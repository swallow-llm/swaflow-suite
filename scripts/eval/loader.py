# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Load DCP checkpoints into a distributed evaluation job (flow and AR).

If every rank called `dcp.load`, each checkpoint would be read once per rank; a
single reader is slow because per-shard latency adds up. Instead, one reader per
node (local rank 0) loads a disjoint subset of the state-dict keys with
`dcp.load(..., no_dist=True)`, and all ranks then broadcast every key from its
owner in a fixed (sorted) order. Each checkpoint is read once, in parallel across nodes.

Evaluation does not use FSDP: the model (up to 3B) fits on one GPU, so it is
replicated and only the held-out blocks are split across ranks. Weights are cast
to bf16 for the forward pass to match training numerics (SLERP stays in fp64 and
RoPE in fp32).
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


__all__ = [
    "build_eval_model",
    "checkpoint_steps",
    "load_checkpoint_distributed",
    "loader_ranks_for",
]


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def build_eval_model(
    kind: str,
    flavor: str,
    *,
    alpha_max: float | None = None,
    qk_norm: bool = True,
    max_seq_len: int | None = None,
) -> torch.nn.Module:
    """Build a model from its flavor on CPU (weights uninitialized).

    Args:
        kind: ``"flow"`` (SphereDiT) or ``"ar"`` (AR twin).
        alpha_max: window edge of the flow (`schedule_alpha_max`); ignored for AR.
        qk_norm: must match the checkpoint (on by default).
        max_seq_len: length of the RoPE table. Longer than the input is harmless,
            since `freqs_cis[0:seqlen]` is sliced.
    """
    if kind == "flow":
        from torchtitan.models.swaflow import swaflow_args
        from torchtitan.models.swaflow.model import SphereDiTForPretrain

        args = dataclasses.replace(
            swaflow_args[flavor], schedule_alpha_max=alpha_max, qk_norm=qk_norm
        )
        if max_seq_len is not None:
            args = dataclasses.replace(args, max_seq_len=max_seq_len)
        with torch.device("cpu"):
            return SphereDiTForPretrain(args)
    if kind == "ar":
        from torchtitan.models.swaflow.ar import ar_args
        from torchtitan.models.swaflow.ar.model import ARTransformer

        args = dataclasses.replace(ar_args[flavor], qk_norm=qk_norm)
        if max_seq_len is not None:
            args = dataclasses.replace(args, max_seq_len=max_seq_len)
        with torch.device("cpu"):
            return ARTransformer(args)
    raise ValueError(f"kind must be 'flow' or 'ar' (got {kind!r})")


# --------------------------------------------------------------------------
# Checkpoint enumeration
# --------------------------------------------------------------------------
def checkpoint_steps(
    dump: str | Path, *, folder: str = "checkpoint"
) -> list[int]:
    """Return the N of every `<dump>/<folder>/step-N`, ascending."""
    root = Path(dump) / folder
    steps = []
    for p in root.glob("step-*"):
        if not p.is_dir():
            continue
        # DCP writes .metadata last; skip partially written or copied checkpoints.
        if not (p / ".metadata").exists():
            continue
        try:
            steps.append(int(p.name.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def select_steps(
    steps: list[int],
    *,
    every_tokens: int = 0,
    tokens_per_step: int = 8_388_608,
    limit: int = 0,
) -> list[int]:
    """Thin checkpoints to one per `every_tokens` tokens (the last is always kept).

    Selection is by token position rather than step number, so runs with
    different global batch sizes are sampled at the same token positions.
    """
    if not steps:
        return []
    if every_tokens <= 0:
        out = list(steps)
    else:
        # Pick the checkpoint nearest to each target position; taking "the next one
        # at least `stride` past the previous pick" would accumulate rounding and
        # drop points.
        stride = every_tokens / tokens_per_step
        picked: list[int] = []
        k = 1
        while k * stride <= steps[-1] + stride / 2:
            target = k * stride
            nearest = min(steps, key=lambda s: abs(s - target))
            if nearest not in picked:
                picked.append(nearest)
            k += 1
        if steps[-1] not in picked:
            picked.append(steps[-1])
        out = sorted(picked)
    if limit:
        out = out[:limit]
    return out


# --------------------------------------------------------------------------
# Distributed loading
# --------------------------------------------------------------------------
def loader_ranks_for(world_size: int, local_world_size: int) -> list[int]:
    """One reader rank per node."""
    lw = max(1, local_world_size)
    return list(range(0, world_size, lw))


def load_checkpoint_distributed(
    model: torch.nn.Module,
    ckpt: str | Path,
    *,
    loader_ranks: list[int] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load only the model weights from a DCP checkpoint into every rank, in place.

    The model must already be on the GPU in fp32, the checkpoint dtype (the
    broadcast is in place and `dcp.load` does not convert dtypes); cast to bf16
    after loading. Without torch.distributed this is a plain `no_dist=True` load.

    Returns:
        {"loaded": number of tensors, "elapsed_sec": seconds}
    """
    import torch.distributed.checkpoint as dcp

    ckpt = Path(ckpt)
    t0 = time.time()
    sd = model.state_dict()
    keys = sorted(sd.keys())

    if not (dist.is_available() and dist.is_initialized()):
        md = dcp.FileSystemReader(str(ckpt)).read_metadata()
        _check_keys(keys, set(md.state_dict_metadata.keys()), ckpt)
        staging = {
            k: torch.empty(sd[k].shape, dtype=sd[k].dtype, device="cpu") for k in keys
        }
        dcp.load(staging, checkpoint_id=str(ckpt), no_dist=True)
        for k in keys:
            sd[k].copy_(staging[k])
        return {"loaded": len(keys), "elapsed_sec": time.time() - t0}

    rank = dist.get_rank()
    world = dist.get_world_size()
    if loader_ranks is None:
        loader_ranks = loader_ranks_for(
            world, int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
        )
    n_load = len(loader_ranks)
    owner_of = {k: loader_ranks[i % n_load] for i, k in enumerate(keys)}

    if rank in loader_ranks:
        md = dcp.FileSystemReader(str(ckpt)).read_metadata()
        _check_keys(keys, set(md.state_dict_metadata.keys()), ckpt)
        # Stage on CPU so that reader ranks do not use extra GPU memory.
        mine = {
            k: torch.empty(sd[k].shape, dtype=sd[k].dtype, device="cpu")
            for k in keys
            if owner_of[k] == rank
        }
        # no_dist=True issues no collectives, so only readers call it.
        dcp.load(mine, checkpoint_id=str(ckpt), no_dist=True)
        for k, v in mine.items():
            sd[k].copy_(v)
        del mine

    # All ranks broadcast in the same (sorted) key order; the model is already on GPU.
    for k in keys:
        dist.broadcast(sd[k], src=owner_of[k])
    torch.cuda.synchronize()
    out = {"loaded": len(keys), "elapsed_sec": time.time() - t0, "loaders": n_load}
    if verbose and rank == 0:
        print(f"[loader] {ckpt.name}: {len(keys)} tensors / {n_load} readers "
              f"/ {out['elapsed_sec']:.1f} s", flush=True)
    return out


def _check_keys(want: Iterable[str], have: set[str], ckpt: Path) -> None:
    missing = [k for k in want if k not in have]
    if missing:
        raise RuntimeError(
            f"{ckpt} is missing {len(missing)} model keys: {missing[:5]} ... "
            "(do the QK-norm setting and flavor match the checkpoint?)"
        )
