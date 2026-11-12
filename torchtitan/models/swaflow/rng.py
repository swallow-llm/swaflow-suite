# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Independent random streams per data-parallel rank.

Without pipeline parallelism torchtitan's `set_determinism` seeds every rank
identically, so the default CPU/CUDA generators would draw the same t (hence α)
and the same sphere noise on all DP ranks. After `Trainer.__init__` the default
generators are reseeded per rank of the `batch` mesh (dp_replicate × dp_shard,
the dimension the dataloader shards over); TP ranks, which hold the same batch,
share a stream. Initial weights are unaffected (they are drawn earlier through the
DTensor RNG tracker, which `torch.manual_seed` does not touch), and the seed is a
pure function of (base_seed, batch_rank). Context parallelism is not supported
(the sphere noise would be identical across CP shards); a warning is logged.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

import torch

from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

__all__ = [
    "apply_rank_distinct_rng",
    "batch_rank_of",
    "load_rng_state",
    "prune_rng_states",
    "rank_rng_seed",
    "restore_rank_rng_state",
    "rng_state_dict",
    "save_rank_rng_state",
]

_M64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """splitmix64 finalizer: maps nearby inputs to uncorrelated 64-bit values."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _M64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def rank_rng_seed(base_seed: int, rank: int) -> int:
    """Seed for `(base_seed, rank)`; a pure function.

    Rank 0 keeps base_seed, so single-process runs (tests, the sampler) are
    unaffected by reseeding. Other ranks get a splitmix64-mixed seed (as in
    `validate.noise_seed`).
    """
    if rank < 0:
        raise ValueError(f"rank must be non-negative (got {rank})")
    if rank == 0:
        return base_seed & _M64
    return _splitmix64((base_seed & _M64) ^ _splitmix64(rank))


def batch_rank_of(parallel_dims: ParallelDims) -> int:
    """Rank index of the dataloader shard.

    Same derivation as the `batch_rank` that `Trainer.__init__` passes to the
    dataloader (`torchtitan/train.py`), so ranks that read different data always
    get different streams.
    """
    if not parallel_dims.dp_enabled:
        return 0
    return parallel_dims.get_mesh("batch").get_local_rank()


def apply_rank_distinct_rng(
    parallel_dims: ParallelDims, *, enable: bool = True
) -> dict[str, int | bool]:
    """Reseed the default generators with a per-DP-rank seed.

    Args:
        parallel_dims: parallel configuration of the run
        enable: if False, do nothing (all DP ranks keep the same seed)

    Returns:
        `{rank, base_seed, seed, enabled}` for the run log.
    """
    base_seed = torch.initial_seed()  # the seed set by set_determinism
    rank = batch_rank_of(parallel_dims)
    if not enable:
        logger.warning(
            "[rank-rng] rank_distinct_seed = false: all DP ranks draw the same "
            "random numbers (for diagnostics only)"
        )
        return {"rank": rank, "base_seed": base_seed, "seed": base_seed, "enabled": False}

    if parallel_dims.cp > 1:
        logger.warning(
            "[rank-rng] context parallel is enabled: t matches within a cp group "
            "(correct), but the sphere noise is identical across cp shards"
        )

    seed = rank_rng_seed(base_seed, rank)
    torch.manual_seed(seed)
    logger.info(
        f"[rank-rng] batch rank {rank}: RNG seed set to {seed} "
        f"(base {base_seed}); t and sphere noise are independent across DP ranks"
    )
    return {"rank": rank, "base_seed": base_seed, "seed": seed, "enabled": True}


# --------------------------------------------------------------------------
# Save the random-stream position with each checkpoint
# --------------------------------------------------------------------------
#
# The seed fixes the stream but not its position, so a resumed run would
# otherwise draw different t / noise (and make different spike-skip decisions)
# than an uninterrupted one. Each rank saves the state of its default generators
# to `<dump>/rngstate/step-<N>/rank<r>.json`. No collectives are used (each rank
# writes and reads only its own file), so filesystem lag cannot cause a hang.
#
# In the training loop the default generators are used only by `torch.rand` in
# `sample_t` and `torch.randn_like` for SLERP, both inside train_step; the
# validator uses its own `torch.Generator` and the dataloader draws no random
# numbers. The state taken right after train_step N (at checkpoint time) therefore
# makes step N+1 after resume draw the same values as an uninterrupted run.

_RNG_DIRNAME = "rngstate"


def rng_state_dict() -> dict[str, Any]:
    """State of the default generators (CPU and CUDA) in JSON-serializable form."""
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state().numpy().tobytes().hex(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state().numpy().tobytes().hex()
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    cpu = state.get("cpu")
    if cpu:
        torch.set_rng_state(_bytes_to_tensor(cpu))
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state(_bytes_to_tensor(cuda))


def _bytes_to_tensor(hex_str: str) -> torch.Tensor:
    # not `torch.frombuffer`, which warns on non-writable buffers
    return torch.tensor(bytearray(bytes.fromhex(hex_str)), dtype=torch.uint8)


def _rng_path(dump_folder: str, step: int, rank: int) -> str:
    return os.path.join(dump_folder, _RNG_DIRNAME, f"step-{step}", f"rank{rank}.json")


def save_rank_rng_state(dump_folder: str, step: int, rank: int) -> None:
    """Save this rank's stream position (called only at checkpoint steps)."""
    path = _rng_path(dump_folder, step, rank)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"step": step, "rank": rank, **rng_state_dict()}, f)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(f"[rank-rng] cannot save RNG state ({path}): {e}")


def restore_rank_rng_state(dump_folder: str, step: int, rank: int) -> bool:
    """Restore the saved state if present; otherwise warn and return False.

    Training still works without it, but draws after resume (and spike-skip
    decisions) differ from an uninterrupted run.
    """
    path = _rng_path(dump_folder, step, rank)
    if not os.path.exists(path):
        logger.warning(
            f"[rank-rng] step {step}: no saved RNG state ({path}); "
            "t / sphere noise after resume will differ from an uninterrupted run"
        )
        return False
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"[rank-rng] cannot read RNG state ({path}): {e}")
        return False
    load_rng_state(state)
    logger.info(f"[rank-rng] step {step}: restored the RNG stream position (rank {rank})")
    return True


def prune_rng_states(dump_folder: str, keep: int = 20) -> None:
    """Delete the directories of old steps (called by rank 0 only)."""
    root = os.path.join(dump_folder, _RNG_DIRNAME)
    if not os.path.isdir(root):
        return
    steps = []
    for name in os.listdir(root):
        if name.startswith("step-"):
            try:
                steps.append(int(name[5:]))
            except ValueError:
                continue
    for step in sorted(steps)[:-keep]:
        shutil.rmtree(os.path.join(root, f"step-{step}"), ignore_errors=True)
