# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a run's checkpoints on a chosen α grid: held-out CE and embedding geometry.

The training-time validator logs each run on its own 8-point training grid. This
script re-evaluates saved checkpoints on another grid: `common` (8 points on
[0, 0.10]; CE is compared across sizes only on this grid) or `window32` (32 points
over [0, --alpha-max]; used to locate α_sol). Each grid is written as its own
series, `<dump>/<out-name>/step-N.json`. With `--kind ar` there is no α and the
plain next-token CE is measured.

Conditions match the training-time validator: the first `--max-blocks` blocks of
`<build>/heldout/<source>.npz`; noise from
`validate.noise_seed(seed, source_index, block_index, alpha_index)`, where
`alpha_index` is the position within the grid (so noise differs between grids but
is identical across runs and checkpoints within one); weights cast to bf16.

    torchrun --nproc_per_node=8 -m scripts.eval.ckpt_eval \\
        --dump <dump> --kind flow --flavor 3B --alpha-max 0.10 --grid window32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.eval.loader import (
    build_eval_model,
    checkpoint_steps,
    load_checkpoint_distributed,
    select_steps,
)
from torchtitan.models.swaflow.embed_geometry import embedding_geometry
from torchtitan.models.swaflow.loss import _denoising_ce
from torchtitan.models.swaflow.validate import load_heldout, noise_seed


#: Common grid: 8 points on [0, 0.10], the intersection of all sizes' windows.
#: CE is compared across sizes only on this grid.
COMMON_GRID = [0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.10]

#: 32 points on [0.003125, 0.10]; every fourth point is a common-grid point.
FINE32_GRID = [round(0.003125 * k, 8) for k in range(1, 33)]

#: 16 points with step 0.00625; contains the common grid.
FINE16_GRID = [round(0.00625 * k, 8) for k in range(1, 17)]

GRIDS = {
    "common": COMMON_GRID,
    "fine16": FINE16_GRID,
    "fine32": FINE32_GRID,
    "none": [],
}

#: Window grids over [0, --alpha-max], used to locate α_sol (`a@ce0.1`). The
#: common grid cannot bracket α_sol at every size (at 150M it is about 0.107 at
#: 2T, inside the 0.15 window but above 0.10). α_sol is a value of α rather than a
#: loss, so it may be read on a different grid at each size.
WINDOW_GRIDS = {"window16": 16, "window32": 32}


def window_grid(alpha_max: float, n: int) -> list[float]:
    """n equally spaced points on (0, alpha_max]."""
    return [round(alpha_max * k / n, 10) for k in range(1, n + 1)]


# --------------------------------------------------------------------------
def _cast_params(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast parameters only, leaving buffers untouched.

    `nn.Module.to(dtype=...)` would also cast the complex `freqs_cis` buffer,
    discarding its imaginary part and breaking RoPE. FSDP's MixedPrecisionPolicy
    in training likewise casts only parameters.
    """
    for p in model.parameters():
        if p.data.is_floating_point() and p.data.dtype != dtype:
            p.data = p.data.to(dtype)


def _rank_blocks(n_blocks: int, world: int, rank: int) -> np.ndarray:
    """Same block partition as the training-time validator (remainder dropped)."""
    per_rank = n_blocks // world
    start = rank * per_rank
    return np.arange(start, start + per_rank, dtype=np.int64)


def _noise_for(
    seed: int,
    source_index: int,
    block_ids: np.ndarray,
    alpha_index: int,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    out = torch.empty((len(block_ids), *shape), device=device, dtype=torch.float32)
    for i, b in enumerate(block_ids):
        gen.manual_seed(noise_seed(seed, source_index, int(b), alpha_index))
        out[i] = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    return out


@torch.no_grad()
def eval_source_flow(
    model, source, *, alpha_grid, seed, batch_size, device, world, rank
) -> dict:
    from torchtitan.components.loss import IGNORE_INDEX

    dim = model.model_args.dim
    block_ids = _rank_blocks(source.n_blocks, world, rank)
    stats = torch.zeros(len(alpha_grid), 2, dtype=torch.float64, device=device)
    for ai, a in enumerate(alpha_grid):
        for s in range(0, len(block_ids), batch_size):
            ids = block_ids[s : s + batch_size]
            blocks = torch.from_numpy(source.tokens[ids].astype(np.int64)).to(device)
            labels = torch.empty_like(blocks)
            labels[:, :-1] = blocks[:, 1:]
            labels[:, -1] = IGNORE_INDEX
            alpha = torch.full(
                (blocks.shape[0],), float(a), device=device, dtype=torch.float32
            )
            noise = _noise_for(
                seed, source.index, ids, ai, (source.seq_len, dim), device
            )
            logits, _ = model(blocks, alpha=alpha, noise=noise)
            loss_sum, _ = _denoising_ce(logits, labels)
            n_valid = (labels[:, :-1] != IGNORE_INDEX).sum()
            stats[ai, 0] += loss_sum.double()
            stats[ai, 1] += n_valid.double()
    return _reduce_stats(stats, alpha_grid, source, len(block_ids) * world)


@torch.no_grad()
def eval_source_ar(model, source, *, seed, batch_size, device, world, rank) -> dict:
    """AR: plain next-token CE, with no α and no noise (as in `ar/validate.py`)."""
    from torchtitan.components.loss import IGNORE_INDEX

    del seed
    block_ids = _rank_blocks(source.n_blocks, world, rank)
    stats = torch.zeros(1, 2, dtype=torch.float64, device=device)
    for s in range(0, len(block_ids), batch_size):
        ids = block_ids[s : s + batch_size]
        blocks = torch.from_numpy(source.tokens[ids].astype(np.int64)).to(device)
        labels = torch.empty_like(blocks)
        labels[:, :-1] = blocks[:, 1:]
        labels[:, -1] = IGNORE_INDEX
        logits = model(blocks)
        ce = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        stats[0, 0] += ce.double()
        stats[0, 1] += (labels != IGNORE_INDEX).sum().double()
    return _reduce_stats(stats, [float("nan")], source, len(block_ids) * world)


def _reduce_stats(stats, alpha_grid, source, n_used) -> dict:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    per_alpha = (stats[:, 0] / stats[:, 1].clamp_min(1)).tolist()
    loss = float(stats[:, 0].sum() / stats[:, 1].sum().clamp_min(1))
    return {
        "loss": loss,
        "per_alpha": {f"{a:g}": v for a, v in zip(alpha_grid, per_alpha)},
        "n_blocks": int(n_used),
        "weight": source.weight,
    }


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="dump_folder of the run")
    ap.add_argument("--kind", choices=["flow", "ar"], required=True)
    ap.add_argument("--flavor", required=True, help="150M / 500M / 1B / 3B")
    ap.add_argument("--alpha-max", type=float, default=None,
                    help="window edge of the flow (schedule_alpha_max); not needed for AR")
    ap.add_argument("--grid", default="window32",
                    choices=sorted(set(GRIDS) | set(WINDOW_GRIDS)),
                    help="α grid (ignored for AR); window* grids are built from --alpha-max")
    ap.add_argument("--alpha-list", default="",
                    help="comma-separated α values instead of --grid "
                         "(e.g. to reproduce the training grid)")
    ap.add_argument("--out-name", default="", help="write to <dump>/<out-name>/")
    ap.add_argument("--build", default="", help="held-out build (default: the run's own)")
    ap.add_argument("--max-blocks", type=int, default=2048)
    ap.add_argument("--local-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234,
                    help="`[swaflow.eval] seed`; the default 1234 matches the training-time "
                         "validator (seeds the noise and the embed_geometry probes)")
    ap.add_argument("--embed-geometry", type=int, default=1024)
    ap.add_argument("--every-tokens", type=int, default=0,
                    help="keep one checkpoint per this many tokens (e.g. 100_000_000_000); 0 keeps all")
    ap.add_argument("--tokens-per-step", type=int, default=8_388_608)
    ap.add_argument("--steps", default="", help="comma-separated steps to evaluate")
    ap.add_argument("--folder", default="checkpoint")
    ap.add_argument("--no-qk-norm", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-evaluate steps that already have a JSON")
    ap.add_argument("--time-budget-sec", type=int, default=0,
                    help="stop cleanly, skipping the rest, after this many seconds")
    args = ap.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)
    device = torch.device("cuda", local_rank if world > 1 else 0)
    is_main = rank == 0

    dump = Path(args.dump)
    out_name = args.out_name or (
        "val_ar_recheck" if args.kind == "ar" else f"val_{args.grid}"
    )
    out_dir = dump / out_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- checkpoints ---
    if args.steps:
        steps = [int(s) for s in args.steps.split(",") if s.strip()]
    else:
        steps = select_steps(
            checkpoint_steps(dump, folder=args.folder),
            every_tokens=args.every_tokens,
            tokens_per_step=args.tokens_per_step,
        )
    if not args.overwrite:
        steps = [s for s in steps if not (out_dir / f"step-{s}.json").exists()]
    if is_main:
        print(f"[eval] {dump.name} / {args.kind} {args.flavor}: {len(steps)} ckpts "
              f"-> {out_dir}", flush=True)
    if not steps:
        if world > 1:
            dist.destroy_process_group()
        return 0

    # --- held-out ---
    build = args.build or _build_from_run(dump)
    sources = load_heldout(build, args.max_blocks)
    if args.alpha_list:
        alpha_grid = [float(x) for x in args.alpha_list.split(",") if x.strip()]
    elif args.grid in WINDOW_GRIDS:
        if not args.alpha_max:
            raise SystemExit(f"--grid {args.grid} requires --alpha-max")
        alpha_grid = window_grid(args.alpha_max, WINDOW_GRIDS[args.grid])
    else:
        alpha_grid = GRIDS[args.grid]
    if args.kind == "ar":
        alpha_grid = []
    if is_main:
        total = sum(s.n_blocks for s in sources)
        print(f"[eval] held-out {len(sources)} sources / {total:,} blocks, "
              f"grid={args.grid} ({len(alpha_grid)} pts) build={build}", flush=True)

    # --- model (built and loaded in fp32, run in bf16) ---
    model = build_eval_model(
        args.kind, args.flavor,
        alpha_max=args.alpha_max, qk_norm=not args.no_qk_norm,
    )
    # Do not use `model.to(dtype=...)`: it would cast the complex `freqs_cis`
    # buffer and silently break RoPE. Cast parameters only, as FSDP does in training.
    model.to(device=device)
    _cast_params(model, torch.float32)
    model.eval()

    t_start = time.time()
    for i, step in enumerate(steps):
        if args.time_budget_sec and time.time() - t_start > args.time_budget_sec:
            if is_main:
                print(f"[eval] time budget {args.time_budget_sec}s exceeded; "
                      f"stopping with {len(steps) - i} checkpoints left", flush=True)
            break
        t0 = time.time()
        # Checkpoints load into fp32 parameters, so cast back before each load.
        _cast_params(model, torch.float32)
        info = load_checkpoint_distributed(
            model, dump / args.folder / f"step-{step}", device=device
        )
        geometry = {}
        if args.embed_geometry:
            geometry = embedding_geometry(
                model.tok_embeddings.weight.detach(),
                n_probe=args.embed_geometry, seed=args.seed,
            )
        _cast_params(model, torch.bfloat16)

        results = {}
        for src in sources:
            if args.kind == "flow":
                results[src.name] = eval_source_flow(
                    model, src, alpha_grid=alpha_grid, seed=args.seed,
                    batch_size=args.local_batch, device=device,
                    world=world, rank=rank,
                )
            else:
                results[src.name] = eval_source_ar(
                    model, src, seed=args.seed, batch_size=args.local_batch,
                    device=device, world=world, rank=rank,
                )
        measured = {n: r for n, r in results.items() if r["n_blocks"] > 0}
        if not measured:
            raise RuntimeError("no block evaluated: held-out has fewer blocks than the world size")
        wsum = sum(r["weight"] for r in measured.values()) or 1.0
        overall = sum(r["loss"] * r["weight"] for r in measured.values()) / wsum

        if is_main:
            payload = {
                "step": step,
                "overall_loss": overall,
                "alpha_grid": alpha_grid,
                "grid_name": (
                    "none" if args.kind == "ar"
                    else ("custom" if args.alpha_list else args.grid)
                ),
                "kind": args.kind,
                "flavor": args.flavor,
                "alpha_max": args.alpha_max,
                "build_dir": str(build),
                "max_blocks_per_source": args.max_blocks,
                "seed": args.seed,
                "sources": results,
                "embed_geometry": geometry,
                "elapsed_sec": time.time() - t0,
                "load_sec": info["elapsed_sec"],
            }
            (out_dir / f"step-{step}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False)
            )
            print(f"[eval] step {step:>7}  overall {overall:.4f}  "
                  f"({time.time() - t0:.1f}s, load {info['elapsed_sec']:.1f}s)  "
                  f"[{i + 1}/{len(steps)}]", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _build_from_run(dump: Path) -> str:
    """Reuse the build recorded in `<dump>/val/step-*.json`."""
    vals = sorted((dump / "val").glob("step-*.json"))
    if vals:
        return json.loads(vals[0].read_text())["build_dir"]
    env = os.environ.get("SWAFLOW_DATA_ROOT")
    if env:
        return str(Path(env) / "production_6t")
    raise SystemExit("pass --build (cannot infer the build from the run)")


if __name__ == "__main__":
    sys.exit(main())
