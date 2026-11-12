# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Held-out SFT loss: the SFT objective measured on held-out blocks.

On the `val.{tokens,clean}.bin` split written by `scripts/data/build_sft.py`, computes

    CE(response positions | prompt kept clean, α)

over an α grid. Unlike the pretraining held-out CE (`ckpt_eval.py`), only response
positions are scored and the prompt is not noised. As a control, mode `noised`
(whole sequence noised, still scoring only response positions) is reported too.
With `--kind ar` the same positions of the same blocks are scored for an AR model
(no α, no prefix mode; a single series `ar`), which is the AR SFT objective.

    torchrun --nproc_per_node=8 -m scripts.eval.sft_val \\
        --dump <sft run> --build <sft build> --flavor 3B --alpha-max 0.10
    torchrun --nproc_per_node=8 -m scripts.eval.sft_val \\
        --dump <ar sft run> --build <sft build> --kind ar --flavor 1B
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from scripts.eval.ckpt_eval import GRIDS, _cast_params
from scripts.eval.loader import (
    build_eval_model,
    checkpoint_steps,
    load_checkpoint_distributed,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.swaflow.sft_data import load_sft_blocks
from torchtitan.models.swaflow.validate import noise_seed

#: Both modes score response positions only; they differ only in whether the
#: prompt is noised.
PREFIX_MODES = ("clean", "noised")


@torch.no_grad()
def score_blocks(
    model,
    tokens: np.ndarray,
    clean: np.ndarray,
    *,
    alpha_grid: list[float],
    device,
    batch_size: int = 8,
    seed: int = 1234,
    block_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Score [n_blocks, seq_len] blocks on an α grid.

    Returns:
        mode -> [len(alpha_grid), 2] array of (loss_sum, n_valid)
    """
    out = {m: np.zeros((len(alpha_grid), 2), dtype=np.float64) for m in PREFIX_MODES}
    n = tokens.shape[0]
    # Noise is seeded by the global block index, so draws do not depend on how
    # blocks are sharded across ranks (same invariant as `validate.noise_seed`).
    ids = np.arange(n) if block_ids is None else np.asarray(block_ids)
    for start in range(0, n, batch_size):
        blk = torch.from_numpy(
            np.array(tokens[start : start + batch_size], dtype=np.int64)
        ).to(device)
        cln = torch.from_numpy(
            np.array(clean[start : start + batch_size], dtype=np.uint8)
        ).to(device)
        b = blk.shape[0]
        # label[j] = block[j+1]; prompt positions are IGNORE (same convention as sft_data)
        labels = torch.full_like(blk, IGNORE_INDEX)
        labels[:, :-1] = torch.where(
            cln[:, 1:].bool(), torch.full_like(blk[:, 1:], IGNORE_INDEX), blk[:, 1:]
        )
        target = labels[:, :-1]
        n_valid = int((target != IGNORE_INDEX).sum())
        for ai, a in enumerate(alpha_grid):
            noise = torch.stack(
                [
                    torch.randn(
                        (blk.shape[1], model.model_args.dim),
                        generator=torch.Generator(device="cpu").manual_seed(
                            noise_seed(seed, 0, int(ids[start + k]), ai)
                        ),
                        dtype=torch.float32,
                    )
                    for k in range(b)
                ]
            ).to(device)
            alpha = torch.full((b,), float(a), device=device)
            for mode in PREFIX_MODES:
                mask = cln if mode == "clean" else None
                logits, _ = model(blk, alpha=alpha, noise=noise, clean_mask=mask)
                ce = F.cross_entropy(
                    logits[:, 1:].flatten(0, 1).float(),
                    target.flatten(0, 1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
                out[mode][ai, 0] += float(ce)
                out[mode][ai, 1] += n_valid
    return out


@torch.no_grad()
def score_blocks_ar(
    model,
    tokens: np.ndarray,
    clean: np.ndarray,
    *,
    device,
    batch_size: int = 8,
) -> np.ndarray:
    """AR version; scores exactly the same positions as `score_blocks`.

    For AR, `logits[:, j]` pairs with `labels[:, j] = block[j+1]` without a shift
    (`sflm/ar/loss.py`), which is the `sft_data` label convention, so labels go
    to `cross_entropy` directly. No α, noise or `clean_mask`.

    Returns:
        [1, 2] array of (loss_sum, n_valid)
    """
    out = np.zeros((1, 2), dtype=np.float64)
    n = tokens.shape[0]
    for start in range(0, n, batch_size):
        blk = torch.from_numpy(
            np.array(tokens[start : start + batch_size], dtype=np.int64)
        ).to(device)
        cln = torch.from_numpy(
            np.array(clean[start : start + batch_size], dtype=np.uint8)
        ).to(device)
        labels = torch.full_like(blk, IGNORE_INDEX)
        labels[:, :-1] = torch.where(
            cln[:, 1:].bool(), torch.full_like(blk[:, 1:], IGNORE_INDEX), blk[:, 1:]
        )
        logits = model(blk)
        ce = F.cross_entropy(
            logits.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        out[0, 0] += float(ce)
        out[0, 1] += int((labels != IGNORE_INDEX).sum())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="dump_folder of the run to evaluate")
    ap.add_argument("--build", required=True, help="SFT build (contains val.*.bin)")
    ap.add_argument("--kind", choices=["flow", "ar"], default="flow")
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--grid", default="common", choices=sorted(GRIDS))
    ap.add_argument("--alpha-list", default="")
    ap.add_argument("--steps", default="", help="comma/colon separated; default: latest")
    ap.add_argument("--folder", default="checkpoint")
    ap.add_argument("--max-blocks", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-name", default="sft_val")
    ap.add_argument("--no-qk-norm", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    if args.alpha_list:
        alpha_grid = [
            float(x) for x in args.alpha_list.replace(":", ",").split(",") if x.strip()
        ]
    else:
        alpha_grid = GRIDS[args.grid]

    tokens, clean, meta = load_sft_blocks(args.build, split="val")
    n = min(args.max_blocks, tokens.shape[0])
    # Shard blocks across ranks; the model is replicated on every rank.
    idx = np.arange(n)[rank::world]
    my_tokens = np.asarray(tokens[idx])
    my_clean = np.asarray(clean[idx])

    dump = Path(args.dump)
    out_dir = dump / args.out_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    steps = (
        [int(s) for s in args.steps.replace(":", ",").split(",") if s.strip()]
        if args.steps
        else [checkpoint_steps(dump, folder=args.folder)[-1]]
    )
    if not args.overwrite:
        steps = [s for s in steps if not (out_dir / f"step-{s}.json").exists()]
    if is_main:
        print(f"[sft_val] {dump.name}: {len(steps)} ckpts / {n} val blocks "
              f"/ α {alpha_grid}", flush=True)
    if not steps:
        if world > 1:
            dist.destroy_process_group()
        return 0

    if args.kind == "ar":
        # AR has no α axis; scored positions and held-out blocks match the flow.
        alpha_grid = []
        model = build_eval_model("ar", args.flavor, qk_norm=not args.no_qk_norm)
    else:
        model = build_eval_model(
            "flow", args.flavor, alpha_max=args.alpha_max,
            qk_norm=not args.no_qk_norm,
        )
    model.to(device=device)
    model.eval()

    for step in steps:
        t0 = time.time()
        _cast_params(model, torch.float32)
        load_checkpoint_distributed(
            model, dump / args.folder / f"step-{step}", device=device
        )
        _cast_params(model, torch.bfloat16)
        if args.kind == "ar":
            local = {"ar": score_blocks_ar(
                model, my_tokens, my_clean, device=device, batch_size=args.batch,
            )}
        else:
            local = score_blocks(
                model, my_tokens, my_clean, alpha_grid=alpha_grid, device=device,
                batch_size=args.batch, seed=args.seed, block_ids=idx,
            )
        modes = tuple(local)
        payload = {
            "step": step, "dump": str(dump), "build": str(args.build),
            "kind": args.kind,
            "build_name": meta.get("build_name"), "template": meta.get("template"),
            "alpha_grid": alpha_grid, "n_blocks": int(n), "seed": args.seed,
        }
        for mode in modes:
            t = torch.from_numpy(local[mode]).to(device)
            if world > 1:
                dist.all_reduce(t)
            tot = t.cpu().numpy()
            payload[mode] = {
                "ce": [float(tot[i, 0] / max(tot[i, 1], 1)) for i in range(tot.shape[0])],
                "n_valid": [int(tot[i, 1]) for i in range(tot.shape[0])],
            }
        payload["mean_ce"] = {m: float(np.mean(payload[m]["ce"])) for m in modes}
        if is_main:
            (out_dir / f"step-{step}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            print(f"[sft_val] step {step}: "
                  + " / ".join(f"{m} {payload['mean_ce'][m]:.4f}" for m in modes)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
