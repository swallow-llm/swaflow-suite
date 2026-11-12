# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Score an external Riemannian flow with our MCQ harness, unchanged.

The target is the released S-FLM checkpoint (HF `jdeschena/s-flm`, `owt/sfm.ckpt`;
12 layers, width 768, 1M steps on OpenWebText). Scoring is that of `run_mcq.py`
(`mcq.score_flow`); only the model side follows S-FLM's own setup: the
state_dict written by `export_sflm_owt.py` is loaded through the key mapping in
`swaflow/sflm_weights.py`, tasks are re-encoded with GPT-2 (V = 50,257 + 1 pad), the
sequence cap is its training length 1024 (longer sequences lose context on the
left, as in the harness), and the time input is σ = −log α
(`invert_time_convention=false`), so the model receives 1 − α.

At α = 0 (the main setting) σ = −log 0 is undefined, and S-FLM was trained only
on α in [1e-3, 0.121]. x_t stays pure noise as for our models, but the time input
is set to the noise end of the training range, α = 1e-3 (σ = 6.91, where the
S-FLM sampler starts; `--alpha-floor`). For our models α = 0 is likewise the
noise end of the training range.

    python -m scripts.external.run_mcq_external --weights <dir>/weights_ema.pt \\
        --out-dir $SWAFLOW_OUTPUTS/ext_sflm_owt_ema --out-name mcq_ext_clean_pmi_a0 \\
        --prefix-mode clean --pmi --alpha-list 0.0

Output has the format of `run_mcq.py` (`<out-dir>/<out-name>/step-<global_step>.json`)
plus an `external` field.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.eval import mcq as mcqlib
from scripts.eval.ckpt_eval import GRIDS, _cast_params
from scripts.eval.run_mcq import DEFAULT_TASKS, _gather, _prepare, _shard

#: Lowest α S-FLM (OWT) was trained on (LogLinear eps; min of `noise.alpha_vals`).
SFLM_ALPHA_FLOOR = 1e-3


def sflm_time_cond(alpha: torch.Tensor,
                   floor: float = SFLM_ALPHA_FLOOR) -> torch.Tensor:
    """Map α (our convention) to our model's time input so that σ = −log α'.

    `alpha_to_sigma(x) = −log(1 − x)`, so x = 1 − α' gives σ = −log α', with
    α' = max(α, floor) to stay inside the training range.
    """
    return 1.0 - alpha.clamp(min=floor)


def build_sflm_owt(weights: str, *, dtype: torch.dtype = torch.float32):
    """Build our SphereDiT (`sflm_small` shape) from an exported state_dict."""
    from torchtitan.models.swaflow import swaflow_args
    from torchtitan.models.swaflow.model import SphereDiT
    from torchtitan.models.swaflow.sflm_weights import sflm_to_swaflow_state_dict

    blob = torch.load(weights, map_location="cpu", weights_only=True)
    meta = blob["meta"]
    base = swaflow_args["sflm_small"]
    # Shape of S-FLM's small-sphere-dit (= sflm_small); vocab and length come
    # from the checkpoint.
    assert (meta["hidden_size"], meta["n_blocks"], meta["n_heads"],
            meta["cond_dim"]) == (base.dim, base.n_layers, base.n_heads,
                                  base.cond_dim), meta
    args = dataclasses.replace(base, vocab_size=meta["vocab_size"],
                               max_seq_len=meta["length"])
    with torch.device("cpu"):
        model = SphereDiT(args)
    model.load_state_dict(
        sflm_to_swaflow_state_dict(blob["state_dict"], dim=args.dim), strict=True)
    _cast_params(model, dtype)
    model.meta = meta
    return model


def _n_over(docs, tok, max_len: int) -> int:
    """Number of requests whose context the harness truncates on the left."""
    n = 0
    for d in docs:
        for ctx, cont in zip(d.contexts, d.continuations):
            L = (len(tok(ctx, add_special_tokens=False)["input_ids"])
                 + len(tok(cont, add_special_tokens=False)["input_ids"]))
            n += int(L > max_len)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="weights_{ema,raw}.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", default="mcq_ext_clean_pmi_a0")
    ap.add_argument("--tokenizer", default="openai-community/gpt2")
    ap.add_argument("--max-len", type=int, default=0,
                    help="0: the checkpoint's training length (1024 for S-FLM)")
    ap.add_argument("--alpha-floor", type=float, default=SFLM_ALPHA_FLOOR)
    ap.add_argument("--prepend-bos", action="store_true",
                    help="sensitivity check: prepend <|endoftext|> to every sequence "
                         "(S-FLM training blocks always start with BOS)")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--grid", default="none", choices=sorted(GRIDS))
    ap.add_argument("--alpha-list", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prefix-mode", choices=["clean", "noised"], default="clean")
    ap.add_argument("--noise-draws", type=int, default=1)
    ap.add_argument("--pmi", action="store_true")
    ap.add_argument("--neutral-token", type=int, default=198,
                    help="198 is a newline in GPT-2 too (as in SmolLM2)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dump-nll", default="",
                    help="write the summed choice CE (with prefix / neutral prefix) "
                         "per request (diagnostic; same format as the SFCD-side dump)")
    args = ap.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    model = build_sflm_owt(args.weights)
    meta = model.meta
    step = int(meta["global_step"])
    out_path = Path(args.out_dir) / args.out_name / f"step-{step}.json"
    if out_path.exists() and not args.overwrite:
        if is_main:
            print(f"[mcq-ext] exists: {out_path}")
        if world > 1:
            dist.destroy_process_group()
        return 0

    alpha_grid = ([float(x) for x in args.alpha_list.replace(":", ",").split(",")
                   if x.strip()] if args.alpha_list else GRIDS[args.grid])
    if not alpha_grid:
        raise SystemExit("specify --alpha-list or --grid")
    max_len = args.max_len or int(meta["length"])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.decode([args.neutral_token]) == "\n", "neutral token is not a newline"
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    bos = tok.bos_token_id if args.prepend_bos else None
    prepared = _prepare(tasks, tok, args.limit,
                        max_len - (1 if bos is not None else 0), rank)
    n_over = {}
    for name, (_, docs, reqs) in prepared.items():
        if bos is not None:
            for r in reqs:
                r.ctx_ids = [bos] + r.ctx_ids
        if is_main:
            n_over[name] = _n_over(docs, tok, max_len)
            print(f"[mcq-ext] {name}: {len(docs)} docs / {len(reqs)} requests "
                  f"/ {n_over[name]} over {max_len} tokens", flush=True)

    model.to(device=device)
    _cast_params(model, torch.bfloat16)      # as in the harness (run_mcq.py)
    model.eval()

    def tc(a: torch.Tensor) -> torch.Tensor:
        return sflm_time_cond(a, args.alpha_floor)

    t0 = time.time()
    results = {}
    dumped: dict[str, dict] = {}
    for name, (ti, docs, reqs) in prepared.items():
        mine = _shard(reqs, world, rank)
        kw = dict(alpha_grid=alpha_grid, seed=args.seed, task_index=ti,
                  device=device, batch_size=args.batch, n_total=len(reqs),
                  prefix_mode=args.prefix_mode, noise_draws=args.noise_draws,
                  time_cond=tc)
        local = mcqlib.score_flow(model, mine, **kw)
        if args.pmi:
            local_n = mcqlib.score_flow(
                model, mine, neutral_token=args.neutral_token, **kw)
            if args.dump_nll:
                ce_p = _gather(local, len(reqs), device)
                ce_n = _gather(local_n, len(reqs), device)
                dumped[name] = {"ce_prefix": ce_p.tolist(),
                                "ce_neutral": ce_n.tolist()}
                if is_main:
                    print(f"[mcq-ext] {name}: mean CE-sum prefix "
                          f"{ce_p.mean():.3f} / neutral {ce_n.mean():.3f}",
                          flush=True)
            local = local - local_n
        nll = _gather(local, len(reqs), device)
        results[name] = mcqlib.score_accuracy(docs, reqs, nll)
        if is_main:
            print(f"[mcq-ext] {name}: acc_norm_tok="
                  f"{results[name]['acc_norm_tok']:.4f}", flush=True)

    if is_main:
        mean_tok = float(np.mean([r["acc_norm_tok"] for r in results.values()]))
        payload = {
            "step": step,
            "kind": "flow",
            "flavor": "sflm_small",
            "alpha_grid": alpha_grid,
            "alpha_max": meta["alpha_max"],
            "seed": args.seed,
            "prefix_mode": args.prefix_mode,
            "prompt_template": "none",
            "noise_draws": args.noise_draws,
            "pmi": args.pmi,
            "tasks": results,
            "mean_acc": float(np.mean([r["acc"] for r in results.values()])),
            "mean_acc_norm_tok": mean_tok,
            "mean_acc_norm_byte": float(
                np.mean([r["acc_norm_byte"] for r in results.values()])),
            "elapsed_sec": time.time() - t0,
            "external": {
                **meta,
                "weights": str(args.weights),
                "tokenizer_used": args.tokenizer,
                "max_len": max_len,
                "time_cond": "sigma = -log(max(alpha, alpha_floor))",
                "alpha_floor": args.alpha_floor,
                "prepend_bos": bool(args.prepend_bos),
                "neutral_token": args.neutral_token,
                "n_requests_over_max_len": n_over,
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.dump_nll:
            torch.save({"meta": payload["external"], "tasks": dumped},
                       args.dump_nll)
        per = "  ".join(f"{n}={r['acc_norm_tok']:.3f}" for n, r in results.items())
        print(f"[mcq-ext] mean {mean_tok:.4f}  {per}  -> {out_path}", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
