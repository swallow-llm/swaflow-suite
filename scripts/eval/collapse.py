# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Collapse rate (and optionally Gen-PPL) of prefix-conditioned generation over checkpoints.

Each of 256 samples continues a 64-token held-out prefix to 512 tokens. A sample is
collapsed when the repeated-4-gram coverage (rep4) of its 448 generated tokens
exceeds 0.9; rep4 also catches loops of formatting tokens that an all-identical
test would miss. The same metrics on the held-out text at the same positions are
reported as a real-text floor, and unconditional generation is measured as a
control. With `--gen-ppl`, the same samples are scored by SmolLM2 (same tokenizer,
V = 49,152), excluding the prefix (`score_from=prefix_len`). Generation uses 64
function evaluations, temperature 1.0 and an α grid inside the training window
(built from the model's schedule).

(run, step) conditions are split across ranks without collectives, because their
run times differ widely.

    torchrun --nproc_per_node=8 -m scripts.eval.collapse --outputs $SWAFLOW_OUTPUTS --gen-ppl
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

from scripts.eval.loader import checkpoint_steps, select_steps
from scripts.eval.profiles import SIZES, TOKENS_PER_STEP
from scripts.eval.generate import (
    build_model,
    load_dcp_weights,
    repeated_ngram_coverage,
    run_one,
    sequence_metrics,
)


#: A sample is collapsed when rep4 of its generated tokens exceeds this.
COLLAPSE_REP4 = 0.9


def _prefixes(build_dir: Path, n: int, prefix_len: int):
    """First `n` blocks of the largest held-out source.

    The caller takes their first `prefix_len` tokens as prefixes.
    """
    from torchtitan.models.swaflow.validate import load_heldout

    sources = load_heldout(build_dir)
    src = max(sources, key=lambda s: s.n_blocks)
    if src.n_blocks < n:
        raise RuntimeError(f"not enough held-out blocks: {src.n_blocks} < {n}")
    return np.asarray(src.tokens[:n], dtype=np.int64), src.name


def _rates(seqs, start: int) -> dict:
    """Collapse rate and mean sequence metrics over the tokens from `start` on."""
    tails = [list(s[start:]) for s in seqs]
    rep4 = [repeated_ngram_coverage(t, 4) for t in tails]
    per = [sequence_metrics(t) for t in tails]
    return {
        "n": len(tails),
        "collapse_rate": float(np.mean([r > COLLAPSE_REP4 for r in rep4])),
        "n_collapsed": int(sum(r > COLLAPSE_REP4 for r in rep4)),
        "rep4_mean": float(np.mean(rep4)),
        "rep4_p90": float(np.percentile(rep4, 90)),
        "distinct_1_mean": float(np.mean([m["distinct_1"] for m in per])),
        "distinct_4_mean": float(np.mean([m["distinct_4"] for m in per])),
        "entropy_bits_mean": float(np.mean([m["entropy_bits"] for m in per])),
        "max_run_max": float(max(m["max_run"] for m in per)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default=os.environ.get("SWAFLOW_OUTPUTS", ""))
    ap.add_argument("--build", default="")
    ap.add_argument("--runs", default=",".join(SIZES))
    ap.add_argument(
        "--extra-run", action="append", default=[],
        help="add a run not in SIZES as `name:flavor:alpha_max` (repeatable), "
             "e.g. an SFT run",
    )
    ap.add_argument("--every-tokens", type=int, default=250_000_000_000)
    ap.add_argument(
        "--tokens-per-step", type=int, default=TOKENS_PER_STEP,
        help="tokens per step, used for the `tokens` field of the output JSON. "
             "The default is the pre-training global batch 4096x2048; pass it "
             "for runs with a different global batch, such as SFT",
    )
    ap.add_argument("--steps", default="", help="colon- or comma-separated steps (applied to every run)")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--prefix-len", type=int, default=64)
    ap.add_argument("--nfe", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out-name", default="collapse")
    ap.add_argument("--unconditional", action=argparse.BooleanOptionalAction,
                    default=True, help="also measure unconditional generation as a control")
    ap.add_argument("--gen-ppl", action="store_true",
                    help="also score Gen-PPL of the same samples (with a SmolLM2 model)")
    ap.add_argument("--ppl-model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--ppl-batch", type=int, default=8)
    ap.add_argument("--save-tokens", action="store_true",
                    help="also save the raw tokens (to rescore later with another model)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--time-budget-sec", type=int, default=0)
    args = ap.parse_args(argv)

    # No process group: conditions take very different times, so ranks split them
    # using only torchrun's env (a barrier would make every rank wait for the slowest).
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    outputs = Path(args.outputs)
    build = Path(args.build or (outputs.parent / "data" / "production_6t"))

    # Extra runs go into a copy; SIZES itself stays the four pre-training runs,
    # which other tools iterate over.
    sizes = dict(SIZES)
    for spec in args.extra_run:
        name, flavor, amax = spec.split(":")
        sizes[name] = (flavor, 0, float(amax), None)

    conditions = []
    for run in args.runs.split(","):
        label, _dim, amax, _ar = sizes[run]
        dump = outputs / run
        if not (dump / "checkpoint").is_dir():
            continue
        if args.steps:
            steps = [int(s) for s in args.steps.replace(":", ",").split(",")]
            steps = [s for s in steps if (dump / "checkpoint" / f"step-{s}").is_dir()]
        else:
            steps = select_steps(checkpoint_steps(dump),
                                 every_tokens=args.every_tokens)
        for s in steps:
            conditions.append((run, label, amax, s))
    conditions.sort(key=lambda c: (-c[3], c[1]))  # slowest conditions first

    out_root = outputs / args.out_name
    mine = conditions[rank::world]
    if not args.overwrite:
        mine = [c for c in mine
                if not (out_root / c[0] / f"step-{c[3]}.json").exists()]
    if rank == 0:
        print(f"[collapse] {len(conditions)} conditions / world {world} "
              f"-> {out_root}", flush=True)

    prefix_np, prefix_src = _prefixes(build, args.num_samples, args.prefix_len)
    prefix_tokens = torch.from_numpy(prefix_np[:, : args.prefix_len]).to(device)
    prefix_lengths = torch.full(
        (args.num_samples,), args.prefix_len, dtype=torch.long, device=device
    )
    # Real-text floor: the same 448 positions of the held-out text.
    floor = _rates(prefix_np[:, : args.seq_len], args.prefix_len)

    ppl_model = None
    if args.gen_ppl:
        from transformers import AutoModelForCausalLM

        ppl_model = AutoModelForCausalLM.from_pretrained(
            args.ppl_model, dtype=torch.float32
        ).to(device).eval()

    def _ppl(tokens_2d, score_from: int) -> dict:
        from scripts.eval.gen_ppl import score_batches

        return score_batches(
            ppl_model, np.asarray(tokens_2d, dtype=np.int64), bos_id=0,
            batch_size=args.ppl_batch, device=device, score_from=score_from,
        )

    t_start = time.time()
    for run, label, amax, step in mine:
        if args.time_budget_sec and time.time() - t_start > args.time_budget_sec:
            print(f"[collapse] rank {rank}: time budget exceeded, skipping the rest", flush=True)
            break
        t0 = time.time()
        model = build_model(label, alpha_max=amax, qk_norm=True)
        load_dcp_weights(model, outputs / run / "checkpoint" / f"step-{step}")
        model = model.to(device).eval()
        dim = model.model_args.dim
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        noise = torch.randn(
            (args.num_samples, args.seq_len, dim), generator=gen, dtype=torch.float32
        ).to(device)

        pfx = run_one(
            model, steps=args.nfe, temperature=args.temperature, noise=noise,
            naive_full_grid=False, chunk_size=args.chunk_size, seed=args.seed,
            prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths,
            micro_batch=args.micro_batch,
        )
        seqs = pfx.pop("tokens")
        preserved = bool(
            np.array_equal(
                np.asarray(seqs)[:, : args.prefix_len], prefix_np[:, : args.prefix_len]
            )
        )
        payload = {
            "run": run, "size": label, "step": step,
            "tokens": step * args.tokens_per_step,
            "alpha_max": amax, "nfe": args.nfe, "temperature": args.temperature,
            "num_samples": args.num_samples, "seq_len": args.seq_len,
            "prefix_len": args.prefix_len, "prefix_source": prefix_src,
            "seed": args.seed,
            "prefix_preserved": preserved,
            "alpha_grid_source": pfx["alpha_grid_source"],
            "alpha_min": pfx["alpha_min"], "alpha_max_grid": pfx["alpha_max"],
            "conditional": _rates(seqs, args.prefix_len),
            "real_text_floor": floor,
        }
        unc_seqs = None
        if args.unconditional:
            unc = run_one(
                model, steps=args.nfe, temperature=args.temperature, noise=noise,
                naive_full_grid=False, chunk_size=args.chunk_size, seed=args.seed,
                micro_batch=args.micro_batch,
            )
            unc_seqs = unc.pop("tokens")
            payload["unconditional"] = _rates(unc_seqs, args.prefix_len)

        if ppl_model is not None:
            payload["gen_ppl"] = {
                "model": args.ppl_model,
                # The prefix is real text and is excluded from scoring.
                "conditional": _ppl(seqs, args.prefix_len),
                "real_text_floor": _ppl(
                    prefix_np[:, : args.seq_len], args.prefix_len
                ),
            }
            if unc_seqs is not None:
                payload["gen_ppl"]["unconditional"] = _ppl(unc_seqs, args.prefix_len)

        d = out_root / run
        d.mkdir(parents=True, exist_ok=True)
        if args.save_tokens:
            np.savez_compressed(
                d / f"step-{step}.tokens.npz",
                conditional=np.asarray(seqs, dtype=np.int32),
                **({"unconditional": np.asarray(unc_seqs, dtype=np.int32)}
                   if unc_seqs is not None else {}),
            )
        (d / f"step-{step}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        c = payload["conditional"]
        u = payload.get("unconditional", {})
        g = payload.get("gen_ppl", {})
        ppl = (f"  ppl {g['conditional']['ppl']:.1f} "
               f"(floor {g['real_text_floor']['ppl']:.2f})" if g else "")
        print(f"[collapse] rank {rank} {label} step {step:>7} "
              f"({step * TOKENS_PER_STEP / 1e9:.0f}B): "
              f"cond {c['collapse_rate']:.3%} / uncond "
              f"{u.get('collapse_rate', float('nan')):.3%} "
              f"prefix_ok={preserved}{ppl} ({time.time() - t0:.0f}s)", flush=True)
        del model, noise
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
