# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Generate from a trained checkpoint and report structural-health metrics.

The α grid stays inside the training window: training sees only α in [0, α_max]
(truncation), so a full [0, 1] grid would take the model through α it never
saw. The grid comes from the model's own schedule (`TruncatedScheduleWrapper`,
as in training): it spans α from ~0 to α_max and ends with argmax decoding at
α_max (`sampling.alpha_grid`). `--naive-full-grid` adds a control on the
untruncated [0, 1] grid.

The metrics measure structural health, not quality: repetition (distinct-n for
n=1..4, share of tokens covered by repeated 4-grams, longest same-token run,
top-token share, unigram entropy), monotonicity in the number of steps (NFE),
and degenerate output at low temperature (one token throughout, endless loops).

    python -m scripts.eval.generate --ckpt <run>/checkpoint/step-N --flavor 3B \\
        --alpha-max 0.10 --qk-norm --steps 4 8 32 64 --temperatures 1.0 0.1 --out <dir>

Checkpoints are DCP, model-only (`snapshots/step-N`) or full state
(`checkpoint/step-N`); both store the model weights under flat unprefixed keys,
and only those keys are read. QK-norm (default off) must match the checkpoint,
otherwise `q_norm.weight` / `k_norm.weight` do not match its keys.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from torchtitan.models.swaflow import swaflow_args
from torchtitan.models.swaflow.model import SphereDiTForPretrain
from torchtitan.models.swaflow.sampling import alpha_grid, generate, SamplerConfig
from torchtitan.models.swaflow.schedules import LogLinear


# --------------------------------------------------------------------------
# Metrics (pure functions; tested in swaflow/tests/test_gen_metrics.py)
# --------------------------------------------------------------------------


def distinct_n(tokens: Sequence[int], n: int) -> float:
    """Unique n-grams / all n-grams: 1.0 if no n-gram repeats, lower if repetitive.

    A sequence too short to hold one n-gram gives 1.0.
    """
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def repeated_ngram_coverage(tokens: Sequence[int], n: int) -> float:
    """Fraction of tokens covered by n-grams that occur at least twice.

    distinct-n measures the share of duplicate n-grams; this measures how much
    of the sequence is taken up by repetition, and saturates at 1.0 for a
    sequence stuck in a loop.
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(tokens)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    return sum(covered) / len(tokens)


def max_run_length(tokens: Sequence[int]) -> int:
    """Length of the longest run of one repeated token."""
    if not tokens:
        return 0
    best = run = 1
    for prev, cur in zip(tokens, tokens[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def unigram_entropy_bits(tokens: Sequence[int]) -> float:
    """Empirical entropy of token frequencies in bits; 0 if all tokens are equal."""
    if not tokens:
        return 0.0
    counts = np.array(list(Counter(tokens).values()), dtype=np.float64)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def top_token_fraction(tokens: Sequence[int]) -> float:
    """Share of the most frequent token; 1.0 if all tokens are equal."""
    if not tokens:
        return 0.0
    return Counter(tokens).most_common(1)[0][1] / len(tokens)


def sequence_metrics(tokens: Sequence[int]) -> dict[str, float]:
    """Metrics of one sequence."""
    return {
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "distinct_3": distinct_n(tokens, 3),
        "distinct_4": distinct_n(tokens, 4),
        "rep4_coverage": repeated_ngram_coverage(tokens, 4),
        "max_run": float(max_run_length(tokens)),
        "top_token_frac": top_token_fraction(tokens),
        "entropy_bits": unigram_entropy_bits(tokens),
        "unique_tokens": float(len(set(tokens))),
        "all_same": float(len(set(tokens)) <= 1),
    }


def batch_metrics(batch: Iterable[Sequence[int]]) -> dict[str, float]:
    """Mean of the per-sequence metrics, plus a cross-sample duplicate rate.

    `cross_sample_distinct_4` checks whether different samples emit the same
    4-grams: mode collapse (every draw gives the same text) is invisible to
    within-sequence metrics.
    """
    seqs = [list(s) for s in batch]
    per = [sequence_metrics(s) for s in seqs]
    keys = per[0].keys()
    out = {k: float(np.mean([m[k] for m in per])) for k in keys}
    out["max_run_max"] = float(max(m["max_run"] for m in per))
    concat_grams = [
        tuple(s[i : i + 4]) for s in seqs for i in range(max(0, len(s) - 3))
    ]
    out["cross_sample_distinct_4"] = (
        len(set(concat_grams)) / len(concat_grams) if concat_grams else 1.0
    )
    return out


def is_monotone_nondecreasing(values: Sequence[float], tol: float = 1e-9) -> bool:
    """Whether `values` is non-decreasing up to `tol`."""
    return all(b >= a - tol for a, b in zip(values, values[1:]))


def monotone_toward(
    values: Sequence[float], reference: float, tol: float = 1e-9
) -> dict[str, bool]:
    """Whether `values` move monotonically toward `reference`, and whether they cross it.

    Which direction is an improvement depends on the metric: distinct-4 and
    entropy approach the real-text value from above (too random), repetition
    from below. Monotonicity is therefore judged as approach to the reference;
    crossing it (overshoot) is reported separately.
    """
    if not values:
        return {"toward_reference": True, "overshoot": False}
    dists = [abs(v - reference) for v in values]
    signs = {v > reference for v in values}
    return {
        "toward_reference": is_monotone_nondecreasing([-d for d in dists], tol),
        "overshoot": len(signs) > 1,
    }


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def build_model(
    flavor: str, *, alpha_max: float | None, qk_norm: bool
) -> SphereDiTForPretrain:
    """Build the model on CPU from the flavor's args with the given window and QK-norm."""
    args = dataclasses.replace(
        swaflow_args[flavor], schedule_alpha_max=alpha_max, qk_norm=qk_norm
    )
    with torch.device("cpu"):
        model = SphereDiTForPretrain(args)
    return model


def load_dcp_weights(model: torch.nn.Module, ckpt: Path) -> dict[str, int]:
    """Load only the model weights from a DCP checkpoint, in place.

    Model-only (`snapshots/step-N`) and full-state (`checkpoint/step-N`)
    checkpoints both store the model weights under flat unprefixed keys. DCP
    reads only the requested keys, so optimizer, dataloader and train state in a
    full state are not touched.

    Returns:
        {"loaded": tensors read, "in_checkpoint": keys in the checkpoint}
    """
    import torch.distributed.checkpoint as dcp

    md = dcp.FileSystemReader(str(ckpt)).read_metadata()
    ckpt_keys = set(md.state_dict_metadata.keys())

    sd = model.state_dict()
    # freqs_cis is non-persistent and so not in state_dict; any other model key
    # missing from the checkpoint is reported explicitly.
    missing = [k for k in sd if k not in ckpt_keys]
    if missing:
        raise RuntimeError(
            f"the checkpoint lacks {len(missing)} model keys: "
            f"{missing[:5]} ... "
            "(does the QK-norm setting match the checkpoint?)"
        )
    dcp.load(sd, checkpoint_id=str(ckpt))
    model.load_state_dict(sd, strict=True)
    return {"loaded": len(sd), "in_checkpoint": len(ckpt_keys)}


# --------------------------------------------------------------------------
# One generation setting
# --------------------------------------------------------------------------


def run_one(
    model: SphereDiTForPretrain,
    *,
    steps: int,
    temperature: float,
    noise: torch.Tensor,
    naive_full_grid: bool,
    chunk_size: int | None,
    seed: int,
    prefix_tokens: torch.Tensor | None = None,
    prefix_lengths: torch.Tensor | None = None,
    noise_removal: str = "greedy",
    velocity: str = "exact",
    micro_batch: int = 0,
) -> dict[str, Any]:
    """Generate one setting and return its metrics and samples.

    `micro_batch` splits the sample axis into chunks of that size.
    `SamplerConfig.chunk_size` chunks only the velocity along the sequence axis;
    the logits [B, L, V] the model returns are not split (25.8 GB in fp32 at
    B=256, L=512, V=49,152, and the log_softmax intermediate is as large), so
    large sample counts need `micro_batch`.
    """
    cfg = SamplerConfig(
        steps=steps,
        temperature=temperature,
        velocity=velocity,
        noise_removal=noise_removal,
        chunk_size=chunk_size,
    )
    schedule = LogLinear() if naive_full_grid else model.schedule
    grid = alpha_grid(schedule, steps, time_eps=cfg.time_eps, device="cpu")

    n = noise.shape[0]
    mb = micro_batch if 0 < micro_batch < n else n
    if mb != n and (velocity == "sample" or noise_removal == "ancestral"):
        # Splitting changes the order in which random numbers are drawn, so
        # stochastic paths would not reproduce the unsplit result; refuse.
        raise NotImplementedError(
            "micro_batch requires velocity=exact and noise_removal=greedy"
        )

    t0 = time.time()
    parts: list[torch.Tensor] = []
    for i in range(0, n, mb):
        j = min(i + mb, n)
        parts.append(
            generate(
                model,
                config=cfg,
                schedule=schedule,
                noise=noise[i:j],
                prefix_tokens=None if prefix_tokens is None else prefix_tokens[i:j],
                prefix_lengths=None if prefix_lengths is None else prefix_lengths[i:j],
                generator=torch.Generator(device=noise.device).manual_seed(seed),
            )
        )
    tokens = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    elapsed = time.time() - t0

    seqs = tokens.cpu().tolist()
    return {
        "steps": steps,
        "nfe": steps,
        "temperature": temperature,
        "velocity": velocity,
        "noise_removal": noise_removal,
        "alpha_grid_source": "naive_full[0,1]" if naive_full_grid else "training_window",
        "alpha_min": float(grid[0]),
        "alpha_max": float(grid[-1]),
        "alpha_grid": [float(a) for a in grid] if steps <= 16 else None,
        "alpha_max_model_input": float(grid[-2]),  # the last grid point is for decoding only
        "metrics": batch_metrics(seqs),
        "micro_batch": mb,
        "elapsed_sec": elapsed,
        "tokens": seqs,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_prefixes(
    build_dir: Path, n: int, prefix_len: int, seq_len: int
) -> tuple[torch.Tensor, str]:
    """Use the first `prefix_len` tokens of held-out blocks as prefixes.

    Held-out blocks never occur in training, so memorization cannot leak into
    the prefix check.
    """
    from torchtitan.models.swaflow.validate import load_heldout

    sources = load_heldout(build_dir)
    src = max(sources, key=lambda s: s.n_blocks)
    tokens = np.asarray(src.tokens[:n, :prefix_len], dtype=np.int64)
    if tokens.shape[0] < n:
        raise RuntimeError(f"not enough held-out blocks: {tokens.shape[0]} < {n}")
    if prefix_len > seq_len:
        raise ValueError("prefix_len > seq_len")
    return torch.from_numpy(tokens), src.name


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path, help="DCP checkpoint directory")
    ap.add_argument("--flavor", default="3B")
    ap.add_argument(
        "--alpha-max",
        type=float,
        default=0.10,
        help="truncation upper end used in training; must match the training run",
    )
    ap.add_argument(
        "--qk-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="must match the checkpoint (default off)",
    )
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8, 32, 64])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument(
        "--micro-batch",
        type=int,
        default=0,
        help="split the sample axis into chunks of this size (0 = no split); logits "
        "[B, L, V] grow with B, so split when generating many samples",
    )
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--tokenizer", type=Path, default=None, help="hf_assets_path")
    ap.add_argument("--build", type=Path, default=None, help="data build dir to read held-out prefixes from")
    ap.add_argument("--prefix-len", type=int, default=64)
    ap.add_argument(
        "--prefix-samples",
        type=int,
        default=8,
        help="number of prefix-conditioned samples (default 8, capped by "
        "`--num-samples`); raise it to measure conditional Gen-PPL",
    )
    ap.add_argument(
        "--naive-full-grid",
        action="store_true",
        help="control: also run one setting on the untruncated [0,1] grid",
    )
    ap.add_argument("--n-text-samples", type=int, default=3)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[gen] building {args.flavor} (alpha_max={args.alpha_max}, qk_norm={args.qk_norm})")
    model = build_model(args.flavor, alpha_max=args.alpha_max, qk_norm=args.qk_norm)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gen] params = {n_params:,}")

    print(f"[gen] loading {args.ckpt}")
    t0 = time.time()
    load_info = load_dcp_weights(model, args.ckpt)
    print(f"[gen] loaded {load_info['loaded']} tensors in {time.time() - t0:.1f}s")

    model = model.to(device).eval()

    tokenizer = None
    if args.tokenizer is not None:
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(str(args.tokenizer))

    dim = model.model_args.dim
    # The same initial noise for every setting, so step counts are compared on paired samples.
    noise_gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(
        (args.num_samples, args.seq_len, dim), generator=noise_gen, dtype=torch.float32
    ).to(device)

    results: list[dict[str, Any]] = []
    for temperature in args.temperatures:
        for steps in args.steps:
            print(f"[gen] steps={steps} T={temperature}", flush=True)
            r = run_one(
                model,
                steps=steps,
                temperature=temperature,
                noise=noise,
                naive_full_grid=False,
                chunk_size=args.chunk_size,
                seed=args.seed,
                micro_batch=args.micro_batch,
            )
            print(
                f"       distinct_4={r['metrics']['distinct_4']:.4f} "
                f"rep4={r['metrics']['rep4_coverage']:.4f} "
                f"max_run={r['metrics']['max_run_max']:.0f} "
                f"H={r['metrics']['entropy_bits']:.3f} "
                f"({r['elapsed_sec']:.1f}s)",
                flush=True,
            )
            results.append(r)

    control: dict[str, Any] | None = None
    if args.naive_full_grid:
        steps = max(s for s in args.steps if s <= 32)
        print(f"[gen] control: naive full [0,1] grid, steps={steps} T=1.0", flush=True)
        control = run_one(
            model,
            steps=steps,
            temperature=1.0,
            noise=noise,
            naive_full_grid=True,
            chunk_size=args.chunk_size,
            seed=args.seed,
            micro_batch=args.micro_batch,
        )
        print(
            f"       distinct_4={control['metrics']['distinct_4']:.4f} "
            f"rep4={control['metrics']['rep4_coverage']:.4f} "
            f"max_run={control['metrics']['max_run_max']:.0f} "
            f"H={control['metrics']['entropy_bits']:.3f}",
            flush=True,
        )

    prefix_case: dict[str, Any] | None = None
    if args.build is not None:
        n_pref = min(args.num_samples, args.prefix_samples)
        prefix_tokens, src_name = _load_prefixes(
            args.build, n_pref, args.prefix_len, args.seq_len
        )
        print(f"[gen] prefix case: {n_pref}×{args.prefix_len} from held-out {src_name}")
        prefix_lengths = torch.full((n_pref,), args.prefix_len, dtype=torch.long)
        steps = max(args.steps)
        prefix_case = run_one(
            model,
            steps=steps,
            temperature=1.0,
            noise=noise[:n_pref],
            naive_full_grid=False,
            chunk_size=args.chunk_size,
            seed=args.seed,
            micro_batch=args.micro_batch,
            prefix_tokens=prefix_tokens,
            prefix_lengths=prefix_lengths,
        )
        out_tok = torch.tensor(prefix_case["tokens"], dtype=torch.long)
        preserved = bool(torch.equal(out_tok[:, : args.prefix_len], prefix_tokens))
        prefix_case["prefix_source"] = src_name
        prefix_case["prefix_len"] = args.prefix_len
        prefix_case["prefix_preserved"] = preserved
        # Recompute the metrics on the continuation alone.
        prefix_case["metrics_continuation"] = batch_metrics(
            [s[args.prefix_len :] for s in prefix_case["tokens"]]
        )
        print(f"       prefix_preserved = {preserved}")

    # ---- Monotonicity in the number of steps (direction not interpreted) ----
    #
    # Which direction is an improvement needs a real-text reference, so only the
    # raw directions are recorded here.
    monotonicity: dict[str, Any] = {}
    metric_names = (
        "distinct_1",
        "distinct_4",
        "rep4_coverage",
        "entropy_bits",
        "cross_sample_distinct_4",
    )
    for temperature in args.temperatures:
        rows = [r for r in results if r["temperature"] == temperature]
        rows.sort(key=lambda r: r["steps"])
        monotonicity[f"T={temperature}"] = {
            "steps": [r["steps"] for r in rows],
            **{
                m: {
                    "values": (vals := [r["metrics"][m] for r in rows]),
                    "monotone_increasing": is_monotone_nondecreasing(vals),
                    "monotone_decreasing": is_monotone_nondecreasing(
                        [-v for v in vals]
                    ),
                }
                for m in metric_names
            },
        }

    summary = {
        "ckpt": str(args.ckpt),
        "flavor": args.flavor,
        "alpha_max": args.alpha_max,
        "qk_norm": args.qk_norm,
        "n_params": n_params,
        "num_samples": args.num_samples,
        "micro_batch": args.micro_batch,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "velocity": "exact",
        "noise_removal": "greedy",
        "results": [{k: v for k, v in r.items() if k != "tokens"} for r in results],
        "control_naive_full_grid": (
            {k: v for k, v in control.items() if k != "tokens"} if control else None
        ),
        "prefix_case": (
            {k: v for k, v in prefix_case.items() if k != "tokens"}
            if prefix_case
            else None
        ),
        "monotonicity": monotonicity,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- Raw token ids and decoded text for inspection ----
    npz_arrays = {
        f"s{r['steps']}_T{r['temperature']}": np.asarray(r["tokens"], dtype=np.int32)
        for r in results
    }
    if prefix_case is not None:
        # The `pfx` key prefix tells gen_ppl.py to use the first `prefix_len`
        # tokens as context but not score them (they are real text, and
        # scoring them would inflate the conditional score).
        npz_arrays[f"pfx{prefix_case['steps']}_T{prefix_case['temperature']}"] = (
            np.asarray(prefix_case["tokens"], dtype=np.int32)
        )
    np.savez_compressed(args.out / "tokens.npz", **npz_arrays)
    if tokenizer is not None:
        lines: list[str] = []
        blocks = [(f"steps={r['steps']} T={r['temperature']}", r) for r in results]
        if control is not None:
            blocks.append(("CONTROL naive full [0,1] grid", control))
        if prefix_case is not None:
            blocks.append(
                (
                    f"PREFIX (first {args.prefix_len} tokens are real held-out text)",
                    prefix_case,
                )
            )
        for title, r in blocks:
            lines.append("=" * 78)
            lines.append(
                f"{title}  alpha_grid=[{r['alpha_min']:.3e}, {r['alpha_max']:.4f}]"
            )
            lines.append("=" * 78)
            for i, seq in enumerate(r["tokens"][: args.n_text_samples]):
                lines.append(f"--- sample {i} ---")
                lines.append(tokenizer.decode(seq))
                lines.append("")
        (args.out / "samples.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"[gen] wrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
