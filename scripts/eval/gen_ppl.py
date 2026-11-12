# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Gen-PPL: score generated text with an external autoregressive model.

For each setting in a `tokens.npz` written by `generate.py`, reports the mean NLL
and perplexity under an external AR model (default SmolLM2-1.7B). SmolLM2 shares
our tokenizer (V=49,152, identical ids), so no re-tokenization is needed
(re-tokenizing would change token counts and shift PPL between settings).

Read Gen-PPL next to the real-text floor and the repetition metrics: degenerate
repetitive text gets very low PPL (`\\n\\n\\n...` is fully predictable).
One BOS (`<|endoftext|>`, id 0) is prepended and all `seq_len` generated tokens
are scored; the real-text floor is scored the same way.

    python -m scripts.eval.gen_ppl \\
        --gen 150M=<out>/150M/step238400 3B=<out>/3B/step238400 \\
        --build <build> --out <out>/gen_ppl.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def score_batches(
    model,
    tokens: np.ndarray,
    *,
    bos_id: int,
    batch_size: int,
    device: torch.device,
    score_from: int = 0,
) -> dict[str, float]:
    """Score [N, L] token sequences with an AR model.

    The first `score_from` tokens are used as context but not scored, so the
    real held-out prefix of a prefix-conditioned generation does not enter its
    score (it would add the same offset to every setting and dilute the
    differences).

    Returns:
        Mean NLL (nats/token), PPL, and the std and percentiles of per-sequence
        NLL, which show whether a few extreme sequences drive the mean
        (degenerate sequences have NLL lower by orders of magnitude).
    """
    if score_from >= tokens.shape[1]:
        raise ValueError(f"score_from={score_from} is not below the sequence length ({tokens.shape[1]})")
    per_seq: list[float] = []
    total_nll = 0.0
    total_tok = 0
    for i in range(0, len(tokens), batch_size):
        chunk = torch.from_numpy(tokens[i : i + batch_size].astype(np.int64))
        bos = torch.full((chunk.shape[0], 1), bos_id, dtype=torch.long)
        inp = torch.cat([bos, chunk], dim=1).to(device)
        with torch.no_grad():
            logits = model(input_ids=inp).logits.float()
        # Logits at position j predict the token at j+1; with BOS prepended,
        # every one of the L generated tokens has a prediction.
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = inp[:, 1:]
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, L]
        nll = nll[:, score_from:]
        per_seq.extend(nll.mean(dim=1).cpu().tolist())
        total_nll += float(nll.sum().item())
        total_tok += int(nll.numel())
    arr = np.asarray(per_seq)
    mean_nll = total_nll / total_tok
    return {
        "n_seq": int(len(tokens)),
        "score_from": int(score_from),
        "n_tokens": total_tok,
        "nll_nat": mean_nll,
        "ppl": math.exp(mean_nll),
        "nll_std_over_seq": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "nll_p05": float(np.percentile(arr, 5)),
        "nll_p50": float(np.percentile(arr, 50)),
        "nll_p95": float(np.percentile(arr, 95)),
        "ppl_of_median_seq": math.exp(float(np.percentile(arr, 50))),
    }


def heldout_pool(source, seq_len: int) -> np.ndarray:
    """Cut one held-out source into contiguous `seq_len` slices, [K, seq_len].

    Every block (length 2048) yields as many slices as fit. Taking only its
    first `seq_len` tokens would cap the count at the number of blocks, and
    small sources would leave the reference with fewer sequences than the
    generated side.
    """
    arr = np.asarray(source.tokens, dtype=np.int64)
    per_block = arr.shape[1] // seq_len
    if per_block < 1:
        raise ValueError(
            f"held-out block length {arr.shape[1]} is shorter than seq_len {seq_len}."
        )
    return arr[:, : per_block * seq_len].reshape(-1, seq_len)


def _round_robin(
    pools: dict[str, np.ndarray], order: Sequence[str], n: int
) -> list[np.ndarray]:
    """Collect n sequences, one from each source in turn (skipping exhausted ones).

    With an even split (`n // len(sources)`) the share of a small source would
    go unfilled; round-robin passes it on to the larger sources.
    """
    picked: list[np.ndarray] = []
    cursor = {name: 0 for name in order}
    while len(picked) < n:
        progressed = False
        for name in order:
            if len(picked) >= n:
                break
            i = cursor[name]
            if i < len(pools[name]):
                picked.append(pools[name][i])
                cursor[name] = i + 1
                progressed = True
        if not progressed:
            total = sum(len(p) for p in pools.values())
            raise RuntimeError(
                f"not enough held-out sequences: only {total} available, {n} needed"
            )
    return picked


def real_text_tokens(build_dir: Path, n: int, seq_len: int) -> np.ndarray:
    """Real-text floor sequences: held-out blocks drawn round-robin across sources."""
    from torchtitan.models.swaflow.validate import load_heldout

    sources = load_heldout(build_dir)
    pools = {s.name: heldout_pool(s, seq_len) for s in sources}
    order = [s.name for s in sorted(sources, key=lambda s: -s.n_blocks)]
    return np.stack(_round_robin(pools, order, n), axis=0)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gen",
        nargs="+",
        required=True,
        metavar="LABEL=DIR",
        help="generation directory containing `tokens.npz`, as `150M=/path/...`",
    )
    ap.add_argument("--build", type=Path, required=True, help="data build whose held-out split gives the real-text floor")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--bos-id", type=int, default=0, help="<|endoftext|>")
    ap.add_argument(
        "--prefix-len",
        type=int,
        default=64,
        help="for `pfx*` keys (prefix-conditioned generation): number of leading "
        "tokens used as context but not scored",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"[ppl] loading evaluator {args.model} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(device).eval()
    n_eval = sum(p.numel() for p in model.parameters())
    print(f"[ppl] evaluator params = {n_eval:,}", flush=True)

    results: dict[str, dict] = {}
    shapes: set[tuple[int, int]] = set()

    for spec in args.gen:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--gen expects LABEL=DIR: {spec}")
        npz = np.load(Path(path) / "tokens.npz")
        results[label] = {"dir": path, "conditions": {}}
        for key in sorted(
            npz.keys(),
            key=lambda k: (
                k.startswith("pfx"),
                float(k.split("_T")[1]),
                int("".join(c for c in k.split("_T")[0] if c.isdigit())),
            ),
        ):
            toks = npz[key]
            shapes.add(tuple(toks.shape))
            # Prefix settings: the first prefix_len tokens are real text, not scored.
            score_from = args.prefix_len if key.startswith("pfx") else 0
            r = score_batches(
                model,
                toks,
                bos_id=args.bos_id,
                batch_size=args.batch_size,
                device=device,
                score_from=score_from,
            )
            results[label]["conditions"][key] = r
            print(
                f"[ppl] {label:6s} {key:12s} ppl={r['ppl']:9.2f} "
                f"nll={r['nll_nat']:.4f} (p05 {r['nll_p05']:.3f} / "
                f"p95 {r['nll_p95']:.3f})",
                flush=True,
            )

    if len(shapes) != 1:
        print(f"[ppl] WARNING: sample shapes differ: {sorted(shapes)}")
    n, seq_len = sorted(shapes)[0]

    real = real_text_tokens(args.build, n, seq_len)
    floor = score_batches(
        model, real, bos_id=args.bos_id, batch_size=args.batch_size, device=device
    )
    print(f"[ppl] REAL held-out {n}×{seq_len}: ppl={floor['ppl']:.2f}", flush=True)
    # Floor comparable to the prefix settings: first prefix_len tokens as context.
    floor_pfx = score_batches(
        model,
        real,
        bos_id=args.bos_id,
        batch_size=args.batch_size,
        device=device,
        score_from=args.prefix_len,
    )
    print(
        f"[ppl] REAL (prefix {args.prefix_len} tokens as context, rest scored): "
        f"ppl={floor_pfx['ppl']:.2f}",
        flush=True,
    )

    payload = {
        "evaluator": args.model,
        "evaluator_params": n_eval,
        "dtype": args.dtype,
        "bos_id": args.bos_id,
        "scoring": "one BOS prepended, all L generated tokens scored "
        "(for `pfx*` keys the first prefix_len tokens are context only)",
        "shapes": sorted(f"{a}x{b}" for a, b in shapes),
        "build": str(args.build),
        "real_text_floor": floor,
        "real_text_floor_from_prefix": floor_pfx,
        "prefix_len": args.prefix_len,
        "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ppl] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
