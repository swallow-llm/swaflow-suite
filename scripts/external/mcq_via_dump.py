# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Score an external model on our MCQ tasks through tensor dumps.

The target is SFCD (HF `Jugc/SFCD`, OpenWebText). We do not reimplement it: its
own code runs in its own environment and returns only summed choice CEs.

    1. export (our environment): build requests as the harness does; write the
       token ids, choice spans and per-question noise seeds
    2. forward (SFCD environment, `sfcd_mcq_forward.py`): clean prefix and
       pure-noise choices, one forward at SFCD's pure-noise time input; returns
       the choice CE with the prefix and with the prefix replaced by a neutral token
    3. score (our environment): PMI difference, length normalization and standard
       errors (`mcq.score_accuracy`); output has the format of `run_mcq.py`

The score is that of `mcq.score_flow` with clean prefix, PMI and α=0.

    python -m scripts.external.mcq_via_dump export --out <req.pt>
    <SFCD env>/python sfcd_mcq_forward.py --requests <req.pt> --out <ce.pt> ...
    python -m scripts.external.mcq_via_dump score --requests <req.pt> --ce <ce.pt> \\
        --out-dir $SWAFLOW_OUTPUTS/ext_sfcd_owt_ema --out-name mcq_ext_clean_pmi_a0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.eval import mcq as mcqlib
from scripts.eval.run_mcq import DEFAULT_TASKS
from torchtitan.models.swaflow.validate import noise_seed


def _digest(reqs) -> str:
    h = hashlib.sha256()
    for r in reqs:
        h.update(np.asarray(r.ctx_ids + [-1] + r.cont_ids + [-2],
                            dtype=np.int64).tobytes())
    return h.hexdigest()


def _build(tasks, tokenizer, max_len, limit):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    out = {}
    for ti, name in enumerate(tasks):
        docs = mcqlib.load_task(name, limit=limit)
        reqs = mcqlib.build_requests(docs, tok, max_len=max_len)
        out[name] = (ti, docs, reqs)
    return tok, out


def cmd_export(args) -> int:
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    tok, built = _build(tasks, args.tokenizer, args.max_len, args.limit)
    assert tok.decode([args.neutral_token]) == "\n", "neutral token is not a newline"
    payload = {
        "meta": {"tokenizer": args.tokenizer, "max_len": args.max_len,
                 "seed": args.seed, "neutral_token": args.neutral_token,
                 "limit": args.limit, "tasks": tasks,
                 "noise": "noise_seed(seed, task_index, doc_id, 0) per request; "
                          "randn((L, d)) on a CUDA generator, then L2-normalised"},
        "tasks": {},
    }
    for name, (ti, docs, reqs) in built.items():
        payload["tasks"][name] = {
            "task_index": ti,
            "ctx": [r.ctx_ids for r in reqs],
            "cont": [r.cont_ids for r in reqs],
            "noise_seed": [noise_seed(args.seed, ti, r.doc_id, 0) for r in reqs],
            "digest": _digest(reqs),
        }
        print(f"[export] {name}: {len(docs)} docs / {len(reqs)} requests "
              f"/ digest {payload['tasks'][name]['digest'][:12]}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_score(args) -> int:
    req = torch.load(args.requests, weights_only=True)
    # The SFCD-side dump stores `torch.__version__` as a TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ce = torch.load(args.ce, weights_only=True)
    m = req["meta"]
    tasks = m["tasks"]
    _, built = _build(tasks, m["tokenizer"], m["max_len"], m["limit"])
    results = {}
    for name, (ti, docs, reqs) in built.items():
        d = ce["tasks"][name]
        # The sequences SFCD scored must equal the ones rebuilt here.
        assert d["digest"] == req["tasks"][name]["digest"] == _digest(reqs), name
        prefix = np.asarray(d["ce_prefix"], dtype=np.float64)
        neutral = np.asarray(d["ce_neutral"], dtype=np.float64)
        assert prefix.shape == neutral.shape == (len(reqs),)
        nll = prefix - neutral if args.pmi else prefix
        results[name] = mcqlib.score_accuracy(docs, reqs, nll)
        print(f"[score] {name}: acc_norm_tok={results[name]['acc_norm_tok']:.4f}",
              flush=True)
    step = int(ce["meta"].get("step", 0))
    mean_tok = float(np.mean([r["acc_norm_tok"] for r in results.values()]))
    payload = {
        "step": step,
        "kind": "flow",
        "flavor": "external",
        "alpha_grid": [0.0],
        "alpha_max": None,
        "seed": m["seed"],
        "prefix_mode": "clean",
        "prompt_template": "none",
        "noise_draws": 1,
        "pmi": bool(args.pmi),
        "tasks": results,
        "mean_acc": float(np.mean([r["acc"] for r in results.values()])),
        "mean_acc_norm_tok": mean_tok,
        "mean_acc_norm_byte": float(
            np.mean([r["acc_norm_byte"] for r in results.values()])),
        "external": {**ce["meta"], "requests": str(args.requests),
                     "ce": str(args.ce), "tokenizer_used": m["tokenizer"],
                     "max_len": m["max_len"],
                     "neutral_token": m["neutral_token"]},
    }
    out = Path(args.out_dir) / args.out_name / f"step-{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    per = "  ".join(f"{n}={r['acc_norm_tok']:.3f}" for n, r in results.items())
    print(f"[score] mean {mean_tok:.4f}  {per}  -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--tokenizer", default="openai-community/gpt2")
    e.add_argument("--max-len", type=int, default=1024)
    e.add_argument("--tasks", default=DEFAULT_TASKS)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--seed", type=int, default=1234)
    e.add_argument("--neutral-token", type=int, default=198)
    s = sub.add_parser("score")
    s.add_argument("--requests", required=True)
    s.add_argument("--ce", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--out-name", default="mcq_ext_clean_pmi_a0")
    s.add_argument("--no-pmi", dest="pmi", action="store_false")
    args = ap.parse_args(argv)
    return cmd_export(args) if args.cmd == "export" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
