# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Prefix probe: can a scoring rule tell a document's continuation from another's?

A synthetic multiple-choice task with a very large effect size. The prefix is the
first P tokens of a held-out document, the correct choice its next L tokens, and
the distractors L tokens from other documents. A model that uses its context should
be almost always right, so chance-level accuracy here means the score does not use
the prefix, not that the task is hard. The probe sweeps α, the prefix mode (clean:
prefix held at α=1; noised: the whole sequence noised at α) and the model kind (AR
validates the task itself), with and without the PMI correction.

    torchrun --nproc_per_node=1 -m scripts.eval.prefix_probe \\
        --dump <dump> --kind flow --flavor 3B --alpha-max 0.10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.eval.ckpt_eval import _cast_params
from scripts.eval.loader import build_eval_model, load_checkpoint_distributed
from scripts.eval.mcq import Doc, Request, score_ar, score_flow, score_accuracy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--kind", choices=["flow", "ar"], default="flow")
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--step", type=int, default=0, help="0 selects the latest checkpoint")
    ap.add_argument("--build", default="")
    ap.add_argument("--n-docs", type=int, default=128)
    ap.add_argument("--n-choices", type=int, default=4)
    ap.add_argument("--prefix-len", type=int, default=64)
    ap.add_argument("--cont-len", type=int, default=32)
    ap.add_argument("--alphas", default="")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--neutral-token", type=int, default=198,
                    help="token id that replaces the prefix in the PMI baseline (default: newline)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    from scripts.eval.loader import checkpoint_steps
    from torchtitan.models.swaflow.validate import load_heldout

    dump = Path(args.dump)
    step = args.step or checkpoint_steps(dump)[-1]
    build = args.build or json.loads(
        sorted((dump / "val").glob("step-*.json"))[0].read_text()
    )["build_dir"]

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)

    sources = load_heldout(build)
    src = max(sources, key=lambda s: s.n_blocks)
    need = args.n_docs * args.n_choices
    toks = np.asarray(src.tokens[:need], dtype=np.int64)

    # Question i: prefix from doc i; choices continue doc i (correct) or other docs.
    rng = np.random.default_rng(args.seed)
    docs, reqs, idx = [], [], 0
    P, L = args.prefix_len, args.cont_len
    for i in range(args.n_docs):
        others = [j for j in range(args.n_docs) if j != i]
        wrong = rng.choice(others, size=args.n_choices - 1, replace=False)
        gold = int(rng.integers(args.n_choices))
        picks = list(wrong)
        picks.insert(gold, i)
        docs.append(Doc([""] * args.n_choices, [""] * args.n_choices, gold))
        for c, j in enumerate(picks):
            reqs.append(Request(
                i, c, list(toks[i, :P]), list(toks[j, P : P + L]), L * 4, idx
            ))
            idx += 1

    model = build_eval_model(args.kind, args.flavor, alpha_max=args.alpha_max)
    model.to(device=device)
    _cast_params(model, torch.float32)
    model.eval()
    load_checkpoint_distributed(model, dump / "checkpoint" / f"step-{step}",
                                device=device)
    _cast_params(model, torch.bfloat16)

    out = {"dump": str(dump), "kind": args.kind, "flavor": args.flavor,
           "step": step, "n_docs": args.n_docs, "n_choices": args.n_choices,
           "prefix_len": P, "cont_len": L, "chance": 1.0 / args.n_choices,
           "results": []}

    if args.kind == "ar":
        nll = score_ar(model, reqs, device=device, batch_size=args.batch,
                       n_total=len(reqs))
        acc = score_accuracy(docs, reqs, nll)
        out["results"].append({"mode": "ar", **acc})
        print(f"[probe] AR: acc {acc['acc']:.4f} (chance {1 / args.n_choices:.2f})",
              flush=True)
    else:
        amax = args.alpha_max or 0.1
        alphas = ([float(x) for x in args.alphas.replace(":", ",").split(",")]
                  if args.alphas
                  else [0.0] + [round(amax * k / 8, 6) for k in range(1, 9)]
                  + [0.25, 0.5, 1.0])
        for mode in ("clean", "noised"):
            for a in alphas:
                nll = score_flow(model, reqs, alpha_grid=[a], seed=args.seed,
                                 task_index=0, device=device,
                                 batch_size=args.batch, n_total=len(reqs),
                                 prefix_mode=mode)
                # PMI: subtract the score with the context replaced by a neutral
                # token. Choice tokens and noise are identical, so per-token
                # reconstruction difficulty cancels.
                nll0 = score_flow(model, reqs, alpha_grid=[a], seed=args.seed,
                                  task_index=0, device=device,
                                  batch_size=args.batch, n_total=len(reqs),
                                  prefix_mode=mode,
                                  neutral_token=args.neutral_token)
                acc_pmi = score_accuracy(docs, reqs, nll - nll0)
                acc = score_accuracy(docs, reqs, nll)
                acc["acc_pmi"] = acc_pmi["acc"]
                gold_nll = float(np.mean(
                    [nll[r.index] for r in reqs
                     if r.choice_id == docs[r.doc_id].gold]))
                bad_nll = float(np.mean(
                    [nll[r.index] for r in reqs
                     if r.choice_id != docs[r.doc_id].gold]))
                rec = {"mode": mode, "alpha": a, "gold_nll_per_tok": gold_nll / L,
                       "wrong_nll_per_tok": bad_nll / L,
                       "gap_per_tok": (bad_nll - gold_nll) / L, **acc}
                out["results"].append(rec)
                print(f"[probe] {mode:6s} a={a:<9.6g} acc {acc['acc']:.4f}  "
                      f"pmi {acc['acc_pmi']:.4f}  "
                      f"gold {gold_nll / L:8.4f}  wrong {bad_nll / L:8.4f}  "
                      f"gap {(bad_nll - gold_nll) / L:+8.4f}", flush=True)

    dest = Path(args.out or (dump / f"prefix_probe_step{step}_{args.kind}.json"))
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
