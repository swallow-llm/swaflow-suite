# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Check our model loaded from the released S-FLM OpenWebText checkpoint.

Compares our forward pass with an fp32 reference forward that
`export_sflm_owt.py` computes from the same (EMA) weights in the S-FLM
environment (logits must agree to max abs 1e-3). Two checkpoint-specific points:

1. Time convention: this checkpoint conditions on σ = −log α, while our
   `alpha_to_sigma` computes σ = −log(1 − α), so the model's time input is
   1 − α (`run_mcq_external.sflm_time_cond`); x_t is built in our
   convention. The logits without this mapping are also reported, to show the
   check is sensitive to it.
2. Tokenizer: whether HF `openai-community/gpt2` and the tokenizer pickled in
   the checkpoint return the same ids (compared on the S-FLM side).

    python -m scripts.external.check_sflm_owt --dir <--out of export_sflm_owt.py>
"""

from __future__ import annotations

import argparse
import sys

import torch

from scripts.external.run_mcq_external import build_sflm_owt, sflm_time_cond
from torchtitan.models.swaflow.model import SphereDiT
from torchtitan.models.swaflow.sphere_ops import slerp, sphere_normalize

LOGITS_TOL = 1e-3


def _report(name, got, want, tol) -> bool:
    diff = (got.double() - want.double()).abs()
    mx = diff.max().item()
    ok = mx <= tol
    print(f"  {name:34s} max={mx:.3e}  mean={diff.mean().item():.3e}  "
          f"(tol {tol:.0e}) {'OK' if ok else 'NG'}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args(argv)

    ref = torch.load(f"{args.dir}/reference.pt", weights_only=False)
    print("=== S-FLM OWT: weight mapping + time convention check (EMA weights, fp32) ===")
    print(f"  meta: {ref['meta']}")

    model = build_sflm_owt(f"{args.dir}/weights_ema.pt", dtype=torch.float32)
    model.eval()

    tokens, alpha = ref["tokens"], ref["alpha"]
    b, L = tokens.shape
    plen = int(ref["prefix_len"])
    ok = True

    with torch.no_grad():
        x1 = model.get_sphere_embeddings(tokens)
        x0 = sphere_normalize(ref["noise"])
        # Samples 0-2: the whole sequence at α (slerp in our convention).
        x_t = slerp(x0.double(), x1.double(),
                    alpha.double().view(b, 1, 1)).float()
        # Sample 3: per-position α as in the harness (prefix 1.0, rest 0.0).
        a_pos = torch.zeros((L, 1), dtype=torch.float32)
        a_pos[:plen] = 1.0
        x_t[3] = slerp(x0[3], x1[3], a_pos)
        ok &= _report("x_t (slerp, 4 samples)", x_t, ref["x_t"], 1e-5)

        logits = SphereDiT.forward(model, ref["x_t"], sflm_time_cond(alpha))
        ok &= _report("logits (time_cond = 1 - α)", logits, ref["logits"],
                      LOGITS_TOL)
        wrong = SphereDiT.forward(model, ref["x_t"], alpha)
        d_wrong = (wrong - ref["logits"]).abs().max().item()
        print(f"  (without the mapping, i.e. σ=−log(1−α): max={d_wrong:.3e}; "
              f"{'check is sensitive' if d_wrong > 100 * LOGITS_TOL else 'check is NOT sensitive'})")
        ok &= d_wrong > 100 * LOGITS_TOL

    ce = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), tokens.flatten(0, 1), reduction="none").view(b, L)
    print(f"  CE per sample titan={[round(v, 4) for v in ce.mean(1).tolist()]}  "
          f"reference={[round(v, 4) for v in ref['ce_per_sample']]}")
    print(f"  CE after the clean prefix titan={ce[3, plen:].mean().item():.4f}  "
          f"reference={ref['ce_clean_prefix_cont']:.4f}")

    ids_ck = ref["ids_ckpt_tokenizer"]
    if isinstance(ids_ck, list):
        same = ids_ck == ref["ids_hf_gpt2"]
        print(f"  tokenizer: HF gpt2 ids match the checkpoint's training tokenizer = {same}")
        ok &= same
    else:
        print(f"  tokenizer: cannot compare with the checkpoint tokenizer ({ids_ck}); using HF gpt2")

    print("CHECK_SFLM_OWT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
