# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Export the weights of a training state to a Hugging Face Hub layout.

Reads only the model tensors of a torchtitan DCP checkpoint (optimizer and data
state are left out) and writes `config.json` + `model.safetensors`, loadable with
`torchtitan.models.swaflow.hf.from_pretrained`.

    python -m scripts.convert_to_hf --checkpoint $SWAFLOW_OUTPUTS/flow_150m/checkpoint/step-238400 \\
        --kind flow --flavor 150M --alpha-max 0.15 --out hf/SwaFlow-150M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.eval.loader import build_eval_model, load_checkpoint_distributed
from torchtitan.models.swaflow.hf import save_pretrained

TOKENS_PER_STEP = 4096 * 2048


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="<dump>/checkpoint/step-N (a DCP directory)")
    ap.add_argument("--kind", choices=["flow", "ar"], required=True)
    ap.add_argument("--flavor", required=True, help="150M / 500M / 1B / 3B")
    ap.add_argument("--alpha-max", type=float, default=None, help="window edge of a flow model")
    ap.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not (ckpt / ".metadata").is_file():
        raise SystemExit(f"not a DCP checkpoint: {ckpt}")
    if args.kind == "flow" and args.alpha_max is None:
        raise SystemExit("--alpha-max is required for a flow model")
    step = int(ckpt.name.removeprefix("step-")) if ckpt.name.startswith("step-") else None

    model = build_eval_model(args.kind, args.flavor, alpha_max=args.alpha_max)
    load_checkpoint_distributed(model, ckpt)
    extra = {"training": {"step": step, "tokens": step * args.tokens_per_step if step else None}}
    out = save_pretrained(model, args.out, kind=args.kind, flavor=args.flavor, extra=extra)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {out} ({n / 1e6:.1f}M parameters, step {step})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
