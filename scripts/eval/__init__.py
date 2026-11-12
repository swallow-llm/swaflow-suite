# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Offline evaluation harness for trained checkpoints.

Unlike `torchtitan/models/swaflow/validate.py`, which runs during training, these
tools read full training states after the fact. Flow and AR models share one
interface (`kind = "flow" | "ar"`), so task definitions, prompts and scoring are
identical for both. Main entry points: `ckpt_eval` (held-out CE on an α grid and
embedding geometry), `run_mcq`, `gsm8k`, `generate` and `gen_ppl`, `collapse`,
`prefix_probe`, `sft_val`, `equivalence` (α_eq).
"""
