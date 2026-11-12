# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Training entry point for the AR models.

    torchrun --standalone --nproc_per_node=8 -m torchtitan.models.swaflow.ar.train \\
        --job.config_file <config.toml> --job.dump_folder <dir>

Uses the flow `SwaFlowTrainer` unchanged: its checkpoint-schedule sharing,
diagnostics and per-rank random streams do not depend on the model, so the AR
models get the same checkpointing, watchdog and resume path.

Importing `torchtitan.models.swaflow.ar` registers the TrainSpec; set
`[model] name = "swaflow_ar"` in the config.

Without α, `alpha_ce/` logging and the watchdog's α-contrast and CE-ceiling
conditions are inactive (configs also set `stop_clip_rate = 0`); attention
statistics and gradient-norm logging work as usual. The remaining stop
conditions are the non-finite count and the spike rate.
"""

import torchtitan.models.swaflow.ar  # noqa: F401  (registers `swaflow_ar`)
from torchtitan.models.swaflow.train import SwaFlowTrainer
from torchtitan.train import main

__all__ = ["ARTrainer"]


class ARTrainer(SwaFlowTrainer):
    """Identical to `SwaFlowTrainer`; a separate name makes AR runs easy to
    recognize in job logs."""


if __name__ == "__main__":
    main(ARTrainer)
