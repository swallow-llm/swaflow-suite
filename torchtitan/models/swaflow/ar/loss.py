# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Loss for the AR models: next-token cross-entropy.

The `indexed` dataloader returns labels[j] = block[j+1] with IGNORE_INDEX at the
end, which is already the AR convention:

    logits[:, j]  ↔  labels[:, j]  ( = block[j+1] )

with no shift. The last position is dropped (~0.05% at seq_len 2048). The flow
loss instead pairs logits[:, 1:] with labels[:, :-1] and drops the first position
(see `test_ar_loss_predicts_the_next_token`).

The uniform-prediction CE is ln V = 10.8027. The flow's unigram CE (7.53) is its
α -> 0 limit and has no AR counterpart; do not compare AR loss against it.
"""

import torch

from torchtitan.components.loss import cross_entropy_loss, IGNORE_INDEX
from torchtitan.config import JobConfig
from torchtitan.models.swaflow.metrics import get_attention_stats
from torchtitan.tools.logging import logger

__all__ = ["ar_next_token_loss", "build_ar_next_token_loss"]


def ar_next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token CE (sum reduction).

    Upstream `cross_entropy_loss` (CE on fp32 logits; `ignore_index` drops the
    last position), wrapped under its own name because, unlike the flow loss, it
    does not shift.
    """
    return cross_entropy_loss(logits, labels)


def build_ar_next_token_loss(job_config: JobConfig, **kwargs):
    """`build_loss_fn` of the TrainSpec.

    Of what `build_swaflow_denoising_loss` does, only the attention-statistics
    microbatch boundaries are kept. There is no α, so there is no (α, per-sample CE)
    logging: `alpha_ce/` is not written and the watchdog's α-based conditions
    receive no data and stay inactive.
    """
    del kwargs
    core = ar_next_token_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the AR next-token loss with torch.compile")
        core = torch.compile(core, backend=job_config.compile.backend)

    def loss_fn(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_sum = core(pred, labels)
        stats = get_attention_stats()
        if stats is not None:
            stats.end_microbatch()
            stats.begin_microbatch()
        return loss_sum

    stats = get_attention_stats()
    if stats is not None:
        stats.begin_microbatch()  # once, before the first forward

    return loss_fn


def n_valid_targets(labels: torch.Tensor) -> torch.Tensor:
    """Number of positions that enter the loss (not `IGNORE_INDEX`); shared with evaluation."""
    return (labels != IGNORE_INDEX).sum()
