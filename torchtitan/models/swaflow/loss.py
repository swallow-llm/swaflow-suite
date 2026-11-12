# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

import os

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import JobConfig
from torchtitan.models.swaflow.metrics import AlphaCELogger, get_attention_stats
from torchtitan.tools.logging import logger


def _denoising_ce(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss_sum, per_sample_ce); a pure function, compiled with torch.compile.

    Logits at position i predict the clean token at position i (not the next
    token). The dataloader returns next-token-shifted labels
    (labels[i] = input[i+1]), so pred[:, i] is paired with labels[:, i−1]. The
    first token is dropped from the loss (~0.05% at seq_len 2048).

    CE is computed on fp32 logits. per_sample_ce is the mean over valid tokens
    (ignore_index positions contribute 0).
    """
    b = logits.shape[0]
    target = labels[:, :-1]
    ce = torch.nn.functional.cross_entropy(
        logits[:, 1:].flatten(0, 1).float(),
        target.flatten(0, 1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(b, -1)
    loss_sum = ce.sum()
    n_valid = (target != IGNORE_INDEX).sum(dim=-1).clamp_min(1)
    per_sample_ce = ce.sum(dim=-1) / n_valid
    return loss_sum, per_sample_ce


def swaflow_denoising_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Denoising CE (sum reduction), without logging."""
    return _denoising_ce(pred, labels)[0]


def build_swaflow_denoising_loss(job_config: JobConfig, **kwargs):
    """Build the training loss.

    The model (SphereDiTForPretrain) returns (logits, alpha). (α, per-sample CE) is
    always logged by AlphaCELogger, outside the compiled region (only
    _denoising_ce is compiled).

    Attention-statistics microbatch boundaries are also marked here: the loss is
    called exactly once per microbatch, while attention forward cannot see the
    boundaries.
    """
    del kwargs
    alpha_ce_logger = AlphaCELogger(
        os.path.join(job_config.job.dump_folder, "alpha_ce")
    )
    core = _denoising_ce
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the swaflow denoising loss with torch.compile")
        core = torch.compile(core, backend=job_config.compile.backend)

    def loss_fn(pred, labels: torch.Tensor) -> torch.Tensor:
        logits, alpha = pred
        loss_sum, per_sample_ce = core(logits, labels)
        alpha_ce_logger.record(alpha, per_sample_ce)
        stats = get_attention_stats()
        if stats is not None:
            # Finalize this microbatch's attention record and decide whether to
            # measure the next one.
            stats.end_microbatch()
            stats.begin_microbatch()
        return loss_sum

    stats = get_attention_stats()
    if stats is not None:
        stats.begin_microbatch()  # once, before the first forward

    return loss_fn
