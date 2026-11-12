# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Held-out evaluation at each checkpoint for the AR models.

Subclasses `SwaFlowValidator` and changes only `_eval_source`, which measures
plain next-token CE once instead of CE over an α grid with deterministic noise.
Loading, rank splitting, mixture weighting, missing-source handling and writing
`val/step-*.json` reuse the flow code, so AR and flow val losses are defined on
the same blocks.

For AR, `validation_metrics/loss` is next-token CE and `val/alpha_profile/*` is
absent. `val/embed_geometry/*` is logged, but `alpha_star` has no meaning without
SLERP; `c_mean` (anisotropy) and `emb_norm_*` remain comparable. AR CE is not
directly comparable to the flow val loss; compare via α_eq, the α at which the
flow CE equals the AR CE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import BaseValidator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.swaflow.validate import HeldoutSource, SwaFlowValidator
from torchtitan.tools.logging import logger

__all__ = ["ARValidator", "build_ar_validator"]


class ARValidator(SwaFlowValidator):
    """Held-out next-token CE per source."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.alpha_grid:
            # An α grid left in the config would suggest that AR sweeps α; drop it
            # and warn instead of ignoring it silently.
            logger.warning(
                f"[val] the AR models do not use alpha_grid (the configured "
                f"{self.alpha_grid} is ignored); set it to [] in the config"
            )
            self.alpha_grid = []
        total = sum(s.n_blocks for s in self.sources)
        logger.info(
            f"[val] AR: held-out {len(self.sources)} sources / {total:,} blocks "
            f"× {self.sources[0].seq_len} tokens (next-token CE, no α)"
        )

    @torch.no_grad()
    def _eval_source(
        self, model: nn.Module, source: HeldoutSource, device: torch.device
    ) -> dict[str, Any]:
        """Next-token CE for one source.

        Differs from the flow version only in having no α loop and no noise;
        block selection (`_rank_blocks`) and aggregation (all-reduce of loss sum
        and valid tokens) are the same. `per_alpha` is an empty dict, so the α
        loops in the parent `validate()` are no-ops.
        """
        block_ids = self._rank_blocks(source.n_blocks)
        n_used = len(block_ids) * self.dp_world_size
        stats = torch.zeros(2, dtype=torch.float64, device=device)

        for s in range(0, len(block_ids), self.batch_size):
            ids = block_ids[s : s + self.batch_size]
            blocks = torch.from_numpy(source.tokens[ids].astype(np.int64)).to(device)
            labels = torch.empty_like(blocks)
            # same label convention as training (`indexed_data.py`); the last position is dropped
            labels[:, :-1] = blocks[:, 1:]
            labels[:, -1] = IGNORE_INDEX
            with self.validation_context(), self.maybe_enable_amp:
                logits = model(blocks)
            ce = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(),
                labels.flatten(0, 1),
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )
            n_valid = (labels != IGNORE_INDEX).sum()
            stats[0] += ce.double()
            stats[1] += n_valid.double()
            self.metrics_processor.ntokens_since_last_log += int(n_valid)

        if self.parallel_dims.dp_enabled:
            torch.distributed.all_reduce(
                stats,
                op=torch.distributed.ReduceOp.SUM,
                group=self.parallel_dims.get_mesh("batch").get_group(),
            )
        return {
            "loss": float(stats[0] / stats[1].clamp_min(1)),
            "per_alpha": {},
            "n_blocks": int(n_used),
            "weight": source.weight,
        }


def build_ar_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer,
    parallel_dims: ParallelDims,
    loss_fn,
    validation_context,
    maybe_enable_amp,
    metrics_processor: MetricsProcessor,
    pp_schedule=None,
    pp_has_first_stage=None,
    pp_has_last_stage=None,
) -> BaseValidator:
    """`build_validator_fn` of the TrainSpec (same signature as the flow version)."""
    del tokenizer, loss_fn
    if pp_schedule is not None:
        raise ValueError("AR evaluation does not support PP")
    del pp_has_first_stage, pp_has_last_stage
    return ARValidator(
        job_config=job_config,
        parallel_dims=parallel_dims,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
