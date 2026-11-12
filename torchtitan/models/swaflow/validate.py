# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Held-out evaluation at each checkpoint.

The denoising loss depends strongly on α, and the training loss fluctuates with
each batch's α composition. To make val loss comparable across checkpoints, the
data (held-out blocks `<build>/heldout/<source>.npz`), α (the fixed grid
`swaflow.eval.alpha_grid`) and noise (seeded by source, block and α index) are
all fixed. The same computation gives the per-α loss profile.

Results are reported per source; the overall value is weighted by the plan's
mixture weights (val loss under the training distribution).
"""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import BaseValidator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.swaflow.ckpt_schedule import CheckpointSchedule
from torchtitan.models.swaflow.embed_geometry import embedding_geometry
from torchtitan.models.swaflow.loss import _denoising_ce
from torchtitan.tools.logging import logger

__all__ = ["HeldoutSource", "SwaFlowValidator", "build_swaflow_validator", "noise_seed"]


_M64 = (1 << 64) - 1


def noise_seed(base_seed: int, source_index: int, block_index: int, alpha_index: int) -> int:
    """Independent, reproducible seed per (source, block, α).

    It depends only on the block index, so the noise is the same for any world
    size or batch size and val loss is comparable across model sizes. Mixing
    uses the splitmix64 finalizer.
    """
    x = base_seed & _M64
    for part in (source_index, alpha_index, block_index):
        x = (x * 0x9E3779B97F4A7C15 + (part & _M64) + 0x165667B19E3779F9) & _M64
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & _M64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & _M64
        x ^= x >> 31
    return x >> 1  # torch.Generator.manual_seed takes a signed 64-bit value


class HeldoutSource:
    """Held-out slice of one source."""

    def __init__(self, name: str, weight: float, tokens: np.ndarray, index: int):
        self.name = name
        self.weight = weight
        self.tokens = tokens  # [n_blocks, seq_len] uint16
        self.index = index

    @property
    def n_blocks(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])


def load_heldout(build_dir: str | Path, max_blocks: int = 0) -> list[HeldoutSource]:
    """Read `<build>/heldout/<source>.npz` and `plan.json`.

    The plan's mixture weights are carried as source weights, so the overall
    value is the val loss under the training distribution.
    """
    build_dir = Path(build_dir)
    plan_path = build_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"{plan_path} not found; swaflow.eval.build_dir must point to a data "
            "build directory (containing plan.json)"
        )
    plan = json.loads(plan_path.read_text())
    sources = []
    for i, sp in enumerate(plan["sources"]):
        npz = build_dir / "heldout" / f"{sp['name']}.npz"
        if not npz.exists():
            raise FileNotFoundError(
                f"{npz} not found (this build has no held-out split)"
            )
        tokens = np.load(npz)["tokens"]
        if max_blocks:
            tokens = tokens[:max_blocks]  # first N (deterministic)
        sources.append(HeldoutSource(sp["name"], float(sp["weight"]), tokens, i))
    return sources


class SwaFlowValidator(BaseValidator):
    """Held-out val loss and α profile at each checkpoint.

    `should_validate` follows the checkpoint schedule (`SwaFlowTrainer` passes
    the same `CheckpointSchedule` to both), so every saved checkpoint is evaluated.
    """

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        validation_context,
        maybe_enable_amp: AbstractContextManager[None],
        metrics_processor: MetricsProcessor,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        schedule: CheckpointSchedule | None = None,
    ) -> None:
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.schedule = schedule

        cfg = job_config.swaflow.eval
        build_dir = cfg.build_dir or job_config.training.dataset_path
        self.sources = load_heldout(build_dir, cfg.max_blocks_per_source)
        self.alpha_grid = list(cfg.alpha_grid)
        self.batch_size = cfg.local_batch_size
        self.seed = cfg.seed
        self.embed_geometry = cfg.embed_geometry
        self.build_dir = build_dir
        self.out_dir = os.path.join(job_config.job.dump_folder, "val")

        seq_len = job_config.training.seq_len
        for s in self.sources:
            if s.seq_len != seq_len:
                raise ValueError(
                    f"held-out '{s.name}': seq_len {s.seq_len} does not match "
                    f"training.seq_len {seq_len}."
                )
        total = sum(s.n_blocks for s in self.sources)
        logger.info(
            f"[val] held-out: {len(self.sources)} sources / {total:,} blocks × "
            f"{seq_len} tokens, alpha grid {self.alpha_grid} "
            f"({total * len(self.alpha_grid):,} forwards per checkpoint)"
        )

    # ------------------------------------------------------------------
    def should_validate(self, step: int) -> bool:
        if self.schedule is None:
            return super().should_validate(step)
        # exactly the steps at which checkpoints are saved
        return self.schedule.should_save(step) or step == self.job_config.training.steps

    # ------------------------------------------------------------------
    def _rank_blocks(self, n_blocks: int) -> np.ndarray:
        """Split blocks evenly across DP ranks.

        The remainder is dropped: all ranks must run the same number of
        forwards, or FSDP collectives deadlock.
        """
        per_rank = n_blocks // self.dp_world_size
        start = self.dp_rank * per_rank
        return np.arange(start, start + per_rank, dtype=np.int64)

    def _noise_for(
        self,
        source: HeldoutSource,
        block_ids: np.ndarray,
        alpha_index: int,
        shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Deterministic noise [B, L, d], seeded per block."""
        gen = torch.Generator(device=device)
        out = torch.empty((len(block_ids), *shape), device=device, dtype=dtype)
        for i, b in enumerate(block_ids):
            gen.manual_seed(noise_seed(self.seed, source.index, int(b), alpha_index))
            out[i] = torch.randn(shape, generator=gen, device=device, dtype=dtype)
        return out

    @torch.no_grad()
    def _eval_source(
        self, model: nn.Module, source: HeldoutSource, device: torch.device
    ) -> dict[str, Any]:
        from torchtitan.components.loss import IGNORE_INDEX

        dim = model.model_args.dim if hasattr(model, "model_args") else None
        if dim is None:  # model_args survives FSDP wrapping; fallback only
            dim = model.tok_embeddings.weight.shape[1]

        block_ids = self._rank_blocks(source.n_blocks)
        n_used = len(block_ids) * self.dp_world_size
        # (loss sum, valid tokens) per α as [α, 2]; one all-reduce at the end
        stats = torch.zeros(len(self.alpha_grid), 2, dtype=torch.float64, device=device)

        for ai, a in enumerate(self.alpha_grid):
            for s in range(0, len(block_ids), self.batch_size):
                ids = block_ids[s : s + self.batch_size]
                blocks = torch.from_numpy(
                    source.tokens[ids].astype(np.int64)
                ).to(device)
                labels = torch.empty_like(blocks)
                labels[:, :-1] = blocks[:, 1:]
                labels[:, -1] = IGNORE_INDEX  # unused by the loss (as in pretokenized)
                alpha = torch.full(
                    (blocks.shape[0],), float(a), device=device, dtype=torch.float32
                )
                noise = self._noise_for(
                    source, ids, ai, (source.seq_len, dim), device, torch.float32
                )
                with self.validation_context(), self.maybe_enable_amp:
                    logits, _ = model(blocks, alpha=alpha, noise=noise)
                loss_sum, _ = _denoising_ce(logits, labels)
                n_valid = (labels[:, :-1] != IGNORE_INDEX).sum()
                stats[ai, 0] += loss_sum.double()
                stats[ai, 1] += n_valid.double()
                self.metrics_processor.ntokens_since_last_log += int(n_valid)

        if self.parallel_dims.dp_enabled:
            # dist_utils.dist_sum handles scalars only (asserts numel == 1), so
            # reduce the [α, 2] tensor directly.
            torch.distributed.all_reduce(
                stats,
                op=torch.distributed.ReduceOp.SUM,
                group=self.parallel_dims.get_mesh("batch").get_group(),
            )
        per_alpha = (stats[:, 0] / stats[:, 1].clamp_min(1)).tolist()
        # token-weighted mean over α (this source's val loss)
        loss = float(stats[:, 0].sum() / stats[:, 1].sum().clamp_min(1))
        return {
            "loss": loss,
            "per_alpha": {f"{a:g}": v for a, v in zip(self.alpha_grid, per_alpha)},
            "n_blocks": int(n_used),
            "weight": source.weight,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _embedding_geometry(self, model: nn.Module) -> dict[str, float]:
        """Embedding geometry (`embed_geometry.py`).

        Called on all ranks: under FSDP the embedding is a sharded DTensor and
        `full_tensor()` is a collective, so calling it on rank 0 only would hang.
        The value is identical on all ranks; the caller records it on rank 0.
        """
        if not self.embed_geometry:
            return {}
        weight = model.tok_embeddings.weight
        full = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
        return embedding_geometry(full, n_probe=self.embed_geometry, seed=self.seed)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        assert len(model_parts) == 1, "SwaFlow evaluation does not support PP"
        model = model_parts[0]
        was_training = model.training
        model.eval()
        # take the device from the model (utils.device_type disagrees in CPU tests)
        device = next(model.parameters()).device

        results = {s.name: self._eval_source(model, s, device) for s in self.sources}
        geometry = self._embedding_geometry(model)

        # Sources with n_blocks == 0 are missing: _rank_blocks drops the
        # remainder, so a source with fewer held-out blocks than dp_world_size is
        # not evaluated and returns loss 0.0. Including it would silently lower
        # the weighted overall value, so it is excluded.
        measured = {n: r for n, r in results.items() if r["n_blocks"] > 0}
        missing = sorted(set(results) - set(measured))
        if not measured:
            # No measured source is a config error (held-out set too small or
            # world size too large); returning 0.0 would look like a good val loss.
            raise RuntimeError(
                f"no held-out source has at least dp_world_size={self.dp_world_size} "
                f"blocks; nothing can be evaluated "
                f"({ {n: r['n_blocks'] for n, r in results.items()} })"
            )
        if missing:
            logger.warning(
                f"[val] excluding sources with fewer held-out blocks than dp_world_size={self.dp_world_size} "
                f"from the overall value: {', '.join(missing)} "
                f"(excluded mixture weight {sum(results[n]['weight'] for n in missing):.4f})"
            )
        for name in missing:
            results[name]["loss"] = float("nan")
            results[name]["per_alpha"] = {k: float("nan") for k in results[name]["per_alpha"]}

        # overall value weighted by the plan's mixture (val loss under the training distribution)
        wsum = sum(r["weight"] for r in measured.values()) or 1.0
        overall = sum(r["loss"] * r["weight"] for r in measured.values()) / wsum

        metrics: dict[str, Any] = {}
        for name, r in measured.items():
            metrics[f"val/loss/{name}"] = r["loss"]
            for a, v in r["per_alpha"].items():
                metrics[f"val/alpha_profile/{name}/a{a}"] = v
        # α profile across sources (used to measure the window)
        for i, a in enumerate(self.alpha_grid):
            num = sum(r["per_alpha"][f"{a:g}"] * r["weight"] for r in measured.values())
            metrics[f"val/alpha_profile/a{a:g}"] = num / wsum
        metrics["val/n_blocks"] = sum(r["n_blocks"] for r in results.values())
        metrics["val/weight_measured"] = wsum
        for k, v in geometry.items():
            metrics[f"val/embed_geometry/{k}"] = v

        self.metrics_processor.log_validation(
            loss=overall, step=step, extra_metrics=metrics
        )

        # also write JSON for analysis without wandb
        if self.dp_rank == 0 and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            os.makedirs(self.out_dir, exist_ok=True)
            payload = {
                "step": step,
                "overall_loss": overall,
                "alpha_grid": self.alpha_grid,
                "build_dir": str(self.build_dir),
                "sources": results,
                "embed_geometry": geometry,
            }
            with open(os.path.join(self.out_dir, f"step-{step}.json"), "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[val] step {step}: overall {overall:.4f}  "
            + "  ".join(f"{n}={r['loss']:.4f}" for n, r in results.items())
        )
        if was_training:
            model.train()


def build_swaflow_validator(
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
    """`build_validator_fn` of the TrainSpec.

    `loss_fn` is unused: the training loss_fn logs α as a side effect, so
    evaluation calls the pure `_denoising_ce` directly. PP is not supported.
    """
    del tokenizer, loss_fn
    if pp_schedule is not None:
        raise ValueError("SwaFlow evaluation does not support PP")
    del pp_has_first_stage, pp_has_last_stage
    return SwaFlowValidator(
        job_config=job_config,
        parallel_dims=parallel_dims,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
