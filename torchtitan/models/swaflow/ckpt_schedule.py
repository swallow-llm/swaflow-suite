# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint schedule.

Log-spaced saves early (1, 2, 4, 8, ...), then a fixed interval; optimizer state
only at selected steps. Representations change quickly early in training, and
AdamW state is twice the size of the weights (~48 GB per checkpoint at 3B), so it
is kept only where resuming is needed.

Model-only saves go to `<dump>/<snapshot_folder>/step-N`, separate from full
checkpoints: upstream `CheckpointManager._find_load_step` picks the largest step
in the checkpoint folder, so mixing in model-only saves would make a resume start
silently without optimizer state. Upstream has no hook for the save decision, so
`ScheduledCheckpointManager` overrides only `save()` and `_create_checkpoint_id()`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from torchtitan.components.checkpoint import AsyncMode, CheckpointManager
from torchtitan.tools.logging import logger

__all__ = ["CheckpointSchedule", "ScheduledCheckpointManager", "attach_schedule"]


@dataclass(frozen=True)
class CheckpointSchedule:
    """Which steps to save, and which of them include optimizer state."""

    log_spaced_until: int = 0
    interval: int = 0
    full_state_every: int = 0
    full_state_steps: tuple[int, ...] = ()
    snapshot_folder: str = "snapshots"
    tokens_per_step: int = 0
    """Tokens per step used to convert token intervals (0 = not converted); used
    only by `describe()`."""

    @classmethod
    def from_job_config(cls, job_config) -> "CheckpointSchedule | None":
        cfg = getattr(getattr(job_config, "swaflow", None), "checkpoint_schedule", None)
        if cfg is None or not cfg.enable:
            return None

        interval = cfg.interval
        full_state_every = cfg.full_state_every
        tokens_per_step = 0
        interval_tokens = getattr(cfg, "interval_tokens", 0)
        full_tokens = getattr(cfg, "full_state_every_tokens", 0)
        if interval_tokens or full_tokens:
            tokens_per_step = cls._tokens_per_step(job_config)
            if interval_tokens:
                interval = cls._steps_for_tokens(interval_tokens, tokens_per_step)
            if full_tokens:
                full_state_every = cls._steps_for_tokens(full_tokens, tokens_per_step)

        return cls(
            log_spaced_until=cfg.log_spaced_until,
            interval=interval or job_config.checkpoint.interval,
            full_state_every=full_state_every,
            full_state_steps=tuple(sorted(set(cfg.full_state_steps))),
            snapshot_folder=cfg.snapshot_folder,
            tokens_per_step=tokens_per_step,
        )

    # ------------------------------------------------------------------
    # Token-based intervals
    #
    # A step interval changes meaning when the global batch changes, so intervals
    # can be given in tokens and are converted to steps at startup.
    # ------------------------------------------------------------------
    @staticmethod
    def _tokens_per_step(job_config) -> int:
        global_batch_size = getattr(job_config.training, "global_batch_size", -1)
        if not global_batch_size or global_batch_size < 0:
            raise ValueError(
                "swaflow.checkpoint_schedule *_tokens requires "
                "training.global_batch_size to be set explicitly; otherwise "
                "tokens/step depends on the DP degree and the token interval "
                "changes with the number of nodes"
            )
        return int(global_batch_size) * int(job_config.training.seq_len)

    @staticmethod
    def _steps_for_tokens(tokens: int, tokens_per_step: int) -> int:
        return max(1, round(int(tokens) / tokens_per_step))

    # ------------------------------------------------------------------
    def should_save(self, step: int, last_step: bool = False) -> bool:
        if last_step:
            return True
        if step <= 0:
            return False
        if step in self.full_state_steps:
            # Steps in full_state_steps are always saved, regardless of the
            # interval (otherwise an off-schedule entry would be ignored).
            return True
        if step <= self.log_spaced_until and (step & (step - 1)) == 0:
            return True  # power of two (log-spaced)
        return self.interval > 0 and step % self.interval == 0

    def saves_full_state(self, step: int, last_step: bool = False) -> bool:
        """Whether to include optimizer state (i.e. resumable from this step)."""
        if last_step:
            return True  # the final step is always resumable
        if step in self.full_state_steps:
            return True
        return self.full_state_every > 0 and step % self.full_state_every == 0

    def scheduled_steps(self, max_step: int) -> list[int]:
        """Steps in [1, max_step] that are saved (for tests and dashboards)."""
        return [s for s in range(1, max_step + 1) if self.should_save(s)]

    def describe(self, max_step: int) -> str:
        steps = self.scheduled_steps(max_step)
        full = [s for s in steps if self.saves_full_state(s)]
        tok = ""
        if self.tokens_per_step:
            tok = (
                f" (= every {self.interval * self.tokens_per_step / 1e9:.3g}B tokens,"
                f" full every {self.full_state_every * self.tokens_per_step / 1e9:.3g}B;"
                f" {self.tokens_per_step / 1e6:.3f}M tokens/step)"
            )
        return (
            f"checkpoint schedule: log-spaced until {self.log_spaced_until}, "
            f"then every {self.interval} -> {len(steps)} saves "
            f"({len(full)} with optimizer state) in {max_step} steps{tok}; "
            f"first: {steps[:10]}"
        )


class ScheduledCheckpointManager(CheckpointManager):
    """`CheckpointManager` that saves according to a `CheckpointSchedule`.

    Overrides only:
    - `save()`: decides by the schedule instead of the interval and switches
      between full and model-only saves per step
    - `_create_checkpoint_id()`: model-only saves go to the snapshot folder

    Because saves go through the upstream always-save path (`last_step=True`):
    - asynchronous saving (`checkpoint.async_mode`) is not used; saves are synchronous
    - old checkpoints are never purged; `checkpoint.keep_latest_k` has no effect
      (upstream `_purge_stale_checkpoints` comes after `_save_last_step` and is
      not reached), and a warning is logged if it is set
    """

    _schedule: CheckpointSchedule
    _snapshot_root: str
    _saving_model_only: bool = False

    def save(self, curr_step: int, last_step: bool = False) -> None:
        if not self.enable or self.load_only:
            return
        if not self._schedule.should_save(curr_step, last_step):
            return

        full = self._schedule.saves_full_state(curr_step, last_step)
        # Upstream save() can only express "full at the interval" or "follow
        # last_save_model_only at last_step". To choose per step, temporarily
        # override last_save_model_only and take the always-save path.
        prev_model_only = self.last_save_model_only
        self.last_save_model_only = not full
        self._saving_model_only = not full
        kind = "full (with optimizer state)" if full else "model only (snapshot)"
        logger.info(f"[ckpt-schedule] step {curr_step}: saving {kind}")
        begin = time.monotonic()
        try:
            super().save(curr_step, last_step=True)
        finally:
            self.last_save_model_only = prev_model_only
            self._saving_model_only = False
        # Upstream's "Finished saving..." log comes after `_save_last_step` and is
        # not reached on this path, so log the save time here.
        logger.info(
            f"[ckpt-schedule] step {curr_step}: saved {kind} "
            f"in {time.monotonic() - begin:.2f} s"
        )

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        if self._saving_model_only and not folder:
            return os.path.join(self._snapshot_root, f"step-{step}")
        return super()._create_checkpoint_id(step, folder)


def attach_schedule(
    manager: CheckpointManager,
    schedule: CheckpointSchedule,
    dump_folder: str,
) -> CheckpointManager:
    """Apply a schedule to an existing `CheckpointManager`.

    `Trainer.__init__` constructs the `CheckpointManager` directly and the
    TrainSpec has no hook to replace it. Rebuilding it would re-register the
    state dicts and (with async) recreate process groups, so only the class of
    the existing instance is swapped. `ScheduledCheckpointManager` defines no
    `__init__` and overrides two methods, so the state stays consistent.
    """
    if not isinstance(manager, CheckpointManager):
        raise TypeError(f"expected a CheckpointManager (got {type(manager).__name__})")
    # With enable=False upstream __init__ returns early and later attributes
    # do not exist, hence getattr.
    if getattr(manager, "async_mode", AsyncMode.DISABLED) != AsyncMode.DISABLED:
        logger.warning(
            f"[ckpt-schedule] async_mode={manager.async_mode} is ignored; saves are "
            "synchronous (a known limitation of ScheduledCheckpointManager)"
        )
    if getattr(manager, "keep_latest_k", 0) > 0:
        logger.warning(
            f"[ckpt-schedule] keep_latest_k={manager.keep_latest_k} is not applied: "
            "this schedule never deletes old checkpoints; remove the setting "
            "from the config to avoid confusion"
        )
    manager.__class__ = ScheduledCheckpointManager
    manager._schedule = schedule
    manager._snapshot_root = os.path.join(dump_folder, schedule.snapshot_folder)
    manager._saving_model_only = False
    return manager
