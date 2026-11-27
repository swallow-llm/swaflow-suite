# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Training entry point for SwaFlow.

    torchrun --standalone --nproc_per_node=4 -m torchtitan.models.swaflow.train \\
        --job.config_file <config.toml> --job.dump_folder <dir>

Passes a Trainer subclass to upstream `torchtitan.train.main(TrainerClass)`, the
torchtitan extension pattern (as in `experiments/ft/train.py`). The subclass:

1. shares one checkpoint-schedule object between the checkpointer and the
   validator, so every saved checkpoint is evaluated;
2. installs diagnostics (`install_diagnostics`: maximum attention logit and
   pre-clipping gradient norm);
3. reseeds the default generators per DP rank (`rng.py`), since upstream
   `set_determinism` seeds all ranks identically without PP;
4. lets a checkpoint saved with one number of data-parallel ranks be resumed
   with another (`data_reshard.py`).

With `swaflow.checkpoint_schedule.enable = false` (default) the upstream
interval-based schedule is used.
"""

import os

import torchtitan.distributed.utils as dist_utils
from torchtitan.config import JobConfig
from torchtitan.models.swaflow.ckpt_schedule import attach_schedule, CheckpointSchedule
from torchtitan.models.swaflow.data_reshard import attach_dataloader_resharding
from torchtitan.models.swaflow.metrics import (
    configure_attention_stats,
    GradNormLogger,
    set_alpha_ce_observer,
)
from torchtitan.models.swaflow.rng import (
    apply_rank_distinct_rng,
    batch_rank_of,
    prune_rng_states,
    restore_rank_rng_state,
    save_rank_rng_state,
)
from torchtitan.models.swaflow.watchdog import (
    build_watchdog,
    get_watchdog,
    harden_metric_logger,
    RunWatchdog,
)
from torchtitan.tools.logging import logger
from torchtitan.train import main, Trainer

__all__ = ["SwaFlowTrainer", "attach_watchdog", "install_diagnostics"]


def install_diagnostics(job_config: JobConfig) -> None:
    """Install diagnostics. Recording only; training is unchanged.

    1. Maximum attention logit (`<dump>/attn_stats/rank*.jsonl`). Held in a
       process global read by `Attention.forward`; microbatch boundaries are
       marked by the loss (`build_swaflow_denoising_loss`). Must be configured
       before `build_loss_fn`, so this runs before `Trainer.__init__`.

    2. Pre-clipping gradient norm (`<dump>/grad_norm/rank*.jsonl`). Upstream
       `grad_norm` is logged only every `metrics.log_freq` steps and misses
       isolated spikes. `Trainer.train_step` calls `dist_utils.clip_grad_norm_`
       as a module attribute, so wrapping it records every step without
       modifying upstream files. Tensors are buffered (no `.item()`), so no
       syncs are added.
    """
    diag = getattr(getattr(job_config, "swaflow", None), "diagnostics", None)
    if diag is None:
        return
    dump = job_config.job.dump_folder

    if diag.attn_stats_every > 0:
        if job_config.compile.enable and "model" in job_config.compile.components:
            logger.warning(
                "[diagnostics] compile.enable with attn_stats_every > 0 triggers "
                "recompilation of the attention block; set "
                "swaflow.diagnostics.attn_stats_every = 0 when compiling"
            )
        configure_attention_stats(
            os.path.join(dump, "attn_stats"),
            every=diag.attn_stats_every,
            query_stride=diag.attn_stats_query_stride,
        )
        logger.info(
            f"[diagnostics] measuring the max attention logit every {diag.attn_stats_every} "
            f"microbatches (query stride {diag.attn_stats_query_stride})"
        )

    # The watchdog also needs the pre-clipping ‖g‖ every step, so the wrapper is
    # installed even when recording is off. One wrapper feeds both.
    watchdog_cfg = getattr(getattr(job_config, "swaflow", None), "watchdog", None)
    want_wrapper = diag.grad_norm_every_step or (
        watchdog_cfg is not None and watchdog_cfg.enable
    )
    if want_wrapper and not getattr(
        dist_utils.clip_grad_norm_, "_swaflow_wrapped", False
    ):
        grad_logger = (
            GradNormLogger(os.path.join(dump, "grad_norm"))
            if diag.grad_norm_every_step
            else None
        )
        inner = dist_utils.clip_grad_norm_

        def _logging_clip_grad_norm_(*args, **kwargs):
            total_norm = inner(*args, **kwargs)
            if grad_logger is not None:
                grad_logger.record(total_norm)
            watchdog = get_watchdog()
            if watchdog is not None:
                # The skip decision is read by the optimizers.step wrapper via
                # `consume_skip()`.
                watchdog.observe_grad_norm(total_norm)
            return total_norm

        _logging_clip_grad_norm_._swaflow_wrapped = True
        _logging_clip_grad_norm_._swaflow_inner = inner
        dist_utils.clip_grad_norm_ = _logging_clip_grad_norm_
        if grad_logger is not None:
            logger.info("[diagnostics] logging the pre-clipping gradient norm every step")


def attach_watchdog(trainer: Trainer, watchdog: RunWatchdog) -> None:
    """Attach the watchdog to a running Trainer without modifying upstream files.

    Each wrapper only adds work around the original call:
    - `optimizers.step`: skip only the update at a step judged a spike
    - `metrics_processor.log`: update the status file and run health checks
      (on already-computed values, so no extra syncs)
    - `metrics_processor.logger`: wandb / TensorBoard failures do not stop training
    - `checkpointer.save`: save the watchdog state at the checkpoint step
    - `checkpointer.load`: restore the watchdog state on resume
    `clip_grad_norm_` is already wrapped by `install_diagnostics`.

    `checkpointer.save` must be wrapped after `attach_schedule`, which replaces
    the manager's class; a bound method taken earlier would belong to the old class.
    """
    optimizers = trainer.optimizers
    inner_step = optimizers.step

    def _guarded_step(*args, **kwargs):
        if watchdog.consume_skip():
            # Only the optimizer update is skipped. forward/backward and data
            # consumption have already happened, so the random streams and slot
            # coverage are unaffected; the gradients are cleared by the next
            # zero_grad(). The lr_scheduler still advances, keeping LR a function
            # of the step count as upstream resume logic assumes.
            logger.warning(f"[watchdog] step {trainer.step}: skipping the update (spike)")
            return
        return inner_step(*args, **kwargs)

    optimizers.step = _guarded_step

    mp = trainer.metrics_processor
    harden_metric_logger(mp)
    inner_log = mp.log

    def _logging(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        inner_log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)
        watchdog.observe_metrics(step, global_avg_loss, grad_norm, extra_metrics)

    mp.log = _logging

    ckpt = trainer.checkpointer
    inner_save, inner_load = ckpt.save, ckpt.load

    def _save(curr_step: int, last_step: bool = False, *args, **kwargs):
        out = inner_save(curr_step, last_step, *args, **kwargs)
        watchdog.save_state(curr_step)
        return out

    def _load(*args, **kwargs):
        out = inner_load(*args, **kwargs)
        watchdog.restore(trainer.step)
        return out

    ckpt.save = _save
    ckpt.load = _load

    set_alpha_ce_observer(watchdog.observe_alpha_ce)


class SwaFlowTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        # Before super(): attention statistics must exist before build_loss_fn.
        install_diagnostics(job_config)
        super().__init__(job_config)

        # Right after super(): model initialization (DTensor RNG) is done and the
        # checkpoint load (inside `train()`) has not happened, so reseeding here
        # makes t / sphere noise per-rank without changing the initial weights.
        rng_cfg = getattr(getattr(job_config, "swaflow", None), "rng", None)
        # `rng_info["base_seed"]` is the same on all ranks (set by set_determinism),
        # whereas `torch.initial_seed()` differs per rank after reseeding;
        # diagnostics that need a common seed use it.
        self.rng_info = apply_rank_distinct_rng(
            self.parallel_dims, enable=rng_cfg is None or rng_cfg.rank_distinct_seed
        )

        self._attach_checkpoint_schedule(job_config)
        self._attach_rng_checkpointing(job_config)
        attach_dataloader_resharding(
            self.checkpointer,
            step_getter=lambda: self.step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        # Attach the watchdog after the checkpoint schedule: attach_schedule
        # replaces the manager's class, so wrapping save earlier would capture the
        # old class's method.
        self.watchdog = build_watchdog(
            job_config,
            is_primary=(int(os.environ.get("RANK", "0")) == 0),
            step_getter=lambda: self.step,
            world_size=self.parallel_dims.world_size,
        )
        if self.watchdog is not None:
            attach_watchdog(self, self.watchdog)
            logger.info(f"[watchdog] {self.watchdog.describe()}")

    def _attach_checkpoint_schedule(self, job_config: JobConfig) -> None:
        schedule = CheckpointSchedule.from_job_config(job_config)
        if schedule is None:
            return

        attach_schedule(self.checkpointer, schedule, job_config.job.dump_folder)
        logger.info(f"[ckpt-schedule] {schedule.describe(job_config.training.steps)}")

        validator = getattr(self, "validator", None)
        if validator is None:
            logger.warning(
                "[ckpt-schedule] validation.enable = false; "
                "checkpoints will not be evaluated"
            )
            return
        if not hasattr(validator, "schedule"):
            raise TypeError(
                "swaflow.checkpoint_schedule requires the validator to be a "
                "SwaFlowValidator "
                f"(got {type(validator).__name__})"
            )
        # share the checkpointer's schedule object
        validator.schedule = schedule

    def _attach_rng_checkpointing(self, job_config: JobConfig) -> None:
        """Save the random-stream position with each checkpoint (see `rng.py`).

        Without this a resumed run draws different t / sphere noise from step N+1
        on, and `spike_skip` would skip different steps. Independent of the
        watchdog.
        """
        rng_cfg = getattr(getattr(job_config, "swaflow", None), "rng", None)
        if rng_cfg is not None and not rng_cfg.checkpoint_rng_state:
            logger.warning(
                "[rank-rng] checkpoint_rng_state = false: t / sphere noise after "
                "resume will differ from an uninterrupted run (and so will spike skips)"
            )
            return
        dump = job_config.job.dump_folder
        ckpt_dir = os.path.join(dump, job_config.checkpoint.folder)
        # Keyed by batch rank, the unit the streams are split by; TP ranks that
        # see the same batch write identical state.
        rank = batch_rank_of(self.parallel_dims)
        is_global_zero = int(os.environ.get("RANK", "0")) == 0
        ckpt = self.checkpointer
        inner_save, inner_load = ckpt.save, ckpt.load

        def _save(curr_step: int, last_step: bool = False, *args, **kwargs):
            out = inner_save(curr_step, last_step, *args, **kwargs)
            # `save` is called every step and decides internally whether to save,
            # so record only at steps where a checkpoint was written.
            if os.path.isdir(os.path.join(ckpt_dir, f"step-{curr_step}")):
                save_rank_rng_state(dump, curr_step, rank)
                if is_global_zero:
                    prune_rng_states(dump)
            return out

        def _load(*args, **kwargs):
            out = inner_load(*args, **kwargs)
            if self.step > 0:
                restore_rank_rng_state(dump, self.step, rank)
            return out

        ckpt.save = _save
        ckpt.load = _load

    def close(self) -> None:
        watchdog = getattr(self, "watchdog", None)
        if watchdog is not None:
            # write the final state
            watchdog.close()
            set_alpha_ce_observer(None)
        super().close()


if __name__ == "__main__":
    main(SwaFlowTrainer)
