# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Monitoring of long runs: health checks, gradient-spike handling and a status file.

Divergence does not necessarily produce NaN / Inf: a run can keep going with
clipping active on almost every step (so the update size is set by the clip,
not the LR) and a flat α profile. The primary indicators are therefore

1. the shape of CE across α (has it flattened?) -- `AlphaProfileTracker`
2. the clip rate (is clipping persistent?) -- `ClipRateTracker`
3. loss / gradient-norm spikes -- `SpikeDetector`. The per-sample CE ceiling is
   ln V = 10.8027, not the unigram CE 7.53: the latter is the best a converged
   model reaches as α -> 0, not an upper bound, and would flag the whole early
   phase.

NaN / Inf is only a backstop.

## Spike handling (default: record only)

Gradient clipping already bounds outlier updates, so by default spikes are
only recorded. With `spike_skip = true`, the optimizer update is skipped on a
step with a detected gradient-norm spike:

- only `optimizers.step()` is skipped; forward / backward and data loading run
  as usual, so RNG streams and data consumption are unchanged
- the decision uses only ‖g‖, which is identical on all ranks. Per-sample CE
  is rank-local and is used for alerts only
- the decision state (window of recent ‖g‖, skip history) is saved to
  `<dump>/watchdog/step-<N>.json` at checkpoint steps and restored on resume,
  so a resumed run makes the same decisions
- in skip mode the decision is also all-reduced (MAX of an int), in case of
  floating-point differences between ranks

## Cost

- `spike_skip = false` (default): ‖g‖ tensors are buffered and read with
  `.item()` every `grad_flush_every` steps, so no per-step GPU->CPU sync is
  added (as in `GradNormLogger`); detection lags by up to `grad_flush_every`
  steps
- `spike_skip = true`: the decision is needed at once, so it syncs every step
  (negligible next to the step time of large models)
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from torchtitan.tools.logging import logger

__all__ = [
    "AlphaProfileTracker",
    "ClipRateTracker",
    "LN_V",
    "RunWatchdog",
    "SpikeDetector",
    "SpikeEvent",
    "UNIGRAM_CE",
    "WatchdogStop",
    "build_watchdog",
    "get_watchdog",
    "harden_metric_logger",
    "set_watchdog",
]

#: CE of a uniform prediction (SmolLM2, V = 49,152); the ceiling used for spike checks.
LN_V = 10.8027

#: Unigram CE: the best a converged model reaches as α -> 0. Only for expressing
#: loss levels relative to a ceiling; not for spike checks.
UNIGRAM_CE = 7.53


class WatchdogStop(RuntimeError):
    """Raised to stop the run on a condition that needs manual inspection.

    torchrun then exits non-zero. A stop flag is written at the same time, so an
    automatic resubmission chain stops when its next job starts.
    """


# --------------------------------------------------------------------------
# 1. spike detection (‖g‖-based; deterministic, state saved with checkpoints)
# --------------------------------------------------------------------------
@dataclass
class SpikeEvent:
    step: int
    grad_norm: float
    baseline: float
    threshold: float
    kind: str
    """`grad` (jump above the running median) or `nonfinite` (NaN / Inf)."""
    skipped: bool


class SpikeDetector:
    """Detect jumps of ‖g‖ above the running median of recent values.

    The median, unlike the mean, is not dragged along by the spikes themselves
    (a few outliers in a window of 50 do not move it). Spikes are therefore not
    excluded from the window; excluding them would freeze the baseline after a
    lasting rise in ‖g‖ and flag every later step (see `commit`).

    `floor` is an absolute lower bound, by default `training.max_norm`: a ‖g‖
    below the clipping threshold is not a jump (the update is bounded anyway),
    and flagging it only adds noise early in training.

    All decision state is in `state_dict()`. Decisions must be identical after
    resume, so no time or randomness may be used.
    """

    def __init__(
        self,
        factor: float = 10.0,
        window: int = 50,
        min_history: int = 20,
        floor: float = 0.0,
    ) -> None:
        self.factor = float(factor)
        self.window = int(window)
        self.min_history = int(min_history)
        self.floor = float(floor)
        self._hist: deque[float] = deque(maxlen=self.window)
        self.total_spikes = 0
        self.total_skips = 0
        self.total_nonfinite = 0
        self.consecutive_spikes = 0
        self.consecutive_skips = 0
        # The spike rate is taken over the recent window (default 200 steps), so
        # these must cover a whole window; a shorter history would cap the rate.
        self.recent_spike_steps: deque[int] = deque(maxlen=512)
        self.recent_skip_steps: deque[int] = deque(maxlen=512)

    # -- decision --------------------------------------------------------
    def propose(self, grad_norm: float) -> tuple[bool, float, float]:
        """Return (is_spike, baseline, threshold) without updating the history.

        `commit` updates the history. The two are separate because in skip mode
        the decision is first agreed across ranks by an all-reduce; updating the
        history before that would let rank states diverge when decisions differ.
        """
        if not math.isfinite(grad_norm):
            return True, float("nan"), float("nan")
        if len(self._hist) < self.min_history:
            return False, float("nan"), float("nan")
        baseline = statistics.median(self._hist)
        threshold = max(self.factor * baseline, self.floor)
        return grad_norm > threshold, baseline, threshold

    def commit(
        self, step: int, grad_norm: float, is_spike: bool, skipped: bool
    ) -> SpikeEvent | None:
        # Spikes enter the history too. The median ignores a few outliers, so
        # excluding them is unnecessary, and harmful: after a lasting rise in ‖g‖
        # (a regime change, not an outlier) the baseline would freeze at the old
        # level and every later step would be a spike. Once half the window is
        # high, the median follows and the alerts stop.
        if math.isfinite(grad_norm):
            self._hist.append(grad_norm)
        if not is_spike:
            self.consecutive_spikes = 0
            self.consecutive_skips = 0
            return None
        self.total_spikes += 1
        self.consecutive_spikes += 1
        self.recent_spike_steps.append(step)
        if not math.isfinite(grad_norm):
            self.total_nonfinite += 1
        if skipped:
            self.total_skips += 1
            self.consecutive_skips += 1
            self.recent_skip_steps.append(step)
        else:
            self.consecutive_skips = 0
        baseline = (
            statistics.median(self._hist) if len(self._hist) >= self.min_history
            else float("nan")
        )
        threshold = (
            max(self.factor * baseline, self.floor) if math.isfinite(baseline)
            else float("nan")
        )
        return SpikeEvent(
            step=step,
            grad_norm=grad_norm,
            baseline=baseline,
            threshold=threshold,
            kind="grad" if math.isfinite(grad_norm) else "nonfinite",
            skipped=skipped,
        )

    # -- state (consistent across resume) --------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "hist": list(self._hist),
            "total_spikes": self.total_spikes,
            "total_skips": self.total_skips,
            "total_nonfinite": self.total_nonfinite,
            "consecutive_spikes": self.consecutive_spikes,
            "consecutive_skips": self.consecutive_skips,
            "recent_spike_steps": list(self.recent_spike_steps),
            "recent_skip_steps": list(self.recent_skip_steps),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._hist = deque(
            [float(v) for v in state.get("hist", [])], maxlen=self.window
        )
        self.total_spikes = int(state.get("total_spikes", 0))
        self.total_skips = int(state.get("total_skips", 0))
        self.total_nonfinite = int(state.get("total_nonfinite", 0))
        self.consecutive_spikes = int(state.get("consecutive_spikes", 0))
        self.consecutive_skips = int(state.get("consecutive_skips", 0))
        self.recent_spike_steps = deque(
            [int(v) for v in state.get("recent_spike_steps", [])], maxlen=512
        )
        self.recent_skip_steps = deque(
            [int(v) for v in state.get("recent_skip_steps", [])], maxlen=512
        )


# --------------------------------------------------------------------------
# 2. flattening of the α profile
# --------------------------------------------------------------------------
class AlphaProfileTracker:
    """Summarise the shape of the training (α, per-sample CE) pairs in one number.

    `contrast` = mean CE on the low-α side - mean CE on the high-α side. α is
    uniform within the window [0, α_max], so splitting at the observed α
    quartiles cuts at the same place without knowing the window (and means the
    same for runs with different windows).

    There is no absolute threshold: early in training even a healthy run is
    flat, so an absolute threshold would start with false positives. The check
    is how far the contrast has fallen from its own best, i.e. a run that was
    learning and then broke.
    """

    def __init__(
        self,
        capacity: int = 4096,
        min_samples: int = 256,
        ema: float = 0.3,
        update_every: int = 8,
    ) -> None:
        self.capacity = int(capacity)
        self.min_samples = int(min_samples)
        self.ema = float(ema)
        # Computing the contrast sorts the window, so it runs every few
        # microbatches (every 8 is plenty for a window of 4,096).
        self.update_every = max(1, int(update_every))
        self._since_update = 0
        self._alpha: deque[float] = deque(maxlen=self.capacity)
        self._ce: deque[float] = deque(maxlen=self.capacity)
        self.contrast: float | None = None
        self.contrast_max: float = 0.0
        self.ce_max: float = 0.0
        """Maximum per-sample CE in the window.

        Not the maximum since the last log: alpha_ce flushes (every 50
        microbatches) and metric logging (every `log_freq` steps) have different
        periods, so most log intervals would contain no flush and report 0.0.
        """
        self.ce_ceiling_hits: int = 0

    def observe(self, alphas: Sequence[float], ces: Sequence[float]) -> None:
        for a, c in zip(alphas, ces):
            self._alpha.append(float(a))
            self._ce.append(float(c))
            if c > LN_V:
                self.ce_ceiling_hits += 1
            # Raise the running maximum here (`_update` runs only every few
            # calls, so relying on it alone could report a maximum of 0.000).
            # Lowering it as values leave the window is done in `_update`.
            self.ce_max = max(self.ce_max, float(c))
        self._update()

    def _update(self) -> None:
        n = len(self._alpha)
        if n < self.min_samples:
            return
        self._since_update += 1
        if self._since_update < self.update_every:
            return
        self._since_update = 0
        alphas = sorted(self._alpha)
        lo_q = alphas[n // 4]
        hi_q = alphas[(3 * n) // 4]
        lo_ce = [c for a, c in zip(self._alpha, self._ce) if a <= lo_q]
        hi_ce = [c for a, c in zip(self._alpha, self._ce) if a >= hi_q]
        if not lo_ce or not hi_ce:
            return
        self.ce_max = max(self._ce)
        contrast = sum(lo_ce) / len(lo_ce) - sum(hi_ce) / len(hi_ce)
        self.contrast = (
            contrast
            if self.contrast is None
            else (1.0 - self.ema) * self.contrast + self.ema * contrast
        )
        self.contrast_max = max(self.contrast_max, self.contrast)

    def state_dict(self) -> dict[str, Any]:
        return {
            "contrast": self.contrast,
            "contrast_max": self.contrast_max,
            "ce_ceiling_hits": self.ce_ceiling_hits,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        c = state.get("contrast")
        self.contrast = None if c is None else float(c)
        self.contrast_max = float(state.get("contrast_max", 0.0))
        self.ce_ceiling_hits = int(state.get("ce_ceiling_hits", 0))


# --------------------------------------------------------------------------
# 3. clip rate
# --------------------------------------------------------------------------
class ClipRateTracker:
    """Fraction of the last N steps on which clipping was active.

    The rate depends on the window: the initial transient clips heavily, and
    whether warmup covers it shifts the numbers. Checks therefore start only
    after warmup plus a margin. The expected pattern is clipping during the
    transient and rarely afterwards, so the check is only whether clipping
    becomes persistent later.
    """

    def __init__(self, max_norm: float, window: int = 200) -> None:
        self.max_norm = float(max_norm)
        self.window = int(window)
        self._hits: deque[int] = deque(maxlen=self.window)

    def observe(self, grad_norm: float) -> None:
        if not math.isfinite(grad_norm):
            self._hits.append(1)
            return
        self._hits.append(1 if grad_norm > self.max_norm else 0)

    @property
    def n(self) -> int:
        return len(self._hits)

    @property
    def rate(self) -> float | None:
        if not self._hits:
            return None
        return sum(self._hits) / len(self._hits)


# --------------------------------------------------------------------------
# 4. watchdog
# --------------------------------------------------------------------------
class RunWatchdog:
    """Monitoring, spike handling and the status file.

    Unlike the loggers (`AlphaCELogger` / `GradNormLogger`) this makes
    decisions, so its state is saved at checkpoint steps.
    """

    def __init__(
        self,
        cfg,
        *,
        dump_folder: str,
        max_norm: float,
        total_steps: int,
        is_primary: bool,
        step_getter: Callable[[], int],
        world_size: int = 1,
        job_id: str = "",
        checkpoint_dir: str = "",
        warmup_steps: int = 0,
    ) -> None:
        self.cfg = cfg
        self.dump_folder = dump_folder
        self.checkpoint_dir = checkpoint_dir
        self.total_steps = int(total_steps)
        self.is_primary = bool(is_primary)
        self.step_getter = step_getter
        self.world_size = int(world_size)
        self.job_id = job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID", "")
        self.max_norm = float(max_norm)

        floor = cfg.spike_grad_floor if cfg.spike_grad_floor > 0 else max_norm
        self.detector = SpikeDetector(
            factor=cfg.spike_grad_factor,
            window=cfg.spike_grad_window,
            min_history=cfg.spike_min_history,
            floor=floor,
        )
        self.alpha = AlphaProfileTracker()
        self.clip = ClipRateTracker(max_norm, window=cfg.clip_window)

        self.state_dir = os.path.join(dump_folder, "watchdog")
        self.status_path = cfg.status_path or os.path.join(dump_folder, "status.json")
        self.stop_flag_path = cfg.stop_flag_path or os.path.join(
            dump_folder, "autoresume", "STOP"
        )
        self.generation = int(os.environ.get("SWAFLOW_RUN_GENERATION", "0") or 0)

        # Step from which the combined stop condition is active (default derived
        # from warmup). Limiting it to the steady state removes the dependence of
        # the thresholds on warmup (see `_check_pathology_stop`).
        self.stop_enabled_after = cfg.stop_enable_after_steps or max(
            cfg.grace_steps, int(warmup_steps * cfg.stop_warmup_multiple)
        )

        self._pending: list[tuple[int, torch.Tensor]] = []
        self._skip_now = False
        self._alerts: list[dict[str, Any]] = []
        self._alert_last: dict[str, int] = {}
        self._log_failures = 0
        self._t0 = time.time()
        self._last_log_time = self._t0
        self._last_log_step = 0
        self._sec_per_step: float | None = None
        self._recent: dict[str, Any] = {}
        self._restored_from: int | None = None
        self._stopped = False
        self._last_ceiling_hits = 0
        self._stop_confirmations = 0

        if self.is_primary:
            os.makedirs(self.state_dir, exist_ok=True)

    # -- every step (all ranks) -----------------------------------------
    def observe_grad_norm(self, grad_norm: torch.Tensor) -> bool:
        """Take ‖g‖ before clipping; return whether to skip this step's update.

        With `spike_skip = false` the tensor is buffered to avoid extra syncs and
        evaluated every `grad_flush_every` steps (detection is delayed
        accordingly).
        """
        step = self.step_getter()
        if not self.cfg.spike_skip:
            self._pending.append((step, grad_norm.detach().float()))
            if len(self._pending) >= self.cfg.grad_flush_every:
                self._drain()
            return False

        value = float(grad_norm.detach().float().item())
        is_spike, baseline, threshold = self.detector.propose(value)
        is_spike = self._agree(is_spike)
        event = self.detector.commit(step, value, is_spike, skipped=is_spike)
        self.clip.observe(value)
        self._skip_now = is_spike
        if event is not None:
            self._handle_event(event, baseline, threshold)
        return is_spike

    def consume_skip(self) -> bool:
        """Whether to skip this step's update (queried by the `optimizers.step` wrapper).

        The flag is cleared on read. Clipping and the optimizer step each run
        once per step, so a decision is neither lost nor carried over.
        """
        skip = self._skip_now
        self._skip_now = False
        return skip

    def _agree(self, is_spike: bool) -> bool:
        """Agree on the decision across ranks (a spike on any rank is a spike on all).

        ‖g‖ comes from `clip_grad_norm_` after its all-reduce and should match
        across ranks, but a split decision would silently desynchronise the
        optimizer state (inconsistent FSDP shards). One int all-reduce per step
        is a cheap safeguard.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return is_spike
        flag = torch.tensor(
            [1 if is_spike else 0],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    def _drain(self) -> None:
        # Take the buffer before processing it: if `WatchdogStop` is raised
        # midway, `clear()` is never reached and the drain in `close()` would
        # process the same values again, recording them twice.
        pending, self._pending = self._pending, []
        for step, tensor in pending:
            value = float(tensor.item())
            is_spike, baseline, threshold = self.detector.propose(value)
            event = self.detector.commit(step, value, is_spike, skipped=False)
            self.clip.observe(value)
            if event is not None:
                self._handle_event(event, baseline, threshold)
        self._pending.clear()

    def _handle_event(
        self, event: SpikeEvent, baseline: float, threshold: float
    ) -> None:
        text = (
            f"step {event.step}: ‖g‖ = {event.grad_norm:.4g} "
            f"(median {baseline:.4g} / threshold {threshold:.4g}, {event.kind})"
            f"{'; update skipped' if event.skipped else ''}"
        )
        if event.kind == "nonfinite":
            self._alert("nonfinite", text, force=True)
        else:
            self._alert("grad_spike", text)
        if self.is_primary:
            self._append_jsonl(
                "spikes.jsonl",
                {
                    "step": event.step,
                    "grad_norm": event.grad_norm,
                    "baseline": baseline,
                    "threshold": threshold,
                    "kind": event.kind,
                    "skipped": event.skipped,
                    "generation": self.generation,
                },
            )
        self._check_stop_conditions()

    def _check_stop_conditions(self) -> None:
        """Conditions that hand the run over to a human: write the stop flag and stop.

        There is no automatic rollback or hyperparameter change (its failure
        behaviour would be unpredictable). Only conditions that a transient
        cannot trigger are used: stopping a long run on a transient costs more
        than running a little longer while broken.
        """
        cfg = self.cfg
        if (
            cfg.stop_after_nonfinite > 0
            and self.detector.total_nonfinite >= cfg.stop_after_nonfinite
        ):
            self.request_stop(
                f"‖g‖ was non-finite (NaN / Inf) "
                f"{self.detector.total_nonfinite} times "
                f"(limit {cfg.stop_after_nonfinite}); updating would corrupt the weights"
            )
        rate = self._spike_rate()
        if (
            cfg.stop_spike_rate > 0
            and self.step_getter() >= cfg.grace_steps
            and self.clip.n >= cfg.clip_window // 2
            and rate is not None
            and rate > cfg.stop_spike_rate
        ):
            self.request_stop(
                f"over the last {self.clip.n} steps, {100 * rate:.1f}% were spikes "
                f"(threshold {100 * cfg.stop_spike_rate:.0f}%): "
                "persistently broken, not a transient jump"
            )
        limit = cfg.stop_after_consecutive_spikes
        if limit > 0 and self.detector.consecutive_spikes >= limit:
            self.request_stop(
                f"spikes on {self.detector.consecutive_spikes} consecutive steps (limit {limit})"
            )

    def _spike_rate(self) -> float | None:
        """Fraction of the last `clip_window` steps that were spikes."""
        window = self.cfg.clip_window
        if self.clip.n <= 0:
            return None
        recent = [s for s in self.detector.recent_spike_steps if s > self.step_getter() - window]
        return len(recent) / min(self.clip.n, window)

    # -- every microbatch (rank-local, no sync) --------------------------
    def observe_alpha_ce(self, alphas: Sequence[float], ces: Sequence[float]) -> None:
        """Called from the `AlphaCELogger` flush (CPU values; no extra sync).

        These values are rank-local, so they must not decide whether to update;
        they feed alerts and the status file only.
        """
        if not self.is_primary:
            return
        self.alpha.observe(alphas, ces)

    # -- metrics logging interval (rank 0) ------------------------------
    def observe_metrics(
        self, step: int, loss: float, grad_norm: float, extra: dict[str, Any] | None
    ) -> None:
        now = time.time()
        if step > self._last_log_step:
            dt = now - self._last_log_time
            n = step - self._last_log_step
            inst = dt / n
            self._sec_per_step = (
                inst if self._sec_per_step is None
                else 0.7 * self._sec_per_step + 0.3 * inst
            )
        self._last_log_time = now
        self._last_log_step = step

        extra = extra or {}
        self._recent = {
            "loss": loss,
            "grad_norm": grad_norm,
            "lr": extra.get("lr"),
            "tokens_seen": extra.get("n_tokens_seen"),
        }
        self._health_checks(step)
        self.write_status(step)

    def _health_checks(self, step: int) -> None:
        cfg = self.cfg
        if step < cfg.grace_steps:
            return

        rate = self.clip.rate
        if (
            rate is not None
            and self.clip.n >= max(10, cfg.clip_window // 4)
            and rate > cfg.clip_rate_alert
        ):
            self._alert(
                "clip_rate",
                f"step {step}: clip rate over the last {self.clip.n} steps is "
                f"{100 * rate:.1f}% (threshold {100 * cfg.clip_rate_alert:.0f}%). "
                "Persistent clipping means the update size is set by the clip "
                "rather than the LR",
            )

        c = self.alpha.contrast
        if c is not None:
            if (
                self.alpha.contrast_max >= cfg.alpha_contrast_arm
                and c < cfg.alpha_flatten_ratio * self.alpha.contrast_max
            ):
                self._alert(
                    "alpha_flat",
                    f"step {step}: α-profile contrast fell to {c:.3f} "
                    f"(best so far {self.alpha.contrast_max:.3f}, "
                    f"threshold {cfg.alpha_flatten_ratio:.2f}x the best). "
                    "The model is losing the use of α within the window",
                )
            if cfg.alpha_contrast_min > 0 and c < cfg.alpha_contrast_min:
                self._alert(
                    "alpha_flat_abs",
                    f"step {step}: α-profile contrast {c:.3f} is "
                    f"below the absolute minimum ({cfg.alpha_contrast_min:.3f})",
                )

        self._check_pathology_stop(step, rate, c)

        # Alert on new ceiling hits since the last check (a cumulative count
        # would keep alerting after the first hit).
        # ce_ceiling never stops the run: healthy runs also exceed the ceiling
        # occasionally at low α, so it is a weak signal.
        if self.alpha.ce_ceiling_hits > self._last_ceiling_hits:
            new_hits = self.alpha.ce_ceiling_hits - self._last_ceiling_hits
            self._last_ceiling_hits = self.alpha.ce_ceiling_hits
            self._alert(
                "ce_ceiling",
                f"step {step}: per-sample CE exceeded the ceiling ln V = {LN_V} "
                f"({new_hits} new samples, {self.alpha.ce_ceiling_hits} total, "
                f"max in window {self.alpha.ce_max:.3f}): confidently wrong predictions",
            )

    def _check_pathology_stop(
        self, step: int, rate: float | None, contrast: float | None
    ) -> None:
        """Stop on divergence that does not produce NaN.

        `stop_after_nonfinite` / `stop_spike_rate` miss a run that diverges
        without non-finite values or frequent spikes. A high clip rate together
        with a fall of the α contrast from its best (the primary indicators of
        this module) does catch it.

        Both must hold, `stop_confirmations` times in a row: either alone can
        hold temporarily (clipping rises on hard data, the contrast fluctuates).

        Checked only in the steady state (`stop_enable_after_steps`). The clip
        rate depends on warmup only through the initial transient, so the
        thresholds need no retuning when warmup changes, and the transient
        cannot stop a run by mistake.

        The α contrast exists only on rank 0 (per-sample CE is rank-local), so
        the decision is all-reduced to every rank; if rank 0 alone raised, the
        other ranks would hang at the next collective.
        """
        cfg = self.cfg
        if cfg.stop_clip_rate <= 0 or step < self.stop_enabled_after:
            return

        local = (
            rate is not None
            and self.clip.n >= max(10, cfg.clip_window // 4)
            and rate > cfg.stop_clip_rate
            and contrast is not None
            and self.alpha.contrast_max >= cfg.alpha_contrast_arm
            and contrast < cfg.alpha_flatten_ratio * self.alpha.contrast_max
        )
        if self._agree(local):
            self._stop_confirmations += 1
        else:
            self._stop_confirmations = 0
        if self._stop_confirmations < cfg.stop_confirmations:
            return
        self.request_stop(
            f"step {step}: clip rate {100 * (rate or 0):.1f}% "
            f"(threshold {100 * cfg.stop_clip_rate:.0f}%) and "
            f"α contrast drop ({(contrast if contrast is not None else float('nan')):.3f} / "
            f"best {self.alpha.contrast_max:.3f}) held together "
            f"{self._stop_confirmations} times in a row. "
            "This is divergence without NaN; "
            "continuing would only waste compute"
        )

    # -- alerts ------------------------------------------------------------
    def _alert(self, kind: str, message: str, force: bool = False) -> None:
        step = self.step_getter()
        last = self._alert_last.get(kind)
        if not force and last is not None and step - last < self.cfg.alert_cooldown:
            return
        self._alert_last[kind] = step
        record = {"step": step, "kind": kind, "message": message, "time": _now()}
        self._alerts.append(record)
        del self._alerts[:-32]
        logger.error(f"[watchdog] ALERT({kind}) {message}")
        if self.is_primary:
            self._append_jsonl("alerts.jsonl", record | {"generation": self.generation})
            _wandb_alert(f"SwaFlow: {kind}", message)

    def request_stop(self, reason: str) -> None:
        """Write the stop flag and stop the run (this also stops automatic resubmission)."""
        self._stopped = True
        logger.error(f"[watchdog] STOP {reason}")
        self._alert("stop", reason, force=True)
        if self.is_primary:
            try:
                os.makedirs(os.path.dirname(self.stop_flag_path), exist_ok=True)
                with open(self.stop_flag_path, "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "reason": reason,
                                "step": self.step_getter(),
                                "time": _now(),
                                "job_id": self.job_id,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                logger.error(f"[watchdog] wrote stop flag: {self.stop_flag_path}")
            except OSError as e:  # stop even if the flag cannot be written
                logger.error(f"[watchdog] could not write stop flag: {e}")
            self.write_status(self.step_getter(), health="stopped")
        raise WatchdogStop(reason)

    # -- status file -------------------------------------------------------
    def write_status(self, step: int, health: str | None = None) -> None:
        if not self.is_primary:
            return
        tokens = self._recent.get("tokens_seen")
        remaining = max(0, self.total_steps - step)
        eta_h = (
            remaining * self._sec_per_step / 3600.0
            if self._sec_per_step is not None
            else None
        )
        tokens_target = (
            tokens / step * self.total_steps if tokens and step > 0 else None
        )
        if health is None:
            health = "alert" if self._alerts else "ok"
        payload = {
            "updated": _now(),
            "job_id": self.job_id,
            "generation": self.generation,
            "health": health,
            "progress": {
                "step": step,
                "steps_target": self.total_steps,
                "percent": 100.0 * step / self.total_steps if self.total_steps else None,
                "tokens_seen": tokens,
                "tokens_target": tokens_target,
                "sec_per_step": self._sec_per_step,
                "elapsed_hours": (time.time() - self._t0) / 3600.0,
                "eta_hours": eta_h,
                "world_size": self.world_size,
            },
            "recent": {
                **self._recent,
                "clip_rate": self.clip.rate,
                "clip_window": self.clip.n,
                "max_norm": self.max_norm,
                "alpha_contrast": self.alpha.contrast,
                "alpha_contrast_max": self.alpha.contrast_max,
                "ce_max": self.alpha.ce_max,
                "ce_ceiling": LN_V,
            },
            "spikes": {
                "total": self.detector.total_spikes,
                "recent_steps": list(self.detector.recent_spike_steps),
                "consecutive": self.detector.consecutive_spikes,
                "skipped_total": self.detector.total_skips,
                "skipped_steps": list(self.detector.recent_skip_steps),
                "skip_enabled": bool(self.cfg.spike_skip),
                "nonfinite": self.detector.total_nonfinite,
                "rate_recent": self._spike_rate(),
                "stop_confirmations": self._stop_confirmations,
                "stop_enabled_after": self.stop_enabled_after,
                "ce_ceiling_hits": self.alpha.ce_ceiling_hits,
            },
            "resume": {
                "generation": self.generation,
                "restored_from_step": self._restored_from,
                "stop_flag": self.stop_flag_path,
                "stop_flag_present": os.path.exists(self.stop_flag_path),
            },
            "logging": {"metric_log_failures": self._log_failures},
            "alerts": self._alerts[-8:],
        }
        _atomic_write_json(self.status_path, payload)

    def _append_jsonl(self, name: str, record: dict[str, Any]) -> None:
        try:
            with open(os.path.join(self.dump_folder, name), "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except OSError as e:
            logger.warning(f"[watchdog] {name} not writable: {e}")

    # -- save / restore at checkpoint steps -------------------------------
    def save_state(self, step: int) -> None:
        """Called right after a checkpoint save (by all ranks; rank 0 writes).

        The state goes to a separate file rather than into the DCP
        `train_state`: adding keys to the state_dict would make older
        checkpoints unloadable.

        `CheckpointManager.save` is called every step (it decides internally
        whether to save), so state is written only on steps that produced a
        checkpoint. Writing every step would (i) prune the state of the resume
        point too early and (ii) turn the drain into a per-step GPU->CPU sync.
        """
        if self.checkpoint_dir and not os.path.isdir(
            os.path.join(self.checkpoint_dir, f"step-{step}")
        ):
            return
        # Drain on all ranks (keeps the decision state identical across ranks).
        # Only rank 0 writes; on restore every rank reads that file.
        self._drain()
        if not self.is_primary:
            return
        state = {
            "step": step,
            "generation": self.generation,
            "detector": self.detector.state_dict(),
            "alpha": self.alpha.state_dict(),
            "cfg": {
                "spike_grad_factor": self.cfg.spike_grad_factor,
                "spike_grad_window": self.cfg.spike_grad_window,
                "spike_min_history": self.cfg.spike_min_history,
                "spike_skip": bool(self.cfg.spike_skip),
                "floor": self.detector.floor,
            },
        }
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            _atomic_write_json(os.path.join(self.state_dir, f"step-{step}.json"), state)
            _atomic_write_json(os.path.join(self.state_dir, "latest.json"), state)
            self._prune_states()
        except OSError as e:
            logger.warning(f"[watchdog] could not save state: {e}")

    def _prune_states(self, keep: int = 20) -> None:
        steps = []
        for name in os.listdir(self.state_dir):
            if name.startswith("step-") and name.endswith(".json"):
                try:
                    steps.append(int(name[5:-5]))
                except ValueError:
                    continue
        for s in sorted(steps)[:-keep]:
            try:
                os.remove(os.path.join(self.state_dir, f"step-{s}.json"))
            except OSError:
                pass

    def restore(self, step: int) -> None:
        """Load the state of the restored step, right after `checkpointer.load()`.

        Every rank reads the same file, so the decision state agrees across
        ranks in skip mode. A missing file gives a warning: the history starts
        empty, so for `min_history` steps decisions may differ from an
        uninterrupted run.
        """
        if step <= 0:
            return
        path = os.path.join(self.state_dir, f"step-{step}.json")
        if not os.path.exists(path):
            logger.warning(
                f"[watchdog] step {step} has no state file ({path}); "
                "the spike history starts empty (decisions may differ for the first "
                f"{self.detector.min_history} steps after resume)"
            )
            return
        try:
            with open(path) as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[watchdog] cannot read state file ({path}): {e}")
            return
        self.detector.load_state_dict(state.get("detector", {}))
        self.alpha.load_state_dict(state.get("alpha", {}))
        self._restored_from = step
        logger.info(
            f"[watchdog] step {step}: state restored (spikes so far "
            f"{self.detector.total_spikes} / skipped {self.detector.total_skips})"
        )

    def note_log_failure(self) -> None:
        self._log_failures += 1

    def close(self) -> None:
        try:
            self._drain()
            self.write_status(
                self.step_getter(), health="stopped" if self._stopped else "finished"
            )
        except Exception as e:  # noqa: BLE001 (never fail during shutdown)
            logger.warning(f"[watchdog] exception in close: {e}")

    def describe(self) -> str:
        # Print a disabled condition as disabled: a threshold of 0 would read as
        # "stop at 0% clipping" (e.g. for AR runs, which have no α contrast and
        # disable this condition).
        if self.cfg.stop_clip_rate > 0:
            pathology = (
                f" | (clip {self.cfg.stop_clip_rate:.0%} and α fall) ×"
                f" {self.cfg.stop_confirmations} in a row (from step {self.stop_enabled_after})"
            )
        else:
            pathology = " | (clip ∧ α fall) = disabled (stop_clip_rate = 0)"
        return (
            f"spike: ‖g‖ > {self.cfg.spike_grad_factor}× median(last "
            f"{self.cfg.spike_grad_window}) and > {self.detector.floor:g}"
            f" / skip={'on' if self.cfg.spike_skip else 'off'}"
            f" / stop: non-finite x{self.cfg.stop_after_nonfinite} |"
            f" spike rate {self.cfg.stop_spike_rate:.0%}"
            f"{pathology}"
            f" / status={self.status_path}"
        )


# --------------------------------------------------------------------------
# process-global instance (read by the clip wrapper, like `_ATTENTION_STATS`)
# --------------------------------------------------------------------------
_WATCHDOG: RunWatchdog | None = None


def set_watchdog(watchdog: RunWatchdog | None) -> None:
    global _WATCHDOG
    _WATCHDOG = watchdog


def get_watchdog() -> RunWatchdog | None:
    return _WATCHDOG


def build_watchdog(
    job_config,
    *,
    is_primary: bool,
    step_getter: Callable[[], int],
    world_size: int = 1,
) -> RunWatchdog | None:
    cfg = getattr(getattr(job_config, "swaflow", None), "watchdog", None)
    if cfg is None or not cfg.enable:
        return None
    wd = RunWatchdog(
        cfg,
        dump_folder=job_config.job.dump_folder,
        checkpoint_dir=os.path.join(
            job_config.job.dump_folder, job_config.checkpoint.folder
        ),
        max_norm=job_config.training.max_norm,
        total_steps=job_config.training.steps,
        is_primary=is_primary,
        step_getter=step_getter,
        world_size=world_size,
        warmup_steps=getattr(job_config.lr_scheduler, "warmup_steps", 0),
    )
    set_watchdog(wd)
    return wd


# --------------------------------------------------------------------------
# metric logging failures (wandb / tensorboard) must not stop training
# --------------------------------------------------------------------------
class ResilientLogger:
    """Wrap the metric logger and swallow logging failures.

    A transient network failure should not kill a long run. Upstream torchtitan
    catches only `wandb.init` failures, not those of `wandb.log` during
    training. Failures are counted in `logging.metric_log_failures` in
    status.json.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._failures = 0
        self._warned = 0

    def log(self, metrics: dict[str, Any], step: int) -> None:
        try:
            self._inner.log(metrics, step)
        except Exception as e:  # noqa: BLE001 (logging failures do not stop training)
            self._failures += 1
            wd = get_watchdog()
            if wd is not None:
                wd.note_log_failure()
            # Do not flood the log: warn only on failures 1, 2, 4, 8, ...
            if self._failures & (self._failures - 1) == 0:
                self._warned += 1
                logger.warning(
                    f"[metrics] logging failed ({self._failures} failures so far, "
                    f"training continues): {type(e).__name__}: {e}"
                )

    def close(self) -> None:
        try:
            self._inner.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[metrics] exception in logger.close (ignored): {e}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def harden_metric_logger(metrics_processor) -> None:
    inner = getattr(metrics_processor, "logger", None)
    if inner is None or isinstance(inner, ResilientLogger):
        return
    metrics_processor.logger = ResilientLogger(inner)


def _wandb_alert(title: str, text: str) -> None:
    """Send a wandb alert if a wandb run is active (failures are ignored).

    Checks `sys.modules` to avoid importing wandb in runs that do not use it.
    In offline / disabled mode this is a no-op or raises; neither affects
    training.
    """
    wandb = sys.modules.get("wandb")
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    try:
        wandb.alert(title=title, text=text)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[watchdog] wandb.alert failed (ignored): {e}")


# --------------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON atomically (these files are read while the run is going).

    NaN / Inf become `null`: by default `json.dump` writes `NaN`, which is not
    valid JSON and breaks other tools.
    """
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(_sanitize(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque, set)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
