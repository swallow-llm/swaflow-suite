# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""JobConfig extension for SwaFlow.

Adds a `[swaflow]` section through torchtitan's `[job] custom_config_module`
mechanism (`ConfigManager._merge_configs`), leaving the upstream `JobConfig`
unchanged. In the toml:

    [job]
    custom_config_module = "torchtitan.models.swaflow.job_config"

    [swaflow.checkpoint_schedule]
    enable = true
    log_spaced_until = 64
    interval = 50
    full_state_every = 100

    [swaflow.eval]
    enable = true
    build_dir = "<data_root>/<build>"

These values are included in the wandb run config via `job_config.to_dict()`.
"""

from dataclasses import dataclass, field

__all__ = [
    "CheckpointEval",
    "CheckpointSchedule",
    "Diagnostics",
    "JobConfig",
    "Midtrain",
    "ModelOverrides",
    "RNG",
    "SwaFlowConfig",
    "Watchdog",
]


@dataclass
class CheckpointSchedule:
    """Checkpoint schedule: log-spaced (1, 2, 4, 8, ...) early, then a fixed interval.

    Representations change quickly early in training, where a fixed interval
    would be too coarse.
    """

    enable: bool = False
    """If false, use the upstream interval-based schedule (default)."""

    log_spaced_until: int = 0
    """Save at powers of two (1, 2, 4, 8, ...) up to this step."""

    interval: int = 0
    """Fixed interval in steps after log_spaced_until; 0 uses checkpoint.interval.

    Prefer `interval_tokens`: a step interval changes meaning with the global batch.
    """

    interval_tokens: int = 0
    """`interval` in tokens (0 disables; takes precedence over `interval`).

    Converted to steps as
    `round(interval_tokens / (training.global_batch_size × training.seq_len))`.
    `training.global_batch_size` must be set explicitly (otherwise tokens/step
    would depend on the DP degree); `from_job_config` raises if it is not.
    """

    full_state_every: int = 0
    """Interval in steps for saves that include optimizer state (resumable). 0 disables."""

    full_state_every_tokens: int = 0
    """`full_state_every` in tokens (0 disables; takes precedence over the step version)."""

    full_state_steps: list[int] = field(default_factory=list)
    """Additional steps saved with optimizer state."""

    snapshot_folder: str = "snapshots"
    """Folder for model-only saves (under dump_folder).

    Kept separate from the checkpoint folder: upstream `_find_load_step` picks the
    largest step there, so mixing in model-only saves would make a resume start
    silently from a point without optimizer state.
    """


@dataclass
class CheckpointEval:
    """Held-out evaluation at each checkpoint."""

    enable: bool = False

    build_dir: str = ""
    """Data build directory containing the held-out split.

    Reads `<build_dir>/heldout/<source>.npz` and `<build_dir>/plan.json`.
    Empty uses training.dataset_path.
    """

    local_batch_size: int = 8

    alpha_grid: list[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 0.9]
    )
    """Fixed α grid for evaluation (α=1 clean).

    The denoising loss depends strongly on α, so val loss is comparable across
    checkpoints only on a fixed grid. The default covers the window ~[0, 0.13]
    with four points and the rest coarsely. Changing the grid changes the absolute
    val loss; keep it identical across compared runs.
    """

    max_blocks_per_source: int = 0
    """Maximum number of evaluation blocks per source (0 = all held-out blocks).

    When truncated, the first N blocks are used (deterministic) and the number
    actually used is logged.
    """

    seed: int = 1234
    """Noise seed. Independent seeds are derived per block and α, so the noise does
    not depend on world size or batch size."""

    embed_geometry: int = 0
    """Number of probes for measuring the embedding geometry (0 disables).

    The upper end of the window is where context-free nearest-neighbour decoding
    starts to succeed, which depends on the embedding geometry (max cos;
    `embed_geometry.py`). This gives a view of how the window scales with width
    that is independent of the loss profile. Logged under `embed_geometry` in
    `<dump>/val/step-*.json` at evaluation steps. Costs one [n_probe, V] matmul;
    about 1024 is a reasonable value.
    """


@dataclass
class ModelOverrides:
    """Per-run overrides of flavor (`swaflow_args`) values.

    Shapes (dim / n_layers / ...) are fixed by the flavor; these values can be set
    from the config: `schedule_alpha_max` (upper end of the truncation window),
    `qk_norm` and `antithetic_t_sampling`. `None` (default) keeps the flavor
    value. Only static constants can be set here.
    """

    schedule_alpha_max: float | None = None
    """Upper end of the window [0, alpha_max] (α=1 clean).

    Not the S-FLM `alpha_min`, which uses the opposite convention; convert with
    α_new = 1 − α_old.
    """

    qk_norm: bool | None = None
    """Apply per-head RMSNorm to q and k."""

    antithetic_t_sampling: bool | None = None
    """Stratify t within each rank (`SphereDiTForPretrain.sample_t`)."""


@dataclass
class Diagnostics:
    """Diagnostics for attention logits (QK-norm) and gradient norms.

    Recording only; nothing here changes training.
    """

    attn_stats_every: int = 50
    """Interval in microbatches for measuring the maximum attention logit (0 disables).

    The exact max needs the [L, L] logit matrix, so it is not measured every
    step; at 50 the overhead is below 0.1%. Do not combine with
    `compile.enable = true`: a Python bool flipping inside the attention block
    triggers recompilation. Set 0 when compiling (a warning is logged).
    """

    attn_stats_query_stride: int = 16
    """Query stride when measuring logits; the reported max is a lower bound."""

    grad_norm_every_step: bool = True
    """Log the pre-clipping gradient norm to JSONL every step.

    Upstream `grad_norm` is logged only every `metrics.log_freq` steps and misses
    isolated spikes. Values are buffered, adding no GPU->CPU syncs.
    """


@dataclass
class Midtrain:
    """Clean-prefix mid-training (`midtrain_data.py`).

    Active only with `training.dataset = "indexed_midtrain"`. Uses the same data
    and order as pretraining; each sample draws a split point k, keeps positions
    0..k−1 clean, and noises and scores only positions k onward.
    """

    clean_prefix_prob: float = 0.5
    """Probability of drawing k > 0; otherwise k = 0, identical to pretraining.

    Keeping part of the samples at k = 0 preserves the pretraining objective.
    0.0 is bit-identical to `indexed` (checked by a test).
    """

    k_max: int = 1024
    """k ~ U{0, …, k_max−1}; must not exceed seq_len.

    k varies so that the model does not learn a positional rule that fails to
    transfer to prompts of varying length.
    """

    seed: int = 20260903
    """Seed of the hash that determines k. It does not consume the default torch
    generator, so α and sphere-noise draws are identical to pretraining."""


@dataclass
class RNG:
    """Per-DP-rank random streams (`torchtitan/models/swaflow/rng.py`)."""

    rank_distinct_seed: bool = True
    """Seed each DP rank independently (default on).

    Off keeps the `set_determinism` seeding, so all ranks draw the same t /
    sphere noise; for diagnostics only. A warning is logged when off.
    """

    checkpoint_rng_state: bool = True
    """Save the random-stream position with each checkpoint.

    The seed fixes the stream but not its position, so without this a resumed
    run draws different t / sphere noise from step N+1 on, and `spike_skip` would
    skip different steps. Saved to `<dump>/rngstate/step-<N>/rank<r>.json`; each
    rank writes and reads its own file (no collectives).
    """


@dataclass
class Watchdog:
    """Monitoring and spike handling for long runs (`watchdog.py`).

    A run can diverge without NaN / Inf: clipping fires constantly and the α
    profile stays flat. The main signals are therefore the shape of the
    per-α-bin CE and the clipping rate; gradient spikes are secondary.
    """

    enable: bool = True
    """On by default. Detection and logging only; training is unchanged while
    `spike_skip` is off."""

    status_path: str = ""
    """Status file; empty uses `<dump>/status.json`."""

    stop_flag_path: str = ""
    """Stop-flag path; empty uses `<dump>/autoresume/STOP` (the default state
    directory of the autoresume script). Written when the watchdog stops the run
    for manual inspection."""

    grace_steps: int = 100
    """No alerts before this step.

    Early on even healthy runs have a flat α profile and frequent clipping;
    set this well after the warmup."""

    grad_flush_every: int = 50
    """Interval for checking the buffered ‖g‖ when `spike_skip = false`.

    Buffering avoids a GPU->CPU sync every step, so detection lags by up to this
    many steps (harmless, since it only logs). With `spike_skip = true` the
    decision is needed immediately, so ‖g‖ is synced every step."""

    # ---- spike definition ----
    spike_grad_factor: float = 10.0
    """A step is a spike when ‖g‖ exceeds this multiple of the recent median."""

    spike_grad_window: int = 50
    """Number of recent steps for the median."""

    spike_min_history: int = 20
    """No spike decisions until this many steps of history (avoids false alarms at start)."""

    spike_grad_floor: float = 0.0
    """Absolute floor on ‖g‖ for a spike; 0 uses `training.max_norm`.

    A ‖g‖ below the clipping threshold is not a spike (the update is bounded)."""

    # ---- spike handling ----
    spike_skip: bool = False
    """Off by default. When on, skip the optimizer update at a detected spike
    (forward/backward, data consumption and the random streams advance as usual,
    so data coverage and reproducibility are preserved).

    Off by default because gradient clipping already handles isolated outliers."""

    # ---- stop conditions (write the stop flag and exit) ----
    # No automatic rollback or hyperparameter changes. Only conditions that a
    # transient cannot trigger are used: stopping a long run on a transient costs
    # more than running a little longer in a bad state.
    stop_after_nonfinite: int = 3
    """Stop when the cumulative count of non-finite ‖g‖ reaches this (0 disables)."""

    stop_spike_rate: float = 0.3
    """Stop when the fraction of spikes over the last `clip_window` steps exceeds this.

    A rate is used rather than a run length, since healthy runs can show several
    consecutive spikes."""

    stop_after_consecutive_spikes: int = 0
    """Stop after this many consecutive spikes (default 0 = disabled).

    Transient bursts easily reach small counts; if used, set it to 20 or more."""

    # ---- stop on divergence without NaNs ----
    # A run can diverge without non-finite gradients or frequent spikes; instead
    # the clipping rate stays high and the α-profile contrast collapses. The stop
    # condition is the conjunction of the two, since either alone can hold
    # temporarily (e.g. in a hard stretch of data).
    stop_clip_rate: float = 0.6
    """Stop when, in steady state, the clipping rate exceeds this and the α contrast
    has regressed, on `stop_confirmations` consecutive checks (0 disables).

    Set between the steady-state clipping rates of healthy and diverging runs."""

    stop_confirmations: int = 3
    """Number of consecutive checks on which the conjunction must hold.

    Checks run every `metrics.log_freq` steps; requiring several avoids stopping
    a run on one noisy window."""

    stop_enable_after_steps: int = 0
    """Step from which the conjunction stop is enabled; 0 derives it from the warmup
    (`lr_scheduler.warmup_steps × stop_warmup_multiple`, at least `grace_steps`).

    Restricting to the steady state removes the dependence of the clipping rate
    on the early transient, so thresholds need no recalibration when the warmup
    changes and the transient cannot stop the run. Before this step the
    watchdog only notifies."""

    stop_warmup_multiple: float = 3.0
    """Multiplier on the warmup when `stop_enable_after_steps = 0`."""

    # ---- alert thresholds ----
    clip_window: int = 200
    """Number of recent steps for the clipping rate."""

    clip_rate_alert: float = 0.5
    """Alert when the clipping rate exceeds this after the early transient.

    Expected: clipping fires during the transient, then drops to 0-10%."""

    alpha_flatten_ratio: float = 0.5
    """Alert when the α-profile contrast falls below this fraction of its best so far.

    Regression is used instead of an absolute threshold because even healthy
    runs start flat."""

    alpha_contrast_arm: float = 1.0
    """The flattening alert is armed only after the best contrast exceeds this."""

    alpha_contrast_min: float = 0.0
    """Absolute floor on the contrast (0 disables)."""

    alert_cooldown: int = 500
    """Minimum number of steps between repeated alerts of the same kind."""


@dataclass
class SFT:
    """SFT dataloader options (active only with `training.dataset = "sft"`).

    By default `SFTBlockDataset` makes one pass over the build. Runs longer than
    one pass (e.g. TinyGSM SFT) can instead cycle over the build here rather than
    materializing repeated copies with `build_sft.py --repeats`.
    """

    allow_multiple_epochs: bool = False
    """Cycle over the build indefinitely. If False (default), data ends after one pass.

    The default avoids a silent failure: a one-epoch run whose `training.steps`
    exceeds one pass should stop rather than silently start a second pass. Each
    pass replays the same blocks in the same order (`_sample_index` wraps to 0),
    unlike `build_sft.py --repeats`, which re-packs each repeat.
    """


@dataclass
class SwaFlowConfig:
    checkpoint_schedule: CheckpointSchedule = field(
        default_factory=CheckpointSchedule
    )
    eval: CheckpointEval = field(default_factory=CheckpointEval)
    model: ModelOverrides = field(default_factory=ModelOverrides)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    midtrain: Midtrain = field(default_factory=Midtrain)
    sft: SFT = field(default_factory=SFT)
    rng: RNG = field(default_factory=RNG)
    watchdog: Watchdog = field(default_factory=Watchdog)


@dataclass
class JobConfig:
    """Extension merged into the upstream JobConfig."""

    swaflow: SwaFlowConfig = field(default_factory=SwaFlowConfig)
