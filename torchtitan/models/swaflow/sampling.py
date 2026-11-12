# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0
#
# The velocity fields `_velocity_exact` / `_velocity_sample`, the geodesic step
# size `geodesic_step_size` and the categorical sampler `sample_categorical` are
# adapted from S-FLM (https://github.com/jdeschena/s-flm), samplers.py:
# `sfm_compute_velocity` / `sfm_step_size` / `sample_categorical` / `SFMSampler`,
# Copyright (c) the S-FLM authors, licensed under the Apache License, Version 2.0.
# Changes: rewritten for the time convention alpha=1 clean, adapted to torchtitan's
# dtype policy, chunked velocity computation, and refactored into pure functions.

"""S-FLM generation: integrate the geodesic ODE on the sphere to produce tokens.

Time convention: α = 1 is clean, α = 0 is pure noise. Generation starts from
pure noise (uniform on the sphere) and runs α: 0 -> 1, so the training time t
(t=0 clean, t=1 noise) runs backwards, 1 -> 0. A reversed direction still gives
finite numbers but meaningless output. The S-FLM reference defaults to the
opposite convention (invert_time_convention=true); convert with α = 1 − α_ref.

One step:

    1. logits = model(x_t, α_t)                       # denoiser
    2. p = softmax(logits / T)                        # posterior (temperature on logits)
    3. v = Σ_k p_k · log_{x_t}(e_k)                   # marginalize over the embedding table
    4. dt = (α_s − α_t) / (1 − α_t)                   # fraction of the remaining distance
    5. x_s = exp_{x_t}(dt · v)                        # geodesic step on the sphere
    The last step skips 3-5 and decodes by argmax or categorical sampling.

v is the posterior expectation of the tangent vectors from x_t toward each
embedding e_k; if p is a point mass, exp_{x_t}(dt·v) = slerp(x_t, e_k; dt).

Precision: logits are cast to fp32 before log_softmax; sphere operations
(velocity, exp_map) run in fp64 by default (`slerp_float64=True`);
`use_float64` keeps log_p in fp64.

Memory: velocity="exact" builds [B, L, V] intermediates (cosω, ω, scale). In fp64
with V=49152, L=2048, B=8 each is 6.4 GB; use `chunk_size` (sequence-axis
chunks) or `top_k_velocity` for long sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from torchtitan.models.swaflow.model import SphereDiT, SphereDiTForPretrain
from torchtitan.models.swaflow.schedules import LogLinear, NoiseSchedule
from torchtitan.models.swaflow.sphere_ops import exp_map, log_map, sphere_normalize

__all__ = [
    "SamplerConfig",
    "alpha_grid",
    "generate",
    "geodesic_step_size",
    "sample_categorical",
    "sfm_velocity",
    "sphere_prior",
]

VELOCITY_MODES = ("exact", "sample")
NOISE_REMOVAL_MODES = ("greedy", "ancestral")


@dataclass
class SamplerConfig:
    """Sampler settings. The defaults give a deterministic path.

    Attributes:
        steps: number of ODE steps = NFE. The time grid has steps+1 points; the
            model is evaluated at the first steps points, and the last of these
            is used for decoding (steps−1 integration steps, counted as in the
            S-FLM SFMSampler).
        velocity: "exact" marginalizes over the vocabulary, Σ_k p_k log_x(e_k);
            "sample" samples one token from the posterior and uses its log_map.
            Default exact (deterministic).
        top_k_velocity: vocabulary size used for the velocity; −1 for all. A
            positive k keeps the top-k of log_p, renormalizes them with
            log_softmax and marginalizes over those.
        noise_removal: decoding at the last step: "greedy" (argmax) or
            "ancestral" (categorical sample, Gumbel-max).
        temperature: temperature on the logits (softmax of logits/T).
        eps: geometry clamp (cosω to [−1+eps, 1−eps]; sin ω and 1−α from below).
            The S-FLM `algo.eps`; default 1e-6.
        time_eps: end of the time grid; the α grid starts at α(t=1−time_eps),
            avoiding exactly α=0. The S-FLM sampler `eps`; default 1e-5.
            Distinct from eps.
        use_float64: keep the posterior (log_p) in fp64.
        slerp_float64: run sphere operations (x, E, velocity, exp_map) in fp64.
        chunk_size: split the velocity computation along the sequence axis into
            chunks of this length (None: no split). Same values, lower peak
            memory for the [B, L, V] intermediates.
        top_k: logits filter; not implemented (raises unless −1).
        p_nucleus: nucleus filter; not implemented (raises unless 1.0).
    """

    steps: int = 32
    velocity: str = "exact"
    top_k_velocity: int = -1
    noise_removal: str = "greedy"
    temperature: float = 1.0
    eps: float = 1e-6
    time_eps: float = 1e-5
    use_float64: bool = False
    slerp_float64: bool = True
    chunk_size: int | None = None
    # Not implemented; a non-default value raises instead of being ignored.
    top_k: int = -1
    p_nucleus: float = 1.0

    def __post_init__(self) -> None:
        if self.velocity not in VELOCITY_MODES:
            raise ValueError(
                f"velocity must be one of {VELOCITY_MODES}, got {self.velocity!r}"
            )
        if self.noise_removal not in NOISE_REMOVAL_MODES:
            raise ValueError(
                f"noise_removal must be one of {NOISE_REMOVAL_MODES}, "
                f"got {self.noise_removal!r}"
            )
        if self.steps < 2:
            raise ValueError(f"steps must be >= 2, got {self.steps}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k_velocity == 0 or self.top_k_velocity < -1:
            raise ValueError(
                f"top_k_velocity must be -1 or a positive int, "
                f"got {self.top_k_velocity}"
            )
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.top_k != -1 or self.p_nucleus != 1.0:
            raise NotImplementedError(
                "top_k / p_nucleus filtering is not implemented; "
                "call with top_k=-1, p_nucleus=1.0"
            )

    @property
    def sphere_dtype(self) -> torch.dtype | None:
        return torch.float64 if self.slerp_float64 else None


# --------------------------------------------------------------------------
# Pure functions (framework-independent)
# --------------------------------------------------------------------------


def sphere_prior(
    shape: tuple[int, ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample uniformly on the sphere (the starting point α=0 of generation).

    A normalized isotropic Gaussian is uniform on the unit sphere.
    """
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return sphere_normalize(x)


def alpha_grid(
    schedule: NoiseSchedule,
    steps: int,
    *,
    time_eps: float = 1e-5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """α grid [steps+1] for generation, ascending from the noise end to the clean end.

    t=0 is clean and t=1 is noise, so generation runs t from 1−time_eps down to
    0 and α increases through the schedule. time_eps avoids exactly pure noise
    (t=1), as S-FLM's `SFMSampler.init_state(eps=1e-5)` does (its
    `linspace(eps, 1)` avoids the same end under the opposite convention).
    """
    t = torch.linspace(1.0 - time_eps, 0.0, steps + 1, device=device, dtype=dtype)
    return schedule.alpha(t).to(dtype)


def geodesic_step_size(alpha_t: Tensor, alpha_s: Tensor, eps: float) -> Tensor:
    """Fraction of the geodesic to travel, dt = (α_s − α_t) / (1 − α_t).

    The slerp parameter that moves from α_t to α_s along the geodesic from x_t
    toward a target e. If the posterior is a point mass,
    `exp_{x_t}(dt · log_{x_t}(e)) = slerp(x_t, e; dt)` holds exactly.

    The denominator 1 − α_t (remaining noise) vanishes at the clean end, so it is
    clamped from below by eps (as in S-FLM's `sfm_step_size`).
    """
    return (alpha_s - alpha_t) / (1.0 - alpha_t).clamp_min(eps)


def sample_categorical(
    probs: Tensor, *, generator: torch.Generator | None = None
) -> Tensor:
    """Sample from a categorical distribution over the last axis (Gumbel-max trick).

    argmax_k (p_k / g_k) with g_k = −log u_k has the same distribution as
    argmax_k (log p_k + Gumbel). 1e-10 guards log(0) (as in S-FLM's
    `sample_categorical`).
    """
    u = torch.rand(probs.shape, device=probs.device, dtype=probs.dtype, generator=generator)
    gumbel_noise = 1e-10 - (u + 1e-10).log()
    return (probs / gumbel_noise).argmax(dim=-1)


def _velocity_exact(x: Tensor, embed: Tensor, p: Tensor, eps: float) -> Tensor:
    """Compute v = Σ_k p_k · log_x(e_k) in expanded form.

        log_x(e) = ω/sin ω · (e − cos ω · x),  cos ω = ⟨x, e⟩

    so Σ_k p_k log_x(e_k) = Σ_k (p_k·ω_k/sin ω_k)·e_k
                             − x · Σ_k (p_k·ω_k/sin ω_k)·cos ω_k.
    The second term removes the component along x (projection onto the tangent space).

    Args:
        x: [B, L, d] unit vectors
        embed: [V, d] (full vocabulary) or [B, L, k, d] (after top-k)
        p: [B, L, V] or [B, L, k], posterior probabilities matching embed
        eps: clamp for cos ω / sin ω
    """
    if embed.ndim == 2:
        cos_omega = torch.einsum("bld,vd->blv", x, embed)
    else:
        cos_omega = torch.einsum("bld,blkd->blk", x, embed)
    cos_omega = cos_omega.clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(cos_omega)
    p_scale = p * (omega / omega.sin().clamp_min(eps))
    if embed.ndim == 2:
        term1 = torch.einsum("blv,vd->bld", p_scale, embed)
    else:
        term1 = torch.einsum("blk,blkd->bld", p_scale, embed)
    term2 = x * (p_scale * cos_omega).sum(dim=-1, keepdim=True)
    return term1 - term2


def _velocity_sample(
    x: Tensor,
    embed: Tensor,
    p: Tensor,
    eps: float,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample one token from the posterior and return the log_map toward its embedding.

    A one-sample estimate of the exact expectation (unbiased, with variance).
    """
    target_idx = sample_categorical(p, generator=generator)  # [B, L]
    if embed.ndim == 2:
        target = embed[target_idx]
    else:
        b, seq = target_idx.shape
        target = embed[
            torch.arange(b, device=embed.device)[:, None],
            torch.arange(seq, device=embed.device)[None, :],
            target_idx,
        ]
    return log_map(x, target, eps=eps)


def sfm_velocity(
    x: Tensor,
    embed: Tensor,
    log_p: Tensor,
    *,
    mode: str = "exact",
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Velocity field v(x, p): [B, L, d] in the tangent space at x.

    Args:
        x: [B, L, d] current points on the sphere (unit vectors)
        embed: [V, d] normalized embedding table, or [B, L, k, d] (top-k)
        log_p: [B, L, V] or [B, L, k] log posterior
        mode: "exact" (marginalize) / "sample" (one sample)
    """
    p = log_p.exp()
    if mode == "exact":
        return _velocity_exact(x, embed, p, eps)
    if mode == "sample":
        return _velocity_sample(x, embed, p, eps, generator)
    raise ValueError(f"unknown velocity mode: {mode!r}")


def _select_top_k_velocity(
    log_p: Tensor, embed: Tensor, k: int
) -> tuple[Tensor, Tensor]:
    """Keep the top-k entries of log_p and renormalize them with log_softmax.

    Returns:
        (log_p_k [B, L, k], embed_k [B, L, k, d])
    """
    log_p_k, top_idx = torch.topk(log_p, k, dim=-1)
    return torch.log_softmax(log_p_k, dim=-1), F.embedding(top_idx, embed)


def _project_prefix(
    x: Tensor, prefix: Tensor | None, prefix_lengths: Tensor | None
) -> Tensor:
    """Overwrite the prefix positions with the prefix values (in place).

    x is either continuous vectors [B, L, d] or token ids [B, L]; prefix is the
    matching [B, P, d] / [B, P]. Positions at or beyond prefix_lengths [B] are
    left unchanged (variable-length prefixes).
    """
    if prefix is None:
        return x
    p_len = prefix.shape[1]
    positions = torch.arange(p_len, device=x.device)
    mask = positions[None, :] < prefix_lengths[:, None]  # [B, P]
    if x.ndim == 3:
        mask = mask.unsqueeze(-1)
    x[:, :p_len] = torch.where(mask, prefix.to(x.dtype), x[:, :p_len])
    return x


# --------------------------------------------------------------------------
# Generation loop
# --------------------------------------------------------------------------


@dataclass
class _StepTrace:
    """Intermediate values of one step, passed to the `on_step` hook (diagnostics)."""

    step_idx: int
    alpha_t: float
    alpha_s: float | None
    log_p: Tensor
    velocity: Tensor | None
    dt: Tensor | None
    x_next: Tensor
    is_last: bool
    extra: dict = field(default_factory=dict)


def _denoise(model: SphereDiT, x: Tensor, alpha: Tensor) -> Tensor:
    """Call the backbone forward ([B, L, d], [B] -> logits [B, L, V]).

    `SphereDiTForPretrain` overrides forward for training (it takes token ids and
    adds noise itself), so the base-class forward is called explicitly there.
    Otherwise `model(x, alpha)` is called, so subclasses that replace forward
    (e.g. oracle denoisers in tests) also work.
    """
    if isinstance(model, SphereDiTForPretrain):
        return SphereDiT.forward(model, x, alpha)
    return model(x, alpha)


@torch.no_grad()
def generate(
    model: SphereDiT,
    *,
    num_samples: int | None = None,
    seq_len: int | None = None,
    config: SamplerConfig | None = None,
    schedule: NoiseSchedule | None = None,
    device: torch.device | str | None = None,
    noise: Tensor | None = None,
    alphas: Tensor | None = None,
    prefix_tokens: Tensor | None = None,
    prefix_lengths: Tensor | None = None,
    generator: torch.Generator | None = None,
    on_step: Callable[[_StepTrace], None] | None = None,
) -> Tensor:
    """Generate tokens by integrating the geodesic ODE from pure noise.

    Args:
        model: `SphereDiT` (or its training subclass), preferably in eval mode.
        num_samples, seq_len: output shape; may be omitted when `noise` is given.
        config: `SamplerConfig` (default: exact / greedy / T=1 / steps=32).
        schedule: noise schedule. None uses `model.schedule` (the training
            schedule, including its truncation window), or `LogLinear()` if the
            model has none.
        device: defaults to the device of the model parameters.
        noise: initial points [B, L, d]; None samples uniformly on the sphere.
            Given noise is normalized onto the sphere as well.
        alphas: the α grid [steps+1], given directly (ignores schedule/time_eps).
        prefix_tokens: [B, P] token ids to condition on.
        prefix_lengths: [B] valid prefix length per sample (may be < P).
        generator: random source for ancestral decoding / velocity="sample".
        on_step: hook called with a `_StepTrace` after each step (diagnostics).

    Returns:
        token ids [B, L] (int64)

    Generation runs α: 0 -> 1 (noise -> clean); see the module docstring.
    """
    cfg = config or SamplerConfig()
    if not isinstance(model, SphereDiT):
        raise TypeError(f"model must be a SphereDiT, got {type(model).__name__}")

    param = next(model.parameters())
    device = torch.device(device) if device is not None else param.device
    dim = model.model_args.dim

    # ---- initial points (pure noise, α=0) ----
    if noise is not None:
        x_t = sphere_normalize(noise.to(device=device, dtype=torch.float32))
    else:
        if num_samples is None or seq_len is None:
            raise ValueError("num_samples and seq_len are required when noise is None")
        x_t = sphere_prior(
            (num_samples, seq_len, dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    bsz, seq = x_t.shape[0], x_t.shape[1]

    # ---- prefix embeddings (projected at the start and after every step) ----
    prefix_embeds: Tensor | None = None
    if (prefix_tokens is None) != (prefix_lengths is None):
        raise ValueError("prefix_tokens and prefix_lengths must both be set or unset")
    if prefix_tokens is not None:
        prefix_tokens = prefix_tokens.to(device)
        prefix_lengths = prefix_lengths.to(device)
        if prefix_tokens.shape[0] != bsz:
            raise ValueError(
                f"prefix batch {prefix_tokens.shape[0]} != num_samples {bsz}"
            )
        if prefix_tokens.shape[1] > seq:
            raise ValueError("prefix is longer than the sequence")
        prefix_embeds = model.get_sphere_embeddings(prefix_tokens).to(x_t.dtype)
        _project_prefix(x_t, prefix_embeds, prefix_lengths)

    # ---- α grid ----
    if alphas is not None:
        alpha_seq = alphas.to(device=device, dtype=torch.float64).reshape(-1)
        if alpha_seq.numel() != cfg.steps + 1:
            raise ValueError(
                f"alphas must have steps+1 = {cfg.steps + 1} entries, "
                f"got {alpha_seq.numel()}"
            )
    else:
        sched = schedule or getattr(model, "schedule", None) or LogLinear()
        alpha_seq = alpha_grid(
            sched, cfg.steps, time_eps=cfg.time_eps, device=device
        )

    # ---- normalized embedding table (for marginalizing the velocity) ----
    embed_table = sphere_normalize(model.tok_embeddings.weight.detach())
    if cfg.sphere_dtype is not None:
        embed_table = embed_table.to(cfg.sphere_dtype)

    tokens: Tensor | None = None
    for step_idx in range(cfg.steps):
        is_last = step_idx == cfg.steps - 1
        alpha_t = alpha_seq[step_idx]

        logits = _denoise(model, x_t, alpha_t.to(torch.float32).expand(bsz))
        log_p = torch.log_softmax(logits.float() / cfg.temperature, dim=-1)
        if cfg.use_float64:
            log_p = log_p.to(torch.float64)

        if is_last:
            # last step: decode instead of integrating
            if cfg.noise_removal == "greedy":
                tokens = log_p.argmax(dim=-1)
            else:
                tokens = sample_categorical(log_p.exp(), generator=generator)
            _project_prefix(tokens, prefix_tokens, prefix_lengths)
            if on_step is not None:
                on_step(
                    _StepTrace(
                        step_idx=step_idx,
                        alpha_t=float(alpha_t),
                        alpha_s=None,
                        log_p=log_p,
                        velocity=None,
                        dt=None,
                        x_next=tokens,
                        is_last=True,
                    )
                )
            break

        alpha_s = alpha_seq[step_idx + 1]
        x = x_t if cfg.sphere_dtype is None else x_t.to(cfg.sphere_dtype)
        dt = geodesic_step_size(
            alpha_t.to(x.dtype), alpha_s.to(x.dtype), cfg.eps
        )

        vel = _velocity_chunked(x, embed_table, log_p, cfg, generator)
        x_next = exp_map(x, dt * vel)
        x_t = x_next.to(x_t.dtype)
        _project_prefix(x_t, prefix_embeds, prefix_lengths)

        if on_step is not None:
            on_step(
                _StepTrace(
                    step_idx=step_idx,
                    alpha_t=float(alpha_t),
                    alpha_s=float(alpha_s),
                    log_p=log_p,
                    velocity=vel,
                    dt=dt,
                    x_next=x_t,
                    is_last=False,
                )
            )

    assert tokens is not None  # steps >= 2 guarantees a decode step
    return tokens


def _velocity_chunked(
    x: Tensor,
    embed_table: Tensor,
    log_p: Tensor,
    cfg: SamplerConfig,
    generator: torch.Generator | None,
) -> Tensor:
    """Compute the velocity in chunks along the sequence axis (same values as unchunked).

    `chunk_size=None` computes it in one pass. The exact [B, L, V] intermediates
    are large in fp64, so chunking caps the peak memory for long sequences.

    log_p is not cast: posterior precision (`use_float64`) and sphere precision
    (`slerp_float64`) are independent. p = exp(log_p) stays in log_p's dtype and
    meets the fp64 sphere tensors by type promotion (same order as the reference).
    """
    chunk = cfg.chunk_size or x.shape[1]
    outs = []
    for start in range(0, x.shape[1], chunk):
        sl = slice(start, start + chunk)
        x_c, log_p_c = x[:, sl], log_p[:, sl]
        embed_c = embed_table
        if cfg.top_k_velocity > 0:
            log_p_c, embed_c = _select_top_k_velocity(
                log_p_c, embed_table, cfg.top_k_velocity
            )
        outs.append(
            sfm_velocity(
                x_c,
                embed_c,
                log_p_c,
                mode=cfg.velocity,
                eps=cfg.eps,
                generator=generator,
            )
        )
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
