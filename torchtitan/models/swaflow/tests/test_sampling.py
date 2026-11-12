# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generation core (sampling.py).

Numerical comparison against the reference implementation needs GPU dumps and
is done separately. Only CPU property tests live here:

- direction of the time convention (alpha=1 is clean). A reversed direction
  still gives finite, plausible-looking values, so it is pinned directly first
- geometric properties of the velocity field (tangency; equality with slerp
  for a one-point posterior)
- invariants of the options (top_k_velocity / chunk_size / prefix / noise_removal)
"""

from __future__ import annotations

import pytest
import torch

from torchtitan.models.swaflow.model import (
    PROFILE_SFLM,
    SwaFlowModelArgs,
    SphereDiT,
    SphereDiTForPretrain,
)
from torchtitan.models.swaflow.sampling import (
    SamplerConfig,
    alpha_grid,
    generate,
    geodesic_step_size,
    sample_categorical,
    sfm_velocity,
    sphere_prior,
)
from torchtitan.models.swaflow.schedules import LogLinear, TruncatedScheduleWrapper
from torchtitan.models.swaflow.sphere_ops import exp_map, slerp, sphere_normalize

V, D, L, B = 41, 32, 8, 2


def make_args(**kw) -> SwaFlowModelArgs:
    base = dict(
        dim=D, n_layers=2, n_heads=4, vocab_size=V, cond_dim=16,
        sigma_freq_dim=32, max_seq_len=L, backbone_profile=PROFILE_SFLM,
    )
    base.update(kw)
    return SwaFlowModelArgs(**base)


def randomized_model(seed: int = 0, cls=SphereDiT, std: float = 0.05, **kw):
    """A model with randomized weights.

    A fresh init zero-initializes adaLN and lm_head, so the logits are all zero
    and any implementation gives the same result, making the tests meaningless.

    Smaller std keeps the adaLN gates near 0 and the blocks near the identity.
    Tests of interactions between positions (attention) should use a larger std.
    """
    model = cls(make_args(**kw))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.empty(p.shape).normal_(0.0, std, generator=g))
    return model.eval()


# --------------------------------------------------------------------------
# Time convention (checked first)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [2, 8, 32])
def test_alpha_grid_runs_noise_to_clean(steps: int) -> None:
    """The alpha grid must increase from the noise end to the clean end."""
    grid = alpha_grid(LogLinear(), steps, time_eps=1e-5)
    assert grid.numel() == steps + 1
    assert torch.all(grid.diff() > 0), "alpha decreases: generation direction reversed"
    assert grid[0] < 1e-4, "start is not pure noise (alpha ~ 0)"
    assert grid[-1] == pytest.approx(1.0), "end is not clean (alpha = 1)"


def test_alpha_grid_respects_truncation_window() -> None:
    """With a truncation window the grid ends at alpha_max."""
    sched = TruncatedScheduleWrapper(LogLinear(0.0), alpha_max=0.999)
    grid = alpha_grid(sched, 8, time_eps=1e-5)
    assert grid[-1] == pytest.approx(0.999)
    assert torch.all(grid.diff() > 0)


def test_step_size_is_fraction_of_remaining_noise() -> None:
    """dt = (alpha_s - alpha_t)/(1 - alpha_t) is a progress fraction in [0, 1], equal to 1 on the last interval."""
    alpha = alpha_grid(LogLinear(), 8, time_eps=1e-5)
    dt = geodesic_step_size(alpha[:-1], alpha[1:], eps=1e-6)
    assert torch.all(dt > 0), "non-positive dt: time direction reversed"
    assert torch.all(dt <= 1.0 + 1e-9)
    assert dt[-1] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Geometry of the velocity field
# --------------------------------------------------------------------------


def test_velocity_is_tangent_to_the_sphere() -> None:
    """v lies in the tangent space at x (<x, v> = 0)."""
    g = torch.Generator().manual_seed(1)
    x = sphere_normalize(torch.randn(B, L, D, generator=g, dtype=torch.float64))
    embed = sphere_normalize(torch.randn(V, D, generator=g, dtype=torch.float64))
    log_p = torch.log_softmax(torch.randn(B, L, V, generator=g, dtype=torch.float64), -1)
    v = sfm_velocity(x, embed, log_p)
    assert (x * v).sum(-1).abs().max() < 1e-10


def test_one_hot_posterior_step_equals_slerp() -> None:
    """For a one-point posterior, exp_x(dt*v) equals slerp(x, e; dt) exactly.

    This is the definition of the velocity field (v = log_x(e)); errors in the
    marginalization formula fail here.
    """
    g = torch.Generator().manual_seed(2)
    x = sphere_normalize(torch.randn(B, L, D, generator=g, dtype=torch.float64))
    embed = sphere_normalize(torch.randn(V, D, generator=g, dtype=torch.float64))
    target = torch.randint(0, V, (B, L), generator=g)
    log_p = torch.full((B, L, V), -1e30, dtype=torch.float64)
    log_p.scatter_(-1, target.unsqueeze(-1), 0.0)

    v = sfm_velocity(x, embed, log_p, eps=1e-12)
    for dt in (0.1, 0.5, 1.0):
        got = exp_map(x, dt * v)
        want = slerp(x, embed[target], dt, eps=1e-12)
        assert (got - want).abs().max() < 1e-9, dt


def test_top_k_velocity_matches_full_when_k_is_vocab_size() -> None:
    """top-k with k = V gives the same result as the full vocabulary (selection and renormalization are sound)."""
    model = randomized_model(seed=3)
    noise = torch.randn(B, L, D)
    full = generate(model, config=SamplerConfig(steps=6), noise=noise)
    topk = generate(
        model, config=SamplerConfig(steps=6, top_k_velocity=V), noise=noise
    )
    assert torch.equal(full, topk)


def test_top_k_velocity_changes_the_trajectory() -> None:
    """A small k changes the result (the argument is actually used)."""
    model = randomized_model(seed=4)
    noise = torch.randn(B, L, D)
    full = generate(model, config=SamplerConfig(steps=6), noise=noise)
    topk = generate(model, config=SamplerConfig(steps=6, top_k_velocity=2), noise=noise)
    assert not torch.equal(full, topk)


def test_chunking_does_not_change_numerics() -> None:
    model = randomized_model(seed=5)
    noise = torch.randn(B, L, D)
    whole = generate(model, config=SamplerConfig(steps=6), noise=noise)
    chunked = generate(model, config=SamplerConfig(steps=6, chunk_size=3), noise=noise)
    assert torch.equal(whole, chunked)


# --------------------------------------------------------------------------
# Generation loop (end to end)
# --------------------------------------------------------------------------


class OracleDenoiser(SphereDiT):
    """A denoiser that always predicts a fixed target sequence (end-to-end check of the integration path)."""

    def set_target(self, target: torch.Tensor) -> None:
        self._target = target

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (x.shape[0], x.shape[1], V), -30.0, dtype=torch.float32, device=x.device
        )
        return logits.scatter(-1, self._target.unsqueeze(-1), 30.0)


def test_oracle_denoiser_converges_to_its_target() -> None:
    """With a perfect denoiser, generation converges to its target sequence.

    It does not converge if the time direction, dt or the geodesic step is
    broken (with a reversed direction, dt < 0 moves toward the noise side).
    """
    model = randomized_model(seed=6, cls=OracleDenoiser)
    target = torch.randint(0, V, (B, L), generator=torch.Generator().manual_seed(7))
    model.set_target(target)

    tokens = generate(
        model,
        num_samples=B,
        seq_len=L,
        config=SamplerConfig(steps=16),
        generator=torch.Generator().manual_seed(8),
    )
    assert torch.equal(tokens, target)


def test_trajectory_approaches_the_target_monotonically() -> None:
    """Under the oracle, the geodesic distance from x_t to the target embedding shrinks monotonically.

    With a one-point posterior each step is a slerp toward the target, and for
    equally spaced alpha the remaining distance falls linearly as
    omega*(1 - alpha_k). The last recorded state is at alpha[steps-1] (the
    alpha=1 grid point is used only for decoding), so the remaining distance is
    ~1/steps of the initial distance.
    """
    steps = 16
    model = randomized_model(seed=9, cls=OracleDenoiser)
    target = torch.randint(0, V, (B, L), generator=torch.Generator().manual_seed(10))
    model.set_target(target)
    e_target = sphere_normalize(model.tok_embeddings.weight.detach())[target]

    dists: list[float] = []

    def hook(trace) -> None:
        if not trace.is_last:
            cos = (trace.x_next.double() * e_target.double()).sum(-1).clamp(-1, 1)
            dists.append(cos.arccos().mean().item())

    generate(
        model, num_samples=B, seq_len=L, config=SamplerConfig(steps=steps),
        generator=torch.Generator().manual_seed(11), on_step=hook,
    )
    assert len(dists) == steps - 1
    assert all(b < a for a, b in zip(dists, dists[1:])), dists
    assert dists[-1] < dists[0] * 1.5 / steps, dists


def test_prefix_tokens_are_preserved() -> None:
    model = randomized_model(seed=12)
    prefix = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    lengths = torch.tensor([4, 2])
    tokens = generate(
        model, num_samples=B, seq_len=L, config=SamplerConfig(steps=6),
        prefix_tokens=prefix, prefix_lengths=lengths,
        generator=torch.Generator().manual_seed(13),
    )
    assert torch.equal(tokens[0, :4], prefix[0])
    assert torch.equal(tokens[1, :2], prefix[1, :2])


def _final_state(model, **kw) -> torch.Tensor:
    """Returns the continuous state z_t just before decoding (more sensitive than tokens)."""
    states: list[torch.Tensor] = []
    generate(model, on_step=lambda t: states.append(t.x_next) if not t.is_last else None, **kw)
    return states[-1]


def test_prefix_conditioning_changes_the_continuation() -> None:
    """The prefix affects later positions (with fixed initial noise the continuation changes).

    Compared on the continuous state before decoding rather than on tokens: a
    random-weight model's logits tend to stick to a few tokens, so argmax alone
    may show no difference. A larger std is used because small adaLN gates
    suppress the attention contribution and hide interactions between
    positions (with std=0.05 the difference is ~1e-6).
    """
    model = randomized_model(seed=14, std=0.3)
    noise = torch.randn(B, L, D)
    cfg = SamplerConfig(steps=6)
    plain = _final_state(model, config=cfg, noise=noise)
    conditioned = _final_state(
        model, config=cfg, noise=noise,
        prefix_tokens=torch.tensor([[3, 4], [5, 6]]),
        prefix_lengths=torch.tensor([2, 2]),
    )
    assert (plain[:, 2:] - conditioned[:, 2:]).abs().max() > 1e-3


def test_pretrain_wrapper_is_usable_as_a_denoiser() -> None:
    """The training wrapper (forward takes tokens) also works as a denoiser."""
    model = randomized_model(seed=15, cls=SphereDiTForPretrain)
    tokens = generate(
        model, num_samples=B, seq_len=L, config=SamplerConfig(steps=4),
        generator=torch.Generator().manual_seed(16),
    )
    assert tokens.shape == (B, L) and tokens.dtype == torch.int64
    assert int(tokens.max()) < V


@pytest.mark.parametrize("velocity", ["exact", "sample"])
@pytest.mark.parametrize("noise_removal", ["greedy", "ancestral"])
def test_all_option_combinations_run(velocity: str, noise_removal: str) -> None:
    model = randomized_model(seed=17)
    tokens = generate(
        model, num_samples=B, seq_len=L,
        config=SamplerConfig(steps=5, velocity=velocity, noise_removal=noise_removal),
        generator=torch.Generator().manual_seed(18),
    )
    assert tokens.shape == (B, L)
    assert int(tokens.min()) >= 0 and int(tokens.max()) < V


def test_greedy_is_deterministic_and_ancestral_is_not() -> None:
    model = randomized_model(seed=19)
    noise = torch.randn(B, L, D)
    greedy = [
        generate(model, config=SamplerConfig(steps=6), noise=noise) for _ in range(2)
    ]
    assert torch.equal(*greedy)
    anc = [
        generate(
            model, config=SamplerConfig(steps=6, noise_removal="ancestral"),
            noise=noise, generator=torch.Generator().manual_seed(seed),
        )
        for seed in (0, 1)
    ]
    assert not torch.equal(*anc)


def test_temperature_sharpens_the_posterior() -> None:
    """Temperature scales the logits (low temperature sharpens the posterior and changes the path).

    argmax is temperature-invariant, so tokens cannot detect it; the check uses
    the posterior's maximum probability and the resulting velocity field.
    """
    model = randomized_model(seed=20)
    noise = torch.randn(B, L, D)
    step0: dict[float, tuple[float, torch.Tensor]] = {}
    for temp in (1.0, 0.1):
        traces: list = []
        generate(
            model, config=SamplerConfig(steps=6, temperature=temp), noise=noise,
            on_step=traces.append,
        )
        t0 = traces[0]
        step0[temp] = (t0.log_p.exp().max().item(), t0.velocity)
    assert step0[0.1][0] > step0[1.0][0], "posterior not sharper at low temperature"
    assert (step0[0.1][1] - step0[1.0][1]).abs().max() > 1e-4


def test_explicit_alpha_grid_overrides_the_schedule() -> None:
    model = randomized_model(seed=21)
    noise = torch.randn(B, L, D)
    alphas = alpha_grid(LogLinear(), 6, time_eps=1e-5)
    with_grid = generate(
        model, config=SamplerConfig(steps=6), noise=noise, alphas=alphas
    )
    with_sched = generate(
        model, config=SamplerConfig(steps=6), noise=noise, schedule=LogLinear()
    )
    assert torch.equal(with_grid, with_sched)


# --------------------------------------------------------------------------
# Helpers and argument validation
# --------------------------------------------------------------------------


def test_sphere_prior_is_on_the_unit_sphere() -> None:
    x = sphere_prior((4, 5, D), generator=torch.Generator().manual_seed(22))
    assert (x.norm(dim=-1) - 1.0).abs().max() < 1e-6


def test_sample_categorical_matches_the_target_distribution() -> None:
    probs = torch.tensor([0.1, 0.6, 0.3]).expand(40000, 3).contiguous()
    draws = sample_categorical(probs, generator=torch.Generator().manual_seed(23))
    freq = torch.bincount(draws, minlength=3).float() / draws.numel()
    assert (freq - probs[0]).abs().max() < 0.01


def test_unimplemented_filters_raise() -> None:
    with pytest.raises(NotImplementedError):
        SamplerConfig(top_k=5)
    with pytest.raises(NotImplementedError):
        SamplerConfig(p_nucleus=0.9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"velocity": "euler"},
        {"noise_removal": "none"},
        {"steps": 1},
        {"temperature": 0.0},
        {"top_k_velocity": 0},
        {"chunk_size": 0},
    ],
)
def test_invalid_config_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SamplerConfig(**kwargs)


def test_prefix_args_must_be_paired() -> None:
    model = randomized_model(seed=24)
    with pytest.raises(ValueError):
        generate(
            model, num_samples=B, seq_len=L, config=SamplerConfig(steps=4),
            prefix_tokens=torch.zeros(B, 2, dtype=torch.long),
        )


def test_alphas_length_is_validated() -> None:
    model = randomized_model(seed=25)
    with pytest.raises(ValueError):
        generate(
            model, num_samples=B, seq_len=L, config=SamplerConfig(steps=4),
            alphas=torch.linspace(0.0, 1.0, 3),
        )
