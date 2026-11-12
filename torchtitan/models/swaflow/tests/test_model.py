# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torchtitan.models.swaflow.loss import swaflow_denoising_loss
from torchtitan.models.swaflow.model import (
    PROFILE_SFLM,
    PROFILE_PRODUCTION,
    SwaFlowModelArgs,
    SphereDiT,
    SphereDiTForPretrain,
    alpha_to_sigma,
)
from torchtitan.models.swaflow.schedules import LogLinear, TruncatedScheduleWrapper
from torchtitan.models.swaflow.sphere_ops import sphere_normalize

B, L, D, V = 2, 16, 64, 128


def tiny_args(**overrides) -> SwaFlowModelArgs:
    defaults = dict(
        dim=D,
        n_layers=2,
        n_heads=4,
        vocab_size=V,
        cond_dim=32,
        sigma_freq_dim=32,
        max_seq_len=L,
    )
    defaults.update(overrides)
    return SwaFlowModelArgs(**defaults)


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = SphereDiT(tiny_args())
    with torch.no_grad():  # trunc_normal_ assumes no_grad (as in the train loop)
        m.init_weights()
    return m


def rand_inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = sphere_normalize(torch.randn(B, L, D, generator=g))
    sigma = torch.rand(B, generator=g)
    return x, sigma


def test_forward_logits_shape(model):
    x, sigma = rand_inputs()
    logits = model(x, sigma)
    assert logits.shape == (B, L, V)
    assert torch.isfinite(logits).all()


def raw_model():
    """A model without init_weights().

    After adaLN-Zero init the gates are 0 and every block is the identity, so
    the wiring of attention mixing and sigma conditioning is tested on a model
    whose adaLN keeps the default random init.
    """
    torch.manual_seed(0)
    return SphereDiT(tiny_args())


def test_forward_is_bidirectional():
    """Changing a later token changes earlier logits (not causal)."""
    m = raw_model()
    x, sigma = rand_inputs()
    logits = m(x, sigma)
    x2 = x.clone()
    x2[:, -1] = sphere_normalize(torch.randn(B, D))  # change only the last token
    logits2 = m(x2, sigma)
    # if the logits at position 0 change, every position sees the last token
    assert not torch.allclose(logits[:, 0], logits2[:, 0])


def test_sigma_conditions_output():
    """Changing sigma changes the output (adaLN conditioning is active)."""
    m = raw_model()
    x, _ = rand_inputs()
    out1 = m(x, torch.full((B,), 0.1))
    out2 = m(x, torch.full((B,), 0.9))
    assert not torch.allclose(out1, out2)


def test_get_sphere_embeddings_unit_norm(model):
    tokens = torch.randint(0, V, (B, L))
    emb = model.get_sphere_embeddings(tokens)
    assert emb.shape == (B, L, D)
    torch.testing.assert_close(
        emb.norm(dim=-1), torch.ones(B, L), rtol=0, atol=1e-5
    )


def test_lm_head_not_tied(model):
    """lm_head is not tied to the embedding."""
    assert model.output.weight is not model.tok_embeddings.weight
    assert model.output.weight.shape == (V, D)


def test_pretrain_wrapper_forward_and_backward():
    torch.manual_seed(0)
    m = SphereDiTForPretrain(tiny_args())
    with torch.no_grad():
        m.init_weights()
    tokens = torch.randint(0, V, (B, L))
    logits, alpha = m(tokens)
    assert logits.shape == (B, L, V)
    assert alpha.shape == (B,)
    assert (alpha >= 0).all() and (alpha <= 1).all()
    labels = torch.randint(0, V, (B, L))
    loss = swaflow_denoising_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.tok_embeddings.weight.grad is not None
    assert m.output.weight.grad is not None


def test_pretrain_wrapper_truncated_schedule():
    m = SphereDiTForPretrain(tiny_args(schedule_alpha_max=0.131))
    assert isinstance(m.schedule, TruncatedScheduleWrapper)
    assert m.schedule.alpha_max == 0.131
    m2 = SphereDiTForPretrain(tiny_args())
    assert isinstance(m2.schedule, LogLinear)


# ==========================================================================
# backbone_profile (sflm / production)
# ==========================================================================


def parity_args(**overrides) -> SwaFlowModelArgs:
    overrides.setdefault("backbone_profile", PROFILE_SFLM)
    return tiny_args(**overrides)


# ---- both profiles: sigma = -log(1 - alpha) ------------------------------


def test_alpha_to_sigma_matches_formula():
    alpha = torch.tensor([0.0, 0.1, 0.5, 0.9, 0.999])
    expected = -torch.log1p(-alpha.double())
    torch.testing.assert_close(alpha_to_sigma(alpha).double(), expected, atol=1e-6, rtol=0)


def test_alpha_to_sigma_endpoints():
    """alpha=0 (pure noise) -> sigma=0; alpha=1 (clean) -> clamped to a finite value."""
    assert alpha_to_sigma(torch.tensor([0.0])).item() == 0.0
    sigma_clean = alpha_to_sigma(torch.tensor([1.0]))
    assert torch.isfinite(sigma_clean).all()
    assert sigma_clean.item() > 10.0


def test_model_forward_takes_alpha_and_converts():
    """SphereDiT.forward takes alpha and converts it to sigma internally."""
    m = raw_model()
    x, _ = rand_inputs()
    alpha = torch.tensor([0.3, 0.7])
    got = m(x, alpha)
    # matches passing sigma directly to sigma_map
    cond = m.sigma_map(alpha_to_sigma(alpha))
    h = x
    for layer in m.layers.values():
        h = layer(h, cond, m.freqs_cis)
    h = m.norm(h)
    shift, scale = m.final_modulation(torch.nn.functional.silu(cond)).chunk(2, dim=-1)
    expected = m.output(h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1))
    torch.testing.assert_close(got, expected)


# ---- CenteredLayerNorm ---------------------------------------------------


# ---- GELU(tanh) MLP ------------------------------------------------------


# ---- rotate-half RoPE ----------------------------------------------------


# ---- lm_head -------------------------------------------------------------


# ---- sflm init ----------------------------------------------------------


# ---- forward / backward --------------------------------------------------


def test_sflm_profile_forward_and_backward():
    torch.manual_seed(0)
    m = SphereDiTForPretrain(parity_args())
    with torch.no_grad():
        m.init_weights()
    tokens = torch.randint(0, V, (B, L))
    logits, alpha = m(tokens)
    assert logits.shape == (B, L, V)
    assert torch.isfinite(logits).all()
    loss = swaflow_denoising_loss(logits, torch.randint(0, V, (B, L)))
    loss.backward()
    assert m.output.weight.grad is not None
    assert m.layers["0"].feed_forward.w1.weight.grad is not None


# ---- t sampling (antithetic / eps) ----------------------------------------


def test_sample_t_antithetic_is_stratified():
    """Stratified: exactly one u falls in each [i/n, (i+1)/n)."""
    n = 64
    m = SphereDiTForPretrain(tiny_args(antithetic_t_sampling=True))
    torch.manual_seed(0)
    t = m.sample_t(n, torch.device("cpu"))
    strata = (t * n).floor().long()
    assert sorted(strata.tolist()) == list(range(n))


def test_sample_t_eps_bounds_t():
    """t_sampling_eps caps t at (1 - eps).

    With alpha=1 as clean, t=1 is the pure-noise end, so this avoids the noise end.
    """
    eps = 1e-3
    m = SphereDiTForPretrain(tiny_args(t_sampling_eps=eps))
    torch.manual_seed(0)
    t = m.sample_t(4096, torch.device("cpu"))
    assert t.max().item() <= 1.0 - eps
    assert t.min().item() >= 0.0


def test_denoising_loss_alignment():
    """CE is computed with pred[:, i] <-> labels[:, i-1].

    Concentrating the logits at position i on class labels[i-1] gives loss ~ 0.
    """
    pred = torch.full((B, L, V), -20.0)
    labels = torch.randint(0, V, (B, L))
    for b in range(B):
        for i in range(1, L):
            pred[b, i, labels[b, i - 1]] = 20.0
    loss = swaflow_denoising_loss(pred, labels)
    assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# Flavor overrides via [swaflow.model]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QK-norm on by default, and diagnostic logging
# ---------------------------------------------------------------------------


def test_qk_norm_default_is_on():
    """QK-norm is on by default."""
    assert SwaFlowModelArgs().qk_norm is True
    m = SphereDiT(tiny_args())
    for block in m.layers.values():
        assert block.attention.q_norm is not None
        assert block.attention.k_norm is not None


def test_sflm_profile_forces_qk_norm_off():
    """The sflm profile matches the reference implementation's components, so it also drops QK-norm."""
    assert tiny_args(backbone_profile=PROFILE_SFLM).qk_norm is False
    assert tiny_args(backbone_profile=PROFILE_SFLM, qk_norm=True).qk_norm is False


# ---------------------------------------------------------------------------
# Ladder shapes


def test_ladder_flavor_shapes_match_the_paper():
    """Fails if `swaflow_args` drifts from the shape table.

    head_dim is 64 for 150M only and 128 for 500M / 1B / 3B. When reading the
    d dependence of the window, keep in mind that 150M is an outlier in head
    shape. The window is set by the embedding geometry (V and d), so this
    should not matter; comparing the trend over 500M / 1B / 3B with the
    four-point trend confirms it.
    """
    from torchtitan.models.swaflow import swaflow_args

    table = {
        "150M": (768, 12, 12),
        "500M": (1280, 24, 10),
        "1B": (2048, 24, 16),
        "3B": (3072, 28, 24),
    }
    for name, (dim, n_layers, n_heads) in table.items():
        a = swaflow_args[name]
        assert (a.dim, a.n_layers, a.n_heads) == (dim, n_layers, n_heads), name
        assert a.vocab_size == 49152, name  # ln V = 10.8027
        assert 256 <= a.cond_dim <= 512, name
        assert a.backbone_profile == PROFILE_PRODUCTION, name
        assert a.schedule_alpha_max is None, (
            f"{name}: the window is set per run via `[swaflow.model] schedule_alpha_max`, "
            "not in the flavor; baking it into the flavor would rule out untruncated window-measurement runs"
        )
    head_dims = {n: swaflow_args[n].dim // swaflow_args[n].n_heads for n in table}
    assert head_dims == {"150M": 64, "500M": 128, "1B": 128, "3B": 128}
