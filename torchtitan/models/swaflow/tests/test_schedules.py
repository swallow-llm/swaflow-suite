# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torchtitan.models.swaflow.schedules import (
    CosineSquared,
    LogLinear,
    TruncatedScheduleWrapper,
)

BASES = [LogLinear(), CosineSquared()]
IDS = ["loglinear", "cosine_squared"]

T_GRID = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)


# ---------------------------------------------------------------- base schedules


@pytest.mark.parametrize("sched", BASES, ids=IDS)
def test_boundaries(sched):
    """alpha(0) = 1 (clean), alpha(1) = 0 (pure noise). Convention: alpha=1 is clean."""
    assert sched.alpha(torch.tensor(0.0, dtype=torch.float64)) == pytest.approx(1.0)
    assert sched.alpha(torch.tensor(1.0, dtype=torch.float64)) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("sched", BASES, ids=IDS)
def test_monotone_decreasing(sched):
    alphas = sched.alpha(T_GRID)
    assert (alphas.diff() < 0).all(), "alpha(t) must be strictly decreasing"


@pytest.mark.parametrize("sched", BASES, ids=IDS)
def test_alpha_in_unit_interval(sched):
    alphas = sched.alpha(T_GRID)
    assert (alphas >= 0).all() and (alphas <= 1).all()


@pytest.mark.parametrize(
    "sched",
    BASES + [TruncatedScheduleWrapper(b, alpha_max=0.131) for b in BASES],
    ids=IDS + [f"trunc_{i}" for i in IDS],
)
def test_alpha_prime_matches_finite_difference(sched):
    t = torch.linspace(0.01, 0.99, 51, dtype=torch.float64)
    h = 1e-6
    fd = (sched.alpha(t + h) - sched.alpha(t - h)) / (2.0 * h)
    torch.testing.assert_close(sched.alpha_prime(t), fd, rtol=0, atol=1e-6)
    assert (sched.alpha_prime(t) < 0).all(), "alpha_prime must be negative"


@pytest.mark.parametrize("sched", BASES, ids=IDS)
def test_inverse_roundtrip(sched):
    t = T_GRID
    torch.testing.assert_close(sched.alpha_inv(sched.alpha(t)), t, rtol=0, atol=1e-9)
    a = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    torch.testing.assert_close(sched.alpha(sched.alpha_inv(a)), a, rtol=0, atol=1e-9)


def test_loglinear_eps():
    sched = LogLinear(eps=1e-4)
    assert sched.alpha(torch.tensor(1.0, dtype=torch.float64)) == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        LogLinear(eps=1.0)
    with pytest.raises(ValueError):
        LogLinear(eps=-0.1)


def test_float_input_returns_tensor():
    out = LogLinear().alpha(0.5)
    assert isinstance(out, torch.Tensor)
    assert out == pytest.approx(0.5)


# ---------------------------------------------------------------- truncation


def _t_remap_alpha(base, t, alpha_max, alpha_min=0.0):
    """Reference formula for the t-remap scheme (not used; only for comparison tests).

    Maps t affinely onto the base's t interval
    [alpha_inv(alpha_max), alpha_inv(alpha_min)] before applying the base.
    """
    t_clean = base.alpha_inv(alpha_max)
    t_noise = base.alpha_inv(alpha_min)
    return base.alpha(t_clean + (t_noise - t_clean) * t)


def test_alpha_remap_equals_t_remap_for_linear_base():
    """For a linear base (LogLinear), alpha-remap and t-remap coincide."""
    base = LogLinear()
    w = TruncatedScheduleWrapper(base, alpha_max=0.131)
    torch.testing.assert_close(
        w.alpha(T_GRID), _t_remap_alpha(base, T_GRID, 0.131), rtol=0, atol=1e-12
    )


def test_alpha_remap_differs_from_t_remap_for_nonlinear_base():
    """For a nonlinear base the two schemes differ (the alpha density within the window changes).

    Fails if the implementation is accidentally reverted to t-remap.
    """
    base = CosineSquared()
    w = TruncatedScheduleWrapper(base, alpha_max=0.131)
    diff = (w.alpha(T_GRID) - _t_remap_alpha(base, T_GRID, 0.131)).abs()
    assert diff.max() > 0.01, "expected alpha-remap and t-remap to differ"


@pytest.mark.parametrize("base", BASES, ids=IDS)
def test_truncated_boundaries_and_window(base):
    # alpha_max=0.131: the measured window for d=768 (alpha=1 is clean)
    # alpha-remap: alpha(0)=alpha_max exactly, alpha(1) near alpha_min
    # (for bases with alpha(1) not exactly 0, a residual scaled by the factor remains)
    w = TruncatedScheduleWrapper(base, alpha_max=0.131)
    torch.testing.assert_close(
        w.alpha(torch.tensor(0.0, dtype=torch.float64)),
        torch.tensor(0.131, dtype=torch.float64),
        rtol=0,
        atol=1e-9,
    )
    torch.testing.assert_close(
        w.alpha(torch.tensor(1.0, dtype=torch.float64)),
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0,
        atol=1e-9,
    )
    alphas = w.alpha(T_GRID)
    assert (alphas >= 0).all() and (alphas <= 0.131 + 1e-12).all()
    assert (alphas.diff() < 0).all()


@pytest.mark.parametrize("base", BASES, ids=IDS)
def test_truncated_nonzero_alpha_min(base):
    w = TruncatedScheduleWrapper(base, alpha_min=0.05, alpha_max=0.131)
    torch.testing.assert_close(
        w.alpha(torch.tensor(0.0, dtype=torch.float64)),
        torch.tensor(0.131, dtype=torch.float64),
        rtol=0,
        atol=1e-9,
    )
    torch.testing.assert_close(
        w.alpha(torch.tensor(1.0, dtype=torch.float64)),
        torch.tensor(0.05, dtype=torch.float64),
        rtol=0,
        atol=1e-9,
    )
    alphas = w.alpha(T_GRID)
    assert (alphas >= 0.05 - 1e-12).all() and (alphas <= 0.131 + 1e-12).all()


@pytest.mark.parametrize("base", BASES, ids=IDS)
def test_truncated_inverse_roundtrip(base):
    w = TruncatedScheduleWrapper(base, alpha_max=0.131)
    torch.testing.assert_close(
        w.alpha_inv(w.alpha(T_GRID)), T_GRID, rtol=0, atol=1e-8
    )


@pytest.mark.parametrize("base", BASES, ids=IDS)
def test_truncated_full_range_is_identity(base):
    w = TruncatedScheduleWrapper(base, alpha_min=0.0, alpha_max=1.0)
    torch.testing.assert_close(
        w.alpha(T_GRID), base.alpha(T_GRID), rtol=0, atol=1e-12
    )


def test_truncated_validation():
    base = LogLinear()
    with pytest.raises(ValueError):
        TruncatedScheduleWrapper(base, alpha_min=0.2, alpha_max=0.1)
    with pytest.raises(ValueError):
        TruncatedScheduleWrapper(base, alpha_min=0.1, alpha_max=0.1)
    with pytest.raises(ValueError):
        TruncatedScheduleWrapper(base, alpha_max=1.5)
    with pytest.raises(ValueError):
        TruncatedScheduleWrapper(base, alpha_min=-0.1, alpha_max=0.5)


def test_truncated_wrapper_constructs_under_meta_device():
    """Regression: torchtitan builds the model under the meta device.

    If scalar constants pick up the ambient default device and become meta
    tensors, float() in TruncatedScheduleWrapper.__init__ fails with
    "Tensor.item() cannot be called on meta tensors".
    """
    with torch.device("meta"):
        sched = TruncatedScheduleWrapper(LogLinear(), alpha_max=1.0 - 1e-3)
    assert sched.alpha(0.0).item() == pytest.approx(1.0 - 1e-3)
    assert sched.alpha(1.0).item() == pytest.approx(0.0, abs=1e-12)


def test_model_with_alpha_max_constructs_under_meta_device():
    """Same as above, via the model construction path (as in the sflm_small flavor)."""
    from torchtitan.models.swaflow.model import SwaFlowModelArgs, SphereDiTForPretrain

    with torch.device("meta"):
        SphereDiTForPretrain(
            SwaFlowModelArgs(
                dim=32,
                n_layers=1,
                n_heads=2,
                vocab_size=16,
                cond_dim=16,
                sigma_freq_dim=16,
                max_seq_len=8,
                schedule_alpha_max=1.0 - 1e-3,
            )
        )
