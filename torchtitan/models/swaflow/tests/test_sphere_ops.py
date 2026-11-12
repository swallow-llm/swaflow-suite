# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torchtitan.models.swaflow.sphere_ops import (
    exp_map,
    log_map,
    slerp,
    sphere_normalize,
)

# fp32 includes arithmetic error; fp64 is close to rounding error only
ATOL = {torch.float32: 1e-4, torch.float64: 1e-9}

B, D = 16, 64


def rand_unit(*shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return sphere_normalize(torch.randn(*shape, generator=g, dtype=dtype))


def rand_tangent(p, scale_max, seed):
    """Random tangent vector at p (norm uniform in (0.05, scale_max))."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn_like(p)
    v = v - (v * p).sum(-1, keepdim=True) * p
    v = sphere_normalize(v)
    norms = 0.05 + (scale_max - 0.05) * torch.rand(
        *p.shape[:-1], 1, generator=g, dtype=p.dtype
    )
    return v * norms


# ---------------------------------------------------------------- normalize


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sphere_normalize_unit_norm(dtype):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, D, generator=g, dtype=dtype) * 10.0
    norms = sphere_normalize(x).norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(B, dtype=dtype))


def test_sphere_normalize_zero_vector_safe():
    out = sphere_normalize(torch.zeros(3, D))
    assert torch.isfinite(out).all()


def test_sphere_normalize_dim():
    x = torch.randn(4, D, 5)
    norms = sphere_normalize(x, dim=1).norm(dim=1)
    torch.testing.assert_close(norms, torch.ones(4, 5))


# ---------------------------------------------------------------- slerp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_slerp_endpoints(dtype):
    a = rand_unit(B, D, dtype=dtype, seed=1)
    b = rand_unit(B, D, dtype=dtype, seed=2)
    torch.testing.assert_close(slerp(a, b, 0.0), a, rtol=0, atol=ATOL[dtype])
    torch.testing.assert_close(slerp(a, b, 1.0), b, rtol=0, atol=ATOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_slerp_norm_preserved(dtype):
    a = rand_unit(B, D, dtype=dtype, seed=3)
    b = rand_unit(B, D, dtype=dtype, seed=4)
    g = torch.Generator().manual_seed(5)
    t = torch.rand(B, 1, generator=g, dtype=dtype)
    norms = slerp(a, b, t).norm(dim=-1)
    torch.testing.assert_close(
        norms, torch.ones(B, dtype=dtype), rtol=0, atol=ATOL[dtype]
    )


def test_slerp_matches_geodesic():
    """slerp(a, b; t) = exp_a(t * log_a(b))."""
    a = rand_unit(B, D, dtype=torch.float64, seed=6)
    b = rand_unit(B, D, dtype=torch.float64, seed=7)
    g = torch.Generator().manual_seed(8)
    t = torch.rand(B, 1, generator=g, dtype=torch.float64)
    expected = exp_map(a, t * log_map(a, b))
    torch.testing.assert_close(slerp(a, b, t), expected, rtol=0, atol=1e-9)


def test_slerp_exact_antipodal_finite():
    """Exact antipodes are degenerate (see docstring); clamping keeps the result finite."""
    a = rand_unit(B, D, dtype=torch.float32, seed=9)
    g = torch.Generator().manual_seed(90)
    t = torch.rand(B, 1, generator=g, dtype=torch.float32)
    assert torch.isfinite(slerp(a, -a, t)).all()
    assert torch.isfinite(slerp(a, -a, 0.5)).all()


@pytest.mark.parametrize(
    "dtype,atol",
    [
        # fp32 inputs have rounding error ~1e-7 against 1+cos(theta) ~ 5e-7, so a
        # norm error of a few percent remains even with fp64 compute (sanity check only)
        (torch.float32, 0.15),
        (torch.float64, 1e-9),
    ],
)
def test_slerp_near_antipodal_unit_norm(dtype, atol):
    """Near antipodes, clamping keeps the result finite and unit-norm (up to input precision)."""
    a = rand_unit(B, D, dtype=dtype, seed=9)
    perturb = 1e-3 * rand_unit(B, D, dtype=dtype, seed=91)
    b = sphere_normalize(-a + perturb)
    g = torch.Generator().manual_seed(92)
    t = torch.rand(B, 1, generator=g, dtype=dtype)
    out = slerp(a, b, t)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.norm(dim=-1), torch.ones(B, dtype=dtype), rtol=0, atol=atol
    )


def test_slerp_identical_endpoints():
    a = rand_unit(B, D, dtype=torch.float32, seed=10)
    torch.testing.assert_close(slerp(a, a, 0.5), a, rtol=0, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slerp_output_dtype_preserved(dtype):
    """With the default fp64 compute, the output keeps the input dtype."""
    a = rand_unit(B, D, dtype=torch.float32, seed=11).to(dtype)
    b = rand_unit(B, D, dtype=torch.float32, seed=12).to(dtype)
    assert slerp(a, b, 0.5).dtype == dtype


def test_slerp_compute_dtype_none_matches_fp64():
    a = rand_unit(B, D, dtype=torch.float32, seed=13)
    b = rand_unit(B, D, dtype=torch.float32, seed=14)
    native = slerp(a, b, 0.3, compute_dtype=None)
    ref = slerp(a, b, 0.3)  # default: fp64 compute
    torch.testing.assert_close(native, ref, rtol=0, atol=1e-5)


# ---------------------------------------------------------------- log/exp map


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_map_norm_is_geodesic_distance(dtype):
    p = rand_unit(B, D, dtype=dtype, seed=15)
    q = rand_unit(B, D, dtype=dtype, seed=16)
    theta = torch.acos((p * q).sum(-1).clamp(-1.0, 1.0))
    torch.testing.assert_close(
        log_map(p, q).norm(dim=-1), theta, rtol=0, atol=ATOL[dtype]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_map_is_tangent(dtype):
    p = rand_unit(B, D, dtype=dtype, seed=17)
    q = rand_unit(B, D, dtype=dtype, seed=18)
    inner = (p * log_map(p, q)).sum(-1)
    torch.testing.assert_close(
        inner, torch.zeros(B, dtype=dtype), rtol=0, atol=ATOL[dtype]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp_log_roundtrip(dtype):
    """exp_p(log_p(q)) = q (round trip)."""
    p = rand_unit(B, D, dtype=dtype, seed=19)
    q = rand_unit(B, D, dtype=dtype, seed=20)
    torch.testing.assert_close(
        exp_map(p, log_map(p, q)), q, rtol=0, atol=ATOL[dtype]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_exp_roundtrip(dtype):
    """log_p(exp_p(v)) = v (for ||v|| < pi)."""
    p = rand_unit(B, D, dtype=dtype, seed=21)
    v = rand_tangent(p, scale_max=3.0, seed=22)
    torch.testing.assert_close(
        log_map(p, exp_map(p, v)), v, rtol=0, atol=ATOL[dtype]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp_map_norm_preserved(dtype):
    p = rand_unit(B, D, dtype=dtype, seed=23)
    v = rand_tangent(p, scale_max=3.0, seed=24)
    torch.testing.assert_close(
        exp_map(p, v).norm(dim=-1),
        torch.ones(B, dtype=dtype),
        rtol=0,
        atol=ATOL[dtype],
    )


def test_exp_map_zero_vector():
    p = rand_unit(B, D, dtype=torch.float32, seed=25)
    torch.testing.assert_close(exp_map(p, torch.zeros_like(p)), p)
