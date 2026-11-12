# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Geometry on the unit hypersphere S^{d-1}.

The `dim` axis is the embedding dimension; other axes are batch axes.
SLERP is computed in fp64 by default and returned in the input dtype
(`compute_dtype=None` disables this). cosθ is clamped to [−1+eps, 1−eps] so the
antipodal (cosθ -> −1) and parallel (cosθ -> 1, 0/0) limits stay finite.
"""

import torch
from torch import Tensor

__all__ = ["exp_map", "log_map", "slerp", "sphere_normalize"]


def sphere_normalize(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """Normalize to unit norm along `dim`; zero vectors are guarded by eps."""
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def slerp(
    a: Tensor,
    b: Tensor,
    t: Tensor | float,
    dim: int = -1,
    compute_dtype: torch.dtype | None = torch.float64,
    eps: float = 1e-7,
) -> Tensor:
    """Spherical linear interpolation slerp(a, b; t); t=0 gives a, t=1 gives b.

        slerp(a, b; t) = [sin((1−t)θ)·a + sin(tθ)·b] / sin θ,  θ = arccos⟨a, b⟩

    a, b are unit vectors along `dim`. t is a scalar or a Tensor broadcastable in
    keepdim shape (e.g. () or (B, 1) when a is (B, d)). Computed in fp64 by
    default; returns a's dtype.

    The clamp keeps near-antipodal inputs finite and stable, but at exactly
    b = −a the geodesic is undefined and the formula gives the zero vector at
    t=0.5. This has measure zero for continuous inputs, so there is no branch
    (keeps the function torch.compile friendly).
    """
    out_dtype = a.dtype
    if compute_dtype is not None:
        a = a.to(compute_dtype)
        b = b.to(compute_dtype)
    t = torch.as_tensor(t, dtype=a.dtype, device=a.device)

    cos_theta = (a * b).sum(dim=dim, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    out = (torch.sin((1.0 - t) * theta) * a + torch.sin(t * theta) * b) / sin_theta
    return out.to(out_dtype)


def log_map(p: Tensor, q: Tensor, dim: int = -1, eps: float = 1e-7) -> Tensor:
    """Logarithm map log_p(q): the tangent vector at p with norm equal to the geodesic distance θ.

        log_p(q) = θ/sin θ · (q − cos θ · p),  θ = arccos⟨p, q⟩

    At q = −p the direction is undefined; the clamp returns a finite value whose
    direction depends on rounding. As q -> p, log_p(q) -> q − p continuously.
    """
    cos_theta = (p * q).sum(dim=dim, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    return theta / torch.sin(theta) * (q - cos_theta * p)


def exp_map(p: Tensor, v: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """Exponential map exp_p(v): move a distance ||v|| along the geodesic in direction v.

        exp_p(v) = cos θ · p + sin θ/θ · v,  θ = ||v||

    v is assumed tangent at p (⟨p, v⟩ = 0). As θ -> 0, exp_p(v) -> p (sin θ/θ is
    evaluated with an eps-guarded division).
    """
    theta = v.norm(p=2, dim=dim, keepdim=True)
    return torch.cos(theta) * p + torch.sin(theta) / theta.clamp_min(eps) * v
