# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Embedding-geometry threshold α* for the truncation window, from trained embeddings.

In high dimension the noise z is nearly orthogonal to the embeddings, so SLERP gives

    x_t ≈ cos(απ/2)·z + sin(απ/2)·e_i        (Θ ≈ π/2)

The correct e_i has inner product ≈ sin(απ/2) with x_t; a competitor e_j has
cos(απ/2)·⟨z, e_j⟩ + sin(απ/2)·⟨e_i, e_j⟩. With m = max_{j≠i} ⟨z, e_j⟩ and
c = max_{j≠i} ⟨e_i, e_j⟩, context-free nearest-neighbour decoding recovers e_i when

    sin(απ/2)·(1 − c) > cos(απ/2)·m   ⇔   tan(απ/2) > m / (1 − c)

so the threshold is

    α* = (2/π)·arctan( m / (1 − c) )

For random unit vectors m ≈ √(2 ln V / d) and c ≈ 0, giving α* ∝ 1/√d. Trained
embeddings are anisotropic, so m and c are measured here. α* is a context-free
bound, not the transition the model reaches: using context, the model succeeds
below α* (compare with the measured α profile).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["embedding_geometry", "trivial_alpha", "iid_max_cosine"]


def trivial_alpha(m: float, c: float = 0.0) -> float:
    """Lowest α at which nearest-neighbour decoding succeeds, α* = (2/π)·arctan(m / (1 − c))."""
    denom = max(1e-12, 1.0 - c)
    return 2.0 / math.pi * math.atan(m / denom)


def iid_max_cosine(vocab_size: int, dim: int) -> float:
    """Reference max cosine √(2 ln V / d) for V isotropic unit vectors.

    Leading term of the Gaussian maximum only (no constant-factor correction);
    a baseline for comparison with measured values.
    """
    return math.sqrt(2.0 * math.log(vocab_size) / dim)


@torch.no_grad()
def embedding_geometry(
    weight: Tensor,
    *,
    n_probe: int = 1024,
    seed: int = 20260808,
    chunk: int = 128,
) -> dict[str, float]:
    """Measure m / c / α* from an embedding matrix [V, d].

    Args:
        weight: raw embedding weights [V, d]; normalized onto the unit sphere
            internally (as in `get_sphere_embeddings`).
        n_probe: number of noise vectors z and of sampled tokens for c.
        seed: seed for z and the token sample (keep it fixed across sizes and steps).
        chunk: rows per [chunk, V] block (avoids materializing n_probe × V at once).

    Returns:
        `m_mean` / `m_p50` / `m_p95`:  distribution of max_j ⟨z, e_j⟩
        `c_mean` / `c_p95`:           distribution of max_{j≠i} ⟨e_i, e_j⟩ (anisotropy)
        `alpha_star`:                 α*(m_mean, c_mean)
        `alpha_star_isotropic`:       α* with c = 0 (no anisotropy term)
        `m_iid`:                      √(2 ln V / d) (isotropic baseline)
        `alpha_star_iid`:             α* from the baseline
        `emb_norm_mean` / `emb_norm_cv`:  norms before normalization (for reference)
    """
    if weight.dim() != 2:
        raise ValueError(f"embedding must have shape [V, d] (got {tuple(weight.shape)})")
    vocab_size, dim = weight.shape
    if n_probe <= 0:
        raise ValueError(f"n_probe must be positive, got {n_probe}")

    device = weight.device
    # Compute in fp32 so the max over V is not taken at reduced precision
    # (bf16 weights give the same values).
    e = weight.float()
    norms = e.norm(dim=-1)
    e = e / norms.clamp_min(1e-12).unsqueeze(-1)

    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n_probe, dim, generator=g, dtype=torch.float32)
    z = (z / z.norm(dim=-1, keepdim=True)).to(device)
    idx = torch.randint(0, vocab_size, (n_probe,), generator=g).to(device)

    m_vals: list[Tensor] = []
    c_vals: list[Tensor] = []
    for lo in range(0, n_probe, chunk):
        hi = min(lo + chunk, n_probe)
        m_vals.append((z[lo:hi] @ e.T).max(dim=1).values)
        sims = e[idx[lo:hi]] @ e.T
        # Exclude the token itself (inner product 1) by token id, so duplicated
        # embedding vectors are still counted as competitors.
        sims.scatter_(1, idx[lo:hi].unsqueeze(1), -1.0)
        c_vals.append(sims.max(dim=1).values)
    m = torch.cat(m_vals).double()
    c = torch.cat(c_vals).double()

    m_mean, c_mean = float(m.mean()), float(c.mean())
    m_iid = iid_max_cosine(vocab_size, dim)
    return {
        "vocab_size": float(vocab_size),
        "dim": float(dim),
        "n_probe": float(n_probe),
        "m_mean": m_mean,
        "m_p50": float(m.median()),
        "m_p95": float(m.quantile(0.95)),
        "c_mean": c_mean,
        "c_p95": float(c.quantile(0.95)),
        "alpha_star": trivial_alpha(m_mean, c_mean),
        "alpha_star_isotropic": trivial_alpha(m_mean, 0.0),
        "m_iid": m_iid,
        "alpha_star_iid": trivial_alpha(m_iid, 0.0),
        "emb_norm_mean": float(norms.double().mean()),
        "emb_norm_cv": float(norms.double().std() / norms.double().mean().clamp_min(1e-12)),
    }
