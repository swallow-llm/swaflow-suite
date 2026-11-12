# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Tests for the embedding-geometry probe.

The 1/sqrt(d) extrapolation assumes the window's upper edge is set by where
context-free nearest-neighbor lookup starts to solve the task. Pinned here: the
formula itself, and that isotropic embeddings reproduce its theoretical value.
Behavior on trained embeddings is measured in actual runs.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchtitan.models.swaflow.embed_geometry import (
    embedding_geometry,
    iid_max_cosine,
    trivial_alpha,
)


def _isotropic(vocab: int, dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(vocab, dim, generator=g)
    return w / w.norm(dim=-1, keepdim=True)


def test_trivial_alpha_matches_closed_form():
    # inverse of tan(alpha* * pi/2) = m / (1 - c)
    for m, c in ((0.1, 0.0), (0.2, 0.1), (0.05, 0.3)):
        a = trivial_alpha(m, c)
        assert math.tan(a * math.pi / 2) == pytest.approx(m / (1 - c), rel=1e-9)


def test_trivial_alpha_is_monotone_in_m():
    assert trivial_alpha(0.05) < trivial_alpha(0.10) < trivial_alpha(0.20)


def test_iid_max_cosine_scales_as_inv_sqrt_d():
    """The baseline sqrt(2 ln V / d) halves exactly when d is quadrupled."""
    assert iid_max_cosine(49152, 3072) == pytest.approx(
        iid_max_cosine(49152, 768) / 2, rel=1e-12
    )


def test_isotropic_embedding_reproduces_the_iid_baseline():
    """For isotropic embeddings the measured m is close to the baseline sqrt(2 ln V / d).

    Only the leading term of the Gaussian maximum is used, so they do not match
    exactly (there is a constant-factor correction); only the order of magnitude
    is pinned. An order-of-magnitude miss suggests a probe normalization or
    dimension mix-up.
    """
    w = _isotropic(4096, 256)
    g = embedding_geometry(w, n_probe=256, seed=7)
    assert g["m_mean"] == pytest.approx(g["m_iid"], rel=0.25)
    assert g["c_mean"] == pytest.approx(g["m_iid"], rel=0.25)


def test_measured_alpha_star_follows_inv_sqrt_d_for_isotropic_embeddings():
    """For isotropic embeddings alpha* roughly halves when d is quadrupled (the extrapolation's form).

    Whether this holds for trained embeddings is the question the window
    measurement answers, so the isotropic case is pinned first.
    """
    a_small = embedding_geometry(_isotropic(4096, 256, seed=1), n_probe=256)[
        "alpha_star_isotropic"
    ]
    a_large = embedding_geometry(_isotropic(4096, 1024, seed=2), n_probe=256)[
        "alpha_star_isotropic"
    ]
    assert a_large / a_small == pytest.approx(0.5, rel=0.12)


def test_anisotropy_raises_alpha_star():
    """Embeddings concentrated in one direction (larger c) raise the trivialization threshold."""
    w = _isotropic(2048, 128, seed=3)
    w_aniso = w.clone()
    bias = torch.zeros(128)
    bias[0] = 1.0
    w_aniso = w_aniso + 1.5 * bias  # shift the whole vocabulary toward one direction
    g_iso = embedding_geometry(w, n_probe=256, seed=5)
    g_ani = embedding_geometry(w_aniso, n_probe=256, seed=5)
    assert g_ani["c_mean"] > g_iso["c_mean"]
    assert g_ani["alpha_star"] > g_iso["alpha_star"]


def test_probe_is_deterministic_and_scale_invariant():
    """Same seed gives the same values; scaling the embeddings changes nothing
    (consistent with `get_sphere_embeddings` normalizing onto the unit sphere)."""
    w = _isotropic(1024, 64, seed=11)
    a = embedding_geometry(w, n_probe=128, seed=99)
    b = embedding_geometry(w * 3.7, n_probe=128, seed=99)
    for k in ("m_mean", "c_mean", "alpha_star"):
        assert a[k] == pytest.approx(b[k], rel=1e-5), k
    assert b["emb_norm_mean"] == pytest.approx(3.7, rel=1e-5)


def test_chunking_does_not_change_the_result():
    w = _isotropic(1024, 64, seed=13)
    a = embedding_geometry(w, n_probe=128, seed=4, chunk=128)
    b = embedding_geometry(w, n_probe=128, seed=4, chunk=7)
    for k in ("m_mean", "m_p95", "c_mean", "alpha_star"):
        assert a[k] == pytest.approx(b[k], rel=1e-6), k


def test_self_similarity_is_excluded():
    """c is max over j != i, so the self inner product of 1 must not be picked up."""
    g = embedding_geometry(_isotropic(512, 32, seed=17), n_probe=64, seed=1)
    assert g["c_p95"] < 0.99


def test_rejects_non_matrix():
    with pytest.raises(ValueError, match=r"\[V, d\]"):
        embedding_geometry(torch.randn(8), n_probe=4)
