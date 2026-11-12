# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-DP-rank random streams (`torchtitan/models/swaflow/rng.py`).

Three properties:

1. independent across ranks: different DP ranks draw different t / sphere noise
   (prevents a regression where all ranks drew the same values)
2. reproducible: the same (seed, rank) gives the same draws, independent of
   execution order and wall-clock time
3. nothing else changes: rank 0 and single-process runs are bit-identical to
   the behavior before per-rank seeding

`rank_rng_seed` is a pure function so the tests need no distributed setup; they
loop over ranks and compare what the model draws after seeding with it.
"""

import pytest
import torch

from torchtitan.models.swaflow.model import SwaFlowModelArgs, SphereDiTForPretrain
from torchtitan.models.swaflow.rng import rank_rng_seed

BASE_SEED = 20260807
WORLD = 8
LOCAL_BATCH = 16


def tiny_model(**overrides) -> SphereDiTForPretrain:
    args = SwaFlowModelArgs(
        dim=32,
        n_layers=1,
        n_heads=2,
        vocab_size=64,
        cond_dim=16,
        sigma_freq_dim=16,
        max_seq_len=8,
        schedule_alpha_max=0.10,
        **overrides,
    )
    return SphereDiTForPretrain(args)


def draws_for_rank(rank: int, *, base_seed: int = BASE_SEED, antithetic: bool = False):
    """Reproduces what training draws after seeding with the rank's seed.

    The training path (`SphereDiTForPretrain.forward`) draws only two things:
    `torch.rand` in `sample_t` and `torch.randn_like` in SLERP.
    """
    torch.manual_seed(rank_rng_seed(base_seed, rank))
    model = tiny_model(antithetic_t_sampling=antithetic)
    t = model.sample_t(LOCAL_BATCH, torch.device("cpu"))
    noise = torch.randn(LOCAL_BATCH, 8, 32)
    return t, noise


# ---------------------------------------------------------------------------
# 1. Seed derivation
# ---------------------------------------------------------------------------
def test_seed_is_distinct_per_rank():
    seeds = [rank_rng_seed(BASE_SEED, r) for r in range(64)]
    assert len(set(seeds)) == 64


def test_seed_is_pure_function():
    assert rank_rng_seed(BASE_SEED, 3) == rank_rng_seed(BASE_SEED, 3)
    # a different base gives a different stream (varying the seed yields independent replicas)
    assert rank_rng_seed(BASE_SEED, 3) != rank_rng_seed(BASE_SEED + 1, 3)


def test_seed_fits_in_64bit():
    for rank in (0, 1, 7, 63, 255, 4095):
        seed = rank_rng_seed(BASE_SEED, rank)
        assert 0 <= seed < 2**64
        torch.manual_seed(seed)  # within the range torch accepts


def test_rank0_keeps_base_seed():
    """Rank 0 passes the base seed through, so single-process runs (reference dumps / sampler) are unchanged."""
    assert rank_rng_seed(BASE_SEED, 0) == BASE_SEED


def test_negative_rank_rejected():
    with pytest.raises(ValueError):
        rank_rng_seed(BASE_SEED, -1)


# ---------------------------------------------------------------------------
# 2. Actual draws are independent across ranks (regression guard)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("antithetic", [False, True])
def test_t_and_noise_are_independent_across_ranks(antithetic):
    ts, noises = zip(
        *(draws_for_rank(r, antithetic=antithetic) for r in range(WORLD))
    )
    for r in range(1, WORLD):
        assert not torch.equal(ts[0], ts[r]), f"rank {r} has the same t as rank 0"
        assert not torch.equal(noises[0], noises[r]), f"rank {r} has the same noise as rank 0"
    # across all ranks the number of distinct alphas equals the global batch.
    # (Without per-rank seeds there were only LOCAL_BATCH, duplicated WORLD times.)
    assert torch.cat(ts).unique().numel() == WORLD * LOCAL_BATCH


def test_broken_behaviour_is_what_the_fix_removes():
    """Control: without reseeding, every rank draws the same values.

    If this fails, upstream `set_determinism` may have changed behavior; this
    guards the premise of the fix.
    """
    same = []
    for _ in range(WORLD):
        torch.manual_seed(BASE_SEED)  # the state set_determinism gives every rank
        same.append(tiny_model().sample_t(LOCAL_BATCH, torch.device("cpu")))
    assert all(torch.equal(same[0], x) for x in same[1:])


# ---------------------------------------------------------------------------
# 3. Reproducibility (same seed and world size give the same result)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rank", [0, 1, 7])
def test_draws_are_reproducible(rank):
    t1, n1 = draws_for_rank(rank)
    t2, n2 = draws_for_rank(rank)
    assert torch.equal(t1, t2)
    assert torch.equal(n1, n2)


def test_draws_are_independent_of_evaluation_order():
    """Seeding ranks in a different order (e.g. on resume) gives the same values.

    The seed is a pure function that depends on neither the step nor the execution
    order, so a run killed and resumed (going through `__init__` again) returns to
    the same stream.
    """
    forward = [draws_for_rank(r)[0] for r in range(WORLD)]
    backward = {r: draws_for_rank(r)[0] for r in reversed(range(WORLD))}
    for r in range(WORLD):
        assert torch.equal(forward[r], backward[r])


def test_antithetic_still_stratifies_within_rank():
    """Antithetic sampling still stratifies within each rank.

    With a shared grid across ranks, stratification was global; with per-rank
    seeds each rank gets an independent grid. The granularity is coarser, but
    each rank's 16 points still fall one per interval of [0,1) split into 16
    (local stratification is kept).
    """
    for rank in range(WORLD):
        torch.manual_seed(rank_rng_seed(BASE_SEED, rank))
        model = tiny_model(antithetic_t_sampling=True, t_sampling_eps=0.0)
        t = model.sample_t(LOCAL_BATCH, torch.device("cpu"))
        bins = (t * LOCAL_BATCH).floor().long().sort().values
        assert torch.equal(bins, torch.arange(LOCAL_BATCH))


# --------------------------------------------------------------------------
# Save the random stream position with the checkpoint
# --------------------------------------------------------------------------
def test_rng_state_round_trips_the_stream_position(tmp_path) -> None:
    """The seed alone is not enough: without restoring the position, draws after resume differ."""
    from torchtitan.models.swaflow.rng import (
        restore_rank_rng_state,
        save_rank_rng_state,
    )

    torch.manual_seed(20260809)
    torch.rand(128)                       # stands in for having advanced 200 steps
    save_rank_rng_state(str(tmp_path), step=200, rank=3)
    expected = torch.rand(8).tolist()     # what would come next without interruption

    torch.manual_seed(20260809)           # on resume: only the seed has been reset
    assert torch.rand(8).tolist() != expected, "without restoring the position the draws differ"

    torch.manual_seed(20260809)
    assert restore_rank_rng_state(str(tmp_path), step=200, rank=3)
    assert torch.rand(8).tolist() == expected


def test_restore_without_a_saved_state_is_loud_but_not_fatal(tmp_path) -> None:
    from torchtitan.models.swaflow.rng import restore_rank_rng_state

    assert not restore_rank_rng_state(str(tmp_path), step=999, rank=0)


def test_rng_states_are_per_rank(tmp_path) -> None:
    """One file per rank; the point is that no collective is used."""
    from torchtitan.models.swaflow.rng import (
        restore_rank_rng_state,
        save_rank_rng_state,
    )

    torch.manual_seed(1)
    save_rank_rng_state(str(tmp_path), step=10, rank=0)
    torch.manual_seed(2)
    save_rank_rng_state(str(tmp_path), step=10, rank=1)

    torch.manual_seed(1)
    zero = torch.rand(4).tolist()
    restore_rank_rng_state(str(tmp_path), step=10, rank=0)
    assert torch.rand(4).tolist() == zero
    restore_rank_rng_state(str(tmp_path), step=10, rank=1)
    assert torch.rand(4).tolist() != zero


def test_prune_keeps_the_recent_steps(tmp_path) -> None:
    from torchtitan.models.swaflow.rng import prune_rng_states, save_rank_rng_state

    for step in range(1, 26):
        save_rank_rng_state(str(tmp_path), step=step, rank=0)
    prune_rng_states(str(tmp_path), keep=20)
    kept = sorted(
        int(p.name[5:]) for p in (tmp_path / "rngstate").iterdir()
    )
    assert kept == list(range(6, 26))
