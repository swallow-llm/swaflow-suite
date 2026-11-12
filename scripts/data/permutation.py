# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shuffle: a pseudo-random permutation with no stored array
(Feistel network + cycle walking).

2T tokens at seq_len 1024 is ~2e9 samples, so an explicit int64 permutation
would take 16 GB to store and as much RAM to generate. The order log only has
to recover which data a step consumed, so the permutation is a pure function
of the seed, evaluated in O(1) for any slot.

Balanced Feistel network, x = (L, R) -> 6 rounds -> y, with a splitmix64 round
function. The domain is a power of two (2^(2h) >= n); for other n, cycle
walking (re-apply while out of range) gives a bijection on [0, n). The domain
is < 4n, so the expected number of walks is below 4.

The inverse is also O(1) (rounds in reverse); `plan.py` uses it to carve out
the held-out set.

Everything is uint64 ring arithmetic in numpy, with no version-dependent RNG,
so the order is stable across numpy / Python versions (pinned by a
known-answer test in `torchtitan/models/swaflow/tests/test_data_pipeline.py`).
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["FEISTEL_ROUNDS", "FeistelPermutation", "derive_key"]

#: Number of rounds. Changing it changes the order of existing builds
#: (detected by the plan.json checksum).
FEISTEL_ROUNDS = 6
_ROUNDS = FEISTEL_ROUNDS

# splitmix64 constants (64-bit ring arithmetic)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer on uint64 arrays."""
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> _S30)) * _MIX1
        z = (z ^ (z >> _S27)) * _MIX2
        return z ^ (z >> _S31)


def derive_key(seed: int, label: str) -> int:
    """Derive a 64-bit key from (seed, label).

    Uses blake2b because Python's `hash()` depends on PYTHONHASHSEED. Distinct
    labels (global / holdout / epoch) give independent permutations from one seed.
    """
    # The salt string is fixed: it determines the released data order.
    digest = hashlib.blake2b(
        f"titan-flowlm/perm/{seed}/{label}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


class FeistelPermutation:
    """Deterministic pseudo-random permutation of [0, n); forward and inverse are O(1)."""

    def __init__(self, n: int, key: int, rounds: int = _ROUNDS) -> None:
        if n < 0:
            raise ValueError(f"n must be non-negative (got {n})")
        self.n = int(n)
        self.key = int(key) & 0xFFFFFFFFFFFFFFFF
        self.rounds = int(rounds)

        bits = max(1, (self.n - 1).bit_length()) if self.n > 1 else 1
        self.half_bits = (bits + 1) // 2
        if self.half_bits > 31:
            # Beyond 2^62 (~4.6e18) samples the domain no longer fits in uint64.
            raise ValueError(f"n is too large (got {n}, limit 2^62)")
        self._mask = np.uint64((1 << self.half_bits) - 1)
        self._shift = np.uint64(self.half_bits)

        # Round keys: a splitmix64 chain started from key
        keys = []
        cur = np.uint64(self.key)
        for _ in range(self.rounds):
            cur = _splitmix64(cur)
            keys.append(cur)
        self._round_keys = keys

    # ------------------------------------------------------------------
    # Bijection on [0, 2^(2h)) (no cycle walking)
    # ------------------------------------------------------------------
    def _block_forward(self, x: np.ndarray) -> np.ndarray:
        left = x >> self._shift
        right = x & self._mask
        for rk in self._round_keys:
            left, right = right, left ^ (_splitmix64(right ^ rk) & self._mask)
        return (left << self._shift) | right

    def _block_inverse(self, y: np.ndarray) -> np.ndarray:
        left = y >> self._shift
        right = y & self._mask
        for rk in reversed(self._round_keys):
            left, right = right ^ (_splitmix64(left ^ rk) & self._mask), left
        return (left << self._shift) | right

    # ------------------------------------------------------------------
    # Bijection on [0, n) (with cycle walking)
    # ------------------------------------------------------------------
    def _walk(self, x: np.ndarray, step) -> np.ndarray:
        n = np.uint64(self.n)
        y = step(x)
        # Expected iterations < 4 (domain < 4n); the cap guards against an infinite loop.
        for _ in range(1024):
            out_of_range = y >= n
            if not out_of_range.any():
                return y
            y = np.where(out_of_range, step(y), y)
        raise RuntimeError("cycle walking did not converge (implementation bug)")

    def forward(self, x: np.ndarray | int) -> np.ndarray | int:
        """Slot index -> permuted index."""
        scalar = np.isscalar(x)
        arr = np.atleast_1d(np.asarray(x, dtype=np.uint64))
        if self.n <= 1:
            out = arr
        else:
            out = self._walk(arr, self._block_forward)
        out = out.astype(np.int64)
        return int(out[0]) if scalar else out

    def inverse(self, y: np.ndarray | int) -> np.ndarray | int:
        """Permuted index -> slot index (inverse of `forward`)."""
        scalar = np.isscalar(y)
        arr = np.atleast_1d(np.asarray(y, dtype=np.uint64))
        if self.n <= 1:
            out = arr
        else:
            out = self._walk(arr, self._block_inverse)
        out = out.astype(np.int64)
        return int(out[0]) if scalar else out
