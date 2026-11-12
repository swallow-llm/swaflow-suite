# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""The training order: mixture, deterministic shuffle and held-out split.

Training consumes slots 0, 1, 2, ... in order. `BuildPlan` is a pure function
that maps a slot to (source, shard, block within the shard). No permutation
array is stored (see `permutation.py`).

    slot --global permutation--> p
      p --mixture interval--> (source s, local slot l)
      l --epoch split--> (epoch e, position i within the epoch)
      i --epoch permutation--> within
      within --inverse held-out permutation--> block (held-out excluded)
      block --shard boundaries--> (shard, block within the shard)

Held-out: each source has its own permutation h. The first `n_holdout` images
of h are held out and the rest are for training. Feistel inverses are O(1), so
the k-th training block is h^-1(n_holdout + k), found in constant time without
storing the excluded set. Held-out blocks never reach training in any epoch.

With more than one epoch, each epoch gets an independent permutation, so
repeats of a block are not clustered; every block appears
floor/ceil(n_slots / n_train) times.

Mixture: source s occupies a contiguous interval [offset_s, offset_s + n_s) of
the slot space, which the global permutation spreads uniformly. The run-level
ratio is exactly w_s and local ratios are hypergeometric (no drift).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .permutation import FEISTEL_ROUNDS, FeistelPermutation, derive_key

__all__ = [
    "BuildPlan",
    "ShardRef",
    "SourcePlan",
    "slot_range_for_steps",
    "slots_for_rank",
]

PLAN_FILENAME = "plan.json"
PLAN_FORMAT_VERSION = 1


# ----------------------------------------------------------------------
# rank / step <-> slot mapping (convention shared with the dataloader)
# ----------------------------------------------------------------------
def slots_for_rank(sample_index, dp_rank: int, dp_world_size: int):
    """Rank-local sample index j -> global slot index.

    Round-robin convention: `slot = j * world + rank`. The union over all ranks
    and all j covers every slot exactly once, so resuming only needs j (one
    integer per rank) and yields no duplicates or gaps.
    """
    return np.asarray(sample_index, dtype=np.int64) * dp_world_size + dp_rank


def slot_range_for_steps(
    step_start: int, step_end: int, global_batch_size: int
) -> tuple[int, int]:
    """Slot range [a, b) consumed by steps [step_start, step_end).

    Steps are 0-based; `global_batch_size = local_batch_size * dp_world_size`.
    The data seen by checkpoint N is `slot_range_for_steps(0, N, gb)`.
    """
    return step_start * global_batch_size, step_end * global_batch_size


# ----------------------------------------------------------------------
# plan components
# ----------------------------------------------------------------------
@dataclass
class ShardRef:
    name: str
    #: Path relative to the build directory (prefix without extension)
    prefix: str
    source_file: str
    n_docs: int
    n_tokens: int
    n_blocks: int


@dataclass
class SourcePlan:
    name: str
    weight: float
    shards: list[ShardRef]
    n_blocks: int
    n_holdout: int
    n_train: int
    slot_offset: int
    n_slots: int
    origin: str = ""
    revision: str = ""
    block_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))

    @property
    def epochs(self) -> float:
        return self.n_slots / self.n_train if self.n_train else 0.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "origin": self.origin,
            "revision": self.revision,
            "n_blocks": self.n_blocks,
            "n_holdout": self.n_holdout,
            "n_train": self.n_train,
            "slot_offset": self.slot_offset,
            "n_slots": self.n_slots,
            "epochs": round(self.epochs, 6),
            "shards": [
                {
                    "name": s.name,
                    "prefix": s.prefix,
                    "source_file": s.source_file,
                    "n_docs": s.n_docs,
                    "n_tokens": s.n_tokens,
                    "n_blocks": s.n_blocks,
                }
                for s in self.shards
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> "SourcePlan":
        shards = [ShardRef(**s) for s in d["shards"]]
        plan = cls(
            name=d["name"],
            weight=d["weight"],
            shards=shards,
            n_blocks=d["n_blocks"],
            n_holdout=d["n_holdout"],
            n_train=d["n_train"],
            slot_offset=d["slot_offset"],
            n_slots=d["n_slots"],
            origin=d.get("origin", ""),
            revision=d.get("revision", ""),
        )
        plan.block_offsets = _cumulative([s.n_blocks for s in shards])
        return plan


def _cumulative(counts) -> np.ndarray:
    out = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(np.asarray(counts, dtype=np.int64), out=out[1:])
    return out


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """Integer allocation that sums exactly to `total` (largest remainder; deterministic)."""
    exact = [w * total for w in weights]
    floors = [int(np.floor(e)) for e in exact]
    remainder = total - sum(floors)
    order = sorted(
        range(len(weights)), key=lambda i: (-(exact[i] - floors[i]), i)
    )
    for i in order[:remainder]:
        floors[i] += 1
    return floors


# ----------------------------------------------------------------------
# BuildPlan
# ----------------------------------------------------------------------
class BuildPlan:
    def __init__(
        self,
        build_name: str,
        build_dir: str | Path,
        seq_len: int,
        seed: int,
        sources: list[SourcePlan],
        total_slots: int,
        config_digest: str = "",
        tokenizer: dict | None = None,
        holdout_fraction: float = 0.0,
    ) -> None:
        self.build_name = build_name
        self.build_dir = Path(build_dir)
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.sources = sources
        self.total_slots = int(total_slots)
        self.config_digest = config_digest
        self.tokenizer = tokenizer or {}
        self.holdout_fraction = holdout_fraction

        self._slot_offsets = np.array(
            [s.slot_offset for s in sources] + [total_slots], dtype=np.int64
        )
        self._global_perm = FeistelPermutation(
            total_slots, derive_key(self.seed, "global")
        )
        self._holdout_perms: dict[int, FeistelPermutation] = {}
        self._epoch_perms: dict[tuple[int, int], FeistelPermutation] = {}

    # -- permutations -------------------------------------------------
    def _holdout_perm(self, si: int) -> FeistelPermutation:
        if si not in self._holdout_perms:
            sp = self.sources[si]
            self._holdout_perms[si] = FeistelPermutation(
                sp.n_blocks, derive_key(self.seed, f"holdout:{sp.name}")
            )
        return self._holdout_perms[si]

    def _epoch_perm(self, si: int, epoch: int) -> FeistelPermutation:
        key = (si, epoch)
        if key not in self._epoch_perms:
            sp = self.sources[si]
            self._epoch_perms[key] = FeistelPermutation(
                sp.n_train, derive_key(self.seed, f"epoch:{sp.name}:{epoch}")
            )
        return self._epoch_perms[key]

    # -- resolution ---------------------------------------------------
    def resolve(self, slots) -> dict[str, np.ndarray]:
        """Slot indices -> location of each training sample.

        Every returned array has the input's length: `source_index` /
        `shard_index` / `block_in_shard` / `source_block` (block index within
        the source) / `epoch`.
        """
        slots = np.atleast_1d(np.asarray(slots, dtype=np.int64))
        if slots.size and (slots.min() < 0 or slots.max() >= self.total_slots):
            raise IndexError(
                f"slot index outside [0, {self.total_slots}) "
                f"(min {slots.min()}, max {slots.max()})"
            )
        permuted = np.asarray(self._global_perm.forward(slots), dtype=np.int64)
        src_index = (
            np.searchsorted(self._slot_offsets, permuted, side="right") - 1
        ).astype(np.int64)

        shard_index = np.empty_like(permuted)
        block_in_shard = np.empty_like(permuted)
        source_block = np.empty_like(permuted)
        epoch_out = np.empty_like(permuted)

        for si, sp in enumerate(self.sources):
            mask = src_index == si
            if not mask.any():
                continue
            local = permuted[mask] - sp.slot_offset
            epoch = local // sp.n_train
            within = local % sp.n_train
            for e in np.unique(epoch):
                sel = epoch == e
                within[sel] = self._epoch_perm(si, int(e)).forward(within[sel])
            block = np.asarray(
                self._holdout_perm(si).inverse(sp.n_holdout + within), dtype=np.int64
            )
            shard = (
                np.searchsorted(sp.block_offsets, block, side="right") - 1
            ).astype(np.int64)
            shard_index[mask] = shard
            block_in_shard[mask] = block - sp.block_offsets[shard]
            source_block[mask] = block
            epoch_out[mask] = epoch

        return {
            "slot": slots,
            "source_index": src_index,
            "shard_index": shard_index,
            "block_in_shard": block_in_shard,
            "source_block": source_block,
            "epoch": epoch_out,
        }

    def heldout_blocks(self, si: int) -> np.ndarray:
        """Held-out blocks of source si (block indices within the source)."""
        sp = self.sources[si]
        if sp.n_holdout == 0:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(
            self._holdout_perm(si).inverse(np.arange(sp.n_holdout, dtype=np.int64)),
            dtype=np.int64,
        )

    def locate_block(self, si: int, source_block) -> tuple[np.ndarray, np.ndarray]:
        """Block index within the source -> (shard_index, block_in_shard)."""
        sp = self.sources[si]
        b = np.atleast_1d(np.asarray(source_block, dtype=np.int64))
        shard = (np.searchsorted(sp.block_offsets, b, side="right") - 1).astype(np.int64)
        return shard, b - sp.block_offsets[shard]

    # -- statistics ---------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.total_slots * self.seq_len

    def summary(self) -> str:
        lines = [
            f"build      : {self.build_name}",
            f"seq_len    : {self.seq_len}",
            f"seed       : {self.seed}",
            f"total      : {self.total_slots:,} slots = "
            f"{self.total_tokens / 1e9:.3f}B tokens",
        ]
        for sp in self.sources:
            lines.append(
                f"  {sp.name:<10} w={sp.weight:6.4f} "
                f"slots={sp.n_slots:,} ({sp.n_slots / max(self.total_slots, 1):6.2%}) "
                f"blocks={sp.n_blocks:,} holdout={sp.n_holdout:,} "
                f"epochs={sp.epochs:.3f} shards={len(sp.shards)}"
            )
        return "\n".join(lines)

    # -- serialization ------------------------------------------------
    def to_json(self) -> dict:
        return {
            "format_version": PLAN_FORMAT_VERSION,
            "build_name": self.build_name,
            "seq_len": self.seq_len,
            "seed": self.seed,
            "total_slots": self.total_slots,
            "total_tokens": self.total_tokens,
            "config_digest": self.config_digest,
            "tokenizer": self.tokenizer,
            "holdout_fraction": self.holdout_fraction,
            "permutation": {
                "kind": "feistel-splitmix64",
                "rounds": FEISTEL_ROUNDS,
                "slot_convention": "slot = sample_index * dp_world_size + dp_rank",
            },
            "sources": [s.to_json() for s in self.sources],
            "checksum": self.checksum(),
        }

    def checksum(self, n: int = 4096) -> str:
        """Digest of the resolution of the first n slots.

        A fingerprint that reloading plan.json reproduces the same order,
        without storing the permutation. Cached, because the dataloader's
        state_dict calls it at every checkpoint.
        """
        import hashlib

        n = min(n, self.total_slots)
        if n == 0:
            return ""
        cached = getattr(self, "_checksum_cache", None)
        if cached is not None and cached[0] == n:
            return cached[1]
        r = self.resolve(np.arange(n, dtype=np.int64))
        blob = np.stack(
            [r["source_index"], r["shard_index"], r["block_in_shard"]]
        ).tobytes()
        digest = hashlib.blake2b(blob, digest_size=16).hexdigest()
        self._checksum_cache = (n, digest)
        return digest

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.build_dir / PLAN_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, build_dir: str | Path) -> "BuildPlan":
        build_dir = Path(build_dir)
        path = build_dir if build_dir.is_file() else build_dir / PLAN_FILENAME
        d = json.loads(path.read_text())
        if d["format_version"] != PLAN_FORMAT_VERSION:
            raise ValueError(
                f"{path}: format_version {d['format_version']} is not supported"
            )
        plan = cls(
            build_name=d["build_name"],
            build_dir=path.parent,
            seq_len=d["seq_len"],
            seed=d["seed"],
            sources=[SourcePlan.from_json(s) for s in d["sources"]],
            total_slots=d["total_slots"],
            config_digest=d.get("config_digest", ""),
            tokenizer=d.get("tokenizer", {}),
            holdout_fraction=d.get("holdout_fraction", 0.0),
        )
        expected = d.get("checksum", "")
        if expected and plan.checksum() != expected:
            raise ValueError(
                f"{path}: the order checksum does not reproduce "
                f"(stored {expected} / recomputed {plan.checksum()}). "
                "The permutation implementation or its parameters have changed"
            )
        return plan


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
def build_plan(
    build_name: str,
    build_dir: str | Path,
    seq_len: int,
    seed: int,
    sources: list[dict],
    holdout_fraction: float,
    target_tokens: int = 0,
    config_digest: str = "",
    tokenizer: dict | None = None,
) -> BuildPlan:
    """Build a `BuildPlan` from the list of tokenized shards.

    Each element of `sources` is
    `{"name", "weight", "origin", "revision", "shards": [ShardRef, ...]}`.
    """
    total_weight = sum(s["weight"] for s in sources)
    prepared: list[SourcePlan] = []
    for s in sources:
        shards = list(s["shards"])
        n_blocks = sum(sh.n_blocks for sh in shards)
        n_holdout = int(np.ceil(n_blocks * holdout_fraction)) if holdout_fraction else 0
        n_train = n_blocks - n_holdout
        if n_train <= 0:
            raise ValueError(
                f"source '{s['name']}': no training blocks left "
                f"(blocks {n_blocks}, holdout {n_holdout}); the data is too small"
            )
        sp = SourcePlan(
            name=s["name"],
            weight=s["weight"] / total_weight,
            shards=shards,
            n_blocks=n_blocks,
            n_holdout=n_holdout,
            n_train=n_train,
            slot_offset=0,
            n_slots=0,
            origin=s.get("origin", ""),
            revision=s.get("revision", ""),
        )
        sp.block_offsets = _cumulative([sh.n_blocks for sh in shards])
        prepared.append(sp)

    weights = [sp.weight for sp in prepared]
    if target_tokens > 0:
        total_slots = target_tokens // seq_len
    else:
        # Longest run that keeps the mixture weights with no source beyond one epoch
        total_slots = int(min(sp.n_train / w for sp, w in zip(prepared, weights)))
    if total_slots <= 0:
        raise ValueError("total_slots is 0 (not enough data, or misconfigured weights)")

    counts = _largest_remainder(weights, total_slots)
    offset = 0
    for sp, n in zip(prepared, counts):
        sp.slot_offset = offset
        sp.n_slots = n
        offset += n
    assert offset == total_slots

    return BuildPlan(
        build_name=build_name,
        build_dir=build_dir,
        seq_len=seq_len,
        seed=seed,
        sources=prepared,
        total_slots=total_slots,
        config_digest=config_digest,
        tokenizer=tokenizer,
        holdout_fraction=holdout_fraction,
    )
