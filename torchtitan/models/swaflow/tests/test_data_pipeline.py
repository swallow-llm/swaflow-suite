# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the scripts/data/ preprocessing pipeline.

Dummy local parquet sources run through the same code path as production,
without touching real data. In particular this pins, as a regression test,
that adding a source requires only a config change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.data.config import SourceConfig
from scripts.data.indexed_dataset import Shard, ShardWriter
from scripts.data.permutation import derive_key, FeistelPermutation
from scripts.data.plan import BuildPlan, slot_range_for_steps, slots_for_rank
from scripts.data.tokenize_corpus import main as tokenize_main

TEST_TOKENIZER = "tests/assets/tokenizer"  # small tokenizer, V=2002
SEQ_LEN = 32


# ======================================================================
# permutation
# ======================================================================
@pytest.mark.parametrize("n", [1, 2, 3, 7, 64, 100, 1000, 4096, 65537])
def test_feistel_is_a_bijection_and_invertible(n):
    perm = FeistelPermutation(n, derive_key(42, "t"))
    x = np.arange(n, dtype=np.int64)
    y = perm.forward(x)
    assert sorted(y.tolist()) == list(range(n))
    assert np.array_equal(perm.inverse(y), x)


def test_feistel_is_stable_across_versions():
    """Known-vector regression: if this fails, existing builds change order.

    The permutation array is never stored (see the permutation.py docstring),
    so the implementation must return the same mapping forever. It uses no
    numpy / Python RNG and should be stable; this test guards it.
    """
    perm = FeistelPermutation(1000, derive_key(1234, "global"))
    assert perm.forward(np.arange(8)).tolist() == [
        559, 211, 936, 108, 188, 549, 623, 201
    ]


def test_feistel_shuffles_neighbours_apart():
    """Adjacent slots do not map to adjacent blocks (the order is actually shuffled)."""
    n = 10_000
    perm = FeistelPermutation(n, derive_key(7, "global"))
    y = perm.forward(np.arange(n, dtype=np.int64))
    # a uniform random permutation has E|diff| ~ n/3; the identity gives 1.
    assert np.abs(np.diff(y)).mean() > n / 5


def test_derive_key_separates_purposes():
    assert derive_key(1, "global") != derive_key(1, "holdout:math")
    assert derive_key(1, "global") == derive_key(1, "global")


# ======================================================================
# Indexed dataset format
# ======================================================================
def test_shard_roundtrip(tmp_path):
    prefix = tmp_path / "s0"
    docs = [np.arange(5, dtype=np.uint16), np.arange(7, 20, dtype=np.uint16)]
    with ShardWriter(prefix, vocab_size=2048, store_doc_ids=True) as w:
        for row, d in enumerate(docs):
            w.add_document(d, row=row * 3, doc_id=f"id-{row}")
        meta = w.finalize({"source": "dummy", "source_file": "f.parquet"})

    assert meta["n_docs"] == 2
    assert meta["n_tokens"] == 18

    shard = Shard(prefix)
    assert shard.n_docs == 2
    assert shard.n_tokens == 18
    assert shard.tokens.dtype == np.uint16
    assert np.array_equal(list(shard.iter_documents())[1], docs[1])
    assert shard.doc_row.tolist() == [0, 3]
    assert prefix.with_suffix(".docids.txt").read_text().splitlines() == [
        "0\tid-0",
        "3\tid-1",
    ]


def test_shard_path_appends_and_never_replaces(tmp_path):
    """`Path.with_suffix` would eat `.partial` and point at the same path."""
    from scripts.data.indexed_dataset import shard_path

    assert shard_path(tmp_path / "s0.partial", ".bin").name == "s0.partial.bin"
    assert shard_path(tmp_path / "3__part_0", ".meta.json").name == "3__part_0.meta.json"

    # prefixes containing dots still round-trip
    prefix = tmp_path / "a.b.c"
    with ShardWriter(prefix, vocab_size=64) as w:
        w.add_document(np.arange(4, dtype=np.uint16), row=0)
        w.finalize({"source": "s", "source_file": "f"})
    assert Shard(prefix).n_tokens == 4


def test_shard_rejects_out_of_vocab_tokens(tmp_path):
    with ShardWriter(tmp_path / "s", vocab_size=100) as w:
        with pytest.raises(ValueError, match="vocabulary range"):
            w.add_document(np.array([5, 200], dtype=np.uint16), row=0)


def test_shard_writer_rejects_vocab_too_large_for_uint16(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        ShardWriter(tmp_path / "s", vocab_size=70000)


def test_docs_in_block_maps_back_to_source_rows(tmp_path):
    prefix = tmp_path / "s0"
    with ShardWriter(prefix, vocab_size=2048) as w:
        for row in range(6):
            w.add_document(np.full(10, row + 1, dtype=np.uint16), row=row * 2)
        w.finalize({"source": "dummy", "source_file": "f.parquet"})
    shard = Shard(prefix)
    # block 0 at seq_len 25 = tokens [0, 25) = spans documents 0, 1, 2
    refs = shard.docs_in_block(0, 25)
    assert [r.source_row for r in refs] == [0, 2, 4]
    refs = shard.docs_in_block(1, 25)  # [25, 50) = documents 2, 3, 4
    assert [r.source_row for r in refs] == [4, 6, 8]


# ======================================================================
# config
# ======================================================================
def test_source_rejects_ambiguous_origin():
    with pytest.raises(ValueError, match="exactly one of"):
        SourceConfig(name="x", weight=1.0, files=("a",), repo_id="r", local_dir="/d")


def test_source_glob_expansion_is_sorted_and_capped():
    src = SourceConfig(
        name="x", weight=1.0, files=("d/part_*.parquet",), local_dir="/d", max_files=2
    )
    got = src.match_files(
        ["d/part_002.parquet", "d/part_000.parquet", "d/part_001.parquet", "d/other.txt"]
    )
    assert got == ["d/part_000.parquet", "d/part_001.parquet"]


def test_source_glob_with_no_match_is_an_error():
    src = SourceConfig(name="x", weight=1.0, files=("nope/*",), local_dir="/d")
    with pytest.raises(ValueError, match="no file matching"):
        src.match_files(["d/a.parquet"])


# ======================================================================
# End to end (dummy local parquet sources)
# ======================================================================
def _write_parquet(path: Path, n_docs: int, seed: int, words: int = 40) -> None:
    rng = np.random.default_rng(seed)
    texts, ids = [], []
    for i in range(n_docs):
        n = int(rng.integers(words // 2, words * 2))
        texts.append(" ".join(f"w{int(v)}" for v in rng.integers(0, 500, n)))
        ids.append(f"{path.stem}-{i:05d}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts, "id": ids}), path)


def _config_text(sources: list[tuple[str, float, str]], name: str, seed: int = 99) -> str:
    head = f"""
[build]
name = "{name}"
seq_len = {SEQ_LEN}
seed = {seed}
holdout_fraction = 0.01

[tokenizer]
name = "test"
path = "{TEST_TOKENIZER}"
vocab_size = 2002
eos_id = 0
"""
    blocks = "".join(
        f"""
[[sources]]
name = "{sname}"
weight = {weight}
local_dir = "{local_dir}"
files = "*.parquet"
text_column = "text"
id_column = "id"
store_doc_ids = true
"""
        for sname, weight, local_dir in sources
    )
    return head + blocks


@pytest.fixture(scope="module")
def two_source_build(tmp_path_factory):
    """Builds two local parquet sources weighted 3:1, end to end."""
    base = tmp_path_factory.mktemp("pipeline")
    for base_seed, (src, n_files, n_docs) in enumerate([("alpha", 2, 300), ("beta", 1, 200)]):
        for f in range(n_files):
            _write_parquet(
                base / "raw" / src / f"part_{f}.parquet", n_docs, seed=100 * base_seed + f
            )
    cfg_path = base / "cfg.toml"
    cfg_path.write_text(
        _config_text(
            [("alpha", 3.0, str(base / "raw" / "alpha")), ("beta", 1.0, str(base / "raw" / "beta"))],
            name="dummy_build",
        )
    )
    root = base / "out"
    tokenize_main(["--config", str(cfg_path), "--root", str(root), "--stage", "tokenize"])
    return base, cfg_path, root / "dummy_build"


def test_build_produces_uint16_shards_and_plan(two_source_build):
    _, _, build_dir = two_source_build
    plan = BuildPlan.load(build_dir)
    assert plan.seq_len == SEQ_LEN
    assert [s.name for s in plan.sources] == ["alpha", "beta"]
    assert len(plan.sources[0].shards) == 2
    for sp in plan.sources:
        for ref in sp.shards:
            shard = Shard(build_dir / ref.prefix)
            assert shard.tokens.dtype == np.uint16
            assert shard.n_tokens == ref.n_tokens
            assert ref.n_blocks == ref.n_tokens // SEQ_LEN


def test_mixture_ratio_matches_configured_weights(two_source_build):
    _, _, build_dir = two_source_build
    plan = BuildPlan.load(build_dir)
    r = plan.resolve(np.arange(plan.total_slots))
    counts = np.bincount(r["source_index"], minlength=2)
    assert counts.sum() == plan.total_slots
    # interval assignment + global permutation: error is under one slot (largest-remainder rounding)
    assert abs(counts[0] - 0.75 * plan.total_slots) <= 1
    assert abs(counts[1] - 0.25 * plan.total_slots) <= 1


def test_every_slot_maps_to_a_distinct_block(two_source_build):
    _, _, build_dir = two_source_build
    plan = BuildPlan.load(build_dir)
    r = plan.resolve(np.arange(plan.total_slots))
    keys = list(zip(r["source_index"].tolist(), r["source_block"].tolist()))
    assert len(set(keys)) == plan.total_slots, "duplicates within a single epoch"


def test_heldout_blocks_never_appear_in_training(two_source_build):
    _, _, build_dir = two_source_build
    plan = BuildPlan.load(build_dir)
    r = plan.resolve(np.arange(plan.total_slots))
    for si, sp in enumerate(plan.sources):
        held = set(plan.heldout_blocks(si).tolist())
        assert len(held) == sp.n_holdout > 0
        used = set(r["source_block"][r["source_index"] == si].tolist())
        assert held.isdisjoint(used)


def test_heldout_export_matches_the_plan(two_source_build):
    _, _, build_dir = two_source_build
    plan = BuildPlan.load(build_dir)
    for si, sp in enumerate(plan.sources):
        blob = np.load(build_dir / "heldout" / f"{sp.name}.npz")
        assert blob["tokens"].shape == (sp.n_holdout, SEQ_LEN)
        assert blob["tokens"].dtype == np.uint16
        assert np.array_equal(blob["source_block"], plan.heldout_blocks(si))
        # contents match the corresponding block in the shard
        shard = Shard(build_dir / sp.shards[int(blob["shard_index"][0])].prefix)
        assert np.array_equal(
            blob["tokens"][0], shard.read_block(int(blob["block_in_shard"][0]), SEQ_LEN)
        )


def test_plan_reload_reproduces_the_same_order(two_source_build):
    _, _, build_dir = two_source_build
    a = BuildPlan.load(build_dir)
    b = BuildPlan.load(build_dir)  # with checksum verification
    assert a.checksum() == b.checksum()
    assert np.array_equal(
        a.resolve(np.arange(50))["source_block"], b.resolve(np.arange(50))["source_block"]
    )


def test_plan_checksum_detects_a_changed_order(two_source_build, tmp_path):
    _, _, build_dir = two_source_build
    doc = json.loads((build_dir / "plan.json").read_text())
    doc["seed"] += 1  # change only the seed = the order changes
    tampered = tmp_path / "plan.json"
    tampered.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="checksum does not reproduce"):
        BuildPlan.load(tampered)


def test_shards_are_reused_and_a_new_source_needs_only_config(two_source_build):
    """Adding a source requires only a config change (no code change)."""
    base, cfg_path, build_dir = two_source_build
    for f in range(1):
        _write_parquet(base / "raw" / "gamma" / f"part_{f}.parquet", 150, seed=11)

    shards_dir = build_dir / "shards"
    before = {
        str(p.relative_to(shards_dir)): p.stat().st_mtime_ns
        for p in shards_dir.rglob("*.bin")
    }
    cfg2 = base / "cfg_with_gamma.toml"
    cfg2.write_text(
        _config_text(
            [
                ("alpha", 3.0, str(base / "raw" / "alpha")),
                ("beta", 1.0, str(base / "raw" / "beta")),
                ("gamma", 4.0, str(base / "raw" / "gamma")),
            ],
            name="dummy_build",
        )
    )
    tokenize_main(
        ["--config", str(cfg2), "--root", str(build_dir.parent), "--stage", "tokenize"]
    )

    plan = BuildPlan.load(build_dir)
    assert [s.name for s in plan.sources] == ["alpha", "beta", "gamma"]
    weights = {s.name: s.weight for s in plan.sources}
    assert weights == pytest.approx({"alpha": 0.375, "beta": 0.125, "gamma": 0.5})

    # existing shards are not rebuilt (re-tokenizing is expensive)
    after = {
        str(p.relative_to(shards_dir)): p.stat().st_mtime_ns
        for p in shards_dir.rglob("*.bin")
    }
    for name, mtime in before.items():
        assert after[name] == mtime, f"{name} was rebuilt"

    r = plan.resolve(np.arange(plan.total_slots))
    counts = np.bincount(r["source_index"], minlength=3)
    expected = np.array([0.375, 0.125, 0.5]) * plan.total_slots
    assert np.abs(counts - expected).max() <= 1  # integer-allocation rounding only

    # restore the original 2-source config so other tests are unaffected
    tokenize_main(
        ["--config", str(cfg_path), "--root", str(build_dir.parent), "--stage", "plan"]
    )
    assert [s.name for s in BuildPlan.load(build_dir).sources] == ["alpha", "beta"]


def test_target_tokens_allows_multiple_epochs(two_source_build, tmp_path):
    """Multiple epochs are allowed, each with an independent permutation."""
    _, _, build_dir = two_source_build
    plan = BuildPlan.load(build_dir)
    from scripts.data.plan import build_plan

    big = build_plan(
        build_name="x",
        build_dir=build_dir,
        seq_len=plan.seq_len,
        seed=plan.seed,
        sources=[
            {"name": s.name, "weight": s.weight, "shards": s.shards} for s in plan.sources
        ],
        holdout_fraction=plan.holdout_fraction,
        target_tokens=int(plan.total_tokens * 2.5),
    )
    assert big.sources[0].epochs == pytest.approx(2.5, rel=0.02)
    r = big.resolve(np.arange(big.total_slots))
    for si, sp in enumerate(big.sources):
        used = r["source_block"][r["source_index"] == si]
        counts = np.bincount(used, minlength=sp.n_blocks)
        held = big.heldout_blocks(si)
        assert counts[held].sum() == 0, "held-out leaks with more than one epoch"
        train = np.setdiff1d(np.arange(sp.n_blocks), held)
        # every block appears floor or ceil times
        assert set(np.unique(counts[train]).tolist()) <= {2, 3}


# ======================================================================
# Slot convention (basis of the order log)
# ======================================================================
def test_step_to_slot_range_covers_every_rank_exactly_once():
    world, local_bs, step = 4, 3, 7
    gb = world * local_bs
    a, b = slot_range_for_steps(step, step + 1, gb)
    collected = []
    for rank in range(world):
        j = np.arange(step * local_bs, (step + 1) * local_bs)
        collected.append(slots_for_rank(j, rank, world))
    got = np.sort(np.concatenate(collected))
    assert got.tolist() == list(range(a, b))


def _write_shards(tmp_path, n_shards: int, docs_per_shard: int) -> list:
    """Writes n_shards small shards for the mmap-pool tests."""
    from scripts.data.indexed_dataset import ShardWriter

    prefixes = []
    for i in range(n_shards):
        prefix = tmp_path / f"probe_{i:03d}"
        with ShardWriter(prefix, vocab_size=64) as w:
            for d in range(docs_per_shard):
                w.add_document(np.arange(8, dtype=np.uint16) + (i * 16 + d) % 40,
                               row=d, doc_id=f"s{i}d{d}")
            w.finalize({"source": "probe", "source_file": f"{i}.parquet"})
        prefixes.append(prefix)
    return prefixes


# ======================================================================
# mmap pool. Guards the fix for long runs failing periodically with
# `mmap: Cannot allocate memory`.
# ======================================================================
def test_read_block_survives_pool_eviction(tmp_path):
    """`read_block` returns correct values even after eviction.

    Over its limit the pool `munmap`s old maps. If `read_block` returned a view,
    that would be use-after-unmap, so it must return a copy. With limit 1 every
    read evicts; the values must match those read without a limit.
    """
    from scripts.data.indexed_dataset import MMAP_POOL, Shard

    prefixes = _write_shards(tmp_path, n_shards=4, docs_per_shard=6)
    seq_len = 8
    expect = {}
    prev_max = MMAP_POOL.max_open
    try:
        MMAP_POOL.max_open = 64                       # everything fits
        for p in prefixes:
            s = Shard(p)
            expect[p] = [s.read_block(b, seq_len).copy()
                         for b in range(s.n_tokens // seq_len)]

        MMAP_POOL.clear()
        MMAP_POOL.max_open = 1                        # evict every time
        evictions_before = MMAP_POOL.evictions
        shards = [Shard(p) for p in prefixes]
        for _ in range(3):                            # same values on every pass
            for p, s in zip(prefixes, shards):
                for b, want in enumerate(expect[p]):
                    assert np.array_equal(s.read_block(b, seq_len), want), (p, b)
        assert MMAP_POOL.evictions > evictions_before, "no eviction happened"
    finally:
        MMAP_POOL.max_open = prev_max
        MMAP_POOL.clear()


def test_shard_does_not_load_idx_arrays_until_needed(tmp_path):
    """Document boundaries in `.idx` are not loaded until `docs_in_block` is called.

    The training path (`read_block`) does not use them. Keeping them resident
    costs 85 GB per rank for a 63,911-shard build and adds anonymous mappings
    that push against `vm.max_map_count`.
    """
    from scripts.data.indexed_dataset import Shard

    prefixes = _write_shards(tmp_path, n_shards=1, docs_per_shard=5)
    s = Shard(prefixes[0])
    assert s._idx is None, ".idx arrays loaded at construction"
    s.read_block(0, 8)
    assert s._idx is None, "read_block loaded the .idx arrays"
    s.docs_in_block(0, 8)
    assert s._idx is not None, "docs_in_block did not load them"
