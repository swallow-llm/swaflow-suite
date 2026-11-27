# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Tests for the indexed dataloader.

Two main properties:

1. kill -> resume produces no duplicated and no missing samples
2. the order log reconstructs the data of any step

Both compare the token sequences actually produced against an independent
recomputation from plan.json + run_order.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.data.indexed_dataset import Shard
from scripts.data.order_log import describe_slots, write_run_order
from scripts.data.plan import BuildPlan, slot_range_for_steps
from scripts.data.tokenize_corpus import main as tokenize_main
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.swaflow.indexed_data import IndexedMixtureDataset

SEQ_LEN = 32
TEST_TOKENIZER = "tests/assets/tokenizer"


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("indexed")
    rng = np.random.default_rng(0)
    for src, n_docs in [("alpha", 400), ("beta", 250)]:
        texts = [
            " ".join(f"w{int(v)}" for v in rng.integers(0, 500, int(rng.integers(20, 80))))
            for _ in range(n_docs)
        ]
        ids = [f"{src}-{i:05d}" for i in range(n_docs)]
        path = base / "raw" / src / "part_0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"text": texts, "id": ids}), path)

    cfg = base / "cfg.toml"
    cfg.write_text(
        f"""
[build]
name = "dl_build"
seq_len = {SEQ_LEN}
seed = 4242
holdout_fraction = 0.01

[tokenizer]
path = "{TEST_TOKENIZER}"
vocab_size = 2002
eos_id = 0

[[sources]]
name = "alpha"
weight = 2.0
local_dir = "{base / "raw" / "alpha"}"
files = "*.parquet"

[[sources]]
name = "beta"
weight = 1.0
local_dir = "{base / "raw" / "beta"}"
files = "*.parquet"
"""
    )
    root = base / "out"
    tokenize_main(["--config", str(cfg), "--root", str(root), "--stage", "tokenize"])
    return root / "dl_build"


def _drain(ds: IndexedMixtureDataset, n: int) -> list[torch.Tensor]:
    out = []
    it = iter(ds)
    for _ in range(n):
        inp, label = next(it)
        assert label[-1].item() == IGNORE_INDEX
        assert torch.equal(label[:-1], inp["input"][1:])
        out.append(inp["input"])
    return out


def _make(build_dir, rank, world, infinite=True) -> IndexedMixtureDataset:
    return IndexedMixtureDataset(
        build_dir=str(build_dir),
        seq_len=SEQ_LEN,
        dp_rank=rank,
        dp_world_size=world,
        infinite=infinite,
    )


# ======================================================================
# Basics
# ======================================================================
def test_rejects_seq_len_mismatch(build_dir):
    with pytest.raises(ValueError, match="seq_len"):
        IndexedMixtureDataset(build_dir=str(build_dir), seq_len=SEQ_LEN * 2)


def test_blocks_match_the_plan(build_dir):
    ds = _make(build_dir, 0, 1)
    plan = BuildPlan.load(build_dir)
    got = _drain(ds, 20)
    resolved = plan.resolve(np.arange(20))
    for k, block in enumerate(got):
        sp = plan.sources[int(resolved["source_index"][k])]
        shard = Shard(build_dir / sp.shards[int(resolved["shard_index"][k])].prefix)
        expected = shard.read_block(int(resolved["block_in_shard"][k]), SEQ_LEN)
        assert np.array_equal(block.numpy(), expected.astype(np.int64))


# ======================================================================
# Resume
# ======================================================================
@pytest.mark.parametrize("world", [1, 2, 4])
def test_resume_has_no_duplication_and_no_omission(build_dir, world):
    """Killing and resuming from the state leaves the produced sequence unchanged."""
    n_before, n_after = 37, 60
    for rank in range(world):
        full = _drain(_make(build_dir, rank, world), n_before + n_after)

        ds = _make(build_dir, rank, world)
        head = _drain(ds, n_before)
        state = ds.state_dict()
        assert state["sample_index"] == n_before

        resumed = _make(build_dir, rank, world)
        resumed.load_state_dict(state)
        tail = _drain(resumed, n_after)

        joined = head + tail
        assert len(joined) == len(full)
        for a, b in zip(joined, full):
            assert torch.equal(a, b)


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_ranks_partition_the_slot_space(build_dir, world):
    """The union over ranks covers every slot exactly once (no gaps, no duplicates)."""
    plan = BuildPlan.load(build_dir)
    n_samples = plan.total_slots // world
    seen: list[int] = []
    for rank in range(world):
        seen.extend((np.arange(n_samples) * world + rank).tolist())
    seen.sort()
    # only the remainder (< world slots) is dropped
    assert seen == list(range(n_samples * world))
    assert plan.total_slots - len(seen) < world


def test_resume_across_the_resolve_chunk_boundary(build_dir):
    """Positions stay correct across the internal resolve-chunk boundary (1024)."""
    from torchtitan.models.swaflow.indexed_data import _RESOLVE_CHUNK

    n = _RESOLVE_CHUNK + 5
    ds = _make(build_dir, 0, 1)
    head = _drain(ds, n)
    state = ds.state_dict()
    resumed = _make(build_dir, 0, 1)
    resumed.load_state_dict(state)
    tail = _drain(resumed, 10)
    full = _drain(_make(build_dir, 0, 1), n + 10)
    for a, b in zip(head + tail, full):
        assert torch.equal(a, b)


def test_state_dict_refuses_a_foreign_build(build_dir):
    ds = _make(build_dir, 0, 1)
    with pytest.raises(ValueError, match="build"):
        ds.load_state_dict({"sample_index": 0, "build_name": "other"})
    with pytest.raises(ValueError, match="does not match"):
        ds.load_state_dict(
            {"sample_index": 0, "build_name": "dl_build", "plan_checksum": "deadbeef"}
        )


def test_finite_mode_stops_at_the_end_of_the_plan(build_dir):
    plan = BuildPlan.load(build_dir)
    ds = _make(build_dir, 0, 1, infinite=False)
    got = sum(1 for _ in ds)
    assert got == plan.total_slots


def test_resume_through_the_parallel_aware_dataloader(build_dir):
    """torchtitan checkpoints the dataloader state, not the dataset state.

    Checks that the position is restored exactly through the StatefulDataLoader
    layer (`ParallelAwareDataloader.state_dict()` is pickled per rank).
    """
    from torchtitan.components.dataloader import ParallelAwareDataloader

    world, local_bs = 2, 4

    def make(rank):
        return ParallelAwareDataloader(
            dataset=_make(build_dir, rank, world),
            dp_rank=rank,
            dp_world_size=world,
            batch_size=local_bs,
        )

    for rank in range(world):
        dl = make(rank)
        it = iter(dl)
        for _ in range(5):
            next(it)
        state = dl.state_dict()

        resumed = make(rank)
        resumed.load_state_dict(state)
        it_resumed = iter(resumed)
        for _ in range(10):
            exp_input, exp_label = next(it)
            got_input, got_label = next(it_resumed)
            assert torch.equal(got_input["input"], exp_input["input"])
            assert torch.equal(got_label, exp_label)


# ======================================================================
# Order log
# ======================================================================
def test_order_log_restores_the_data_of_an_arbitrary_step(build_dir, tmp_path):
    """Reproduces the data seen at checkpoint N from plan + run_order alone."""
    world, local_bs = 2, 3
    gb = world * local_bs
    plan = BuildPlan.load(build_dir)
    run_path = write_run_order(tmp_path, build_dir, plan, gb, world)
    run = json.loads(Path(run_path).read_text())
    assert run["plan_checksum"] == plan.checksum()

    step = 11
    # (a) blocks the dataloader actually produces at step 11
    actual = []
    for rank in range(world):
        ds = _make(build_dir, rank, world)
        blocks = _drain(ds, (step + 1) * local_bs)
        actual.extend(b.numpy().tolist() for b in blocks[step * local_bs :])

    # (b) blocks reconstructed from the order log
    a, b = slot_range_for_steps(step, step + 1, run["global_batch_size"])
    assert (a, b) == (step * gb, (step + 1) * gb)
    info = describe_slots(plan, np.arange(a, b), with_documents=True)
    restored = []
    for entry in info["slots"]:
        sp = next(s for s in plan.sources if s.name == entry["source"])
        ref = next(s for s in sp.shards if s.name == entry["shard"])
        shard = Shard(build_dir / ref.prefix)
        restored.append(shard.read_block(entry["block_in_shard"], SEQ_LEN).tolist())

    assert sorted(actual) == sorted(restored)
    # the log contains no text; references are row numbers and file names only.
    blob = json.dumps(info)
    assert "text" not in blob
    for entry in info["slots"]:
        assert entry["documents"] and all(
            set(d) == {"row", "doc_index", "n_tokens"} for d in entry["documents"]
        )


def test_order_log_documents_point_at_real_source_rows(build_dir):
    plan = BuildPlan.load(build_dir)
    info = describe_slots(plan, np.arange(8), with_documents=True)
    for entry in info["slots"]:
        sp = next(s for s in plan.sources if s.name == entry["source"])
        ref = next(s for s in sp.shards if s.name == entry["shard"])
        shard = Shard(build_dir / ref.prefix)
        rows = [d["row"] for d in entry["documents"]]
        assert rows == sorted(rows)
        assert all(0 <= r < shard.meta["n_rows_read"] for r in rows)
        # documents spanned by a block always total at least seq_len tokens
        assert sum(d["n_tokens"] for d in entry["documents"]) >= SEQ_LEN


# ======================================================================
# Resume with a different number of data-parallel ranks
# ======================================================================
def _loader(build_dir, rank, world, local_bs, midtrain=False):
    from torchtitan.components.dataloader import ParallelAwareDataloader
    from torchtitan.models.swaflow.midtrain_data import NonUniformAlphaDataset

    if midtrain:
        ds = NonUniformAlphaDataset(
            build_dir=str(build_dir),
            seq_len=SEQ_LEN,
            dp_rank=rank,
            dp_world_size=world,
            clean_prefix_prob=0.5,
            k_max=SEQ_LEN // 2,
            k_seed=7,
        )
    else:
        ds = _make(build_dir, rank, world)
    return ParallelAwareDataloader(
        dataset=ds, dp_rank=rank, dp_world_size=world, batch_size=local_bs
    )


def _step_rows(iters, accum):
    """Rows (tokens and clean mask) read by all ranks in one optimizer step."""
    rows = []
    for it in iters:
        for _ in range(accum):
            inp, _label = next(it)
            mask = inp.get("clean_mask")
            for i in range(inp["input"].shape[0]):
                rows.append(
                    (
                        tuple(inp["input"][i].tolist()),
                        None if mask is None else tuple(mask[i].tolist()),
                    )
                )
    return sorted(rows)


def _save_loader_states(path, states, extra=None):
    """Write per-rank dataloader states the way torchtitan's checkpoint does."""
    import torch.distributed.checkpoint as dcp

    merged = {}
    for state in states:
        merged.update(state)
    dcp.save({"dataloader": merged, **(extra or {})}, checkpoint_id=str(path), no_dist=True)


def _checkpointer(dl):
    """A stand-in for CheckpointManager that runs torchtitan's own `dcp_load`."""
    import types

    from torchtitan.components.checkpoint import CheckpointManager

    ckpt = types.SimpleNamespace(states={"dataloader": dl})
    ckpt.dcp_load = types.MethodType(CheckpointManager.dcp_load, ckpt)
    return ckpt


@pytest.mark.parametrize("midtrain", [False, True])
@pytest.mark.parametrize(
    "saved_world, new_world, new_local_bs",
    [(4, 2, 2), (4, 1, 2), (4, 8, 1), (2, 4, 2)],
)
def test_a_new_world_size_reads_the_same_data_in_every_step(
    build_dir, tmp_path, saved_world, new_world, new_local_bs, midtrain
):
    """After resuming with another number of ranks, each step reads the same
    sequences (and clean prefixes) as the original run would have."""
    from torchtitan.models.swaflow.data_reshard import attach_dataloader_resharding

    global_batch, steps_before, steps_after = 8, 5, 6
    saved_local_bs = global_batch // saved_world
    original = [_loader(build_dir, r, saved_world, saved_local_bs, midtrain) for r in range(saved_world)]
    orig_iters = [iter(dl) for dl in original]
    for _ in range(steps_before):
        _step_rows(orig_iters, 1)
    _save_loader_states(tmp_path / "ckpt", [dl.state_dict() for dl in original])
    expected = [_step_rows(orig_iters, 1) for _ in range(steps_after)]

    accum = global_batch // (new_local_bs * new_world)
    resumed = []
    for r in range(new_world):
        dl = _loader(build_dir, r, new_world, new_local_bs, midtrain)
        ckpt = _checkpointer(dl)
        assert attach_dataloader_resharding(
            ckpt, step_getter=lambda: steps_before, gradient_accumulation_steps=accum
        )
        ckpt.dcp_load(
            {"dataloader": dl}, checkpoint_id=str(tmp_path / "ckpt"), from_hf=False, from_quantized=False
        )
        resumed.append(dl)
    new_iters = [iter(dl) for dl in resumed]
    got = [_step_rows(new_iters, accum) for _ in range(steps_after)]
    assert got == expected
    for dl in resumed:
        assert dl.dataset._sample_index == (steps_before + steps_after) * global_batch // new_world


def test_the_same_world_size_takes_the_usual_path(build_dir, tmp_path, monkeypatch):
    from torchtitan.models.swaflow import data_reshard

    world, local_bs = 2, 2
    original = [_loader(build_dir, r, world, local_bs) for r in range(world)]
    iters = [iter(dl) for dl in original]
    for _ in range(3):
        _step_rows(iters, 1)
    _save_loader_states(tmp_path / "ckpt", [dl.state_dict() for dl in original])

    def _fail(*args, **kwargs):
        raise AssertionError("the conversion must not run when the world size is unchanged")

    monkeypatch.setattr(data_reshard, "reshard_loader_state", _fail)
    dl = _loader(build_dir, 1, world, local_bs)
    ckpt = _checkpointer(dl)
    data_reshard.attach_dataloader_resharding(ckpt, step_getter=lambda: 3, gradient_accumulation_steps=1)
    ckpt.dcp_load({"dataloader": dl}, checkpoint_id=str(tmp_path / "ckpt"), from_hf=False, from_quantized=False)
    next(iter(dl))
    assert dl.dataset._sample_index == 3 * local_bs + local_bs


def test_the_saved_world_size_comes_from_the_checkpoint(build_dir, tmp_path):
    from torchtitan.models.swaflow.data_reshard import saved_dp_world_size

    states = [_loader(build_dir, r, 3, 1).state_dict() for r in range(3)]
    _save_loader_states(tmp_path / "with", states)
    assert saved_dp_world_size(str(tmp_path / "with")) == 3

    import torch.distributed.checkpoint as dcp

    dcp.save({"train_state": {"step": 1}}, checkpoint_id=str(tmp_path / "without"), no_dist=True)
    assert saved_dp_world_size(str(tmp_path / "without")) is None


@pytest.mark.parametrize("saved, new", [(256, 16), (256, 32), (128, 16), (8, 3), (2, 8), (128, 256)])
def test_source_ranks_check_every_saved_state(saved, new):
    from torchtitan.models.swaflow.data_reshard import source_ranks

    read = [source_ranks(r, new, saved) for r in range(new)]
    assert all(0 in ranks and all(0 <= s < saved for s in ranks) for ranks in read)
    if new <= saved:
        assert sorted({s for ranks in read for s in ranks}) == list(range(saved))


def _loader_state(sample_index, local_bs=16):
    return {
        "_sampler_iter_state": {"samples_yielded": sample_index},
        "_sampler_iter_yielded": sample_index // local_bs,
        "_num_yielded": sample_index // local_bs,
        "dataset_state": {"sample_index": sample_index, "build_name": "b", "plan_checksum": "c"},
        "_iterator_finished": False,
    }


def test_reshard_converts_the_position_and_the_counters():
    from torchtitan.models.swaflow.data_reshard import reshard_loader_state

    # the 150M state at step 238,400: 256 ranks x 16 sequences per step
    saved = {r: _loader_state(238_400 * 16) for r in (0, 16, 32)}
    out = reshard_loader_state(saved, 256, 16)
    assert out["dataset_state"]["sample_index"] == 238_400 * 4096 // 16
    assert out["dataset_state"]["build_name"] == "b"
    assert out["_sampler_iter_state"]["samples_yielded"] == 238_400 * 256
    assert out["_num_yielded"] == out["_sampler_iter_yielded"] == 238_400 * 16
    # the input is not modified
    assert saved[0]["dataset_state"]["sample_index"] == 238_400 * 16


def test_reshard_refuses_inconsistent_or_indivisible_states():
    from torchtitan.models.swaflow.data_reshard import reshard_loader_state

    with pytest.raises(ValueError, match="different data positions"):
        reshard_loader_state({0: _loader_state(32), 5: _loader_state(48)}, 8, 4)
    with pytest.raises(ValueError, match="split evenly"):
        reshard_loader_state({0: _loader_state(3)}, 8, 5)
    with pytest.raises(ValueError, match="rank 0"):
        reshard_loader_state({1: _loader_state(32)}, 8, 4)


def test_the_order_log_records_the_global_batch_with_accumulation():
    import types

    from torchtitan.models.swaflow.indexed_data import global_batch_size

    cfg = types.SimpleNamespace(training=types.SimpleNamespace(global_batch_size=4096, local_batch_size=16))
    assert global_batch_size(cfg, 16) == 4096
    cfg.training.global_batch_size = -1
    assert global_batch_size(cfg, 16) == 256


def _publish_index(build_dir: Path, root: Path) -> Path:
    """Lay the shard index out as on the Hub: shards.parquet plus sub-foldered .idx files."""
    import shutil

    plan = BuildPlan.load(build_dir)
    index = root / "production_6t" / "index"
    rows = []
    for sp in plan.sources:
        for i, ref in enumerate(sorted(sp.shards, key=lambda r: r.name)):
            rel = f"{sp.name}/dir/{i // 2}/{ref.name}.idx"
            (index / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(build_dir / f"{ref.prefix}.idx", index / rel)
            meta = json.loads((build_dir / f"{ref.prefix}.meta.json").read_text())
            rows.append({"name": ref.name, "path": rel, "source": meta["source"],
                         "source_file": meta["source_file"], "n_docs": meta["n_docs"]})
    pq.write_table(pa.Table.from_pylist(rows), index / "shards.parquet")
    shutil.copyfile(build_dir / "plan.json", root / "production_6t" / "plan.json")
    return index


def test_order_log_reads_the_published_shard_index(build_dir, tmp_path):
    """Documents expanded from the published index equal those from the build's shards."""
    from scripts.data.order_log import ShardIndex, describe_slots

    plan = BuildPlan.load(build_dir)
    index = ShardIndex(_publish_index(build_dir, tmp_path))
    slots = np.arange(0, plan.total_slots, max(1, plan.total_slots // 50))
    got = describe_slots(plan, slots, True, index=index)
    want = describe_slots(plan, slots, True)
    for g, w in zip(got.pop("slots"), want.pop("slots")):
        assert g == w, {k: (g.get(k), w.get(k)) for k in set(g) | set(w) if g.get(k) != w.get(k)}
    assert got == want


def test_order_log_lists_only_the_index_files_a_step_needs(build_dir, tmp_path, capsys):
    from scripts.data.order_log import main as order_log_main

    _publish_index(build_dir, tmp_path)
    build = tmp_path / "production_6t"
    args = ["--build", str(build), "--global-batch", "6", "--step", "3",
            "--index", str(build / "index")]
    assert order_log_main(args + ["--index-files"]) == 0
    files = capsys.readouterr().out.split()
    plan = BuildPlan.load(build_dir)
    r = plan.resolve(np.arange(18, 24))
    assert len(files) == len({(int(a), int(b)) for a, b in zip(r["source_index"], r["shard_index"])})
    assert all(f.startswith("production_6t/index/") and (tmp_path / f).is_file() for f in files)

    # with only those files present, the step still expands to documents
    for p in (build / "index").rglob("*.idx"):
        if p.relative_to(tmp_path).as_posix() not in files:
            p.unlink()
    assert order_log_main(args + ["--documents"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["slots"]) == 6 and all(s["documents"] for s in out["slots"])
