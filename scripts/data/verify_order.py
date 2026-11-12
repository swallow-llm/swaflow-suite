# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Checks for the data order.

The pure mechanisms (Feistel mapping and inverse, shard format, slot
convention, epoch split) are covered by unit tests. These checks target what
depends on the real data and scale: input schema handling, doc_row recording,
shard layout, the data-parallel world size and global batch, and plans
shorter than one epoch.

Checks
------
``roundtrip`` For the batch of one step, compare

    the tokens yielded by the dataloader
      == plan.resolve -> shard.read_block
      == the matching rows of the input file, re-read and re-tokenized

The third path bypasses the shards, so it checks schema handling, doc_row
recording and the plan mapping end to end.

``shuffle`` (rebuild) Rebuild the plan with the same seed and check that the
digest of the first N slots, plan.json and the held-out files are identical.
The real build is only read: the plan stage runs in a separate directory with
the shards symlinked.

``heldout`` Check over every slot of the plan that no held-out block appears
in training, that all slots are in epoch 0, and that no block is consumed
twice.

Usage::

    python -m scripts.data.verify_order --build <build_dir> \\
        --run <dump_folder>/run_order.json --step 1234 --checks all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.data"

from .indexed_dataset import Shard
from .plan import BuildPlan
from .tokenize_corpus import _load_tokenizer, iter_record_batches  # noqa: PLC2701


def _ok(label: str, passed: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return passed


# ----------------------------------------------------------------------
# round trip of the order log
# ----------------------------------------------------------------------
def _rows_from_input(build_dir: Path, source_name: str, source_file: str,
                     wanted_rows: set[int], text_column: str, id_column: str):
    """Read the text of the given rows of an input file (row number = doc_row).

    Uses the same reader as the build (`iter_record_batches`); a separate
    implementation could miss convention details such as empty lines not
    shifting row numbers.
    """
    inputs = json.loads((build_dir / "inputs.json").read_text())
    entry = next(
        e for e in inputs["sources"][source_name] if e["rel"] == source_file
    )
    cfg = _source_config(build_dir, source_name)
    path = Path(entry["local_path"])
    last = max(wanted_rows)
    out: dict[int, str] = {}
    row = 0
    for texts, _ids, _lb in iter_record_batches(path, cfg, batch_size=2048):
        for t in texts:
            if row in wanted_rows:
                out[row] = t
            row += 1
            if row > last:
                return out, entry
    return out, entry


def _source_config(build_dir: Path, source_name: str):
    from .config import load_config

    cfg = load_config(build_dir / "config.toml")
    return next(s for s in cfg.sources if s.name == source_name)


def check_roundtrip(plan: BuildPlan, run: dict, step: int, n_deep: int) -> bool:
    """Compare the batch of one step along three paths."""
    from torchtitan.models.swaflow.indexed_data import IndexedMixtureDataset

    gb = int(run["global_batch_size"])
    dp_ws = int(run["dp_world_size"])
    local_bs = gb // dp_ws
    seq_len = plan.seq_len
    build_dir = plan.build_dir
    print(
        f"\n=== round trip: step {step} / global_batch {gb} "
        f"(dp_world_size {dp_ws} × local {local_bs}) ==="
    )

    passed = True

    # --- path 1: run the dataloader and collect the tokens it yields -----
    #     For each rank, move sample_index to the step and take local_bs samples.
    loader_tokens: dict[int, np.ndarray] = {}  # slot -> tokens
    for rank in range(dp_ws):
        ds = IndexedMixtureDataset(
            str(build_dir), seq_len, dp_rank=rank, dp_world_size=dp_ws, infinite=False
        )
        ds.load_state_dict(
            {
                "sample_index": step * local_bs,
                "build_name": plan.build_name,
                "plan_checksum": plan.checksum(),
            }
        )
        it = iter(ds)
        for i in range(local_bs):
            batch, _label = next(it)
            slot = (step * local_bs + i) * dp_ws + rank
            loader_tokens[slot] = batch["input"].numpy().astype(np.uint16)

    # --- the slot set must follow the step convention -----------------
    expected_slots = set(range(step * gb, (step + 1) * gb))
    passed &= _ok(
        "slot set of the step = [t·B, (t+1)·B)",
        set(loader_tokens) == expected_slots,
        f"{len(loader_tokens)} slots, difference {len(set(loader_tokens) ^ expected_slots)}",
    )

    # --- path 2: plan.resolve -> shard.read_block ---------------------
    slots = np.array(sorted(loader_tokens), dtype=np.int64)
    r = plan.resolve(slots)
    shards: dict[tuple[int, int], Shard] = {}
    n_mismatch = 0
    for k, slot in enumerate(slots):
        si, sh_i = int(r["source_index"][k]), int(r["shard_index"][k])
        key = (si, sh_i)
        if key not in shards:
            shards[key] = Shard(build_dir / plan.sources[si].shards[sh_i].prefix)
        blk = shards[key].read_block(int(r["block_in_shard"][k]), seq_len)
        if not np.array_equal(blk, loader_tokens[int(slot)]):
            n_mismatch += 1
    passed &= _ok(
        "dataloader tokens == plan.resolve -> shard.read_block",
        n_mismatch == 0,
        f"{len(slots)} slots, mismatches {n_mismatch}",
    )

    # --- path 3: re-read and re-tokenize the rows of the input file ----
    #     Cover every source present once first, then fill up with the rest.
    order: list[int] = []
    picked: set[int] = set()
    seen_src: set[int] = set()
    for k in range(len(slots)):
        si = int(r["source_index"][k])
        if si not in seen_src:
            seen_src.add(si)
            order.append(k)
            picked.add(k)
    order += [k for k in range(len(slots)) if k not in picked]
    order = order[:n_deep]

    tokenizer = _load_tokenizer(plan.tokenizer["path"])
    eos = np.uint16(plan.tokenizer["eos_id"])
    n_deep_bad = 0
    for k in order:
        si, sh_i = int(r["source_index"][k]), int(r["shard_index"][k])
        sp = plan.sources[si]
        shard = shards[(si, sh_i)]
        block = int(r["block_in_shard"][k])
        refs = shard.docs_in_block(block, seq_len)
        src_cfg = _source_config(build_dir, sp.name)
        rows = {d.source_row for d in refs}
        texts, entry = _rows_from_input(
            build_dir, sp.name, refs[0].source_file, rows,
            src_cfg.text_column, src_cfg.id_column,
        )
        # Concatenate documents as the build does: encode(add_special_tokens=False) + EOS
        pieces = []
        for d in refs:
            enc = tokenizer.encode(texts[d.source_row], add_special_tokens=False)
            pieces.append(np.append(np.asarray(enc.ids, dtype=np.uint16), eos))
        stream = np.concatenate(pieces)
        first_doc = refs[0].doc_index
        start = block * seq_len - int(shard.doc_offset[first_doc])
        rebuilt = stream[start : start + seq_len]
        expect = loader_tokens[int(slots[k])]
        good = rebuilt.size == seq_len and np.array_equal(rebuilt, expect)
        n_deep_bad += not good
        print(
            f"    slot {int(slots[k]):>8}  {sp.name:<20} "
            f"{refs[0].source_file}  rows {sorted(rows)}  "
            f"docs {len(refs)}  -> {'match' if good else 'MISMATCH'}"
        )
        print(
            f"        origin={sp.origin}  revision={sp.revision}  "
            f"sha256={entry['sha256'][:16]}…  shard={shard.name}  block={block}"
        )
    passed &= _ok(
        "re-tokenized input rows == dataloader tokens",
        n_deep_bad == 0,
        f"{len(order)} slots, mismatches {n_deep_bad}",
    )
    return passed


# ----------------------------------------------------------------------
# reproducibility of the deterministic shuffle
# ----------------------------------------------------------------------
def check_shuffle(plan: BuildPlan, n_slots: int, workdir: Path) -> bool:
    print(f"\n=== shuffle: rebuild the plan (first {n_slots} slots) ===")
    build_dir = plan.build_dir
    root = workdir / "rebuild"
    new_dir = root / build_dir.name
    if root.exists():
        shutil.rmtree(root)
    new_dir.mkdir(parents=True)
    # Symlink the shards (no re-tokenization); the real build is only read.
    (new_dir / "shards").symlink_to(build_dir / "shards")
    # Copy only inputs.json. The real build's config.toml is passed via --config
    # (tokenize_corpus copies it into build_dir, so a pre-placed copy would make
    # source and destination the same file).
    shutil.copyfile(build_dir / "inputs.json", new_dir / "inputs.json")

    cmd = [
        sys.executable, "-m", "scripts.data.tokenize_corpus",
        "--config", str(build_dir / "config.toml"),
        "--stage", "plan",
        "--root", str(root),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2])
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        return _ok("rebuilding the plan", False, f"rc={proc.returncode}")

    rebuilt = BuildPlan.load(new_dir)
    a, b = plan.checksum(n_slots), rebuilt.checksum(n_slots)
    passed = _ok(
        f"digest of the first {n_slots} slots matches", a == b, f"{a} / {b}"
    )
    old = json.loads((build_dir / "plan.json").read_text())
    new = json.loads((new_dir / "plan.json").read_text())
    passed &= _ok("plan.json identical", old == new)

    # Materialised held-out blocks must match too (checks the block mapping
    # independently of the round trip).
    n_bad = 0
    for f in sorted((build_dir / "heldout").glob("*.npz")):
        h1 = hashlib.blake2b((build_dir / "heldout" / f.name).read_bytes()).hexdigest()
        h2 = hashlib.blake2b((new_dir / "heldout" / f.name).read_bytes()).hexdigest()
        n_bad += h1 != h2
    passed &= _ok("held-out npz byte-identical", n_bad == 0, f"mismatched: {n_bad} files")
    return passed


# ----------------------------------------------------------------------
# held-out blocks never used for training
# ----------------------------------------------------------------------
def check_heldout(plan: BuildPlan, chunk: int = 1 << 20) -> bool:
    print("\n=== held-out: no leakage into training (whole plan) ===")
    held = {si: set(plan.heldout_blocks(si).tolist()) for si in range(len(plan.sources))}
    n_held = sum(len(v) for v in held.values())
    seen: dict[int, set] = {si: set() for si in held}
    max_epoch = 0
    n_leak = 0
    for start in range(0, plan.total_slots, chunk):
        end = min(start + chunk, plan.total_slots)
        r = plan.resolve(np.arange(start, end, dtype=np.int64))
        max_epoch = max(max_epoch, int(r["epoch"].max()))
        for si in held:
            m = r["source_index"] == si
            if not m.any():
                continue
            blocks = r["source_block"][m]
            n_leak += len(held[si] & set(blocks.tolist()))
            seen[si].update(blocks.tolist())

    passed = _ok(
        f"held-out {n_held:,} blocks never appear in training slots",
        n_leak == 0,
        f"leaked {n_leak}",
    )
    passed &= _ok(
        "plan stays within one epoch (all slots in epoch 0)", max_epoch == 0,
        f"max epoch = {max_epoch}",
    )
    # Within one epoch no block is drawn twice, i.e. no duplicates
    n_dup = 0
    for si, sp in enumerate(plan.sources):
        n_dup += sp.n_slots - len(seen[si])
        held_hit = len(held[si] & seen[si])
        print(
            f"    {sp.name:<20} slots {sp.n_slots:>7,}  unique blocks "
            f"{len(seen[si]):>7,}  train blocks {sp.n_train:>7,}  "
            f"held-out {sp.n_holdout:>4,} (leaked {held_hit})"
        )
    passed &= _ok("no block consumed twice", n_dup == 0, f"duplicates {n_dup}")
    return passed


# ----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True)
    ap.add_argument("--run", default="", help="run_order.json (needed for roundtrip)")
    ap.add_argument("--step", type=int, default=0, help="step checked by roundtrip")
    ap.add_argument("--global-batch", type=int, default=0, help="used without --run")
    ap.add_argument("--dp-world-size", type=int, default=0, help="used without --run")
    ap.add_argument("--deep-slots", type=int, default=6,
                    help="number of slots traced back to the input file")
    ap.add_argument("--digest-slots", type=int, default=4096)
    ap.add_argument("--workdir", default="", help="working directory for the shuffle check")
    ap.add_argument("--checks", default="all",
                    help="comma-separated subset of roundtrip,shuffle,heldout (default: all)")
    args = ap.parse_args(argv)

    plan = BuildPlan.load(args.build)
    checks = (
        {"roundtrip", "shuffle", "heldout"}
        if args.checks == "all"
        else set(args.checks.split(","))
    )
    results: dict[str, bool] = {}

    if "roundtrip" in checks:
        if args.run:
            run = json.loads(Path(args.run).read_text())
        else:
            run = {
                "global_batch_size": args.global_batch,
                "dp_world_size": args.dp_world_size,
            }
        if not run.get("global_batch_size"):
            raise SystemExit("roundtrip needs --run or --global-batch/--dp-world-size")
        results["roundtrip"] = check_roundtrip(plan, run, args.step, args.deep_slots)

    if "shuffle" in checks:
        workdir = Path(args.workdir) if args.workdir else plan.build_dir.parent / "_verify"
        results["rebuild"] = check_shuffle(plan, args.digest_slots, workdir)

    if "heldout" in checks:
        results["heldout"] = check_heldout(plan)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
