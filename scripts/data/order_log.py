# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Order log: recover the data a checkpoint consumed, without storing text.

The order log holds only document references and indices. Each slot expands to

    (source name, input file path, repository revision, row)

so anyone who downloads the same input files reaches the same documents.
Neither text nor tokens are stored.

It has three parts:

1. `plan.json` (build side): seed, mixture weights, shard layout and
   permutation parameters. Enough to recompute any slot.
2. `run_order.json` (run side): the build, global batch size and slot
   convention of a training run. Written by the dataloader into dump_folder
   at startup.
3. This module: a CLI that expands steps / checkpoints into document
   references from the two files above.

Usage::

    # summary of the range consumed up to checkpoint step 1000
    python -m scripts.data.order_log --build <build_dir> \\
        --run <dump_folder>/run_order.json --checkpoint 1000

    # expand the batch of step 42 down to documents
    python -m scripts.data.order_log --build <build_dir> \\
        --global-batch 64 --step 42 --documents

With the published data order log, which holds the plan and the shard index but
not the tokenized shards, `--index` points to its `index/` folder (with
`shards.parquet`). `--index-files` lists the index files a range needs, so that
only those are downloaded; see `scripts/data/README.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.data"

from .indexed_dataset import Shard
from .plan import BuildPlan, slot_range_for_steps

RUN_ORDER_FILENAME = "run_order.json"


class ShardIndex:
    """The shard index as published: `shards.parquet` plus one `.idx` per shard.

    `shards.parquet` has one row per shard with its `source` and `name` (the
    pair identifies the shard), the `path` of its `.idx` relative to the index
    folder, and the metadata the order log reads (`source_file`, ...). Only the
    `.idx` files of the shards that are actually expanded need to be present.
    """

    def __init__(self, root: str | Path) -> None:
        import pyarrow.parquet as pq

        self.root = Path(root)
        rows = pq.read_table(self.root / "shards.parquet").to_pylist()
        self.rows = {(r["source"], r["name"]): r for r in rows}

    def path(self, source: str, shard_name: str) -> Path:
        return self.root / self.rows[(source, shard_name)]["path"]

    def shard(self, build_dir: Path, source: str, ref) -> Shard:
        return Shard(
            build_dir / ref.prefix,
            meta=self.rows[(source, ref.name)],
            idx_path=self.path(source, ref.name),
        )


def index_files(plan: BuildPlan, slots: np.ndarray, index: ShardIndex, base: str | Path) -> list[str]:
    """Paths (relative to `base`) of the `.idx` files needed to expand `slots`."""
    r = plan.resolve(slots)
    keys = {
        (plan.sources[int(si)].name, plan.sources[int(si)].shards[int(sh)].name)
        for si, sh in zip(r["source_index"], r["shard_index"])
    }
    base = Path(base).resolve()
    return sorted(index.path(*k).resolve().relative_to(base).as_posix() for k in keys)


def write_run_order(
    dump_folder: str | Path,
    build_dir: str | Path,
    plan: BuildPlan,
    global_batch_size: int,
    dp_world_size: int,
    extra: dict | None = None,
) -> Path:
    """Write the run-side order log (called once on rank 0 by the dataloader)."""
    path = Path(dump_folder) / RUN_ORDER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "build_dir": str(Path(build_dir).resolve()),
        "build_name": plan.build_name,
        "seq_len": plan.seq_len,
        "seed": plan.seed,
        "total_slots": plan.total_slots,
        "plan_checksum": plan.checksum(),
        "config_digest": plan.config_digest,
        "global_batch_size": int(global_batch_size),
        "dp_world_size": int(dp_world_size),
        "slot_convention": "slot = sample_index * dp_world_size + dp_rank",
        "step_to_slots": "step t (0-based) consumes slots [t*global_batch, (t+1)*global_batch)",
        **(extra or {}),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def describe_slots(
    plan: BuildPlan, slots: np.ndarray, with_documents: bool, index: ShardIndex | None = None
) -> dict:
    """Slots -> summary of their origin (and optionally a per-document listing).

    With `index`, the document boundaries come from the published shard index
    instead of the build's shard files.
    """
    r = plan.resolve(slots)
    out: dict = {
        "n_slots": int(slots.size),
        "n_tokens": int(slots.size) * plan.seq_len,
        "slot_min": int(slots.min()) if slots.size else None,
        "slot_max": int(slots.max()) if slots.size else None,
        "by_source": {},
    }
    for si, sp in enumerate(plan.sources):
        mask = r["source_index"] == si
        n = int(mask.sum())
        if not n:
            continue
        out["by_source"][sp.name] = {
            "n_slots": n,
            "fraction": n / slots.size,
            "target_fraction": sp.weight,
            "epochs_touched": sorted({int(e) for e in r["epoch"][mask]}),
            "unique_blocks": int(np.unique(r["source_block"][mask]).size),
        }

    if with_documents:
        docs = []
        shard_cache: dict[tuple[int, int], Shard] = {}
        for k in range(slots.size):
            si = int(r["source_index"][k])
            sh_i = int(r["shard_index"][k])
            sp = plan.sources[si]
            key = (si, sh_i)
            if key not in shard_cache:
                ref = sp.shards[sh_i]
                shard_cache[key] = (
                    index.shard(plan.build_dir, sp.name, ref) if index else Shard(plan.build_dir / ref.prefix)
                )
            shard = shard_cache[key]
            refs = shard.docs_in_block(int(r["block_in_shard"][k]), plan.seq_len)
            docs.append(
                {
                    "slot": int(r["slot"][k]),
                    "source": sp.name,
                    "origin": sp.origin,
                    "revision": sp.revision,
                    "shard": sp.shards[sh_i].name,
                    "source_file": sp.shards[sh_i].source_file,
                    "block_in_shard": int(r["block_in_shard"][k]),
                    "epoch": int(r["epoch"][k]),
                    "documents": [
                        {"row": d.source_row, "doc_index": d.doc_index, "n_tokens": d.n_tokens}
                        for d in refs
                    ],
                }
            )
        out["slots"] = docs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True, help="build directory (containing plan.json)")
    ap.add_argument("--run", default="", help="path to run_order.json")
    ap.add_argument("--global-batch", type=int, default=0)
    ap.add_argument("--step", type=int, default=None, help="expand the batch of this step")
    ap.add_argument("--checkpoint", type=int, default=None, help="summarise steps [0, N)")
    ap.add_argument("--documents", action="store_true", help="expand down to documents")
    ap.add_argument("--index", default="", help="folder of the published shard index (with shards.parquet)")
    ap.add_argument(
        "--index-files",
        action="store_true",
        help="print the index files the range needs (relative to the parent of --build) and exit",
    )
    ap.add_argument("--max-slots", type=int, default=4096, help="maximum number of slots to expand")
    ap.add_argument(
        "--max-summary-slots",
        type=int,
        default=10_000_000,
        help="maximum number of slots to summarise (a late checkpoint covers hundreds of millions)",
    )
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    plan = BuildPlan.load(args.build)
    gb = args.global_batch
    if args.run:
        run = json.loads(Path(args.run).read_text())
        gb = gb or run["global_batch_size"]
        if run.get("plan_checksum") and run["plan_checksum"] != plan.checksum():
            raise SystemExit(
                "plan_checksum in run_order.json does not match plan.json; "
                "this is a different build"
            )
    if not gb:
        raise SystemExit("either --global-batch or --run is required")

    if args.step is not None:
        a, b = slot_range_for_steps(args.step, args.step + 1, gb)
    elif args.checkpoint is not None:
        a, b = slot_range_for_steps(0, args.checkpoint, gb)
    else:
        raise SystemExit("specify --step or --checkpoint")

    end = min(b, plan.total_slots)
    limit = args.max_slots if (args.documents or args.index_files) else args.max_summary_slots
    # Always record truncation so a partial view is not read as the full range.
    truncated = (end - a) > limit
    slots = np.arange(a, min(end, a + limit), dtype=np.int64)
    if slots.size == 0:
        raise SystemExit(f"slot range [{a}, {b}) is outside the plan")

    index = ShardIndex(args.index) if args.index else None
    if args.index_files:
        if index is None:
            raise SystemExit("--index-files needs --index")
        print("\n".join(index_files(plan, slots, index, base=Path(args.build).resolve().parent)))
        return 0
    summary = describe_slots(plan, slots, with_documents=args.documents, index=index)
    summary["range"] = {
        "slot_start": int(a),
        "slot_end": int(b),
        "inspected_slots": int(slots.size),
        "truncated": truncated,
    }
    summary["global_batch_size"] = gb
    summary["build"] = plan.build_name

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(text)
        print(f"wrote {args.json_out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
