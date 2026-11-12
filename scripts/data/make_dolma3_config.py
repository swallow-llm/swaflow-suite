# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Generate Dolma 3 Mix build configs. This module defines the selection rule;
the TOML files are its output.

Repository layout:

    data/<partition>/<file>.jsonl.zst

There are 163 partitions across 6 sources: common_crawl is split by topic and
crawl period (124), olmocr by topic (21), stack_edu by language (15), and the
other three sources have one partition each. Partitions within a source differ
in content, so for a slice (the default):

1. Across sources, tokens are allocated by the token shares on the dataset
   card (76.07 / 13.57 / 6.89 / 2.56 / 0.86 / 0.04).
2. Within a source, the first file of each partition is taken and read from
   the start up to a per-file cap at a document boundary
   (`max_tokens_per_file`); the source's quota is split equally across its
   partitions. A file holds ~20-1500M tokens, so a ~1B-token slice cannot
   cover the partitions with whole files.
3. No random selection. Raising `--target-tokens` only appends the following
   rows of the same files; nothing is re-fetched.

Equal per-partition quotas under-weight large partitions; this is accepted for
a small validation slice.

`--all-files` takes every file in the repository without truncation (the full
build). Partition proportions then follow the raw mix, and the weights across
sources are enforced by the plan's slot allocation.

Usage::

    python -m scripts.data.make_dolma3_config \
        --target-tokens 1_000_000_000 \
        --out scripts/data/configs/dolma3_1b_val.toml
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

REPO_ID = "allenai/dolma3_mix-6T"

#: Token shares from the dataset card (dolma3_mix-6T-1025-7B), used unchanged.
#: partition prefix -> (source name, mix %)
SOURCES: list[tuple[str, str, float]] = [
    ("common_crawl-", "common_crawl", 76.07),
    ("olmocr_science_pdfs-", "olmocr_science_pdfs", 13.57),
    ("stack_edu-", "stack_edu", 6.89),
    ("finemath-3plus", "finemath_3plus", 2.56),
    ("rpj-proofpile-arxiv", "rpj_proofpile_arxiv", 0.86),
    ("dolma1_7-wiki-en", "dolma1_7_wiki_en", 0.04),
]

#: The tokenizer location is site-specific; set `SWAFLOW_TOKENIZER_DIR` to
#: override it. `load_config` reads the same variable, so generated configs
#: work unchanged on another machine.
TOKENIZER = {
    "name": "HuggingFaceTB/SmolLM2-135M",
    "path": os.environ.get(
        "SWAFLOW_TOKENIZER_DIR",
        "./assets/hf/SmolLM2-135M",
    ),
    "vocab_size": 49152,
    "eos_id": 0,
}


def source_of(partition: str) -> tuple[str, float]:
    for prefix, name, share in SOURCES:
        if partition.startswith(prefix):
            return name, share
    raise ValueError(f"partition belongs to no known source: {partition}")


def list_repo(revision: str) -> tuple[str, list[str]]:
    """(commit sha, relative paths of the files under data/)."""
    from huggingface_hub import HfApi

    api = HfApi()
    sha = api.dataset_info(REPO_ID, revision=revision).sha
    paths = [
        f.path
        for f in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
        if f.path.startswith("data/") and f.path.endswith(".jsonl.zst")
    ]
    return sha, sorted(paths)


def select_all(paths: list[str]) -> list[dict]:
    """Take every file (the full Dolma 3 Mix build).

    Unlike `select()`, there is no per-partition selection and no truncation
    (`max_tokens_per_file = 0`): the rule is "the repository as is", so
    partition proportions follow the raw mix.

    The global permutation spreads even the first slots of the plan over every
    shard, so a prefix of the plan cannot be served by a partial download.
    With every file present, runs of different lengths share one plan and one
    order.

    The weights across sources (`weight`, the card's token shares) are enforced
    by the plan's slot allocation, so file counts need no adjustment.
    """
    partitions: dict[str, list[str]] = collections.OrderedDict()
    for p in paths:
        partitions.setdefault(p.split("/")[1], []).append(p)

    by_source: dict[str, dict] = collections.OrderedDict()
    for _, name, share in SOURCES:
        by_source[name] = {"name": name, "weight": share, "files": [], "n_partitions": 0}
    for part, files in partitions.items():
        name, _ = source_of(part)
        by_source[name]["files"].extend(files)
        by_source[name]["n_partitions"] += 1
    for s in by_source.values():
        if not s["files"]:
            raise ValueError(f"source '{s['name']}': no files selected")
        s["max_tokens_per_file"] = 0   # no truncation: read whole files
        s["quota_tokens"] = 0          # the total is set by the plan's target_tokens
    return list(by_source.values())


def select(paths: list[str], target_tokens: int, files_per_partition: int = 1) -> list[dict]:
    """Take the first N files of each partition; split each source's quota equally across them."""
    partitions: dict[str, list[str]] = collections.OrderedDict()
    for p in paths:
        partitions.setdefault(p.split("/")[1], []).append(p)

    by_source: dict[str, dict] = collections.OrderedDict()
    for _, name, share in SOURCES:
        by_source[name] = {"name": name, "weight": share, "files": [], "n_partitions": 0}
    for part, files in partitions.items():
        name, _ = source_of(part)
        by_source[name]["files"].extend(files[:files_per_partition])
        by_source[name]["n_partitions"] += 1

    total_share = sum(s["weight"] for s in by_source.values())
    for s in by_source.values():
        if not s["files"]:
            raise ValueError(f"source '{s['name']}': no files selected")
        quota = target_tokens * s["weight"] / total_share
        s["quota_tokens"] = int(round(quota))
        s["max_tokens_per_file"] = int(round(quota / len(s["files"])))
    return list(by_source.values())


def render(sources: list[dict], sha: str, revision: str, args) -> str:
    total = sum(s["quota_tokens"] for s in sources)
    all_files = getattr(args, "all_files", False)
    n_files = sum(len(s["files"]) for s in sources)
    if all_files:
        head = [
            "# SwaFlow: full Dolma 3 Mix build (generated; do not edit by hand)",
            "#",
            "# Command: python -m scripts.data.make_dolma3_config --all-files \\",
            f"#            --plan-target-tokens {args.plan_target_tokens} \\",
            f"#            --revision {revision} --out <this file>",
            "#",
            "# Selection rule: every file in the repository, untruncated.",
            "# The global permutation spreads even the first slots of the plan",
            "# over all shards, so a prefix of the plan cannot be served by a",
            "# partial download.",
            "#",
            "# Mixture weights across sources (weight = token share on the dataset",
            "# card) are enforced by the plan's slot allocation, not by file counts.",
            "#",
            f"# Revision pinned to commit sha: {sha}",
            f"# {n_files:,} files in total",
            "",
            "[build]",
            f'name = "{args.name}"',
            f'description = "{args.description}"',
            f"seq_len = {args.seq_len}",
            f"seed = {args.seed}",
            f"holdout_fraction = {args.holdout_fraction}",
            "# Token budget of the plan. Changing it changes the order (the blocks",
            "# behind the first slots are not preserved), so runs that stop",
            "# earlier keep this value and end through their own number of",
            "# training steps.",
            f"target_tokens = {args.plan_target_tokens}",
            "# Guard against processing this large config by accident:",
            "# tokenize_corpus refuses it unless --allow-design-only is",
            "# given explicitly.",
            "design_only = true",
            "",
        ]
    else:
        head = [
            "# SwaFlow: Dolma 3 Mix validation slice (generated; do not edit by hand)",
            "#",
            "# Command: python -m scripts.data.make_dolma3_config \\",
            f"#            --target-tokens {args.target_tokens} \\",
            f"#            --revision {revision} --out <this file>",
            "#",
            "# Selection rule (see make_dolma3_config.py): the first file of each",
            "# of the 163 partitions; tokens split across sources by the dataset",
            "# card's token shares and equally across partitions within a source,",
            "# truncated at document boundaries (max_tokens_per_file). No random selection.",
            "#",
            f"# Revision pinned to commit sha: {sha}",
            f"# Total quota: {total / 1e9:.3f}B tokens (before held-out)",
            "",
            "[build]",
            f'name = "{args.name}"',
            f'description = "{args.description}"',
            f"seq_len = {args.seq_len}",
            f"seed = {args.seed}",
            f"holdout_fraction = {args.holdout_fraction}",
            "# 0 = the longest run that keeps the mixture with every source within",
            "# one epoch; the total is already set by max_tokens_per_file.",
            "target_tokens = 0",
            "",
        ]
    out = [
        *head,
        "[tokenizer]",
        f'name = "{TOKENIZER["name"]}"',
        f'path = "{TOKENIZER["path"]}"',
        f"vocab_size = {TOKENIZER['vocab_size']}",
        f"eos_id = {TOKENIZER['eos_id']}",
    ]
    for s in sources:
        if all_files:
            title = (
                f"# --- {s['name']}: mix {s['weight']}%, "
                f"{s['n_partitions']} partitions / {len(s['files']):,} files (all) ---"
            )
            notes = (
                f'notes = "ODC-By 1.0. {s["n_partitions"]} partitions, all '
                f'{len(s["files"])} files in full. '
                f'sa_remove_ranges not applied."'
            )
        else:
            title = (
                f"# --- {s['name']}: mix {s['weight']}% = "
                f"{s['quota_tokens'] / 1e6:.1f}M tokens, {s['n_partitions']} partitions ---"
            )
            notes = (
                f'notes = "ODC-By 1.0. {s["n_partitions"]} partitions x first file. '
                f'sa_remove_ranges not applied."'
            )
        out += [
            "",
            title,
            "[[sources]]",
            f'name = "{s["name"]}"',
            f"weight = {s['weight']}",
            f'repo_id = "{REPO_ID}"',
            f'revision = "{sha}"',
            f"max_tokens_per_file = {s['max_tokens_per_file']}",
            'text_column = "text"',
            'id_column = "id"',
            notes,
            "files = [",
        ]
        out += [f'    "{f}",' for f in s["files"]]
        out += ["]"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--files-per-partition", type=int, default=1)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--name", default="dolma3_1b_val")
    ap.add_argument(
        "--description", default="Dolma 3 Mix 1B-token slice"
    )
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--holdout-fraction", type=float, default=0.001)
    ap.add_argument(
        "--all-files", action="store_true",
        help="take every file in the repository without truncation (full build)",
    )
    ap.add_argument(
        "--plan-target-tokens", type=int, default=6_000_000_000_000,
        help="plan token budget (target_tokens) for --all-files; changing it changes the order",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    sha, paths = list_repo(args.revision)
    print(f"{REPO_ID}@{sha}: {len(paths)} files", file=sys.stderr)
    if args.all_files:
        sources = select_all(paths)
    else:
        sources = select(paths, args.target_tokens, args.files_per_partition)
    for s in sources:
        print(
            f"  {s['name']:22s} {s['weight']:6.2f}%  {s['n_partitions']:4d} partitions"
            f"  {len(s['files']):5d} files  cap {s['max_tokens_per_file'] / 1e6:8.3f}M tok/file"
            f"  quota {s['quota_tokens'] / 1e6:8.1f}M",
            file=sys.stderr,
        )
    Path(args.out).write_text(render(sources, sha, args.revision, args))
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
