# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""TOML build config: data sources with mixture weights.

Adding a source needs no code change. Every source is described the same way
(a set of input files plus text / id column names), with no dataset-specific
branches in code.

Example::

    [build]
    name = "dolma3_1b_val"
    seq_len = 2048
    seed = 20260806
    holdout_fraction = 0.001

    [tokenizer]
    path = "/path/to/smollm2-135m-tokenizer"
    vocab_size = 49152

    [[sources]]
    name = "common_crawl"
    weight = 76.07                      # relative; normalised over sources
    repo_id = "allenai/dolma3_mix-6T"
    files = "data/common_crawl-*/*.jsonl.zst"   # fnmatch glob, or a list
    text_column = "text"
    id_column = "id"

The tokenizer location is site-specific: if ``SWAFLOW_TOKENIZER_DIR`` is set
it overrides ``[tokenizer] path``, so generated configs work unchanged on
another machine. ``vocab_size`` / ``eos_id`` are never overridden (changes
to content must be explicit in the config).

Use `local_dir` instead of `repo_id` for local files. The read path is the
same, so test sources go through the same code as real ones.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["BuildConfig", "SourceConfig", "TokenizerConfig", "load_config"]

#: fnmatch metacharacters. Patterns without them are resolved by set lookup.
_GLOB_META = re.compile(r"[*?\[]")


@dataclass(frozen=True)
class TokenizerConfig:
    path: str
    vocab_size: int
    eos_id: int = 0
    name: str = ""


@dataclass(frozen=True)
class SourceConfig:
    name: str
    weight: float
    files: tuple[str, ...]
    repo_id: str = ""
    revision: str = "main"
    local_dir: str = ""
    text_column: str = "text"
    id_column: str = "id"
    max_files: int = 0  # if >0, keep the first N files after glob expansion
    #: If >0, cap on tokens per shard (= per input file). Truncation happens at
    #: a document boundary, so the actual count slightly exceeds the cap. Files
    #: are read from the start, so raising the cap only appends the continuation
    #: of the same file.
    max_tokens_per_file: int = 0
    store_doc_ids: bool = False
    #: Free-form note (subset, license, ...). Does not affect processing.
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("source.name is required")
        if self.weight <= 0:
            raise ValueError(f"source '{self.name}': weight must be positive")
        if bool(self.repo_id) == bool(self.local_dir):
            raise ValueError(
                f"source '{self.name}': specify exactly one of repo_id and local_dir"
            )
        if not self.files:
            raise ValueError(f"source '{self.name}': files is empty")

    @property
    def origin(self) -> str:
        return self.repo_id if self.repo_id else self.local_dir

    def match_files(self, available: list[str]) -> list[str]:
        """Expand glob patterns (or literal paths) into file names.

        `available` lists paths relative to the repo / directory. The result is
        sorted (deterministic) and truncated to `max_files`.

        Literal paths are resolved by set lookup. The full Dolma 3 Mix config
        lists ~64k literal paths; matching each with `fnmatch.filter` would
        cost patterns x files comparisons.
        """
        available_set = set(available)
        picked: list[str] = []
        seen: set[str] = set()
        for pattern in self.files:
            if _GLOB_META.search(pattern):
                hits = sorted(fnmatch.filter(available, pattern))
            else:
                hits = [pattern] if pattern in available_set else []
            if not hits:
                raise ValueError(
                    f"source '{self.name}': no file matching pattern '{pattern}' "
                    f"in {self.origin}."
                )
            for h in hits:
                if h not in seen:
                    seen.add(h)
                    picked.append(h)
        picked.sort()
        if self.max_files:
            picked = picked[: self.max_files]
        return picked


@dataclass(frozen=True)
class BuildConfig:
    name: str
    seq_len: int
    seed: int
    tokenizer: TokenizerConfig
    sources: tuple[SourceConfig, ...]
    holdout_fraction: float = 0.001
    #: Total training tokens (token budget of the plan). 0 means the longest
    #: run that keeps the mixture weights with every source within one epoch
    #: (set by the most constrained source).
    target_tokens: int = 0
    description: str = ""
    source_path: str = ""
    #: Refuse to process unless explicitly allowed (guards large configs
    #: against accidental runs).
    design_only: bool = False
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("build.seq_len must be positive")
        if not 0.0 <= self.holdout_fraction < 0.5:
            raise ValueError("build.holdout_fraction must be in [0, 0.5)")
        names = [s.name for s in self.sources]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate source.name: {names}")
        if not self.sources:
            raise ValueError("no [[sources]] defined")

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(s.weight for s in self.sources)
        return {s.name: s.weight / total for s in self.sources}

    def digest(self) -> str:
        """Content hash of the config; stored in plan.json to detect mix-ups."""
        payload = {
            "build": {
                "name": self.name,
                "seq_len": self.seq_len,
                "seed": self.seed,
                "holdout_fraction": self.holdout_fraction,
                "target_tokens": self.target_tokens,
            },
            "tokenizer": asdict(self.tokenizer),
            "sources": [asdict(s) for s in self.sources],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()


def _as_file_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise ValueError(f"files must be a string or a list (got {type(value)})")


def load_config(path: str | Path) -> BuildConfig:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    build = raw.get("build", {})
    tok = raw.get("tokenizer", {})
    if "path" not in tok:
        raise ValueError("[tokenizer] path is required")
    tokenizer = TokenizerConfig(
        path=str(os.environ.get("SWAFLOW_TOKENIZER_DIR") or tok["path"]),
        vocab_size=int(tok["vocab_size"]),
        eos_id=int(tok.get("eos_id", 0)),
        name=str(tok.get("name", "")),
    )

    sources = []
    for entry in raw.get("sources", []):
        entry = dict(entry)
        entry["files"] = _as_file_tuple(entry.get("files", []))
        entry["weight"] = float(entry.get("weight", 1.0))
        sources.append(SourceConfig(**entry))

    return BuildConfig(
        name=str(build["name"]),
        seq_len=int(build["seq_len"]),
        seed=int(build["seed"]),
        tokenizer=tokenizer,
        sources=tuple(sources),
        holdout_fraction=float(build.get("holdout_fraction", 0.001)),
        target_tokens=int(build.get("target_tokens", 0)),
        description=str(build.get("description", "")),
        design_only=bool(build.get("design_only", False)),
        source_path=str(path),
        raw=raw,
    )
