# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""uint16 indexed dataset.

One shard per input file, stored as three files:

    <prefix>.bin        tokens: raw uint16 little-endian array; documents
                        are concatenated, each followed by EOS
    <prefix>.idx        document boundaries: header + doc_offset[int64, N+1]
                        + doc_row[int64, N]
    <prefix>.meta.json  provenance and checks (source name, input file,
                        revision, tokenizer, counts, doc_id digest)

With one shard per input file, document j is identified by (input file, row),
so the order log needs no text. The input file is also the unit of
parallelism.

uint16 suffices because V = 49,152 < 65,536. The writer checks the vocabulary
bound and fails instead of silently corrupting data.

doc_ids are not stored by default (tens of GB at 2T tokens). Instead each
shard's meta holds a blake2b digest of its concatenated doc_ids, so a
re-downloaded input file can be checked to hold the same documents without any
text. With `store_doc_ids`, raw IDs also go to `<prefix>.docids.txt` (for small
sources and checks).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

#: Maximum number of `.bin` mmaps kept open: well below the default
#: `vm.max_map_count` (65,530) with a useful hit rate. Set via the environment.
_MMAP_MAX_OPEN = int(os.environ.get("SWAFLOW_MMAP_MAX_OPEN", "1024"))

__all__ = [
    "MMAP_POOL",
    "read_idx_header",
    "IDX_MAGIC",
    "TOKEN_DTYPE",
    "Shard",
    "ShardWriter",
    "read_idx",
    "shard_path",
]


def shard_path(prefix: str | Path, ext: str) -> Path:
    """`<prefix><ext>` by concatenation, never `Path.with_suffix`.

    `with_suffix` replaces an existing extension, so a prefix containing `.`
    (the in-progress `.partial` prefix, or a shard named after a dotted file
    name) would silently point to another path.
    """
    return Path(str(prefix) + ext)

#: Shard index header; fixed, so that the released shards stay readable.
IDX_MAGIC = b"TFLMIDX\x00"
IDX_VERSION = 1
TOKEN_DTYPE = np.uint16
_DTYPE_CODE = {1: np.uint16}
_CODE_OF_DTYPE = {np.dtype(np.uint16): 1}

_HEADER_DTYPE = np.dtype(
    [
        ("version", "<u4"),
        ("dtype_code", "<u4"),
        ("n_docs", "<u8"),
        ("n_tokens", "<u8"),
    ]
)


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
class ShardWriter:
    """Write one shard (= one input file).

    Usage::

        with ShardWriter(prefix, vocab_size=49152) as w:
            for row, tokens in enumerate(...):
                w.add_document(tokens, row=row, doc_id="...")
            meta = w.finalize(extra_meta)
    """

    def __init__(
        self,
        prefix: str | Path,
        vocab_size: int,
        dtype: type = TOKEN_DTYPE,
        store_doc_ids: bool = False,
    ) -> None:
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = np.dtype(dtype)
        if self.dtype not in _CODE_OF_DTYPE:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if vocab_size > np.iinfo(self.dtype).max + 1:
            raise ValueError(
                f"vocab_size {vocab_size} does not fit in {self.dtype} "
                f"(limit {np.iinfo(self.dtype).max + 1}); token storage assumes uint16"
            )
        self.vocab_size = int(vocab_size)
        self.store_doc_ids = store_doc_ids

        self._bin = open(shard_path(self.prefix, ".bin"), "wb")
        self._offsets: list[int] = [0]
        self._rows: list[int] = []
        self._n_tokens = 0
        self._id_hash = hashlib.blake2b(digest_size=16)
        self._docids = (
            open(shard_path(self.prefix, ".docids.txt"), "w") if store_doc_ids else None
        )

    @property
    def n_docs(self) -> int:
        return len(self._rows)

    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    def add_document(self, tokens: np.ndarray, row: int, doc_id: str = "") -> None:
        arr = np.asarray(tokens)
        if arr.size == 0:
            raise ValueError("do not pass empty documents to add_document")
        if arr.max() >= self.vocab_size or arr.min() < 0:
            raise ValueError(
                f"token ID outside the vocabulary range [0, {self.vocab_size}) "
                f"(min {arr.min()}, max {arr.max()})"
            )
        self._bin.write(arr.astype(self.dtype, copy=False).tobytes())
        self._n_tokens += int(arr.size)
        self._offsets.append(self._n_tokens)
        self._rows.append(int(row))
        self._id_hash.update(doc_id.encode())
        self._id_hash.update(b"\n")
        if self._docids is not None:
            self._docids.write(f"{row}\t{doc_id}\n")

    def finalize(self, extra_meta: dict | None = None) -> dict:
        self._bin.close()
        if self._docids is not None:
            self._docids.close()

        header = np.zeros(1, dtype=_HEADER_DTYPE)
        header["version"] = IDX_VERSION
        header["dtype_code"] = _CODE_OF_DTYPE[self.dtype]
        header["n_docs"] = len(self._rows)
        header["n_tokens"] = self._n_tokens
        with open(shard_path(self.prefix, ".idx"), "wb") as f:
            f.write(IDX_MAGIC)
            f.write(header.tobytes())
            f.write(np.asarray(self._offsets, dtype="<i8").tobytes())
            f.write(np.asarray(self._rows, dtype="<i8").tobytes())

        meta = {
            "format_version": IDX_VERSION,
            "token_dtype": self.dtype.name,
            "vocab_size": self.vocab_size,
            "n_docs": len(self._rows),
            "n_tokens": self._n_tokens,
            "doc_id_digest": self._id_hash.hexdigest(),
            "store_doc_ids": self.store_doc_ids,
            **(extra_meta or {}),
        }
        with open(shard_path(self.prefix, ".meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=True)
        return meta

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        if not self._bin.closed:
            self._bin.close()
        if self._docids is not None and not self._docids.closed:
            self._docids.close()


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
def read_idx_header(path: str | Path) -> dict:
    """Read only the 32-byte header of an `.idx` file.

    `read_idx` loads the whole file and keeps doc_offset / doc_row resident
    (~1.3 MB per shard on average), but the training path (`read_block`) never
    uses them; with ~64k shards that would reach tens of GB per rank.

    The header holds n_tokens / dtype_code, so `.bin` is still checked against
    `.idx` (not `.meta.json`).
    """
    with open(path, "rb") as f:
        raw = f.read(len(IDX_MAGIC) + _HEADER_DTYPE.itemsize)
    if raw[: len(IDX_MAGIC)] != IDX_MAGIC:
        raise ValueError(f"{path} is not a SwaFlow .idx file")
    header = np.frombuffer(raw, dtype=_HEADER_DTYPE, count=1, offset=len(IDX_MAGIC))[0]
    if int(header["version"]) != IDX_VERSION:
        raise ValueError(
            f"{path}: unsupported format_version {int(header['version'])} "
            f"(this code reads {IDX_VERSION})"
        )
    return {k: int(header[k]) for k in _HEADER_DTYPE.names}


class _MmapPool:
    """LRU that caps the number of open `.bin` mmaps.

    `vm.max_map_count` (default 65,530) limits mappings per process, which is
    close to the number of shards in the full build. The global permutation
    spreads samples over all shards, so the distinct shards a rank touches grow
    with the step count until they reach that limit.

    Eviction calls `munmap`, so `read_block` must return a copy (a view would
    be a use-after-unmap). Re-mapping is a syscall only and rereads no data.
    """

    def __init__(self, max_open: int = _MMAP_MAX_OPEN) -> None:
        self.max_open = max_open
        self._open: "OrderedDict[str, np.memmap]" = OrderedDict()
        self.opens = 0
        self.evictions = 0

    def get(self, prefix: Path, dtype, n_tokens: int) -> np.memmap:
        key = str(prefix)
        m = self._open.get(key)
        if m is not None:
            self._open.move_to_end(key)
            return m
        while len(self._open) >= self.max_open:
            _, old = self._open.popitem(last=False)
            old._mmap.close()
            self.evictions += 1
        m = np.memmap(shard_path(prefix, ".bin"), dtype=dtype, mode="r")
        if m.size != n_tokens:
            raise ValueError(
                f"{prefix}.bin has {m.size} tokens but .idx says {n_tokens}"
                " (possibly corrupted)"
            )
        self._open[key] = m
        self.opens += 1
        return m

    def clear(self) -> None:
        while self._open:
            _, m = self._open.popitem()
            m._mmap.close()


#: Process-wide mmap pool: the number of mappings stays capped no matter how
#: many `Shard` objects exist.
MMAP_POOL = _MmapPool()


def read_idx(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read an .idx file; returns (doc_offset[N+1], doc_row[N], header)."""
    raw = Path(path).read_bytes()
    if raw[: len(IDX_MAGIC)] != IDX_MAGIC:
        raise ValueError(f"{path} is not a SwaFlow .idx file")
    pos = len(IDX_MAGIC)
    header = np.frombuffer(raw, dtype=_HEADER_DTYPE, count=1, offset=pos)[0]
    if int(header["version"]) != IDX_VERSION:
        raise ValueError(
            f"{path}: unsupported format_version {int(header['version'])} "
            f"(this code reads {IDX_VERSION})"
        )
    pos += _HEADER_DTYPE.itemsize
    n_docs = int(header["n_docs"])
    offsets = np.frombuffer(raw, dtype="<i8", count=n_docs + 1, offset=pos).copy()
    pos += (n_docs + 1) * 8
    rows = np.frombuffer(raw, dtype="<i8", count=n_docs, offset=pos).copy()
    return offsets, rows, {k: int(header[k]) for k in _HEADER_DTYPE.names}


@dataclass
class DocRef:
    """One document referenced by the order log (no text)."""

    source: str
    shard: str
    doc_index: int
    source_file: str
    source_row: int
    n_tokens: int


class Shard:
    """Read a written shard (tokens via a pooled memmap; boundaries loaded lazily)."""

    def __init__(self, prefix: str | Path) -> None:
        self.prefix = Path(prefix)
        self.meta = json.loads(shard_path(self.prefix, ".meta.json").read_text())
        # Read only the 32-byte .idx header. doc_offset / doc_row are needed only
        # by the order log and are loaded on first access.
        header = read_idx_header(shard_path(self.prefix, ".idx"))
        self.dtype = _DTYPE_CODE[header["dtype_code"]]
        self.n_docs = header["n_docs"]
        self.n_tokens = header["n_tokens"]
        self._idx: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def name(self) -> str:
        return self.prefix.name

    @property
    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Document boundaries from `.idx` (order log only; loaded on first access)."""
        if self._idx is None:
            offsets, rows, header = read_idx(shard_path(self.prefix, ".idx"))
            if header["n_tokens"] != self.n_tokens:
                raise ValueError(f"{self.prefix}.idx: header and body disagree")
            self._idx = (offsets, rows)
        return self._idx

    @property
    def doc_offset(self) -> np.ndarray:
        return self._arrays[0]

    @property
    def doc_row(self) -> np.ndarray:
        return self._arrays[1]

    @property
    def tokens(self) -> np.ndarray:
        """mmap of `.bin`, borrowed from the pool (not owned).

        The returned array, and any view of it, becomes invalid once more than
        `MMAP_POOL.max_open` other shards have been touched (it is evicted and
        unmapped). Use it only to

        1. read immediately (copy with `np.array(...)`), or
        2. work inside a short operation on a single shard.

        Keeping views across shards is a use-after-unmap. For a block, use
        `read_block`, which returns a copy.
        """
        return MMAP_POOL.get(self.prefix, self.dtype, self.n_tokens)

    def n_blocks(self, seq_len: int) -> int:
        return self.n_tokens // seq_len

    def read_block(self, block: int, seq_len: int) -> np.ndarray:
        start = block * seq_len
        if block < 0 or start + seq_len > self.n_tokens:
            raise IndexError(f"block {block} is out of range (shard {self.name})")
        # Return a copy: a view would be a use-after-unmap once the pool evicts the
        # mmap. A block is a few KB, so the copy is cheap.
        return np.array(self.tokens[start : start + seq_len])

    def docs_in_block(self, block: int, seq_len: int) -> list[DocRef]:
        """Documents overlapping the block (used to expand the order log)."""
        start = block * seq_len
        end = start + seq_len
        first = int(np.searchsorted(self.doc_offset, start, side="right") - 1)
        last = int(np.searchsorted(self.doc_offset, end, side="left") - 1)
        refs = []
        for d in range(max(first, 0), min(last, self.n_docs - 1) + 1):
            refs.append(
                DocRef(
                    source=self.meta.get("source", ""),
                    shard=self.name,
                    doc_index=d,
                    source_file=self.meta.get("source_file", ""),
                    source_row=int(self.doc_row[d]),
                    n_tokens=int(self.doc_offset[d + 1] - self.doc_offset[d]),
                )
            )
        return refs

    def iter_documents(self) -> Iterator[np.ndarray]:
        for d in range(self.n_docs):
            tokens = self.tokens          # re-fetch from the pool each time
            yield np.array(tokens[self.doc_offset[d] : self.doc_offset[d + 1]])
