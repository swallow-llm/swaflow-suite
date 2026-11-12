# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing pipeline: input files -> tokenization -> uint16 indexed dataset.

Inputs may be parquet or JSON Lines (`.jsonl` / `.jsonl.zst`; Dolma 3 Mix uses
`.jsonl.zst`). Either way one input file becomes one shard, and document j is
identified by (input file, row).

    # 1) download only
    python -m scripts.data.tokenize_corpus --config <cfg.toml> --stage download

    # 2) tokenize + build the plan
    python -m scripts.data.tokenize_corpus --config <cfg.toml> --stage tokenize

The stages are separate so that downloading can run where network access is
available; `--stage all` runs both.

One shard per input file. Completed shards are skipped, so an interrupted run
resumes with the same command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # also run as `python scripts/data/tokenize_corpus.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.data"

from .config import BuildConfig, SourceConfig, load_config
from .indexed_dataset import shard_path, Shard, ShardWriter
from .plan import ShardRef, build_plan
from .progress import ProgressReporter

DEFAULT_ROOT = Path(
    os.environ.get("SWAFLOW_DATA_ROOT", "./data")
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# resolving / fetching source files
# ----------------------------------------------------------------------
def _list_available(src: SourceConfig) -> list[str]:
    if src.repo_id:
        from huggingface_hub import HfApi

        return HfApi().list_repo_files(
            src.repo_id, repo_type="dataset", revision=src.revision
        )
    root = Path(src.local_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"source '{src.name}': local_dir not found: {root}")
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_source_files(src: SourceConfig) -> list[str]:
    return src.match_files(_list_available(src))


_COMMIT_CACHE: dict[tuple[str, str], str] = {}


def _repo_commit(api, src: SourceConfig) -> str:
    """Resolve a revision (branch name etc.) to a commit sha."""
    key = (src.repo_id, src.revision)
    if key not in _COMMIT_CACHE:
        _COMMIT_CACHE[key] = api.dataset_info(src.repo_id, revision=src.revision).sha
    return _COMMIT_CACHE[key]


def paths_info_map(src: SourceConfig, rel_paths: list[str], batch: int = 500) -> dict:
    """Query `get_paths_info` in batches; returns rel_path -> (sha256, size).

    One call per file would hit the metadata API ~64k times for the full build,
    which is slow and can run into API rate limits. The API accepts a list of
    paths, so each source is queried in batches.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    out: dict[str, dict] = {}
    for i in range(0, len(rel_paths), batch):
        chunk = rel_paths[i : i + batch]
        for e in api.get_paths_info(
            src.repo_id, paths=chunk, repo_type="dataset", revision=src.revision
        ):
            lfs = getattr(e, "lfs", None)
            out[e.path] = {
                "sha256": getattr(lfs, "sha256", "") if lfs else "",
                "size": getattr(e, "size", 0) or 0,
            }
    return out


def fetch_file(
    src: SourceConfig, rel_path: str, raw_dir: Path, info: dict | None = None
) -> tuple[Path, dict]:
    """Return the local copy of an input file and its provenance metadata.

    HF sources use the `hf_hub_download` cache directly (no second copy).
    Passing `info` from `paths_info_map` avoids one metadata request per file.
    """
    if src.local_dir:
        path = Path(src.local_dir) / rel_path
        return path, {
            "origin": src.local_dir,
            "revision": "",
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    if info is None:
        entry = next(
            iter(
                api.get_paths_info(
                    src.repo_id, paths=[rel_path], repo_type="dataset", revision=src.revision
                )
            ),
            None,
        )
        lfs = getattr(entry, "lfs", None)
        info = {
            "sha256": getattr(lfs, "sha256", "") if lfs else "",
            "size": getattr(entry, "size", 0) if entry else 0,
        }
    commit = _repo_commit(api, src)
    path = Path(
        hf_hub_download(
            src.repo_id,
            rel_path,
            repo_type="dataset",
            revision=src.revision,
            cache_dir=str(raw_dir) if raw_dir else None,
        )
    )
    return path, {
        "origin": src.repo_id,
        "revision": commit,
        # The LFS oid is the sha256, so identity can be checked without re-downloading
        "sha256": info.get("sha256", ""),
        "size": info.get("size", 0),
    }


# ----------------------------------------------------------------------
# reading input files (parquet / JSON Lines)
# ----------------------------------------------------------------------
#: Extensions stripped to form shard names, longest first (so `.jsonl.zst` is
#: not cut at `.zst`, leaving `foo.jsonl`).
DATA_SUFFIXES = (".jsonl.zst", ".json.zst", ".jsonl.gz", ".jsonl", ".parquet")


def strip_data_suffix(rel_path: str) -> str:
    for suffix in DATA_SUFFIXES:
        if rel_path.endswith(suffix):
            return rel_path[: -len(suffix)]
    return rel_path


def _open_text_lines(path: Path):
    """Open `.jsonl` / `.jsonl.zst` / `.jsonl.gz` as a line iterator.

    zstd uses the Python 3.14 standard library (`compression.zstd`), which adds
    no dependency; older Pythons fall back to pyarrow's codec.
    """
    name = path.name
    if name.endswith(".gz"):
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    if not name.endswith(".zst"):
        return open(path, "rt", encoding="utf-8")
    try:
        from compression import zstd  # Python 3.14+
    except ImportError:
        pass
    else:
        return zstd.open(path, "rt", encoding="utf-8")
    import io

    import pyarrow as pa

    stream = pa.CompressedInputStream(pa.OSFile(str(path), "rb"), "zstd")
    return io.TextIOWrapper(io.BufferedReader(stream), encoding="utf-8")


def iter_record_batches(path: Path, src: SourceConfig, batch_size: int):
    """One input file -> batches of `(texts, ids, line_bytes)`.

    Input row order is preserved: the row number (parquet row / jsonl line)
    becomes `doc_row`, so the order log works the same for both formats.

    `line_bytes` holds the uncompressed input bytes of each line (recorded in
    the shard meta). It is per line rather than a total because of truncation:
    the caller adds only the lines it consumed. Empty for parquet.
    """
    if path.name.endswith(".parquet"):
        import pyarrow.parquet as pq

        columns = [src.text_column] + ([src.id_column] if src.id_column else [])
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size, columns=columns
        ):
            texts = batch.column(src.text_column).to_pylist()
            ids = (
                batch.column(src.id_column).to_pylist()
                if src.id_column
                else [""] * len(texts)
            )
            yield texts, ids, []
        return

    texts: list[str] = []
    ids: list[str] = []
    line_bytes: list[int] = []
    with _open_text_lines(path) as f:
        for line in f:
            line_bytes.append(len(line.encode()))
            if not line.strip():
                # Empty lines still count as rows (keeps row number == doc_row)
                texts.append("")
                ids.append("")
            else:
                record = json.loads(line)
                texts.append(record.get(src.text_column) or "")
                ids.append(str(record.get(src.id_column, "")) if src.id_column else "")
            if len(texts) >= batch_size:
                yield texts, ids, line_bytes
                texts, ids, line_bytes = [], [], []
    if texts:
        yield texts, ids, line_bytes


# ----------------------------------------------------------------------
# tokenization
# ----------------------------------------------------------------------
#: Per-process tokenizer cache. `tokenize_file` runs once per input file, so
#: loading the tokenizer each time would repeat the load ~64k times for the
#: full build.
_TOKENIZER_CACHE: dict[str, object] = {}


def _load_tokenizer(path: str):
    from tokenizers import Tokenizer

    cached = _TOKENIZER_CACHE.get(path)
    if cached is not None:
        return cached
    p = Path(path)
    tok_file = p / "tokenizer.json" if p.is_dir() else p
    if not tok_file.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tok_file}")
    tok = Tokenizer.from_file(str(tok_file))
    _TOKENIZER_CACHE[path] = tok
    return tok


def tokenize_file(
    input_path: Path,
    prefix: Path,
    cfg: BuildConfig,
    src: SourceConfig,
    origin_meta: dict,
    rel_path: str,
    batch_size: int = 2048,
    limit_docs: int = 0,
) -> dict:
    """One input file -> one shard.

    With `src.max_tokens_per_file` / `limit_docs`, only that much is read from
    the start of the input. Truncation is at a document boundary, so the shard
    covers exactly the input rows [0, n_rows_read).
    """
    tokenizer = _load_tokenizer(cfg.tokenizer.path)
    eos = np.uint16(cfg.tokenizer.eos_id)
    max_tokens = int(src.max_tokens_per_file)
    n_read = 0
    t0 = time.time()

    tmp_prefix = prefix.with_name(prefix.name + ".partial")
    writer = ShardWriter(
        tmp_prefix,
        vocab_size=cfg.tokenizer.vocab_size,
        store_doc_ids=src.store_doc_ids,
    )

    stop = False
    input_bytes = text_bytes = 0
    for texts, ids, line_bytes in iter_record_batches(input_path, src, batch_size):
        keep = [i for i, t in enumerate(texts) if t and t.strip()]
        encoded = tokenizer.encode_batch([texts[i] for i in keep], add_special_tokens=False)
        n_used = len(texts)  # rows consumed from this batch (fewer if truncated)
        for i, enc in zip(keep, encoded):
            row = n_read + i
            toks = np.asarray(enc.ids, dtype=np.uint16)
            text_bytes += len(texts[i].encode())
            writer.add_document(np.append(toks, eos), row=row, doc_id=ids[i] or "")
            if limit_docs and writer.n_docs >= limit_docs:
                stop = True
            if max_tokens and writer.n_tokens >= max_tokens:
                stop = True
            if stop:
                # Count rows up to the last kept document (keeps the row range exact)
                n_used = i + 1
                break
        n_read += n_used
        input_bytes += sum(line_bytes[:n_used])
        if stop:
            break

    meta = writer.finalize(
        {
            "source": src.name,
            "source_file": rel_path,
            "origin": origin_meta["origin"],
            "revision": origin_meta["revision"],
            "source_sha256": origin_meta["sha256"],
            "source_size": origin_meta["size"],
            "text_column": src.text_column,
            "id_column": src.id_column,
            "tokenizer": cfg.tokenizer.name or cfg.tokenizer.path,
            "tokenizer_path": cfg.tokenizer.path,
            "eos_id": cfg.tokenizer.eos_id,
            "n_rows_read": n_read,
            # Rows read but dropped as empty (row numbers are not compacted)
            "n_rows_skipped_empty": n_read - writer.n_docs,
            # Input size of the consumed rows (for tokens/GB statistics):
            # raw line bytes for jsonl, 0 for parquet (not measurable)
            "n_input_bytes": input_bytes,
            "n_text_bytes": text_bytes,
            "truncated": bool(stop),
            "max_tokens_per_file": max_tokens,
            "seq_len": cfg.seq_len,
        }
    )
    # Rename only when complete, so an interrupted shard never looks finished
    for suffix in (".bin", ".idx", ".meta.json", ".docids.txt"):
        s = shard_path(tmp_prefix, suffix)
        if s.exists():
            s.replace(shard_path(prefix, suffix))

    dt = time.time() - t0
    log(
        f"  {rel_path}: {meta['n_docs']:,} docs / {meta['n_tokens'] / 1e6:.1f}M tokens "
        f"({dt:.0f}s, {meta['n_tokens'] / max(dt, 1e-9) / 1e6:.2f}M tok/s), "
        f"rows [0, {n_read:,}), empty skipped {meta['n_rows_skipped_empty']}"
        + (" [truncated]" if stop else "")
    )
    return meta


# ----------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------
#: Ledger of input files resolved by the download stage. The tokenize stage
#: reads it, so it needs no network access.
INPUTS_FILENAME = "inputs.json"


def _load_inputs(build_dir: Path) -> dict:
    path = build_dir / INPUTS_FILENAME
    return json.loads(path.read_text()) if path.exists() else {}


def shard_prefix_for(build_dir: Path, src_name: str, rel: str) -> Path:
    """Input file relative path -> shard prefix. Resuming relies on this map.

    The download, tokenize and plan stages all derive the same prefix, so an
    existing `.meta.json` is enough to tell that a shard is complete.
    """
    return build_dir / "shards" / src_name / strip_data_suffix(rel).replace("/", "__")


def _assign_to_node(items: list, node_rank: int, n_nodes: int) -> list:
    """Assign items to nodes round-robin.

    Not contiguous ranges: input files are sorted by partition and file sizes
    differ a lot between sources (a common_crawl file holds ~3x the tokens of
    an olmocr file), so contiguous splits would unbalance the nodes.
    """
    if n_nodes <= 1:
        return items
    return [x for i, x in enumerate(items) if i % n_nodes == node_rank]


# ---- download stage -------------------------------------------------
def _fetch_task(task: tuple) -> dict:
    """Fetch one file in a worker thread; exceptions are returned in the result.

    One failure does not stop the stage. Failed files are left out of the
    ledger, so rerunning retries them (completed files hit the cache).
    """
    src, rel, raw_dir, info = task
    try:
        path, meta = fetch_file(src, rel, raw_dir, info=info)
        return {"rel": rel, "local_path": str(path), "ok": True, **meta}
    except Exception as exc:
        return {"rel": rel, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def stage_download(
    cfg: BuildConfig,
    build_dir: Path,
    raw_dir: Path,
    workers: int = 16,
    node_rank: int = 0,
    n_nodes: int = 1,
    progress_interval: float = 60.0,
) -> dict:
    """Download the input files and write the ledger (`inputs.json`).

    Idempotent: `hf_hub_download` does not touch the network when the cache
    already holds the file, so an interrupted run resumes with the same command.

    Network-bound, so it uses threads rather than processes.

    On multiple nodes each node writes only its part of the ledger
    (`inputs.rank<N>.json`); `merge_inputs()` combines them afterwards.
    """
    from concurrent.futures import ThreadPoolExecutor

    inputs: dict = {"config_digest": cfg.digest(), "sources": {}, "errors": []}
    for src in cfg.sources:
        files = resolve_source_files(src)
        mine = _assign_to_node(files, node_rank, n_nodes)
        log(
            f"[download] {src.name}: {len(files):,} files total, "
            f"{len(mine):,} on this node (rank {node_rank}/{n_nodes}, workers {workers})"
            f" from {src.origin}"
        )
        rep = ProgressReporter(
            total_items=len(mine),
            label=f"download:{src.name}",
            unit_name="bytes",
            interval_sec=progress_interval,
            jsonl_path=build_dir / "progress" / f"download.{src.name}.rank{node_rank}.jsonl",
            log=log,
        )
        # Fetch metadata in batches (one request per file would mean ~64k API calls)
        info = paths_info_map(src, mine) if src.repo_id else {}
        entries: list[dict] = []
        tasks = [(src, rel, raw_dir, info.get(rel)) for rel in mine]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for res in pool.map(_fetch_task, tasks):
                rep.advance(units=res.get("size", 0) or 0, failed=not res["ok"])
                if res["ok"]:
                    entries.append(res)
                else:
                    inputs["errors"].append({"source": src.name, **res})
        rep.finish()
        inputs["sources"][src.name] = entries

    name = INPUTS_FILENAME if n_nodes == 1 else f"inputs.rank{node_rank}.json"
    with open(build_dir / name, "w") as f:
        json.dump(inputs, f, indent=2, ensure_ascii=False)
    log(f"[download] ledger -> {build_dir / name}")
    if inputs["errors"]:
        log(f"[download] **{len(inputs['errors'])} files failed**; rerun to resume")
    return inputs


def merge_inputs(build_dir: Path, n_nodes: int) -> dict:
    """Merge `inputs.rank*.json` into one `inputs.json` (after a multi-node download)."""
    merged: dict = {"config_digest": "", "sources": {}, "errors": []}
    for rank in range(n_nodes):
        path = build_dir / f"inputs.rank{rank}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (download for rank {rank} has not finished)")
        part = json.loads(path.read_text())
        merged["config_digest"] = part.get("config_digest", "")
        merged["errors"].extend(part.get("errors", []))
        for name, entries in part.get("sources", {}).items():
            merged["sources"].setdefault(name, []).extend(entries)
    for name in merged["sources"]:
        merged["sources"][name].sort(key=lambda e: e["rel"])
    with open(build_dir / INPUTS_FILENAME, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    total = sum(len(v) for v in merged["sources"].values())
    log(f"[download] merged ledger -> {build_dir / INPUTS_FILENAME} ({total:,} files)")
    return merged


# ---- tokenize stage -------------------------------------------------
def light_task_payload(cfg: BuildConfig, src: SourceConfig) -> tuple[BuildConfig, SourceConfig]:
    """Copy of the config without file lists, for sending to workers.

    `ProcessPoolExecutor.map` pickles the arguments of every task. A full
    `BuildConfig` carries every source's `files` tuple (~64k paths for the full
    build) and the parsed TOML in `raw`, i.e. several MB per task; over ~64k
    tasks the transfer alone would swamp the workers.

    `tokenize_file` uses only `cfg.tokenizer` / `cfg.seq_len` and the source's
    column names and cap; `fetch_file` uses repo_id / revision / local_dir.
    None of these depend on the file list.
    """
    light_src = replace(src, files=("*",))
    return replace(cfg, sources=(light_src,), raw={}), light_src


def _tokenize_task(task: tuple) -> dict:
    """Tokenize one file in a worker process.

    Exceptions are returned in the result, so one broken input does not stop
    the stage. A failed shard leaves no `.meta.json` and is retried on the next
    run.
    """
    cfg, src, entry, build_dir, limit_docs, raw_dir = task
    rel = entry["rel"]
    prefix = shard_prefix_for(build_dir, src.name, rel)
    try:
        if "local_path" in entry:
            path, origin_meta = Path(entry["local_path"]), entry
        else:
            path, origin_meta = fetch_file(src, rel, raw_dir)
        meta = tokenize_file(
            path, prefix, cfg, src, origin_meta, rel, limit_docs=limit_docs
        )
        return {"rel": rel, "ok": True, "n_tokens": meta["n_tokens"]}
    except Exception as exc:
        return {"rel": rel, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def stage_tokenize(
    cfg: BuildConfig,
    build_dir: Path,
    raw_dir: Path,
    limit_docs: int = 0,
    force: bool = False,
    workers: int = 0,
    node_rank: int = 0,
    n_nodes: int = 1,
    progress_interval: float = 60.0,
) -> dict:
    """Input files -> shards. Does not build the plan (see `collect_shard_refs`).

    Parallel over shards, each an independent input file. Uses processes,
    since zstd decompression and JSON parsing are Python CPU work (threads
    would contend for the GIL).

    Workers set `TOKENIZERS_PARALLELISM=false`; otherwise each of the N
    processes would start a thread per core in the Rust tokenizer and
    oversubscribe the CPU.

    Idempotent: shards with a `.meta.json` are skipped. Output is written under
    a `.partial` prefix and renamed at the end, so a killed job never leaves a
    half-written shard that looks complete.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    cached = _load_inputs(build_dir)
    if cached and cached.get("config_digest") != cfg.digest():
        log(
            f"[warn] {INPUTS_FILENAME} was written with a different config digest; "
            "sources missing from it are resolved directly"
        )
    if workers <= 0:
        workers = _default_workers()

    stats = {"done": 0, "skipped": 0, "failed": 0, "n_tokens": 0, "errors": []}
    for src in cfg.sources:
        entries = cached.get("sources", {}).get(src.name)
        if entries is None:
            entries = [{"rel": rel} for rel in resolve_source_files(src)]
        mine = _assign_to_node(entries, node_rank, n_nodes)

        todo, skipped = [], 0
        for entry in mine:
            prefix = shard_prefix_for(build_dir, src.name, entry["rel"])
            if shard_path(prefix, ".meta.json").exists() and not force:
                skipped += 1
            else:
                todo.append(entry)
        stats["skipped"] += skipped
        log(
            f"[tokenize] {src.name}: assigned {len(mine):,} / {len(entries):,} files"
            f" (rank {node_rank}/{n_nodes}), skipping {skipped:,} existing,"
            f" {len(todo):,} to process with {workers} workers"
        )
        if not todo:
            continue

        rep = ProgressReporter(
            total_items=len(todo),
            label=f"tokenize:{src.name}",
            unit_name="tokens",
            interval_sec=progress_interval,
            jsonl_path=build_dir / "progress" / f"tokenize.{src.name}.rank{node_rank}.jsonl",
            log=log,
        )
        # Build the file-list-free config once and share it across all tasks
        # (the full cfg would cost several MB per task).
        light_cfg, light_src = light_task_payload(cfg, src)
        tasks = [(light_cfg, light_src, e, build_dir, limit_docs, raw_dir) for e in todo]
        try:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_worker_init
            ) as pool:
                # chunksize=1: per-file time ranges from seconds to minutes, so
                # larger chunks leave a single worker running at the end
                for res in pool.map(_tokenize_task, tasks, chunksize=1):
                    rep.advance(units=res.get("n_tokens", 0) or 0, failed=not res["ok"])
                    if res["ok"]:
                        stats["done"] += 1
                        stats["n_tokens"] += res["n_tokens"]
                    else:
                        stats["failed"] += 1
                        stats["errors"].append({"source": src.name, **res})
                        log(f"  [FAILED] {res['rel']}: {res['error']}")
        except BrokenProcessPool as exc:
            # A worker died (e.g. OOM killer). Completed shards are kept, so fix
            # the cause (usually too many --workers) and rerun.
            rep.finish()
            log(
                f"[tokenize] worker process died ({exc})."
                f" {src.name}: {stats['done']:,} files done."
                " Completed shards are kept; lower --workers and rerun to"
                " continue (most likely out of memory)"
            )
            stats["failed"] += 1
            stats["errors"].append(
                {"source": src.name, "rel": "<pool>", "ok": False,
                 "error": f"BrokenProcessPool: {exc}"}
            )
            return stats
        rep.finish()

    log(
        f"[tokenize] rank {node_rank}: done {stats['done']:,} / skipped "
        f"{stats['skipped']:,} / failed {stats['failed']:,} / "
        f"{stats['n_tokens'] / 1e9:.3f}G tokens"
    )
    if stats["failed"]:
        log("[tokenize] failed shards leave no .meta.json and are retried on the next run")
    return stats


def _worker_init() -> None:
    """Worker process initializer.

    N processes each running a thread per core in the Rust tokenizer would
    oversubscribe the CPU, so parallelism is kept to one level (processes).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _default_workers() -> int:
    """Default worker count.

    `os.process_cpu_count()` honours cgroups / affinity, so only the CPUs
    assigned to the job are counted. The default is half of them (about the
    physical cores): hyperthreads help little for integer work such as zstd
    decompression and add I/O pressure. Override with --workers.
    """
    n = os.process_cpu_count() or os.cpu_count() or 8
    return max(1, n // 2)


def collect_shard_refs(cfg: BuildConfig, build_dir: Path) -> list[dict]:
    """Read the `.meta.json` of every completed shard and build the plan input.

    Call once, after tokenization has finished on all nodes. Missing shards are
    listed and the call fails (silently building a smaller plan would drop data
    from training unnoticed).
    """
    cached = _load_inputs(build_dir)
    source_entries: list[dict] = []
    missing: list[str] = []
    for src in cfg.sources:
        entries = cached.get("sources", {}).get(src.name)
        if entries is None:
            entries = [{"rel": rel} for rel in resolve_source_files(src)]
        shard_refs: list[ShardRef] = []
        for entry in entries:
            rel = entry["rel"]
            prefix = shard_prefix_for(build_dir, src.name, rel)
            meta_path = shard_path(prefix, ".meta.json")
            if not meta_path.exists():
                missing.append(f"{src.name}/{rel}")
                continue
            meta = json.loads(meta_path.read_text())
            n_blocks = meta["n_tokens"] // cfg.seq_len
            if n_blocks == 0:
                # Shards without a single full block are left out of the plan:
                # repeated values in block_offsets could make searchsorted point
                # at an empty shard (which would fail when reading data).
                log(f"  [skip] {rel}: {meta['n_tokens']} tokens, less than one block")
                continue
            shard_refs.append(
                ShardRef(
                    name=prefix.name,
                    prefix=str(prefix.relative_to(build_dir)),
                    source_file=rel,
                    n_docs=meta["n_docs"],
                    n_tokens=meta["n_tokens"],
                    n_blocks=n_blocks,
                )
            )
        log(
            f"[plan] {src.name}: {len(shard_refs):,} shards, "
            f"{sum(s.n_tokens for s in shard_refs) / 1e9:.3f}G tokens"
        )
        source_entries.append(
            {
                "name": src.name,
                "weight": src.weight,
                "origin": src.origin,
                "revision": src.revision,
                "shards": shard_refs,
            }
        )
    if missing:
        head = "\n  ".join(missing[:20])
        raise RuntimeError(
            f"{len(missing):,} shards are incomplete; finish the tokenize stage"
            f" before building the plan:\n  {head}"
            + ("\n  ..." if len(missing) > 20 else "")
        )
    return source_entries


#: Maximum number of held-out blocks materialised per source.
#:
#: `holdout_fraction` (0.1%) is the amount withheld from training, not the amount
#: used for evaluation. For the full build 0.1% is ~2.9M blocks (~12 GB as
#: uint16), too large to load as one array and far more than validation needs.
#: Only the first N blocks are materialised; the whole 0.1% stays excluded from
#: training (as defined by `plan.heldout_blocks`).
DEFAULT_HELDOUT_EXPORT_CAP = 16384


def export_heldout(plan, build_dir: Path, cap: int = DEFAULT_HELDOUT_EXPORT_CAP) -> None:
    """Materialise the held-out blocks (`heldout/<source>.npz`) for validation.

    Shards are opened one at a time. The full build has ~49k common_crawl
    shards, and opening them all would exhaust file descriptors and memory, so
    blocks are grouped by shard and read shard by shard.
    """
    out_dir = build_dir / "heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    for si, sp in enumerate(plan.sources):
        blocks = plan.heldout_blocks(si)
        if blocks.size == 0:
            continue
        n_total = blocks.size
        if cap and blocks.size > cap:
            # The first `cap` blocks in held-out permutation order (deterministic)
            blocks = blocks[:cap]
        shard_idx, block_in_shard = plan.locate_block(si, blocks)
        arr = np.empty((blocks.size, plan.seq_len), dtype=np.uint16)
        order = np.argsort(shard_idx, kind="stable")
        opened = 0
        k = 0
        while k < order.size:
            s_i = int(shard_idx[order[k]])
            shard = Shard(build_dir / sp.shards[s_i].prefix)
            opened += 1
            while k < order.size and int(shard_idx[order[k]]) == s_i:
                pos = order[k]
                arr[pos] = shard.read_block(int(block_in_shard[pos]), plan.seq_len)
                k += 1
            del shard  # release the memmap and .idx here
        np.savez(
            out_dir / f"{sp.name}.npz",
            tokens=arr,
            source_block=blocks,
            shard_index=shard_idx,
            block_in_shard=block_in_shard,
        )
        log(
            f"[heldout] {sp.name}: {blocks.size:,} blocks "
            f"({blocks.size * plan.seq_len / 1e6:.2f}M tokens, {opened:,} shards touched)"
            + (f" (of {n_total:,} withheld, first {cap:,} materialised)" if n_total > blocks.size else "")
            + f" -> {out_dir / (sp.name + '.npz')}"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--stage", choices=["download", "tokenize", "plan", "all"], default="all"
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="output root for builds")
    ap.add_argument("--raw-dir", default="", help="download dir for input files (default: HF cache)")
    ap.add_argument("--limit-docs", type=int, default=0, help="max documents per shard (testing)")
    ap.add_argument("--force", action="store_true", help="rebuild existing shards")
    ap.add_argument("--allow-design-only", action="store_true")
    ap.add_argument(
        "--workers", type=int, default=0,
        help="parallelism: processes for tokenize, threads for download; 0 = auto",
    )
    ap.add_argument("--node-rank", type=int, default=0, help="this node's rank (multi-node runs)")
    ap.add_argument("--n-nodes", type=int, default=1, help="number of nodes in a multi-node run")
    ap.add_argument(
        "--progress-interval", type=float, default=60.0,
        help="interval between progress lines in seconds (also written to JSONL)",
    )
    ap.add_argument(
        "--merge-inputs", action="store_true",
        help="merge the download stage's inputs.rank*.json into inputs.json and exit",
    )
    ap.add_argument(
        "--heldout-export-cap", type=int, default=DEFAULT_HELDOUT_EXPORT_CAP,
        help="max held-out blocks materialised per source; "
             "the amount withheld from training is still holdout_fraction",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.design_only and not args.allow_design_only:
        raise SystemExit(
            f"{args.config} has design_only = true; "
            "pass --allow-design-only to process it"
        )

    build_dir = Path(args.root) / cfg.name
    build_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    shutil.copyfile(cfg.source_path, build_dir / "config.toml")

    log(f"build_dir = {build_dir}")
    log(f"config digest = {cfg.digest()}")

    if args.merge_inputs:
        merge_inputs(build_dir, args.n_nodes)
        log("TOKENIZE_CORPUS_DONE")
        return 0

    if args.stage in ("download", "all"):
        stage_download(
            cfg, build_dir, raw_dir,
            workers=args.workers if args.workers > 0 else 16,
            node_rank=args.node_rank, n_nodes=args.n_nodes,
            progress_interval=args.progress_interval,
        )
    make_plan = args.stage in ("plan", "all")
    if args.stage in ("tokenize", "all"):
        stats = stage_tokenize(
            cfg, build_dir, raw_dir, limit_docs=args.limit_docs, force=args.force,
            workers=args.workers, node_rank=args.node_rank, n_nodes=args.n_nodes,
            progress_interval=args.progress_interval,
        )
        if stats["failed"]:
            log(f"[tokenize] {stats['failed']} files failed; not building the plan")
            return 1
        # On a single node, continue and build the plan. On multiple nodes each
        # node holds only its own shards, so rank 0 runs `--stage plan` once all
        # nodes have finished.
        make_plan = make_plan or args.n_nodes == 1
    if make_plan:
        entries = collect_shard_refs(cfg, build_dir)
        plan = build_plan(
            build_name=cfg.name,
            build_dir=build_dir,
            seq_len=cfg.seq_len,
            seed=cfg.seed,
            sources=entries,
            holdout_fraction=cfg.holdout_fraction,
            target_tokens=cfg.target_tokens,
            config_digest=cfg.digest(),
            tokenizer={
                "name": cfg.tokenizer.name,
                "path": cfg.tokenizer.path,
                "vocab_size": cfg.tokenizer.vocab_size,
                "eos_id": cfg.tokenizer.eos_id,
            },
        )
        path = plan.save()
        log(f"plan -> {path}")
        print(plan.summary(), flush=True)
        export_heldout(plan, build_dir, cap=args.heldout_export_cap)
    log("TOKENIZE_CORPUS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
