# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Build SFT data as 2048-token blocks with a clean prompt and a noised response.

Sources: Dolci Instruct SFT (`--source dolci`, chat messages) and TinyGSM
(`--source tinygsm`, question + code).

Output (`val.*` in the same format):

    <out>/train.tokens.bin   uint16 [n_blocks, seq_len]   token ids
    <out>/train.clean.bin    uint8  [n_blocks, seq_len]   1 = prompt (condition), 0 = response
    <out>/meta.json          shapes, template, statistics, checksum

The val split measures the SFT objective itself; the pretraining held-out CE
only checks that nothing broke.

Design choices:

1. The template uses plain tokens only. `<|im_start|>` (id 1) / `<|im_end|>`
   (id 2) practically never occur in the pretraining corpus (Dolma 3), so their
   embeddings are untrained. Only ``User: `` / ``Assistant:``, newlines and
   `<|endoftext|>` (id 0, the pretraining document separator) are used.

2. ``Assistant:`` belongs to the prompt (clean): at inference the cue for where
   the prefix ends must be on the conditioning side (like ``Answer:`` in an MCQ
   context). The response keeps its leading space, matching how the MCQ
   harness splits continuations (`" yes"`), so token boundaries agree.

3. Each segment is encoded separately, as the MCQ harness (`build_requests`)
   encodes context and continuation. Encoding the whole text at once lets BPE
   merge across the boundary, leaving the response tokens ill-defined.

4. By default there is no padding: examples are concatenated and cut into
   blocks (as in pretraining). The tail of an example that crosses a block
   boundary gives a block with no clean position, which matches the
   pretraining distribution. Pad tokens would be read by the bidirectional
   attention (the model has no padding mask). `--one-per-block` instead puts
   each example in its own block padded with EOS (see `pad_to_block`).

5. Examples longer than seq_len are truncated rather than dropped; only those
   whose remaining response is shorter than `--min-response` tokens are
   dropped. Dropping long reasoning traces (e.g. from OpenThoughts3) entirely
   would remove many tokens and shift the mixture.

6. Example order is shuffled globally (fixed seed). The parquet files are
   sorted by source, so packing them in order would show different domains
   early and late in the LR schedule.

Usage:

    python -m scripts.data.build_sft \\
        --repo <snapshot dir>/data --out $SWAFLOW_DATA_ROOT/dolci_sft_2048 \\
        --tokenizer $SWAFLOW_TOKENIZER_DIR --workers 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

__all__ = ["encode_example", "encode_segments", "render_segments",
           "render_tinygsm", "TEMPLATE", "TINYGSM_TEMPLATE", "SOURCES"]


#: Document separator; the same id as in pretraining (`scripts/data/tokenize_corpus.py`).
EOS_ID = 0

#: Template identifier (stored in meta.json; rename it whenever the template changes)
TEMPLATE = "plain-user-assistant-v1"

_PROMPT_ROLES = ("system", "user")


def render_segments(messages: list[dict]) -> list[tuple[str, int]] | None:
    """Turn one example into a list of `(text, is_clean)` segments.

    `is_clean = 1` marks prompt positions (held clean as the condition); `0`
    marks response positions (noised and scored).

    Returns None if the example is not used (tool calls, empty content, or no
    assistant turn).
    """
    out: list[tuple[str, int]] = []
    n_response = 0
    for m in messages:
        if m.get("function_calls") or m.get("functions"):
            return None  # tool calls have no plain-text form
        content = (m.get("content") or "").strip()
        if not content:
            return None
        role = m.get("role") or ""
        if role in _PROMPT_ROLES:
            out.append((f"User: {content}\n\n", 1))
        elif role == "assistant":
            out.append(("Assistant:", 1))
            out.append((f" {content}", 0))
            out.append(("\n\n", 0))
            n_response += 1
        else:
            return None  # do not silently include unknown roles
    if not out or n_response == 0:
        return None
    if out[0][1] != 1:
        return None  # an example that starts with assistant has no condition
    if out[-1][0] == "\n\n":
        out.pop()  # the trailing separator is replaced by EOS
    return out


#: TinyGSM template identifier (distinct from `plain-user-assistant-v1`).
TINYGSM_TEMPLATE = "tinygsm-question-code-v1"


def render_tinygsm(question: str, code: str) -> list[tuple[str, int]] | None:
    """Turn one TinyGSM row into `(text, is_clean)` segments.

    Same format as S-FLM's `dataloader.get_tiny_gsm_dataset`:
    ``question + "\n" + code``. There is no ``User:`` / ``Assistant:``
    wrapping, so the zero-shot evaluation prompt matches S-FLM and results are
    not confounded by a format difference.

    clean (prompt) = question + newline; response (noised, scored) = Python code.
    """
    q = (question or "").strip()
    c = (code or "").strip()
    if not q or not c:
        return None
    return [(q + "\n", 1), (c, 0)]


#: `--source` -> (parquet columns, template identifier, (dataset, license)).
SOURCES = {
    "dolci": (("messages",), TEMPLATE, ("allenai/Dolci-Instruct-SFT", "ODC-BY-1.0")),
    "tinygsm": (("question", "code"), TINYGSM_TEMPLATE,
                ("TinyGSM/TinyGSM", "MIT")),
}


def encode_example(
    messages: list[dict], tokenizer, *, seq_len: int, min_response: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """`render_segments` + `encode_segments` (for tests and one-off use)."""
    segs = render_segments(messages)
    if segs is None:
        return None
    return encode_segments(segs, tokenizer, seq_len=seq_len, min_response=min_response)


def encode_segments(
    segs: list[tuple[str, int]], tokenizer, *, seq_len: int, min_response: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Segments -> (tokens uint16, clean uint8).

    - each segment is encoded on its own, without special tokens (design choice 3)
    - EOS is appended (response side)
    - beyond `seq_len` the example is truncated, keeping the start; a truncated
      example whose remaining response is shorter than `min_response` gives
      None (design choice 5)

    `min_response` applies only to truncated examples. Naturally short answers
    (e.g. one-word answers to FLAN v2 classification tasks) are kept: MCQ
    answers have exactly that shape (4-12 tokens).
    """
    ids: list[int] = []
    clean: list[int] = []
    for text, is_clean in segs:
        enc = tokenizer.encode(text).ids
        ids.extend(enc)
        clean.extend([is_clean] * len(enc))
    ids.append(EOS_ID)
    clean.append(0)
    truncated = len(ids) > seq_len
    if truncated:
        ids = ids[:seq_len]
        clean = clean[:seq_len]
    n_resp = len(clean) - int(sum(clean))
    if n_resp < (min_response if truncated else 1):
        return None
    return (
        np.asarray(ids, dtype=np.uint16),
        np.asarray(clean, dtype=np.uint8),
    )


def pad_to_block(ids: np.ndarray, clean: np.ndarray, seq_len: int,
                 pad_id: int = EOS_ID) -> tuple[np.ndarray, np.ndarray] | None:
    """Put one example alone in a block of `seq_len` tokens.

    Same rule as S-FLM's `pad_and_mask` (`dataloader.get_tiny_gsm_dataset`
    with `wrap=False` and `train_on_pad=True`):

        [prompt ....][response ....][pad pad ... pad]
        clean=1       clean=0        clean=0
        no loss       loss           loss

    Padding positions are scored, so the model learns to fill the rest of the
    sequence once the answer is complete.

    Examples longer than `seq_len` give None (dropped; S-FLM's
    `filter_too_long=True`). They are not truncated, since cut-off code is not a
    useful target.

    The pad token is EOS (id 0) because the tokenizer has no pad_token. S-FLM
    uses `tokenizer.pad_token_id`; the rule (fill after the answer with one
    fixed token) is the same.
    """
    n = len(ids)
    if n > seq_len:
        return None
    out_ids = np.full(seq_len, pad_id, dtype=np.uint16)
    out_cln = np.zeros(seq_len, dtype=np.uint8)
    out_ids[:n] = ids
    out_cln[:n] = clean
    return out_ids, out_cln


# --------------------------------------------------------------------------
# phase 1: one parquet file per worker
# --------------------------------------------------------------------------
def _encode_file(job: tuple[str, str, str, int, int, int, str, bool]) -> dict:
    path, out_dir, tok_file, seq_len, min_response, part, source, one_per_block = job
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    columns = list(SOURCES[source][0])
    tok = Tokenizer.from_file(tok_file)
    out = Path(out_dir)
    tok_buf: list[np.ndarray] = []
    cln_buf: list[np.ndarray] = []
    lens: list[int] = []
    stats = Counter()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=1000, columns=columns):
        for row in batch.to_pylist():
            stats["rows"] += 1
            if source == "tinygsm":
                segs = render_tinygsm(row["question"], row["code"])
            else:
                segs = render_segments(row["messages"])
            if segs is None:
                stats["dropped"] += 1
                stats["dropped_unrenderable"] += 1
                continue
            enc = encode_segments(
                # No truncation with one_per_block (long examples are dropped instead).
                segs, tok, seq_len=10**9 if one_per_block else seq_len,
                min_response=min_response,
            )
            if enc is None:
                stats["dropped"] += 1
                stats["dropped_short_response"] += 1
                continue
            ids, clean = enc
            if one_per_block:
                # One example per block; examples that are too long are dropped
                blk = pad_to_block(ids, clean, seq_len)
                if blk is None:
                    stats["dropped"] += 1
                    stats["dropped_too_long"] += 1
                    continue
                ids, clean = blk
                stats["padded_tokens"] += seq_len - len(enc[0])
            elif len(ids) == seq_len:
                stats["truncated"] += 1
            tok_buf.append(ids)
            cln_buf.append(clean)
            lens.append(len(ids))
            stats["tokens"] += len(ids)
            stats["response_tokens"] += len(clean) - int(clean.sum())
    # A part may keep no examples at all (e.g. a parquet file of tool-use data);
    # write empty files so that phase 2 sees zero examples.
    empty16, empty8 = np.empty(0, np.uint16), np.empty(0, np.uint8)
    (np.concatenate(tok_buf) if tok_buf else empty16).tofile(
        out / f"part-{part:03d}.tokens.u16")
    (np.concatenate(cln_buf) if cln_buf else empty8).tofile(
        out / f"part-{part:03d}.clean.u8")
    np.asarray(lens, dtype=np.int64).tofile(out / f"part-{part:03d}.lengths.i64")
    return {"part": part, "path": path, **stats}


# --------------------------------------------------------------------------
# phase 2: shuffle and cut into blocks
# --------------------------------------------------------------------------
def _pack(parts_dir: Path, out: Path, seq_len: int, seed: int, val_blocks: int,
          repeats: int = 1) -> dict:
    """Shuffle the encoded examples and pack them into blocks.

    `repeats > 1` packs the corpus again with a different shuffle per pass.
    Repeating on the data side rather than looping the dataloader
    (`SFTBlockDataset(infinite=True)`) gives each pass a different packing, so
    blocks do not repeat verbatim, and needs no change to the training code.
    """
    stems = [p.name[: -len(".lengths.i64")] for p in sorted(parts_dir.glob("part-*.lengths.i64"))]
    lengths = [np.fromfile(parts_dir / f"{s}.lengths.i64", dtype=np.int64) for s in stems]
    tokens = [np.fromfile(parts_dir / f"{s}.tokens.u16", dtype=np.uint16) for s in stems]
    cleans = [np.fromfile(parts_dir / f"{s}.clean.u8", dtype=np.uint8) for s in stems]
    offsets = [np.concatenate([[0], np.cumsum(l)]) for l in lengths]
    index = np.concatenate(
        [
            np.stack([np.full(len(l), i, dtype=np.int64), np.arange(len(l))], axis=1)
            for i, l in enumerate(lengths)
        ]
    )
    parts = []
    for r in range(max(1, repeats)):
        idx = index.copy()
        np.random.default_rng(seed + r).shuffle(idx)
        parts.append(idx)
    index = np.concatenate(parts)

    total = int(sum(int(l.sum()) for l in lengths)) * max(1, repeats)
    n_blocks = total // seq_len
    tok_out = np.empty(n_blocks * seq_len, dtype=np.uint16)
    cln_out = np.empty(n_blocks * seq_len, dtype=np.uint8)
    pos = 0
    limit = n_blocks * seq_len
    for p, e in index:
        s, t = offsets[p][e], offsets[p][e + 1]
        n = min(int(t - s), limit - pos)
        if n <= 0:
            break
        tok_out[pos : pos + n] = tokens[p][s : s + n]
        cln_out[pos : pos + n] = cleans[p][s : s + n]
        pos += n
    tok_out = tok_out.reshape(n_blocks, seq_len)
    cln_out = cln_out.reshape(n_blocks, seq_len)

    digest.update(tok_out[:n_train][:: max(n_train // 512, 1)].tobytes())
    return {
        "n_blocks": int(n_blocks),
        "n_train_blocks": int(n_train),
        "n_val_blocks": int(n_val),
        "packed_tokens": int(pos),
        "dropped_tail_tokens": int(total - n_blocks * seq_len),
        "repeats": int(max(1, repeats)),
        "clean_fraction": float(cln_out[:n_train].mean()),
        "blocks_without_clean": int((cln_out[:n_train].sum(axis=1) == 0).sum()),
        "sample_checksum": digest.hexdigest(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="directory containing the parquet files")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer",
        default=os.environ.get("SWAFLOW_TOKENIZER_DIR", ""),
        help="directory containing tokenizer.json",
    )
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--min-response", type=int, default=32)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--val-blocks", type=int, default=2048)
    ap.add_argument("--limit-files", type=int, default=0, help="for quick tests")
    ap.add_argument("--source", default="dolci", choices=sorted(SOURCES),
                    help="parquet columns and template (dolci = messages / "
                         "tinygsm = question+code)")
    ap.add_argument("--one-per-block", action="store_true",
                    help="put each example in its own padded block "
                         "(as S-FLM's wrap=False + train_on_pad=True); "
                         "examples that are too long are dropped")
    ap.add_argument("--repeats", type=int, default=1,
                    help="number of passes over the corpus "
                         "(a new shuffle per pass)")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args(argv)

    tok_file = str(Path(args.tokenizer) / "tokenizer.json")
    if not Path(tok_file).is_file():
        raise SystemExit(f"tokenizer.json not found: {tok_file}")
    files = sorted(Path(args.repo).glob("*.parquet"))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"no parquet files found: {args.repo}")

    out = Path(args.out)
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs = [
        (str(f), str(parts_dir), tok_file, args.seq_len, args.min_response, i,
         args.source, args.one_per_block)
        for i, f in enumerate(files)
    ]
    stats: list[dict] = []
    # The Rust tokenizer is parallel already; use one thread per worker process.
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for s in ex.map(_encode_file, jobs):
            stats.append(s)
            print(
                f"[build_sft] part {s['part']:3d}: {s.get('rows', 0):,} rows / "
                f"{s.get('tokens', 0) / 1e6:.1f}M tokens "
                f"(dropped {s.get('dropped', 0):,}, truncated {s.get('truncated', 0):,})",
                flush=True,
            )
    t_enc = time.time() - t0

    packed = _pack(parts_dir, out, args.seq_len, args.seed, args.val_blocks,
                   repeats=args.repeats)
    totals = Counter()
    for s in stats:
        for k in ("rows", "dropped", "dropped_unrenderable", "dropped_short_response",
                  "dropped_too_long", "truncated", "tokens", "response_tokens",
                  "padded_tokens"):
            totals[k] += s.get(k, 0)

    meta = {
        "build_name": out.name,
        "source": SOURCES[args.source][2][0],
        "license": SOURCES[args.source][2][1],
        "template": SOURCES[args.source][1],
        "seq_len": args.seq_len,
        "min_response": args.min_response,
        "one_per_block": bool(args.one_per_block),
        "seed": args.seed,
        "eos_id": EOS_ID,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "files": [f.name for f in files],
        "totals": dict(totals),
        **packed,
        "encode_sec": round(t_enc, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    if not args.keep_parts:
        for p in parts_dir.glob("part-*"):
            p.unlink()
        parts_dir.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
