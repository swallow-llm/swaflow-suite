#!/usr/bin/env python
# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc EMA-K weight averages from saved checkpoints.

This is a post-hoc sparse-checkpoint approximation, not an exact reconstruction of the
online EMA: only the checkpoints saved every 20B tokens are available. EMA-K follows
"Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging"
(main setting EMA-25), whose online update at optimizer step t is

    ema_t = β_t · ema_{t-1} + (1 − β_t) · θ_t,        β_t = 2^(−K/t).

Between consecutive saved checkpoints (training age T_prev < T_cur) the updates are
folded into one:

    β_interval = Π_{t=T_prev+1}^{T_cur} 2^(−K/t) ≈ (T_prev / T_cur) ** (K · ln2)
    ema_0 = θ(T_0),   ema_i = β_interval,i · ema_{i−1} + (1 − β_interval,i) · θ(T_i)
    ema_N = Σ_i w_i θ(T_i),  w_0 = (T_0/T_N)^(K·ln2),  w_i = (1 − β_interval,i) · (T_i/T_N)^(K·ln2)

Differences from the online EMA (also recorded in the output metadata): the online
weight of the iterates in (T_{i−1}, T_i] goes to θ(T_i); the harmonic sum over an
interval is replaced by ln(T_cur/T_prev) (error O(1/T), shown by `--dry-run`); θ(T_0)
absorbs the weight of all earlier iterates; training age is counted in global tokens
(the global batch is constant, so ratios equal optimizer-step ratios).

Inputs are torchtitan DCP checkpoints (`<dump>/checkpoint/step-N/`). Only the model's
floating-point tensors are averaged (fp32, untied embeddings, no buffers in the state
dict); optimizer, scheduler, dataloader and train_state are dropped. Token positions
are `train_state.step` × tokens/step, with tokens/step = global_batch_size × seq_len
from `<dump>/run_order.json`; `train_state.ntokens_seen` counts per rank and is only
cross-checked. A run whose first checkpoint is a symlink into another run (a branch,
such as the 3B clean-prefix mid-training run (3T->4T)) is followed back: steps before
the branch point come from the parent run.

Output has the layout of a run dump, so `scripts/eval/` reads it directly:

    <output-dir>/<name>/checkpoint/step-<final>/   DCP with model keys only (fp32 by default)
    <output-dir>/<name>/posthoc_ema.json           approximation, inputs and checks

    python scripts/posthoc_ema.py --checkpoint-root $SWAFLOW_OUTPUTS/<run> \\
        --target-tokens 4000000000000 --checkpoint-interval-tokens 20000000000 \\
        --ema-k 12.5 25 50 100 --output-dir $SWAFLOW_OUTPUTS --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:  # so that scripts.eval imports under `python scripts/posthoc_ema.py`
    sys.path.insert(0, str(_REPO))

from scripts.eval.loader import checkpoint_steps  # noqa: E402

# Harmless notice emitted by every single-process DCP read/write.
warnings.filterwarnings(
    "ignore", message="torch.distributed is disabled, unavailable or uninitialized"
)

APPROXIMATION = "post-hoc sparse-checkpoint approximation"
APPROXIMATION_NOTE = (
    "Post-hoc sparse-checkpoint approximation of EMA-K; NOT an exact reconstruction "
    "of the online EMA. The optimizer iterates between saved checkpoints are not "
    "available, so all online-EMA weight that falls in (T_prev, T_cur] is placed on "
    "the checkpoint at T_cur, the per-step product of 2^(-K/t) over the interval is "
    "replaced by (T_prev/T_cur)^(K ln 2), and the weight of iterates before the first "
    "used checkpoint is absorbed by that checkpoint."
)
GENERATOR = "scripts/posthoc_ema.py"
METADATA_NAME = "posthoc_ema.json"
DEFAULT_KS = (12.5, 25.0, 50.0, 100.0)
ACCUMULATOR_DTYPE = torch.float32

#: Non-model state written by torchtitan's CheckpointManager (OPTIMIZER / LR_SCHEDULER /
#: DATALOADER / TRAIN_STATE in `torchtitan/components/checkpoint.py`); neither
#: averaged nor written out.
NON_MODEL_PREFIXES = ("optimizer", "lr_scheduler", "dataloader", "train_state")

_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}


def _log(msg: str) -> None:
    print(f"[posthoc-ema {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==========================================================================
# EMA coefficients (pure functions, tested directly)
# ==========================================================================
def ema_power(k: float) -> float:
    """β_interval = (T_prev/T_cur) ** ema_power(K)."""
    return float(k) * math.log(2.0)


def interval_coefficients(tokens: Sequence[int], k: float) -> list[tuple[float, float]]:
    """(β_interval,i, 1 − β_interval,i) for update i; i = 0 is the initialisation, (0, 1).

    1 − β is computed with expm1 to avoid cancellation when β ≈ 1.
    """
    p = ema_power(k)
    out = [(0.0, 1.0)]
    for i in range(1, len(tokens)):
        if not tokens[i] > tokens[i - 1] > 0:
            raise ValueError(f"token positions are not strictly increasing: {tokens[i - 1]} -> {tokens[i]}")
        x = p * math.log(tokens[i - 1] / tokens[i])
        out.append((math.exp(x), -math.expm1(x)))
    return out


def final_weights(tokens: Sequence[int], k: float) -> list[float]:
    """Closed-form weights w_i of ema_N = Σ w_i θ(T_i); Σ w_i = 1."""
    p = ema_power(k)
    tn = tokens[-1]
    coefs = interval_coefficients(tokens, k)
    w = [(tokens[0] / tn) ** p]
    for i in range(1, len(tokens)):
        w.append(coefs[i][1] * (tokens[i] / tn) ** p)
    return w


def _harmonic(n: int) -> float:
    """H(n) = Σ_{t=1}^{n} 1/t."""
    if n < 64:
        return math.fsum(1.0 / t for t in range(1, n + 1))
    inv = 1.0 / n
    return (math.log(n) + 0.5772156649015329 + 0.5 * inv
            - inv * inv / 12.0 + inv ** 4 / 120.0)


def online_interval_beta(step_prev: int, step_cur: int, k: float) -> float:
    """Exact online-EMA product Π_{t=step_prev+1}^{step_cur} 2^(−K/t) over optimizer steps.

    Only reported, to show the error of the approximation; not used for averaging.
    """
    if step_prev >= 64:
        # H(b) − H(a) by subtraction cancels; fold the leading term into log1p
        a, b = step_prev, step_cur
        d = (math.log1p((b - a) / a) + 0.5 * (1.0 / b - 1.0 / a)
             - (1.0 / b**2 - 1.0 / a**2) / 12.0 + (1.0 / b**4 - 1.0 / a**4) / 120.0)
    else:
        d = _harmonic(step_cur) - _harmonic(step_prev)
    return 2.0 ** (-float(k) * d)


def update_accumulator(
    acc: dict[str, torch.Tensor],
    theta: dict[str, torch.Tensor],
    keys: Iterable[str],
    beta: float,
    one_minus_beta: float,
    *,
    first: bool,
) -> None:
    """acc ← β·acc + (1−β)·θ in place (acc is fp32); acc ← θ if `first`."""
    with torch.no_grad():
        for key in keys:
            if first:
                acc[key].copy_(theta[key])
            else:
                acc[key].mul_(beta).add_(theta[key], alpha=one_minus_beta)


# ==========================================================================
# Checkpoint discovery (lineage / token positions)
# ==========================================================================
@dataclasses.dataclass
class RunRoot:
    ckpt_dir: Path
    dump: Path | None
    via: str
    taken: list[int] = dataclasses.field(default_factory=list)
    dropped: list[int] = dataclasses.field(default_factory=list)
    branch_step: int | None = None
    dp_world_size: int | None = None


@dataclasses.dataclass
class Source:
    root_index: int
    path: str
    realpath: str
    dir_step: int
    step: int = -1
    tokens: int = -1
    ntokens_seen: int | None = None
    ntokens_seen_consistent: bool | None = None


def _as_root(path: Path, via: str) -> RunRoot:
    path = Path(path)
    if (path / "checkpoint").is_dir():
        return RunRoot(path / "checkpoint", path, via)
    if path.is_dir() and any(path.glob("step-*")):
        dump = path.parent if path.name == "checkpoint" else None
        return RunRoot(path, dump, via)
    raise SystemExit(
        f"{path}: no checkpoint found (pass a run dump or its checkpoint/ directory)"
    )


def _steps_of(root: RunRoot) -> list[int]:
    # Steps without `.metadata` (partially written) are skipped, as in scripts.eval.loader
    return checkpoint_steps(root.ckpt_dir.parent, folder=root.ckpt_dir.name)


def resolve_roots(paths: Sequence[str | Path], follow_branch: bool = True) -> list[RunRoot]:
    """Order runs along the lineage, oldest first.

    With `follow_branch`, while the first checkpoint of the oldest run is a symlink
    into another run's checkpoint (the run branched from there), the parent run is
    prepended.
    """
    roots = [_as_root(Path(p), "cli") for p in paths]
    if follow_branch:
        seen = {r.ckpt_dir.resolve() for r in roots}
        for _ in range(16):
            first = roots[0]
            steps = _steps_of(first)
            if not steps:
                break
            head = first.ckpt_dir / f"step-{steps[0]}"
            if not head.is_symlink():
                break
            parent = head.resolve().parent
            if parent in seen:
                break
            roots.insert(0, _as_root(parent, f"branch symlink {head} -> {head.resolve()}"))
            seen.add(parent)
    return roots


def lineage_sources(roots: list[RunRoot]) -> list[Source]:
    """Checkpoints along the lineage, in increasing step order.

    The newest run contributes all its steps; each older run only the steps before
    the first step of the next run (later steps belong to another branch).
    """
    by_step: dict[int, Source] = {}
    upper: int | None = None
    for idx in range(len(roots) - 1, -1, -1):
        root = roots[idx]
        steps = _steps_of(root)
        if not steps:
            raise SystemExit(f"{root.ckpt_dir} has no complete checkpoint")
        root.taken = [s for s in steps if upper is None or s < upper]
        root.dropped = [s for s in steps if upper is not None and s >= upper]
        root.branch_step = upper
        for s in root.taken:
            p = root.ckpt_dir / f"step-{s}"
            by_step[s] = Source(idx, str(p), str(p.resolve()), s)
        if root.taken:
            upper = min(root.taken) if upper is None else min(upper, min(root.taken))
    return [by_step[s] for s in sorted(by_step)]


def resolve_tokens_per_step(roots: list[RunRoot], cli_value: int | None) -> tuple[int, str]:
    found: dict[str, int] = {}
    for root in roots:
        ro_path = root.dump / "run_order.json" if root.dump else None
        if ro_path is None or not ro_path.exists():
            continue
        ro = json.loads(ro_path.read_text())
        found[str(ro_path)] = int(ro["global_batch_size"]) * int(ro["seq_len"])
        if ro.get("dp_world_size"):
            root.dp_world_size = int(ro["dp_world_size"])
    values = set(found.values())
    if len(values) > 1:
        raise SystemExit(f"tokens/step differs between runs: {found} (the approximation assumes it is constant)")
    if cli_value:
        if values and cli_value not in values:
            raise SystemExit(f"--tokens-per-step {cli_value} and run_order.json {found} disagree")
        return int(cli_value), "--tokens-per-step"
    if not values:
        raise SystemExit("no run_order.json found; pass --tokens-per-step")
    return values.pop(), "global_batch_size * seq_len in " + ", ".join(sorted(found))


# ==========================================================================
# DCP loading (model keys only, split across workers)
# ==========================================================================
class _CachedReader(FileSystemReader):
    """Reuses already parsed Metadata (parsing .metadata takes ~2.4 s at 3B)."""

    def __init__(self, path: str, metadata: Metadata):
        super().__init__(path)
        self._cached_metadata = metadata

    def read_metadata(self, *args, **kwargs) -> Metadata:  # noqa: D102
        return self._cached_metadata


def _prune_metadata(md: Metadata, keys: Iterable[str]) -> Metadata:
    keys = set(keys)
    planner = md.planner_data
    if isinstance(planner, dict):
        planner = {k: v for k, v in planner.items() if k in keys}
    return dataclasses.replace(
        md,
        state_dict_metadata={k: v for k, v in md.state_dict_metadata.items() if k in keys},
        storage_data={i: v for i, v in md.storage_data.items() if i.fqn in keys},
        planner_data=planner,
    )


def _model_signature(md: Metadata) -> tuple[dict[str, tuple[tuple[int, ...], str]], list[str], list[str]]:
    sig: dict[str, tuple[tuple[int, ...], str]] = {}
    other: set[str] = set()
    non_tensor: list[str] = []
    for key, value in md.state_dict_metadata.items():
        top = key.split(".", 1)[0]
        if top in NON_MODEL_PREFIXES:
            other.add(top)
            continue
        if not isinstance(value, TensorStorageMetadata):
            non_tensor.append(key)
            continue
        sig[key] = (
            tuple(int(x) for x in value.size),
            str(value.properties.dtype).replace("torch.", ""),
        )
    return sig, sorted(other), sorted(non_tensor)


def _scan_checkpoint(path: str) -> dict[str, Any]:
    """Read metadata and train_state only (no weights)."""
    md = FileSystemReader(path).read_metadata()
    sig, other, non_tensor = _model_signature(md)
    train_state: dict[str, int] = {}
    want = [k for k in ("train_state.step", "train_state.ntokens_seen")
            if k in md.state_dict_metadata]
    if want:
        sd: dict[str, Any] = {k: 0 for k in want}
        dcp.load(sd, storage_reader=_CachedReader(path, md), no_dist=True)
        train_state = {k.split(".", 1)[1]: int(v) for k, v in sd.items()}
    return {"path": path, "sig": sig, "other_state": other,
            "non_tensor_model_entries": non_tensor, "train_state": train_state}


_STAGING: dict[str, torch.Tensor] = {}


def _worker_init() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings(
        "ignore", message="torch.distributed is disabled, unavailable or uninitialized"
    )


def _load_group(task: tuple[str, Metadata, list[str]]) -> float:
    """Load this worker's keys into the shared-memory `_STAGING` inherited via fork."""
    path, md, keys = task
    t0 = time.time()
    dcp.load({k: _STAGING[k] for k in keys},
             storage_reader=_CachedReader(path, md), no_dist=True)
    return time.time() - t0


def _read_group_metadata(path: str, groups: list[list[str]]) -> list[Metadata]:
    md = FileSystemReader(path).read_metadata()
    return [_prune_metadata(md, g) for g in groups]


def _balance_groups(sizes: dict[str, int], n: int) -> list[list[str]]:
    n = max(1, min(n, len(sizes)))
    groups: list[list[str]] = [[] for _ in range(n)]
    load = [0] * n
    for key in sorted(sizes, key=lambda k: (-sizes[k], k)):
        i = load.index(min(load))
        groups[i].append(key)
        load[i] += sizes[key]
    return [sorted(g) for g in groups if g]


class ParallelLoader:
    """Loads the model keys into shared memory with n worker processes.

    Staging is allocated before forking so that workers inherit `_STAGING`.
    The next checkpoint's `.metadata` is parsed ahead of time.
    """

    def __init__(self, sig: dict[str, tuple[tuple[int, ...], str]], workers: int):
        global _STAGING
        _STAGING = {
            k: torch.empty(shape, dtype=getattr(torch, dt)).share_memory_()
            for k, (shape, dt) in sig.items()
        }
        self.staging = _STAGING
        self.groups = _balance_groups({k: t.numel() for k, t in _STAGING.items()}, workers)
        self.pool = mp.get_context("fork").Pool(len(self.groups) + 1, initializer=_worker_init)
        self._prefetch: dict[str, Any] = {}

    def prefetch(self, path: str) -> None:
        if path not in self._prefetch:
            self._prefetch[path] = self.pool.apply_async(
                _read_group_metadata, (path, self.groups))

    def load(self, path: str) -> float:
        self.prefetch(path)
        mds = self._prefetch.pop(path).get()
        t0 = time.time()
        self.pool.map(_load_group, [(path, md, g) for md, g in zip(mds, self.groups)],
                      chunksize=1)
        return time.time() - t0

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


# ==========================================================================
# Output
# ==========================================================================
def token_label(tokens: int) -> str:
    for unit, div in (("t", 10**12), ("b", 10**9), ("m", 10**6), ("k", 10**3)):
        if tokens >= div:
            return f"{tokens / div:g}{unit}"
    return str(tokens)


def k_label(k: float) -> str:
    return f"{k:g}"


def _parse_tokens(text: str) -> int:
    text = str(text).replace("_", "")
    value = float(text) if any(c in text for c in ".eE") else int(text)
    if value != int(value):
        raise argparse.ArgumentTypeError(f"token count must be an integer: {text}")
    return int(value)


def _is_ours(run_dir: Path) -> bool:
    marker = run_dir / METADATA_NAME
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text()).get("generator") == GENERATOR
    except (OSError, ValueError):
        return False


def check_output_slot(run_dir: Path, overwrite: bool) -> None:
    """Check, before any heavy work, whether an existing output may be replaced."""
    if not run_dir.exists():
        return
    if not overwrite:
        raise SystemExit(f"{run_dir} already exists (pass --overwrite to replace it)")
    if not _is_ours(run_dir):
        raise SystemExit(
            f"{run_dir} has no {METADATA_NAME}(generator={GENERATOR}); "
            "it is not an output of this script, so it is left untouched"
        )
    extra = sorted(p.name for p in run_dir.iterdir()
                   if p.name not in (METADATA_NAME, "checkpoint"))
    if extra:
        raise SystemExit(
            f"{run_dir} holds files not written by this script: {extra}. "
            "Replacing the weights would make them inconsistent; remove them first"
        )


def check_output_is_not_a_source(run_dir: Path, roots: list[RunRoot], sources: list[Source]) -> None:
    out = run_dir.resolve()
    guarded = [r.ckpt_dir.resolve() for r in roots]
    guarded += [r.dump.resolve() for r in roots if r.dump is not None]
    for g in guarded:
        if out == g or g in out.parents or out in g.parents:
            raise SystemExit(f"output {out} overlaps input {g}; input checkpoints are never written")
    for s in sources:
        if out in Path(s.realpath).parents:
            raise SystemExit(f"output {out} has input {s.realpath} under it")


def _git_info() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "script_status": run("git", "status", "--porcelain", "--", GENERATOR),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ==========================================================================
# Plan (shared by --dry-run and the actual run)
# ==========================================================================
@dataclasses.dataclass
class Plan:
    roots: list[RunRoot]
    all_sources: list[Source]
    sources: list[Source]
    offgrid_skipped: list[int]
    irregular_gaps: list[dict[str, int]]
    missing_grid_steps: list[int]
    tokens_per_step: int
    tokens_per_step_source: str
    interval_steps: int
    horizon_tokens: int
    sig: dict[str, tuple[tuple[int, ...], str]]
    float_keys: list[str]
    other_keys: list[str]
    other_state: list[str]


def build_plan(args: argparse.Namespace) -> Plan:
    roots = resolve_roots(args.checkpoint_root, follow_branch=not args.no_follow_branch)
    all_sources = lineage_sources(roots)
    tps, tps_src = resolve_tokens_per_step(roots, args.tokens_per_step)
    interval_steps = round(args.checkpoint_interval_tokens / tps)
    if interval_steps < 1 or abs(interval_steps * tps - args.checkpoint_interval_tokens) >= tps:
        raise SystemExit(
            f"--checkpoint-interval-tokens {args.checkpoint_interval_tokens} is not "
            f"a whole number of steps (tokens/step {tps})")

    horizon = args.end_tokens if args.end_tokens is not None else args.target_tokens
    for s in all_sources:  # provisional token position from the directory name, for range selection
        s.tokens = s.dir_step * tps
    upto = [s for s in all_sources if s.tokens <= horizon]
    if not upto:
        raise SystemExit(f"{horizon} tokens: no checkpoint at or before this position")
    final = upto[-1]
    # The target checkpoint is the last one at or before the target and must lie within half
    # an interval of it (grid points can fall just short: 4T is step 476,800 = 3.99969T).
    tolerance = max(tps, interval_steps * tps // 2)
    if args.end_tokens is None and args.target_tokens - final.tokens >= tolerance:
        raise SystemExit(
            f"target {args.target_tokens}: no checkpoint (the last one at or before it is step "
            f"{final.dir_step} = {final.tokens} tokens; tolerance is under {tolerance} tokens). "
            "Use --end-tokens to stop earlier")
    start = args.start_tokens or 0
    offgrid = [s.dir_step for s in upto
               if s.dir_step % interval_steps and s is not final and s.tokens >= start]
    chosen = [s for s in upto if s.tokens >= start
              and (s is final or args.include_offgrid or s.dir_step % interval_steps == 0)]
    if len(chosen) < 2:
        raise SystemExit(f"only {len(chosen)} checkpoints to average")

    # ---- metadata scan (keys / shapes / dtypes must agree; train_state) ----
    _log(f"scanning metadata: {len(chosen)} checkpoints / {args.scan_workers} workers")
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.scan_workers,
                                mp_context=mp.get_context("fork"),
                                initializer=_worker_init) as ex:
        scans = list(ex.map(_scan_checkpoint, [s.path for s in chosen]))
    _log(f"scan done {time.time() - t0:.1f}s")

    ref = scans[-1]
    if not ref["sig"]:
        raise SystemExit(f"{final.path} has no model tensors")
    problems = []
    for s, scan in zip(chosen, scans):
        if scan["non_tensor_model_entries"]:
            problems.append(f"{s.path}: non-tensor model entries {scan['non_tensor_model_entries'][:3]}")
        if scan["sig"] != ref["sig"]:
            a, b = set(scan["sig"]), set(ref["sig"])
            diff = sorted(k for k in a & b if scan["sig"][k] != ref["sig"][k])
            problems.append(
                f"{s.path}: keys/shapes/dtypes differ from the final checkpoint "
                f"(missing {sorted(b - a)[:3]} / extra {sorted(a - b)[:3]} / mismatched "
                f"{[(k, scan['sig'][k], ref['sig'][k]) for k in diff[:3]]})")
        ts = scan["train_state"]
        s.step = ts.get("step", s.dir_step)
        if s.step != s.dir_step:
            problems.append(f"{s.path}: train_state.step = {s.step} differs from the directory name")
        s.tokens = s.step * tps
        s.ntokens_seen = ts.get("ntokens_seen")
        dp = roots[s.root_index].dp_world_size
        if s.ntokens_seen is not None and dp:
            s.ntokens_seen_consistent = s.ntokens_seen * dp == s.tokens
    if problems:
        raise SystemExit("input checkpoint check failed:\n  " + "\n  ".join(problems[:20]))
    inconsistent = [s.step for s in chosen if s.ntokens_seen_consistent is False]
    if inconsistent:
        _log(f"warning: ntokens_seen × dp_world_size != step × tokens/step at steps: "
             f"{inconsistent[:10]} (ntokens_seen is rank-local; dp may have changed mid-run)")

    steps = [s.step for s in chosen]
    gaps = [{"from": a, "to": b, "steps": b - a}
            for a, b in zip(steps, steps[1:]) if b - a != interval_steps]
    grid = range(steps[0] - steps[0] % interval_steps + interval_steps, steps[-1], interval_steps)
    have = set(steps)
    missing = [g for g in grid if g not in have and g > steps[0]]
    sig = ref["sig"]
    float_keys = sorted(k for k, (_, dt) in sig.items() if dt in _FLOAT_DTYPES)
    other_keys = sorted(k for k in sig if k not in float_keys)
    return Plan(roots, all_sources, chosen, offgrid, gaps, missing, tps, tps_src,
                interval_steps, horizon, sig, float_keys, other_keys, ref["other_state"])


def weight_summary(plan: Plan, k: float) -> dict[str, Any]:
    tokens = [s.tokens for s in plan.sources]
    w = final_weights(tokens, k)
    coefs = interval_coefficients(tokens, k)
    exact = [None] + [online_interval_beta(a.step, b.step, k)
                      for a, b in zip(plan.sources, plan.sources[1:])]
    beta_err = max(abs(coefs[i][0] - exact[i]) for i in range(1, len(tokens)))
    tail = 0.0
    cover: dict[str, int] = {}
    earliest: dict[str, int] = {}
    for n, i in enumerate(range(len(w) - 1, -1, -1), start=1):
        tail += w[i]
        for q in (0.5, 0.9, 0.99, 0.999):
            if f"{q:g}" not in cover and tail >= q:
                cover[f"{q:g}"] = n
                earliest[f"{q:g}"] = tokens[i]
    per_root: dict[str, float] = {}
    for s, wi in zip(plan.sources, w):
        key = str(plan.roots[s.root_index].dump or plan.roots[s.root_index].ckpt_dir)
        per_root[key] = per_root.get(key, 0.0) + wi
    half_life = 1.0 - 0.5 ** (1.0 / ema_power(k))  # final fraction of the horizon holding half the weight
    return {
        "K": k,
        "weights": w,
        "betas": [c[0] for c in coefs],
        "online_betas_exact_in_steps": exact,
        "max_abs_beta_minus_exact_online_product": beta_err,
        "sum_weights": math.fsum(w),
        "init_weight": w[0],
        "final_checkpoint_weight": w[-1],
        "effective_num_checkpoints": 1.0 / math.fsum(x * x for x in w),
        "center_of_mass_tokens": math.fsum(wi * t for wi, t in zip(w, tokens)),
        "num_checkpoints_covering_mass": cover,
        "earliest_tokens_covering_mass": earliest,
        "mass_per_run": per_root,
        "half_mass_fraction_of_horizon": half_life,
    }


def print_plan(plan: Plan, ks: Sequence[float], outputs: dict[float, Path], final_step: int) -> None:
    print("\n==== post-hoc EMA-K (" + APPROXIMATION + ") ====")
    print(f"tokens/step        : {plan.tokens_per_step:,}  ({plan.tokens_per_step_source})")
    print(f"interval           : {plan.interval_steps} steps = "
          f"{plan.interval_steps * plan.tokens_per_step:,} tokens")
    print(f"horizon            : {plan.horizon_tokens:,} tokens")
    print("lineage (oldest first):")
    for i, r in enumerate(plan.roots):
        taken = [s.dir_step for s in plan.all_sources if s.root_index == i]
        used = [s.step for s in plan.sources if s.root_index == i]
        rng = f"{taken[0]}..{taken[-1]}" if taken else "-"
        print(f"  [{i}] {r.ckpt_dir}  ({r.via})")
        print(f"      lineage steps {rng} ({len(taken)} points, used {len(used)})")
        if r.dropped:
            print(f"      step >= {r.branch_step} (start of the next run): {len(r.dropped)} points "
                  f"({r.dropped[0]}..{r.dropped[-1]}) not taken from this run"
                  " (they come from the next run; at the branch point a symlink is the same file)")
    print(f"model tensors      : {len(plan.sig)}(EMA {len(plan.float_keys)} / "
          f"copied from final ckpt {len(plan.other_keys)}) "
          f"numel {sum(math.prod(s) for s, _ in plan.sig.values()):,}")
    print(f"dtypes             : {sorted({dt for _, dt in plan.sig.values()})}  "
          f"(accumulator {str(ACCUMULATOR_DTYPE).replace('torch.', '')})")
    print(f"not averaged       : {plan.other_state}")
    print("keys/shapes/dtypes : identical to the final checkpoint in every used checkpoint")
    nts = [s.ntokens_seen_consistent for s in plan.sources]
    print(f"ntokens_seen check : match {nts.count(True)} / mismatch {nts.count(False)} / "
          f"unchecked {nts.count(None)} (rank-local × dp_world_size)")
    if plan.offgrid_skipped:
        print(f"off-grid, skipped  : {plan.offgrid_skipped}")
    if plan.irregular_gaps:
        print(f"irregular gaps     : {plan.irregular_gaps}")
    if plan.missing_grid_steps:
        print(f"warning: grid gaps : {plan.missing_grid_steps[:20]}")
    summaries = {k: weight_summary(plan, k) for k in ks}
    head = " ".join(f"{'b(K=' + k_label(k) + ')':>11}" for k in ks)
    headw = " ".join(f"{'w(K=' + k_label(k) + ')':>11}" for k in ks)
    print(f"\n{'#':>4} {'step':>7} {'tokens':>17} {'run':>3} {head} {headw}")
    for i, s in enumerate(plan.sources):
        b = " ".join(f"{'init':>11}" if i == 0 else f"{summaries[k]['betas'][i]:11.6f}"
                     for k in ks)
        w = " ".join(f"{summaries[k]['weights'][i]:11.3e}" for k in ks)
        print(f"{i:>4} {s.step:>7} {s.tokens:>17,} {s.root_index:>3} {b} {w}")
    print(f"\ncheckpoints used   : {len(plan.sources)}  "
          f"(step {plan.sources[0].step} = {plan.sources[0].tokens:,} tokens as init -> "
          f"step {final_step} = {plan.sources[-1].tokens:,} tokens)")
    for k in ks:
        sm = summaries[k]
        print(f"K={k_label(k):>5}: init weight {sm['init_weight']:.2e} / final ckpt {sm['final_checkpoint_weight']:.4f} / "
              f"effective n {sm['effective_num_checkpoints']:.1f} / center of mass {sm['center_of_mass_tokens'] / 1e12:.4f}T / "
              f"50/90/99% of mass in the last {sm['num_checkpoints_covering_mass'].get('0.5')}/"
              f"{sm['num_checkpoints_covering_mass'].get('0.9')}/"
              f"{sm['num_checkpoints_covering_mass'].get('0.99')} points / "
              f"|β − exact online product| ≤ {sm['max_abs_beta_minus_exact_online_product']:.1e}")
        for run, m in sm["mass_per_run"].items():
            print(f"        mass {m:.3e}  {run}")
        print(f"        -> {outputs[k]}")


# ==========================================================================
# Averaging
# ==========================================================================
def _group_of(key: str) -> str:
    return key.split(".", 1)[0]


def compute_sanity(
    accs: dict[float, dict[str, torch.Tensor]],
    final_theta: dict[str, torch.Tensor],
    float_keys: list[str],
) -> dict[str, Any]:
    """NaN/Inf check, L2 distance to the raw final checkpoint, and ordering in K."""
    out: dict[str, Any] = {"per_k": {}, "pairwise_l2": {}}
    n2_final = math.fsum(float(final_theta[k].double().pow(2).sum()) for k in float_keys)
    out["l2_norm_final"] = math.sqrt(n2_final)
    for k_val, acc in accs.items():
        nonfinite = [k for k in float_keys if not bool(torch.isfinite(acc[k]).all())]
        groups: dict[str, float] = {}
        n2 = 0.0
        for key in float_keys:
            d = float((acc[key].double() - final_theta[key].double()).pow_(2).sum())
            groups[_group_of(key)] = groups.get(_group_of(key), 0.0) + d
            n2 += float(acc[key].double().pow(2).sum())
        d2 = math.fsum(groups.values())
        out["per_k"][k_label(k_val)] = {
            "finite": not nonfinite,
            "nonfinite_keys": nonfinite[:20],
            "l2_to_final": math.sqrt(d2),
            "rel_l2_to_final": math.sqrt(d2 / n2_final),
            "l2_to_final_by_group": {g: math.sqrt(v) for g, v in sorted(groups.items())},
            "l2_norm": math.sqrt(n2),
        }
    ks = sorted(accs)
    for a, b in zip(ks, ks[1:]):
        d2 = math.fsum(float((accs[a][key].double() - accs[b][key].double()).pow_(2).sum())
                       for key in float_keys)
        out["pairwise_l2"][f"{k_label(a)}-{k_label(b)}"] = math.sqrt(d2)
    dists = [out["per_k"][k_label(k)]["l2_to_final"] for k in ks]
    out["closer_to_final_as_k_grows"] = all(x > y for x, y in zip(dists, dists[1:]))
    return out


def _source_record(plan: Plan, s: Source) -> dict[str, Any]:
    return {"step": s.step, "tokens": s.tokens, "path": s.path, "realpath": s.realpath,
            "run": str(plan.roots[s.root_index].dump or plan.roots[s.root_index].ckpt_dir)}


def build_metadata(
    plan: Plan, args: argparse.Namespace, k: float, name: str, display: str,
    summary: dict[str, Any], out_dtype: torch.dtype,
) -> dict[str, Any]:
    first, last = plan.sources[0], plan.sources[-1]
    schedule = [
        {"index": i, **_source_record(plan, s),
         "beta_interval": None if i == 0 else summary["betas"][i],
         "beta_interval_exact_online_product_in_steps": summary["online_betas_exact_in_steps"][i],
         "final_weight": summary["weights"][i],
         "ntokens_seen_rank_local": s.ntokens_seen}
        for i, s in enumerate(plan.sources)
    ]
    return {
        "generator": GENERATOR,
        "status": "in_progress",
        "name": name,
        "display_name": display,
        "method": "EMA-K (Anytime Pretraining: Horizon-Free Learning-Rate Schedules "
                  "with Weight Averaging)",
        "approximation": APPROXIMATION,
        "approximation_note": APPROXIMATION_NOTE,
        "K": k,
        "online_beta_formula": "beta_t = 2 ** (-K / t)   (t = optimizer step; not used directly)",
        "beta_formula": "beta_interval_i = (T_{i-1} / T_i) ** (K * ln 2) "
                        "= exp(-K * ln 2 * ln(T_i / T_{i-1}))",
        "recurrence": "ema_0 = theta(T_0); ema_i = beta_interval_i * ema_{i-1} "
                      "+ (1 - beta_interval_i) * theta(T_i)",
        "closed_form_weights": "w_0 = (T_0/T_N)^(K ln 2); "
                               "w_i = (1 - beta_interval_i) * (T_i/T_N)^(K ln 2)",
        "training_age_unit": "global tokens = train_state.step * tokens_per_step "
                             "(constant global batch, so ratios equal optimizer-step ratios)",
        "target_tokens": args.target_tokens,
        "end_tokens": plan.horizon_tokens,
        "start_tokens": args.start_tokens,
        "tokens_per_step": plan.tokens_per_step,
        "tokens_per_step_source": plan.tokens_per_step_source,
        "checkpoint_interval_tokens": args.checkpoint_interval_tokens,
        "checkpoint_interval_steps": plan.interval_steps,
        "num_checkpoints": len(plan.sources),
        "source_range": {"first": _source_record(plan, first), "last": _source_record(plan, last)},
        "lineage": [
            {"checkpoint_dir": str(r.ckpt_dir), "dump": str(r.dump) if r.dump else None,
             "via": r.via, "dp_world_size": r.dp_world_size,
             "steps_in_lineage": [r.taken[0], r.taken[-1]] if r.taken else None,
             "num_in_lineage": len(r.taken),
             "next_run_starts_at_step": r.branch_step,
             "not_taken_from_this_run": r.dropped}
            for r in plan.roots
        ],
        "offgrid_skipped_steps": plan.offgrid_skipped,
        "irregular_gaps": plan.irregular_gaps,
        "missing_grid_steps": plan.missing_grid_steps,
        "accumulator_dtype": str(ACCUMULATOR_DTYPE).replace("torch.", ""),
        "output_dtype": str(out_dtype).replace("torch.", ""),
        "source_dtypes": sorted({dt for _, dt in plan.sig.values()}),
        "num_model_tensors": len(plan.sig),
        "ema_keys": len(plan.float_keys),
        "copied_from_final_checkpoint_keys": plan.other_keys,
        "not_averaged_state": list(NON_MODEL_PREFIXES) + ["rng state (not in checkpoint)"],
        "weights_summary": {k2: v for k2, v in summary.items()
                            if k2 not in ("weights", "betas", "online_betas_exact_in_steps")},
        "schedule": schedule,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "command": [sys.executable] + sys.argv,
        "git": _git_info(),
    }


def run(args: argparse.Namespace, plan: Plan, outputs: dict[float, Path], names: dict[float, tuple[str, str]]) -> dict[float, Path]:
    final = plan.sources[-1]
    out_dtype = getattr(torch, args.dtype)
    summaries = {k: weight_summary(plan, k) for k in args.ema_k}
    metas = {k: build_metadata(plan, args, k, *names[k], summaries[k], out_dtype)
             for k in args.ema_k}
    for k, run_dir in outputs.items():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoint").mkdir(exist_ok=True)
        # Remove partial outputs left by a crashed earlier run (this script's naming only)
        for stale in (run_dir / "checkpoint").glob(".step-*.partial-*"):
            shutil.rmtree(stale)
        _write_json(run_dir / METADATA_NAME, metas[k])

    loader = ParallelLoader(plan.sig, args.workers)  # forks here
    torch.set_num_threads(args.threads)
    _log(f"staging {sum(t.numel() for t in loader.staging.values()) / 1e9:.3f}B params "
         f"(shared memory) / reader {len(loader.groups)} / torch threads {args.threads}")
    accs = {k: {key: torch.empty(plan.sig[key][0], dtype=ACCUMULATOR_DTYPE)
                for key in plan.float_keys} for k in args.ema_k}
    coefs = {k: interval_coefficients([s.tokens for s in plan.sources], k) for k in args.ema_k}
    checksums: list[float] = []
    t_start = time.time()
    try:
        for i, src in enumerate(plan.sources):
            if i + 1 < len(plan.sources):
                loader.prefetch(plan.sources[i + 1].path)
            t_read = loader.load(src.path)
            t0 = time.time()
            with torch.no_grad():
                csum = math.fsum(float(loader.staging[key].sum(dtype=torch.float64))
                                 for key in plan.float_keys)
            if not math.isfinite(csum):
                raise RuntimeError(f"{src.path} contains NaN/Inf (sum = {csum})")
            checksums.append(csum)
            for k in args.ema_k:
                beta, one_minus = coefs[k][i]
                update_accumulator(accs[k], loader.staging, plan.float_keys,
                                   beta, one_minus, first=(i == 0))
            t_upd = time.time() - t0
            done = time.time() - t_start
            eta = done / (i + 1) * (len(plan.sources) - i - 1)
            _log(f"[{i + 1}/{len(plan.sources)}] step {src.step} ({src.tokens / 1e12:.4f}T) "
                 f"read {t_read:.1f}s update {t_upd:.1f}s | elapsed {done / 60:.1f} min, "
                 f"ETA {eta / 60:.1f} min")

        final_theta = {key: t.clone() for key, t in loader.staging.items()}
        _log("sanity checks (NaN/Inf, distance to raw final)")
        sanity = compute_sanity(accs, final_theta, plan.float_keys)
        for k in args.ema_k:
            p = sanity["per_k"][k_label(k)]
            _log(f"  K={k_label(k):>5}: finite={p['finite']} ‖ema − raw‖ = {p['l2_to_final']:.4f} "
                 f"(relative {p['rel_l2_to_final']:.3e})")
        _log(f"  closer to raw as K grows: {sanity['closer_to_final_as_k_grows']} "
             f"/ pairwise {sanity['pairwise_l2']}")
        if not all(p["finite"] for p in sanity["per_k"].values()):
            raise RuntimeError(f"NaN/Inf in output: {sanity['per_k']}")

        for k in args.ema_k:
            run_dir = outputs[k]
            ckpt_dir = run_dir / "checkpoint" / f"step-{final.step}"
            tmp = ckpt_dir.parent / f".step-{final.step}.partial-{os.getpid()}"
            state = {key: (accs[k][key].to(out_dtype) if key in accs[k] else final_theta[key])
                     for key in plan.sig}
            t0 = time.time()
            dcp.save(state, storage_writer=FileSystemWriter(str(tmp)), no_dist=True)
            t_save = time.time() - t0
            roundtrip = None
            if not args.no_roundtrip_check:
                loader.load(str(tmp))
                bad = [key for key in plan.sig
                       if not torch.equal(loader.staging[key], state[key].to(loader.staging[key].dtype))]
                if bad:
                    raise RuntimeError(f"re-read of the written checkpoint does not match: {bad[:5]}")
                roundtrip = {"bitwise_equal": True, "tensors": len(plan.sig)}
            meta = dict(metas[k])
            meta.update({
                "status": "complete",
                "checkpoint": str(ckpt_dir),
                "final_step": final.step,
                "sanity": {
                    **sanity["per_k"][k_label(k)],
                    "l2_norm_final": sanity["l2_norm_final"],
                    "pairwise_l2_between_k": sanity["pairwise_l2"],
                    "closer_to_final_as_k_grows": sanity["closer_to_final_as_k_grows"],
                    "source_checksums_float64_sum": checksums,
                    "roundtrip": roundtrip,
                },
                "elapsed_sec": time.time() - t_start,
            })
            _write_json(tmp / METADATA_NAME, meta)
            if ckpt_dir.exists():  # --overwrite (check_output_slot confirmed it is ours)
                shutil.rmtree(ckpt_dir)
            os.rename(tmp, ckpt_dir)
            _write_json(run_dir / METADATA_NAME, meta)
            _log(f"wrote K={k_label(k)} -> {ckpt_dir} (save {t_save:.1f}s)")
            del state
    finally:
        loader.close()
    return {k: outputs[k] / "checkpoint" / f"step-{final.step}" for k in args.ema_k}


# ==========================================================================
# Check that the outputs load through the evaluation code
# ==========================================================================
def verify_with_eval_code(
    entries: list[tuple[str, Path]],
    *,
    flavor: str,
    alpha_max: float | None,
    build: str,
    device: str,
    n_blocks: int,
    batch_size: int,
    seed: int,
    expected_l2: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Load each checkpoint through `scripts/eval` and compute CE on the first n blocks.

    Loading, casting, noise and CE use the same functions as `scripts/eval/ckpt_eval.py`,
    so this matches the common-grid computation on the first n blocks (a small sample,
    not the reported CE). With `expected_l2` (label -> L2 norm recorded at creation),
    the norm of the fp32 weights loaded by the evaluation path is checked; for the raw
    final checkpoint this confirms that this script's parallel loader and the
    evaluation loader return the same weights.
    """
    from scripts.eval.ckpt_eval import COMMON_GRID, _cast_params, eval_source_flow
    from scripts.eval.loader import build_eval_model, load_checkpoint_distributed
    from torchtitan.models.swaflow.validate import load_heldout

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    model = build_eval_model("flow", flavor, alpha_max=alpha_max)
    model.to(device=dev)
    model.eval()
    sources = load_heldout(build, n_blocks)
    results: dict[str, Any] = {
        "flavor": flavor, "alpha_max": alpha_max, "build_dir": build,
        "blocks_per_source": n_blocks, "alpha_grid": COMMON_GRID, "seed": seed,
        "device": str(dev), "forward_dtype": "bfloat16" if dev.type == "cuda" else "float32",
        "note": "quick CE check on the first blocks only; not the val_common number",
        "models": {},
    }
    for label, path in entries:
        t0 = time.time()
        _cast_params(model, torch.float32)
        info = load_checkpoint_distributed(model, path, device=dev)
        with torch.no_grad():
            l2 = math.sqrt(math.fsum(float(t.double().pow(2).sum())
                                     for t in model.state_dict().values()
                                     if t.is_floating_point()))
        want = (expected_l2 or {}).get(label)
        l2_match = None if want is None else math.isclose(l2, want, rel_tol=1e-9)
        if dev.type == "cuda":
            _cast_params(model, torch.bfloat16)
        per_source = {
            src.name: eval_source_flow(model, src, alpha_grid=COMMON_GRID, seed=seed,
                                       batch_size=batch_size, device=dev, world=1, rank=0)
            for src in sources
        }
        wsum = sum(r["weight"] for r in per_source.values())
        overall = sum(r["loss"] * r["weight"] for r in per_source.values()) / wsum
        per_alpha = {
            a: sum(r["per_alpha"][a] * r["weight"] for r in per_source.values()) / wsum
            for a in next(iter(per_source.values()))["per_alpha"]
        }
        finite = math.isfinite(overall) and all(math.isfinite(v) for v in per_alpha.values())
        results["models"][label] = {
            "checkpoint": str(path), "loaded_tensors": info["loaded"],
            "l2_norm_loaded_fp32": l2, "l2_norm_recorded": want, "l2_match": l2_match,
            "overall_ce": overall, "per_alpha_ce": per_alpha, "finite": finite,
            "elapsed_sec": time.time() - t0,
        }
        _log(f"  eval-load {label}: {info['loaded']} tensors, ‖θ‖ {l2:.6f} "
             f"(matches record: {l2_match}), quick-check CE {overall:.4f} "
             f"(finite={finite}, {time.time() - t0:.1f}s)")
    return results


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="EMA-K from saved checkpoints (" + APPROXIMATION + ")",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--checkpoint-root", nargs="+", required=True,
                    help="run dump (or its checkpoint/); several allowed, oldest first along the lineage")
    ap.add_argument("--no-follow-branch", action="store_true",
                    help="do not follow a symlinked first checkpoint back to the parent run")
    ap.add_argument("--target-tokens", type=_parse_tokens, required=True,
                    help="end point; uses the last checkpoint at or before it (within half an interval)")
    ap.add_argument("--checkpoint-interval-tokens", type=_parse_tokens,
                    default=20_000_000_000)
    ap.add_argument("--tokens-per-step", type=_parse_tokens, default=None,
                    help="default: global_batch_size × seq_len from <dump>/run_order.json")
    ap.add_argument("--start-tokens", type=_parse_tokens, default=None,
                    help="initialise from the first checkpoint at or after this token position")
    ap.add_argument("--end-tokens", type=_parse_tokens, default=None,
                    help="stop before the target (the output name uses this position)")
    ap.add_argument("--include-offgrid", action="store_true",
                    help="also use checkpoints off the interval grid (the final one is always used)")
    ap.add_argument("--ema-k", nargs="+", type=float, default=list(DEFAULT_KS))
    ap.add_argument("--output-dir", required=True,
                    help="parent directory of outputs (<output-dir>/<name>/checkpoint/step-N)")
    ap.add_argument("--name-template", default="ema{k}_3b_{target}",
                    help="output run name; fields {k}, {target} (e.g. 4t), {TARGET} (e.g. 4T)")
    ap.add_argument("--display-name-template", default="SwaFlow-3B-{TARGET}-ema{k}")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="output dtype (the accumulator is always float32); "
                         "scripts/eval expects fp32 checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without reading weights (metadata is read)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing outputs, only those written by this script")
    ap.add_argument("--workers", type=int, default=32, help="processes reading weights")
    ap.add_argument("--threads", type=int, default=16, help="torch threads for the accumulator update")
    ap.add_argument("--scan-workers", type=int, default=8, help="processes scanning metadata")
    ap.add_argument("--no-roundtrip-check", action="store_true")
    ap.add_argument("--verify-flavor", default=None,
                    help="if set, load the outputs (and the raw final) through scripts/eval and report CE")
    ap.add_argument("--verify-alpha-max", type=float, default=None)
    ap.add_argument("--verify-build", default=None,
                    help="held-out build (default: the build recorded in the input run's val/)")
    ap.add_argument("--verify-device", default=None, help="default: cuda:0 if available")
    ap.add_argument("--verify-blocks", type=int, default=4)
    ap.add_argument("--verify-batch", type=int, default=8)
    ap.add_argument("--verify-seed", type=int, default=1234)
    ap.add_argument("--verify-only", action="store_true",
                    help="skip averaging; only run --verify-flavor on existing outputs")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.ema_k = sorted(set(args.ema_k))
    if any(k <= 0 for k in args.ema_k):
        raise SystemExit("K must be positive")
    if args.verify_only and not args.verify_flavor:
        raise SystemExit("--verify-only requires --verify-flavor")

    plan = build_plan(args)
    final = plan.sources[-1]
    label = token_label(plan.horizon_tokens)
    names = {k: (args.name_template.format(k=k_label(k), target=label.lower(), TARGET=label.upper()),
                 args.display_name_template.format(k=k_label(k), target=label.lower(),
                                                   TARGET=label.upper()))
             for k in args.ema_k}
    if len({n for n, _ in names.values()}) != len(names):
        raise SystemExit(f"output names collide: {names}")
    out_root = Path(args.output_dir)
    outputs = {k: out_root / names[k][0] for k in args.ema_k}
    for run_dir in outputs.values():
        check_output_is_not_a_source(run_dir, plan.roots, plan.sources)

    if args.dry_run:
        print_plan(plan, args.ema_k, {k: outputs[k] / "checkpoint" / f"step-{final.step}"
                                      for k in args.ema_k}, final.step)
        for k, run_dir in outputs.items():
            if run_dir.exists():
                state = "existing output of this script" if _is_ours(run_dir) else "NOT written by this script"
                print(f"note: {run_dir} already exists ({state})")
        return 0

    if args.verify_only:
        ckpts = {k: outputs[k] / "checkpoint" / f"step-{final.step}" for k in args.ema_k}
        missing = [str(p) for p in ckpts.values() if not (p / ".metadata").exists()]
        if missing:
            raise SystemExit(f"missing outputs: {missing}")
    else:
        for run_dir in outputs.values():
            check_output_slot(run_dir, args.overwrite)
        t0 = time.time()
        ckpts = run(args, plan, outputs, names)
        _log(f"EMA done {(time.time() - t0) / 60:.1f} min")

    if args.verify_flavor:
        device = args.verify_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        build = args.verify_build
        if build is None:
            from scripts.eval.ckpt_eval import _build_from_run
            dump = plan.roots[final.root_index].dump
            if dump is None:
                raise SystemExit("pass --verify-build")
            build = _build_from_run(dump)
        entries = [("raw", Path(final.path))] + [(f"ema{k_label(k)}", ckpts[k]) for k in args.ema_k]
        expected_l2: dict[str, float] = {}
        for k in args.ema_k:
            sanity = json.loads((outputs[k] / METADATA_NAME).read_text()).get("sanity", {})
            if "l2_norm" in sanity:
                expected_l2[f"ema{k_label(k)}"] = sanity["l2_norm"]
            if "l2_norm_final" in sanity:
                expected_l2["raw"] = sanity["l2_norm_final"]
        _log(f"loading through scripts/eval: {len(entries)} models / {device}")
        ver = verify_with_eval_code(
            entries, flavor=args.verify_flavor, alpha_max=args.verify_alpha_max,
            build=build, device=device, n_blocks=args.verify_blocks,
            batch_size=args.verify_batch, seed=args.verify_seed, expected_l2=expected_l2)
        for k in args.ema_k:
            path = outputs[k] / METADATA_NAME
            meta = json.loads(path.read_text())
            meta["verification_eval_load"] = {
                **ver, "this_model": ver["models"][f"ema{k_label(k)}"]}
            _write_json(path, meta)
            _write_json(ckpts[k] / METADATA_NAME, meta)
        if not all(m["finite"] for m in ver["models"].values()):
            raise SystemExit("NaN/Inf in CE from the evaluation path")
        if any(m["l2_match"] is False for m in ver["models"].values()):
            raise SystemExit("norm of weights loaded by the evaluation path does not match the recorded norm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
