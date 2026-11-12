# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Progress reporting for long jobs, readable from outside the job.

Downloading and tokenizing the full corpus takes hours to tens of hours, so:

* stdout is flushed on every line
* a line with done / total, elapsed time, rate and ETA is printed at a fixed
  interval
* the same record is appended to a JSONL file, so how far a failed job got
  can be read by a program instead of from the log

Progress lines are printed by the aggregating thread, not by the workers
(with many workers the lines would interleave).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

__all__ = ["ProgressReporter", "format_duration"]


def format_duration(sec: float) -> str:
    """Seconds -> a string such as `1d02h03m`."""
    if sec < 0 or sec != sec or sec == float("inf"):
        return "?"
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


class ProgressReporter:
    """Collect completion events and print one summary line per interval.

    Each completed item (file) may carry `units` of work (bytes or tokens),
    which give the rate. Pass `total_units = 0` if the total is not known in
    advance. The ETA is extrapolated from the item count.

    Thread-safe; `advance()` is meant to be called from the side that collects
    executor results.
    """

    def __init__(
        self,
        total_items: int,
        label: str,
        unit_name: str = "tokens",
        total_units: int = 0,
        interval_sec: float = 60.0,
        jsonl_path: str | Path | None = None,
        log=print,
    ) -> None:
        self.total_items = int(total_items)
        self.total_units = int(total_units)
        self.label = label
        self.unit_name = unit_name
        self.interval = float(interval_sec)
        self.log = log
        self._lock = threading.Lock()
        self._items = 0
        self._units = 0
        self._skipped = 0
        self._failed = 0
        self._t0 = time.time()
        self._next = self._t0 + self.interval
        self._jsonl = Path(jsonl_path) if jsonl_path else None
        if self._jsonl:
            self._jsonl.parent.mkdir(parents=True, exist_ok=True)
            # Append, so a restarted job keeps the earlier records.
            self._jsonl.touch(exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def advance(self, units: int = 0, skipped: bool = False, failed: bool = False) -> None:
        with self._lock:
            self._items += 1
            self._units += int(units)
            self._skipped += bool(skipped)
            self._failed += bool(failed)
            due = time.time() >= self._next
            if due:
                self._next = time.time() + self.interval
        if due:
            self.emit()

    def emit(self, final: bool = False) -> None:
        with self._lock:
            snap = dict(
                items=self._items,
                total_items=self.total_items,
                units=self._units,
                total_units=self.total_units,
                skipped=self._skipped,
                failed=self._failed,
                elapsed_sec=round(self.elapsed, 1),
            )
        el = max(snap["elapsed_sec"], 1e-9)
        rate = snap["units"] / el
        frac = snap["items"] / self.total_items if self.total_items else 0.0
        eta = (el / frac - el) if frac > 0 else float("inf")
        snap["rate_per_sec"] = round(rate, 1)
        snap["eta_sec"] = None if eta == float("inf") else round(eta, 1)
        snap["final"] = final
        snap["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        self.log(
            f"[progress:{self.label}] {snap['items']:,}/{self.total_items:,} files"
            f" ({frac:6.2%})"
            + (f" skipped={snap['skipped']:,}" if snap["skipped"] else "")
            + (f" FAILED={snap['failed']:,}" if snap["failed"] else "")
            + f" | {snap['units'] / 1e9:.3f}G {self.unit_name}"
            + f" | {format_duration(el)} elapsed"
            + f" | {rate / 1e6:.2f}M {self.unit_name}/s"
            + f" | ETA {format_duration(eta)}"
        )
        if self._jsonl:
            with open(self._jsonl, "a") as f:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")

    def finish(self) -> dict:
        self.emit(final=True)
        with self._lock:
            return dict(
                items=self._items,
                units=self._units,
                skipped=self._skipped,
                failed=self._failed,
                elapsed_sec=self.elapsed,
            )
