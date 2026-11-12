# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Read α profiles and threshold crossings from val dumps (training-time or offline).

A thin reader over `<dump>/<series>/step-*.json`, where the series is `val`
(training grid), `val_common` (common grid) or `val_window32` (window grid).

α_sol (key `a@ce0.1`) is the α at which held-out CE crosses 0.1 nats, found by
linear interpolation in log CE between neighbouring grid points:

    a = a0 + (a1 - a0) * (ln c0 - ln 0.1) / (ln c0 - ln c1)

Inside the window CE falls roughly exponentially in α, so linear interpolation in
CE overestimates the crossing on a coarse grid; the linear value is reported
alongside for comparison. `scripts/analysis/alpha_window.py` uses linear
interpolation on an untruncated pilot run and is a separate tool.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

__all__ = [
    "SIZES",
    "crossing",
    "load_series",
    "overall_profile",
    "series_rows",
]

#: run name -> (display name, d, window edge α_max, AR twin run name)
SIZES: dict[str, tuple[str, int, float, str | None]] = {
    "flow_150m": ("150M", 768, 0.15, "ar_150m"),
    "flow_500m": ("500M", 1280, 0.125, "ar_500m"),
    "flow_1b": ("1B", 2048, 0.105, "ar_1b"),
    "flow_3b": ("3B", 3072, 0.10, None),
}

#: global batch in tokens (same for all runs)
TOKENS_PER_STEP = 4096 * 2048


def load_series(dump: str | Path, name: str = "val") -> list[dict]:
    """Read `<dump>/<name>/step-*.json` in ascending step order."""
    root = Path(dump) / name
    out = []
    for p in sorted(root.glob("step-*.json"),
                    key=lambda q: int(q.stem.split("-")[1])):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def overall_profile(payload: dict) -> list[tuple[float, float]]:
    """(α, CE) weighted by the data-mixture weights.

    Sources with `n_blocks == 0` are dropped, as in the training-time validator.
    """
    measured = {n: r for n, r in payload["sources"].items() if r["n_blocks"] > 0}
    wsum = sum(r["weight"] for r in measured.values()) or 1.0
    return [
        (float(a),
         sum(r["per_alpha"][f"{a:g}"] * r["weight"] for r in measured.values()) / wsum)
        for a in payload["alpha_grid"]
    ]


def crossing(
    points: list[tuple[float, float]], level: float, *, log: bool = True
) -> float | None:
    """α at which CE crosses `level` from above; returns the crossing at the largest α.

    `log=True` (default) interpolates linearly in log CE.
    """
    pts = sorted((a, c) for a, c in points if not math.isnan(c))
    found = None
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if c0 >= level >= c1 and c0 != c1:
            if log and c0 > 0 and c1 > 0 and level > 0:
                num = math.log(c0) - math.log(level)
                den = math.log(c0) - math.log(c1)
            else:
                num, den = c0 - level, c0 - c1
            found = a0 + (a1 - a0) * (num / den)
    return found


def series_rows(
    dump: str | Path, name: str = "val", *, level: float = 0.1
) -> list[dict]:
    """Time series for one run (step, tokens, CE, a@ce{level}, α_geo, c_mean, ...)."""
    rows = []
    for payload in load_series(dump, name):
        pts = overall_profile(payload)
        geom = payload.get("embed_geometry") or {}
        a_log = crossing(pts, level, log=True)
        a_lin = crossing(pts, level, log=False)
        astar = geom.get("alpha_star")
        rows.append(
            {
                "step": payload["step"],
                "tokens": payload["step"] * TOKENS_PER_STEP,
                "overall_loss": payload["overall_loss"],
                "grid_name": payload.get("grid_name", "train"),
                "n_alpha": len(payload["alpha_grid"]),
                f"a@ce{level:g}": a_log,
                f"a@ce{level:g}_linear": a_lin,
                "alpha_star": astar,
                "alpha_star_isotropic": geom.get("alpha_star_isotropic"),
                "c_mean": geom.get("c_mean"),
                "m_mean": geom.get("m_mean"),
                "emb_norm_mean": geom.get("emb_norm_mean"),
                "ratio_astar_over_a": (
                    astar / a_log if (astar and a_log) else None
                ),
                "profile": pts,
            }
        )
    return rows
