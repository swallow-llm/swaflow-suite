# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Equivalence point α_eq: the α at which the flow's held-out CE equals its AR twin's.

Flow and AR losses mean different things, so they are compared by difficulty:
α_eq(N, D) solves L_flow(α_eq; N, D) = CE_AR(N, D), with both models measured on
the same held-out blocks, sources and mixture weights. Interpolation is linear in
log CE.

At α_eq the two CEs are equal by definition, so their improvement rates coincide
trivially. Rates are instead compared with α fixed at `alpha_ref`, the final
checkpoint's α_eq for each size: d ln L_flow(alpha_ref) / d ln D and
d ln CE_AR / d ln D are least-squares fitted over the same range of D.

    python -m scripts.eval.equivalence --series val_window32 --out <outdir>
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

from scripts.eval.profiles import SIZES, load_series, overall_profile


def loglog_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope s of log y = b + s * log x."""
    pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y and y > 0]
    if len(pts) < 3:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    if sxx <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx


def interp_alpha_at(points: list[tuple[float, float]], level: float) -> float | None:
    """α with L(α) = level, interpolated linearly in log CE (as `profiles.crossing`)."""
    pts = sorted((a, c) for a, c in points if not math.isnan(c) and c > 0)
    found = None
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if c0 >= level >= c1 and c0 != c1:
            num = math.log(c0) - math.log(level)
            den = math.log(c0) - math.log(c1)
            found = a0 + (a1 - a0) * (num / den)
    return found


def interp_ce_at(points: list[tuple[float, float]], alpha: float) -> float | None:
    """CE at α, interpolated linearly in log CE; None outside the grid."""
    pts = sorted((a, c) for a, c in points if not math.isnan(c) and c > 0)
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if a0 <= alpha <= a1:
            if a1 == a0:
                return c0
            f = (alpha - a0) / (a1 - a0)
            return math.exp(math.log(c0) + f * (math.log(c1) - math.log(c0)))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default=os.environ.get("SWAFLOW_OUTPUTS", ""))
    ap.add_argument("--series", default="val_window32")
    ap.add_argument("--fallback-series", default="val")
    ap.add_argument("--ar-series", default="val")
    ap.add_argument("--fit-from-tokens", type=float, default=2e11,
                   help="fit slopes from this many tokens on (default 200B, skipping the early transient)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    outputs = Path(args.outputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tps = 4096 * 2048

    rows: list[dict] = []
    per_size: dict[str, list[dict]] = {}
    for run, (label, dim, amax, ar_run) in SIZES.items():
        if ar_run is None:
            continue  # no AR twin at 3B
        dump = outputs / run
        name = args.series
        if not list((dump / name).glob("step-*.json")):
            name = args.fallback_series
        flow = {p["step"]: p for p in load_series(dump, name)}
        ar = {p["step"]: p["overall_loss"]
              for p in load_series(outputs / ar_run, args.ar_series)}
        common = sorted(set(flow) & set(ar))
        got = []
        for step in common:
            prof = overall_profile(flow[step])
            ce_ar = ar[step]
            a_eq = interp_alpha_at(prof, ce_ar)
            got.append({
                "run": run, "size": label, "dim": dim, "alpha_max": amax,
                "step": step, "tokens": step * tps,
                "ce_ar": ce_ar,
                "flow_overall": flow[step]["overall_loss"],
                "alpha_eq": a_eq,
                "alpha_eq_over_amax": (a_eq / amax) if a_eq else None,
                "series": name, "n_alpha": len(flow[step]["alpha_grid"]),
            })
        per_size[run] = got
        rows += got
        print(f"[eq] {label}: {len(got)} common steps "
              f"(flow {name} / ar {args.ar_series})")

    if not rows:
        raise SystemExit("no common steps (was the AR val series found?)")

    # ---- slopes with α fixed at the final equivalence point ----
    slopes = []
    for run, got in per_size.items():
        label, dim, amax, ar_run = SIZES[run]
        defined = [g for g in got if g["alpha_eq"]]
        if not defined:
            continue
        a_ref = defined[-1]["alpha_eq"]
        dump = outputs / run
        name = defined[-1]["series"]
        flow = {p["step"]: p for p in load_series(dump, name)}
        xs, y_flow, y_ar = [], [], []
        for g in got:
            if g["tokens"] < args.fit_from_tokens:
                continue
            ce = interp_ce_at(overall_profile(flow[g["step"]]), a_ref)
            if ce is None:
                continue
            xs.append(g["tokens"])
            y_flow.append(ce)
            y_ar.append(g["ce_ar"])
            g["flow_ce_at_alpha_ref"] = ce
        slopes.append({
            "size": label, "dim": dim, "alpha_ref": a_ref,
            "n_fit": len(xs),
            "slope_flow": loglog_slope(xs, y_flow),
            "slope_ar": loglog_slope(xs, y_ar),
            "slope_alpha_eq": loglog_slope(
                [g["tokens"] for g in defined if g["tokens"] >= args.fit_from_tokens],
                [g["alpha_eq"] for g in defined if g["tokens"] >= args.fit_from_tokens],
            ),
            "alpha_eq_final": defined[-1]["alpha_eq"],
            "ce_ar_final": defined[-1]["ce_ar"],
        })

    # ---- output ----
    cols = ["run", "size", "dim", "alpha_max", "series", "n_alpha", "step",
            "tokens", "ce_ar", "flow_overall", "alpha_eq", "alpha_eq_over_amax",
            "flow_ce_at_alpha_ref"]
    with (out / "equivalence_series.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "equivalence_slopes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slopes[0]) if slopes else ["size"])
        w.writeheader()
        w.writerows(slopes)

    md = ["# Equivalence point α_eq (the α at which the flow CE equals the AR CE)", "",
          f"- flow series: {args.series} (fallback: {args.fallback_series}), "
          f"AR series: {args.ar_series}",
          "- interpolation: linear in log CE (as `profiles.crossing`)",
          f"- slopes fitted from {args.fit_from_tokens:.3g} tokens on", "",
          "## 1. Position of the equivalence point", "",
          "| size | final α_eq | α_eq / α_max | final CE (AR) | α_eq at 200B | "
          "d ln α_eq / d ln D |", "|---|---|---|---|---|---|"]
    for s in slopes:
        run = next(r for r, v in SIZES.items() if v[0] == s["size"])
        got = per_size[run]
        early = next((g["alpha_eq"] for g in got
                      if g["tokens"] >= 2e11 and g["alpha_eq"]), None)
        amax = SIZES[run][2]
        md.append(
            f"| {s['size']} | {s['alpha_eq_final']:.5f} | "
            f"{s['alpha_eq_final'] / amax:.3f} | {s['ce_ar_final']:.4f} | "
            f"{early:.5f} | {_f(s['slope_alpha_eq'])} |"
        )
    md += ["", "## 2. Improvement rate (log-log slopes)", "",
           "With α fixed at the final equivalence point `alpha_ref`, the log-log slopes",
           "of the flow CE and the AR CE are compared over the same range of D (at the",
           "equivalence point itself the two CEs are equal by definition).", "",
           "| size | alpha_ref | points | d ln CE_flow / d ln D | "
           "d ln CE_AR / d ln D | ratio |", "|---|---|---|---|---|---|"]
    for s in slopes:
        sf, sa = s["slope_flow"], s["slope_ar"]
        ratio = f"{sf / sa:.3f}" if (sf and sa) else "-"
        md.append(f"| {s['size']} | {s['alpha_ref']:.5f} | {s['n_fit']} | "
                  f"{_f(sf)} | {_f(sa)} | {ratio} |")
    (out / "equivalence_summary.md").write_text("\n".join(md) + "\n",
                                                encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/equivalence_*.csv/.md")
    return 0


def _f(v, nd: int = 4) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    sys.exit(main())
