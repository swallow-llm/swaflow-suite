# Copyright (c) 2026 Daisuke Oba
# SPDX-License-Identifier: Apache-2.0

"""Window edge α_max from an untruncated pilot run.

The window [0, α_max] of each size is set where an untruncated pilot's CE falls below
0.1 nats. This reads the per-sample training log (`<dump>/alpha_ce/rank*.jsonl`, binned
in α at several training points) and the validator's fixed α grid (`<dump>/val/step-*.json`),
reports where the CE profile crosses fixed thresholds (linear interpolation), and across
sizes fits the log-log slope against width d (geometric prediction: -1/2). Plots are plain SVG.

    python -m scripts.analysis.alpha_window \\
        --run 150M:768:<dump150m> --run 500M:1280:<dump500m> --run 1B:2048:<dump1b> \\
        --extrapolate-to 3072 --svg <out>/window.svg --csv <out>/window.csv

With gradient accumulation pass `--accum`: `alpha_ce` has one line per microbatch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LN_V = 10.8027  # loss ceiling ln V for the SmolLM2 vocabulary (V = 49,152)

#: CE thresholds at which the transition is read. The small ones locate where the
#: loss rises from ~0; the larger ones only check the shape.
DEFAULT_LEVELS = (0.05, 0.1, 0.5, 1.0, LN_V / 2)


# ---------------------------------------------------------------- training-side log
def _segments(records: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for rec in records:
        if rec["call"] == 0 or not out:
            out.append([])
        out[-1].append(rec)
    return out


def load_train_rows(
    dump: Path, segment: str = "last", start_step: int = 1, accum: int = 1
) -> list[tuple[int, float, float]]:
    """`<dump>/alpha_ce/rank*.jsonl` -> [(step, α, CE)]."""
    rows: list[tuple[int, float, float]] = []
    for path in sorted((dump / "alpha_ce").glob("rank*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        if not records:
            continue
        segs = _segments(records)
        chosen = segs if segment == "all" else [segs[-1 if segment == "last" else 0]]
        for seg in chosen:
            for rec in seg:
                step = rec["call"] // max(1, accum) + start_step
                for a, ce in zip(rec["alpha"], rec["ce"], strict=True):
                    rows.append((step, float(a), float(ce)))
    return rows


def bin_profile(
    rows: list[tuple[int, float, float]],
    lo_step: int,
    hi_step: int,
    edges: list[float],
) -> list[tuple[float, float, int, float]]:
    """(left edge, right edge, n, mean CE) per bin; empty bins give nan.

    Edges come from the caller (not uniform, to resolve low α finely). Only the
    last bin includes its right edge.
    """
    sel = [(a, ce) for step, a, ce in rows if lo_step <= step <= hi_step]
    sel.sort()
    out = []
    i = 0
    for b in range(len(edges) - 1):
        x, y = edges[b], edges[b + 1]
        last = b == len(edges) - 2
        total, n = 0.0, 0
        while i < len(sel) and sel[i][0] < y:
            total += sel[i][1]
            n += 1
            i += 1
        if last:
            while i < len(sel) and sel[i][0] <= y:
                total += sel[i][1]
                n += 1
                i += 1
        out.append((x, y, n, total / n if n else math.nan))
    return out


def log_edges(lo: float, hi: float, n: int) -> list[float]:
    """Log-spaced bin edges (the transition lies around 0.02-0.2)."""
    return [lo * (hi / lo) ** (i / n) for i in range(n + 1)]


# ---------------------------------------------------------------- transition
def crossing(
    points: list[tuple[float, float]], level: float
) -> float | None:
    """α at which CE crosses `level` from above, by linear interpolation.

    The profile decreases in α (ceiling at noise, 0 at clean). The crossing at the
    largest α is returned, so spurious crossings from noise at low α are ignored.
    """
    pts = [(a, ce) for a, ce in points if not math.isnan(ce)]
    pts.sort()
    if len(pts) < 2:
        return None
    found = None
    for (a0, c0), (a1, c1) in zip(pts, pts[1:], strict=False):
        if c0 >= level >= c1 and c0 != c1:
            found = a0 + (a1 - a0) * (c0 - level) / (c0 - c1)
    if found is None:
        # All points below `level`: trivial everywhere; all above: transition outside the range.
        return None
    return found


def transitions(
    points: list[tuple[float, float]], levels=DEFAULT_LEVELS
) -> dict[str, float | None]:
    return {f"a@ce{lvl:g}": crossing(points, lvl) for lvl in levels}


# ---------------------------------------------------------------- validator
def load_val(dump: Path) -> list[dict]:
    """`<dump>/val/step-*.json`, in increasing step order."""
    out = []
    for p in sorted(
        (dump / "val").glob("step-*.json"),
        key=lambda q: int(q.stem.split("-")[1]),
    ):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def val_profile(payload: dict, source: str | None = None) -> list[tuple[float, float]]:
    """(α, CE) on the validator's fixed α grid.

    With `source`, only that source. Use one when comparing sizes: the overall value
    is a weighted mean over sources that filled at least dp_world_size blocks, so its
    composition differs between runs with different data parallelism.
    """
    grid = payload["alpha_grid"]
    srcs = payload["sources"]
    if source is not None:
        per = srcs[source]["per_alpha"]
        if srcs[source]["n_blocks"] == 0:
            raise SystemExit(f"source {source} was not measured (n_blocks=0)")
        return [(float(a), float(per[f"{a:g}"])) for a in grid]
    measured = {n: r for n, r in srcs.items() if r["n_blocks"] > 0}
    wsum = sum(r["weight"] for r in measured.values()) or 1.0
    return [
        (
            float(a),
            sum(r["per_alpha"][f"{a:g}"] * r["weight"] for r in measured.values()) / wsum,
        )
        for a in grid
    ]


# ---------------------------------------------------------------- trend in d
def loglog_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit of log y = b + s·log x; returns (s, b)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True))
    s = sxy / sxx
    return s, my - s * mx


# ---------------------------------------------------------------- SVG
def write_svg(
    path: Path,
    series: list[tuple[str, list[tuple[float, float]]]],
    *,
    title: str,
    xlabel: str = "alpha (1 = clean)",
    ylabel: str = "mean CE",
    xlog: bool = True,
    ymax: float = LN_V,
    marks: list[tuple[str, float]] | None = None,
) -> None:
    """Minimal line plot (lines, points, legend) without a plotting dependency."""
    W, H = 900, 560
    ml, mr, mt, mb = 78, 210, 46, 58
    pw, ph = W - ml - mr, H - mt - mb
    pts = [p for _n, s in series for p in s if not math.isnan(p[1])]
    if not pts:
        raise SystemExit("no points to plot")
    xs = [p[0] for p in pts if p[0] > 0] if xlog else [p[0] for p in pts]
    x0, x1 = min(xs), max(xs)

    def fx(x: float) -> float:
        if xlog:
            x = max(x, x0)
            return ml + pw * (math.log(x / x0) / math.log(x1 / x0))
        return ml + pw * (x - x0) / (x1 - x0)

    def fy(y: float) -> float:
        return mt + ph * (1.0 - min(max(y, 0.0), ymax) / ymax)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    o: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{ml}" y="26" font-size="15" font-weight="bold">{title}</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>',
    ]
    # y axis
    for i in range(6):
        v = ymax * i / 5
        y = fy(v)
        o.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#eee"/>')
        o.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end">{v:.1f}</text>')
    # x axis
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] if xlog else None
    if ticks is None:
        ticks = [x0 + (x1 - x0) * i / 5 for i in range(6)]
    for t in ticks:
        if not (x0 <= t <= x1):
            continue
        x = fx(t)
        o.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#eee"/>')
        o.append(f'<text x="{x:.1f}" y="{mt+ph+18}" text-anchor="middle">{t:g}</text>')
    o.append(
        f'<text x="{ml+pw/2:.0f}" y="{H-14}" text-anchor="middle">{xlabel}</text>'
    )
    o.append(
        f'<text x="18" y="{mt+ph/2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {mt+ph/2:.0f})">{ylabel}</text>'
    )
    for i, (a) in enumerate(marks or []):
        name, xv = a
        if not (x0 <= xv <= x1):
            continue
        x = fx(xv)
        o.append(
            f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" '
            f'stroke="#888" stroke-dasharray="4 3"/>'
        )
        o.append(
            f'<text x="{x+3:.1f}" y="{mt+14+14*i}" fill="#666">{name}</text>'
        )
    for i, (name, s) in enumerate(series):
        col = colors[i % len(colors)]
        good = [(a, c) for a, c in s if not math.isnan(c) and (not xlog or a > 0)]
        if not good:
            continue
        d = " ".join(
            f"{'M' if k == 0 else 'L'}{fx(a):.1f},{fy(c):.1f}"
            for k, (a, c) in enumerate(good)
        )
        o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for a, c in good:
            o.append(f'<circle cx="{fx(a):.1f}" cy="{fy(c):.1f}" r="2.6" fill="{col}"/>')
        ly = mt + 18 + 20 * i
        o.append(
            f'<line x1="{ml+pw+16}" y1="{ly}" x2="{ml+pw+44}" y2="{ly}" '
            f'stroke="{col}" stroke-width="2"/>'
        )
        o.append(f'<text x="{ml+pw+50}" y="{ly+4}">{name}</text>')
    o.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(o), encoding="utf-8")


# ---------------------------------------------------------------- CLI
def _fmt(v: float | None, w: int = 8) -> str:
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.4f}"


def parse_run(spec: str) -> tuple[str, float, Path]:
    """Split `<label>:<dim>:<dump>`."""
    label, dim, dump = spec.split(":", 2)
    return label, float(dim), Path(dump)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL:DIM:DUMP",
                    help="one model size; may be repeated")
    ap.add_argument("--dump", type=Path, help="single dump when --run is not used")
    ap.add_argument("--label", default="run")
    ap.add_argument("--dim", type=float, default=0.0)
    ap.add_argument("--accum", type=int, default=1,
                    help="microbatches per optimizer step (gradient accumulation)")
    ap.add_argument("--segment", choices=("last", "first", "all"), default="last")
    ap.add_argument("--start-step", type=int, default=1)
    ap.add_argument("--windows", type=int, nargs="*", default=None, metavar="STEP",
                    help="end steps of the training-side profiles (default: 20/50/100%% of the run)")
    ap.add_argument("--window-frac", type=float, default=0.25,
                    help="length of each step range, as a fraction of its end step")
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--alpha-range", type=float, nargs=2, default=(0.002, 1.0))
    ap.add_argument("--val-source", default="common_crawl",
                    help="source for the validator profile (default suits comparing sizes); "
                         "'overall' for the weighted mean")
    ap.add_argument("--extrapolate-to", type=float, default=3072.0,
                    help="width d to extrapolate the trend to (3B = 3072)")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="factor applied to the extrapolated transition for the suggested alpha_max")
    ap.add_argument("--svg", type=Path)
    ap.add_argument("--svg-train", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    runs = [parse_run(r) for r in args.run]
    if args.dump is not None:
        runs.append((args.label, args.dim, args.dump))
    if not runs:
        raise SystemExit("--run or --dump is required")

    edges = log_edges(args.alpha_range[0], args.alpha_range[1], args.bins)
    csv_rows: list[str] = ["kind,label,dim,step,alpha,n,ce"]
    val_series: list[tuple[str, list[tuple[float, float]]]] = []
    train_series: list[tuple[str, list[tuple[float, float]]]] = []
    summary: list[dict] = []

    for label, dim, dump in runs:
        print(f"\n{'='*76}\n{label}  (d = {dim:g})  {dump}\n{'='*76}")

        # ---- training side: CE per α bin at several training points ----
        rows = load_train_rows(dump, args.segment, args.start_step, args.accum)
        if rows:
            last = max(s for s, _a, _c in rows)
            ends = args.windows or [
                max(1, int(last * f)) for f in (0.2, 0.5, 1.0)
            ]
            for end in ends:
                lo = max(1, int(end * (1.0 - args.window_frac)))
                prof = bin_profile(rows, lo, end, edges)
                pts = [((x + y) / 2, ce) for x, y, n, ce in prof if n]
                tr = transitions(pts)
                print(f"\n-- training side (α, CE)  step [{lo}, {end}]  "
                      f"{sum(n for _x, _y, n, _c in prof):,} samples")
                print(f"{'α bin':>18} {'n':>9} {'mean CE':>9}")
                for x, y, n, ce in prof:
                    bar = "" if math.isnan(ce) else "#" * int(round(ce / LN_V * 34))
                    print(f"[{x:.4f}, {y:.4f}) {n:>9,} {ce:>9.4f}  {bar}")
                    csv_rows.append(
                        f"train,{label},{dim:g},{end},{(x+y)/2:.6f},{n},"
                        f"{'' if math.isnan(ce) else f'{ce:.6f}'}"
                    )
                print("  transition: " + "  ".join(f"{k}={_fmt(v)}" for k, v in tr.items()))
                if end == ends[-1]:
                    train_series.append((f"{label} step≤{end}", pts))
        else:
            print("  (no alpha_ce records)")

        # ---- validator: fixed α grid ----
        vals = load_val(dump)
        src = None if args.val_source == "overall" else args.val_source
        last_tr: dict[str, float | None] = {}
        geom: dict = {}
        if vals:
            print(f"\n-- validator fixed α grid (source = {args.val_source})")
            head = vals[-1]["alpha_grid"]
            print("  step  " + "".join(f"{a:>8g}" for a in head) + "   a@ce0.1")
            for pay in vals:
                pts = val_profile(pay, src)
                tr = transitions(pts)
                print(f"{pay['step']:>6}  "
                      + "".join(f"{c:>8.3f}" for _a, c in pts)
                      + f"  {_fmt(tr['a@ce0.1'])}")
                for a, c in pts:
                    csv_rows.append(f"val,{label},{dim:g},{pay['step']},{a:.6f},,{c:.6f}")
                last_tr = tr
            val_series.append((f"{label} (d={dim:g})", val_profile(vals[-1], src)))
            geom = vals[-1].get("embed_geometry") or {}
            if geom:
                print(f"\n-- embedding geometry (step {vals[-1]['step']})")
                print(f"  max cos(z, e)   measured m = {geom['m_mean']:.4f}  "
                      f"(isotropic baseline √(2lnV/d) = {geom['m_iid']:.4f}, "
                      f"ratio {geom['m_mean']/geom['m_iid']:.2f})")
                print(f"  max cos(e_i,e_j) measured c = {geom['c_mean']:.4f}")
                print(f"  geometric threshold α_geo = {geom['alpha_star']:.4f}  "
                      f"(without anisotropy {geom['alpha_star_isotropic']:.4f}, "
                      f"isotropic baseline {geom['alpha_star_iid']:.4f})")
        else:
            print("  (no val/step-*.json)")

        print("\n-- transition (last step)")
        for k, v in last_tr.items():
            print(f"  {k:>12} = {_fmt(v)}")
        summary.append({"label": label, "dim": dim, "tr": last_tr, "geom": geom})

    # ---- trend in d and extrapolation ----
    usable = [s for s in summary if s["dim"] > 0]
    if len(usable) >= 2:
        print(f"\n{'='*76}\ntrend in d (log-log slope; geometric prediction −0.5)\n{'='*76}")
        print(f"{'quantity':>22} {'slope':>8} {'slope/-0.5':>9}  " +
              "  ".join(f"{s['label']:>9}" for s in usable) +
              f" {'-> d=' + f'{args.extrapolate_to:g}':>12}")
        keys = [f"a@ce{lvl:g}" for lvl in DEFAULT_LEVELS]
        rows_out = [(k, [s["tr"].get(k) for s in usable]) for k in keys]
        if all(s["geom"] for s in usable):
            rows_out.append(("α_geo", [s["geom"]["alpha_star"] for s in usable]))
            rows_out.append(("m (max cos)", [s["geom"]["m_mean"] for s in usable]))
        for name, vs in rows_out:
            ok = [(s["dim"], v) for s, v in zip(usable, vs, strict=True)
                  if v is not None and v > 0]
            if len(ok) < 2:
                print(f"{name:>22} {'-':>8} {'-':>9}  " +
                      "  ".join(_fmt(v, 9) for v in vs))
                continue
            slope, b = loglog_slope([d for d, _ in ok], [v for _, v in ok])
            pred = math.exp(b) * args.extrapolate_to**slope
            print(f"{name:>22} {slope:>8.3f} {slope/-0.5:>9.2f}  " +
                  "  ".join(_fmt(v, 9) for v in vs) +
                  f" {pred:>12.4f}")
            csv_rows.append(f"trend,{name},{args.extrapolate_to:g},,{pred:.6f},,{slope:.6f}")
        print(f"\n  suggested alpha_max with margin ×{args.margin:g} "
              f"(too wide is safer than too narrow):")
        for name, vs in rows_out:
            ok = [(s["dim"], v) for s, v in zip(usable, vs, strict=True)
                  if v is not None and v > 0]
            if len(ok) < 2 or name.startswith("m "):
                continue
            slope, b = loglog_slope([d for d, _ in ok], [v for _, v in ok])
            print(f"    {name:>18}: {math.exp(b) * args.extrapolate_to**slope * args.margin:.4f}")

    if args.svg and val_series:
        write_svg(args.svg, val_series,
                  title="alpha profile (validator, fixed alpha grid, final step)",
                  marks=[(f"{s['label']} α*", s["geom"]["alpha_star"])
                         for s in summary if s.get("geom")])
        print(f"\nwrote {args.svg}")
    if args.svg_train and train_series:
        write_svg(args.svg_train, train_series,
                  title="alpha profile (training-side per-sample CE)")
        print(f"wrote {args.svg_train}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
