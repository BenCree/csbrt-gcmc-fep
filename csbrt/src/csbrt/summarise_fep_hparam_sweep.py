#!/usr/bin/env python3
"""Tabulate a submit_fep_hparam_sweep.sh sweep, replicate-aware.

Run roots are `<point>__rep<N>`. For every (point, edge) this reports the replicate
mean DDG, the replicate range (the seed noise of that protocol), and the worst
adjacent-window overlap. Each point is then compared to a baseline point on the same
edges: a shift only counts as signal when it exceeds the replicate range, the same
discriminator as G3/C1 in the GCMC A/B (there the between-arm change tracked the
replicate spread, i.e. it was seed noise). Verifies by artefact, never exit status.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def ddg(payload: dict) -> float | None:
    value = payload.get("relative_binding_free_energy")
    try:
        first = value[0]
        first = first[0] if isinstance(first, (list, tuple)) else first
        return float(first)
    except (TypeError, IndexError, ValueError):
        return None


def num_windows(leg: Path) -> int:
    return len(list(leg.glob("energy_traj_*.parquet")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--baseline", help="point name to compare against "
                        "(default: the first no-water point alphabetically)")
    parser.add_argument("--csv", type=Path, help="also write the per-point table here")
    opt = parser.parse_args()

    runs = defaultdict(list)  # (point, edge) -> [row per replicate]
    for root in sorted(p for p in opt.sweep_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        point = re.sub(r"__rep\d+$", "", root.name)
        for edge in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            row = {"ddg": None, "ov": None, "complete": False}
            analysis = edge / "analysis.json"
            if analysis.is_file():
                payload = json.loads(analysis.read_text())
                ovs = [v for v in (payload.get("bound_adjacent_overlap_minimum"),
                                   payload.get("free_adjacent_overlap_minimum")) if v is not None]
                row.update(ddg=ddg(payload), ov=min(ovs) if ovs else None,
                           complete=payload.get("status") == "completed")
            runs[(point, edge.name)].append(row)
    if not runs:
        sys.exit(f"no runs found under {opt.sweep_root}")

    table = {}
    for (point, edge), reps in runs.items():
        vals = [r["ddg"] for r in reps if r["ddg"] is not None]
        ovs = [r["ov"] for r in reps if r["ov"] is not None]
        table[(point, edge)] = {
            "point": point, "edge": edge, "n_done": len(vals), "n_runs": len(reps),
            "mean": st.fmean(vals) if vals else None,
            "range": (max(vals) - min(vals)) if len(vals) > 1 else None,
            "min_overlap": min(ovs) if ovs else None,
        }

    points = sorted({p for p, _ in table})
    baseline = opt.baseline or next((p for p in points if p.startswith("nowater")), points[0])
    if baseline not in points:
        sys.exit(f"baseline {baseline!r} not found; points: {points}")

    def f(v, w=7):
        return f"{'--':>{w}}" if v is None else f"{v:{w}.3f}"

    print(f"baseline: {baseline}\n")
    print(f"{'point':58s} {'edges':>5s} {'med range':>9s} {'med |shift|':>11s} "
          f"{'shift>range':>11s} {'worst ov':>8s}")
    out_rows = []
    for point in points:
        edges = sorted(e for p, e in table if p == point)
        rng = [table[(point, e)]["range"] for e in edges if table[(point, e)]["range"] is not None]
        ovs = [table[(point, e)]["min_overlap"] for e in edges if table[(point, e)]["min_overlap"] is not None]
        shifts, beats = [], 0
        for e in edges:
            a, b = table[(point, e)], table.get((baseline, e))
            if point == baseline or not b or a["mean"] is None or b["mean"] is None:
                continue
            shift = abs(a["mean"] - b["mean"])
            noise = max(x for x in (a["range"], b["range"]) if x is not None) if (
                a["range"] is not None or b["range"] is not None) else None
            shifts.append(shift)
            beats += noise is not None and shift > noise
        out_rows.append({"point": point, "edges": len(edges),
                         "median_replicate_range": st.median(rng) if rng else None,
                         "median_abs_shift": st.median(shifts) if shifts else None,
                         "edges_shift_gt_range": beats if shifts else None,
                         "worst_min_overlap": min(ovs) if ovs else None})
        r = out_rows[-1]
        frac = f"{beats}/{len(shifts)}" if shifts else "--"
        print(f"{point[:58]:58s} {len(edges):5d} {f(r['median_replicate_range'], 9)} "
              f"{f(r['median_abs_shift'], 11)} {frac:>11s} {f(r['worst_min_overlap'], 8)}")

    print("\nRead: a setting is only interesting if 'med range' (seed noise) is SMALLER than the\n"
          "baseline's, or 'shift>range' is a clear majority of edges. A point whose shift is\n"
          "inside its own replicate range has not been distinguished from re-running the seed.")

    if opt.csv:
        with opt.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"wrote {opt.csv}")


if __name__ == "__main__":
    main()
