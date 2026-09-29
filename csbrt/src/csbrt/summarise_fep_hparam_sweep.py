#!/usr/bin/env python3
"""Tabulate a submit_fep_hparam_sweep.sh sweep: one row per (grid point, edge).

Reports DDG, minimum adjacent-window overlap per leg, and completeness, so the
waters vs no-waters arms and each hparam setting can be compared on the same edges.
Verifies by artefact (analysis.json / parquet count), never by exit status.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def ddg(payload: dict) -> float | None:
    value = payload.get("relative_binding_free_energy")
    try:
        # BSS difference is [[dG, err], ...] or (dG, err) depending on version.
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
    parser.add_argument("--csv", type=Path, help="also write the table here")
    opt = parser.parse_args()

    rows = []
    for point in sorted(p for p in opt.sweep_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for edge in sorted(p for p in point.iterdir() if p.is_dir() and not p.name.startswith("_")):
            analysis = edge / "analysis.json"
            row = {"point": point.name, "edge": edge.name,
                   "arm": point.name.split("_", 1)[0],
                   "bound_windows": num_windows(edge / "bound"),
                   "free_windows": num_windows(edge / "free"),
                   "ddg": None, "min_overlap_bound": None, "min_overlap_free": None,
                   "status": "incomplete"}
            if analysis.is_file():
                payload = json.loads(analysis.read_text())
                row.update(ddg=ddg(payload),
                           min_overlap_bound=payload.get("bound_adjacent_overlap_minimum"),
                           min_overlap_free=payload.get("free_adjacent_overlap_minimum"),
                           status=payload.get("status", "?"))
            rows.append(row)

    if not rows:
        sys.exit(f"no runs found under {opt.sweep_root}")

    def fmt(v):
        return "--" if v is None else f"{v:.3f}"

    print(f"{'point':60s} {'edge':22s} {'DDG':>8s} {'ov_bnd':>7s} {'ov_free':>7s} win(b/f) status")
    for r in rows:
        print(f"{r['point'][:60]:60s} {r['edge'][:22]:22s} {fmt(r['ddg']):>8s} "
              f"{fmt(r['min_overlap_bound']):>7s} {fmt(r['min_overlap_free']):>7s} "
              f"{r['bound_windows']}/{r['free_windows']:<5} {r['status']}")

    # Waters vs no-waters, per edge: DDG shift between arms at otherwise equal hparams.
    print("\narm summary (mean over completed rows):")
    for arm in sorted({r["arm"] for r in rows}):
        done = [r for r in rows if r["arm"] == arm and r["ddg"] is not None]
        if done:
            mean_ddg = sum(r["ddg"] for r in done) / len(done)
            worst = min(min(r["min_overlap_bound"] or 0, r["min_overlap_free"] or 0) for r in done)
            print(f"  {arm:8s} n={len(done):3d}  mean DDG={mean_ddg:7.3f}  worst min-overlap={worst:.3f}")
        else:
            print(f"  {arm:8s} no completed runs")

    if opt.csv:
        with opt.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {opt.csv}")


if __name__ == "__main__":
    main()
