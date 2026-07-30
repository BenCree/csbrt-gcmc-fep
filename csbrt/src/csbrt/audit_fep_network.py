#!/usr/bin/env python3
"""Sweep a SOMD2 relative-FEP network for incomplete edges and replicate spread.

    python -m csbrt.audit_fep_network RUNS_DIR [--overlap-threshold 0.15]

run_fep_leg.py already refuses to publish a completion marker when a leg is
missing lambda windows, so individual failures are caught. What this adds is the
network-level view: across 50+ edges and several replicates, one absent marker is
easy to miss, and SOMD2 can exit 0 with every window dead. Run this after any
series, and before trusting a network fit.

Discovers every fep-runs-rep* directory and reports:

  1. Per-replicate completeness -- energy parquets per leg vs the configured
     lambda schedule, and the ISLAND STRUCTURE of any incomplete leg (a leg
     missing interior windows has no thermodynamic path from 0 to 1, so it
     yields no DDG no matter how it is reanalysed).
  2. A genuine-replicate test -- compares prepared-input hashes and DDG values
     across replicates to distinguish "same inputs, independent sampling" from
     "duplicate computation".
  3. Cross-replicate DDG agreement, stratified by adjacent-window overlap,
     because low-overlap edges dominate the raw spread.

Read-only: never writes into the run tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

DDG_KEY = "relative_binding_free_energy"
LEGS = ("bound", "free")


# ----------------------------------------------------------------- parsing helpers

def quantity(value) -> float | None:
    """'-1669.1386 kcal/mol' -> -1669.1386."""
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def leg_lambdas(leg_dir: Path) -> list[float]:
    out = []
    for p in leg_dir.glob("energy_traj_*.parquet"):
        try:
            out.append(float(p.name[len("energy_traj_"):-len(".parquet")]))
        except ValueError:
            pass
    return sorted(out)


def configured_num_lambda(leg_dir: Path) -> int | None:
    """num_lambda from the leg's own written config (authoritative per leg)."""
    cfg = leg_dir / "config.yaml"
    if not cfg.is_file():
        return None
    for line in cfg.read_text().splitlines():
        if line.startswith("num_lambda:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def islands(present: list[float], expected: list[float]) -> list[list[float]]:
    """Group present windows into runs that are contiguous in the expected schedule.

    Two present windows are connected only if adjacent in the FULL schedule --
    a missing interior window breaks the chain, and free energies cannot be
    summed across a break.
    """
    if not present:
        return []
    index = {round(v, 5): i for i, v in enumerate(expected)}
    idxs = sorted(index[round(v, 5)] for v in present if round(v, 5) in index)
    groups, run = [], [idxs[0]]
    for a, b in zip(idxs, idxs[1:]):
        if b == a + 1:
            run.append(b)
        else:
            groups.append(run)
            run = [b]
    groups.append(run)
    return [[expected[i] for i in g] for g in groups]


def standard_schedule(n: int) -> list[float]:
    return [i / (n - 1) for i in range(n)] if n and n > 1 else []


# ----------------------------------------------------------------- collection

def collect_edge(edge_dir: Path) -> dict:
    rec = {"edge": edge_dir.name, "legs": {}, "complete": True, "islands": {}}

    for leg in LEGS:
        d = edge_dir / leg
        if not d.is_dir():
            rec["legs"][leg] = {"present": [], "n": 0, "expected": None, "done": False}
            rec["complete"] = False
            continue
        present = leg_lambdas(d)
        n_expected = configured_num_lambda(d)
        done = (d / "fep_leg.complete.json").is_file()
        rec["legs"][leg] = {
            "present": present, "n": len(present),
            "expected": n_expected, "done": done,
        }
        if n_expected and len(present) != n_expected:
            rec["complete"] = False
            rec["islands"][leg] = islands(present, standard_schedule(n_expected))
        elif not done:
            rec["complete"] = False

    a = read_json(edge_dir / "analysis.json")
    if a and DDG_KEY in a:
        rec["ddg"] = quantity(a[DDG_KEY][0])
        rec["ddg_err"] = quantity(a[DDG_KEY][1]) if len(a[DDG_KEY]) > 1 else None
        rec["ovlp"] = min(
            (a.get("bound_adjacent_overlap_minimum", float("nan")),
             a.get("free_adjacent_overlap_minimum", float("nan"))))
        rec["status"] = a.get("status")
    else:
        rec["ddg"] = rec["ddg_err"] = rec["ovlp"] = None
        rec["status"] = None

    prep = read_json(edge_dir / "setup" / "fep_preparation.complete.json") or {}
    sha = prep.get("output_sha256", {}) or {}
    rec["bound_bss_sha"] = next((v for k, v in sha.items() if k.endswith("_bound.bss")), None)
    rec["free_bss_sha"] = next((v for k, v in sha.items() if k.endswith("_free.bss")), None)
    rec["bound_frame"] = prep.get("bound_input_coordinates")
    rec["mapped_heavy_fraction"] = prep.get("mapped_heavy_fraction")
    return rec


def collect_replicate(rep_dir: Path) -> dict[str, dict]:
    edges = {}
    for d in sorted(p for p in rep_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if d.name == "network_analysis":
            continue
        edges[d.name] = collect_edge(d)
    return edges


# ----------------------------------------------------------------- reporting

def report_completeness(reps: dict[str, dict]) -> None:
    print("=" * 78)
    print("1. PER-REPLICATE COMPLETENESS")
    print("=" * 78)
    for rep, edges in reps.items():
        bad = [e for e in edges.values() if not e["complete"]]
        n = len(edges)
        pct = (len(bad) / n * 100) if n else 0.0
        print(f"\n{rep}: {n} edges, {len(bad)} incomplete ({pct:.1f}% hard-failure rate)")
        for e in bad:
            for leg in LEGS:
                L = e["legs"][leg]
                if L["expected"] and L["n"] != L["expected"]:
                    print(f"  {e['edge']}  {leg}: {L['n']}/{L['expected']} windows")
                    print(f"    present : {' '.join(f'{v:.2f}' for v in L['present'])}")
                    isl = e["islands"].get(leg, [])
                    print(f"    islands : {len(isl)} disconnected segment(s)")
                    for seg in isl:
                        print(f"       [{' '.join(f'{v:.2f}' for v in seg)}]")
                    if len(isl) > 1:
                        print("    -> NO thermodynamic path 0->1; this edge yields no DDG.")
                        full = standard_schedule(L["expected"])
                        missing = [v for v in full
                                   if round(v, 5) not in {round(x, 5) for x in L["present"]}]
                        print(f"    -> rerun only: {missing}")


def report_replicate_genuineness(reps: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("2. ARE THESE GENUINE REPLICATES?")
    print("=" * 78)
    names = list(reps)
    if len(names) < 2:
        print("  only one replicate found; nothing to compare")
        return

    shared = set.intersection(*(set(reps[n]) for n in names))
    same_in, diff_in, same_ddg, diff_ddg = 0, 0, 0, 0
    same_frame = 0
    for edge in sorted(shared):
        recs = [reps[n][edge] for n in names]
        shas = {(r["bound_bss_sha"], r["free_bss_sha"]) for r in recs
                if r["bound_bss_sha"] or r["free_bss_sha"]}
        if len(shas) == 1:
            same_in += 1
        elif len(shas) > 1:
            diff_in += 1
        frames = {r["bound_frame"] for r in recs if r["bound_frame"]}
        if len(frames) == 1:
            same_frame += 1
        ddgs = [r["ddg"] for r in recs if r["ddg"] is not None]
        if len(ddgs) == len(names):
            if len(set(ddgs)) == 1:
                same_ddg += 1
            else:
                diff_ddg += 1

    print(f"  edges present in all {len(names)} replicates : {len(shared)}")
    print(f"  identical prepared inputs (.bss sha256)     : {same_in}")
    print(f"  differing prepared inputs                   : {diff_in}")
    print(f"  identical bound starting frame              : {same_frame}")
    print(f"  bit-identical DDG across replicates         : {same_ddg}")
    print(f"  differing DDG across replicates             : {diff_ddg}")
    print("\n  interpretation:")
    if same_in and not diff_in:
        print("   - inputs are SHARED: positions/topology identical across replicates.")
        print("     Variation can only come from velocities + the (unseeded) Langevin stream.")
    if diff_in:
        print("   - inputs DIFFER: replicates start from different prepared systems,")
        print("     so the spread includes preparation/conformational variability.")
    if same_ddg and not diff_ddg:
        print("   - DDGs are bit-identical -> NOT independent. You have 1 replicate.")
    elif diff_ddg:
        print("   - DDGs differ -> sampling is genuinely independent. Valid replicates,")
        print("     though velocity-only variation samples a narrower ensemble than")
        print("     varying the starting frame would.")


def report_agreement(reps: dict[str, dict], threshold: float) -> None:
    print("\n" + "=" * 78)
    print(f"3. CROSS-REPLICATE DDG AGREEMENT  (overlap threshold {threshold})")
    print("=" * 78)
    names = list(reps)
    shared = sorted(set.intersection(*(set(reps[n]) for n in names)))

    hdr = f"{'edge':30s}" + "".join(f"{n.replace('fep-runs-',''):>9s}" for n in names)
    print(hdr + f"{'spread':>9s}{'minOvlp':>9s}")
    good, bad, skipped = [], [], []
    for edge in shared:
        vals = [reps[n][edge]["ddg"] for n in names]
        ovs = [reps[n][edge]["ovlp"] for n in names]
        if any(v is None for v in vals) or any(o is None for o in ovs):
            missing = [n for n in names if reps[n][edge]["ddg"] is None]
            skipped.append((edge, missing))
            continue
        spread = max(vals) - min(vals)
        ov = min(ovs)
        row = f"{edge:30s}" + "".join(f"{v:9.3f}" for v in vals) + f"{spread:9.3f}{ov:9.3f}"
        flag = "" if ov >= threshold else "  <-- low overlap"
        print(row + flag)
        (good if ov >= threshold else bad).append((edge, vals, spread, ov))

    def stats(label, rows):
        if not rows:
            print(f"\n  {label}: none")
            return
        spreads = [r[2] for r in rows]
        sds = [statistics.stdev(r[1]) for r in rows if len(r[1]) > 1]
        print(f"\n  {label}: n={len(rows)}")
        print(f"    mean spread   = {statistics.fmean(spreads):.3f} kcal/mol")
        print(f"    RMS  spread   = {statistics.fmean(x * x for x in spreads) ** 0.5:.3f} kcal/mol")
        print(f"    max  spread   = {max(spreads):.3f} kcal/mol")
        if sds:
            print(f"    mean per-edge SD = {statistics.fmean(sds):.3f} kcal/mol")

    if skipped:
        print(f"\n  EXCLUDED from the statistics below: {len(skipped)} edge(s) with no "
              f"usable DDG in at least one replicate --")
        for edge, missing in skipped:
            print(f"    {edge:30s} missing in: {', '.join(missing)}")
        print("    These are NOT counted as agreement; they are unfinished work.")

    stats(f"WELL-OVERLAPPING (>= {threshold})", good)
    stats(f"LOW OVERLAP      (<  {threshold})", bad)
    if good and bad:
        rg = statistics.fmean(r[2] ** 2 for r in good) ** 0.5
        rb = statistics.fmean(r[2] ** 2 for r in bad) ** 0.5
        print(f"\n  low-overlap edges are {rb / rg:.1f}x noisier than well-overlapping ones."
              if rg else "")
        print("  Quote the WELL-OVERLAPPING RMS as your replicate uncertainty;")
        print("  the low-overlap edges need more lambda windows, not averaging.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path,
                    help="Directory containing the replicate roots")
    ap.add_argument("--overlap-threshold", type=float, default=0.15)
    ap.add_argument("--glob", default="fep-runs-rep*")
    opt = ap.parse_args()

    rep_dirs = sorted(p for p in opt.runs.glob(opt.glob) if p.is_dir())
    if not rep_dirs:
        sys.exit(f"no replicate directories matching {opt.glob!r} under {opt.runs}")
    print(f"runs root  : {opt.runs}")
    print(f"replicates : {[p.name for p in rep_dirs]}\n")

    reps = {p.name: collect_replicate(p) for p in rep_dirs}
    report_completeness(reps)
    report_replicate_genuineness(reps)
    report_agreement(reps, opt.overlap_threshold)


if __name__ == "__main__":
    main()
