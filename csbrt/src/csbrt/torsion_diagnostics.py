#!/usr/bin/env python
"""Torsional-sampling diagnostics for a ligand trajectory.

An edge can show healthy adjacent-window overlap and still be wrong, because a
rotatable bond never crossed its barrier during the run. This finds the ligand's
rotatable torsions, tracks each one across the trajectory, and reports whether it
actually sampled its rotameric states or sat in one well the whole time.

    python -m csbrt.torsion_diagnostics \\
        --topology system0.prm7 --trajectory traj_0.00000.dcd \\
        --ligand-sdf ligand.sdf --output torsions.json

Why this is implemented here rather than delegated to `slow-rotations`: that
package (0.0.1) cannot run unmodified -- `torsions.py` imports
`MDAnalysis.tests.datafiles` (four names it never uses, requiring the separate
MDAnalysisTests package) and its `LigandTorsionFinder.__init__` writes
unconditionally to a hardcoded `/Users/megosato/Desktop/` path, so constructing
it raises OSError on any other machine. Both are fixable only by editing
installed source, which does not survive a reinstall. Everything needed here --
rdkit, mdtraj, pymbar, numpy -- is already in environment.yml.

Metrics per torsion:

  n_states     rotameric wells found by histogram peak detection
  transitions  well-to-well crossings observed
  occupancy    fraction of frames in each well
  g            statistical inefficiency (pymbar): correlation time in frames
  n_eff        effective independent samples = n_frames / g

A torsion with n_states > 1 and transitions == 0 is STUCK: the trajectory
carries no information about the relative populations of its wells, and any free
energy derived from it is conditioned on the arbitrary starting rotamer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Below this many frames, well detection is fitting noise.
MIN_FRAMES = 50


def rotatable_torsions(sdf_path: Path) -> tuple[list[tuple[int, int, int, int]], list[str]]:
    """Rotatable-bond torsion quadruplets, as indices into the SDF's atom order."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"no readable molecule in {sdf_path}")

    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    quads, names = [], []
    for b, c in mol.GetSubstructMatches(pattern):
        bond = mol.GetBondBetweenAtoms(b, c)
        if bond is None or bond.IsInRing():
            continue
        a = next((n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors()
                  if n.GetIdx() != c), None)
        d = next((n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()
                  if n.GetIdx() != b), None)
        if a is None or d is None:
            continue
        quads.append((a, b, c, d))
        sym = lambda i: mol.GetAtomWithIdx(i).GetSymbol()
        names.append(f"{sym(a)}{a}-{sym(b)}{b}-{sym(c)}{c}-{sym(d)}{d}")
    return quads, names


def wells(degrees: np.ndarray, bins: int = 36, min_occupancy: float = 0.02):
    """Locate rotameric wells as histogram peaks on the periodic angle axis.

    Returns (labels, n_states, occupancies). Labels assign each frame to a well.
    """
    hist, edges = np.histogram(degrees, bins=bins, range=(-180.0, 180.0))
    total = hist.sum()
    if total == 0:
        return np.zeros(len(degrees), dtype=int), 0, []

    # Peaks that are local maxima on the wrapped axis and carry real population.
    peaks = [i for i in range(bins)
             if hist[i] >= hist[(i - 1) % bins] and hist[i] >= hist[(i + 1) % bins]
             and hist[i] / total >= min_occupancy]
    if not peaks:
        peaks = [int(np.argmax(hist))]

    centres = np.array([(edges[i] + edges[i + 1]) / 2 for i in peaks])

    # Assign frames to the nearest peak centre, respecting periodicity.
    delta = np.abs(degrees[:, None] - centres[None, :])
    delta = np.minimum(delta, 360.0 - delta)
    labels = np.argmin(delta, axis=1)

    occ = [float(np.mean(labels == k)) for k in range(len(centres))]
    keep = [k for k, o in enumerate(occ) if o >= min_occupancy]
    if len(keep) < len(centres):
        centres = centres[keep]
        delta = np.abs(degrees[:, None] - centres[None, :])
        delta = np.minimum(delta, 360.0 - delta)
        labels = np.argmin(delta, axis=1)
        occ = [float(np.mean(labels == k)) for k in range(len(centres))]
    return labels, len(centres), occ


def transitions(labels: np.ndarray) -> int:
    return int(np.count_nonzero(labels[1:] != labels[:-1]))


def inefficiency(degrees: np.ndarray) -> float:
    """Statistical inefficiency g, on the unwrapped-safe sin/cos embedding."""
    try:
        from pymbar import timeseries
    except ImportError:
        return float("nan")
    rad = np.deg2rad(degrees)
    gs = []
    for series in (np.sin(rad), np.cos(rad)):
        if np.std(series) < 1e-9:
            continue
        try:
            gs.append(float(timeseries.statistical_inefficiency(series)))
        except Exception:
            pass
    return max(gs) if gs else 1.0


def analyse(topology: Path, trajectory: Path, ligand_sdf: Path,
            resname: str, stride: int) -> dict:
    import mdtraj

    traj = mdtraj.load(str(trajectory), top=str(topology), stride=stride)
    quads, names = rotatable_torsions(ligand_sdf)
    if not quads:
        return {"status": "no_rotatable_torsions", "n_frames": int(traj.n_frames),
                "torsions": []}

    sel = traj.topology.select(f"resname {resname}")
    if len(sel) == 0:
        raise ValueError(f"no atoms with resname {resname} in {topology}")
    # SDF atom order is assumed to match the ligand's order in the topology --
    # true for this pipeline, where both come from the same prepared ligand.
    if max(max(q) for q in quads) >= len(sel):
        raise ValueError(
            f"ligand in topology has {len(sel)} atoms but the SDF needs at least "
            f"{max(max(q) for q in quads) + 1}; SDF and topology disagree")
    indices = np.array([[sel[i] for i in q] for q in quads])

    angles = np.rad2deg(mdtraj.compute_dihedrals(traj, indices))

    out = []
    for k, name in enumerate(names):
        deg = angles[:, k]
        labels, n_states, occ = wells(deg)
        n_trans = transitions(labels)
        g = inefficiency(deg)
        n_eff = float(traj.n_frames / g) if g and g > 0 else float("nan")
        verdict = "ok"
        if n_states > 1 and n_trans == 0:
            verdict = "STUCK"
        elif n_states > 1 and n_trans < 5:
            verdict = "undersampled"
        elif n_eff < 10:
            verdict = "correlated"
        out.append({
            "torsion": name, "atom_indices": [int(i) for i in quads[k]],
            "n_states": n_states, "transitions": n_trans,
            "occupancy": [round(o, 3) for o in occ],
            "statistical_inefficiency": round(g, 2),
            "n_effective_samples": round(n_eff, 1),
            "verdict": verdict,
        })

    stuck = [t for t in out if t["verdict"] == "STUCK"]
    report = {
        "status": "completed",
        "topology": str(topology), "trajectory": str(trajectory),
        "n_frames": int(traj.n_frames), "n_torsions": len(out),
        "n_stuck": len(stuck), "n_undersampled": sum(1 for t in out
                                                     if t["verdict"] == "undersampled"),
        "torsions": out,
    }
    # Well detection needs enough frames to populate a histogram. Below that the
    # verdicts are artefacts of fitting wells to a handful of points, so say so
    # rather than letting a smoke-length trajectory look like a finding.
    if traj.n_frames < MIN_FRAMES:
        report["status"] = "insufficient_sampling"
        report["warning"] = (
            f"only {traj.n_frames} frames (< {MIN_FRAMES}); state assignment and "
            "transition counts are not meaningful. Treat as plumbing validation only.")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--ligand-sdf", type=Path, required=True,
                   help="prepared ligand SDF; supplies bond orders for rotatable-bond perception")
    p.add_argument("--ligand-resname", default="LIG")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output", type=Path)
    opt = p.parse_args()

    report = analyse(opt.topology, opt.trajectory, opt.ligand_sdf,
                     opt.ligand_resname, opt.stride)

    if report.get("warning"):
        print(f"WARNING: {report['warning']}")
    print(f"frames={report['n_frames']}  torsions={report.get('n_torsions', 0)}  "
          f"stuck={report.get('n_stuck', 0)}  undersampled={report.get('n_undersampled', 0)}")
    for t in report.get("torsions", []):
        print(f"  {t['torsion']:24s} states={t['n_states']} trans={t['transitions']:5d} "
              f"g={t['statistical_inefficiency']:7.2f} n_eff={t['n_effective_samples']:7.1f} "
              f" {t['verdict']}")
    if report.get("n_stuck"):
        print("\nSTUCK torsions never crossed between populated wells. Any free energy "
              "from this leg is conditioned on the starting rotamer.")

    if opt.output:
        opt.output.parent.mkdir(parents=True, exist_ok=True)
        opt.output.write_text(json.dumps(report, indent=2))
        print(f"\nTORSION_DIAGNOSTICS={opt.output}")


if __name__ == "__main__":
    main()
