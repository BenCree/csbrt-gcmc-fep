#!/usr/bin/env python3
"""Build a consensus-hydration bound frame per ligand from all replicates.

Unlike select_bound_frames.py (which returns one replicate's endpoint), this uses
every replicate's water data. It keeps the medoid replicate's protein/ligand/bulk
unchanged and *repositions the pocket waters onto the all-replicate consensus
hydration sites*, then minimizes. So a consensus-vs-medoid comparison differs only
in pocket-water placement.

Method (density-based consensus water placement, cf. GIST / 3D-RISM / WATsite):
  1. scaffold = medoid replicate's production-final (topology + protein/ligand/bulk).
  2. consensus sites = shared-catalog sites whose mean occupancy across the
     ligand's replicates exceeds --min-occupancy, mapped into the scaffold frame.
  3. optimally assign the scaffold's in-sphere pocket waters to those sites
     (linear_sum_assignment) and rigidly translate each onto its site.
  4. energy-minimize to relieve residual clashes; validate ghost-free + finite.

Topology is unchanged from the medoid, so no prmtop surgery and the ghost-free
guarantee is inherited. Output matches select_bound_frames' layout, so it drops
into `make_fep_manifest.py --bound-frame-root`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pipeline_utils import require_file, write_json_atomic
from select_bound_frames import occupancy_vector, one_glob, production_final


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series-root", type=Path, required=True)
    p.add_argument("--common-catalog", type=Path, required=True,
                   help="Frozen common site catalog CSV (site_id, x/y/z_angstrom)")
    p.add_argument("--alignment-reference", type=Path, required=True,
                   help="Receptor PDB in the catalog's frame (density alignment target)")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--ligands", type=Path)
    p.add_argument("--replicate-glob", default="rep*")
    p.add_argument("--occupancy-column", default="occupancy")
    p.add_argument("--site-metrics-subdir", default="common_site_analysis")
    p.add_argument("--min-occupancy", type=float, default=0.5)
    p.add_argument("--sphere-radius", type=float, default=10.0,
                   help="Pocket radius (A) around the ligand to re-consensus")
    p.add_argument("--minimize-steps", type=int, default=500)
    p.add_argument("--ligand-resname", default="LIG")
    return p.parse_args()


def read_catalog(path: Path) -> tuple[list[str], np.ndarray]:
    with require_file(path).open(newline="") as h:
        rows = list(csv.DictReader(h))
    ids = [r["site_id"] for r in rows]
    xyz = np.array([[float(r["x_angstrom"]), float(r["y_angstrom"]), float(r["z_angstrom"])]
                    for r in rows], dtype=float)
    return ids, xyz


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t mapping mobile onto target (target ~ R@mobile+t)."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, tc - r @ mc


AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "CYM", "GLN", "GLU", "GLY", "HIS",
    "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "ASH", "GLH", "ACE", "NME", "NMA",
}


def protein_ca(struct) -> np.ndarray:
    """Protein C-alpha coordinates in residue order (Amber prmtops carry no chain
    IDs and renumber residues, so ordinal correspondence is the reliable match)."""
    xyz = struct.coordinates
    return np.array([xyz[a.idx] for a in struct.atoms
                     if a.name == "CA" and a.residue.name in AMINO_ACIDS])


def matched_ca(reference, scaffold) -> tuple[np.ndarray, np.ndarray]:
    ref, sca = protein_ca(reference), protein_ca(scaffold)
    if len(ref) < 3 or len(sca) < 3:
        raise ValueError(f"Too few protein CA atoms (reference {len(ref)}, scaffold {len(sca)})")
    if len(ref) != len(sca):
        raise ValueError(
            f"Reference has {len(ref)} protein CA atoms but scaffold has {len(sca)}; "
            "they must be the same receptor for ordinal CA matching")
    return ref, sca


def water_oxygens(struct) -> list[int]:
    idx = []
    for res in struct.residues:
        if res.name in ("WAT", "HOH"):
            oxy = [a for a in res.atoms if (a.element_name or a.name[:1]) == "O"]
            if len(oxy) == 1:
                idx.append(oxy[0].idx)
    return idx


def place_consensus_waters(*, scaffold_prmtop: Path, scaffold_rst7: Path,
                           reference_pdb: Path, site_coords_common: np.ndarray,
                           ligand_resname: str, sphere_radius: float,
                           minimize_steps: int, out_prmtop: Path, out_rst7: Path) -> dict:
    import parmed as pmd

    struct = pmd.load_file(str(scaffold_prmtop), str(scaffold_rst7))
    reference = pmd.load_file(str(reference_pdb))
    xyz = struct.coordinates.copy()

    # 1. map catalog sites (common frame) into the scaffold frame
    ref_ca, sca_ca = matched_ca(reference, struct)
    rot, trans = kabsch(ref_ca, sca_ca)
    sites = (rot @ site_coords_common.T).T + trans

    # 2. pocket = sites and waters within sphere_radius of any ligand heavy atom
    lig = np.array([xyz[a.idx] for a in struct.atoms
                    if a.residue.name == ligand_resname and (a.element_name or a.name[:1]) != "H"])
    if len(lig) == 0:
        raise ValueError(f"No {ligand_resname} heavy atoms in scaffold")

    def within(points: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(points[:, None, :] - lig[None, :, :], axis=2).min(axis=1)
        return d <= sphere_radius

    pocket_sites = sites[within(sites)]
    o_idx = np.array(water_oxygens(struct))
    o_xyz = xyz[o_idx]
    pocket_mask = within(o_xyz)
    pocket_o_idx = o_idx[pocket_mask]

    # 3. optimal water->site assignment, then rigid translation of each matched water
    from scipy.optimize import linear_sum_assignment
    displacements = []
    repositioned = 0
    if len(pocket_o_idx) and len(pocket_sites):
        cost = np.linalg.norm(xyz[pocket_o_idx][:, None, :] - pocket_sites[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for w, s in zip(rows, cols):
            o_atom = struct.atoms[int(pocket_o_idx[w])]
            delta = pocket_sites[s] - xyz[o_atom.idx]
            for a in o_atom.residue.atoms:      # move the whole rigid water
                xyz[a.idx] = xyz[a.idx] + delta
            displacements.append(float(np.linalg.norm(delta)))
            repositioned += 1
    struct.coordinates = xyz
    struct.save(str(out_prmtop), overwrite=True)
    struct.save(str(out_rst7), overwrite=True)

    # 4. minimize (topology is the equilibrated medoid's; PME as in production)
    from openmm import app, unit, LangevinMiddleIntegrator, Platform
    prm = app.AmberPrmtopFile(str(out_prmtop))
    inp = app.AmberInpcrdFile(str(out_rst7))
    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                              constraints=app.HBonds)
    integ = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    try:
        platform = Platform.getPlatformByName("CUDA")
    except Exception:
        platform = Platform.getPlatformByName("CPU")
    sim = app.Simulation(prm.topology, system, integ, platform)
    sim.context.setPositions(inp.positions)
    if inp.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*inp.boxVectors)
    sim.minimizeEnergy(maxIterations=minimize_steps)
    state = sim.context.getState(getPositions=True, getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    if not np.isfinite(energy):
        raise ValueError("Minimized potential energy is not finite")
    minimized = pmd.load_file(str(out_prmtop))
    minimized.positions = state.getPositions()
    minimized.box = struct.box
    minimized.save(str(out_rst7), overwrite=True)

    # inherit the ghost-free guarantee explicitly: reject a scaffold that still
    # carried zero-interaction ghost waters (an unfinalized handoff)
    try:
        from ev71_loch_common import physical_water_audit
    except ImportError:
        from loch_common import physical_water_audit
    import sire as sr
    audit = physical_water_audit(sr.load(str(out_prmtop), str(out_rst7)))
    zero = int(audit["zero_interaction_water_count"])
    if zero:
        raise ValueError(f"scaffold carried {zero} zero-interaction ghost waters; "
                         "not a clean physical handoff")

    return {
        "physical_waters": int(audit["water_molecules"]),
        "consensus_sites_in_pocket": int(len(pocket_sites)),
        "pocket_waters": int(len(pocket_o_idx)),
        "repositioned_waters": repositioned,
        "mean_displacement_angstrom": float(np.mean(displacements)) if displacements else 0.0,
        "max_displacement_angstrom": float(np.max(displacements)) if displacements else 0.0,
        "minimized_potential_energy_kcal_mol": float(energy),
        "shared_ca_atoms": int(len(ref_ca)),
    }


def consensus_occupancy(ligand_dir: Path, opt) -> dict[str, float]:
    reps = sorted(p for p in ligand_dir.glob(opt.replicate_glob) if p.is_dir())
    vectors = [occupancy_vector(one_glob(r / opt.site_metrics_subdir, "*-site-metrics.csv"),
                                opt.occupancy_column) for r in reps]
    shared = set.intersection(*(set(v) for v in vectors))
    return {s: float(np.mean([v[s] for v in vectors])) for s in shared}, len(reps)


def medoid_scaffold(ligand_dir: Path, opt) -> Path:
    reps = sorted(p for p in ligand_dir.glob(opt.replicate_glob) if p.is_dir())
    vectors = {r.name: occupancy_vector(one_glob(r / opt.site_metrics_subdir, "*-site-metrics.csv"),
                                        opt.occupancy_column) for r in reps}
    shared = sorted(set.intersection(*(set(v) for v in vectors.values())))
    names = [r.name for r in reps]
    m = np.array([[vectors[n][s] for s in shared] for n in names])
    return reps[int(np.argmin(np.linalg.norm(m - m.mean(0), axis=1)))]


def main() -> None:
    opt = options()
    opt.output_root.mkdir(parents=True, exist_ok=True)
    site_ids, site_xyz = read_catalog(opt.common_catalog)
    catalog_index = {sid: i for i, sid in enumerate(site_ids)}

    if opt.ligands is not None:
        ligands = [l for l in require_file(opt.ligands).read_text().split() if l]
    else:
        ligands = sorted(c.name for c in opt.series_root.iterdir()
                         if c.is_dir() and any(c.glob(opt.replicate_glob)))

    results, failures = [], []
    for ligand in ligands:
        try:
            ligand_dir = opt.series_root / ligand
            occ, n_reps = consensus_occupancy(ligand_dir, opt)
            keep = [s for s, o in occ.items() if o >= opt.min_occupancy and s in catalog_index]
            if not keep:
                raise ValueError(f"No sites above occupancy {opt.min_occupancy}")
            coords = np.array([site_xyz[catalog_index[s]] for s in keep])
            scaffold = medoid_scaffold(ligand_dir, opt)
            top, rst = production_final(scaffold)
            out_dir = opt.output_root / ligand
            out_dir.mkdir(parents=True, exist_ok=True)
            report = place_consensus_waters(
                scaffold_prmtop=top, scaffold_rst7=rst, reference_pdb=opt.alignment_reference,
                site_coords_common=coords, ligand_resname=opt.ligand_resname,
                sphere_radius=opt.sphere_radius, minimize_steps=opt.minimize_steps,
                out_prmtop=out_dir / "production-final.prmtop",
                out_rst7=out_dir / "production-final.rst7")
            report.update({"ligand": ligand, "replicates": n_reps,
                           "scaffold_replicate": scaffold.name,
                           "consensus_sites_selected": len(keep)})
            results.append(report)
            print(f"{ligand}: scaffold {scaffold.name}, {report['repositioned_waters']} waters "
                  f"-> consensus (mean {report['mean_displacement_angstrom']:.2f} A), "
                  f"E={report['minimized_potential_energy_kcal_mol']:.0f} kcal/mol", flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append({"ligand": ligand, "error": f"{type(error).__name__}: {error}"})
            print(f"{ligand}: FAILED — {type(error).__name__}: {error}", flush=True)

    if results:
        with (opt.output_root / "consensus_report.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    write_json_atomic(opt.output_root / "consensus_report.json",
                      {"selected": results, "failures": failures})
    print(f"\nBuilt {len(results)}/{len(ligands)} consensus frames.", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
