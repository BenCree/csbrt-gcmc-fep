#!/usr/bin/env python3
"""Build a hydrated receptor PDB for round-two docking.

GNINA cannot place waters, which is the gap the Loch GCMC stage fills. This takes a
finished GCMC run, removes the ligand, and keeps the pocket waters GCMC placed, so the
second docking round searches a site that is hydrated rather than empty.

Two filters decide which waters survive:

* distance -- only waters within ``--water-radius`` of the ligand centroid, since bulk
  solvent tens of Angstrom away is irrelevant to docking and would bloat the receptor;
* occupancy -- optionally, only waters sitting on a hydration site whose occupancy in
  ``{prefix}-site-metrics.csv`` is at least ``--minimum-occupancy``. A site occupied 20%
  of the time is not a structural water and should not obstruct the docking search.

The ligand centroid is taken *before* stripping, so the geometry is that of the
equilibrated complex rather than the docked input.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import parmed as pmd

WATER_RESIDUE_NAMES = {"WAT", "HOH", "SOL", "TIP", "TP3"}
DEFAULT_WATER_RADIUS = 12.0


def read_site_occupancies(path: Path, minimum: float) -> np.ndarray:
    """Coordinates of hydration sites at or above the occupancy threshold."""
    kept = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                occupancy = float(row["occupancy"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path} lacks a usable 'occupancy' column") from error
            if occupancy < minimum:
                continue
            kept.append(
                [
                    float(row["x_angstrom"]),
                    float(row["y_angstrom"]),
                    float(row["z_angstrom"]),
                ]
            )
    return np.asarray(kept, dtype=float).reshape(-1, 3)


def carbon_alpha(structure: pmd.Structure) -> np.ndarray:
    indices = [atom.idx for atom in structure.atoms if atom.name == "CA"]
    if not indices:
        raise ValueError("Structure contains no CA atoms to align on")
    return np.asarray(structure.coordinates)[indices]


def kabsch_transform(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation/translation carrying `mobile` onto `target`, plus the residual RMSD."""
    if mobile.shape != target.shape:
        raise ValueError(
            f"Cannot align {mobile.shape[0]} atoms onto {target.shape[0]}"
        )
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    centered_mobile = mobile - mobile_centre
    centered_target = target - target_centre
    left, _, right_transpose = np.linalg.svd(centered_mobile.T @ centered_target)
    if np.linalg.det(left @ right_transpose) < 0:
        left[:, -1] *= -1
    rotation = left @ right_transpose
    translation = target_centre - mobile_centre @ rotation
    residual = centered_mobile @ rotation - centered_target
    rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return rotation, translation, rmsd


def map_sites_into_frame(
    sites: np.ndarray,
    alignment_reference: Path,
    structure: pmd.Structure,
    maximum_rmsd: float,
) -> tuple[np.ndarray, float]:
    """Carry hydration sites from the catalog frame into the production frame.

    `ev71_density_sites.py` superposes every trajectory onto a shared
    `--alignment-reference` receptor so that sites are comparable across runs. Those
    coordinates therefore live in the reference frame, which is tens of Angstrom from
    the production frame these coordinates are being filtered against. Comparing them
    directly silently matches nothing.
    """
    reference = pmd.load_file(str(alignment_reference))
    if reference.coordinates is None:
        raise ValueError(f"No coordinates in {alignment_reference}")
    rotation, translation, rmsd = kabsch_transform(
        carbon_alpha(reference), carbon_alpha(structure)
    )
    if rmsd > maximum_rmsd:
        raise ValueError(
            f"CA superposition of {alignment_reference} onto the production frame gave "
            f"RMSD {rmsd:.2f} A, above --maximum-alignment-rmsd; the two structures are "
            "probably not the same receptor"
        )
    return sites @ rotation + translation, rmsd


def ligand_centroid(structure: pmd.Structure, resname: str) -> np.ndarray:
    residues = [r for r in structure.residues if r.name == resname]
    if len(residues) != 1:
        raise ValueError(
            f"Expected exactly one {resname} residue; found {len(residues)}"
        )
    indices = [atom.idx for atom in residues[0].atoms]
    return np.asarray(structure.coordinates)[indices].mean(axis=0)


def select_waters(
    structure: pmd.Structure,
    centre: np.ndarray,
    radius: float,
    sites: np.ndarray | None,
    site_radius: float,
) -> list[int]:
    """Residue indices of waters to keep, by distance and optional site occupancy."""
    coordinates = np.asarray(structure.coordinates)
    keep = []
    for index, residue in enumerate(structure.residues):
        if residue.name not in WATER_RESIDUE_NAMES:
            continue
        oxygens = [a.idx for a in residue.atoms if a.atomic_number == 8]
        if not oxygens:
            continue
        position = coordinates[oxygens[0]]
        if np.linalg.norm(position - centre) > radius:
            continue
        if sites is not None and len(sites):
            if np.min(np.linalg.norm(sites - position, axis=1)) > site_radius:
                continue
        keep.append(index)
    return keep


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True,
                        help="{prefix}-production-final.prmtop from the GCMC run")
    parser.add_argument("--rst7", type=Path, required=True,
                        help="{prefix}-production-final.rst7 from the GCMC run")
    parser.add_argument("--output", type=Path, required=True,
                        help="Hydrated receptor PDB for GNINA")
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--water-radius", type=float, default=DEFAULT_WATER_RADIUS,
                        help="Keep waters within this distance of the ligand centroid")
    parser.add_argument("--site-metrics", type=Path, default=None,
                        help="{prefix}-site-metrics.csv; enables the occupancy filter")
    parser.add_argument("--alignment-reference", type=Path, default=None,
                        help="The receptor PDB ev71_density_sites.py aligned on "
                             "(preparation/receptor_input.pdb). Required with "
                             "--site-metrics: site coordinates are in that frame, not "
                             "the production frame, and are mapped across by CA Kabsch.")
    parser.add_argument("--maximum-alignment-rmsd", type=float, default=2.0,
                        help="Reject the site mapping above this CA superposition RMSD")
    parser.add_argument("--minimum-occupancy", type=float, default=0.5,
                        help="Occupancy threshold when --site-metrics is given")
    parser.add_argument("--site-assignment-radius", type=float, default=1.4,
                        help="How close a water must sit to a site to count as on it")
    parser.add_argument("--ligand-out", type=Path, default=None,
                        help="Write the stripped ligand here, in the production frame. "
                             "Round-two docking needs this as --autobox-reference: the "
                             "round-one pose is in the template frame, tens of Angstrom "
                             "away, and would box the wrong region of the receptor.")
    parser.add_argument("--keep-ligand", action="store_true",
                        help="Leave the ligand in place (diagnostic only; GNINA should "
                             "dock into an empty site)")
    opt = parser.parse_args()
    if opt.water_radius <= 0:
        raise SystemExit("--water-radius must be positive")
    if not 0.0 <= opt.minimum_occupancy <= 1.0:
        raise SystemExit("--minimum-occupancy must be in [0, 1]")
    if opt.site_metrics is not None and opt.alignment_reference is None:
        raise SystemExit(
            "--site-metrics requires --alignment-reference: hydration-site coordinates "
            "are written in the alignment frame, which is tens of Angstrom from the "
            "production frame, so filtering without the mapping silently keeps nothing"
        )
    return opt


def main() -> None:
    opt = options()
    structure = pmd.load_file(str(opt.prmtop), xyz=str(opt.rst7))
    if structure.coordinates is None:
        raise SystemExit(f"No coordinates loaded from {opt.rst7}")

    centre = ligand_centroid(structure, opt.ligand_resname)

    if opt.ligand_out is not None:
        ligand_only = pmd.load_file(str(opt.prmtop), xyz=str(opt.rst7))
        ligand_only.strip([a.residue.name != opt.ligand_resname for a in ligand_only.atoms])
        opt.ligand_out.parent.mkdir(parents=True, exist_ok=True)
        ligand_only.save(str(opt.ligand_out), overwrite=True)
        print(f"ligand (production frame) -> {opt.ligand_out}")

    sites = None
    alignment_rmsd = None
    if opt.site_metrics is not None:
        sites = read_site_occupancies(opt.site_metrics, opt.minimum_occupancy)
        if not len(sites):
            raise SystemExit(
                f"No hydration sites in {opt.site_metrics} at occupancy "
                f">= {opt.minimum_occupancy}; lower --minimum-occupancy"
            )
        sites, alignment_rmsd = map_sites_into_frame(
            sites, opt.alignment_reference, structure, opt.maximum_alignment_rmsd
        )
        print(
            f"mapped {len(sites)} sites into the production frame "
            f"(CA superposition RMSD {alignment_rmsd:.2f} A)"
        )

    total_waters = sum(1 for r in structure.residues if r.name in WATER_RESIDUE_NAMES)
    keep_waters = set(
        select_waters(structure, centre, opt.water_radius, sites,
                      opt.site_assignment_radius)
    )

    def drop_residue(index: int, residue) -> bool:
        if residue.name == opt.ligand_resname:
            return not opt.keep_ligand
        if residue.name in WATER_RESIDUE_NAMES:
            return index not in keep_waters
        # Counterions are box-neutralising bulk, not part of the site.
        return residue.name in {"Na+", "Cl-", "NA", "CL"}

    # parmed strips on a per-atom boolean mask in one pass; deleting residues
    # individually would invalidate the indices as it went.
    strip_mask = [
        drop_residue(atom.residue.idx, atom.residue) for atom in structure.atoms
    ]
    structure.strip(strip_mask)

    opt.output.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(opt.output), renumber=True)

    remaining_waters = sum(
        1 for r in structure.residues if r.name in WATER_RESIDUE_NAMES
    )
    report = {
        "prmtop": str(opt.prmtop),
        "rst7": str(opt.rst7),
        "output": str(opt.output),
        "ligand_centroid_angstrom": centre.tolist(),
        "ligand_stripped": not opt.keep_ligand,
        "waters_in_system": total_waters,
        "waters_kept": remaining_waters,
        "water_radius_angstrom": opt.water_radius,
        "occupancy_filter": None if sites is None else {
            "site_metrics": str(opt.site_metrics),
            "alignment_reference": str(opt.alignment_reference),
            "alignment_ca_rmsd_angstrom": alignment_rmsd,
            "minimum_occupancy": opt.minimum_occupancy,
            "sites_above_threshold": int(len(sites)),
            "site_assignment_radius_angstrom": opt.site_assignment_radius,
        },
    }
    report_path = opt.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if remaining_waters == 0:
        raise SystemExit(
            "No waters survived filtering; the hydrated receptor would be identical to "
            "a dry one. Raise --water-radius or lower --minimum-occupancy."
        )
    print(f"ligand centroid : {np.round(centre, 2).tolist()}")
    print(f"waters in system: {total_waters}")
    print(f"waters kept     : {remaining_waters}")
    print(f"HYDRATED_RECEPTOR={opt.output}")


if __name__ == "__main__":
    main()
