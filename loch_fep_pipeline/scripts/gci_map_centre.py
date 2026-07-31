#!/usr/bin/env python3
"""Carry a GCMC sphere centre from one equilibrated frame into another.

A hydration-site coordinate is only meaningful in the frame it was measured in.
Every equilibration run produces its own frame, so a centre taken from an earlier
study cannot be used directly: it lands somewhere arbitrary, and because every
other check on the sphere is self-consistent, the resulting run looks clean while
titrating the wrong region.

This helper superposes the protein C-alpha atoms of a target structure onto a
reference structure, applies the same rigid transform to each named site, and
then runs the *same* environment guard that ``gci_window.py`` applies. It
therefore tells you whether a mapped centre is usable before any GPU time is
spent, and records the transform so the choice is reproducible.

It performs no simulation and modifies no input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import sire as sr

from gci_common import (
    add_dummy_atom,
    sphere_environment,
    suggested_num_ghosts,
    validate_sphere_environment,
)
from pipeline_utils import implementation_signature, require_file, sha256, write_json_atomic

DEFAULT_SELECTION = "name CA and protein"


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-structure",
        type=Path,
        required=True,
        help="Structure the site coordinates belong to (PDB, or a topology used "
        "together with --reference-coordinates).",
    )
    parser.add_argument(
        "--reference-coordinates",
        type=Path,
        default=None,
        help="Coordinates for --reference-structure when it is a topology.",
    )
    parser.add_argument(
        "--target-prmtop",
        type=Path,
        required=True,
        help="Topology of the frame the GCI windows will run in.",
    )
    parser.add_argument("--target-rst7", type=Path, required=True)
    parser.add_argument(
        "--site",
        nargs=5,
        action="append",
        metavar=("NAME", "X", "Y", "Z", "RADIUS"),
        required=True,
        help="A named site in the reference frame and the sphere radius to check "
        "it with, in Angstrom. Repeatable.",
    )
    parser.add_argument(
        "--selection",
        default=DEFAULT_SELECTION,
        help=f"mdtraj selection used for superposition (default: {DEFAULT_SELECTION!r}).",
    )
    parser.add_argument(
        "--max-rmsd",
        type=float,
        default=3.0,
        help="Reject the superposition above this C-alpha RMSD (Angstrom). A poor "
        "fit means the two structures are not the same protein in comparable "
        "conformations, so a mapped coordinate would be meaningless.",
    )
    parser.add_argument("--min-solute-clearance", type=float, default=2.0)
    parser.add_argument("--max-solute-distance", type=float, default=None)
    parser.add_argument("--require-waters-in-sphere", type=int, default=0)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Only map the coordinates; do not check them against the target "
        "structure. Not recommended.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write a JSON record here.")
    return parser.parse_args()


def load_frame(structure: Path, coordinates: Path | None) -> md.Trajectory:
    require_file(structure)
    if coordinates is None:
        return md.load(str(structure))
    require_file(coordinates)
    return md.load(str(coordinates), top=str(structure))


def carbon_alpha_indices(frame: md.Trajectory, selection: str) -> np.ndarray:
    indices = frame.topology.select(selection)
    if indices.size == 0:
        raise ValueError(f"Selection {selection!r} matched no atoms")
    return indices


def residue_names(frame: md.Trajectory, indices: np.ndarray) -> list[str]:
    """Return the residue name of each selected atom, in selection order.

    Names only, deliberately: residue *numbering* is a file-format convention
    (a PDB is typically 1-based while a topology read from AMBER is 0-based), so
    comparing numbers would reject a perfectly good correspondence. The ordered
    name sequence is what establishes that the two selections describe the same
    chain in the same order.
    """
    return [str(frame.topology.atom(int(index)).residue.name) for index in indices]


def residue_numbers(frame: md.Trajectory, indices: np.ndarray) -> list[int]:
    return [int(frame.topology.atom(int(index)).residue.resSeq) for index in indices]


def kabsch(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (rotation, translation, rmsd) mapping reference points onto target.

    A point ``p`` in the reference frame maps to ``rotation @ p + translation``.
    """
    reference_centroid = reference.mean(axis=0)
    target_centroid = target.mean(axis=0)
    p = reference - reference_centroid
    q = target - target_centroid
    u, _, vt = np.linalg.svd(p.T @ q)
    # Guard against an improper rotation (a reflection), which would mirror the
    # structure and place the mapped site on the wrong side of the pocket.
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rotation = vt.T @ correction @ u.T
    translation = target_centroid - rotation @ reference_centroid
    fitted = (rotation @ reference.T).T + translation
    rmsd = float(np.sqrt(((fitted - target) ** 2).sum(axis=1).mean()))
    return rotation, translation, rmsd


def main() -> None:
    opt = options()

    sites = []
    seen: set[str] = set()
    for name, x, y, z, radius in opt.site:
        if name in seen:
            raise ValueError(f"Duplicate site name {name!r}")
        seen.add(name)
        values = [float(x), float(y), float(z)]
        radius_value = float(radius)
        if not radius_value > 0.0:
            raise ValueError(f"Site {name!r} has non-positive radius {radius_value}")
        if not all(np.isfinite(values)):
            raise ValueError(f"Site {name!r} has a non-finite coordinate")
        sites.append({"name": name, "reference_centre": values, "radius_angstrom": radius_value})

    reference_frame = load_frame(opt.reference_structure, opt.reference_coordinates)
    target_frame = load_frame(opt.target_prmtop, opt.target_rst7)

    reference_indices = carbon_alpha_indices(reference_frame, opt.selection)
    target_indices = carbon_alpha_indices(target_frame, opt.selection)
    if reference_indices.size != target_indices.size:
        raise ValueError(
            f"Selection {opt.selection!r} matched {reference_indices.size} atoms in the "
            f"reference but {target_indices.size} in the target. The two structures must "
            "contain the same protein for a coordinate to be transferable."
        )
    reference_residues = residue_names(reference_frame, reference_indices)
    target_residues = residue_names(target_frame, target_indices)
    if reference_residues != target_residues:
        mismatches = [
            (position, a, b)
            for position, (a, b) in enumerate(zip(reference_residues, target_residues))
            if a != b
        ]
        raise ValueError(
            f"The superposition atoms do not correspond: {len(mismatches)} residue-name "
            f"mismatches, first at position {mismatches[0][0]} "
            f"({mismatches[0][1]} vs {mismatches[0][2]}). Refusing to map a coordinate "
            "through an ambiguous alignment."
        )
    # Informational only: a constant offset is just a numbering convention.
    offsets = {
        b - a
        for a, b in zip(
            residue_numbers(reference_frame, reference_indices),
            residue_numbers(target_frame, target_indices),
        )
    }
    residue_number_offset = offsets.pop() if len(offsets) == 1 else None
    if residue_number_offset:
        print(
            f"Residue numbering differs by a constant {residue_number_offset:+d} "
            "(a file-format convention, not a correspondence problem)",
            flush=True,
        )

    # mdtraj works in nanometres; every coordinate here is Angstrom.
    reference_points = reference_frame.xyz[0][reference_indices] * 10.0
    target_points = target_frame.xyz[0][target_indices] * 10.0
    rotation, translation, rmsd = kabsch(reference_points, target_points)
    print(
        f"Superposed {reference_indices.size} atoms ({opt.selection!r}): "
        f"RMSD {rmsd:.2f} A",
        flush=True,
    )
    if rmsd > opt.max_rmsd:
        raise RuntimeError(
            f"C-alpha RMSD {rmsd:.2f} A exceeds --max-rmsd {opt.max_rmsd:.2f} A. The "
            "structures are too dissimilar for a site coordinate to carry over."
        )

    target_system = None
    if not opt.skip_validation:
        target_system = sr.load(str(opt.target_prmtop.resolve()), str(opt.target_rst7.resolve()))

    failures = []
    for site in sites:
        centre = np.asarray(site["reference_centre"], dtype=float)
        mapped = rotation @ centre + translation
        site["mapped_centre"] = [round(float(value), 4) for value in mapped]
        print(
            f"  {site['name']}: {site['reference_centre']} -> {site['mapped_centre']} "
            f"(r = {site['radius_angstrom']:g} A)",
            flush=True,
        )
        site["suggested_num_ghosts"] = suggested_num_ghosts(site["radius_angstrom"])
        if target_system is None:
            continue
        # Validate with the same guard gci_window.py uses, so a centre that would
        # be rejected there is rejected here instead of after submission.
        with_dummy, _, reference_selection = add_dummy_atom(target_system, site["mapped_centre"])
        try:
            site["environment"] = validate_sphere_environment(
                with_dummy,
                reference_selection,
                site["radius_angstrom"],
                min_solute_clearance_angstrom=opt.min_solute_clearance,
                max_solute_distance_angstrom=opt.max_solute_distance,
                require_waters_in_sphere=opt.require_waters_in_sphere,
            )
            site["usable"] = True
            print(
                f"    PASS nearest solute heavy atom "
                f"{site['environment']['nearest_solute_heavy_atom_angstrom']:.2f} A; "
                f"inside the sphere "
                f"{site['environment']['solute_heavy_atoms_within_radius']} solute heavy "
                f"atoms and {site['environment']['water_oxygens_within_radius']} water "
                f"oxygens",
                flush=True,
            )
        except RuntimeError as error:
            # Report every site before failing, so one bad coordinate does not
            # hide the state of the others.
            site["usable"] = False
            site["rejection"] = str(error)
            site["environment"] = sphere_environment(
                with_dummy,
                reference_selection,
                site["radius_angstrom"],
                probe_radius_angstrom=site["radius_angstrom"] + 8.0,
            )
            failures.append(site["name"])
            print(f"    REJECTED {error}", flush=True)

    if target_system is not None and not failures:
        print("\nReady-to-run centres:", flush=True)
        for site in sites:
            x, y, z = site["mapped_centre"]
            print(
                f"  # {site['name']}\n"
                f"  --sphere-centre {x} {y} {z} --sphere-radius "
                f"{site['radius_angstrom']:g} --num-ghosts {site['suggested_num_ghosts']}",
                flush=True,
            )

    record = {
        "stage": "gci_map_centre",
        "reference_structure": str(opt.reference_structure.resolve()),
        "reference_structure_sha256": sha256(opt.reference_structure),
        "reference_coordinates": str(opt.reference_coordinates.resolve())
        if opt.reference_coordinates
        else None,
        "target_prmtop": str(opt.target_prmtop.resolve()),
        "target_prmtop_sha256": sha256(opt.target_prmtop),
        "target_rst7": str(opt.target_rst7.resolve()),
        "target_rst7_sha256": sha256(opt.target_rst7),
        "selection": opt.selection,
        "superposed_atoms": int(reference_indices.size),
        "residue_number_offset": residue_number_offset,
        "rmsd_angstrom": rmsd,
        "max_rmsd_angstrom": opt.max_rmsd,
        "rotation": [[float(value) for value in row] for row in rotation],
        "translation": [float(value) for value in translation],
        "validated": not opt.skip_validation,
        "sites": sites,
        "implementation": implementation_signature(
            sources={
                "gci_map_centre.py": Path(__file__),
                "gci_common.py": Path(__file__).with_name("gci_common.py"),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("sire", "OpenMM", "mdtraj"),
            modules=("sire", "openmm", "mdtraj"),
        ),
    }
    if opt.output is not None:
        write_json_atomic(opt.output, record)
        print(f"\nWrote {opt.output}", flush=True)
    else:
        print(json.dumps({key: record[key] for key in ("rmsd_angstrom", "sites")}, indent=2))

    if failures:
        raise SystemExit(
            f"{len(failures)} mapped centre(s) are not usable in the target frame: "
            f"{', '.join(failures)}"
        )


if __name__ == "__main__":
    main()
