#!/usr/bin/env python3
"""Scalable hydration-density and common-site analysis for EV71 Loch runs.

This is the scientific-series companion to ``ev71_postprocess.py``.  The
original postprocessor preserves Ludovic's exact average-linkage clustering
definition for parity and crystal-water validation.  This script instead uses
an O(frames * waters) density grid, extracts reproducible candidate hydration
sites, and assigns at most one physical water to each site in each frame.

Run without ``--site-catalog`` to discover provisional sites.  Re-run every
ligand/replica with the same saved catalog to obtain directly comparable site
occupancy, ligand-overlap, and geometric water-bridge measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time
from typing import Any

import mdtraj as md
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    implementation_signature,
    invalidate_checkpoint,
    read_ghost_records,
    require_file,
    sha256,
    write_json_atomic,
)


WATER_NAMES = {"wat", "hoh"}
BULK_WATER_NUMBER_DENSITY_A3 = 0.0334


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--ghost-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument(
        "--alignment-reference",
        type=Path,
        help="Optional common receptor/PDB reference for cross-run coordinates",
    )
    parser.add_argument(
        "--site-catalog",
        type=Path,
        help="CSV catalog to reuse instead of discovering sites from this run",
    )
    parser.add_argument("--sphere-radius", type=float, default=10.0)
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--smoothing-sigma", type=float, default=0.75)
    parser.add_argument("--minimum-relative-density", type=float, default=1.5)
    parser.add_argument("--minimum-peak-separation", type=float, default=2.4)
    parser.add_argument("--site-assignment-radius", type=float, default=1.4)
    parser.add_argument("--minimum-site-occupancy", type=float, default=0.20)
    parser.add_argument("--maximum-sites", type=int, default=250)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--ligand-overlap-cutoff", type=float, default=2.4)
    parser.add_argument("--hbond-distance-cutoff", type=float, default=3.5)
    parser.add_argument("--hbond-angle-cutoff", type=float, default=150.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_options(opt: argparse.Namespace) -> None:
    positive = {
        "sphere radius": opt.sphere_radius,
        "grid spacing": opt.grid_spacing,
        "smoothing sigma": opt.smoothing_sigma,
        "minimum peak separation": opt.minimum_peak_separation,
        "site assignment radius": opt.site_assignment_radius,
        "ligand overlap cutoff": opt.ligand_overlap_cutoff,
        "hydrogen-bond distance cutoff": opt.hbond_distance_cutoff,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    if opt.minimum_relative_density < 0 or not 0 <= opt.minimum_site_occupancy <= 1:
        raise ValueError("Density threshold must be nonnegative and occupancy must be in [0, 1]")
    if opt.maximum_sites < 1 or opt.blocks < 1:
        raise ValueError("--maximum-sites and --blocks must be positive integers")
    if not 0 < opt.hbond_angle_cutoff <= 180:
        raise ValueError("--hbond-angle-cutoff must be in (0, 180]")


def ligand_indices(topology: md.Topology, resname: str) -> tuple[np.ndarray, np.ndarray]:
    residues = [residue for residue in topology.residues if residue.name == resname]
    if len(residues) != 1:
        raise ValueError(f"Expected exactly one {resname!r} residue; found {len(residues)}")
    atoms = list(residues[0].atoms)
    all_indices = np.asarray([atom.index for atom in atoms], dtype=np.int64)
    heavy = np.asarray(
        [atom.index for atom in atoms if atom.element is None or atom.element.symbol != "H"],
        dtype=np.int64,
    )
    if not len(heavy):
        raise ValueError("Ligand has no heavy atoms")
    return all_indices, heavy


def water_records(topology: md.Topology) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    oxygen_atoms: list[int] = []
    residue_indices: list[int] = []
    residue_hydrogens: dict[int, np.ndarray] = {}
    for residue in topology.residues:
        if residue.name.lower() not in WATER_NAMES:
            continue
        oxygens = [atom.index for atom in residue.atoms if atom.element is not None and atom.element.symbol == "O"]
        hydrogens = [atom.index for atom in residue.atoms if atom.element is not None and atom.element.symbol == "H"]
        if len(oxygens) != 1:
            raise ValueError(f"Water residue {residue.index} has {len(oxygens)} oxygens")
        oxygen_atoms.append(oxygens[0])
        residue_indices.append(residue.index)
        residue_hydrogens[residue.index] = np.asarray(hydrogens, dtype=np.int64)
    if not oxygen_atoms:
        raise ValueError("No water oxygen atoms found")
    return (
        np.asarray(oxygen_atoms, dtype=np.int64),
        np.asarray(residue_indices, dtype=np.int64),
        residue_hydrogens,
    )


def ca_key(atom: md.core.topology.Atom) -> tuple[int, int, str]:
    return atom.residue.chain.index, atom.residue.resSeq, atom.name


def image_and_align(
    trajectory: md.Trajectory,
    reference_path: Path | None,
) -> tuple[list[int], str]:
    trajectory.image_molecules(inplace=True)
    mobile_atoms = [
        atom
        for atom in trajectory.topology.atoms
        if atom.residue.is_protein and atom.name == "CA"
    ]
    if not mobile_atoms:
        raise ValueError("No protein C-alpha atoms found for alignment")
    mobile_by_key = {ca_key(atom): atom.index for atom in mobile_atoms}
    if len(mobile_by_key) != len(mobile_atoms):
        raise ValueError("Protein C-alpha alignment keys are not unique")

    if reference_path is None:
        indices = [atom.index for atom in mobile_atoms]
        trajectory.superpose(trajectory, frame=0, atom_indices=indices)
        return indices, "trajectory_frame_0"

    reference = md.load(str(require_file(reference_path)), discard_overlapping_frames=False)
    reference_atoms = [
        atom
        for atom in reference.topology.atoms
        if atom.residue.is_protein and atom.name == "CA"
    ]
    reference_by_key = {ca_key(atom): atom.index for atom in reference_atoms}
    common = sorted(set(mobile_by_key) & set(reference_by_key))
    if len(common) == len(mobile_atoms) == len(reference_atoms):
        mobile_indices = [mobile_by_key[key] for key in common]
        reference_indices = [reference_by_key[key] for key in common]
    else:
        # tLEaP can renumber an otherwise unchanged receptor when it combines
        # receptor and ligand.  Ordered matching is safe only after proving
        # that the complete C-alpha residue-name sequence is identical.
        mobile_sequence = [atom.residue.name for atom in mobile_atoms]
        reference_sequence = [atom.residue.name for atom in reference_atoms]
        if len(mobile_atoms) != len(reference_atoms) or mobile_sequence != reference_sequence:
            missing_mobile = sorted(set(reference_by_key) - set(mobile_by_key))[:5]
            missing_reference = sorted(set(mobile_by_key) - set(reference_by_key))[:5]
            raise ValueError(
                "Alignment reference C-alpha identity differs from trajectory: "
                f"missing_mobile={missing_mobile}, missing_reference={missing_reference}"
            )
        mobile_indices = [atom.index for atom in mobile_atoms]
        reference_indices = [atom.index for atom in reference_atoms]
    trajectory.superpose(
        reference,
        frame=0,
        atom_indices=mobile_indices,
        ref_atom_indices=reference_indices,
    )
    return mobile_indices, str(reference_path.resolve())


def physical_sphere_waters(
    trajectory: md.Trajectory,
    inactive: list[list[int]],
    ligand_atoms: np.ndarray,
    oxygen_atoms: np.ndarray,
    oxygen_residues: np.ndarray,
    radius: float,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    if len(inactive) != trajectory.n_frames:
        raise ValueError(
            f"Ghost history has {len(inactive)} records; trajectory has {trajectory.n_frames} frames"
        )
    frame_xyz: list[np.ndarray] = []
    frame_residues: list[np.ndarray] = []
    centres = np.empty((trajectory.n_frames, 3), dtype=np.float64)
    for frame in range(trajectory.n_frames):
        centre_nm = trajectory.xyz[frame, ligand_atoms].mean(axis=0)
        centres[frame] = 10.0 * centre_nm
        oxygen_xyz = 10.0 * trajectory.xyz[frame, oxygen_atoms]
        distance = np.linalg.norm(oxygen_xyz - centres[frame], axis=1)
        inactive_mask = np.isin(oxygen_residues, inactive[frame])
        inside = (~inactive_mask) & (distance <= radius)
        frame_xyz.append(oxygen_xyz[inside].astype(np.float64, copy=False))
        frame_residues.append(oxygen_residues[inside].copy())
    return frame_xyz, frame_residues, centres


def grid_geometry(centres: np.ndarray, radius: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    centre = np.median(centres, axis=0)
    origin = np.floor((centre - radius - spacing) / spacing) * spacing
    upper = np.ceil((centre + radius + spacing) / spacing) * spacing
    shape = np.ceil((upper - origin) / spacing).astype(int) + 1
    return origin, shape


def accumulate_density(
    waters: list[np.ndarray],
    origin: np.ndarray,
    shape: np.ndarray,
    spacing: float,
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    counts = np.zeros(tuple(shape), dtype=np.float32)
    observations = 0
    for coordinates in waters:
        if not len(coordinates):
            continue
        voxels = np.floor((coordinates - origin) / spacing).astype(np.int64)
        valid = np.all((voxels >= 0) & (voxels < shape), axis=1)
        voxels = voxels[valid]
        np.add.at(counts, (voxels[:, 0], voxels[:, 1], voxels[:, 2]), 1.0)
        observations += len(voxels)
    smoothed = gaussian_filter(
        counts,
        sigma=smoothing_sigma / spacing,
        mode="constant",
        cval=0.0,
    )
    density = smoothed / (len(waters) * spacing**3)
    relative = density / BULK_WATER_NUMBER_DENSITY_A3
    return density.astype(np.float32), relative.astype(np.float32), observations


def assign_sites(
    waters: list[np.ndarray],
    water_residues: list[np.ndarray],
    sites: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    n_frames = len(waters)
    n_sites = len(sites)
    occupied = np.zeros((n_frames, n_sites), dtype=bool)
    assigned_residue = np.full((n_frames, n_sites), -1, dtype=np.int32)
    assigned_xyz: list[list[np.ndarray]] = [[] for _ in range(n_sites)]
    for frame, (coordinates, residues) in enumerate(zip(waters, water_residues, strict=True)):
        if not len(coordinates) or not n_sites:
            continue
        distance = np.linalg.norm(coordinates[:, None, :] - sites[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(distance)
        keep = distance[rows, columns] <= cutoff
        rows = rows[keep]
        columns = columns[keep]
        occupied[frame, columns] = True
        assigned_residue[frame, columns] = residues[rows]
        for water_index, site_index in zip(rows, columns, strict=True):
            assigned_xyz[int(site_index)].append(coordinates[int(water_index)])
    return occupied, assigned_residue, assigned_xyz


def detect_sites(
    relative_density: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    waters: list[np.ndarray],
    water_residues: list[np.ndarray],
    threshold: float,
    separation: float,
    assignment_radius: float,
    minimum_occupancy: float,
    maximum_sites: int,
) -> tuple[np.ndarray, np.ndarray]:
    half_width = max(1, int(math.ceil(separation / spacing)))
    size = 2 * half_width + 1
    local_max = maximum_filter(relative_density, size=size, mode="constant")
    candidate_voxels = np.argwhere(
        (relative_density >= threshold) & np.isclose(relative_density, local_max)
    )
    if not len(candidate_voxels):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
    candidate_values = relative_density[tuple(candidate_voxels.T)]
    order = np.argsort(-candidate_values)
    accepted: list[np.ndarray] = []
    accepted_density: list[float] = []
    for index in order:
        xyz = origin + (candidate_voxels[index].astype(np.float64) + 0.5) * spacing
        if accepted and np.min(np.linalg.norm(np.asarray(accepted) - xyz, axis=1)) < separation:
            continue
        accepted.append(xyz)
        accepted_density.append(float(candidate_values[index]))
        if len(accepted) >= maximum_sites:
            break
    sites = np.asarray(accepted, dtype=np.float64)
    density_values = np.asarray(accepted_density, dtype=np.float64)
    occupied, _, assigned_xyz = assign_sites(
        waters, water_residues, sites, assignment_radius
    )
    keep = occupied.mean(axis=0) >= minimum_occupancy
    sites = sites[keep]
    density_values = density_values[keep]
    assigned_xyz = [values for values, retain in zip(assigned_xyz, keep, strict=True) if retain]
    for index, values in enumerate(assigned_xyz):
        if values:
            sites[index] = np.asarray(values).mean(axis=0)
    return sites, density_values


def read_catalog(path: Path) -> tuple[list[str], np.ndarray]:
    with require_file(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"site_id", "x_angstrom", "y_angstrom", "z_angstrom"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Site catalog must contain {sorted(required)}")
    identifiers = [row["site_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Site catalog identifiers are not unique")
    coordinates = np.asarray(
        [[float(row[axis]) for axis in ("x_angstrom", "y_angstrom", "z_angstrom")] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("Site catalog contains non-finite coordinates")
    return identifiers, coordinates


def block_statistics(occupied: np.ndarray, blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = [indices for indices in np.array_split(np.arange(len(occupied)), blocks) if len(indices)]
    values = np.asarray([occupied[indices].mean(axis=0) for indices in splits])
    mean = occupied.mean(axis=0)
    if len(values) > 1:
        standard_error = values.std(axis=0, ddof=1) / math.sqrt(len(values))
    else:
        standard_error = np.zeros(occupied.shape[1], dtype=np.float64)
    return mean, standard_error, values


def site_density_values(
    relative_density: np.ndarray, origin: np.ndarray, spacing: float, sites: np.ndarray
) -> np.ndarray:
    voxels = np.floor((sites - origin) / spacing).astype(int)
    values = np.zeros(len(sites), dtype=np.float64)
    shape = np.asarray(relative_density.shape)
    valid = np.all((voxels >= 0) & (voxels < shape), axis=1)
    values[valid] = relative_density[tuple(voxels[valid].T)]
    return values


def ligand_overlap(
    trajectory: md.Trajectory,
    ligand_heavy: np.ndarray,
    sites: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = 10.0 * trajectory.xyz[:, ligand_heavy, :]
    distance = np.linalg.norm(xyz[:, :, None, :] - sites[None, None, :, :], axis=3).min(axis=1)
    return (distance <= cutoff).mean(axis=0), np.median(distance, axis=0), distance.min(axis=0)


def donor_acceptor_groups(
    topology: md.Topology, ligand_resname: str
) -> dict[str, dict[str, Any]]:
    bonded: dict[int, set[int]] = {atom.index: set() for atom in topology.atoms}
    for left, right in topology.bonds:
        bonded[left.index].add(right.index)
        bonded[right.index].add(left.index)

    groups: dict[str, dict[str, Any]] = {}
    for name in ("protein", "ligand"):
        if name == "protein":
            atoms = [atom for atom in topology.atoms if atom.residue.is_protein]
        else:
            atoms = [atom for atom in topology.atoms if atom.residue.name == ligand_resname]
        atom_set = {atom.index for atom in atoms}
        acceptors = [
            atom.index
            for atom in atoms
            if atom.element is not None and atom.element.symbol in {"N", "O", "S"}
        ]
        donors: list[tuple[int, int]] = []
        for atom in atoms:
            if atom.element is None or atom.element.symbol not in {"N", "O", "S"}:
                continue
            for neighbour in bonded[atom.index]:
                other = topology.atom(neighbour)
                if neighbour in atom_set and other.element is not None and other.element.symbol == "H":
                    donors.append((atom.index, neighbour))
        groups[name] = {
            "acceptors": np.asarray(acceptors, dtype=np.int64),
            "donors": np.asarray(donors, dtype=np.int64).reshape((-1, 2)),
        }
    return groups


def angle_degrees(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> np.ndarray:
    left = first - vertex
    right = last - vertex
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    cosine = np.divide(
        np.sum(left * right, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 0,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def water_hbonds_group(
    frame_xyz: np.ndarray,
    water_oxygen: int,
    water_hydrogens: np.ndarray,
    group: dict[str, Any],
    distance_cutoff: float,
    angle_cutoff: float,
) -> bool:
    oxygen_xyz = frame_xyz[water_oxygen]
    acceptors = group["acceptors"]
    if len(water_hydrogens) and len(acceptors):
        acceptor_xyz = frame_xyz[acceptors]
        close = np.linalg.norm(acceptor_xyz - oxygen_xyz, axis=1) <= distance_cutoff
        if np.any(close):
            acceptor_xyz = acceptor_xyz[close]
            for hydrogen in water_hydrogens:
                h_xyz = frame_xyz[hydrogen]
                first = np.broadcast_to(oxygen_xyz, acceptor_xyz.shape)
                vertex = np.broadcast_to(h_xyz, acceptor_xyz.shape)
                if np.any(angle_degrees(first, vertex, acceptor_xyz) >= angle_cutoff):
                    return True
    donors = group["donors"]
    if len(donors):
        donor_xyz = frame_xyz[donors[:, 0]]
        close = np.linalg.norm(donor_xyz - oxygen_xyz, axis=1) <= distance_cutoff
        if np.any(close):
            donor_xyz = donor_xyz[close]
            hydrogen_xyz = frame_xyz[donors[close, 1]]
            water_xyz = np.broadcast_to(oxygen_xyz, donor_xyz.shape)
            if np.any(angle_degrees(donor_xyz, hydrogen_xyz, water_xyz) >= angle_cutoff):
                return True
    return False


def bridge_statistics(
    trajectory: md.Trajectory,
    assigned_residues: np.ndarray,
    oxygen_residues: np.ndarray,
    oxygen_atoms: np.ndarray,
    water_hydrogens: dict[int, np.ndarray],
    ligand_resname: str,
    distance_cutoff: float,
    angle_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    residue_to_oxygen = {
        int(residue): int(oxygen)
        for residue, oxygen in zip(oxygen_residues, oxygen_atoms, strict=True)
    }
    groups = donor_acceptor_groups(trajectory.topology, ligand_resname)
    n_frames, n_sites = assigned_residues.shape
    protein_contact = np.zeros((n_frames, n_sites), dtype=bool)
    ligand_contact = np.zeros((n_frames, n_sites), dtype=bool)
    for frame in range(n_frames):
        xyz = 10.0 * trajectory.xyz[frame]
        for site in np.flatnonzero(assigned_residues[frame] >= 0):
            residue = int(assigned_residues[frame, site])
            oxygen = residue_to_oxygen[residue]
            hydrogens = water_hydrogens[residue]
            protein_contact[frame, site] = water_hbonds_group(
                xyz, oxygen, hydrogens, groups["protein"], distance_cutoff, angle_cutoff
            )
            ligand_contact[frame, site] = water_hbonds_group(
                xyz, oxygen, hydrogens, groups["ligand"], distance_cutoff, angle_cutoff
            )
    bridge = protein_contact & ligand_contact
    note = (
        "Geometry-only candidate hydrogen bonds: N/O/S donors and acceptors, "
        "heavy-atom distance and D-H-A angle cutoffs; no protonation-aware acceptor typing."
    )
    return (
        protein_contact.mean(axis=0),
        ligand_contact.mean(axis=0),
        bridge.mean(axis=0),
        bridge,
        note,
    )


def write_dx(path: Path, density: np.ndarray, origin: np.ndarray, spacing: float) -> None:
    nx, ny, nz = density.shape
    values = density.ravel(order="C")
    with path.open("w") as handle:
        handle.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        handle.write(f"origin {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
        handle.write(f"delta {spacing:.6f} 0 0\n")
        handle.write(f"delta 0 {spacing:.6f} 0\n")
        handle.write(f"delta 0 0 {spacing:.6f}\n")
        handle.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        handle.write(f"object 3 class array type double rank 0 items {len(values)} data follows\n")
        for start in range(0, len(values), 3):
            handle.write(" ".join(f"{value:.7g}" for value in values[start : start + 3]) + "\n")
        handle.write('attribute "dep" string "positions"\n')
        handle.write('object "water relative density" class field\n')
        handle.write('component "positions" value 1\n')
        handle.write('component "connections" value 2\n')
        handle.write('component "data" value 3\n')


def write_catalog(path: Path, identifiers: list[str], sites: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("site_id", "x_angstrom", "y_angstrom", "z_angstrom"))
        for identifier, xyz in zip(identifiers, sites, strict=True):
            writer.writerow((identifier, *[f"{value:.6f}" for value in xyz]))


def write_site_pdb(
    path: Path,
    identifiers: list[str],
    sites: np.ndarray,
    occupancy: np.ndarray,
    relative_density: np.ndarray,
) -> None:
    with path.open("w") as handle:
        handle.write("REMARK Hydration sites: occupancy in occupancy, relative density in B-factor\n")
        for serial, (identifier, xyz, occ, density) in enumerate(
            zip(identifiers, sites, occupancy, relative_density, strict=True), start=1
        ):
            handle.write(f"REMARK SITE {serial} ID {identifier}\n")
            handle.write(
                "HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2s}\n".format(
                    serial, "O", "WAT", serial, *xyz, min(float(occ), 1.0), min(float(density), 99.99), "O"
                )
            )
        handle.write("END\n")


def main() -> None:
    started = time.time()
    opt = options()
    validate_options(opt)
    topology_path = require_file(opt.topology)
    trajectory_path = require_file(opt.trajectory)
    ghost_path = require_file(opt.ghost_file)
    reference_path = require_file(opt.alignment_reference) if opt.alignment_reference else None
    catalog_input = require_file(opt.site_catalog) if opt.site_catalog else None
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    density_npz = output / f"{opt.prefix}-water-density.npz"
    density_dx = output / f"{opt.prefix}-water-relative-density.dx"
    catalog_csv = output / f"{opt.prefix}-site-catalog.csv"
    metrics_csv = output / f"{opt.prefix}-site-metrics.csv"
    frame_npz = output / f"{opt.prefix}-frame-site-series.npz"
    sites_pdb = output / f"{opt.prefix}-hydration-sites.pdb"
    summary_json = output / f"{opt.prefix}-density-analysis.json"
    marker = output / "density_analysis.complete.json"
    expected = [density_npz, density_dx, catalog_csv, metrics_csv, frame_npz, sites_pdb, summary_json]

    signature = {
        "topology_sha256": sha256(topology_path),
        "trajectory_sha256": sha256(trajectory_path),
        "ghost_file_sha256": sha256(ghost_path),
        "alignment_reference_sha256": sha256(reference_path) if reference_path else None,
        "site_catalog_sha256": sha256(catalog_input) if catalog_input else None,
        "prefix": opt.prefix,
        "ligand_resname": opt.ligand_resname,
        "sphere_radius_angstrom": opt.sphere_radius,
        "grid_spacing_angstrom": opt.grid_spacing,
        "smoothing_sigma_angstrom": opt.smoothing_sigma,
        "minimum_relative_density": opt.minimum_relative_density,
        "minimum_peak_separation_angstrom": opt.minimum_peak_separation,
        "site_assignment_radius_angstrom": opt.site_assignment_radius,
        "minimum_site_occupancy": opt.minimum_site_occupancy,
        "maximum_sites": opt.maximum_sites,
        "blocks": opt.blocks,
        "ligand_overlap_cutoff_angstrom": opt.ligand_overlap_cutoff,
        "hbond_distance_cutoff_angstrom": opt.hbond_distance_cutoff,
        "hbond_angle_cutoff_degrees": opt.hbond_angle_cutoff,
        "implementation": implementation_signature(
            sources={
                "ev71_density_sites.py": Path(__file__),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("mdtraj", "numpy", "scipy"),
            modules=("mdtraj", "scipy.ndimage", "scipy.optimize"),
        ),
    }
    if not opt.force and checkpoint_matches(marker, signature=signature, outputs=expected):
        print(f"Density-analysis checkpoint is valid: {marker}", flush=True)
        return
    invalidate_checkpoint(marker)

    print(f"Loading {trajectory_path}", flush=True)
    trajectory = md.load(str(trajectory_path), top=str(topology_path), discard_overlapping_frames=False)
    ca_atoms, alignment_reference = image_and_align(trajectory, reference_path)
    inactive = read_ghost_records(ghost_path)
    ligand_all, ligand_heavy = ligand_indices(trajectory.topology, opt.ligand_resname)
    oxygen_atoms, oxygen_residues, water_hydrogens = water_records(trajectory.topology)
    waters, water_residue_frames, sphere_centres = physical_sphere_waters(
        trajectory,
        inactive,
        ligand_all,
        oxygen_atoms,
        oxygen_residues,
        opt.sphere_radius,
    )
    origin, shape = grid_geometry(sphere_centres, opt.sphere_radius, opt.grid_spacing)
    density, relative_density, observations = accumulate_density(
        waters, origin, shape, opt.grid_spacing, opt.smoothing_sigma
    )

    if catalog_input:
        identifiers, sites = read_catalog(catalog_input)
        catalog_mode = "reused_common_catalog"
    else:
        sites, _ = detect_sites(
            relative_density,
            origin,
            opt.grid_spacing,
            waters,
            water_residue_frames,
            opt.minimum_relative_density,
            opt.minimum_peak_separation,
            opt.site_assignment_radius,
            opt.minimum_site_occupancy,
            opt.maximum_sites,
        )
        identifiers = [f"HS{index:03d}" for index in range(1, len(sites) + 1)]
        catalog_mode = "discovered_from_this_run"

    occupied, assigned_residues, assigned_xyz = assign_sites(
        waters, water_residue_frames, sites, opt.site_assignment_radius
    )
    occupancy, occupancy_se, block_values = block_statistics(occupied, opt.blocks)
    relative_at_site = site_density_values(relative_density, origin, opt.grid_spacing, sites)
    position_rmsf = np.zeros(len(sites), dtype=np.float64)
    for index, values in enumerate(assigned_xyz):
        if values:
            coordinates = np.asarray(values)
            position_rmsf[index] = float(
                np.sqrt(np.mean(np.sum((coordinates - sites[index]) ** 2, axis=1)))
            )
        else:
            position_rmsf[index] = math.nan
    overlap_fraction, ligand_median_distance, ligand_minimum_distance = ligand_overlap(
        trajectory, ligand_heavy, sites, opt.ligand_overlap_cutoff
    )
    protein_hbond, ligand_hbond, bridge_fraction, bridge_series, bridge_note = bridge_statistics(
        trajectory,
        assigned_residues,
        oxygen_residues,
        oxygen_atoms,
        water_hydrogens,
        opt.ligand_resname,
        opt.hbond_distance_cutoff,
        opt.hbond_angle_cutoff,
    )

    np.savez_compressed(
        density_npz,
        density_per_angstrom3=density,
        relative_to_bulk=relative_density,
        origin_angstrom=origin,
        spacing_angstrom=np.asarray(opt.grid_spacing),
        sphere_centres_angstrom=sphere_centres,
    )
    write_dx(density_dx, relative_density, origin, opt.grid_spacing)
    write_catalog(catalog_csv, identifiers, sites)
    write_site_pdb(sites_pdb, identifiers, sites, occupancy, relative_at_site)
    np.savez_compressed(
        frame_npz,
        site_ids=np.asarray(identifiers),
        occupied=occupied,
        assigned_water_residue=assigned_residues,
        geometric_bridge=bridge_series,
        block_occupancy=block_values,
    )

    fieldnames = [
        "site_id",
        "x_angstrom",
        "y_angstrom",
        "z_angstrom",
        "occupancy",
        "occupancy_block_standard_error",
        "relative_water_density",
        "assigned_position_rmsf_angstrom",
        "ligand_overlap_fraction",
        "ligand_median_distance_angstrom",
        "ligand_minimum_distance_angstrom",
        "water_protein_hbond_fraction",
        "water_ligand_hbond_fraction",
        "water_bridge_fraction",
    ]
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, identifier in enumerate(identifiers):
            writer.writerow(
                {
                    "site_id": identifier,
                    "x_angstrom": f"{sites[index, 0]:.6f}",
                    "y_angstrom": f"{sites[index, 1]:.6f}",
                    "z_angstrom": f"{sites[index, 2]:.6f}",
                    "occupancy": f"{occupancy[index]:.8f}",
                    "occupancy_block_standard_error": f"{occupancy_se[index]:.8f}",
                    "relative_water_density": f"{relative_at_site[index]:.8f}",
                    "assigned_position_rmsf_angstrom": f"{position_rmsf[index]:.8f}",
                    "ligand_overlap_fraction": f"{overlap_fraction[index]:.8f}",
                    "ligand_median_distance_angstrom": f"{ligand_median_distance[index]:.8f}",
                    "ligand_minimum_distance_angstrom": f"{ligand_minimum_distance[index]:.8f}",
                    "water_protein_hbond_fraction": f"{protein_hbond[index]:.8f}",
                    "water_ligand_hbond_fraction": f"{ligand_hbond[index]:.8f}",
                    "water_bridge_fraction": f"{bridge_fraction[index]:.8f}",
                }
            )

    sphere_counts = np.asarray([len(values) for values in waters], dtype=np.int64)
    summary = {
        "status": "completed",
        "purpose": "Project 2 hydration-site occupancy and ligand-displacement analysis",
        "trajectory_frames": trajectory.n_frames,
        "protein_ca_atoms": len(ca_atoms),
        "alignment_reference": alignment_reference,
        "physical_water_observations": observations,
        "sphere_water_count": {
            "minimum": int(sphere_counts.min()),
            "mean": float(sphere_counts.mean()),
            "maximum": int(sphere_counts.max()),
        },
        "grid": {
            "origin_angstrom": origin.tolist(),
            "shape": shape.tolist(),
            "spacing_angstrom": opt.grid_spacing,
            "smoothing_sigma_angstrom": opt.smoothing_sigma,
            "bulk_water_number_density_per_angstrom3": BULK_WATER_NUMBER_DENSITY_A3,
        },
        "site_catalog_mode": catalog_mode,
        "site_catalog_input": str(catalog_input) if catalog_input else None,
        "hydration_sites": len(sites),
        "site_assignment": "one-to-one minimum-distance assignment per frame",
        "blocks_used": len(block_values),
        "hydrogen_bond_interpretation": bridge_note,
        "outputs": {
            "density_npz": density_npz.name,
            "relative_density_dx": density_dx.name,
            "site_catalog_csv": catalog_csv.name,
            "site_metrics_csv": metrics_csv.name,
            "frame_site_series_npz": frame_npz.name,
            "hydration_sites_pdb": sites_pdb.name,
        },
        "wall_seconds": time.time() - started,
    }
    write_json_atomic(summary_json, summary)
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=expected,
        details={"summary": summary, "wall_seconds": time.time() - started},
    )
    print(
        f"Density analysis complete: {len(sites)} sites from {observations} physical-water observations",
        flush=True,
    )
    print(f"RESULT={summary_json}", flush=True)


if __name__ == "__main__":
    main()
