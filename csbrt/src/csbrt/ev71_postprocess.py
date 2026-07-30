#!/usr/bin/env python3
"""Post-process Loch output using GRAND clustering with safe ghost handling.

The imaging, alignment, sphere, and clustering definitions follow Ludovic's
GRAND workflow. Inactive ghosts are shifted *after* imaging and are also
explicitly excluded per frame. GRAND's original shift-before-image order can
image noninteracting ghosts back toward the ligand and contaminate clusters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import mdtraj as md
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

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


SPHERE_RADIUS_ANGSTROM = 10.0
CLUSTER_CUTOFF_ANGSTROM = 2.4
GHOST_SHIFT_BOX_LENGTHS = 5.0


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True, help="PDB topology containing Loch ghost waters")
    parser.add_argument("--trajectory", type=Path, required=True, help="Raw production DCD")
    parser.add_argument("--ghost-file", type=Path, required=True, help="Per-frame inactive Loch residue indices")
    parser.add_argument("--output-dir", type=Path, default=Path("postprocessing"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=SPHERE_RADIUS_ANGSTROM,
        help="Sphere radius in angstrom",
    )
    parser.add_argument(
        "--cluster-cutoff",
        type=float,
        default=CLUSTER_CUTOFF_ANGSTROM,
        help="Average-linkage cutoff in angstrom",
    )
    parser.add_argument(
        "--cluster-stride",
        type=int,
        default=1,
        help="Use every Nth frame for clustering only; 1 exactly follows Ludovic",
    )
    parser.add_argument(
        "--max-distance-memory-gb",
        type=float,
        default=32.0,
        help="Fail before allocating a larger condensed distance matrix",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_inactive_ghosts(path: Path) -> list[list[int]]:
    return read_ghost_records(path)


def ligand_atom_indices(topology: md.Topology, resname: str) -> list[int]:
    residues = [residue for residue in topology.residues if residue.name == resname]
    if len(residues) != 1:
        raise ValueError(
            f"Expected exactly one residue named {resname!r}; found {len(residues)}"
        )
    indices = [atom.index for atom in residues[0].atoms]
    return indices


def shift_inactive_ghosts(trajectory: md.Trajectory, inactive: list[list[int]]) -> None:
    """Translate inactive waters by five unit-cell lengths, as grand.utils does."""
    if trajectory.unitcell_lengths is None:
        raise ValueError("Trajectory has no periodic unit-cell lengths")
    if len(inactive) != trajectory.n_frames:
        raise ValueError(
            f"Ghost history has {len(inactive)} frames but trajectory has {trajectory.n_frames}"
        )

    residue_atoms = {
        residue.index: np.fromiter((atom.index for atom in residue.atoms), dtype=np.int64)
        for residue in trajectory.topology.residues
    }
    water_residues = {
        residue.index
        for residue in trajectory.topology.residues
        if residue.name.lower() in {"wat", "hoh"} and len(list(residue.atoms)) == 3
    }
    for frame, residue_indices in enumerate(inactive):
        invalid = [index for index in residue_indices if index not in residue_atoms]
        if invalid:
            raise IndexError(f"Frame {frame}: ghost residue indices out of range: {invalid[:5]}")
        nonwater = [index for index in residue_indices if index not in water_residues]
        if nonwater:
            raise ValueError(f"Frame {frame}: inactive IDs are not three-site waters: {nonwater[:5]}")
        atom_blocks = [residue_atoms[index] for index in residue_indices]
        if not atom_blocks:
            continue
        atom_indices = np.concatenate(atom_blocks)
        trajectory.xyz[frame, atom_indices, :] += (
            GHOST_SHIFT_BOX_LENGTHS * trajectory.unitcell_lengths[frame]
        )


def image_and_align(trajectory: md.Trajectory) -> list[int]:
    trajectory.image_molecules(inplace=True)
    ca_indices = [
        atom.index
        for atom in trajectory.topology.atoms
        if atom.residue.is_protein and atom.name == "CA"
    ]
    if not ca_indices:
        raise ValueError("No protein C-alpha atoms found for trajectory alignment")
    trajectory.superpose(trajectory, frame=0, atom_indices=ca_indices)
    return ca_indices


def write_sphere_trajectory(
    output: Path,
    trajectory: md.Trajectory,
    topology_path: Path,
    ligand_indices: list[int],
    radius_angstrom: float,
) -> None:
    """Write the same one-point multi-model sphere PDB as grand.utils."""
    initial = md.load(str(topology_path), discard_overlapping_frames=False)
    if initial.n_atoms != trajectory.n_atoms:
        raise ValueError("Topology PDB and trajectory atom counts differ")

    with output.open("w") as handle:
        handle.write("HEADER GCMC SPHERE\n")
        handle.write(f"REMARK RADIUS = {radius_angstrom} ANGSTROMS\n")
        initial_centre = 10.0 * initial.xyz[0, ligand_indices, :].mean(axis=0)
        handle.write("MODEL\n")
        handle.write(
            "HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(
                1, "CTR", "SPH", 1, *initial_centre
            )
        )
        handle.write("ENDMDL\n")
        for frame in range(trajectory.n_frames):
            centre = 10.0 * trajectory.xyz[frame, ligand_indices, :].mean(axis=0)
            handle.write(f"MODEL {frame + 1}\n")
            handle.write(
                "HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(
                    1, "CTR", "SPH", 1, *centre
                )
            )
            handle.write("ENDMDL\n")


def water_oxygen_records(topology: md.Topology) -> tuple[list[int], list[int]]:
    atom_indices: list[int] = []
    residue_indices: list[int] = []
    for residue in topology.residues:
        if residue.name.lower() not in {"wat", "hoh"}:
            continue
        oxygen = [atom.index for atom in residue.atoms if atom.name.lower() == "o"]
        if len(oxygen) != 1:
            raise ValueError(
                f"Water residue {residue.index} ({residue.name}) has {len(oxygen)} O atoms"
            )
        atom_indices.extend(oxygen)
        residue_indices.append(residue.index)
    if not atom_indices:
        raise ValueError("No WAT/HOH oxygen atoms found")
    return atom_indices, residue_indices


def condensed_index(n_items: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Indices of pairs i<j in scipy's condensed distance representation."""
    return n_items * i - i * (i + 1) // 2 + j - i - 1


def cluster_waters(
    trajectory: md.Trajectory,
    ligand_indices: list[int],
    radius_angstrom: float,
    cutoff_angstrom: float,
    stride: int,
    max_memory_gb: float,
    inactive: list[list[int]],
    output: Path,
) -> dict[str, object]:
    """Apply GRAND average-linkage clustering to physical waters only."""
    oxygen_atoms, oxygen_residues = water_oxygen_records(trajectory.topology)
    oxygen_indices = np.asarray(oxygen_atoms, dtype=np.int64)
    oxygen_residue_indices = np.asarray(oxygen_residues, dtype=np.int64)
    coordinates: list[np.ndarray] = []
    source_frames: list[np.ndarray] = []
    sampled_frames = list(range(0, trajectory.n_frames, stride))
    minimum_inactive_distance = math.inf

    for sampled_index, frame in enumerate(sampled_frames):
        centre = trajectory.xyz[frame, ligand_indices, :].mean(axis=0)
        oxygen_xyz = trajectory.xyz[frame, oxygen_indices, :]
        distances = 10.0 * np.linalg.norm(oxygen_xyz - centre, axis=1)
        inactive_mask = np.isin(oxygen_residue_indices, inactive[frame])
        if inactive_mask.any():
            minimum_inactive_distance = min(
                minimum_inactive_distance, float(distances[inactive_mask].min())
            )
            if np.any(distances[inactive_mask] <= radius_angstrom):
                raise RuntimeError(
                    f"Frame {frame} still has an inactive ghost inside the "
                    f"{radius_angstrom:g} A GCMC sphere after shifting"
                )
        inside = (~inactive_mask) & (distances <= radius_angstrom)
        selected = 10.0 * oxygen_xyz[inside]
        if len(selected):
            coordinates.append(selected.astype(np.float64, copy=False))
            source_frames.append(np.full(len(selected), sampled_index, dtype=np.int32))

    if coordinates:
        water_xyz = np.concatenate(coordinates, axis=0)
        water_frames = np.concatenate(source_frames)
    else:
        water_xyz = np.empty((0, 3), dtype=np.float64)
        water_frames = np.empty(0, dtype=np.int32)

    n_observations = len(water_xyz)
    n_distances = n_observations * (n_observations - 1) // 2
    distance_gb = n_distances * np.dtype(np.float64).itemsize / 1.0e9
    print(
        f"Clustering {n_observations} water observations from {len(sampled_frames)} frames; "
        f"condensed distances={n_distances} ({distance_gb:.3f} GB)",
        flush=True,
    )
    if distance_gb > max_memory_gb:
        raise MemoryError(
            f"Exact clustering needs about {distance_gb:.2f} GB for distances, above "
            f"--max-distance-memory-gb={max_memory_gb}. Increase the limit if the node "
            "has enough RAM, or use --cluster-stride > 1 and record that approximation."
        )

    if n_observations == 0:
        cluster_ids = np.empty(0, dtype=np.int32)
    elif n_observations == 1:
        cluster_ids = np.ones(1, dtype=np.int32)
    else:
        distances = pdist(water_xyz)
        # GRAND assigns a huge distance to waters observed in the same frame so
        # a cluster cannot count two simultaneous waters as repeated occupancy.
        for frame in range(len(sampled_frames)):
            members = np.flatnonzero(water_frames == frame)
            if len(members) < 2:
                continue
            left, right = np.triu_indices(len(members), k=1)
            i = members[left]
            j = members[right]
            distances[condensed_index(n_observations, i, j)] = 1.0e8
        tree = hierarchy.linkage(distances, method="average")
        cluster_ids = hierarchy.fcluster(tree, t=cutoff_angstrom, criterion="distance")

    clusters: list[tuple[int, int, np.ndarray]] = []
    for cluster_id in np.unique(cluster_ids):
        members = np.flatnonzero(cluster_ids == cluster_id)
        centre = water_xyz[members].mean(axis=0)
        representative = members[np.argmin(np.linalg.norm(water_xyz[members] - centre, axis=1))]
        clusters.append((int(cluster_id), len(members), water_xyz[representative]))
    clusters.sort(key=lambda item: -item[1])

    with output.open("w") as handle:
        handle.write("REMARK Clustered GCMC water positions written for Loch using GRAND's method\n")
        for index, (_, count, xyz) in enumerate(clusters, start=1):
            occupancy = count / float(len(sampled_frames))
            handle.write(
                "ATOM  {:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(
                    1, "O", "WAT", index, *xyz, occupancy, occupancy
                )
            )
            handle.write("TER\n")
        handle.write("END")

    return {
        "trajectory_frames": trajectory.n_frames,
        "sampled_frames": len(sampled_frames),
        "cluster_stride": stride,
        "water_observations": n_observations,
        "clusters": len(clusters),
        "distance_matrix_gb": distance_gb,
        "minimum_inactive_ghost_distance_angstrom": (
            minimum_inactive_distance if math.isfinite(minimum_inactive_distance) else None
        ),
        "inactive_ghosts_explicitly_excluded": True,
    }


def main() -> None:
    started = time.time()
    opt = options()
    if opt.cluster_stride < 1:
        raise ValueError("--cluster-stride must be a positive integer")
    if opt.sphere_radius <= 0 or opt.cluster_cutoff <= 0:
        raise ValueError("Sphere radius and cluster cutoff must be positive")
    if opt.max_distance_memory_gb <= 0:
        raise ValueError("--max-distance-memory-gb must be positive")

    topology_path = require_file(opt.topology)
    trajectory_path = require_file(opt.trajectory)
    ghost_path = require_file(opt.ghost_file)
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    processed_dcd = output / f"{opt.prefix}-gcmc.dcd"
    sphere_pdb = output / "gcmc_sphere.pdb"
    cluster_pdb = output / f"{opt.prefix}-lig-clusts.pdb"
    topology_copy = output / f"{opt.prefix}-ghosts.pdb"
    metrics_path = output / f"{opt.prefix}-postprocess.json"
    marker = output / "postprocessing.complete.json"
    expected_outputs = [processed_dcd, sphere_pdb, cluster_pdb, topology_copy, metrics_path]
    signature = {
        "topology_sha256": sha256(topology_path),
        "trajectory_sha256": sha256(trajectory_path),
        "ghost_file_sha256": sha256(ghost_path),
        "prefix": opt.prefix,
        "ligand_resname": opt.ligand_resname,
        "sphere_radius_angstrom": opt.sphere_radius,
        "cluster_cutoff_angstrom": opt.cluster_cutoff,
        "cluster_stride": opt.cluster_stride,
        "max_distance_memory_gb": opt.max_distance_memory_gb,
        "ghost_handling_version": 2,
        "implementation": implementation_signature(
            sources={
                "ev71_postprocess.py": Path(__file__),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("mdtraj", "numpy", "scipy"),
            modules=("mdtraj", "scipy.cluster.hierarchy"),
        ),
    }
    if not opt.force and checkpoint_matches(
        marker, signature=signature, outputs=expected_outputs
    ):
        print(f"Postprocessing checkpoint is valid: {marker}", flush=True)
        return
    invalidate_checkpoint(marker)

    print(f"Loading {trajectory_path}", flush=True)
    trajectory = md.load(
        str(trajectory_path),
        top=str(topology_path),
        discard_overlapping_frames=False,
    )
    inactive = read_inactive_ghosts(ghost_path)
    ca_indices = image_and_align(trajectory)
    # Deliberate correctness fix relative to GRAND's historical order: imaging
    # after a shift can wrap inactive ghosts back beside the ligand.
    shift_inactive_ghosts(trajectory, inactive)
    ligand_indices = ligand_atom_indices(trajectory.topology, opt.ligand_resname)

    trajectory.save_dcd(str(processed_dcd))
    write_sphere_trajectory(
        sphere_pdb,
        trajectory,
        topology_path,
        ligand_indices,
        opt.sphere_radius,
    )
    metrics = cluster_waters(
        trajectory,
        ligand_indices,
        opt.sphere_radius,
        opt.cluster_cutoff,
        opt.cluster_stride,
        opt.max_distance_memory_gb,
        inactive,
        cluster_pdb,
    )

    if topology_copy.resolve() != topology_path:
        shutil.copy2(topology_path, topology_copy)
    metrics.update(
        {
            "protein_ca_atoms": len(ca_indices),
            "ligand_atoms": len(ligand_indices),
            "sphere_radius_angstrom": opt.sphere_radius,
            "cluster_cutoff_angstrom": opt.cluster_cutoff,
            "processed_trajectory": processed_dcd.name,
            "cluster_pdb": cluster_pdb.name,
            "sphere_pdb": sphere_pdb.name,
            "ghost_topology": topology_copy.name,
            "wall_seconds": time.time() - started,
        }
    )
    write_json_atomic(metrics_path, metrics)
    for path in (processed_dcd, sphere_pdb, cluster_pdb, topology_copy, metrics_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Post-processing did not create {path}")
        print(f"Created {path}", flush=True)
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=expected_outputs,
        details={
            "metrics": metrics,
            "wall_seconds": time.time() - started,
        },
    )
    print(f"POSTPROCESS_TOTAL_WALL_SECONDS={time.time() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
