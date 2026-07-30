#!/usr/bin/env python3
"""Independently audit EV71 pipeline artifacts and every physical handoff."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shlex

import mdtraj as md
import numpy as np
import openmm
from openmm import app, unit
import parmed as pmd
from rdkit import Chem

from pipeline_utils import (
    finite_csv,
    ghost_history,
    read_ghost_records,
    read_json,
    require_file,
    sha256,
    validate_recorded_outputs,
    write_json_atomic,
)


WATER_NAMES = {"WAT", "HOH", "TIP3", "SOL"}
ION_NAMES = {"Na+", "Cl-", "NA", "CL", "Na", "Cl"}
AMBER_CHARGE_ROUNDING_TOLERANCE = 1.0e-2
CHEMISTRY_CHARGE_TOLERANCE = 1.0e-2
POSE_RMSD_TOLERANCE_ANGSTROM = 1.0e-2
TIP3P_PARAMETER_TOLERANCE = 2.0e-6
TIP3P_BOND_TOLERANCE = 2.0e-4
TIP3P_ATOMS = {
    "O": {"atomic_number": 8, "mass": 16.0, "charge": -0.834,
          "epsilon": 0.152, "rmin": 1.7683},
    "H1": {"atomic_number": 1, "mass": 1.008, "charge": 0.417,
           "epsilon": 0.0, "rmin": 0.0},
    "H2": {"atomic_number": 1, "mass": 1.008, "charge": 0.417,
           "epsilon": 0.0, "rmin": 0.0},
}
TIP3P_BONDS = {
    frozenset(("O", "H1")): (553.0, 0.9572),
    frozenset(("O", "H2")): (553.0, 0.9572),
    frozenset(("H1", "H2")): (553.0, 1.5136),
}
PROFILE_PROTOCOLS = {
    "full": {
        "uvt1": {
            "batch_size": 50,
            "initial_attempts": 10_000,
            "cycles": 100,
            "attempts_per_cycle": 1_000,
            "md_steps_per_cycle": 5,
            "report_interval": 100,
        },
        "npt": {"steps": 1_000_000, "report_interval": 2_500},
        "uvt2": {
            "batch_size": 50,
            "cycles": 125,
            "attempts_per_cycle": 800,
            "md_steps_per_cycle": 2_000,
            "report_interval": 500,
        },
        "production": {
            "batch_size": 50,
            "cycles": 2_500,
            "md_steps_per_cycle": 2_000,
            "attempts_per_cycle": 200,
            "report_interval": 500,
        },
    },
    "smoke": {
        "uvt1": {
            "batch_size": 50,
            "initial_attempts": 100,
            "cycles": 2,
            "attempts_per_cycle": 100,
            "md_steps_per_cycle": 5,
            "report_interval": 5,
        },
        "npt": {"steps": 500, "report_interval": 100},
        "uvt2": {
            "batch_size": 50,
            "cycles": 2,
            "attempts_per_cycle": 100,
            "md_steps_per_cycle": 100,
            "report_interval": 100,
        },
        "production": {
            "batch_size": 50,
            "cycles": 3,
            "md_steps_per_cycle": 100,
            "attempts_per_cycle": 100,
            "report_interval": 100,
        },
    },
}
EXPECTED_PHYSICAL_PROTOCOL = {
    "temperature_K": 300.0,
    "friction_per_ps": 1.0,
    "cutoff": "12 A",
    "switch_distance_nm": 1.0,
    "ewald_error_tolerance": 5.0e-4,
    "sphere_radius": "10 A",
    "excess_chemical_potential": "-6.09 kcal/mol",
    "standard_volume": "30.345 A^3",
    "num_ghost_waters": 45,
    "timestep_fs": 2.0,
    "pressure": "1 bar",
    "ca_restraint_k_kj_mol_nm2": 100.0,
    "minimization_tolerance_kj_mol_nm": 10.0,
    "com_reset_frequency": 1,
    "barostat_frequency": 25,
    "cutoff_type": "pme",
    "constraint": "h_bonds",
    "integrator": "langevin_middle",
    "bulk_sampling_probability": 0.0,
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ligand-id", required=True)
    parser.add_argument("--profile", required=True, choices=("full", "smoke"))
    parser.add_argument(
        "--through",
        choices=("preparation", "equilibration", "production", "postprocessing"),
        default="postprocessing",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def recorded_path(
    value: object, *, base: Path, allow_absolute: bool = False
) -> Path:
    """Resolve a manifest path without allowing an artifact to escape its trust root."""
    path = Path(str(value))
    if path.is_absolute():
        if not allow_absolute:
            raise ValueError(f"Manifest artifact path must be relative: {value!r}")
        return path.resolve()
    if not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe manifest path: {value!r}")
    root = base.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes {root}: {value!r}") from error
    return resolved


def integer_property(molecule: Chem.Mol, name: str) -> int:
    if not molecule.HasProp(name):
        raise ValueError(f"Ligand SDF is missing required {name!r} metadata")
    value = float(molecule.GetProp(name))
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"Ligand SDF {name!r} is not an integer: {value}")
    return int(value)


def ligand_aliases(molecule: Chem.Mol) -> set[str]:
    aliases = {molecule.GetProp("_Name").strip()}
    if molecule.HasProp("ligand_name"):
        aliases.add(molecule.GetProp("ligand_name").strip())
    return {value for value in aliases if value}


def sdf_records(path: Path) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier(
        str(require_file(path)), removeHs=False, sanitize=True, strictParsing=True
    )
    records = list(supplier)
    invalid = [index for index, molecule in enumerate(records) if molecule is None]
    if invalid:
        raise ValueError(f"Invalid SDF records in {path} at zero-based indices {invalid}")
    return [molecule for molecule in records if molecule is not None]


def molecule_coordinates(molecule: Chem.Mol) -> np.ndarray:
    if molecule.GetNumConformers() != 1:
        raise ValueError("Selected ligand must have exactly one 3D conformer")
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise ValueError("Selected ligand conformer is not marked as three-dimensional")
    coordinates = np.asarray(conformer.GetPositions(), dtype=float)
    if coordinates.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Selected ligand has missing or non-finite coordinates")
    return coordinates


def molecule_edges(molecule: Chem.Mol) -> set[tuple[int, int]]:
    return {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in molecule.GetBonds()
    }


def aligned_rmsd(reference: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return proper-rotation RMSD and maximum displacement in Angstrom."""
    if reference.shape != target.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Cannot align coordinate arrays with different shapes")
    if len(reference) < 3:
        raise ValueError("At least three coordinates are required for a rigid alignment")
    reference_centred = reference - reference.mean(axis=0)
    target_centred = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(reference_centred.T @ target_centred)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1, :] *= -1
        rotation = right.T @ left.T
    delta = reference_centred @ rotation.T - target_centred
    distances = np.linalg.norm(delta, axis=1)
    return float(np.sqrt(np.mean(distances**2))), float(distances.max())


def ligand_sdf_audit(molecule: Chem.Mol) -> dict[str, object]:
    title = molecule.GetProp("_Name").strip()
    if not title:
        raise ValueError("Selected ligand has an empty SDF title")
    coordinates = molecule_coordinates(molecule)
    metadata_charge = integer_property(molecule, "charge")
    multiplicity = integer_property(molecule, "multiplicity")
    formal_charge = int(Chem.GetFormalCharge(molecule))
    radicals = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    if metadata_charge != formal_charge:
        raise ValueError(
            f"Ligand charge metadata ({metadata_charge}) differs from its formal charge "
            f"({formal_charge})"
        )
    if multiplicity < 1 or multiplicity != radicals + 1:
        raise ValueError(
            f"Ligand multiplicity {multiplicity} is incompatible with {radicals} "
            "explicit radical electrons"
        )
    return {
        "title": title,
        "aliases": sorted(ligand_aliases(molecule)),
        "atoms": molecule.GetNumAtoms(),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "explicit_hydrogens": sum(
            atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()
        ),
        "formal_charge": formal_charge,
        "multiplicity": multiplicity,
        "radical_electrons": radicals,
        "bonds": len(molecule_edges(molecule)),
        "coordinate_extent_angstrom": [
            float(value) for value in np.ptp(coordinates, axis=0)
        ],
    }


def compare_source_and_extracted_ligand(
    source: Chem.Mol, extracted: Chem.Mol
) -> dict[str, object]:
    source_audit = ligand_sdf_audit(source)
    extracted_audit = ligand_sdf_audit(extracted)
    identity_keys = {
        "title",
        "aliases",
        "atoms",
        "heavy_atoms",
        "explicit_hydrogens",
        "formal_charge",
        "multiplicity",
        "radical_electrons",
        "bonds",
    }
    if any(source_audit[key] != extracted_audit[key] for key in identity_keys):
        raise ValueError("Extracted ligand chemistry/metadata differs from the source record")
    source_elements = [atom.GetAtomicNum() for atom in source.GetAtoms()]
    extracted_elements = [atom.GetAtomicNum() for atom in extracted.GetAtoms()]
    if source_elements != extracted_elements or molecule_edges(source) != molecule_edges(extracted):
        raise ValueError("Extracted ligand atom order or connectivity differs from the source")
    rmsd, maximum = aligned_rmsd(
        molecule_coordinates(source), molecule_coordinates(extracted)
    )
    if rmsd > 2.0e-3:
        raise ValueError(f"Extracted ligand pose differs from its source: RMSD={rmsd:.6f} A")
    return {
        "source_to_extracted_rmsd_angstrom": rmsd,
        "source_to_extracted_max_displacement_angstrom": maximum,
    }


def amber_ligand_chemistry_audit(
    sdf_molecule: Chem.Mol,
    prmtop: Path,
    coordinates: Path,
) -> dict[str, object]:
    structure = pmd.load_file(str(require_file(prmtop)), str(require_file(coordinates)))
    ligands = [residue for residue in structure.residues if residue.name == "LIG"]
    if len(ligands) != 1:
        raise ValueError(f"{prmtop} contains {len(ligands)} LIG residues")
    ligand = ligands[0]
    atoms = list(ligand.atoms)
    if len(atoms) != sdf_molecule.GetNumAtoms():
        raise ValueError("AMBER ligand atom count differs from the selected SDF")
    sdf_elements = [atom.GetAtomicNum() for atom in sdf_molecule.GetAtoms()]
    amber_elements = [atom.atomic_number for atom in atoms]
    if amber_elements != sdf_elements:
        raise ValueError("AMBER ligand element order differs from the selected SDF")
    local = {atom.idx: index for index, atom in enumerate(atoms)}
    amber_edges: set[tuple[int, int]] = set()
    cross_bonds = 0
    ligand_indices = set(local)
    for bond in structure.bonds:
        first, second = bond.atom1.idx, bond.atom2.idx
        inside = (first in ligand_indices, second in ligand_indices)
        if all(inside):
            amber_edges.add(tuple(sorted((local[first], local[second]))))
        elif any(inside):
            cross_bonds += 1
    if cross_bonds:
        raise ValueError("Prepared ligand is covalently connected outside its LIG residue")
    if amber_edges != molecule_edges(sdf_molecule):
        raise ValueError("AMBER ligand connectivity differs from the selected SDF")
    amber_coordinates = np.asarray(
        [[atom.xx, atom.xy, atom.xz] for atom in atoms], dtype=float
    )
    if not np.isfinite(amber_coordinates).all():
        raise ValueError("AMBER ligand contains non-finite coordinates")
    sdf_coordinates = molecule_coordinates(sdf_molecule)
    heavy = np.asarray(
        [atom.GetAtomicNum() != 1 for atom in sdf_molecule.GetAtoms()], dtype=bool
    )
    rmsd, maximum = aligned_rmsd(sdf_coordinates[heavy], amber_coordinates[heavy])
    if rmsd > POSE_RMSD_TOLERANCE_ANGSTROM:
        raise ValueError(
            f"AMBER ligand pose differs from the selected SDF: RMSD={rmsd:.6f} A"
        )
    partial_charge = float(sum(atom.charge for atom in atoms))
    formal_charge = int(Chem.GetFormalCharge(sdf_molecule))
    if not math.isclose(
        partial_charge, formal_charge, abs_tol=CHEMISTRY_CHARGE_TOLERANCE
    ):
        raise ValueError(
            f"AMBER ligand charge {partial_charge} differs from formal charge {formal_charge}"
        )
    return {
        "atoms": len(atoms),
        "heavy_atoms": int(heavy.sum()),
        "explicit_hydrogens": int((~heavy).sum()),
        "bonds": len(amber_edges),
        "partial_charge": partial_charge,
        "heavy_atom_pose_rmsd_angstrom": rmsd,
        "maximum_heavy_atom_displacement_angstrom": maximum,
    }


def pdb_receptor(path: Path) -> dict[str, object]:
    lines = require_file(path).read_text().splitlines()
    if any(line.startswith(("MODEL ", "ENDMDL")) for line in lines):
        raise ValueError(f"Receptor {path} must contain a single unmodelled coordinate set")
    heteroatoms = [line for line in lines if line.startswith("HETATM")]
    if heteroatoms:
        raise ValueError(f"Receptor {path} contains {len(heteroatoms)} HETATM records")
    atom_lines = [line for line in lines if line.startswith("ATOM  ")]
    if not atom_lines:
        raise ValueError(f"Receptor {path} has no ATOM records")
    if sum(line.startswith("TER") for line in lines) != 1:
        raise ValueError(f"Receptor {path} must have exactly one terminal TER record")

    residues: list[dict[str, object]] = []
    residue_lookup: dict[tuple[str, str, str], int] = {}
    chains: set[str] = set()
    coordinates: list[list[float]] = []
    for line in atom_lines:
        if len(line) < 54:
            raise ValueError(f"Truncated ATOM record in {path}: {line!r}")
        altloc = line[16]
        if altloc != " ":
            raise ValueError(f"Receptor {path} has alternate location {altloc!r}")
        chain = line[21]
        insertion = line[26]
        if insertion != " ":
            raise ValueError(f"Receptor {path} has insertion code {insertion!r}")
        key = (chain, line[22:26], insertion)
        if key not in residue_lookup:
            residue_lookup[key] = len(residues)
            residues.append(
                {
                    "key": key,
                    "name": line[17:20].strip(),
                    "atoms": [],
                }
            )
        residue = residues[residue_lookup[key]]
        if residue["name"] != line[17:20].strip():
            raise ValueError(f"Inconsistent residue name for {key} in {path}")
        atom_name = line[12:16].strip()
        atoms = residue["atoms"]
        assert isinstance(atoms, list)
        if any(atom["name"] == atom_name for atom in atoms):
            raise ValueError(f"Duplicate atom {atom_name!r} in receptor residue {key}")
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError(f"Non-finite receptor coordinate in {path}")
        atoms.append({"name": atom_name, "element": element, "xyz": xyz})
        coordinates.append(xyz)
        chains.add(chain)
    if len(chains) != 1:
        raise ValueError(f"Receptor {path} contains {len(chains)} chains")
    residue_numbers = [int(str(residue["key"][1]).strip()) for residue in residues]
    expected_numbers = list(range(residue_numbers[0], residue_numbers[-1] + 1))
    if residue_numbers != expected_numbers:
        raise ValueError(f"Receptor {path} has residue gaps or non-monotonic numbering")
    ca_counts = [
        sum(atom["name"] == "CA" for atom in residue["atoms"])
        for residue in residues
    ]
    if any(count != 1 for count in ca_counts):
        raise ValueError(f"Receptor {path} does not have exactly one CA per residue")
    return {
        "path": path,
        "residues": residues,
        "coordinates": np.asarray(coordinates, dtype=float),
        "chains": chains,
        "residue_numbers": residue_numbers,
    }


def inferred_histidine_template(residue: dict[str, object]) -> str:
    atoms = {atom["name"] for atom in residue["atoms"]}
    has_hd1, has_he2 = "HD1" in atoms, "HE2" in atoms
    if has_hd1 and has_he2:
        return "HIP"
    if has_hd1:
        return "HID"
    if has_he2:
        return "HIE"
    raise ValueError(f"Cannot infer protonation for histidine {residue['key']}")


def source_receptor_audit(
    source_path: Path,
    normalized_path: Path,
    prepared_prmtop: Path,
    prepared_coordinates: Path,
    normalization_marker: object,
) -> dict[str, object]:
    source = pdb_receptor(source_path)
    normalized = pdb_receptor(normalized_path)
    source_residues = source["residues"]
    normalized_residues = normalized["residues"]
    assert isinstance(source_residues, list) and isinstance(normalized_residues, list)
    if len(source_residues) != len(normalized_residues):
        raise ValueError("Receptor normalization changed the residue count")

    inferred_histidines: dict[str, str] = {}
    expected_n_terminal_renames = 0
    for residue_index, (before, after) in enumerate(
        zip(source_residues, normalized_residues, strict=True)
    ):
        if before["key"] != after["key"]:
            raise ValueError("Receptor normalization changed residue identity/order")
        expected_name = before["name"]
        if before["name"] in {"HIS", "HID", "HIE", "HIP"}:
            expected_name = inferred_histidine_template(before)
            key = before["key"]
            inferred_histidines[
                f"{key[0]}:{str(key[1]).strip()}{str(key[2]).strip()}"
            ] = expected_name
        if after["name"] != expected_name:
            raise ValueError(
                f"Unexpected receptor residue normalization at {before['key']}: "
                f"{before['name']} -> {after['name']}"
            )
        before_atoms = before["atoms"]
        after_atoms = after["atoms"]
        assert isinstance(before_atoms, list) and isinstance(after_atoms, list)
        if len(before_atoms) != len(after_atoms):
            raise ValueError("Receptor normalization changed an atom count")
        for atom_index, (old_atom, new_atom) in enumerate(
            zip(before_atoms, after_atoms, strict=True)
        ):
            expected_atom_name = old_atom["name"]
            if residue_index == 0 and old_atom["name"] == "H":
                expected_atom_name = "H1"
                expected_n_terminal_renames += 1
            if (
                new_atom["name"] != expected_atom_name
                or new_atom["element"] != old_atom["element"]
                or not np.allclose(new_atom["xyz"], old_atom["xyz"], atol=1.0e-8)
            ):
                raise ValueError("Receptor normalization changed atom identity or coordinates")

    if expected_n_terminal_renames != 1:
        # A receptor protonated by PDB2PQR with --ffout=AMBER already carries
        # H1/H2/H3 at the N-terminus, so normalization has no bare "H" to rename
        # and legitimately performs zero renames. Mirrors the same allowance in
        # prepare_ev71_system.py.
        first_residue_atoms = (
            {atom["name"] for atom in normalized_residues[0]["atoms"]}
            if normalized_residues else set()
        )
        if not (expected_n_terminal_renames == 0 and "H1" in first_residue_atoms):
            raise ValueError(
                "Expected exactly one N-terminal H -> H1 normalization (and the "
                "first residue does not already carry an AMBER-style H1)"
            )
    if not isinstance(normalization_marker, dict):
        raise ValueError("Preparation marker has no receptor normalization audit")
    if normalization_marker.get("histidine_templates") != inferred_histidines:
        raise ValueError("Recorded histidine templates differ from the supplied protonation")
    if normalization_marker.get("n_terminal_h_renamed") != expected_n_terminal_renames:
        raise ValueError("Recorded N-terminal normalization differs from the source")
    if normalization_marker.get("residues") != len(source_residues):
        raise ValueError("Recorded receptor residue count differs from the source")

    structure = pmd.load_file(
        str(require_file(prepared_prmtop)), str(require_file(prepared_coordinates))
    )
    protein = [
        residue
        for residue in structure.residues
        if residue.name not in WATER_NAMES | ION_NAMES | {"LIG"}
    ]
    if len(protein) != len(normalized_residues):
        raise ValueError("Prepared receptor residue count differs from the supplied receptor")
    reference_coordinates: list[list[float]] = []
    prepared_coordinates_matched: list[list[float]] = []
    added_atoms: list[str] = []
    for residue_index, (reference, prepared_residue) in enumerate(
        zip(normalized_residues, protein, strict=True)
    ):
        if prepared_residue.name != reference["name"]:
            raise ValueError("Prepared receptor residue sequence differs from normalized input")
        expected_atoms = {atom["name"]: atom for atom in reference["atoms"]}
        actual_atoms = {atom.name: atom for atom in prepared_residue.atoms}
        if len(actual_atoms) != len(prepared_residue.atoms):
            raise ValueError("Prepared receptor has duplicate atom names in a residue")
        missing = set(expected_atoms) - set(actual_atoms)
        extra = set(actual_atoms) - set(expected_atoms)
        allowed_extra = {"OXT"} if residue_index == len(protein) - 1 else set()
        if missing or not extra <= allowed_extra:
            raise ValueError(
                f"Prepared receptor atom identity changed at residue {residue_index}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        added_atoms.extend(
            f"{residue_index}:{name}" for name in sorted(extra)
        )
        for name, reference_atom in expected_atoms.items():
            actual_atom = actual_atoms[name]
            expected_atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(
                str(reference_atom["element"]).title()
            )
            if actual_atom.atomic_number != expected_atomic_number:
                raise ValueError("Prepared receptor element identity differs from source")
            reference_coordinates.append(reference_atom["xyz"])
            prepared_coordinates_matched.append(
                [actual_atom.xx, actual_atom.xy, actual_atom.xz]
            )
    if "OXT" not in {atom.name for atom in protein[-1].atoms}:
        raise ValueError("tLEaP did not materialize the receptor C-terminal OXT")
    rmsd, maximum = aligned_rmsd(
        np.asarray(reference_coordinates, dtype=float),
        np.asarray(prepared_coordinates_matched, dtype=float),
    )
    if rmsd > POSE_RMSD_TOLERANCE_ANGSTROM:
        raise ValueError(f"Prepared receptor coordinates changed: RMSD={rmsd:.6f} A")
    source_atoms = sum(len(residue["atoms"]) for residue in source_residues)
    hydrogens = sum(
        atom["element"] == "H"
        for residue in source_residues
        for atom in residue["atoms"]
    )
    return {
        "source_atoms": source_atoms,
        "source_residues": len(source_residues),
        "source_ca_atoms": len(source_residues),
        "source_explicit_hydrogens": hydrogens,
        "chain": next(iter(source["chains"])),
        "residue_range": [source["residue_numbers"][0], source["residue_numbers"][-1]],
        "histidine_templates": inferred_histidines,
        "normalization": {"n_terminal_h_to_h1": expected_n_terminal_renames},
        "tleap_added_atoms": added_atoms,
        "source_to_prepared_rmsd_angstrom": rmsd,
        "source_to_prepared_max_displacement_angstrom": maximum,
    }


def tleap_log_audit(path: Path) -> dict[str, object]:
    text = require_file(path).read_text(errors="replace")
    matches = re.findall(
        r"Exiting LEaP: Errors = (\d+); Warnings = (\d+); Notes = (\d+)\.", text
    )
    if not matches:
        raise ValueError(f"No final tLEaP status in {path}")
    errors, warnings, notes = map(int, matches[-1])
    if errors:
        raise ValueError(f"tLEaP reported {errors} errors in {path}")
    return {
        "sha256": sha256(path),
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
    }


def logged_command(path: Path) -> list[str]:
    first = require_file(path).read_text(errors="replace").splitlines()[0]
    prefix = "[pipeline] command: "
    if not first.startswith(prefix):
        raise ValueError(f"Preparation log does not record its command: {path}")
    command = shlex.split(first.removeprefix(prefix))
    if not command:
        raise ValueError(f"Empty recorded command in {path}")
    return command


def command_option(command: list[str], option: str) -> str:
    if command.count(option) != 1:
        raise ValueError(f"Recorded command has {command.count(option)} {option} options")
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError(f"Recorded command has no value for {option}")
    return command[index + 1]


def ambertools_provenance_audit(parameterization: object) -> dict[str, object]:
    if not isinstance(parameterization, dict):
        raise ValueError("Preparation marker has no parameterization record")
    provenance = parameterization.get("ambertools")
    if not isinstance(provenance, dict):
        raise ValueError("Preparation marker has no AmberTools provenance")
    root = Path(str(provenance.get("environment_root", ""))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Recorded AmberTools environment is unavailable: {root}")
    commands = provenance.get("resolved_commands")
    binary_hashes = provenance.get("binary_sha256")
    data_hashes = provenance.get("data_sha256")
    package_hashes = provenance.get("conda_package_record_sha256")
    if not all(isinstance(value, dict) and value for value in (
        commands, binary_hashes, data_hashes, package_hashes
    )):
        raise ValueError("AmberTools provenance contains an empty or invalid hash group")
    assert isinstance(commands, dict)
    assert isinstance(binary_hashes, dict)
    assert isinstance(data_hashes, dict)
    assert isinstance(package_hashes, dict)
    if set(commands) != {"antechamber", "parmchk2", "tleap"}:
        raise ValueError("AmberTools provenance has the wrong command set")

    command_paths = {
        name: require_file(Path(str(path)).resolve()) for name, path in commands.items()
    }
    for name, path in command_paths.items():
        if path != (root / "bin" / name).resolve():
            raise ValueError(f"Recorded {name} is outside the AmberTools environment")

    checked: list[tuple[str, str]] = []
    for key, expected in binary_hashes.items():
        if key.startswith("entrypoints/"):
            name = key.split("/", 1)[1]
            path = command_paths.get(name)
            if path is None:
                raise ValueError(f"Unknown AmberTools entry point in provenance: {key}")
        elif key.startswith("backends/"):
            path = root / "bin" / "wrapped_progs" / key.split("/", 1)[1]
        else:
            path = root / key
        actual = sha256(require_file(path))
        if actual != expected:
            raise ValueError(f"AmberTools executable hash changed: {path}")
        checked.append((f"binary:{key}", actual))
    for key, expected in data_hashes.items():
        path = root / key
        actual = sha256(require_file(path))
        if actual != expected:
            raise ValueError(f"AmberTools data hash changed: {path}")
        checked.append((f"data:{key}", actual))
    for name, expected in package_hashes.items():
        path = root / "conda-meta" / name
        actual = sha256(require_file(path))
        if actual != expected:
            raise ValueError(f"AmberTools package record hash changed: {path}")
        checked.append((f"package:{name}", actual))
    digest = hashlib.sha256()
    for name, value in sorted(checked):
        digest.update(f"{name}={value}\n".encode())
    return {
        "environment_root": str(root),
        "commands": {name: str(path) for name, path in command_paths.items()},
        "binary_files": len(binary_hashes),
        "data_files": len(data_hashes),
        "package_records": len(package_hashes),
        "combined_sha256": digest.hexdigest(),
    }


def preparation_log_audit(
    directory: Path,
    ligand_charge: int,
    ligand_multiplicity: int,
    commands: dict[str, str],
) -> dict[str, object]:
    antechamber = require_file(directory / "antechamber.log")
    antechamber_text = antechamber.read_text(errors="replace")
    if not re.search(r"Status:\s*pass\b", antechamber_text):
        raise ValueError("Antechamber log does not contain a passed input-format check")
    charge_matches = re.findall(r"net charge:\s*(-?\d+)", antechamber_text)
    if not charge_matches or int(charge_matches[-1]) != ligand_charge:
        raise ValueError("Antechamber log net charge differs from the selected ligand")
    antechamber_command = logged_command(antechamber)
    if Path(antechamber_command[0]).resolve() != Path(commands["antechamber"]).resolve():
        raise ValueError("Antechamber log was produced by a different executable")
    expected_options = {
        "-c": "bcc",
        "-nc": str(ligand_charge),
        "-m": str(ligand_multiplicity),
        "-rn": "LIG",
        "-at": "gaff2",
    }
    for option, expected in expected_options.items():
        if command_option(antechamber_command, option) != expected:
            raise ValueError(f"Antechamber command has the wrong {option} value")
    sqm = require_file(directory / "sqm.out")
    sqm_text = sqm.read_text(errors="replace")
    if "Calculation Completed" not in sqm_text:
        raise ValueError("SQM output does not contain its completion marker")
    parmchk = require_file(directory / "parmchk2.log")
    parmchk_command = logged_command(parmchk)
    if Path(parmchk_command[0]).resolve() != Path(commands["parmchk2"]).resolve():
        raise ValueError("parmchk2 log was produced by a different executable")
    ligand_tleap_command = logged_command(directory / "ligand_leap.log")
    solvation_tleap_command = logged_command(directory / "solvate.stdout.log")
    for command, script in (
        (ligand_tleap_command, "ligand.in"),
        (solvation_tleap_command, "solvate.in"),
    ):
        if Path(command[0]).resolve() != Path(commands["tleap"]).resolve():
            raise ValueError("tLEaP log was produced by a different executable")
        if command_option(command, "-f") != script:
            raise ValueError("tLEaP log references the wrong command file")
    return {
        "antechamber": {
            "sha256": sha256(antechamber),
            "status": "passed",
            "command": antechamber_command,
        },
        "sqm": {"sha256": sha256(sqm), "status": "completed"},
        "parmchk2": {
            "sha256": sha256(parmchk),
            "bytes": parmchk.stat().st_size,
            "command": parmchk_command,
        },
        "ligand_tleap": tleap_log_audit(directory / "ligand_leap.log"),
        "solvation_tleap_stdout": tleap_log_audit(directory / "solvate.stdout.log"),
        "solvation_tleap_internal": tleap_log_audit(directory / "leap.log"),
    }


def require_signature_values(
    marker: dict[str, object], label: str, expected: dict[str, int]
) -> dict[str, object]:
    signature = marker.get("signature")
    if not isinstance(signature, dict):
        raise ValueError(f"{label} marker has no signature")
    differences = {
        key: {"actual": signature.get(key), "expected": value}
        for key, value in expected.items()
        if signature.get(key) != value
    }
    if differences:
        raise ValueError(f"{label} does not match the selected profile: {differences}")
    return signature


def validate_physical_protocol(signature: dict[str, object], label: str) -> None:
    protocol = signature.get("physical_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{label} signature has no physical protocol")
    differences = {
        key: {"actual": protocol.get(key), "expected": value}
        for key, value in EXPECTED_PHYSICAL_PROTOCOL.items()
        if protocol.get(key) != value
    }
    if differences:
        raise ValueError(f"{label} physical protocol differs from Ludovic: {differences}")


def validate_marker_input(
    signature: dict[str, object], top: Path, coordinates: Path, label: str
) -> None:
    if signature.get("input_prmtop_sha256") != sha256(top):
        raise ValueError(f"{label} marker does not reference its actual input topology")
    if signature.get("input_rst7_sha256") != sha256(coordinates):
        raise ValueError(f"{label} marker does not reference its actual input coordinates")


def q(value: float) -> str:
    """Normalize negligible AMBER/Sire serialization differences."""
    return f"{float(value):.6f}"


def solute_force_field_signature(
    prmtop: Path,
    structure: pmd.Structure,
    solute_indices: set[int],
) -> tuple[str, dict[str, int]]:
    """Hash solute identity and all live OpenMM force-field parameters.

    Sire legitimately drops zero-force torsion records when writing AMBER.
    Excluding only those records compares the actual potential, including LJ,
    bonded terms, and 1-4/exclusion exceptions, at every handoff.
    """
    amber = app.AmberPrmtopFile(str(prmtop))
    system = amber.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )
    local = {global_index: index for index, global_index in enumerate(sorted(solute_indices))}
    records: list[str] = []
    for residue in structure.residues:
        atoms = [atom for atom in residue.atoms if atom.idx in solute_indices]
        if not atoms:
            continue
        records.append(f"R|{residue.name}|{len(atoms)}")
        for atom in atoms:
            records.append(
                f"A|{local[atom.idx]}|{atom.name}|{atom.type}|{atom.atomic_number}"
            )

    force_records: dict[str, list[str]] = {
        "particles": [],
        "bonds": [],
        "angles": [],
        "nonzero_torsions": [],
        "exceptions": [],
    }
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for global_index in sorted(solute_indices):
                charge, sigma, epsilon = force.getParticleParameters(global_index)
                force_records["particles"].append(
                    "P|{}|{}|{}|{}|{}".format(
                        local[global_index],
                        q(system.getParticleMass(global_index).value_in_unit(unit.dalton)),
                        q(charge.value_in_unit(unit.elementary_charge)),
                        q(sigma.value_in_unit(unit.nanometer)),
                        q(epsilon.value_in_unit(unit.kilojoule_per_mole)),
                    )
                )
            for index in range(force.getNumExceptions()):
                i, j, charge_product, sigma, epsilon = force.getExceptionParameters(index)
                i, j = int(i), int(j)
                if i not in solute_indices or j not in solute_indices:
                    continue
                left, right = sorted((local[i], local[j]))
                force_records["exceptions"].append(
                    "X|{}|{}|{}|{}|{}".format(
                        left,
                        right,
                        q(charge_product.value_in_unit(unit.elementary_charge**2)),
                        q(sigma.value_in_unit(unit.nanometer)),
                        q(epsilon.value_in_unit(unit.kilojoule_per_mole)),
                    )
                )
        elif isinstance(force, openmm.HarmonicBondForce):
            for index in range(force.getNumBonds()):
                i, j, length, k = force.getBondParameters(index)
                i, j = int(i), int(j)
                if i not in solute_indices or j not in solute_indices:
                    continue
                left, right = sorted((local[i], local[j]))
                force_records["bonds"].append(
                    "B|{}|{}|{}|{}".format(
                        left,
                        right,
                        q(length.value_in_unit(unit.nanometer)),
                        q(k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)),
                    )
                )
        elif isinstance(force, openmm.HarmonicAngleForce):
            for index in range(force.getNumAngles()):
                i, j, k_atom, angle, k_force = force.getAngleParameters(index)
                atom_tuple = (local[int(i)], local[int(j)], local[int(k_atom)]
                              ) if all(int(value) in solute_indices for value in (i, j, k_atom)) else None
                if atom_tuple is None:
                    continue
                atom_tuple = min(atom_tuple, atom_tuple[::-1])
                force_records["angles"].append(
                    "G|{}|{}|{}|{}|{}".format(
                        *atom_tuple,
                        q(angle.value_in_unit(unit.radian)),
                        q(k_force.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)),
                    )
                )
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for index in range(force.getNumTorsions()):
                i, j, k_atom, l, periodicity, phase, k_force = force.getTorsionParameters(index)
                global_atoms = tuple(int(value) for value in (i, j, k_atom, l))
                if not all(value in solute_indices for value in global_atoms):
                    continue
                force_constant = k_force.value_in_unit(unit.kilojoule_per_mole)
                if abs(force_constant) < 1.0e-10:
                    continue
                atoms = tuple(local[value] for value in global_atoms)
                atoms = min(atoms, atoms[::-1])
                force_records["nonzero_torsions"].append(
                    "T|{}|{}|{}|{}|{}|{}|{}".format(
                        *atoms,
                        int(periodicity),
                        q(phase.value_in_unit(unit.radian)),
                        q(force_constant),
                    )
                )

    digest = hashlib.sha256()
    for record in records:
        digest.update((record + "\n").encode())
    for name in sorted(force_records):
        for record in sorted(force_records[name]):
            digest.update((record + "\n").encode())
    return digest.hexdigest(), {name: len(values) for name, values in force_records.items()}


def close_parameter(actual: float, expected: float, tolerance: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def water_template_audit(structure: pmd.Structure) -> dict[str, object]:
    """Classify every water as exact TIP3P, exact Loch ghost, or malformed."""
    waters = [residue for residue in structure.residues if residue.name in WATER_NAMES]
    water_indices = {residue.idx for residue in waters}
    bonds_by_residue: dict[int, list[pmd.Bond]] = {
        residue.idx: [] for residue in waters
    }
    angles_by_residue: dict[int, list[pmd.Angle]] = {
        residue.idx: [] for residue in waters
    }
    cross_bonds: Counter[int] = Counter()
    for bond in structure.bonds:
        first = bond.atom1.residue.idx
        second = bond.atom2.residue.idx
        if first == second and first in water_indices:
            bonds_by_residue[first].append(bond)
        else:
            if first in water_indices:
                cross_bonds[first] += 1
            if second in water_indices:
                cross_bonds[second] += 1
    for angle in structure.angles:
        residues = {
            angle.atom1.residue.idx,
            angle.atom2.residue.idx,
            angle.atom3.residue.idx,
        }
        if len(residues) == 1:
            residue_index = next(iter(residues))
            if residue_index in water_indices:
                angles_by_residue[residue_index].append(angle)

    physical = 0
    ghosts = 0
    malformed: list[dict[str, object]] = []
    for residue in waters:
        reasons: list[str] = []
        atoms = list(residue.atoms)
        if [atom.name for atom in atoms] != ["O", "H1", "H2"]:
            reasons.append("atom names/order")
        if len(atoms) != 3:
            reasons.append("not three-site")
        names = {atom.name: atom for atom in atoms}
        if len(names) != len(atoms):
            reasons.append("duplicate atom names")

        common_parameters_ok = not reasons
        for name, expected in TIP3P_ATOMS.items():
            atom = names.get(name)
            if atom is None:
                common_parameters_ok = False
                continue
            if atom.atomic_number != expected["atomic_number"] or not close_parameter(
                atom.mass, expected["mass"], TIP3P_PARAMETER_TOLERANCE
            ):
                reasons.append(f"{name} element/mass")
                common_parameters_ok = False

        bonds = bonds_by_residue.get(residue.idx, [])
        actual_bond_keys = {
            frozenset((bond.atom1.name, bond.atom2.name)) for bond in bonds
        }
        if len(bonds) != 3 or actual_bond_keys != set(TIP3P_BONDS):
            reasons.append("rigid bond set")
            common_parameters_ok = False
        else:
            for bond in bonds:
                expected_k, expected_length = TIP3P_BONDS[
                    frozenset((bond.atom1.name, bond.atom2.name))
                ]
                if bond.type is None or not close_parameter(
                    bond.type.k, expected_k, TIP3P_BOND_TOLERANCE
                ) or not close_parameter(
                    bond.type.req, expected_length, TIP3P_BOND_TOLERANCE
                ):
                    reasons.append("rigid bond parameters")
                    common_parameters_ok = False
                    break
        if angles_by_residue.get(residue.idx):
            reasons.append("unexpected intrawater angles")
            common_parameters_ok = False
        if cross_bonds[residue.idx]:
            reasons.append("bond outside water residue")
            common_parameters_ok = False

        physical_nonbonded = common_parameters_ok and all(
            close_parameter(names[name].charge, expected["charge"], TIP3P_PARAMETER_TOLERANCE)
            and close_parameter(names[name].epsilon, expected["epsilon"], TIP3P_PARAMETER_TOLERANCE)
            and close_parameter(names[name].rmin, expected["rmin"], TIP3P_PARAMETER_TOLERANCE)
            for name, expected in TIP3P_ATOMS.items()
        )
        ghost_nonbonded = common_parameters_ok and all(
            close_parameter(names[name].charge, 0.0, TIP3P_PARAMETER_TOLERANCE)
            and close_parameter(names[name].epsilon, 0.0, TIP3P_PARAMETER_TOLERANCE)
            and close_parameter(names[name].rmin, 0.0, TIP3P_PARAMETER_TOLERANCE)
            for name in TIP3P_ATOMS
        )
        if physical_nonbonded:
            physical += 1
        elif ghost_nonbonded:
            ghosts += 1
        else:
            if common_parameters_ok:
                reasons.append("nonbonded parameters are neither TIP3P nor zero ghost")
            malformed.append(
                {"residue_index": residue.idx, "reasons": sorted(set(reasons))}
            )

    return {
        "water_residues": len(waters),
        "physical_tip3p_waters": physical,
        "zero_interaction_ghost_waters": ghosts,
        "nonstandard_waters": len(malformed),
        "nonstandard_examples": malformed[:10],
    }


def topology_audit(prmtop: Path, coordinates: Path) -> tuple[dict[str, object], str]:
    structure = pmd.load_file(str(require_file(prmtop)), str(require_file(coordinates)))
    xyz = np.asarray(structure.coordinates)
    if xyz.shape != (len(structure.atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite or missing coordinates in {coordinates}")
    if (
        structure.box is None
        or len(structure.box) != 6
        or not np.isfinite(structure.box).all()
        or np.any(np.asarray(structure.box[:3]) <= 0.0)
    ):
        raise ValueError(f"Missing, malformed, or non-finite periodic box in {coordinates}")
    water_templates = water_template_audit(structure)
    ligand = [residue for residue in structure.residues if residue.name == "LIG"]
    if len(ligand) != 1:
        raise ValueError(f"{prmtop} contains {len(ligand)} LIG residues")

    solute_atom_indices = {
        atom.idx
        for residue in structure.residues
        if residue.name not in WATER_NAMES | ION_NAMES
        for atom in residue.atoms
    }
    force_field_signature, force_field_counts = solute_force_field_signature(
        prmtop, structure, solute_atom_indices
    )

    audit = {
        "atoms": len(structure.atoms),
        "residues": len(structure.residues),
        "waters": water_templates["water_residues"],
        "zero_interaction_waters": water_templates["zero_interaction_ghost_waters"],
        "water_templates": water_templates,
        "ligand_atoms": len(ligand[0].atoms),
        "protein_ca_atoms": sum(
            atom.name == "CA"
            for residue in structure.residues
            if residue.name not in WATER_NAMES | ION_NAMES | {"LIG"}
            for atom in residue.atoms
        ),
        "total_charge": float(sum(atom.charge for atom in structure.atoms)),
        "box": [float(value) for value in structure.box],
        "solute_force_field_counts": force_field_counts,
    }
    return audit, force_field_signature


def require_physical(label: str, audit: dict[str, object]) -> None:
    templates = audit.get("water_templates")
    if not isinstance(templates, dict):
        raise ValueError(f"{label} has no independent water-template audit")
    if int(templates["nonstandard_waters"]):
        raise ValueError(f"{label} contains malformed/non-TIP3P waters")
    if int(audit["zero_interaction_waters"]):
        raise ValueError(f"{label} contains zero-interaction physical waters")
    if int(templates["physical_tip3p_waters"]) != int(audit["waters"]):
        raise ValueError(f"{label} contains waters outside the exact TIP3P template")
    if not math.isclose(
        float(audit["total_charge"]),
        0.0,
        abs_tol=AMBER_CHARGE_ROUNDING_TOLERANCE,
    ):
        raise ValueError(f"{label} is not neutral: {audit['total_charge']}")


def require_raw_ghost_topology(label: str, audit: dict[str, object]) -> None:
    templates = audit.get("water_templates")
    if not isinstance(templates, dict):
        raise ValueError(f"{label} has no independent water-template audit")
    if int(templates["nonstandard_waters"]):
        raise ValueError(f"{label} contains malformed waters")
    if int(templates["zero_interaction_ghost_waters"]) != 45:
        raise ValueError(f"{label} does not contain exactly 45 zero-interaction ghosts")
    if int(templates["physical_tip3p_waters"]) != int(audit["waters"]) - 45:
        raise ValueError(f"{label} contains non-ghost waters outside exact TIP3P")
    if not math.isclose(
        float(audit["total_charge"]),
        0.0,
        abs_tol=AMBER_CHARGE_ROUNDING_TOLERANCE,
    ):
        raise ValueError(f"{label} is not neutral: {audit['total_charge']}")


def validate_ghost_indices(
    path: Path,
    *,
    valid_water_residues: set[int],
    expected_lines: int,
) -> tuple[list[list[int]], dict[str, object]]:
    records = read_ghost_records(path)
    if len(records) != expected_lines:
        raise ValueError(
            f"{path} has {len(records)} ghost records; expected {expected_lines}"
        )
    for frame, values in enumerate(records):
        invalid = sorted(set(values) - valid_water_residues)
        if invalid:
            raise ValueError(
                f"{path} frame {frame} has out-of-range/non-water ghost IDs: {invalid[:6]}"
            )
    return records, ghost_history(path)


def physical_and_buffer_water_indices(prmtop: Path, coordinates: Path) -> set[int]:
    """Water residue indices valid after Loch appends its 45-water buffer."""
    structure = pmd.load_file(str(require_file(prmtop)), str(require_file(coordinates)))
    valid = {
        residue.idx
        for residue in structure.residues
        if residue.name in WATER_NAMES and len(residue.atoms) == 3
    }
    first_buffer = len(structure.residues)
    valid.update(range(first_buffer, first_buffer + 45))
    return valid


def raw_water_indices(topology_pdb: Path) -> tuple[set[int], int]:
    topology = md.load(str(require_file(topology_pdb))).topology
    valid = {
        residue.index
        for residue in topology.residues
        if residue.name.lower() in {"wat", "hoh"} and len(list(residue.atoms)) == 3
    }
    return valid, topology.n_atoms


def raw_trajectory_audit(
    trajectory_path: Path,
    topology_path: Path,
    *,
    expected_frames: int,
    expected_atoms: int,
) -> dict[str, object]:
    """Stream every raw frame and validate finite coordinates and periodic boxes."""
    frame_count = 0
    atom_count: int | None = None
    minimum_lengths = np.full(3, np.inf, dtype=float)
    maximum_lengths = np.full(3, -np.inf, dtype=float)
    minimum_angles = np.full(3, np.inf, dtype=float)
    maximum_angles = np.full(3, -np.inf, dtype=float)
    for chunk in md.iterload(
        str(require_file(trajectory_path)),
        top=str(require_file(topology_path)),
        chunk=100,
    ):
        if atom_count is None:
            atom_count = chunk.n_atoms
        if chunk.n_atoms != atom_count or chunk.n_atoms != expected_atoms:
            raise ValueError("Raw DCD/topology atom count changed or differs from AMBER")
        if not np.isfinite(chunk.xyz).all():
            raise ValueError("Raw DCD contains non-finite coordinates")
        lengths = chunk.unitcell_lengths
        angles = chunk.unitcell_angles
        if (
            lengths is None
            or angles is None
            or lengths.shape != (chunk.n_frames, 3)
            or angles.shape != (chunk.n_frames, 3)
            or not np.isfinite(lengths).all()
            or not np.isfinite(angles).all()
        ):
            raise ValueError("Raw DCD has missing, malformed, or non-finite unit cells")
        if np.any(lengths <= 0.0) or np.any(angles <= 0.0) or np.any(angles >= 180.0):
            raise ValueError("Raw DCD contains nonphysical unit-cell lengths/angles")
        minimum_lengths = np.minimum(minimum_lengths, lengths.min(axis=0))
        maximum_lengths = np.maximum(maximum_lengths, lengths.max(axis=0))
        minimum_angles = np.minimum(minimum_angles, angles.min(axis=0))
        maximum_angles = np.maximum(maximum_angles, angles.max(axis=0))
        frame_count += chunk.n_frames
    if frame_count != expected_frames:
        raise ValueError(
            f"Raw DCD has {frame_count} frames; expected {expected_frames}"
        )
    if atom_count is None:
        raise ValueError("Raw DCD contains no frames")
    return {
        "frames": frame_count,
        "atoms": atom_count,
        "minimum_box_lengths_nm": [float(value) for value in minimum_lengths],
        "maximum_box_lengths_nm": [float(value) for value in maximum_lengths],
        "minimum_box_angles_degrees": [float(value) for value in minimum_angles],
        "maximum_box_angles_degrees": [float(value) for value in maximum_angles],
    }


def processed_trajectory_audit(
    trajectory_path: Path,
    topology_path: Path,
    ghost_records: list[list[int]],
    ligand_resname: str,
    sphere_radius_angstrom: float,
) -> dict[str, object]:
    """Stream the processed DCD and prove inactive ghosts remain outside."""
    frame_count = 0
    atom_count: int | None = None
    minimum_inactive = math.inf
    record_offset = 0
    for chunk in md.iterload(
        str(require_file(trajectory_path)),
        top=str(require_file(topology_path)),
        chunk=100,
    ):
        if atom_count is None:
            atom_count = chunk.n_atoms
            ligand_residues = [
                residue for residue in chunk.topology.residues if residue.name == ligand_resname
            ]
            if len(ligand_residues) != 1:
                raise ValueError("Processed trajectory does not contain exactly one ligand")
            ligand_indices = np.asarray(
                [atom.index for atom in ligand_residues[0].atoms], dtype=np.int64
            )
            residue_oxygen = {}
            for residue in chunk.topology.residues:
                if residue.name.lower() not in {"wat", "hoh"}:
                    continue
                oxygen = [atom.index for atom in residue.atoms if atom.name.lower() == "o"]
                if len(oxygen) == 1:
                    residue_oxygen[residue.index] = oxygen[0]
        if chunk.n_atoms != atom_count or not np.isfinite(chunk.xyz).all():
            raise ValueError("Processed DCD has changing atom count or non-finite coordinates")
        if chunk.unitcell_lengths is None or not np.isfinite(chunk.unitcell_lengths).all():
            raise ValueError("Processed DCD has missing/non-finite unit-cell lengths")
        for local_frame in range(chunk.n_frames):
            record_index = record_offset + local_frame
            if record_index >= len(ghost_records):
                raise ValueError("Processed DCD has more frames than ghost records")
            ghosts = ghost_records[record_index]
            if not ghosts:
                continue
            missing = [index for index in ghosts if index not in residue_oxygen]
            if missing:
                raise ValueError(f"Processed frame has non-water ghost IDs: {missing[:5]}")
            oxygen_indices = np.asarray([residue_oxygen[index] for index in ghosts])
            centre = chunk.xyz[local_frame, ligand_indices, :].mean(axis=0)
            distances = 10.0 * np.linalg.norm(
                chunk.xyz[local_frame, oxygen_indices, :] - centre, axis=1
            )
            minimum_inactive = min(minimum_inactive, float(distances.min()))
            if np.any(distances <= sphere_radius_angstrom):
                raise ValueError(
                    f"Processed frame {record_index} contains an inactive ghost inside the sphere"
                )
        frame_count += chunk.n_frames
        record_offset += chunk.n_frames
    if frame_count != len(ghost_records):
        raise ValueError("Processed DCD frame count differs from ghost records")
    return {
        "frames": frame_count,
        "atoms": atom_count,
        "minimum_inactive_ghost_distance_angstrom": (
            minimum_inactive if math.isfinite(minimum_inactive) else None
        ),
    }


def pdb_atom_audit(path: Path) -> dict[str, object]:
    atoms = [
        line for line in require_file(path).read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    for line in atoms:
        values = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite coordinate in {path}")
    return {"atoms": len(atoms)}


def main() -> None:
    opt = options()
    run = opt.run_dir.resolve()
    order = ["preparation", "equilibration", "production", "postprocessing"]
    through_index = order.index(opt.through)
    profile = PROFILE_PROTOCOLS[opt.profile]
    report: dict[str, object] = {
        "run_dir": str(run),
        "prefix": opt.prefix,
        "through": opt.through,
        "profile": opt.profile,
        "validation_scope": (
            "full_ludovic_schedule" if opt.profile == "full" else "smoke_plumbing_only"
        ),
        "auditor_source_sha256": sha256(Path(__file__)),
        "audit_utils_source_sha256": sha256(Path(__file__).with_name("pipeline_utils.py")),
    }
    solute_signatures: dict[str, str] = {}

    prep = run / "preparation"
    prep_marker = validate_recorded_outputs(prep / "preparation.complete.json")
    prep_signature = prep_marker.get("signature")
    if not isinstance(prep_signature, dict):
        raise ValueError("Preparation marker has no signature")
    parameterization = prep_signature.get("parameterization")
    if not isinstance(parameterization, dict):
        raise ValueError("Preparation marker has no parameterization protocol")
    expected_parameterization = {
        "charge_method": "am1-bcc",
        "ligand_force_field": "gaff2",
        "protein_force_field": "ff14SB",
        "water_model": "tip3p",
        "solvation": "solvateOct",
    }
    differences = {
        key: {"actual": parameterization.get(key), "expected": value}
        for key, value in expected_parameterization.items()
        if parameterization.get(key) != value
    }
    if differences:
        raise ValueError(
            f"Preparation force fields/parameterization differ from protocol: {differences}"
        )
    ambertools = ambertools_provenance_audit(parameterization)
    if prep_marker.get("ambertools") != parameterization.get("ambertools"):
        raise ValueError("Preparation marker AmberTools details differ from its signature")
    if prep_signature.get("prefix") != opt.prefix:
        raise ValueError("Preparation marker prefix differs from the requested prefix")

    inputs = run / "inputs"
    manifest = read_json(inputs / "manifest.json")
    source_path = require_file(
        recorded_path(manifest.get("source", ""), base=inputs, allow_absolute=True)
    )
    if manifest.get("source_sha256") != sha256(source_path):
        raise ValueError("Ligand manifest source hash differs from the available library")
    manifest_records = manifest.get("ligands")
    if not isinstance(manifest_records, list) or len(manifest_records) != 1:
        raise ValueError("Run input manifest must contain exactly one ligand")
    ligand_record = manifest_records[0]
    if not isinstance(ligand_record, dict):
        raise ValueError("Run input manifest has an invalid ligand record")
    aliases = ligand_record.get("aliases")
    if not isinstance(aliases, list) or opt.ligand_id not in aliases:
        raise ValueError("Run input manifest does not match the requested ligand ID")
    extracted_ligand = require_file(
        recorded_path(ligand_record.get("output", ""), base=inputs)
    )
    extracted_hash = sha256(extracted_ligand)
    if ligand_record.get("output_sha256") != extracted_hash:
        raise ValueError("Extracted ligand differs from the manifest hash")
    if prep_signature.get("ligand_id") != ligand_record.get("ligand_id"):
        raise ValueError("Preparation ligand title differs from the selected manifest record")
    if prep_signature.get("ligand_sha256") != extracted_hash:
        raise ValueError("Preparation ligand hash differs from the selected run input")
    if sha256(prep / "ligand_input.sdf") != extracted_hash:
        raise ValueError("Copied preparation ligand differs from the selected run input")
    receptor_source_file = prep_signature.get("receptor_source_file", "receptor_source.pdb")
    if not isinstance(receptor_source_file, str) or Path(receptor_source_file).name != receptor_source_file:
        raise ValueError("Preparation marker has an unsafe receptor source filename")
    preserved_receptor = require_file(prep / receptor_source_file)
    if prep_signature.get("receptor_sha256") != sha256(preserved_receptor):
        raise ValueError("Preparation receptor hash differs from its preserved source copy")

    source_matches = [
        molecule
        for molecule in sdf_records(source_path)
        if opt.ligand_id in ligand_aliases(molecule)
    ]
    if len(source_matches) != 1:
        raise ValueError(
            f"Ligand selector {opt.ligand_id!r} matches {len(source_matches)} source records"
        )
    extracted_records = sdf_records(extracted_ligand)
    if len(extracted_records) != 1:
        raise ValueError("Extracted ligand file does not contain exactly one record")
    selected_ligand = extracted_records[0]
    extraction_chemistry = compare_source_and_extracted_ligand(
        source_matches[0], selected_ligand
    )
    ligand_sdf = ligand_sdf_audit(selected_ligand)
    manifest_expectations = {
        "ligand_id": ligand_sdf["title"],
        "atoms": ligand_sdf["atoms"],
        "heavy_atoms": ligand_sdf["heavy_atoms"],
        "formal_charge": ligand_sdf["formal_charge"],
        "declared_charge": ligand_sdf["formal_charge"],
        "multiplicity": ligand_sdf["multiplicity"],
        "radical_electrons": ligand_sdf["radical_electrons"],
    }
    manifest_differences = {
        key: {"actual": ligand_record.get(key), "expected": value}
        for key, value in manifest_expectations.items()
        if ligand_record.get(key) != value
    }
    if manifest_differences:
        raise ValueError(f"Ligand manifest chemistry differs: {manifest_differences}")
    if prep_signature.get("ligand_charge") != ligand_sdf["formal_charge"]:
        raise ValueError("Preparation charge differs from the selected ligand")
    if prep_signature.get("ligand_multiplicity") != ligand_sdf["multiplicity"]:
        raise ValueError("Preparation multiplicity differs from the selected ligand")
    ligand_metadata = prep_marker.get("ligand_metadata")
    expected_metadata = {
        "title": ligand_sdf["title"],
        "formal_charge": ligand_sdf["formal_charge"],
        "multiplicity": ligand_sdf["multiplicity"],
    }
    if ligand_metadata != expected_metadata:
        raise ValueError("Preparation marker ligand metadata differs from the selected SDF")

    ligand_only_chemistry = amber_ligand_chemistry_audit(
        selected_ligand,
        prep / "ligand.prmtop",
        prep / "ligand.rst7",
    )
    solvated_ligand_chemistry = amber_ligand_chemistry_audit(
        selected_ligand,
        prep / f"{opt.prefix}_solvated.prmtop",
        prep / f"{opt.prefix}_solvated.inpcrd",
    )
    receptor_chemistry = source_receptor_audit(
        prep / "receptor_source.pdb",
        prep / "receptor_input.pdb",
        prep / f"{opt.prefix}_solvated.prmtop",
        prep / f"{opt.prefix}_solvated.inpcrd",
        prep_marker.get("receptor_normalization"),
    )
    logs = preparation_log_audit(
        prep,
        int(ligand_sdf["formal_charge"]),
        int(ligand_sdf["multiplicity"]),
        ambertools["commands"],
    )
    prepared, signature = topology_audit(
        prep / f"{opt.prefix}_solvated.prmtop",
        prep / f"{opt.prefix}_solvated.inpcrd",
    )
    require_physical("prepared system", prepared)
    expected_residues = int(receptor_chemistry["source_residues"])
    if int(prepared["protein_ca_atoms"]) != expected_residues:
        raise ValueError("Prepared receptor C-alpha count changed during tLEaP")
    report["preparation"] = {
        **prepared,
        "ligand_id": ligand_record.get("ligand_id"),
        "ligand_selector": opt.ligand_id,
        "ligand_library_sha256": manifest.get("source_sha256"),
        "ligand_sha256": extracted_hash,
        "receptor_sha256": prep_signature.get("receptor_sha256"),
        "ligand_sdf": ligand_sdf,
        "source_to_extracted": extraction_chemistry,
        "ligand_only_amber_chemistry": ligand_only_chemistry,
        "solvated_ligand_chemistry": solvated_ligand_chemistry,
        "receptor_chemistry": receptor_chemistry,
        "ambertools_provenance": ambertools,
        "parameterization_logs": logs,
    }
    solute_signatures["preparation"] = signature

    if through_index >= 1:
        equil = run / "equilibration"
        uvt1_marker = validate_recorded_outputs(equil / "uvt1.complete.json")
        npt_marker = validate_recorded_outputs(equil / "npt.complete.json")
        uvt2_marker = validate_recorded_outputs(equil / "uvt2.complete.json")
        uvt1_signature = require_signature_values(uvt1_marker, "UVT1", profile["uvt1"])
        npt_signature = require_signature_values(npt_marker, "NPT", profile["npt"])
        uvt2_signature = require_signature_values(uvt2_marker, "UVT2", profile["uvt2"])
        for label, stage_signature in (
            ("UVT1", uvt1_signature),
            ("NPT", npt_signature),
            ("UVT2", uvt2_signature),
        ):
            validate_physical_protocol(stage_signature, label)
        validate_marker_input(
            uvt1_signature,
            prep / f"{opt.prefix}_solvated.prmtop",
            prep / f"{opt.prefix}_solvated.inpcrd",
            "UVT1",
        )
        validate_marker_input(
            npt_signature,
            equil / f"{opt.prefix}_uvt1.prmtop",
            equil / f"{opt.prefix}_uvt1.rst7",
            "NPT",
        )
        validate_marker_input(
            uvt2_signature,
            equil / f"{opt.prefix}_npt.prmtop",
            equil / f"{opt.prefix}_npt.rst7",
            "UVT2",
        )
        stage_audits: dict[str, object] = {}
        for stage in ("uvt1", "npt", "uvt2"):
            audit, stage_signature = topology_audit(
                equil / f"{opt.prefix}_{stage}.prmtop",
                equil / f"{opt.prefix}_{stage}.rst7",
            )
            require_physical(stage, audit)
            stage_audits[stage] = audit
            solute_signatures[stage] = stage_signature
        if int(stage_audits["npt"]["waters"]) != int(stage_audits["uvt1"]["waters"]):
            raise ValueError("NPT water count differs from UVT1 physical handoff")

        _, uvt1_ghost = validate_ghost_indices(
            equil / f"{opt.prefix}_equilibration_uvt1_ghosts.txt",
            valid_water_residues=physical_and_buffer_water_indices(
                prep / f"{opt.prefix}_solvated.prmtop",
                prep / f"{opt.prefix}_solvated.inpcrd",
            ),
            expected_lines=int(profile["uvt1"]["cycles"]),
        )
        _, uvt2_ghost = validate_ghost_indices(
            equil / f"{opt.prefix}_equilibration_uvt2_ghosts.txt",
            valid_water_residues=physical_and_buffer_water_indices(
                equil / f"{opt.prefix}_npt.prmtop",
                equil / f"{opt.prefix}_npt.rst7",
            ),
            expected_lines=int(profile["uvt2"]["cycles"]),
        )
        if int(stage_audits["uvt1"]["waters"]) != int(prepared["waters"]) + 45 - int(uvt1_ghost["final_state_zero"]):
            raise ValueError("UVT1 physical-water arithmetic failed")
        if int(stage_audits["uvt2"]["waters"]) != int(stage_audits["npt"]["waters"]) + 45 - int(uvt2_ghost["final_state_zero"]):
            raise ValueError("UVT2 physical-water arithmetic failed")

        uvt1_csv = finite_csv(
            equil / f"{opt.prefix}_data_uvt1.csv",
            total_steps=int(profile["uvt1"]["cycles"]) * int(profile["uvt1"]["md_steps_per_cycle"]),
            report_interval=int(profile["uvt1"]["report_interval"]),
        )
        npt_csv = finite_csv(
            equil / f"{opt.prefix}_data_npt.csv",
            total_steps=int(profile["npt"]["steps"]),
            report_interval=int(profile["npt"]["report_interval"]),
        )
        uvt2_csv = finite_csv(
            equil / f"{opt.prefix}_data_uvt2.csv",
            total_steps=int(profile["uvt2"]["cycles"]) * int(profile["uvt2"]["md_steps_per_cycle"]),
            report_interval=int(profile["uvt2"]["report_interval"]),
        )
        report["equilibration"] = {
            "topologies": stage_audits,
            "ghost_histories": {"uvt1": uvt1_ghost, "uvt2": uvt2_ghost},
            "csv": {"uvt1": uvt1_csv, "npt": npt_csv, "uvt2": uvt2_csv},
        }

    if through_index >= 2:
        production = run / "production"
        marker = validate_recorded_outputs(production / "production.complete.json")
        production_signature = require_signature_values(
            marker, "production", profile["production"]
        )
        validate_physical_protocol(production_signature, "production")
        validate_marker_input(
            production_signature,
            run / "equilibration" / f"{opt.prefix}_uvt2.prmtop",
            run / "equilibration" / f"{opt.prefix}_uvt2.rst7",
            "production",
        )
        raw, raw_signature = topology_audit(
            production / f"{opt.prefix}-loch-ghosts.prmtop",
            production / f"{opt.prefix}-loch-ghosts.rst7",
        )
        require_raw_ghost_topology("raw production topology", raw)
        solute_signatures["production_raw"] = raw_signature
        final, final_signature = topology_audit(
            production / f"{opt.prefix}-production-final.prmtop",
            production / f"{opt.prefix}-production-final.rst7",
        )
        require_physical("production final", final)
        solute_signatures["production_final"] = final_signature
        valid_raw_waters, raw_pdb_atoms = raw_water_indices(
            production / f"{opt.prefix}-loch-ghosts.pdb"
        )
        ghost_records, ghosts = validate_ghost_indices(
            production / f"{opt.prefix}-gcmc-ghosts.txt",
            valid_water_residues=valid_raw_waters,
            expected_lines=int(profile["production"]["cycles"]),
        )
        if raw_pdb_atoms != int(raw["atoms"]):
            raise ValueError("Raw production PDB and AMBER topology atom counts differ")
        cycles = int(profile["production"]["cycles"])
        raw_trajectory = raw_trajectory_audit(
            production / f"{opt.prefix}-raw.dcd",
            production / f"{opt.prefix}-loch-ghosts.pdb",
            expected_frames=cycles,
            expected_atoms=int(raw["atoms"]),
        )
        frames = int(raw_trajectory["frames"])
        if ghosts["lines"] != cycles:
            raise ValueError("Production cycles, DCD frames, and ghost lines differ")
        if int(raw["waters"]) != int(report["equilibration"]["topologies"]["uvt2"]["waters"]) + 45:
            raise ValueError("Raw production topology water count is not UVT2 + 45")
        if int(final["waters"]) != int(report["equilibration"]["topologies"]["uvt2"]["waters"]) + 45 - int(ghosts["final_state_zero"]):
            raise ValueError("Production-final physical-water arithmetic failed")
        csv_audit = finite_csv(
            production / f"{opt.prefix}_data_prod.csv",
            total_steps=cycles * int(profile["production"]["md_steps_per_cycle"]),
            report_interval=int(profile["production"]["report_interval"]),
        )
        report["production"] = {
            "raw_topology": raw,
            "final_topology": final,
            "trajectory_frames": frames,
            "raw_trajectory": raw_trajectory,
            "ghost_history": ghosts,
            "csv": csv_audit,
        }

    if through_index >= 3:
        post = run / "postprocessing"
        marker = validate_recorded_outputs(post / "postprocessing.complete.json")
        post_signature = marker.get("signature")
        if not isinstance(post_signature, dict):
            raise ValueError("Postprocessing marker has no signature")
        if post_signature.get("ghost_handling_version") != 2:
            raise ValueError("Postprocessing did not use corrected inactive-ghost handling")
        metrics = read_json(post / f"{opt.prefix}-postprocess.json")
        if marker.get("metrics") != metrics:
            raise ValueError("Postprocessing marker and metrics JSON differ")
        stride = int(metrics["cluster_stride"])
        if stride != int(post_signature.get("cluster_stride", -1)) or stride < 1:
            raise ValueError("Postprocessing stride differs between signature and metrics")
        if opt.profile == "full" and stride > 1:
            report["validation_scope"] = "full_simulation_approximate_postprocessing"
        if int(metrics["trajectory_frames"]) != int(report["production"]["trajectory_frames"]):
            raise ValueError("Postprocessed and raw production frame counts differ")
        raw_pdb = production / f"{opt.prefix}-loch-ghosts.pdb"
        copied_topology = post / f"{opt.prefix}-ghosts.pdb"
        if sha256(raw_pdb) != sha256(copied_topology):
            raise ValueError("Postprocessing topology copy differs from raw topology")
        if post_signature.get("topology_sha256") != sha256(raw_pdb):
            raise ValueError("Postprocessing marker has the wrong topology hash")
        if post_signature.get("trajectory_sha256") != sha256(
            production / f"{opt.prefix}-raw.dcd"
        ):
            raise ValueError("Postprocessing marker has the wrong trajectory hash")
        if post_signature.get("ghost_file_sha256") != sha256(
            production / f"{opt.prefix}-gcmc-ghosts.txt"
        ):
            raise ValueError("Postprocessing marker has the wrong ghost-history hash")
        processed = processed_trajectory_audit(
            post / f"{opt.prefix}-gcmc.dcd",
            copied_topology,
            ghost_records,
            str(post_signature.get("ligand_resname", "LIG")),
            float(post_signature["sphere_radius_angstrom"]),
        )
        if int(processed["frames"]) != int(metrics["trajectory_frames"]):
            raise ValueError("Processed DCD frame count differs from postprocessing metrics")
        if int(processed["atoms"]) != raw_pdb_atoms:
            raise ValueError("Processed DCD and raw topology atom counts differ")
        sphere = post / "gcmc_sphere.pdb"
        sphere_lines = require_file(sphere).read_text().splitlines()
        sphere_models = sum(line.startswith("MODEL") for line in sphere_lines)
        sphere_atoms = pdb_atom_audit(sphere)
        if sphere_models != int(processed["frames"]) + 1 or int(sphere_atoms["atoms"]) != sphere_models:
            raise ValueError("Sphere PDB does not contain initial + one model per frame")
        cluster_audit = pdb_atom_audit(post / f"{opt.prefix}-lig-clusts.pdb")
        if int(cluster_audit["atoms"]) != int(metrics["clusters"]):
            raise ValueError("Cluster PDB atom count differs from postprocessing metrics")
        minimum = processed["minimum_inactive_ghost_distance_angstrom"]
        recorded_minimum = metrics.get("minimum_inactive_ghost_distance_angstrom")
        if minimum is not None and (
            recorded_minimum is None
            or not math.isclose(float(minimum), float(recorded_minimum), abs_tol=0.05)
        ):
            raise ValueError("Independent inactive-ghost distance differs from metrics")
        post_report = dict(metrics)
        post_report["independent_processed_trajectory_audit"] = processed
        post_report["exact_ludovic_clustering_stride"] = stride == 1
        report["postprocessing"] = post_report

    if len(set(solute_signatures.values())) != 1:
        raise ValueError(f"Solute topology changed across physical handoffs: {solute_signatures}")
    report["solute_topology_sha256"] = next(iter(solute_signatures.values()))
    report["solute_topology_stages"] = solute_signatures
    report["status"] = "passed"
    destination = opt.output or run / "pipeline_audit.json"
    write_json_atomic(destination, report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"PIPELINE_AUDIT={destination.resolve()} STATUS=passed", flush=True)


if __name__ == "__main__":
    main()
