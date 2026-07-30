#!/usr/bin/env python3
"""Parameterize one OpenBind ligand and build a solvated EV71 AMBER complex.

The prepared receptor may be PDB or macromolecular CIF/mmCIF. CIF inputs are
materialized to an audited PDB boundary before the unchanged EV71 preparation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

import numpy as np
import parmed as pmd
from rdkit import Chem

from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    implementation_signature,
    invalidate_checkpoint,
    require_file,
    sha256,
)
from receptor_io import (
    materialize_receptor_pdb,
    openbabel_provenance,
    receptor_format,
)


SAFE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
WATER_NAMES = {"WAT", "HOH", "TIP3", "SOL"}
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
# AMBER protonation-state variants emitted by PDB2PQR/PROPKA (`--ffout=AMBER`).
# These are the point of the protonation step -- pH-dependent His tautomers and
# Asp/Glu/Lys/Cys states -- and ff14SB parameterises every one of them natively,
# so accepting them preserves the assignment instead of collapsing it back to a
# tleap default. The audit's purpose (reject residues the force field cannot
# parameterise) is unaffected.
AMBER_PROTONATION_VARIANTS = {
    "HID", "HIE", "HIP",   # histidine: delta, epsilon, doubly protonated
    "CYX", "CYM",          # cystine (disulfide), cysteine thiolate
    "ASH", "GLH", "LYN",   # neutral Asp/Glu, neutral Lys
}
STANDARD_AMINO_ACIDS |= AMBER_PROTONATION_VARIANTS
RECEPTOR_NORMALIZATION_VERSION = 3
AMBER_CHARGE_ROUNDING_TOLERANCE = 1.0e-2
LIGAND_POSE_RMSD_TOLERANCE_ANGSTROM = 1.0e-2


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True, help="Single-record SDF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ligand-charge", type=int)
    parser.add_argument("--ligand-multiplicity", type=int)
    parser.add_argument("--solvent-padding", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Required executable is not active: {name}")
    return path


def run(command: list[str], *, cwd: Path, log: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    with log.open("w") as handle:
        handle.write(f"[pipeline] command: {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    # parmchk2 normally produces no console output. Keep a nonempty, hashable
    # command log so an empty success cannot be confused with an absent log.
    if log.stat().st_size == 0:
        log.write_text("[pipeline] command completed successfully with no console output\n")


def integral_sdf_property(molecule: Chem.Mol, name: str) -> int:
    if not molecule.HasProp(name):
        raise ValueError(f"Ligand SDF is missing required {name!r} metadata")
    raw = molecule.GetProp(name).strip()
    try:
        numeric = float(raw)
    except ValueError as error:
        raise ValueError(f"Ligand SDF {name!r} is not numeric: {raw!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Ligand SDF {name!r} is not an integer: {raw!r}")
    return int(numeric)


def ligand_record(path: Path) -> tuple[Chem.Mol, str, int, int]:
    records = [molecule for molecule in Chem.SDMolSupplier(
        str(path), removeHs=False, sanitize=True, strictParsing=True
    ) if molecule is not None]
    if len(records) != 1:
        raise ValueError(f"Expected exactly one valid ligand in {path}; found {len(records)}")
    molecule = records[0]
    title = molecule.GetProp("_Name").strip()
    if not title:
        raise ValueError("Ligand SDF title is empty")
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise ValueError("Ligand must contain exactly one three-dimensional conformer")
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    if coordinates.shape != (molecule.GetNumAtoms(), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Ligand SDF coordinates are missing or non-finite")
    if any(atom.GetAtomicNum() <= 0 for atom in molecule.GetAtoms()):
        raise ValueError("Ligand contains a dummy or unknown element")
    explicit_hydrogens = sum(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms())
    if explicit_hydrogens == 0:
        raise ValueError("Ligand SDF has no explicit hydrogens")

    charge = integral_sdf_property(molecule, "charge")
    formal_charge = int(Chem.GetFormalCharge(molecule))
    if charge != formal_charge:
        raise ValueError(
            f"Ligand SDF charge metadata ({charge}) does not match its molecular graph "
            f"formal charge ({formal_charge})"
        )
    multiplicity = integral_sdf_property(molecule, "multiplicity")
    if multiplicity < 1:
        raise ValueError(f"Ligand multiplicity must be positive; found {multiplicity}")
    radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    if multiplicity > radical_electrons + 1 or (multiplicity - 1 - radical_electrons) % 2:
        raise ValueError(
            f"Ligand multiplicity {multiplicity} is incompatible with "
            f"{radical_electrons} explicit radical electrons"
        )
    return molecule, title, charge, multiplicity


def ambertools_provenance() -> tuple[dict[str, object], dict[str, str]]:
    """Fingerprint the AmberTools entry points, backends, and data we execute."""
    commands = {name: executable(name) for name in ("antechamber", "parmchk2", "tleap")}
    roots = {str(Path(path).resolve().parents[1]) for path in commands.values()}
    if len(roots) != 1:
        raise ValueError(f"AmberTools commands resolve to different environments: {commands}")
    root = Path(roots.pop())

    binary_paths = {
        f"entrypoints/{name}": Path(path).resolve()
        for name, path in commands.items()
    }
    binary_paths.update({
        f"backends/{path.name}": path
        for path in sorted((root / "bin" / "wrapped_progs").glob("*"))
        if path.is_file()
    })
    for relative in ("bin/sqm", "bin/teLeap", "amber.sh"):
        path = root / relative
        binary_paths[relative] = path

    data_paths: dict[str, Path] = {
        str(path.relative_to(root)): path
        for path in sorted((root / "dat" / "antechamber").glob("*"))
        if path.is_file()
    }
    for relative in (
        "dat/leap/cmd/leaprc.gaff2",
        "dat/leap/cmd/leaprc.protein.ff14SB",
        "dat/leap/cmd/leaprc.water.tip3p",
        "dat/leap/lib/amino12.lib",
        "dat/leap/lib/aminoct12.lib",
        "dat/leap/lib/aminont12.lib",
        "dat/leap/lib/atomic_ions.lib",
        "dat/leap/lib/solvents.lib",
        "dat/leap/parm/parm10.dat",
        "dat/leap/parm/frcmod.ff14SB",
        "dat/leap/parm/gaff2.dat",
        "dat/leap/parm/frcmod.tip3p",
        "dat/leap/parm/frcmod.ions1lm_126_tip3p",
        "dat/leap/parm/frcmod.ionsjc_tip3p",
        "dat/leap/parm/frcmod.ions234lm_126_tip3p",
    ):
        data_paths[relative] = root / relative

    metadata_paths = sorted((root / "conda-meta").glob("ambertools-*.json"))
    if len(metadata_paths) != 1:
        raise FileNotFoundError(
            f"Expected one AmberTools conda package record under {root}; "
            f"found {len(metadata_paths)}"
        )
    for path in [*binary_paths.values(), *data_paths.values(), *metadata_paths]:
        require_file(path)

    return {
        "environment_root": str(root),
        "resolved_commands": {
            name: str(Path(path).resolve()) for name, path in sorted(commands.items())
        },
        "binary_sha256": {
            name: sha256(path) for name, path in sorted(binary_paths.items())
        },
        "data_sha256": {
            name: sha256(path) for name, path in sorted(data_paths.items())
        },
        "conda_package_record_sha256": {
            path.name: sha256(path) for path in metadata_paths
        },
    }, commands


def normalize_receptor(source: Path, output: Path) -> dict[str, object]:
    """Map supplied protonation to ff14SB names without moving atoms."""
    lines = source.read_text().splitlines(keepends=True)
    if any(line.startswith(("MODEL ", "ENDMDL")) for line in lines):
        raise ValueError("EV71 receptor must be a single-model PDB without MODEL records")
    heteroatoms = [line for line in lines if line.startswith("HETATM")]
    if heteroatoms:
        raise ValueError(
            f"EV71 receptor must not contain silently discarded HETATM records; "
            f"found {len(heteroatoms)}"
        )
    atom_line_indices = [index for index, line in enumerate(lines) if line.startswith("ATOM  ")]
    ter_line_indices = [index for index, line in enumerate(lines) if line.startswith("TER")]
    end_line_indices = [index for index, line in enumerate(lines) if line.strip() == "END"]
    if not atom_line_indices:
        raise ValueError(f"No receptor atoms in {source}")
    if len(ter_line_indices) != 1 or ter_line_indices[0] != atom_line_indices[-1] + 1:
        raise ValueError(
            "EV71 receptor must contain exactly one TER immediately after its final ATOM record"
        )
    if len(end_line_indices) != 1 or end_line_indices[0] <= ter_line_indices[0]:
        raise ValueError("EV71 receptor must contain exactly one END record after TER")

    residue_atoms: dict[tuple[str, str, str], set[str]] = {}
    residue_names: dict[tuple[str, str, str], str] = {}
    residue_order: list[tuple[str, str, str]] = []
    for line in lines:
        if not line.startswith("ATOM  "):
            continue
        if len(line) < 54:
            raise ValueError(f"Truncated receptor ATOM record: {line.rstrip()!r}")
        if line[16] != " ":
            raise ValueError(
                f"EV71 receptor contains unsupported alternate location {line[16]!r}"
            )
        if line[26] != " ":
            raise ValueError(
                f"EV71 receptor contains unsupported insertion code {line[26]!r}"
            )
        try:
            coordinates = [float(line[start:stop]) for start, stop in ((30, 38), (38, 46), (46, 54))]
        except ValueError as error:
            raise ValueError(f"Invalid receptor coordinates: {line.rstrip()!r}") from error
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"Non-finite receptor coordinates: {line.rstrip()!r}")
        key = (line[21], line[22:26], line[26])
        name = line[12:16].strip()
        residue_name = line[17:20].strip()
        if not name:
            raise ValueError(f"Blank atom name in receptor record: {line.rstrip()!r}")
        if residue_name not in STANDARD_AMINO_ACIDS:
            raise ValueError(
                f"Unsupported receptor residue {residue_name!r}; only standard amino acids "
                "are accepted in this EV71 preparation path"
            )
        if key not in residue_atoms:
            residue_atoms[key] = set()
            residue_names[key] = residue_name
            residue_order.append(key)
        elif residue_names[key] != residue_name:
            raise ValueError(f"Inconsistent residue names for receptor residue {key}")
        if name in residue_atoms[key]:
            raise ValueError(f"Duplicate atom name {name!r} in receptor residue {key}")
        residue_atoms[key].add(name)

    chains = {key[0] for key in residue_order}
    if chains != {"A"}:
        raise ValueError(f"EV71 receptor must contain only chain A; found {sorted(chains)}")
    residue_numbers = [int(key[1]) for key in residue_order]
    expected_numbers = list(range(residue_numbers[0], residue_numbers[-1] + 1))
    if residue_numbers != expected_numbers:
        raise ValueError(
            f"EV71 receptor residue numbering is not consecutive: "
            f"{residue_numbers[0]}..{residue_numbers[-1]}"
        )
    missing_or_duplicate_ca = [
        key for key in residue_order if sum(name == "CA" for name in residue_atoms[key]) != 1
    ]
    if missing_or_duplicate_ca:
        raise ValueError(
            f"Expected exactly one C-alpha in every receptor residue; bad residues: "
            f"{missing_or_duplicate_ca[:5]}"
        )
    explicit_hydrogens = sum(
        name.startswith("H") for names in residue_atoms.values() for name in names
    )
    if explicit_hydrogens == 0:
        raise ValueError("EV71 receptor must retain its supplied explicit hydrogens")

    ter = lines[ter_line_indices[0]]
    last = residue_order[-1]
    if len(ter) < 26 or ter[21] != last[0] or ter[22:26] != last[1]:
        raise ValueError("Receptor TER does not identify the final residue")

    histidine_names: dict[tuple[str, str, str], str] = {}
    for key in residue_order:
        # Accept histidines that already carry an AMBER template name: a receptor
        # protonated by PDB2PQR/PROPKA arrives as HID/HIE/HIP rather than HIS.
        # The template is still inferred from HD1/HE2 below, which both fixes the
        # name for plain HIS and cross-checks an already-assigned one. The audit
        # infers over this same set, so recording it keeps the two in agreement.
        if residue_names[key] not in {"HIS", "HID", "HIE", "HIP"}:
            continue
        atoms = residue_atoms[key]
        has_hd1 = "HD1" in atoms
        has_he2 = "HE2" in atoms
        if has_hd1 and has_he2:
            histidine_names[key] = "HIP"
        elif has_hd1:
            histidine_names[key] = "HID"
        elif has_he2:
            histidine_names[key] = "HIE"
        else:
            raise ValueError(f"Cannot infer HIS protonation for residue {key}")

    first = residue_order[0]
    normalized: list[str] = []
    renamed_n_terminal_h = 0
    for line in lines:
        if line.startswith("ATOM  "):
            key = (line[21], line[22:26], line[26])
            if key in histidine_names:
                line = line[:17] + f"{histidine_names[key]:>3}" + line[20:]
            if key == first and line[12:16].strip() == "H":
                line = line[:12] + f"{'H1':>4}" + line[16:]
                renamed_n_terminal_h += 1
        normalized.append(line)
    if renamed_n_terminal_h != 1:
        # A receptor protonated by PDB2PQR with --ffout=AMBER already names the
        # N-terminal hydrogens H1/H2/H3, which is precisely the state this rename
        # produces, so there is no bare "H" left to convert. Accept that instead
        # of demanding the un-normalised form.
        already_amber = renamed_n_terminal_h == 0 and "H1" in residue_atoms[first]
        if not already_amber:
            raise ValueError(
                f"Expected one N-terminal backbone H to rename; found "
                f"{renamed_n_terminal_h} (and the first residue does not already "
                f"carry an AMBER-style H1)"
            )
    output.write_text("".join(normalized))
    return {
        "atoms": len(atom_line_indices),
        "residues": len(residue_order),
        "c_alpha_atoms": len(residue_order),
        "chain": "A",
        "first_residue": residue_numbers[0],
        "last_residue": residue_numbers[-1],
        "explicit_hydrogens": explicit_hydrogens,
        "heteroatoms": 0,
        "alternate_locations": 0,
        "insertion_codes": 0,
        "ter_records": 1,
        "histidine_templates": {
            f"{key[0]}:{key[1].strip()}{key[2].strip()}": name
            for key, name in histidine_names.items()
        },
        "n_terminal_h_renamed": renamed_n_terminal_h,
    }


def tleap_succeeded(log: Path) -> None:
    matches = re.findall(r"Exiting LEaP: Errors = (\d+); Warnings = (\d+); Notes = (\d+)\.", log.read_text())
    if not matches:
        raise RuntimeError(f"No final LEaP status found in {log}")
    errors, _, _ = map(int, matches[-1])
    if errors:
        raise RuntimeError(f"LEaP reported {errors} errors; inspect {log}")


def sqm_succeeded(log: Path) -> None:
    if "Calculation Completed" not in require_file(log).read_text():
        raise RuntimeError(f"SQM did not report a completed calculation; inspect {log}")


def proper_kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    if reference.shape != mobile.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Kabsch coordinate arrays have incompatible shapes")
    centered_reference = reference - reference.mean(axis=0)
    centered_mobile = mobile - mobile.mean(axis=0)
    left, _, right_transpose = np.linalg.svd(centered_mobile.T @ centered_reference)
    if np.linalg.det(left @ right_transpose) < 0:
        left[:, -1] *= -1
    rotation = left @ right_transpose
    difference = centered_mobile @ rotation - centered_reference
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def ligand_chemistry_audit(
    source: Chem.Mol,
    structure: pmd.Structure,
    ligand: pmd.Residue,
    *,
    expected_formal_charge: int,
) -> dict[str, object]:
    atoms = list(ligand.atoms)
    source_elements = [atom.GetAtomicNum() for atom in source.GetAtoms()]
    amber_elements = [atom.atomic_number for atom in atoms]
    if amber_elements != source_elements:
        raise ValueError(
            "AMBER ligand element count/order differs from the selected SDF record: "
            f"SDF={source_elements}, AMBER={amber_elements}"
        )

    source_edges = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in source.GetBonds()
    }
    global_to_local = {atom.idx: index for index, atom in enumerate(atoms)}
    amber_edges = {
        tuple(sorted((global_to_local[bond.atom1.idx], global_to_local[bond.atom2.idx])))
        for bond in structure.bonds
        if bond.atom1.idx in global_to_local and bond.atom2.idx in global_to_local
    }
    if amber_edges != source_edges:
        raise ValueError(
            "AMBER ligand connectivity differs from the selected SDF record: "
            f"missing={sorted(source_edges - amber_edges)}, "
            f"extra={sorted(amber_edges - source_edges)}"
        )

    source_hydrogens = sum(number == 1 for number in source_elements)
    amber_hydrogens = sum(number == 1 for number in amber_elements)
    if amber_hydrogens != source_hydrogens:
        raise ValueError(
            f"AMBER ligand has {amber_hydrogens} hydrogens; SDF has {source_hydrogens}"
        )
    partial_charge = float(sum(atom.charge for atom in atoms))
    if not math.isclose(
        partial_charge,
        expected_formal_charge,
        abs_tol=AMBER_CHARGE_ROUNDING_TOLERANCE,
    ):
        raise ValueError(
            f"AMBER ligand partial charges sum to {partial_charge}; expected formal "
            f"charge {expected_formal_charge}"
        )

    source_coordinates = np.asarray(source.GetConformer().GetPositions(), dtype=float)
    amber_coordinates = np.asarray([structure.coordinates[atom.idx] for atom in atoms], dtype=float)
    heavy = np.asarray([number > 1 for number in source_elements], dtype=bool)
    if heavy.sum() < 3:
        raise ValueError("Ligand has fewer than three heavy atoms for a pose-preservation audit")
    pose_rmsd = proper_kabsch_rmsd(source_coordinates[heavy], amber_coordinates[heavy])
    if pose_rmsd > LIGAND_POSE_RMSD_TOLERANCE_ANGSTROM:
        raise ValueError(
            f"AMBER ligand heavy-atom pose changed during preparation: rigid-aligned "
            f"RMSD={pose_rmsd:.6f} A"
        )
    return {
        "atoms": len(atoms),
        "bonds": len(amber_edges),
        "explicit_hydrogens": amber_hydrogens,
        "formal_charge": expected_formal_charge,
        "amber_partial_charge": partial_charge,
        "heavy_atom_rigid_aligned_rmsd_angstrom": pose_rmsd,
        "pose_rmsd_tolerance_angstrom": LIGAND_POSE_RMSD_TOLERANCE_ANGSTROM,
    }


def ligand_amber_audit(
    source: Chem.Mol,
    prmtop: Path,
    coordinates: Path,
    *,
    expected_formal_charge: int,
) -> dict[str, object]:
    structure = pmd.load_file(str(prmtop), str(coordinates))
    xyz = np.asarray(structure.coordinates)
    if xyz.shape != (len(structure.atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError("Ligand-only AMBER coordinates are missing or non-finite")
    if len(structure.residues) != 1 or structure.residues[0].name != "LIG":
        raise ValueError("Ligand-only AMBER topology must contain exactly one LIG residue")
    return ligand_chemistry_audit(
        source,
        structure,
        structure.residues[0],
        expected_formal_charge=expected_formal_charge,
    )


def amber_audit(
    prmtop: Path,
    coordinates: Path,
    *,
    source_ligand: Chem.Mol,
    expected_ligand_charge: int,
) -> dict[str, object]:
    structure = pmd.load_file(str(prmtop), str(coordinates))
    xyz = np.asarray(structure.coordinates)
    if xyz.shape != (len(structure.atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError("Prepared AMBER coordinates are missing or non-finite")
    ligands = [residue for residue in structure.residues if residue.name == "LIG"]
    if len(ligands) != 1:
        raise ValueError(f"Prepared topology contains {len(ligands)} LIG residues")
    waters = [
        residue for residue in structure.residues
        if residue.name in WATER_NAMES and len(residue.atoms) == 3
    ]
    if not waters:
        raise ValueError("Prepared topology contains no three-site waters")
    zero_waters = [
        residue for residue in waters
        if all(abs(atom.charge) < 1.0e-8 and abs(atom.epsilon) < 1.0e-8 for atom in residue.atoms)
    ]
    if zero_waters:
        raise ValueError(f"Prepared topology contains {len(zero_waters)} zero-interaction waters")
    charge = float(sum(atom.charge for atom in structure.atoms))
    if not math.isclose(
        charge, round(charge), abs_tol=AMBER_CHARGE_ROUNDING_TOLERANCE
    ):
        raise ValueError(f"Non-integral prepared-system charge {charge}")
    if not math.isclose(
        charge, 0.0, abs_tol=AMBER_CHARGE_ROUNDING_TOLERANCE
    ):
        raise ValueError(f"Prepared system is not neutral: charge={charge}")
    if structure.box is None or len(structure.box) != 6 or not np.isfinite(structure.box).all():
        raise ValueError("Prepared system has no finite periodic box")
    ligand_chemistry = ligand_chemistry_audit(
        source_ligand,
        structure,
        ligands[0],
        expected_formal_charge=expected_ligand_charge,
    )
    return {
        "atoms": len(structure.atoms),
        "residues": len(structure.residues),
        "waters": len(waters),
        "ligand_atoms": len(ligands[0].atoms),
        "total_charge": charge,
        "zero_interaction_waters": len(zero_waters),
        "box": [float(value) for value in structure.box],
        "ligand_chemistry": ligand_chemistry,
    }


def main() -> None:
    started = time.time()
    opt = options()
    if not SAFE_PREFIX.fullmatch(opt.prefix):
        raise ValueError(f"Unsafe prefix {opt.prefix!r}")
    if opt.solvent_padding <= 0:
        raise ValueError("--solvent-padding must be positive")

    receptor = require_file(opt.receptor)
    receptor_input_format = receptor_format(receptor)
    receptor_converter = openbabel_provenance(receptor)
    ligand = require_file(opt.ligand)
    molecule, ligand_id, metadata_charge, metadata_multiplicity = ligand_record(ligand)
    charge = metadata_charge if opt.ligand_charge is None else opt.ligand_charge
    if opt.ligand_charge is not None and opt.ligand_charge != metadata_charge:
        raise ValueError(
            f"--ligand-charge {opt.ligand_charge} does not match the selected SDF "
            f"record charge {metadata_charge}"
        )
    multiplicity = (
        metadata_multiplicity
        if opt.ligand_multiplicity is None
        else opt.ligand_multiplicity
    )
    if opt.ligand_multiplicity is not None and opt.ligand_multiplicity != metadata_multiplicity:
        raise ValueError(
            f"--ligand-multiplicity {opt.ligand_multiplicity} does not match the selected "
            f"SDF record multiplicity {metadata_multiplicity}"
        )
    ambertools, amber_commands = ambertools_provenance()

    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"{opt.prefix}_solvated"
    receptor_source = output / "receptor_source.pdb"
    receptor_original = (
        output / f"receptor_source{receptor.suffix.lower()}"
        if receptor_input_format != "pdb"
        else receptor_source
    )
    receptor_conversion_log = output / "receptor_conversion.log"
    receptor_copy = output / "receptor_input.pdb"
    ligand_copy = output / "ligand_input.sdf"
    ligand_mol2 = output / "ligand.mol2"
    ligand_frcmod = output / "ligand.frcmod"
    ligand_top = output / "ligand.prmtop"
    ligand_rst = output / "ligand.rst7"
    antechamber_log = output / "antechamber.log"
    parmchk2_log = output / "parmchk2.log"
    sqm_input = output / "sqm.in"
    sqm_log = output / "sqm.out"
    ligand_leap = output / "ligand.in"
    ligand_leap_log = output / "ligand_leap.log"
    complex_leap = output / "solvate.in"
    solvate_stdout_log = output / "solvate.stdout.log"
    leap_log = output / "leap.log"
    final_outputs = [
        prefix.with_suffix(".prmtop"),
        prefix.with_suffix(".inpcrd"),
        prefix.with_suffix(".pdb"),
    ]
    checkpoint_outputs = [
        receptor_original,
        *([] if receptor_original == receptor_source else [receptor_source]),
        receptor_conversion_log,
        receptor_copy,
        ligand_copy,
        ligand_mol2,
        ligand_frcmod,
        ligand_top,
        ligand_rst,
        antechamber_log,
        parmchk2_log,
        sqm_input,
        sqm_log,
        ligand_leap,
        ligand_leap_log,
        complex_leap,
        solvate_stdout_log,
        leap_log,
        *final_outputs,
    ]
    signature = {
        "receptor_sha256": sha256(receptor),
        "receptor_source_file": receptor_original.name,
        "receptor_input_format": receptor_input_format,
        "receptor_converter": receptor_converter,
        "ligand_sha256": sha256(ligand),
        "ligand_id": ligand_id,
        "ligand_charge": charge,
        "ligand_multiplicity": multiplicity,
        "prefix": opt.prefix,
        "solvent_padding_angstrom": opt.solvent_padding,
        "receptor_normalization_version": RECEPTOR_NORMALIZATION_VERSION,
        "parameterization": {
            "charge_method": "am1-bcc",
            "ligand_force_field": "gaff2",
            "protein_force_field": "ff14SB",
            "water_model": "tip3p",
            "solvation": "solvateOct",
            "ambertools": ambertools,
        },
        "implementation": implementation_signature(
            sources={
                "prepare_ev71_system.py": Path(__file__),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
                "receptor_io.py": Path(__file__).with_name("receptor_io.py"),
            },
            distributions=("ParmEd", "rdkit", "biopython"),
        ),
    }
    marker = output / "preparation.complete.json"
    if not opt.force and checkpoint_matches(
        marker, signature=signature, outputs=checkpoint_outputs
    ):
        amber_audit(
            final_outputs[0],
            final_outputs[1],
            source_ligand=molecule,
            expected_ligand_charge=charge,
        )
        sqm_succeeded(sqm_log)
        tleap_succeeded(ligand_leap_log)
        tleap_succeeded(solvate_stdout_log)
        tleap_succeeded(leap_log)
        print(f"Preparation checkpoint is valid: {marker}", flush=True)
        return
    invalidate_checkpoint(marker)

    if receptor_original.resolve() != receptor.resolve():
        shutil.copy2(receptor, receptor_original)
    receptor_conversion = materialize_receptor_pdb(
        receptor_original, receptor_source, receptor_conversion_log
    )
    receptor_normalization = normalize_receptor(receptor_source, receptor_copy)
    shutil.copy2(ligand, ligand_copy)

    parameter_marker = output / "ligand_parameterization.complete.json"
    parameter_outputs = [
        ligand_mol2,
        ligand_frcmod,
        antechamber_log,
        parmchk2_log,
        sqm_input,
        sqm_log,
    ]
    if opt.force or not checkpoint_matches(
        parameter_marker,
        signature=signature,
        outputs=parameter_outputs,
    ):
        invalidate_checkpoint(parameter_marker)
        run(
            [
                amber_commands["antechamber"],
                "-i", ligand_copy.name,
                "-fi", "sdf",
                "-o", ligand_mol2.name,
                "-fo", "mol2",
                "-c", "bcc",
                "-nc", str(charge),
                "-m", str(multiplicity),
                "-rn", "LIG",
                "-at", "gaff2",
            ],
            cwd=output,
            log=antechamber_log,
        )
        run(
            [
                amber_commands["parmchk2"],
                "-i", ligand_mol2.name,
                "-f", "mol2",
                "-o", ligand_frcmod.name,
            ],
            cwd=output,
            log=parmchk2_log,
        )
        sqm_succeeded(sqm_log)
        for path in parameter_outputs:
            require_file(path)
        complete_checkpoint(
            parameter_marker,
            signature=signature,
            outputs=parameter_outputs,
            details={
                "atoms": molecule.GetNumAtoms(),
                "explicit_hydrogens": sum(
                    atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()
                ),
                "formal_charge": charge,
                "multiplicity": multiplicity,
                "ambertools": ambertools,
            },
        )
    else:
        print(f"Ligand parameter checkpoint is valid: {parameter_marker}", flush=True)
        sqm_succeeded(sqm_log)

    ligand_leap.write_text(
        "source leaprc.gaff2\n"
        "LIG = loadmol2 ligand.mol2\n"
        "loadamberparams ligand.frcmod\n"
        "saveamberparm LIG ligand.prmtop ligand.rst7\n"
        "quit\n"
    )
    run(
        [amber_commands["tleap"], "-f", ligand_leap.name],
        cwd=output,
        log=ligand_leap_log,
    )
    tleap_succeeded(ligand_leap_log)
    ligand_audit = ligand_amber_audit(
        molecule,
        ligand_top,
        ligand_rst,
        expected_formal_charge=charge,
    )

    complex_leap.write_text(
        "source leaprc.protein.ff14SB\n"
        "source leaprc.gaff2\n"
        "source leaprc.water.tip3p\n"
        "loadamberparams ligand.frcmod\n"
        "receptor = loadpdb receptor_input.pdb\n"
        "LIG = loadmol2 ligand.mol2\n"
        "complex = combine { receptor LIG }\n"
        f"solvateOct complex TIP3PBOX {opt.solvent_padding:.3f}\n"
        "addionsrand complex Na+ 0\n"
        "addionsrand complex Cl- 0\n"
        f"saveamberparm complex {final_outputs[0].name} {final_outputs[1].name}\n"
        f"savepdb complex {final_outputs[2].name}\n"
        "quit\n"
    )
    run(
        [amber_commands["tleap"], "-f", complex_leap.name],
        cwd=output,
        # tLEaP itself writes leap.log in cwd. Keep redirected console output
        # separate so the two writers cannot corrupt one another.
        log=solvate_stdout_log,
    )
    tleap_succeeded(solvate_stdout_log)
    tleap_succeeded(leap_log)
    for path in [*parameter_outputs, output / "ligand.prmtop", output / "ligand.rst7", *final_outputs]:
        require_file(path)

    audit = amber_audit(
        final_outputs[0],
        final_outputs[1],
        source_ligand=molecule,
        expected_ligand_charge=charge,
    )
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=checkpoint_outputs,
        details={
            "audit": audit,
            "ligand_only_audit": ligand_audit,
            "ligand_metadata": {
                "title": ligand_id,
                "formal_charge": charge,
                "multiplicity": multiplicity,
            },
            "ambertools": ambertools,
            "receptor_conversion": receptor_conversion,
            "receptor_normalization": receptor_normalization,
            "outputs": [path.relative_to(output).as_posix() for path in checkpoint_outputs],
            "wall_seconds": time.time() - started,
        },
    )
    print(json.dumps(audit, indent=2), flush=True)
    print(f"PREPARATION_TOTAL_WALL_SECONDS={time.time() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
