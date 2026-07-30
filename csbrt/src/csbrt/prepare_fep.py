#!/usr/bin/env python3
"""Build SOMD2-ready bound and free perturbable systems for one ligand edge.

Both endpoint preparation directories must have been produced by
``prepare_system.py``.  The state-A complex supplies the bound receptor frame;
the state-A ligand is independently solvated for the free leg.  BioSimSpace
maps, aligns, and merges the parameterised ligands into one perturbable
molecule, which is substituted into both legs and saved as Sire stream files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import BioSimSpace as BSS

from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    implementation_signature,
    read_json,
    require_file,
    sha256,
    validate_recorded_outputs,
)


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-a-preparation", type=Path, required=True)
    parser.add_argument("--state-b-preparation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-id", required=True)
    parser.add_argument("--bound-prmtop", type=Path)
    parser.add_argument("--bound-rst7", type=Path)
    parser.add_argument("--mapping-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-mapped-heavy-fraction", type=float, default=0.50)
    parser.add_argument("--solvent-padding", type=float, default=10.0)
    parser.add_argument("--allow-ring-breaking", action="store_true")
    parser.add_argument("--allow-ring-size-change", action="store_true")
    parser.add_argument("--allow-charge-change", action="store_true")
    parser.add_argument(
        "--align-to-bound-pose",
        action="store_true",
        help="RMSD-align the state-A ligand onto the bound frame's ligand pose before "
        "merging, so the perturbable molecule lands in the equilibrated pocket rather "
        "than the preparation pose (use with a water-equilibrated production frame)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path, log: Path) -> None:
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


def preparation(path: Path) -> tuple[Path, dict[str, object]]:
    directory = path.resolve()
    marker = directory / "preparation.complete.json"
    validate_recorded_outputs(marker)
    return directory, read_json(marker)


def signature(marker: dict[str, object]) -> dict[str, object]:
    value = marker.get("signature")
    if not isinstance(value, dict):
        raise ValueError("Preparation marker has no signature")
    return value


def one_solvated_pair(directory: Path) -> tuple[Path, Path]:
    topologies = sorted(directory.glob("*_solvated.prmtop"))
    coordinates = sorted(directory.glob("*_solvated.inpcrd"))
    if len(topologies) != 1 or len(coordinates) != 1:
        raise ValueError(
            f"Expected one *_solvated.prmtop/inpcrd pair in {directory}; found "
            f"{len(topologies)} and {len(coordinates)}"
        )
    if topologies[0].stem.removesuffix("_solvated") != coordinates[0].stem.removesuffix("_solvated"):
        raise ValueError("Solvated topology and coordinate prefixes differ")
    return require_file(topologies[0]), require_file(coordinates[0])


def ligand_from_preparation(directory: Path):
    system = BSS.IO.readMolecules(
        [str(require_file(directory / "ligand.prmtop")), str(require_file(directory / "ligand.rst7"))]
    )
    if system.nMolecules() != 1:
        raise ValueError(f"Ligand topology in {directory} contains {system.nMolecules()} molecules")
    return system.getMolecule(0)


def ligand_index(system) -> int:
    matches = [
        index
        for index, molecule in enumerate(system.getMolecules())
        if molecule.nResidues() == 1 and molecule.getResidues()[0].name() == "LIG"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one LIG molecule; found {len(matches)}")
    return matches[0]


def main() -> None:
    opt = options()
    if not SAFE_ID.fullmatch(opt.edge_id):
        raise ValueError("--edge-id contains unsafe filename characters")
    if opt.mapping_timeout_seconds <= 0 or opt.solvent_padding <= 0:
        raise ValueError("Mapping timeout and solvent padding must be positive")
    if not 0 < opt.minimum_mapped_heavy_fraction <= 1:
        raise ValueError("--minimum-mapped-heavy-fraction must be in (0, 1]")
    if (opt.bound_prmtop is None) != (opt.bound_rst7 is None):
        raise ValueError("Provide both --bound-prmtop and --bound-rst7, or neither")

    output = opt.output_dir.resolve()
    marker_path = output / "fep_preparation.complete.json"
    output.mkdir(parents=True, exist_ok=True)

    state_a, marker_a = preparation(opt.state_a_preparation)
    state_b, marker_b = preparation(opt.state_b_preparation)
    sig_a, sig_b = signature(marker_a), signature(marker_b)
    default_bound_top, default_bound_rst = one_solvated_pair(state_a)
    bound_topology = require_file(opt.bound_prmtop or default_bound_top)
    bound_coordinates = require_file(opt.bound_rst7 or default_bound_rst)
    implementation = implementation_signature(
        sources={
            "prepare_fep.py": Path(__file__),
            "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
        },
        distributions=("BioSimSpace", "sire"),
        modules=("BioSimSpace", "sire"),
    )
    checkpoint_signature = {
        "edge_id": opt.edge_id,
        "state_a_preparation_signature": sig_a,
        "state_b_preparation_signature": sig_b,
        "mapping_timeout_seconds": opt.mapping_timeout_seconds,
        "minimum_mapped_heavy_fraction": opt.minimum_mapped_heavy_fraction,
        "solvent_padding_angstrom": opt.solvent_padding,
        "allow_ring_breaking": opt.allow_ring_breaking,
        "allow_ring_size_change": opt.allow_ring_size_change,
        "allow_charge_change": opt.allow_charge_change,
        "align_to_bound_pose": opt.align_to_bound_pose,
        "bound_input_topology_sha256": sha256(bound_topology),
        "bound_input_coordinates_sha256": sha256(bound_coordinates),
        "implementation": implementation,
    }
    expected_outputs = [
        output / f"{opt.edge_id}_bound.bss",
        output / f"{opt.edge_id}_free.bss",
        output / "state_a_ligand.mol2",
        output / "state_a_ligand.frcmod",
        output / "state_a_free.prmtop",
        output / "state_a_free.rst7",
        output / "state_a_free.pdb",
        output / "state_a_free.in",
        output / "state_a_free_tleap.log",
    ]
    if not opt.force and checkpoint_matches(
        marker_path, signature=checkpoint_signature, outputs=expected_outputs
    ):
        print(f"FEP preparation checkpoint is valid: {marker_path}", flush=True)
        return
    marker_path.unlink(missing_ok=True)

    charge_a = int(sig_a["ligand_charge"])
    charge_b = int(sig_b["ligand_charge"])
    if charge_a != charge_b and not opt.allow_charge_change:
        raise ValueError(
            f"Charge-changing edge {charge_a} -> {charge_b} requires "
            "--allow-charge-change and a documented finite-size correction"
        )

    ligand_a = ligand_from_preparation(state_a)
    ligand_b = ligand_from_preparation(state_b)
    if opt.align_to_bound_pose:
        # Reposition the state-A ligand onto the ligand pose already present in the
        # (water-equilibrated) bound frame, so the merged perturbable molecule is
        # built in the equilibrated pocket instead of the preparation pose.
        frame_system = BSS.IO.readMolecules([str(bound_topology), str(bound_coordinates)])
        frame_ligand = frame_system.getMolecule(ligand_index(frame_system))
        if frame_ligand.nAtoms() != ligand_a.nAtoms():
            raise ValueError(
                "Bound-frame ligand atom count differs from the state-A ligand; "
                "cannot align to the bound pose"
            )
        identity = {index: index for index in range(ligand_a.nAtoms())}
        ligand_a = BSS.Align.rmsdAlign(ligand_a, frame_ligand, identity)
    if sig_a.get("ligand_sha256") == sig_b.get("ligand_sha256"):
        if ligand_a.nAtoms() != ligand_b.nAtoms():
            raise ValueError("Identical ligand inputs produced different atom counts")
        mapping = dict(enumerate(range(ligand_a.nAtoms())))
    else:
        mapping = BSS.Align.matchAtoms(
            ligand_a,
            ligand_b,
            timeout=opt.mapping_timeout_seconds * BSS.Units.Time.second,
            complete_rings_only=not opt.allow_ring_breaking,
            max_scoring_matches=100,
        )
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("BioSimSpace did not produce a ligand atom mapping")
    atoms_a = ligand_a.getAtoms()
    atoms_b = ligand_b.getAtoms()
    mapped_heavy = sum(
        not atoms_a[index_a].element().lower().startswith("hydrogen")
        and not atoms_b[index_b].element().lower().startswith("hydrogen")
        for index_a, index_b in mapping.items()
    )
    heavy_a = sum(not atom.element().lower().startswith("hydrogen") for atom in atoms_a)
    heavy_b = sum(not atom.element().lower().startswith("hydrogen") for atom in atoms_b)
    mapped_fraction = mapped_heavy / min(heavy_a, heavy_b)
    if mapped_fraction < opt.minimum_mapped_heavy_fraction:
        raise ValueError(
            f"Only {mapped_fraction:.3f} of the smaller ligand's heavy atoms map; "
            f"minimum is {opt.minimum_mapped_heavy_fraction:.3f}"
        )
    reverse_mapping = {state_b_index: state_a_index for state_a_index, state_b_index in mapping.items()}
    aligned_b = BSS.Align.rmsdAlign(ligand_b, ligand_a, reverse_mapping)
    merged = BSS.Align.merge(
        ligand_a,
        aligned_b,
        mapping,
        allow_ring_breaking=opt.allow_ring_breaking,
        allow_ring_size_change=opt.allow_ring_size_change,
    )
    if not merged.isPerturbable():
        raise RuntimeError("Merged ligand is not marked perturbable")

    tleap = shutil.which("tleap")
    if tleap is None:
        sibling = Path(sys.executable).resolve().parent / "tleap"
        tleap = str(sibling) if sibling.is_file() else None
    if tleap is None:
        raise FileNotFoundError("tleap is not active; run inside the preparation Mamba environment")
    local_mol2 = output / "state_a_ligand.mol2"
    local_frcmod = output / "state_a_ligand.frcmod"
    shutil.copy2(require_file(state_a / "ligand.mol2"), local_mol2)
    shutil.copy2(require_file(state_a / "ligand.frcmod"), local_frcmod)
    free_topology = output / "state_a_free.prmtop"
    free_coordinates = output / "state_a_free.rst7"
    free_pdb = output / "state_a_free.pdb"
    free_input = output / "state_a_free.in"
    free_log = output / "state_a_free_tleap.log"
    free_input.write_text(
        "source leaprc.gaff2\n"
        "source leaprc.water.tip3p\n"
        f"loadamberparams {local_frcmod.name}\n"
        f"LIG = loadmol2 {local_mol2.name}\n"
        f"solvateOct LIG TIP3PBOX {opt.solvent_padding:.3f}\n"
        "addionsrand LIG Na+ 0\n"
        "addionsrand LIG Cl- 0\n"
        f"saveamberparm LIG {free_topology.name} {free_coordinates.name}\n"
        f"savepdb LIG {free_pdb.name}\n"
        "quit\n"
    )
    run([tleap, "-f", free_input.name], cwd=output, log=free_log)
    for path in (free_topology, free_coordinates, free_pdb):
        require_file(path)
    if "Exiting LEaP: Errors = 0" not in free_log.read_text(errors="replace"):
        raise RuntimeError("Free-leg tLEaP log does not report zero errors")

    bound = BSS.IO.readMolecules([str(bound_topology), str(bound_coordinates)])
    free = BSS.IO.readMolecules([str(free_topology), str(free_coordinates)])
    bound.updateMolecule(ligand_index(bound), merged)
    free.updateMolecule(ligand_index(free), merged)
    if bound.nPerturbableMolecules() != 1 or free.nPerturbableMolecules() != 1:
        raise RuntimeError("Each FEP leg must contain exactly one perturbable molecule")

    bound_base = output / f"{opt.edge_id}_bound"
    free_base = output / f"{opt.edge_id}_free"
    BSS.Stream.save(bound, str(bound_base))
    BSS.Stream.save(free, str(free_base))
    bound_stream = require_file(bound_base.with_suffix(".bss"))
    free_stream = require_file(free_base.with_suffix(".bss"))
    for stream in (bound_stream, free_stream):
        loaded = BSS.Stream.load(str(stream))
        if loaded.nPerturbableMolecules() != 1:
            raise RuntimeError(f"Reloaded stream {stream} lost its perturbable molecule")

    mapping_rows = [
        {"state_a_atom": int(index_a), "state_b_atom": int(index_b)}
        for index_a, index_b in sorted(mapping.items())
    ]
    details = {
        "edge_id": opt.edge_id,
        "state_a_ligand_id": sig_a.get("ligand_id"),
        "state_b_ligand_id": sig_b.get("ligand_id"),
        "state_a_charge": charge_a,
        "state_b_charge": charge_b,
        "charge_change": charge_b - charge_a,
        "charge_change_correction_required": charge_a != charge_b,
        "mapped_atoms": len(mapping),
        "mapped_heavy_atoms": mapped_heavy,
        "mapped_heavy_fraction": mapped_fraction,
        "mapping": mapping_rows,
        "allow_ring_breaking": opt.allow_ring_breaking,
        "allow_ring_size_change": opt.allow_ring_size_change,
        "align_to_bound_pose": opt.align_to_bound_pose,
        "bound_input_topology": str(bound_topology),
        "bound_input_coordinates": str(bound_coordinates),
        "bound_stream": bound_stream.name,
        "free_stream": free_stream.name,
        "implementation": implementation,
    }
    outputs = [
        bound_stream,
        free_stream,
        local_mol2,
        local_frcmod,
        free_topology,
        free_coordinates,
        free_pdb,
        free_input,
        free_log,
    ]
    complete_checkpoint(
        marker_path,
        signature=checkpoint_signature,
        outputs=outputs,
        details=details,
    )
    payload = validate_recorded_outputs(marker_path)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"FEP_PREPARATION={marker_path}", flush=True)


if __name__ == "__main__":
    main()
