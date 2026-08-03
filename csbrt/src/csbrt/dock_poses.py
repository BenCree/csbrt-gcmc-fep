#!/usr/bin/env python3
"""Dock a ligand with GNINA from a deliberately scrambled starting pose.

A docking program handed a crystal ligand in its crystal position can return that
position without having searched for it, which tells you nothing about whether the pose
is findable. So the ligand is first scrambled -- a uniformly random rotation plus a
random translation within a configurable radius -- and the result is *verified*: the
scrambled input must be far enough from the crystal pose that recovery is a real search
result, and that check fails loudly rather than being assumed.

The docking box still comes from the crystal ligand (via GNINA's ``--autobox_ligand``),
because the question is "can the site be re-found", not "can the site be located".

GNINA is not installable from conda-forge; this calls the standalone static binary as a
subprocess, matching how every other external tool in the pipeline is invoked. Pass
``--gnina-binary`` or leave it on PATH.

Pose ids are ``{ligand_id}_p{NN}`` -- dot-free and matching the SAFE_NAME pattern that
``extract_ligands.py`` enforces, because downstream each pose becomes its own endpoint
with its own run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - registers conformer helpers

DEFAULT_EXHAUSTIVENESS = 16
DEFAULT_NUM_MODES = 20
DEFAULT_AUTOBOX_ADD = 4.0
DEFAULT_SCRAMBLE_RADIUS = 5.0
# Below this the "scramble" has not meaningfully moved the ligand and a recovered pose
# would prove nothing.
MINIMUM_SCRAMBLE_RMSD = 2.0


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation on SO(3) via a normalised random quaternion."""
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def conformer_coordinates(molecule: Chem.Mol) -> np.ndarray:
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=float)


def set_conformer_coordinates(molecule: Chem.Mol, coordinates: np.ndarray) -> None:
    conformer = molecule.GetConformer()
    for index, position in enumerate(coordinates):
        conformer.SetAtomPosition(index, position.tolist())


def heavy_atom_mask(molecule: Chem.Mol) -> np.ndarray:
    return np.asarray(
        [atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()], dtype=bool
    )


def heavy_coordinates(molecule: Chem.Mol) -> np.ndarray:
    """Heavy-atom coordinates of this molecule.

    Selected per molecule rather than by reusing the input's mask: GNINA writes poses
    without hydrogens, so a mask built from an explicit-H input has the wrong length for
    its own output.
    """
    return conformer_coordinates(molecule)[heavy_atom_mask(molecule)]


def proper_kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    """Superposed RMSD with reflection correction (mirrors prepare_ev71_system)."""
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


def in_place_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    """RMSD without superposition -- the pose-relevant measure inside a fixed site."""
    if reference.shape != mobile.shape:
        raise ValueError("Coordinate arrays have incompatible shapes")
    return float(np.sqrt(np.mean(np.sum((reference - mobile) ** 2, axis=1))))


def scramble_ligand(
    molecule: Chem.Mol, rng: np.random.Generator, radius_angstrom: float
) -> Chem.Mol:
    """Random rigid-body rotation about the centroid plus a random translation.

    Rigid-body only: internal geometry is untouched, so the docked result cannot be an
    artefact of a strained input conformer.
    """
    scrambled = Chem.Mol(molecule)
    coordinates = conformer_coordinates(scrambled)
    centroid = coordinates.mean(axis=0)
    rotated = (coordinates - centroid) @ random_rotation(rng).T
    # Uniform within the sphere, not on it: cube-root keeps the radial density even.
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    offset = direction * radius_angstrom * rng.random() ** (1.0 / 3.0)
    set_conformer_coordinates(scrambled, rotated + centroid + offset)
    return scrambled


def gnina_available(binary: str) -> str | None:
    return shutil.which(binary)


def run_gnina(
    *,
    binary: str,
    receptor: Path,
    ligand: Path,
    autobox_ligand: Path,
    output: Path,
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    autobox_add: float,
    cnn_scoring: str,
    extra_args: list[str] | None = None,
    ld_library_path: str | None = None,
) -> None:
    resolved = gnina_available(binary)
    if resolved is None:
        raise FileNotFoundError(
            f"GNINA binary {binary!r} not found on PATH. GNINA is not installable from "
            "conda-forge; download the static binary from the gnina releases page and "
            "pass --gnina-binary /path/to/gnina."
        )
    command = [
        resolved,
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--autobox_ligand", str(autobox_ligand),
        "--autobox_add", f"{autobox_add:g}",
        "--out", str(output),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
        "--cnn_scoring", cnn_scoring,
    ]
    command.extend(extra_args or [])
    environment = None
    if ld_library_path:
        # The release binaries link cuDNN 9 dynamically despite being called "static".
        # Carrying it explicitly beats relying on the caller's ambient environment,
        # which is how this silently breaks under Slurm.
        environment = dict(os.environ)
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            f"{ld_library_path}:{existing}" if existing else ld_library_path
        )
    print("+ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env=environment)
    except subprocess.CalledProcessError as error:
        # The "static" release is not fully static and is built against a recent
        # toolchain. Both of these bite on RHEL-9-era hosts and the raw loader message
        # is cryptic, so name the fix rather than pass the error through untranslated.
        raise RuntimeError(
            f"GNINA exited {error.returncode}. Two common causes:\n"
            "  * 'libcudnn.so.9: cannot open shared object file' -- the binary needs "
            "cuDNN 9 at runtime. Put a conda env that has it on LD_LIBRARY_PATH, e.g.\n"
            "      export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n"
            "  * \"version `GLIBC_2.35' not found\" -- the release is newer than the "
            "host glibc (RHEL 9 ships 2.34). Use an older GNINA release built against "
            "an older glibc, or run in a container."
        ) from error
    if not output.is_file():
        raise RuntimeError(f"GNINA reported success but wrote no output at {output}")


def pose_score(molecule: Chem.Mol) -> tuple[float, float]:
    """(primary, secondary) sort key. CNN score first, then affinity.

    GNINA tags poses with CNNscore/CNNaffinity when CNN scoring is on and
    minimizedAffinity always. Higher CNNscore is better; more negative affinity is
    better. Missing tags sort last so an un-scored pose never outranks a scored one.
    """
    def tag(name: str, default: float) -> float:
        if not molecule.HasProp(name):
            return default
        try:
            return float(molecule.GetProp(name))
        except ValueError:
            return default

    cnn = tag("CNNscore", float("-inf"))
    affinity = tag("minimizedAffinity", tag("CNNaffinity", float("inf")))
    return (-cnn, affinity)


def read_poses(path: Path) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    poses = [molecule for molecule in supplier if molecule is not None]
    if not poses:
        raise RuntimeError(f"No readable poses in {path}")
    return poses


def write_pose_sdfs(
    poses: list[Chem.Mol], out_dir: Path, ligand_id: str, count: int
) -> list[dict[str, object]]:
    """Write the top `count` poses as one single-record SDF each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(poses, key=pose_score)[:count]
    written = []
    for rank, molecule in enumerate(ranked, start=1):
        pose_id = f"{ligand_id}_p{rank:02d}"
        tagged = Chem.Mol(molecule)
        tagged.SetProp("_Name", pose_id)
        path = out_dir / f"{pose_id}.sdf"
        writer = Chem.SDWriter(str(path))
        writer.write(tagged)
        writer.close()
        written.append(
            {
                "pose_id": pose_id,
                "rank": rank,
                "path": str(path),
                "cnn_score": molecule.GetProp("CNNscore")
                if molecule.HasProp("CNNscore") else None,
                "minimized_affinity": molecule.GetProp("minimizedAffinity")
                if molecule.HasProp("minimizedAffinity") else None,
            }
        )
    return written


def single_record(path: Path) -> Chem.Mol:
    if path.suffix.lower() == ".pdb":
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        if molecule is None or molecule.GetNumConformers() != 1:
            raise ValueError(f"Could not read a single conformer from {path}")
        return molecule
    molecules = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
    if len(molecules) != 1:
        raise ValueError(f"Expected exactly one record in {path}; found {len(molecules)}")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise ValueError(f"{path} must hold exactly one 3D conformer")
    return molecule


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", type=Path, required=True,
                        help="Prepared receptor PDB. For round 2 this includes waters.")
    parser.add_argument("--ligand", type=Path, required=True,
                        help="Single-record SDF; the pose to scramble and re-dock")
    parser.add_argument("--ligand-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many ranked poses to keep (default 5)")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--scramble-radius", type=float, default=DEFAULT_SCRAMBLE_RADIUS,
                        help="Max random translation in Angstrom (default 5.0)")
    parser.add_argument("--no-scramble", action="store_true",
                        help="Dock from the input pose unchanged. Only for round 2, "
                             "where the input is already an independent docked pose.")
    parser.add_argument("--minimum-scramble-rmsd", type=float,
                        default=MINIMUM_SCRAMBLE_RMSD,
                        help="Fail if scrambling moved the ligand less than this")
    parser.add_argument("--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS)
    parser.add_argument("--num-modes", type=int, default=DEFAULT_NUM_MODES)
    parser.add_argument("--autobox-add", type=float, default=DEFAULT_AUTOBOX_ADD)
    parser.add_argument("--cnn-scoring", default="rescore",
                        choices=("none", "rescore", "refinement", "all"))
    parser.add_argument("--autobox-reference", type=Path, default=None,
                        help="Structure defining the docking box (default: --ligand). "
                             "Set this whenever the receptor is in a different frame "
                             "from the input ligand, e.g. round-two docking into a "
                             "GCMC-hydrated receptor.")
    parser.add_argument("--gnina-binary", default="gnina")
    parser.add_argument("--gnina-ld-library-path", default=None,
                        help="Prepended to LD_LIBRARY_PATH for the GNINA call. The "
                             "release binaries need libcudnn.so.9 at runtime; point "
                             "this at a conda env lib/ that provides it.")
    parser.add_argument("--scramble-only", action="store_true",
                        help="Write the scrambled ligand and stop. Exercises the "
                             "scramble path without needing the GNINA binary.")
    opt = parser.parse_args()
    if opt.top_n < 1:
        raise SystemExit("--top-n must be positive")
    if opt.scramble_radius < 0:
        raise SystemExit("--scramble-radius must be non-negative")
    if opt.num_modes < opt.top_n:
        raise SystemExit(
            f"--num-modes ({opt.num_modes}) must be at least --top-n ({opt.top_n})"
        )
    return opt


def main() -> None:
    opt = options()
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    crystal = single_record(opt.ligand)
    # Poses are compared against whatever defined the box, because that is the structure
    # in the receptor's frame. With --autobox-reference the input ligand can be tens of
    # Angstrom away (round two: template frame vs production frame) and comparing to it
    # yields a large number that means nothing.
    reference = (
        single_record(opt.autobox_reference)
        if opt.autobox_reference is not None else crystal
    )
    crystal_heavy = heavy_coordinates(reference)

    report: dict[str, object] = {
        "ligand_id": opt.ligand_id,
        "receptor": str(opt.receptor),
        "input_ligand": str(opt.ligand),
        "seed": opt.seed,
        "scrambled": not opt.no_scramble,
        "scramble_radius_angstrom": opt.scramble_radius,
    }

    if opt.no_scramble:
        docking_input = opt.ligand
        report["scramble_rmsd_angstrom"] = None
    else:
        rng = np.random.default_rng(opt.seed)
        scrambled = scramble_ligand(crystal, rng, opt.scramble_radius)
        scramble_rmsd = in_place_rmsd(
            heavy_coordinates(crystal), heavy_coordinates(scrambled)
        )
        report["scramble_rmsd_angstrom"] = scramble_rmsd
        # The verification half: prove the search actually started somewhere else.
        if scramble_rmsd < opt.minimum_scramble_rmsd:
            raise SystemExit(
                f"Scramble moved the ligand only {scramble_rmsd:.2f} A "
                f"(minimum {opt.minimum_scramble_rmsd:.2f}); a recovered pose would not "
                "demonstrate a genuine search. Raise --scramble-radius or change --seed."
            )
        docking_input = opt.output_dir / f"{opt.ligand_id}_scrambled.sdf"
        writer = Chem.SDWriter(str(docking_input))
        scrambled.SetProp("_Name", f"{opt.ligand_id}_scrambled")
        writer.write(scrambled)
        writer.close()
        report["scrambled_ligand"] = str(docking_input)
        print(f"scramble RMSD from input pose: {scramble_rmsd:.2f} A")

    if opt.scramble_only:
        report["status"] = "scramble_only"
        (opt.output_dir / "docking_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(f"DOCKING_REPORT={opt.output_dir / 'docking_report.json'}")
        return

    docked = opt.output_dir / f"{opt.ligand_id}_docked.sdf"
    run_gnina(
        binary=opt.gnina_binary,
        receptor=opt.receptor,
        ligand=docking_input,
        autobox_ligand=opt.autobox_reference or opt.ligand,
        output=docked,
        exhaustiveness=opt.exhaustiveness,
        num_modes=opt.num_modes,
        seed=opt.seed,
        autobox_add=opt.autobox_add,
        cnn_scoring=opt.cnn_scoring,
        ld_library_path=opt.gnina_ld_library_path,
    )

    poses = read_poses(docked)
    written = write_pose_sdfs(poses, opt.output_dir / "poses", opt.ligand_id, opt.top_n)

    # How close did the best pose get back to the input? Reported, never enforced --
    # for a genuinely unknown ligand there is no crystal pose to recover.
    for entry in written:
        pose = single_record(Path(str(entry["path"])))
        pose_heavy = heavy_coordinates(pose)
        if pose_heavy.shape != crystal_heavy.shape:
            # Should not happen for a pose of the same ligand, but a silent shape
            # mismatch would give a meaningless number, so record the reason instead.
            entry["rmsd_to_input_pose_angstrom"] = None
            entry["rmsd_note"] = (
                f"heavy-atom count {pose_heavy.shape[0]} != input {crystal_heavy.shape[0]}"
            )
            continue
        entry["rmsd_to_input_pose_angstrom"] = in_place_rmsd(crystal_heavy, pose_heavy)
        entry["superposed_rmsd_to_input_angstrom"] = proper_kabsch_rmsd(
            crystal_heavy, pose_heavy
        )

    report["poses"] = written
    report["poses_returned"] = len(poses)
    report["best_rmsd_to_input_angstrom"] = min(
        float(entry["rmsd_to_input_pose_angstrom"]) for entry in written
    )
    report["status"] = "completed"
    (opt.output_dir / "docking_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    print(f"poses returned : {len(poses)}")
    print(f"poses kept     : {len(written)}")
    print(f"best RMSD to input pose: {report['best_rmsd_to_input_angstrom']:.2f} A")
    for entry in written:
        print(
            f"  {entry['pose_id']}  CNNscore={entry['cnn_score']}  "
            f"rmsd_to_input={float(entry['rmsd_to_input_pose_angstrom']):.2f} A"
        )
    print(f"DOCKING_REPORT={opt.output_dir / 'docking_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
