#!/usr/bin/env python3
"""Screen candidate FEP edges for allowability, without running any GPU work.

`prepare_fep.py` decides edge validity as a side effect of building the edge, and it
raises on the first failure -- correct when you are committed to an edge, wrong when you
are surveying which edges are possible. This applies the same rules and *records* the
verdict instead, so one unusable pair does not kill the survey.

Two gates, cheapest first:

1. Formal charge equality, read straight from each endpoint's
   `preparation.complete.json` signature. No BioSimSpace, no file parsing beyond JSON,
   so a charge-mismatched pair costs nothing.
2. Maximum common substructure mapping via `BSS.Align.matchAtoms`, giving
   `mapped_heavy_fraction = mapped_heavy / min(heavy_a, heavy_b)` -- the same quantity
   and the same denominator `prepare_fep.py` gates on.

Identical ligands (equal `ligand_sha256`) take the same shortcut `prepare_fep.py` uses
and skip mapping entirely. That matters for pose exploration, where several poses of one
compound are separate endpoints with byte-identical chemistry.

Output is a TSV that `select_fep_edges.py` consumes and that, filtered to allowable
rows, is the `--edges` input `make_fep_manifest.py` expects.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

MINIMUM_MAPPED_HEAVY_FRACTION = 0.50
MAPPING_TIMEOUT_SECONDS = 30.0


def read_endpoint(preparation: Path) -> dict[str, object]:
    """Pull the cheap identity fields out of a completed endpoint preparation."""
    marker = preparation / "preparation.complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"No preparation.complete.json under {preparation}")
    payload = json.loads(marker.read_text())
    if payload.get("status") != "completed":
        raise ValueError(f"{marker} is not marked completed")
    signature = payload.get("signature", {})
    for key in ("ligand_id", "ligand_charge", "ligand_sha256"):
        if key not in signature:
            raise ValueError(f"{marker} signature is missing {key}")
    return {
        "ligand_id": str(signature["ligand_id"]),
        "charge": int(signature["ligand_charge"]),
        "sha256": str(signature["ligand_sha256"]),
        "preparation": preparation,
    }


def discover_endpoints(run_root: Path, replica: int) -> list[dict[str, object]]:
    """Find <run_root>/<ligand>/rep<N>/preparation for every ligand present."""
    found = []
    for ligand_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        preparation = ligand_dir / f"rep{replica}" / "preparation"
        if (preparation / "preparation.complete.json").is_file():
            found.append(read_endpoint(preparation))
    if not found:
        raise SystemExit(
            f"No completed endpoint preparations under {run_root} for rep{replica}"
        )
    return found


def heavy_atom_coordinates(molecule) -> np.ndarray:
    """Heavy-atom coordinates of a BioSimSpace molecule, in Angstrom."""
    coordinates = []
    for atom in molecule.getAtoms():
        if atom.element().lower().startswith("hydrogen"):
            continue
        position = atom.coordinates()
        coordinates.append(
            [position.x().value(), position.y().value(), position.z().value()]
        )
    return np.asarray(coordinates, dtype=float)


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


def screen_pair(
    endpoint_a: dict[str, object],
    endpoint_b: dict[str, object],
    *,
    minimum_fraction: float,
    allow_charge_change: bool,
    allow_ring_breaking: bool,
    timeout_seconds: float,
    skip_mapping: bool,
) -> dict[str, object]:
    """Apply prepare_fep.py's validity rules to one pair, recording rather than raising."""
    record: dict[str, object] = {
        "state_a": endpoint_a["ligand_id"],
        "state_b": endpoint_b["ligand_id"],
        "edge_id": f"{endpoint_a['ligand_id']}_to_{endpoint_b['ligand_id']}",
        "charge_change": int(endpoint_b["charge"]) - int(endpoint_a["charge"]),
        "mapped_heavy_fraction": "",
        "centroid_distance_angstrom": "",
        "pose_rmsd_angstrom": "",
        "allowable": False,
        "reason": "",
    }

    if record["charge_change"] != 0 and not allow_charge_change:
        record["reason"] = (
            f"charge change {endpoint_a['charge']} -> {endpoint_b['charge']} "
            "requires --allow-charge-change and a finite-size correction"
        )
        return record

    if skip_mapping:
        record["allowable"] = True
        record["reason"] = "charge-only screen (mapping skipped)"
        return record

    # Imported lazily: BioSimSpace costs seconds to import and the charge gate above
    # rejects many pairs without ever needing it.
    import BioSimSpace as BSS

    def load(endpoint: dict[str, object]):
        directory = endpoint["preparation"]
        system = BSS.IO.readMolecules(
            [str(directory / "ligand.prmtop"), str(directory / "ligand.rst7")]
        )
        if system.nMolecules() != 1:
            raise ValueError(f"Ligand topology in {directory} is not a single molecule")
        return system.getMolecule(0)

    try:
        ligand_a = load(endpoint_a)
        ligand_b = load(endpoint_b)
    except Exception as error:  # noqa: BLE001 - a bad endpoint must not kill the survey
        record["reason"] = f"could not load ligand topology: {error}"
        return record

    coordinates_a = heavy_atom_coordinates(ligand_a)
    coordinates_b = heavy_atom_coordinates(ligand_b)
    centroid_distance = float(
        np.linalg.norm(coordinates_a.mean(axis=0) - coordinates_b.mean(axis=0))
    )
    record["centroid_distance_angstrom"] = f"{centroid_distance:.3f}"
    if coordinates_a.shape == coordinates_b.shape:
        record["pose_rmsd_angstrom"] = (
            f"{proper_kabsch_rmsd(coordinates_a, coordinates_b):.3f}"
        )

    if endpoint_a["sha256"] == endpoint_b["sha256"]:
        # Same chemistry, different pose. prepare_fep.py takes the identity mapping here.
        if ligand_a.nAtoms() != ligand_b.nAtoms():
            record["reason"] = "identical ligand hashes but different atom counts"
            return record
        record["mapped_heavy_fraction"] = "1.0000"
        record["allowable"] = True
        record["reason"] = "identical ligand chemistry (identity mapping)"
        return record

    try:
        mapping = BSS.Align.matchAtoms(
            ligand_a,
            ligand_b,
            timeout=timeout_seconds * BSS.Units.Time.second,
            complete_rings_only=not allow_ring_breaking,
            max_scoring_matches=100,
        )
    except Exception as error:  # noqa: BLE001
        record["reason"] = f"atom mapping failed: {error}"
        return record
    if not isinstance(mapping, dict) or not mapping:
        record["reason"] = "BioSimSpace produced no atom mapping"
        return record

    atoms_a = ligand_a.getAtoms()
    atoms_b = ligand_b.getAtoms()
    mapped_heavy = sum(
        not atoms_a[index_a].element().lower().startswith("hydrogen")
        and not atoms_b[index_b].element().lower().startswith("hydrogen")
        for index_a, index_b in mapping.items()
    )
    heavy_a = sum(not atom.element().lower().startswith("hydrogen") for atom in atoms_a)
    heavy_b = sum(not atom.element().lower().startswith("hydrogen") for atom in atoms_b)
    fraction = mapped_heavy / min(heavy_a, heavy_b)
    record["mapped_heavy_fraction"] = f"{fraction:.4f}"
    if fraction < minimum_fraction:
        record["reason"] = (
            f"mapped heavy fraction {fraction:.3f} below minimum {minimum_fraction:.3f}"
        )
        return record
    record["allowable"] = True
    record["reason"] = "ok"
    return record


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint-run-root", type=Path,
                        help="Root holding <ligand>/rep<N>/preparation for each endpoint")
    source.add_argument("--preparation-dirs", type=Path, nargs="+",
                        help="Explicit list of preparation directories")
    parser.add_argument("--replica", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mapped-heavy-fraction", type=float,
                        default=MINIMUM_MAPPED_HEAVY_FRACTION)
    parser.add_argument("--mapping-timeout-seconds", type=float,
                        default=MAPPING_TIMEOUT_SECONDS)
    parser.add_argument("--allow-charge-change", action="store_true")
    parser.add_argument("--allow-ring-breaking", action="store_true")
    parser.add_argument("--skip-mapping", action="store_true",
                        help="Charge-only screen; skips BioSimSpace entirely. Fast survey "
                             "of network shape, but the mapping gate is not applied.")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Emit rejected pairs too (default writes allowable only)")
    opt = parser.parse_args()
    if not 0.0 < opt.minimum_mapped_heavy_fraction <= 1.0:
        raise SystemExit("--minimum-mapped-heavy-fraction must be in (0, 1]")
    if opt.mapping_timeout_seconds <= 0:
        raise SystemExit("--mapping-timeout-seconds must be positive")
    return opt


FIELDS = (
    "state_a", "state_b", "edge_id", "mapped_heavy_fraction", "charge_change",
    "centroid_distance_angstrom", "pose_rmsd_angstrom", "allowable", "reason",
)


def main() -> None:
    opt = options()
    if opt.endpoint_run_root is not None:
        endpoints = discover_endpoints(opt.endpoint_run_root, opt.replica)
    else:
        endpoints = [read_endpoint(path) for path in opt.preparation_dirs]
    if len(endpoints) < 2:
        raise SystemExit(f"Need at least two endpoints to form an edge; found {len(endpoints)}")

    identifiers = [endpoint["ligand_id"] for endpoint in endpoints]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit(f"Duplicate ligand ids among endpoints: {sorted(identifiers)}")

    records = [
        screen_pair(
            endpoint_a, endpoint_b,
            minimum_fraction=opt.minimum_mapped_heavy_fraction,
            allow_charge_change=opt.allow_charge_change,
            allow_ring_breaking=opt.allow_ring_breaking,
            timeout_seconds=opt.mapping_timeout_seconds,
            skip_mapping=opt.skip_mapping,
        )
        for endpoint_a, endpoint_b in itertools.combinations(endpoints, 2)
    ]

    emitted = records if opt.all_pairs else [r for r in records if r["allowable"]]
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    with opt.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for record in emitted:
            writer.writerow({key: record[key] for key in FIELDS})

    allowable = sum(1 for record in records if record["allowable"])
    print(f"endpoints      : {len(endpoints)}")
    print(f"candidate pairs: {len(records)}")
    print(f"allowable      : {allowable}")
    print(f"rejected       : {len(records) - allowable}")
    reasons: dict[str, int] = {}
    for record in records:
        if not record["allowable"]:
            key = str(record["reason"]).split(":")[0].split(" below ")[0]
            reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  {count:4d}  {reason}")
    print(f"ALLOWABLE_EDGES={opt.output}")


if __name__ == "__main__":
    main()
