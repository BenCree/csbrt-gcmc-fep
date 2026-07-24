#!/usr/bin/env python3
"""Extract named ligand records from the OpenBind multi-record SDF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rdkit import Chem

from pipeline_utils import require_file, sha256, write_json_atomic


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ligand-id",
        action="append",
        default=[],
        help="SDF title/ligand_name to extract; repeat as needed (default: all)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def record_ids(molecule: Chem.Mol) -> set[str]:
    values = {molecule.GetProp("_Name").strip()}
    if molecule.HasProp("ligand_name"):
        values.add(molecule.GetProp("ligand_name").strip())
    return {value for value in values if value}


def integer_property(molecule: Chem.Mol, name: str, default: int) -> int:
    if not molecule.HasProp(name):
        return default
    raw = float(molecule.GetProp(name))
    if not raw.is_integer():
        raise ValueError(f"Ligand property {name!r} is not an integer: {raw}")
    return int(raw)


def main() -> None:
    opt = options()
    source = require_file(opt.input)
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    supplier = Chem.SDMolSupplier(
        str(source), removeHs=False, sanitize=True, strictParsing=True
    )
    molecules = list(supplier)
    invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
    if invalid:
        raise ValueError(f"Invalid SDF records at zero-based indices {invalid}")

    requested = set(opt.ligand_id)
    matches: dict[str, list[Chem.Mol]] = {name: [] for name in requested}
    selected: list[Chem.Mol] = []
    seen_titles: set[str] = set()
    for molecule in molecules:
        assert molecule is not None
        title = molecule.GetProp("_Name").strip()
        if not title or not SAFE_NAME.fullmatch(title):
            raise ValueError(f"Unsafe or empty SDF title {title!r}")
        if title in seen_titles:
            raise ValueError(f"Duplicate SDF title {title!r}")
        seen_titles.add(title)
        if not requested:
            selected.append(molecule)
        else:
            for name in requested & record_ids(molecule):
                matches[name].append(molecule)

    if requested:
        for name, found in matches.items():
            if len(found) != 1:
                raise ValueError(f"Ligand selector {name!r} matched {len(found)} records")
        selected = [matches[name][0] for name in opt.ligand_id]

    manifest: list[dict[str, object]] = []
    for molecule in selected:
        title = molecule.GetProp("_Name").strip()
        path = output / f"{title}.sdf"
        if path.exists() and not opt.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        writer = Chem.SDWriter(str(path))
        writer.write(molecule)
        writer.close()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}")
        formal_charge = int(Chem.GetFormalCharge(molecule))
        declared_charge = integer_property(molecule, "charge", formal_charge)
        if declared_charge != formal_charge:
            raise ValueError(
                f"Ligand {title!r} declares charge {declared_charge}, but its "
                f"bond graph has formal charge {formal_charge}"
            )
        radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
        multiplicity = integer_property(molecule, "multiplicity", radical_electrons + 1)
        if multiplicity < 1:
            raise ValueError(f"Ligand {title!r} has invalid multiplicity {multiplicity}")
        manifest.append(
            {
                "ligand_id": title,
                "aliases": sorted(record_ids(molecule)),
                "atoms": molecule.GetNumAtoms(),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "formal_charge": formal_charge,
                "declared_charge": declared_charge,
                "multiplicity": multiplicity,
                "radical_electrons": radical_electrons,
                # Keep extracted artifacts portable with the inputs directory.
                "output": path.name,
                "output_sha256": sha256(path),
            }
        )
        print(f"Created {path}", flush=True)

    write_json_atomic(
        output / "manifest.json",
        {
            "source": str(source),
            "source_sha256": sha256(source),
            "source_records": len(molecules),
            "requested_ids": opt.ligand_id,
            "ligands": manifest,
        },
    )
    print(f"Extracted {len(selected)} of {len(molecules)} ligand records", flush=True)


if __name__ == "__main__":
    main()
