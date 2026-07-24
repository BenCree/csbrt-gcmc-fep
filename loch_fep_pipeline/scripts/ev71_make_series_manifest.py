#!/usr/bin/env python3
"""Create a deterministic ligand-by-replica task manifest from an SDF library."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from string import Formatter

from rdkit import Chem

from pipeline_utils import require_file, sha256, write_json_atomic


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MAX_SEED = 2**31 - 1
SEED_BLOCK_SPACING = 1000
LIGAND_SEED_STRIDE = 1_000_000


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-library", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=20260714)
    parser.add_argument("--prefix-template", default="ev71_2a_{ligand_id}")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    opt = options()
    if opt.replicates < 1:
        raise ValueError("--replicates must be positive")
    if opt.replicates * SEED_BLOCK_SPACING > LIGAND_SEED_STRIDE:
        raise ValueError(
            f"--replicates cannot exceed {LIGAND_SEED_STRIDE // SEED_BLOCK_SPACING}"
        )
    if opt.base_seed < 0:
        raise ValueError("--base-seed must be nonnegative")
    fields = [
        (field_name, format_spec, conversion)
        for _, field_name, format_spec, conversion in Formatter().parse(opt.prefix_template)
        if field_name is not None
    ]
    if fields != [("ligand_id", "", None)]:
        raise ValueError("--prefix-template must contain exactly one plain {ligand_id}")
    probe = opt.prefix_template.format(ligand_id="probe")
    if not SAFE_NAME.fullmatch(probe):
        raise ValueError("--prefix-template produces unsafe filenames")

    library = require_file(opt.ligand_library)
    supplier = Chem.SDMolSupplier(
        str(library), removeHs=False, sanitize=True, strictParsing=True
    )
    molecules = list(supplier)
    invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
    if invalid:
        raise ValueError(f"Invalid SDF records at zero-based indices {invalid}")
    ligand_ids: list[str] = []
    for molecule in molecules:
        assert molecule is not None
        ligand_id = molecule.GetProp("_Name").strip()
        if not ligand_id or not SAFE_NAME.fullmatch(ligand_id):
            raise ValueError(f"Unsafe or empty SDF title {ligand_id!r}")
        ligand_ids.append(ligand_id)
    if not ligand_ids or len(set(ligand_ids)) != len(ligand_ids):
        raise ValueError("Ligand library has no records or duplicate titles")

    output = opt.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = opt.run_root.resolve()
    rows: list[dict[str, object]] = []
    for ligand_index, ligand_id in enumerate(ligand_ids):
        for replica in range(1, opt.replicates + 1):
            task_index = ligand_index * opt.replicates + replica - 1
            # Reserve a fixed range for each ligand. Changing --replicates then
            # leaves every existing ligand/replica seed and checkpoint stable.
            seed = (
                opt.base_seed
                + ligand_index * LIGAND_SEED_STRIDE
                + (replica - 1) * SEED_BLOCK_SPACING
            )
            if seed + 3 > MAX_SEED:
                raise ValueError(f"Seed block exceeds signed 32-bit range at task {task_index}")
            prefix = opt.prefix_template.format(ligand_id=ligand_id)
            if not SAFE_NAME.fullmatch(prefix):
                raise ValueError(f"Unsafe prefix generated for {ligand_id!r}: {prefix!r}")
            rows.append(
                {
                    "task_index": task_index,
                    "ligand_id": ligand_id,
                    "replica": replica,
                    "seed": seed,
                    "prefix": prefix,
                    "run_dir": str(run_root / ligand_id / f"rep{replica}"),
                }
            )

    fields = ("task_index", "ligand_id", "replica", "seed", "prefix", "run_dir")
    with output.open("w", newline="") as handle:
        # Explicit LF prevents Bash array workers from retaining a trailing CR
        # in the final run_dir field and creating directories named ``repN\r``.
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed",
        "ligand_library": str(library),
        "ligand_library_sha256": sha256(library),
        "ligand_ids": ligand_ids,
        "ligands": len(ligand_ids),
        "replicates": opt.replicates,
        "tasks": len(rows),
        "base_seed": opt.base_seed,
        "seed_block_spacing": SEED_BLOCK_SPACING,
        "ligand_seed_stride": LIGAND_SEED_STRIDE,
        "prefix_template": opt.prefix_template,
        "run_root": str(run_root),
        "manifest": str(output),
        "manifest_sha256": sha256(output),
    }
    write_json_atomic(output.with_suffix(".json"), summary)
    print(
        f"Created {output}: {len(ligand_ids)} ligands x {opt.replicates} replicas = {len(rows)} tasks",
        flush=True,
    )


if __name__ == "__main__":
    main()
