#!/usr/bin/env python3
"""Build one common site catalog and reanalyze every completed series task."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

from pipeline_utils import (
    read_json,
    require_file,
    sha256,
    validate_recorded_outputs,
    write_json_atomic,
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-catalog-support", type=int, default=2)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    return parser.parse_args()


def read_tasks(path: Path) -> list[dict[str, str]]:
    with require_file(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"task_index", "ligand_id", "replica", "seed", "prefix", "run_dir"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    indices = [int(row["task_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Manifest task indices are not contiguous from zero")
    return rows


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def normalize_run_directory(value: str) -> tuple[Path, dict[str, str] | None]:
    """Return a clean run path, atomically migrating the legacy CR name.

    The first series manifest used csv.DictWriter's CRLF default. Bash removed
    only the newline and retained the carriage return on the final TSV field,
    creating directories named ``repN\r``. The dependent finalizer runs only
    after every array task succeeds, so this is the safe boundary at which to
    rename the complete directory before common-catalog analysis.
    """
    clean = Path(value.rstrip("\r")).resolve()
    legacy = clean.with_name(f"{clean.name}\r")
    clean_exists = clean.exists()
    legacy_exists = legacy.exists()
    if clean_exists and legacy_exists:
        raise FileExistsError(
            f"Both clean and legacy CR-suffixed run directories exist: {clean}, {legacy!r}"
        )
    if legacy_exists:
        if not legacy.is_dir():
            raise NotADirectoryError(legacy)
        legacy.rename(clean)
        migration = {"from": str(legacy), "to": str(clean)}
        print(f"Normalized legacy run directory: {legacy!r} -> {clean}", flush=True)
        return clean, migration
    if not clean.is_dir():
        raise FileNotFoundError(f"Run directory is missing: {clean}")
    return clean, None


def main() -> None:
    started = time.time()
    opt = options()
    if opt.minimum_catalog_support < 1:
        raise ValueError("--minimum-catalog-support must be positive")
    project = opt.project_dir.resolve()
    scripts = project / "scripts"
    receptor = require_file(opt.receptor)
    manifest = require_file(opt.manifest)
    tasks = read_tasks(manifest)
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    common_catalog = output / "common-site-catalog.csv"
    summary_path = output / "series-density-analysis.json"
    summary_path.unlink(missing_ok=True)

    provisional_catalogs: list[Path] = []
    normalized_run_dirs: dict[int, Path] = {}
    legacy_path_migrations: list[dict[str, str]] = []
    for row in tasks:
        task_index = int(row["task_index"])
        run_dir, migration = normalize_run_directory(row["run_dir"])
        normalized_run_dirs[task_index] = run_dir
        if migration is not None:
            legacy_path_migrations.append(migration)
        completion = read_json(run_dir / "series_task.complete.json")
        if completion.get("status") != "completed":
            raise ValueError(f"Series task is incomplete: {run_dir}")
        expected = {
            "task_index": int(row["task_index"]),
            "ligand_id": row["ligand_id"],
            "replica": int(row["replica"]),
            "seed": int(row["seed"]),
            "prefix": row["prefix"],
            "profile": opt.profile,
        }
        mismatches = {
            key: (completion.get(key), value)
            for key, value in expected.items()
            if completion.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Series task metadata differs from manifest at {run_dir}: {mismatches}")
        validate_recorded_outputs(run_dir / "production" / "production.complete.json")
        validate_recorded_outputs(run_dir / "density_analysis" / "density_analysis.complete.json")
        provisional_catalogs.append(
            require_file(run_dir / "density_analysis" / f"{row['prefix']}-site-catalog.csv")
        )

    run(
        [
            sys.executable,
            "-u",
            str(scripts / "ev71_merge_site_catalogs.py"),
            *[str(path) for path in provisional_catalogs],
            "--minimum-support",
            str(opt.minimum_catalog_support),
            "--output",
            str(common_catalog),
        ]
    )

    materialized_reference_hashes = {
        sha256(require_file(run_dir / "preparation" / "receptor_input.pdb"))
        for run_dir in normalized_run_dirs.values()
    }
    if len(materialized_reference_hashes) != 1:
        raise ValueError(
            "Series tasks do not share one identical materialized receptor reference"
        )

    results: list[dict[str, object]] = []
    for count, row in enumerate(tasks, start=1):
        run_dir = normalized_run_dirs[int(row["task_index"])]
        prefix = row["prefix"]
        common_output = run_dir / "common_site_analysis"
        run(
            [
                sys.executable,
                "-u",
                str(scripts / "ev71_density_sites.py"),
                "--topology",
                str(run_dir / "production" / f"{prefix}-loch-ghosts.pdb"),
                "--trajectory",
                str(run_dir / "production" / f"{prefix}-raw.dcd"),
                "--ghost-file",
                str(run_dir / "production" / f"{prefix}-gcmc-ghosts.txt"),
                "--alignment-reference",
                str(require_file(run_dir / "preparation" / "receptor_input.pdb")),
                "--site-catalog",
                str(common_catalog),
                "--output-dir",
                str(common_output),
                "--prefix",
                prefix,
            ]
        )
        marker = common_output / "density_analysis.complete.json"
        validate_recorded_outputs(marker)
        results.append(
            {
                "task_index": int(row["task_index"]),
                "ligand_id": row["ligand_id"],
                "replica": int(row["replica"]),
                "run_dir": str(run_dir),
                "common_metrics": str(common_output / f"{prefix}-site-metrics.csv"),
                "common_analysis_marker_sha256": sha256(marker),
            }
        )
        print(f"Common-catalog analysis {count}/{len(tasks)} complete", flush=True)

    catalog_summary = read_json(common_catalog.with_suffix(".json"))
    replica_counts = {
        ligand: sum(1 for candidate in tasks if candidate["ligand_id"] == ligand)
        for ligand in {row["ligand_id"] for row in tasks}
    }
    if len(set(replica_counts.values())) != 1:
        raise ValueError(f"Ligands do not have a uniform replica count: {replica_counts}")
    summary = {
        "status": "completed",
        "validation_scope": (
            "full_simulation_common_catalog_density_analysis"
            if opt.profile == "full"
            else "smoke_plumbing_common_catalog_density_analysis"
        ),
        "profile": opt.profile,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "receptor": str(receptor),
        "receptor_sha256": sha256(receptor),
        "materialized_receptor_pdb_sha256": next(iter(materialized_reference_hashes)),
        "tasks": len(tasks),
        "ligands": len({row["ligand_id"] for row in tasks}),
        "replicates_per_ligand": next(iter(replica_counts.values())),
        "minimum_catalog_support": opt.minimum_catalog_support,
        "common_catalog": str(common_catalog),
        "common_catalog_sha256": sha256(common_catalog),
        "common_sites": int(catalog_summary["common_sites"]),
        "legacy_path_migrations": legacy_path_migrations,
        "results": results,
        "wall_seconds": time.time() - started,
    }
    write_json_atomic(summary_path, summary)
    print(f"SERIES_RESULT={summary_path}", flush=True)


if __name__ == "__main__":
    main()
