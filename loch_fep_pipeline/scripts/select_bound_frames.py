#!/usr/bin/env python3
"""Select one representative equilibrated bound frame per ligand across replicates.

Given a density-series run root laid out as ``LIGAND/repN/{common_site_analysis,
production}``, pick, for each ligand, the **medoid replicate**: the replicate
whose common-catalog hydration-site occupancy vector is closest to the 6-replicate
mean.  That replicate's ``*-production-final.{prmtop,rst7}`` (waters already placed
by the endpoint GCMC) becomes the ligand's FEP bound frame.

Why medoid, not an average: waters are indistinguishable and sit in different
positions in each replicate, so averaging Cartesian water coordinates is
meaningless.  Selecting the replicate closest to the consensus occupancy keeps a
real, physically valid structure while representing the common water state.

Run this once, standalone, and inspect ``selection_report.csv`` before launching
FEP.  The emitted ``OUTPUT/LIGAND/production-final.{prmtop,rst7}`` layout is what
``make_fep_manifest.py --bound-frame-root`` consumes, so the 52 FEP edges reuse
these placed waters without recomputing any GCMC.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pipeline_utils import require_file, write_json_atomic


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", type=Path, required=True,
                        help="Directory containing LIGAND/repN/ endpoint runs")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Where to write LIGAND/production-final.{prmtop,rst7}")
    parser.add_argument("--ligands", type=Path,
                        help="Optional file with one ligand id per line (default: all "
                        "LIGAND dirs under --series-root that contain replicates)")
    parser.add_argument("--replicate-glob", default="rep*")
    parser.add_argument("--occupancy-column", default="occupancy")
    parser.add_argument("--site-metrics-subdir", default="common_site_analysis",
                        help="Per-rep subdir holding the *-site-metrics.csv scored "
                        "against the shared catalog (comparable across replicates)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy frames instead of symlinking them")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the ghost-free physical-water check (faster; skips "
                        "loading sire). Validation rejects frames with any zero-"
                        "interaction (ghost) waters.")
    return parser.parse_args()


def occupancy_vector(metrics_csv: Path, column: str) -> dict[str, float]:
    with metrics_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "site_id" not in rows[0] or column not in rows[0]:
        raise ValueError(f"{metrics_csv} lacks site_id/{column} columns")
    vector: dict[str, float] = {}
    for row in rows:
        vector[row["site_id"]] = float(row[column])
    if len(vector) != len(rows):
        raise ValueError(f"Duplicate site_id in {metrics_csv}")
    return vector


def one_glob(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}; found {len(matches)}")
    return require_file(matches[0])


def production_final(rep_dir: Path) -> tuple[Path, Path]:
    production = rep_dir / "production"
    top = one_glob(production, "*-production-final.prmtop")
    rst = one_glob(production, "*-production-final.rst7")
    if top.stem.removesuffix("-production-final") != rst.stem.removesuffix("-production-final"):
        raise ValueError(f"Mismatched production-final prefixes in {production}")
    return top, rst


def validate_ghost_free(top: Path, rst: Path) -> int:
    import sire as sr  # local import: only needed when validating
    try:
        from ev71_loch_common import physical_water_audit  # project_2 deployment
    except ImportError:
        from loch_common import physical_water_audit  # portable automated_pipeline
    audit = physical_water_audit(sr.load(str(top), str(rst)))
    zero = int(audit["zero_interaction_water_count"])
    if zero:
        raise ValueError(
            f"{top.name} contains {zero} zero-interaction (ghost) waters; not a clean "
            "physical handoff — do not use as a FEP bound frame"
        )
    return int(audit["water_molecules"])


def ligand_ids(opt: argparse.Namespace) -> list[str]:
    if opt.ligands is not None:
        names = [line.strip() for line in require_file(opt.ligands).read_text().splitlines()]
        return [name for name in names if name]
    root = opt.series_root
    return sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and any(child.glob(opt.replicate_glob))
    )


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        import shutil
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def select_for_ligand(ligand: str, opt: argparse.Namespace) -> dict:
    ligand_dir = opt.series_root / ligand
    reps = sorted(p for p in ligand_dir.glob(opt.replicate_glob) if p.is_dir())
    if not reps:
        raise ValueError(f"No replicates matching {opt.replicate_glob} under {ligand_dir}")

    vectors: dict[str, dict[str, float]] = {}
    for rep in reps:
        metrics = one_glob(rep / opt.site_metrics_subdir, "*-site-metrics.csv")
        vectors[rep.name] = occupancy_vector(metrics, opt.occupancy_column)

    shared = sorted(set.intersection(*(set(v) for v in vectors.values())))
    if not shared:
        raise ValueError(f"{ligand}: replicates share no common hydration sites")
    rep_names = [rep.name for rep in reps]
    matrix = np.array([[vectors[name][site] for site in shared] for name in rep_names])
    mean = matrix.mean(axis=0)
    distances = np.linalg.norm(matrix - mean, axis=1)
    medoid_index = int(np.argmin(distances))
    medoid_name = rep_names[medoid_index]
    # Replicate spread: mean pairwise occupancy distance (large => reps disagree).
    if len(rep_names) > 1:
        pair = [
            float(np.linalg.norm(matrix[i] - matrix[j]))
            for i in range(len(rep_names)) for j in range(i + 1, len(rep_names))
        ]
        spread = float(np.mean(pair))
    else:
        spread = 0.0

    top, rst = production_final(reps[medoid_index])
    waters = None
    if not opt.no_validate:
        waters = validate_ghost_free(top, rst)

    out_dir = opt.output_root / ligand
    link_or_copy(top, out_dir / "production-final.prmtop", opt.copy)
    link_or_copy(rst, out_dir / "production-final.rst7", opt.copy)

    return {
        "ligand": ligand,
        "replicates": len(rep_names),
        "chosen_replicate": medoid_name,
        "occupancy_distance_to_mean": float(distances[medoid_index]),
        "replicate_occupancy_spread": spread,
        "shared_sites": len(shared),
        "physical_waters": waters,
        "source_prmtop": str(top),
        "source_rst7": str(rst),
        "bound_prmtop": str((out_dir / "production-final.prmtop").resolve()),
        "bound_rst7": str((out_dir / "production-final.rst7").resolve()),
    }


def main() -> None:
    opt = options()
    opt.output_root.mkdir(parents=True, exist_ok=True)
    ligands = ligand_ids(opt)
    if not ligands:
        raise SystemExit("No ligands found")
    results, failures = [], []
    for ligand in ligands:
        try:
            record = select_for_ligand(ligand, opt)
            results.append(record)
            print(
                f"{ligand}: chose {record['chosen_replicate']} "
                f"(dist_to_mean={record['occupancy_distance_to_mean']:.3f}, "
                f"rep_spread={record['replicate_occupancy_spread']:.3f})",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - report and continue the batch
            failures.append({"ligand": ligand, "error": f"{type(error).__name__}: {error}"})
            print(f"{ligand}: FAILED — {type(error).__name__}: {error}", flush=True)

    report_csv = opt.output_root / "selection_report.csv"
    if results:
        with report_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    write_json_atomic(
        opt.output_root / "selection_report.json",
        {"selected": results, "failures": failures,
         "validated_ghost_free": not opt.no_validate},
    )
    print(f"\nSelected {len(results)}/{len(ligands)} ligands. Report: {report_csv}", flush=True)
    if failures:
        print(f"{len(failures)} ligand(s) failed — see selection_report.json", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
