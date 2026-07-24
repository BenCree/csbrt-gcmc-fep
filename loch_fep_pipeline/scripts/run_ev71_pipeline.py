#!/usr/bin/env python3
"""Run the four checkpointed EV71 Loch stages for one named OpenBind ligand."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from pipeline_utils import (
    implementation_signature,
    read_json,
    require_file,
    sha256,
    write_json_atomic,
)


STAGES = ("preparation", "equilibration", "production", "postprocessing")
EQUILIBRATION_PROFILE_ARGS = {
    "full": {
        "initial-attempts": 10_000,
        "uvt1-cycles": 100,
        "uvt1-attempts": 1_000,
        "uvt1-md-steps": 5,
        "uvt1-report-interval": 100,
        "npt-steps": 1_000_000,
        "npt-report-interval": 2_500,
        "uvt2-cycles": 125,
        "uvt2-attempts": 800,
        "uvt2-md-steps": 2_000,
        "uvt2-report-interval": 500,
    },
    "smoke": {
        "initial-attempts": 100,
        "uvt1-cycles": 2,
        "uvt1-attempts": 100,
        "uvt1-md-steps": 5,
        "uvt1-report-interval": 5,
        "npt-steps": 500,
        "npt-report-interval": 100,
        "uvt2-cycles": 2,
        "uvt2-attempts": 100,
        "uvt2-md-steps": 100,
        "uvt2-report-interval": 100,
    },
}
PRODUCTION_PROFILE_ARGS = {
    "full": {"cycles": 2_500, "md-steps": 2_000, "attempts": 200, "report-interval": 500},
    "smoke": {"cycles": 3, "md-steps": 100, "attempts": 100, "report-interval": 100},
}

ACTIVE_STATE_PATH: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validation_scope(profile: str, through: str, cluster_stride: int) -> str:
    if profile == "smoke":
        return "smoke_plumbing_only"
    if through == "postprocessing" and cluster_stride > 1:
        return "full_simulation_approximate_postprocessing"
    return "full_ludovic_schedule"


def archive_previous_summaries(run_dir: Path) -> None:
    """Move aggregate reports aside before publishing a new invocation state."""
    existing = [
        run_dir / name
        for name in ("pipeline_state.json", "pipeline_timing.json", "pipeline_audit.json")
        if (run_dir / name).exists()
    ]
    if not existing:
        return
    archive = run_dir / "pipeline_history"
    archive.mkdir(parents=True, exist_ok=True)
    token = f"{time.time_ns()}"
    for path in existing:
        path.replace(archive / f"{token}-{path.name}")


def write_active_state(**updates: Any) -> None:
    if ACTIVE_STATE_PATH is None:
        return
    try:
        payload = read_json(ACTIVE_STATE_PATH)
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
    payload.update(updates)
    write_json_atomic(ACTIVE_STATE_PATH, payload)


def marker_stamps(paths: list[Path]) -> dict[str, tuple[int, int, str]]:
    stamps: dict[str, tuple[int, int, str]] = {}
    for path in paths:
        if path.is_file():
            stat = path.stat()
            stamps[path.name] = (stat.st_mtime_ns, stat.st_size, sha256(path))
    return stamps


def marker_execution(
    before: dict[str, tuple[int, int, str]], paths: list[Path]
) -> dict[str, str]:
    after = marker_stamps(paths)
    return {
        path.stem.removesuffix(".complete"): (
            "reused"
            if path.name in before and before[path.name] == after.get(path.name)
            else "executed"
        )
        for path in paths
    }


def manifest_artifact(inputs: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe extracted-ligand manifest path: {value!r}")
    resolved = (inputs / relative).resolve()
    try:
        resolved.relative_to(inputs.resolve())
    except ValueError as error:
        raise ValueError(f"Extracted ligand escapes inputs directory: {value!r}") from error
    return require_file(resolved)


def options() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument(
        "--receptor",
        type=Path,
    )
    parser.add_argument(
        "--ligand-library",
        type=Path,
    )
    parser.add_argument("--ligand-id", default="x7259a")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--gcmc-platform", default="cuda", choices=("cuda", "opencl"))
    parser.add_argument("--md-platform", default="cuda", choices=("cuda", "opencl", "cpu"))
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--profile", default="full", choices=("full", "smoke"))
    parser.add_argument("--through", default="postprocessing", choices=STAGES)
    parser.add_argument("--force-stage", action="append", choices=STAGES, default=[])
    parser.add_argument("--cluster-stride", type=int, default=1)
    parser.add_argument("--max-distance-memory-gb", type=float, default=32.0)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def audit(
    script_dir: Path,
    run_dir: Path,
    prefix: str,
    ligand_id: str,
    through: str,
    profile: str,
) -> None:
    run(
        [
            sys.executable,
            "-u",
            str(script_dir / "audit_ev71_pipeline.py"),
            "--run-dir", str(run_dir),
            "--prefix", prefix,
            "--ligand-id", ligand_id,
            "--through", through,
            "--profile", profile,
        ]
    )


def main() -> None:
    global ACTIVE_STATE_PATH
    opt = options()
    project = opt.project_dir.resolve()
    scripts = project / "scripts"
    release = project / "openbind_ev71_2a_pyrrolidine_benchmark_release"
    receptor = require_file(
        opt.receptor
        or release / "receptor" / "ev71_2a_x7339a_template_prepared.pdb"
    )
    library = require_file(
        opt.ligand_library
        or release / "ligands" / "pyrrolidine_32_rowan_docked_poses.sdf"
    )
    if opt.seed < 0:
        raise ValueError("Seed must be nonnegative")
    if opt.cluster_stride < 1 or opt.max_distance_memory_gb <= 0:
        raise ValueError("Cluster stride and distance-memory limit must be positive")
    run_dir = opt.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = opt.prefix or f"ev71_2a_{opt.ligand_id}"
    through_index = STAGES.index(opt.through)
    forced = set(opt.force_stage)
    stage_times: dict[str, float] = {}
    stage_execution: dict[str, str] = {}
    pipeline_started = time.time()
    archive_previous_summaries(run_dir)
    ACTIVE_STATE_PATH = run_dir / "pipeline_state.json"
    write_json_atomic(
        ACTIVE_STATE_PATH,
        {
            "status": "running",
            "started_utc": utc_now(),
            "profile": opt.profile,
            "through": opt.through,
            "ligand_id": opt.ligand_id,
            "prefix": prefix,
            "seed": opt.seed,
            "force_stages": sorted(forced),
            "cluster_stride": opt.cluster_stride,
            "validation_scope": validation_scope(
                opt.profile, opt.through, opt.cluster_stride
            ),
            "last_completed_boundary": None,
            "stage_execution": {},
        },
    )

    inputs = run_dir / "inputs"
    # Re-extract deterministically on every invocation. This is cheap for the
    # 32-record library and prevents a stale/swapped cached SDF from running
    # under the requested ligand ID.
    run(
        [
            sys.executable,
            "-u",
            str(scripts / "extract_ligands.py"),
            "--input", str(library),
            "--output-dir", str(inputs),
            "--ligand-id", opt.ligand_id,
            "--overwrite",
        ]
    )
    manifest = read_json(inputs / "manifest.json")
    if manifest.get("source_sha256") != sha256(library):
        raise ValueError("Extracted-ligand manifest does not match the current library")
    records = manifest.get("ligands")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("Expected one ligand in the extraction manifest")
    record = records[0]
    if opt.ligand_id not in record.get("aliases", []):
        raise ValueError("Extracted ligand does not match the requested identifier")
    ligand = manifest_artifact(inputs, record["output"])
    if record.get("output_sha256") != sha256(ligand):
        raise ValueError("Extracted ligand hash differs from its manifest")

    preparation_markers = [run_dir / "preparation" / "preparation.complete.json"]
    before = marker_stamps(preparation_markers)
    started = time.time()
    preparation_command = [
        sys.executable,
        "-u",
        str(scripts / "prepare_ev71_system.py"),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--output-dir", str(run_dir / "preparation"),
        "--prefix", prefix,
    ]
    if "preparation" in forced:
        preparation_command.append("--force")
    run(preparation_command)
    stage_times["preparation"] = time.time() - started
    stage_execution.update(marker_execution(before, preparation_markers))
    audit(scripts, run_dir, prefix, opt.ligand_id, "preparation", opt.profile)
    write_active_state(
        last_completed_boundary="preparation", stage_execution=stage_execution
    )
    if through_index == 0:
        finish(run_dir, opt, prefix, stage_times, stage_execution, pipeline_started)
        return

    equilibration_command = [
        sys.executable,
        "-u",
        str(scripts / "ev71_equilibrate.py"),
        "--prmtop", str(run_dir / "preparation" / f"{prefix}_solvated.prmtop"),
        "--rst7", str(run_dir / "preparation" / f"{prefix}_solvated.inpcrd"),
        "--output-dir", str(run_dir / "equilibration"),
        "--prefix", prefix,
        "--seed", str(opt.seed),
        "--gcmc-platform", opt.gcmc_platform,
        "--md-platform", opt.md_platform,
        "--precision", opt.precision,
    ]
    for stage in ("uvt1", "npt", "uvt2"):
        if "equilibration" in forced:
            equilibration_command.extend(["--force-stage", stage])
    for name, value in EQUILIBRATION_PROFILE_ARGS[opt.profile].items():
        equilibration_command.extend([f"--{name}", str(value)])
    equilibration_markers = [
        run_dir / "equilibration" / f"{stage}.complete.json"
        for stage in ("uvt1", "npt", "uvt2")
    ]
    before = marker_stamps(equilibration_markers)
    started = time.time()
    run(equilibration_command)
    stage_times["equilibration"] = time.time() - started
    stage_execution.update(marker_execution(before, equilibration_markers))
    audit(scripts, run_dir, prefix, opt.ligand_id, "equilibration", opt.profile)
    write_active_state(
        last_completed_boundary="equilibration", stage_execution=stage_execution
    )
    if through_index == 1:
        finish(run_dir, opt, prefix, stage_times, stage_execution, pipeline_started)
        return

    production_command = [
        sys.executable,
        "-u",
        str(scripts / "ev71_production.py"),
        "--prmtop", str(run_dir / "equilibration" / f"{prefix}_uvt2.prmtop"),
        "--rst7", str(run_dir / "equilibration" / f"{prefix}_uvt2.rst7"),
        "--output-dir", str(run_dir / "production"),
        "--prefix", prefix,
        "--seed", str(opt.seed + 3),
        "--gcmc-platform", opt.gcmc_platform,
        "--md-platform", opt.md_platform,
        "--precision", opt.precision,
    ]
    if "production" in forced:
        production_command.append("--force")
    for name, value in PRODUCTION_PROFILE_ARGS[opt.profile].items():
        production_command.extend([f"--{name}", str(value)])
    production_markers = [run_dir / "production" / "production.complete.json"]
    before = marker_stamps(production_markers)
    started = time.time()
    run(production_command)
    stage_times["production"] = time.time() - started
    stage_execution.update(marker_execution(before, production_markers))
    audit(scripts, run_dir, prefix, opt.ligand_id, "production", opt.profile)
    write_active_state(
        last_completed_boundary="production", stage_execution=stage_execution
    )
    if through_index == 2:
        finish(run_dir, opt, prefix, stage_times, stage_execution, pipeline_started)
        return

    postprocess_command = [
        sys.executable,
        "-u",
        str(scripts / "ev71_postprocess.py"),
        "--topology", str(run_dir / "production" / f"{prefix}-loch-ghosts.pdb"),
        "--trajectory", str(run_dir / "production" / f"{prefix}-raw.dcd"),
        "--ghost-file", str(run_dir / "production" / f"{prefix}-gcmc-ghosts.txt"),
        "--output-dir", str(run_dir / "postprocessing"),
        "--prefix", prefix,
        "--cluster-stride", str(opt.cluster_stride),
        "--max-distance-memory-gb", str(opt.max_distance_memory_gb),
    ]
    if "postprocessing" in forced:
        postprocess_command.append("--force")
    postprocess_markers = [
        run_dir / "postprocessing" / "postprocessing.complete.json"
    ]
    before = marker_stamps(postprocess_markers)
    started = time.time()
    run(postprocess_command)
    stage_times["postprocessing"] = time.time() - started
    stage_execution.update(marker_execution(before, postprocess_markers))
    audit(scripts, run_dir, prefix, opt.ligand_id, "postprocessing", opt.profile)
    write_active_state(
        last_completed_boundary="postprocessing", stage_execution=stage_execution
    )
    finish(run_dir, opt, prefix, stage_times, stage_execution, pipeline_started)


def finish(
    run_dir: Path,
    opt: argparse.Namespace,
    prefix: str,
    stage_times: dict[str, float],
    stage_execution: dict[str, str],
    started: float,
) -> None:
    finished_utc = utc_now()
    payload = {
        "status": "completed",
        "profile": opt.profile,
        "through": opt.through,
        "ligand_id": opt.ligand_id,
        "prefix": prefix,
        "seed": opt.seed,
        "gcmc_platform": opt.gcmc_platform,
        "md_platform": opt.md_platform,
        "validation_scope": validation_scope(
            opt.profile, opt.through, opt.cluster_stride
        ),
        "postprocessing_scope": (
            "not_run"
            if opt.through != "postprocessing"
            else (
                "exact_ludovic_stride_1"
                if opt.cluster_stride == 1
                else f"approximate_subsampled_stride_{opt.cluster_stride}"
            )
        ),
        "cluster_stride": opt.cluster_stride,
        "max_distance_memory_gb": opt.max_distance_memory_gb,
        "implementation": implementation_signature(
            sources={
                "run_ev71_pipeline.py": Path(__file__),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            }
        ),
        "stage_execution": stage_execution,
        "stage_wall_seconds": stage_times,
        "total_wall_seconds": time.time() - started,
        "finished_utc": finished_utc,
    }
    write_json_atomic(run_dir / "pipeline_timing.json", payload)
    write_active_state(
        status="completed",
        finished_utc=finished_utc,
        last_completed_boundary=opt.through,
        stage_execution=stage_execution,
        timing_sha256=sha256(run_dir / "pipeline_timing.json"),
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"PIPELINE_COMPLETE={run_dir} THROUGH={opt.through}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_active_state(
            status="failed",
            finished_utc=utc_now(),
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
