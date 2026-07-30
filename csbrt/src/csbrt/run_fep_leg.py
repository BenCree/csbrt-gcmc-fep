#!/usr/bin/env python3
"""Run or resume one SOMD2 FEP leg and publish a verified completion marker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import yaml

from pipeline_utils import (
    complete_checkpoint,
    implementation_signature,
    require_file,
    sha256,
    validate_recorded_outputs,
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--leg", choices=("bound", "free"), required=True)
    parser.add_argument("--platform", default="cuda", choices=("cuda", "opencl"))
    parser.add_argument("--max-gpus", type=int, default=1)
    parser.add_argument("--gcmc-bound", action="store_true")
    parser.add_argument("--gcmc-selection", default="resname LIG")
    parser.add_argument("--gcmc-radius", default="10 A")
    parser.add_argument("--gcmc-num-waters", type=int, default=45)
    parser.add_argument("--gcmc-standard-volume", default="30.345 A^3")
    parser.add_argument("--gcmc-excess-chemical-potential", default="-6.09 kcal/mol")
    parser.add_argument("--gcmc-bulk-sampling-probability", type=float, default=0.0)
    return parser.parse_args()


def somd2_failure_hints(output: Path) -> str:
    """Extract the real SOMD2 error lines from the leg logs, so a zero-Parquet
    failure reports its cause (PTX/CUDA mismatch, minimisation blow-up, missing
    nvcc, OOM, ...) instead of an opaque 'found 0'."""
    import re

    patterns = re.compile(
        r"CUDA_ERROR|UNSUPPORTED_PTX|PTX version|minimis|equilibrat.*fail|"
        r"NaN|not finite|unstable|Error running|Exception|Traceback|"
        r"out of memory|CUDA out|requires nvcc|no kernel image|RuntimeError",
        re.IGNORECASE,
    )
    seen: list[str] = []
    for name in ("log.txt", "runner.stdout.log"):
        path = output / name
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped and patterns.search(stripped) and stripped not in seen:
                seen.append(stripped)
    return "\n".join(seen[-15:])


def main() -> None:
    opt = options()
    stream = require_file(opt.stream)
    config = require_file(opt.config)
    config_payload = yaml.safe_load(config.read_text())
    if not isinstance(config_payload, dict):
        raise ValueError("SOMD2 configuration must be a YAML mapping")
    lambda_values = config_payload.get("lambda_values")
    if lambda_values is not None:
        if not isinstance(lambda_values, list) or len(lambda_values) < 2:
            raise ValueError("lambda_values must contain at least two windows")
        expected_windows = len(lambda_values)
    else:
        expected_windows = int(config_payload.get("num_lambda", 11))
        if expected_windows < 2:
            raise ValueError("num_lambda must be at least two")
    if opt.max_gpus < 1:
        raise ValueError("--max-gpus must be positive")
    if opt.gcmc_bound and opt.leg != "bound":
        raise ValueError("GCMC is only enabled for the bound leg")
    if opt.gcmc_num_waters < 1 or not 0 <= opt.gcmc_bulk_sampling_probability <= 1:
        raise ValueError("Invalid GCMC water count or bulk-sampling probability")
    somd2 = shutil.which("somd2")
    if somd2 is None:
        raise FileNotFoundError(
            "somd2 is not active; create/activate the separate Mamba FEP environment"
        )

    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "fep_leg.complete.json"
    signature = {
        "leg": opt.leg,
        "stream_sha256": sha256(stream),
        "config_sha256": sha256(config),
        "platform": opt.platform,
        "max_gpus": opt.max_gpus,
        "gcmc": opt.gcmc_bound,
        "gcmc_selection": opt.gcmc_selection if opt.gcmc_bound else None,
        "gcmc_radius": opt.gcmc_radius if opt.gcmc_bound else None,
        "gcmc_num_waters": opt.gcmc_num_waters if opt.gcmc_bound else None,
        "gcmc_standard_volume": opt.gcmc_standard_volume if opt.gcmc_bound else None,
        "gcmc_excess_chemical_potential": (
            opt.gcmc_excess_chemical_potential if opt.gcmc_bound else None
        ),
        "gcmc_bulk_sampling_probability": (
            opt.gcmc_bulk_sampling_probability if opt.gcmc_bound else None
        ),
        "implementation": implementation_signature(
            sources={
                "run_fep_leg.py": Path(__file__),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("somd2", "sire"),
            modules=("somd2", "sire"),
        ),
    }
    if marker.is_file():
        payload = json.loads(marker.read_text())
        if (
            payload.get("status") == "completed"
            and payload.get("signature") == signature
        ):
            validate_recorded_outputs(marker)
            print(f"FEP leg checkpoint is valid: {marker}", flush=True)
            return
        marker.unlink()

    command = [
        somd2,
        str(stream),
        "--config", str(config),
        "--output-directory", str(output),
        "--platform", opt.platform,
        "--max-gpus", str(opt.max_gpus),
    ]
    # SOMD2 checkpoint extension is version-dependent (.s3 on older releases,
    # .npz on 2026.1). Matching only one silently falls through to --overwrite,
    # which discards every completed lambda window and re-runs the whole leg.
    checkpoints = [
        path
        for pattern in ("checkpoint_*.s3", "checkpoint_*.npz")
        for path in output.glob(pattern)
    ]
    if checkpoints:
        command.append("--restart")
    elif list(output.glob("config*.yaml")):
        # A prior launch can fail before the first checkpoint (for example at
        # CUDA kernel compilation). SOMD2 otherwise refuses its own stale
        # config/topology files, so rebuild only this uncheckpointed attempt.
        command.append("--overwrite")
    if opt.gcmc_bound:
        command.extend(
            [
                "--gcmc",
                "--gcmc-selection", opt.gcmc_selection,
                "--gcmc-radius", opt.gcmc_radius,
                "--gcmc-num-waters", str(opt.gcmc_num_waters),
                "--gcmc-standard-volume", opt.gcmc_standard_volume,
                "--gcmc-excess-chemical-potential", opt.gcmc_excess_chemical_potential,
                "--gcmc-bulk-sampling-probability", str(opt.gcmc_bulk_sampling_probability),
            ]
        )
    log = output / "runner.stdout.log"
    with log.open("a") as handle:
        handle.write(f"[pipeline] command: {shlex.join(command)}\n")
        handle.flush()
        environment = os.environ.copy()
        if opt.platform == "cuda" and not environment.get("CUDA_VISIBLE_DEVICES"):
            # Slurm normally sets this. Make a one-GPU local invocation behave
            # the same way without overriding scheduler-assigned devices.
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
            env=environment,
        )

    energies = sorted(output.glob("energy_traj_*.parquet"))
    if len(energies) != expected_windows:
        hints = somd2_failure_hints(output)
        detail = (
            f"\nSOMD2 reported (from {output}/log.txt):\n{hints}"
            if hints
            else f"\nNo error lines matched; inspect {output}/log.txt and "
            f"{output}/runner.stdout.log directly."
        )
        raise RuntimeError(
            f"Expected {expected_windows} lambda energy trajectories, found "
            f"{len(energies)} — SOMD2 wrote no/partial energies for this leg.{detail}"
        )
    outputs = [log, *energies]
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=outputs,
        details={
            "leg": opt.leg,
            "stream": str(stream),
            "stream_sha256": sha256(stream),
            "config": str(config),
            "config_sha256": sha256(config),
            "lambda_energy_files": len(energies),
            "expected_lambda_windows": expected_windows,
        },
    )
    print(f"FEP_LEG_COMPLETE={marker}", flush=True)


if __name__ == "__main__":
    main()
