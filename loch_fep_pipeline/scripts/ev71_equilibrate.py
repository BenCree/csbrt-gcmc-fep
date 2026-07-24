#!/usr/bin/env python3
"""Run resumable Ludovic-order UVT1, NPT, and UVT2 equilibration with Loch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import openmm
from openmm import unit
import sire as sr

from ev71_loch_common import (
    BAROSTAT_FREQUENCY,
    CsvStateWriter,
    DEFAULT_BATCH_SIZE,
    MINIMIZATION_TOLERANCE_KJ_MOL_NM,
    NPT_REPORT_INTERVAL,
    NPT_STEPS,
    PRESSURE,
    SEED,
    UVT1_ATTEMPTS,
    UVT1_CYCLES,
    UVT1_INITIAL_ATTEMPTS,
    UVT1_MD_STEPS,
    UVT1_REPORT_INTERVAL,
    UVT2_ATTEMPTS,
    UVT2_CYCLES,
    UVT2_MD_STEPS,
    UVT2_REPORT_INTERVAL,
    ca_restraints,
    finalise_sampler_system,
    image_context,
    make_dynamics,
    make_sampler,
    physical_protocol_signature,
    print_sampler,
    randomise_velocities,
    run_with_csv_reports,
    save_physical_system,
    update_system_from_context,
    validate_gcmc_handoff,
    validate_physical_water_topology,
    validate_single_ligand,
)
from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    finite_csv,
    ghost_history,
    implementation_signature,
    invalidate_checkpoint,
    sha256,
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--rst7", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--gcmc-platform", default="cuda", choices=("cuda", "opencl"))
    parser.add_argument("--md-platform", default="cuda", choices=("cuda", "opencl", "cpu"))
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--initial-attempts", type=int, default=UVT1_INITIAL_ATTEMPTS)
    parser.add_argument("--uvt1-cycles", type=int, default=UVT1_CYCLES)
    parser.add_argument("--uvt1-attempts", type=int, default=UVT1_ATTEMPTS)
    parser.add_argument("--uvt1-md-steps", type=int, default=UVT1_MD_STEPS)
    parser.add_argument("--uvt1-report-interval", type=int, default=UVT1_REPORT_INTERVAL)
    parser.add_argument("--npt-steps", type=int, default=NPT_STEPS)
    parser.add_argument("--npt-report-interval", type=int, default=NPT_REPORT_INTERVAL)
    parser.add_argument("--uvt2-cycles", type=int, default=UVT2_CYCLES)
    parser.add_argument("--uvt2-attempts", type=int, default=UVT2_ATTEMPTS)
    parser.add_argument("--uvt2-md-steps", type=int, default=UVT2_MD_STEPS)
    parser.add_argument("--uvt2-report-interval", type=int, default=UVT2_REPORT_INTERVAL)
    parser.add_argument(
        "--force-stage",
        action="append",
        choices=("uvt1", "npt", "uvt2"),
        default=[],
    )
    return parser.parse_args()


def load_physical(prmtop: Path, rst7: Path, label: str, ligand_resname: str):
    system = sr.load(str(prmtop), str(rst7))
    water = validate_physical_water_topology(system, label=label)
    ligand = validate_single_ligand(system, ligand_resname)
    return system, water, ligand


def stage_signature(
    name: str,
    input_top: Path,
    input_rst: Path,
    values: dict[str, object],
    *,
    gcmc_platform: str | None = None,
) -> dict[str, object]:
    modules = ["sire", "openmm"]
    if gcmc_platform is not None:
        modules.extend(("loch._sampler", f"loch._platforms._{gcmc_platform.lower()}"))
    return {
        "stage": name,
        "input_prmtop_sha256": sha256(input_top),
        "input_rst7_sha256": sha256(input_rst),
        "physical_protocol": physical_protocol_signature(),
        "implementation": implementation_signature(
            sources={
                "ev71_equilibrate.py": Path(__file__),
                "ev71_loch_common.py": Path(__file__).with_name("ev71_loch_common.py"),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("loch", "sire", "OpenMM"),
            modules=tuple(modules),
        ),
        **values,
    }


def main() -> None:
    total_started = time.time()
    opt = options()
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    forced = set(opt.force_stage)
    numeric = {
        "batch_size": opt.batch_size,
        "initial_attempts": opt.initial_attempts,
        "uvt1_cycles": opt.uvt1_cycles,
        "uvt1_attempts": opt.uvt1_attempts,
        "uvt1_md_steps": opt.uvt1_md_steps,
        "uvt1_report_interval": opt.uvt1_report_interval,
        "npt_steps": opt.npt_steps,
        "npt_report_interval": opt.npt_report_interval,
        "uvt2_cycles": opt.uvt2_cycles,
        "uvt2_attempts": opt.uvt2_attempts,
        "uvt2_md_steps": opt.uvt2_md_steps,
        "uvt2_report_interval": opt.uvt2_report_interval,
    }
    if any(value < 0 for key, value in numeric.items() if "interval" not in key):
        raise ValueError("Counts and step totals cannot be negative")
    if any(value < 1 for key, value in numeric.items() if "interval" in key):
        raise ValueError("Report intervals must be positive")
    if opt.uvt1_cycles < 1 or opt.uvt2_cycles < 1 or opt.npt_steps < 1:
        raise ValueError("UVT1, NPT, and UVT2 must each perform work")

    input_top = opt.prmtop.resolve()
    input_rst = opt.rst7.resolve()
    uvt1_top = output / f"{opt.prefix}_uvt1.prmtop"
    uvt1_rst = output / f"{opt.prefix}_uvt1.rst7"
    uvt1_ghost = output / f"{opt.prefix}_equilibration_uvt1_ghosts.txt"
    uvt1_csv_path = output / f"{opt.prefix}_data_uvt1.csv"
    uvt1_outputs = [
        uvt1_top,
        uvt1_rst,
        output / f"{opt.prefix}_uvt1.pdb",
        uvt1_ghost,
        uvt1_csv_path,
    ]
    uvt1_marker = output / "uvt1.complete.json"
    uvt1_signature = stage_signature(
        "uvt1",
        input_top,
        input_rst,
        {
            "seed": opt.seed,
            "batch_size": opt.batch_size,
            "initial_attempts": opt.initial_attempts,
            "cycles": opt.uvt1_cycles,
            "attempts_per_cycle": opt.uvt1_attempts,
            "md_steps_per_cycle": opt.uvt1_md_steps,
            "report_interval": opt.uvt1_report_interval,
            "ligand_resname": opt.ligand_resname,
            "gcmc_platform": opt.gcmc_platform,
            "md_platform": opt.md_platform,
            "precision": opt.precision,
        },
        gcmc_platform=opt.gcmc_platform,
    )
    if "uvt1" not in forced and checkpoint_matches(
        uvt1_marker, signature=uvt1_signature, outputs=uvt1_outputs
    ):
        no_ghosts1, _, _ = load_physical(
            uvt1_top, uvt1_rst, "UVT1 checkpoint", opt.ligand_resname
        )
        uvt1_ghost_audit = ghost_history(uvt1_ghost)
        if int(uvt1_ghost_audit["lines"]) != opt.uvt1_cycles:
            raise ValueError("UVT1 checkpoint ghost history does not match cycles")
        finite_csv(
            uvt1_csv_path,
            total_steps=opt.uvt1_cycles * opt.uvt1_md_steps,
            report_interval=opt.uvt1_report_interval,
        )
        print(f"UVT1 checkpoint is valid: {uvt1_marker}", flush=True)
    else:
        invalidate_checkpoint(uvt1_marker)
        started = time.time()
        original, input_water_audit, _ = load_physical(
            input_top, input_rst, "Loch equilibration input", opt.ligand_resname
        )
        input_waters = int(input_water_audit["water_molecules"])
        sampler = make_sampler(
            original,
            attempts=opt.uvt1_attempts,
            batch_size=opt.batch_size,
            seed=opt.seed,
            log_file=output / f"{opt.prefix}_equilibration_uvt1.log",
            ghost_file=uvt1_ghost,
            ligand_resname=opt.ligand_resname,
            platform=opt.gcmc_platform,
        )
        system = sampler.system()
        dynamics = make_dynamics(
            system,
            restraints=ca_restraints(system),
            platform=opt.md_platform,
            precision=opt.precision,
        )
        sampler.bind_dynamics(dynamics)
        context = dynamics.context()
        sampler.delete_waters(context)
        openmm.LocalEnergyMinimizer.minimize(
            context,
            MINIMIZATION_TOLERANCE_KJ_MOL_NM * unit.kilojoule_per_mole / unit.nanometer,
            0,
        )
        randomise_velocities(context, opt.seed)
        if opt.initial_attempts:
            cycle_attempts = sampler._num_attempts
            sampler._num_attempts = opt.initial_attempts
            sampler.move(context)
            sampler._num_attempts = cycle_attempts
        csv = CsvStateWriter(uvt1_csv_path, context)
        completed = 0
        for cycle in range(opt.uvt1_cycles):
            sampler.move(context)
            sampler.write_ghost_residues()
            completed = run_with_csv_reports(
                dynamics,
                context,
                opt.uvt1_md_steps,
                completed,
                opt.uvt1_report_interval,
                csv,
            )
            print_sampler(f"UVT1 {cycle + 1}/{opt.uvt1_cycles}", sampler)
        csv.close()
        image_context(context)
        no_ghosts1 = finalise_sampler_system(sampler, context)
        handoff = validate_gcmc_handoff(
            no_ghosts1,
            sampler,
            input_water_count=input_waters,
            label="UVT1 finalized handoff",
        )
        save_physical_system(
            no_ghosts1,
            output / f"{opt.prefix}_uvt1",
            expected_water_count=handoff["expected_physical_waters"],
        )
        uvt1_ghost_audit = ghost_history(uvt1_ghost)
        if int(uvt1_ghost_audit["lines"]) != opt.uvt1_cycles:
            raise RuntimeError("UVT1 ghost-history count differs from completed cycles")
        uvt1_csv_audit = finite_csv(
            uvt1_csv_path,
            total_steps=opt.uvt1_cycles * opt.uvt1_md_steps,
            report_interval=opt.uvt1_report_interval,
        )
        complete_checkpoint(
            uvt1_marker,
            signature=uvt1_signature,
            outputs=uvt1_outputs,
            details={
                "handoff": handoff,
                "ghost_history": uvt1_ghost_audit,
                "csv": uvt1_csv_audit,
                "wall_seconds": time.time() - started,
            },
        )
        print(f"UVT1 elapsed: {time.time() - started:.1f} s", flush=True)

    npt_top = output / f"{opt.prefix}_npt.prmtop"
    npt_rst = output / f"{opt.prefix}_npt.rst7"
    npt_csv_path = output / f"{opt.prefix}_data_npt.csv"
    npt_outputs = [npt_top, npt_rst, output / f"{opt.prefix}_npt.pdb", npt_csv_path]
    npt_marker = output / "npt.complete.json"
    npt_signature = stage_signature(
        "npt",
        uvt1_top,
        uvt1_rst,
        {
            "seed": opt.seed + 1,
            "steps": opt.npt_steps,
            "report_interval": opt.npt_report_interval,
            "md_platform": opt.md_platform,
            "precision": opt.precision,
        },
    )
    if "npt" not in forced and checkpoint_matches(
        npt_marker, signature=npt_signature, outputs=npt_outputs
    ):
        npt_system, _, _ = load_physical(
            npt_top, npt_rst, "NPT checkpoint", opt.ligand_resname
        )
        finite_csv(
            npt_csv_path,
            total_steps=opt.npt_steps,
            report_interval=opt.npt_report_interval,
        )
        print(f"NPT checkpoint is valid: {npt_marker}", flush=True)
    else:
        invalidate_checkpoint(npt_marker)
        started = time.time()
        no_ghosts1, uvt1_water_audit, _ = load_physical(
            uvt1_top, uvt1_rst, "NPT input", opt.ligand_resname
        )
        expected_waters = int(uvt1_water_audit["water_molecules"])
        npt = make_dynamics(
            no_ghosts1,
            pressure=PRESSURE,
            barostat_frequency=BAROSTAT_FREQUENCY,
            platform=opt.md_platform,
            precision=opt.precision,
        )
        context = npt.context()
        randomise_velocities(context, opt.seed + 1)
        csv = CsvStateWriter(npt_csv_path, context)
        completed = 0
        while completed < opt.npt_steps:
            chunk = min(opt.npt_report_interval, opt.npt_steps - completed)
            completed = run_with_csv_reports(
                npt,
                context,
                chunk,
                completed,
                opt.npt_report_interval,
                csv,
            )
            print(f"NPT {completed}/{opt.npt_steps}", flush=True)
        csv.close()
        image_context(context)
        npt_system = npt.commit(return_as_system=True)
        npt_system = update_system_from_context(npt_system, context)
        save_physical_system(
            npt_system,
            output / f"{opt.prefix}_npt",
            expected_water_count=expected_waters,
        )
        npt_csv_audit = finite_csv(
            npt_csv_path,
            total_steps=opt.npt_steps,
            report_interval=opt.npt_report_interval,
        )
        complete_checkpoint(
            npt_marker,
            signature=npt_signature,
            outputs=npt_outputs,
            details={
                "physical_waters": expected_waters,
                "csv": npt_csv_audit,
                "wall_seconds": time.time() - started,
            },
        )
        print(f"NPT elapsed: {time.time() - started:.1f} s", flush=True)

    uvt2_top = output / f"{opt.prefix}_uvt2.prmtop"
    uvt2_rst = output / f"{opt.prefix}_uvt2.rst7"
    uvt2_ghost = output / f"{opt.prefix}_equilibration_uvt2_ghosts.txt"
    uvt2_csv_path = output / f"{opt.prefix}_data_uvt2.csv"
    uvt2_outputs = [
        uvt2_top,
        uvt2_rst,
        output / f"{opt.prefix}_uvt2.pdb",
        uvt2_ghost,
        uvt2_csv_path,
    ]
    uvt2_marker = output / "uvt2.complete.json"
    uvt2_signature = stage_signature(
        "uvt2",
        npt_top,
        npt_rst,
        {
            "seed": opt.seed + 2,
            "batch_size": opt.batch_size,
            "cycles": opt.uvt2_cycles,
            "attempts_per_cycle": opt.uvt2_attempts,
            "md_steps_per_cycle": opt.uvt2_md_steps,
            "report_interval": opt.uvt2_report_interval,
            "ligand_resname": opt.ligand_resname,
            "gcmc_platform": opt.gcmc_platform,
            "md_platform": opt.md_platform,
            "precision": opt.precision,
        },
        gcmc_platform=opt.gcmc_platform,
    )
    if "uvt2" not in forced and checkpoint_matches(
        uvt2_marker, signature=uvt2_signature, outputs=uvt2_outputs
    ):
        load_physical(uvt2_top, uvt2_rst, "UVT2 checkpoint", opt.ligand_resname)
        uvt2_ghost_audit = ghost_history(uvt2_ghost)
        if int(uvt2_ghost_audit["lines"]) != opt.uvt2_cycles:
            raise ValueError("UVT2 checkpoint ghost history does not match cycles")
        finite_csv(
            uvt2_csv_path,
            total_steps=opt.uvt2_cycles * opt.uvt2_md_steps,
            report_interval=opt.uvt2_report_interval,
        )
        print(f"UVT2 checkpoint is valid: {uvt2_marker}", flush=True)
    else:
        invalidate_checkpoint(uvt2_marker)
        started = time.time()
        npt_system, npt_water_audit, _ = load_physical(
            npt_top, npt_rst, "UVT2 input", opt.ligand_resname
        )
        input_waters = int(npt_water_audit["water_molecules"])
        sampler = make_sampler(
            npt_system,
            attempts=opt.uvt2_attempts,
            batch_size=opt.batch_size,
            seed=opt.seed + 2,
            log_file=output / f"{opt.prefix}_equilibration_uvt2.log",
            ghost_file=uvt2_ghost,
            ligand_resname=opt.ligand_resname,
            platform=opt.gcmc_platform,
        )
        system = sampler.system()
        dynamics = make_dynamics(
            system,
            restraints=ca_restraints(system),
            platform=opt.md_platform,
            precision=opt.precision,
        )
        sampler.bind_dynamics(dynamics)
        context = dynamics.context()
        randomise_velocities(context, opt.seed + 2)
        csv = CsvStateWriter(uvt2_csv_path, context)
        completed = 0
        for cycle in range(opt.uvt2_cycles):
            sampler.move(context)
            sampler.write_ghost_residues()
            completed = run_with_csv_reports(
                dynamics,
                context,
                opt.uvt2_md_steps,
                completed,
                opt.uvt2_report_interval,
                csv,
            )
            print_sampler(f"UVT2 {cycle + 1}/{opt.uvt2_cycles}", sampler)
        csv.close()
        image_context(context)
        equilibrated = finalise_sampler_system(sampler, context)
        handoff = validate_gcmc_handoff(
            equilibrated,
            sampler,
            input_water_count=input_waters,
            label="UVT2 finalized handoff",
        )
        paths = save_physical_system(
            equilibrated,
            output / f"{opt.prefix}_uvt2",
            expected_water_count=handoff["expected_physical_waters"],
        )
        uvt2_ghost_audit = ghost_history(uvt2_ghost)
        if int(uvt2_ghost_audit["lines"]) != opt.uvt2_cycles:
            raise RuntimeError("UVT2 ghost-history count differs from completed cycles")
        uvt2_csv_audit = finite_csv(
            uvt2_csv_path,
            total_steps=opt.uvt2_cycles * opt.uvt2_md_steps,
            report_interval=opt.uvt2_report_interval,
        )
        complete_checkpoint(
            uvt2_marker,
            signature=uvt2_signature,
            outputs=uvt2_outputs,
            details={
                "handoff": handoff,
                "outputs": {name: Path(path).name for name, path in paths.items()},
                "ghost_history": uvt2_ghost_audit,
                "csv": uvt2_csv_audit,
                "wall_seconds": time.time() - started,
            },
        )
        print(f"UVT2 elapsed: {time.time() - started:.1f} s", flush=True)

    print(f"Production topology: {uvt2_top}", flush=True)
    print(f"Production coordinates: {uvt2_rst}", flush=True)
    print(f"EQUILIBRATION_TOTAL_WALL_SECONDS={time.time() - total_started:.3f}", flush=True)


if __name__ == "__main__":
    main()
