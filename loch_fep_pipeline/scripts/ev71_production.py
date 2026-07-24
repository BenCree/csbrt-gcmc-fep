#!/usr/bin/env python3
"""Run resumable Ludovic-order Loch production for one prepared EV71 complex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mdtraj as md
from openmm import app, unit
import sire as sr

from ev71_loch_common import (
    CsvStateWriter,
    DEFAULT_BATCH_SIZE,
    NUM_GHOSTS,
    PRODUCTION_ATTEMPTS,
    PRODUCTION_CYCLES,
    PRODUCTION_MD_STEPS,
    PRODUCTION_REPORT_INTERVAL,
    SEED,
    TIMESTEP_FS,
    ca_restraints,
    finalise_sampler_system,
    make_dynamics,
    make_sampler,
    physical_water_audit,
    physical_protocol_signature,
    print_sampler,
    randomise_velocities,
    run_with_csv_reports,
    save_physical_system,
    save_system,
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
    parser.add_argument("--seed", type=int, default=SEED + 3)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cycles", type=int, default=PRODUCTION_CYCLES)
    parser.add_argument("--md-steps", type=int, default=PRODUCTION_MD_STEPS)
    parser.add_argument("--attempts", type=int, default=PRODUCTION_ATTEMPTS)
    parser.add_argument("--report-interval", type=int, default=PRODUCTION_REPORT_INTERVAL)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dcd_frames(path: Path) -> int:
    with md.open(str(path)) as handle:
        return len(handle)


def main() -> None:
    total_started = time.time()
    opt = options()
    if min(opt.batch_size, opt.cycles, opt.md_steps, opt.attempts, opt.report_interval) < 1:
        raise ValueError("Production counts and intervals must be positive")
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    input_top = opt.prmtop.resolve()
    input_rst = opt.rst7.resolve()
    raw_prefix = output / f"{opt.prefix}-loch-ghosts"
    raw_pdb = raw_prefix.with_suffix(".pdb")
    raw_dcd = output / f"{opt.prefix}-raw.dcd"
    ghost_file = output / f"{opt.prefix}-gcmc-ghosts.txt"
    csv_file = output / f"{opt.prefix}_data_prod.csv"
    final_prefix = output / f"{opt.prefix}-production-final"
    final_top = final_prefix.with_suffix(".prmtop")
    final_rst = final_prefix.with_suffix(".rst7")
    outputs = [
        raw_prefix.with_suffix(".prmtop"),
        raw_prefix.with_suffix(".rst7"),
        raw_pdb,
        raw_dcd,
        ghost_file,
        csv_file,
        final_top,
        final_rst,
        final_prefix.with_suffix(".pdb"),
    ]
    signature = {
        "input_prmtop_sha256": sha256(input_top),
        "input_rst7_sha256": sha256(input_rst),
        "prefix": opt.prefix,
        "ligand_resname": opt.ligand_resname,
        "gcmc_platform": opt.gcmc_platform,
        "md_platform": opt.md_platform,
        "precision": opt.precision,
        "seed": opt.seed,
        "batch_size": opt.batch_size,
        "cycles": opt.cycles,
        "md_steps_per_cycle": opt.md_steps,
        "attempts_per_cycle": opt.attempts,
        "report_interval": opt.report_interval,
        "physical_protocol": physical_protocol_signature(),
        "implementation": implementation_signature(
            sources={
                "ev71_production.py": Path(__file__),
                "ev71_loch_common.py": Path(__file__).with_name("ev71_loch_common.py"),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("loch", "sire", "OpenMM", "mdtraj"),
            modules=(
                "sire",
                "openmm",
                "loch._sampler",
                f"loch._platforms._{opt.gcmc_platform.lower()}",
            ),
        ),
    }
    marker = output / "production.complete.json"
    if not opt.force and checkpoint_matches(marker, signature=signature, outputs=outputs):
        equilibrated = sr.load(str(input_top), str(input_rst))
        input_audit = validate_physical_water_topology(
            equilibrated, label="Production checkpoint input"
        )
        input_waters = int(input_audit["water_molecules"])
        raw = sr.load(str(raw_prefix.with_suffix(".prmtop")), str(raw_prefix.with_suffix(".rst7")))
        raw_audit = physical_water_audit(raw)
        if int(raw_audit["water_molecules"]) != input_waters + NUM_GHOSTS:
            raise ValueError("Production checkpoint raw topology is not input + 45 waters")
        if int(raw_audit["zero_interaction_water_count"]) != NUM_GHOSTS:
            raise ValueError("Production checkpoint raw topology does not have 45 ghosts")
        final = sr.load(str(final_top), str(final_rst))
        final_audit = validate_physical_water_topology(
            final, label="Production checkpoint final handoff"
        )
        validate_single_ligand(final, opt.ligand_resname)
        ghosts = ghost_history(ghost_file)
        frames = dcd_frames(raw_dcd)
        if ghosts["lines"] != opt.cycles or frames != opt.cycles:
            raise ValueError("Production checkpoint frame/ghost counts do not match cycles")
        expected_final = input_waters + NUM_GHOSTS - int(ghosts["final_state_zero"])
        if int(final_audit["water_molecules"]) != expected_final:
            raise ValueError("Production checkpoint physical-water arithmetic failed")
        finite_csv(
            csv_file,
            total_steps=opt.cycles * opt.md_steps,
            report_interval=opt.report_interval,
        )
        print(f"Production checkpoint is valid: {marker}", flush=True)
        return
    invalidate_checkpoint(marker)

    equilibrated = sr.load(str(input_top), str(input_rst))
    input_audit = validate_physical_water_topology(
        equilibrated, label="Loch production input"
    )
    validate_single_ligand(equilibrated, opt.ligand_resname)
    input_waters = int(input_audit["water_molecules"])
    sampler = make_sampler(
        equilibrated,
        attempts=opt.attempts,
        batch_size=opt.batch_size,
        seed=opt.seed,
        log_file=output / f"{opt.prefix}-gcmc.log",
        ghost_file=ghost_file,
        ligand_resname=opt.ligand_resname,
        platform=opt.gcmc_platform,
    )
    gcmc_system = sampler.system()
    raw_audit = physical_water_audit(gcmc_system)
    if int(raw_audit["water_molecules"]) != input_waters + NUM_GHOSTS:
        raise ValueError("Production sampler did not append exactly 45 waters")
    if int(raw_audit["zero_interaction_water_count"]) != NUM_GHOSTS:
        raise ValueError("Fresh production topology does not contain exactly 45 inactive ghosts")
    topology_paths = save_system(gcmc_system, raw_prefix)
    topology = app.AmberPrmtopFile(topology_paths["prmtop"]).topology

    dynamics = make_dynamics(
        gcmc_system,
        restraints=ca_restraints(gcmc_system),
        platform=opt.md_platform,
        precision=opt.precision,
    )
    sampler.bind_dynamics(dynamics)
    context = dynamics.context()
    randomise_velocities(context, opt.seed)
    csv = CsvStateWriter(csv_file, context)
    dcd_handle = raw_dcd.open("wb")
    dcd = app.DCDFile(
        dcd_handle,
        topology,
        TIMESTEP_FS * unit.femtoseconds,
        firstStep=0,
        interval=opt.md_steps,
    )
    started = time.time()
    completed = 0
    try:
        for cycle in range(opt.cycles):
            md_started = time.time()
            completed = run_with_csv_reports(
                dynamics,
                context,
                opt.md_steps,
                completed,
                opt.report_interval,
                csv,
            )
            md_seconds = time.time() - md_started
            move_started = time.time()
            sampler.move(context)
            move_seconds = time.time() - move_started
            state = context.getState(
                getPositions=True, getEnergy=True, enforcePeriodicBox=True
            )
            dcd.writeModel(
                state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors()
            )
            sampler.write_ghost_residues()
            print_sampler(f"Production {cycle + 1}/{opt.cycles}", sampler)
            print(f"MD={md_seconds:.2f}s Loch={move_seconds:.2f}s", flush=True)
    finally:
        csv.close()
        dcd_handle.close()

    final_system = finalise_sampler_system(sampler, context)
    handoff = validate_gcmc_handoff(
        final_system,
        sampler,
        input_water_count=input_waters,
        label="Production finalized handoff",
    )
    save_physical_system(
        final_system,
        final_prefix,
        expected_water_count=handoff["expected_physical_waters"],
    )
    ghosts = ghost_history(ghost_file)
    frames = dcd_frames(raw_dcd)
    csv_audit = finite_csv(
        csv_file,
        total_steps=opt.cycles * opt.md_steps,
        report_interval=opt.report_interval,
    )
    if ghosts["lines"] != opt.cycles or frames != opt.cycles:
        raise ValueError(
            f"Production emitted {frames} frames and {ghosts['lines']} ghost lines "
            f"for {opt.cycles} cycles"
        )
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=outputs,
        details={
            "raw_topology": {
                "physical_plus_buffer_waters": int(raw_audit["water_molecules"]),
                "zero_interaction_waters": int(raw_audit["zero_interaction_water_count"]),
            },
            "handoff": handoff,
            "trajectory_frames": frames,
            "ghost_history": ghosts,
            "csv": csv_audit,
            "wall_seconds": time.time() - started,
        },
    )
    print(json.dumps(handoff, indent=2), flush=True)
    print(f"Production elapsed: {time.time() - started:.1f} s", flush=True)
    print(f"PRODUCTION_TOTAL_WALL_SECONDS={time.time() - total_started:.3f}", flush=True)


if __name__ == "__main__":
    main()
