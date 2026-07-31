#!/usr/bin/env python3
"""Run one Grand Canonical Integration titration window with Loch.

One invocation samples a single Adams value B (equivalently, a single excess
chemical potential) in a spherical GCMC region anchored on a fixed point in the
binding site. Sweeping B is done by running this script once per window, exactly
as the original ``grand`` implementation did.

The window is defined by three things that must be given explicitly, because the
historical workflow used two different spherical regions (a 7 A sphere holding
seven crystallographic waters and a 4 A sphere holding two) and switched between
them by editing the script in place:

* ``--sphere-centre`` -- the fixed point, in the frame of ``--rst7``
* ``--sphere-radius`` -- the region size
* ``--target-b`` or ``--mu`` -- the titration point

``mu`` is derived from ``--target-b`` for the radius actually in use, so the
chemical potential can never be mismatched to the sphere. Passing ``--mu``
directly is also supported, and the implied B is recorded either way.
"""

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
    SEED,
    TEMPERATURE_K,
    TIMESTEP_FS,
    ca_restraints,
    finalise_sampler_system,
    make_dynamics,
    make_sampler,
    physical_protocol_signature,
    physical_water_audit,
    print_sampler,
    randomise_velocities,
    run_with_csv_reports,
    save_physical_system,
    save_system,
    validate_gcmc_handoff,
    validate_physical_water_topology,
    validate_output_prefix,
    validate_single_ligand,
    with_extension,
)
from gci_common import (
    GCI_ATTEMPTS,
    GCI_CYCLES,
    GCI_DUMMY_ATOMNAME,
    GCI_DUMMY_RESNAME,
    GCI_DUMMY_RESNUM,
    GCI_MD_STEPS,
    GCI_REPORT_INTERVAL,
    GCI_SEED_OFFSET,
    GCI_STANDARD_VOLUME_ANGSTROM3,
    TitrationWriter,
    add_dummy_atom,
    assert_loch_adams_matches,
    b_from_mu,
    equilibrium_b,
    ghost_pool_size,
    kt_kcal_per_mol,
    mu_from_target_b,
    sphere_volume_angstrom3,
    suggested_num_ghosts,
    validate_dummy_atom,
    validate_dummy_particle,
    validate_reference_selection,
    validate_sphere_environment,
    validate_sphere_fits_box,
    validate_target_b,
)
from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    finite_csv,
    ghost_history,
    implementation_signature,
    invalidate_checkpoint,
    sha256,
    write_json_atomic,
)

# The excess chemical potential of bulk TIP3P water, used for the equilibrium B
# reported alongside the window. Not a simulation input.
MU_HYDRATION_DEFAULT = -6.09


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--rst7", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--window-index", type=int, required=True)
    parser.add_argument(
        "--sphere-centre",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Fixed GCMC sphere centre in Angstrom, in the frame of --rst7.",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        required=True,
        help="GCMC sphere radius in Angstrom (no default: the region is a "
        "deliberate choice, not a protocol constant).",
    )
    titration = parser.add_mutually_exclusive_group(required=True)
    titration.add_argument(
        "--target-b",
        type=float,
        help="Adams value for this window; mu is derived for --sphere-radius.",
    )
    titration.add_argument(
        "--mu",
        type=float,
        help="Excess chemical potential in kcal/mol; B is derived for "
        "--sphere-radius.",
    )
    parser.add_argument(
        "--standard-volume", type=float, default=GCI_STANDARD_VOLUME_ANGSTROM3
    )
    parser.add_argument("--adams-shift", type=float, default=0.0)
    # Deliberately no --bulk-sampling-probability. Loch decides per move whether
    # to sample the whole box, and a bulk move uses B computed from the box
    # volume instead of the sphere volume. That is a different Adams value, it is
    # not what assert_loch_adams_matches checks, and it is meaningless for a
    # titration of one region. GCI always samples the sphere only.
    parser.add_argument(
        "--min-solute-clearance",
        type=float,
        default=2.0,
        help="Reject a sphere centre closer than this to a solute heavy atom "
        "(Angstrom); such a centre is buried and can never hold water.",
    )
    parser.add_argument(
        "--max-solute-distance",
        type=float,
        default=None,
        help="Reject a sphere centre further than this from the nearest solute "
        "heavy atom (Angstrom). Defaults to radius + 4 A, so the sphere must "
        "touch the solute rather than sit in bulk solvent.",
    )
    parser.add_argument(
        "--require-waters-in-sphere",
        type=int,
        default=0,
        help="Require at least this many water oxygens inside the sphere in the "
        "input structure.",
    )
    parser.add_argument(
        "--num-ghosts",
        type=int,
        default=None,
        help="Insertion buffer size. Defaults to a value comfortably above the "
        "bulk-equivalent occupancy of the sphere, because a buffer near the "
        "expected occupancy clips the high-B tail of the titration curve.",
    )
    parser.add_argument("--mu-hydration", type=float, default=MU_HYDRATION_DEFAULT)
    parser.add_argument("--dummy-resname", default=GCI_DUMMY_RESNAME)
    parser.add_argument("--dummy-atomname", default=GCI_DUMMY_ATOMNAME)
    parser.add_argument("--dummy-resnum", type=int, default=GCI_DUMMY_RESNUM)
    parser.add_argument("--cycles", type=int, default=GCI_CYCLES)
    parser.add_argument("--attempts", type=int, default=GCI_ATTEMPTS)
    parser.add_argument("--md-steps", type=int, default=GCI_MD_STEPS)
    parser.add_argument("--report-interval", type=int, default=GCI_REPORT_INTERVAL)
    parser.add_argument("--trajectory-stride", type=int, default=1)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--gcmc-platform", default="cuda", choices=("cuda", "opencl"))
    parser.add_argument(
        "--md-platform", default="cuda", choices=("cuda", "opencl", "cpu")
    )
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dcd_frames(path: Path) -> int:
    with md.open(str(path)) as handle:
        return len(handle)


def main() -> None:
    total_started = time.time()
    opt = options()

    if opt.window_index < 0:
        raise ValueError("--window-index must be non-negative")
    # Fail now rather than after the sampler is built: Sire cannot write a
    # topology whose basename carries an extra dot. Windows are distinguished by
    # their output directory, so a dot-free prefix costs nothing.
    if "/" in opt.prefix or "\\" in opt.prefix:
        raise ValueError("--prefix must be a filename component, not a path")
    validate_output_prefix(Path(opt.prefix))
    if (
        min(
            opt.batch_size,
            opt.cycles,
            opt.md_steps,
            opt.attempts,
            opt.report_interval,
            opt.trajectory_stride,
        )
        < 1
    ):
        raise ValueError("GCI counts and intervals must be positive")
    if opt.cycles % opt.trajectory_stride:
        raise ValueError(
            f"--trajectory-stride {opt.trajectory_stride} must divide --cycles "
            f"{opt.cycles}"
        )
    if not opt.sphere_radius > 0.0:
        raise ValueError("--sphere-radius must be positive")
    if not opt.standard_volume > 0.0:
        raise ValueError("--standard-volume must be positive")

    radius = float(opt.sphere_radius)
    standard_volume = float(opt.standard_volume)
    adams_shift = float(opt.adams_shift)
    centre = [float(value) for value in opt.sphere_centre]
    kt = kt_kcal_per_mol(TEMPERATURE_K)

    # Resolve the (B, mu) pair. Whichever the user supplied, the other is
    # derived from the radius actually in use, so they cannot disagree.
    if opt.target_b is not None:
        target_b = validate_target_b(opt.target_b)
        mu = mu_from_target_b(
            target_b,
            radius_angstrom=radius,
            standard_volume_angstrom3=standard_volume,
            adams_shift=adams_shift,
            temperature_K=TEMPERATURE_K,
        )
    else:
        mu = float(opt.mu)
        target_b = validate_target_b(
            b_from_mu(
                mu,
                radius_angstrom=radius,
                standard_volume_angstrom3=standard_volume,
                adams_shift=adams_shift,
                temperature_K=TEMPERATURE_K,
            )
        )

    num_ghosts = (
        int(opt.num_ghosts)
        if opt.num_ghosts is not None
        else suggested_num_ghosts(radius, standard_volume)
    )
    if num_ghosts < 1:
        raise ValueError("--num-ghosts must be positive")
    seed = (
        int(opt.seed)
        if opt.seed is not None
        else SEED + GCI_SEED_OFFSET + opt.window_index
    )

    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_top = opt.prmtop.resolve()
    input_rst = opt.rst7.resolve()

    raw_prefix = output / f"{opt.prefix}-loch-ghosts"
    raw_dcd = output / f"{opt.prefix}-raw.dcd"
    ghost_file = output / f"{opt.prefix}-gcmc-ghosts.txt"
    csv_file = output / f"{opt.prefix}_data_gci.csv"
    titration_csv = output / f"{opt.prefix}_titration.csv"
    titration_json = output / f"{opt.prefix}_titration.json"
    final_prefix = output / f"{opt.prefix}-gci-final"
    final_top = with_extension(final_prefix, "prmtop")
    final_rst = with_extension(final_prefix, "rst7")
    outputs = [
        with_extension(raw_prefix, "prmtop"),
        with_extension(raw_prefix, "rst7"),
        with_extension(raw_prefix, "pdb"),
        raw_dcd,
        ghost_file,
        csv_file,
        titration_csv,
        titration_json,
        final_top,
        final_rst,
        with_extension(final_prefix, "pdb"),
    ]

    reference = f"resname {opt.dummy_resname} and atomname {opt.dummy_atomname}"
    signature = {
        "stage": "gci_window",
        "input_prmtop_sha256": sha256(input_top),
        "input_rst7_sha256": sha256(input_rst),
        "prefix": opt.prefix,
        "window_index": opt.window_index,
        "target_b": target_b,
        "mu_kcal_per_mol": mu,
        "adams_shift": adams_shift,
        "radius_angstrom": radius,
        "standard_volume_angstrom3": standard_volume,
        "bulk_sampling_probability": 0.0,
        "num_ghost_waters": num_ghosts,
        "sphere_centre_angstrom": centre,
        "dummy_resname": opt.dummy_resname,
        "dummy_atomname": opt.dummy_atomname,
        "dummy_resnum": int(opt.dummy_resnum),
        "reference": reference,
        "temperature_K": TEMPERATURE_K,
        "ligand_resname": opt.ligand_resname,
        "gcmc_platform": opt.gcmc_platform,
        "md_platform": opt.md_platform,
        "precision": opt.precision,
        "seed": seed,
        "batch_size": opt.batch_size,
        "cycles": opt.cycles,
        "attempts_per_cycle": opt.attempts,
        "md_steps_per_cycle": opt.md_steps,
        "report_interval": opt.report_interval,
        "trajectory_stride": opt.trajectory_stride,
        # Governs the MD half only; deliberately unchanged so the ev71 audit
        # keeps recognising this protocol.
        "physical_protocol": physical_protocol_signature(),
        "implementation": implementation_signature(
            sources={
                "gci_window.py": Path(__file__),
                "gci_common.py": Path(__file__).with_name("gci_common.py"),
                "ev71_loch_common.py": Path(__file__).with_name(
                    "ev71_loch_common.py"
                ),
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

    marker = output / "gci_window.complete.json"
    expected_frames = opt.cycles // opt.trajectory_stride
    if not opt.force and checkpoint_matches(
        marker, signature=signature, outputs=outputs
    ):
        equilibrated = sr.load(str(input_top), str(input_rst))
        input_audit = validate_physical_water_topology(
            equilibrated, label="GCI checkpoint input"
        )
        input_waters = int(input_audit["water_molecules"])
        raw = sr.load(
            str(with_extension(raw_prefix, "prmtop")),
            str(with_extension(raw_prefix, "rst7")),
        )
        raw_audit = physical_water_audit(raw)
        if int(raw_audit["water_molecules"]) != input_waters + num_ghosts:
            raise ValueError(
                f"GCI checkpoint raw topology is not input + {num_ghosts} waters"
            )
        if int(raw_audit["zero_interaction_water_count"]) != num_ghosts:
            raise ValueError(
                f"GCI checkpoint raw topology does not have {num_ghosts} ghosts"
            )
        validate_dummy_atom(
            raw,
            centre=centre,
            resname=opt.dummy_resname,
            atomname=opt.dummy_atomname,
            label="GCI checkpoint raw topology",
        )
        final = sr.load(str(final_top), str(final_rst))
        final_audit = validate_physical_water_topology(
            final, label="GCI checkpoint final handoff"
        )
        validate_single_ligand(final, opt.ligand_resname)
        ghosts = ghost_history(ghost_file)
        frames = dcd_frames(raw_dcd)
        if ghosts["lines"] != opt.cycles or frames != expected_frames:
            raise ValueError(
                "GCI checkpoint frame/ghost counts do not match the schedule"
            )
        expected_final = input_waters + num_ghosts - int(ghosts["final_state_zero"])
        if int(final_audit["water_molecules"]) != expected_final:
            raise ValueError("GCI checkpoint physical-water arithmetic failed")
        finite_csv(
            csv_file,
            total_steps=opt.cycles * opt.md_steps,
            report_interval=opt.report_interval,
        )
        finite_csv(
            titration_csv,
            total_steps=opt.cycles * opt.attempts,
            report_interval=opt.attempts,
        )
        print(f"GCI window checkpoint is valid: {marker}", flush=True)
        return
    invalidate_checkpoint(marker)

    print(
        f"GCI window {opt.window_index}: B={target_b:+.4f} mu={mu:+.6f} kcal/mol "
        f"radius={radius:g} A centre={centre} ghosts={num_ghosts}",
        flush=True,
    )

    equilibrated = sr.load(str(input_top), str(input_rst))
    input_audit = validate_physical_water_topology(
        equilibrated, label="GCI window input"
    )
    validate_single_ligand(equilibrated, opt.ligand_resname)
    input_waters = int(input_audit["water_molecules"])
    input_zero_interaction = int(input_audit["zero_interaction_water_count"])

    # --- place the fixed sphere centre -------------------------------------
    system, dummy_index, reference_built = add_dummy_atom(
        equilibrated,
        centre,
        resname=opt.dummy_resname,
        atomname=opt.dummy_atomname,
        resnum=opt.dummy_resnum,
    )
    if reference_built != reference:
        raise RuntimeError(
            f"Reference selection mismatch: built {reference_built!r}, "
            f"signed {reference!r}"
        )
    dummy_audit = validate_dummy_atom(
        system,
        centre=centre,
        resname=opt.dummy_resname,
        atomname=opt.dummy_atomname,
        label="GCI dummy atom (in memory)",
    )
    box_audit = validate_sphere_fits_box(system, radius)
    # Every other check on the sphere is self-consistent -- the dummy is exactly
    # where it was asked to be, frozen, and resolvable. This is the only one that
    # can catch a centre carried over from a different structure or frame, the
    # failure that otherwise yields a clean run describing the wrong region.
    environment = validate_sphere_environment(
        system,
        reference,
        radius,
        min_solute_clearance_angstrom=opt.min_solute_clearance,
        max_solute_distance_angstrom=opt.max_solute_distance,
        require_waters_in_sphere=opt.require_waters_in_sphere,
    )
    print(
        f"Sphere environment: nearest solute heavy atom "
        f"{environment['nearest_solute_heavy_atom_angstrom']:.2f} A; inside the "
        f"sphere {environment['solute_heavy_atoms_within_radius']} solute heavy "
        f"atoms and {environment['water_oxygens_within_radius']} water oxygens",
        flush=True,
    )
    if not environment["water_oxygens_within_radius"]:
        print(
            "WARNING: the sphere contains no water in the input structure. That is "
            "expected only if this region is genuinely dehydrated at equilibrium; "
            "otherwise check --sphere-centre against this --rst7.",
            flush=True,
        )
    resolved = validate_reference_selection(system, reference)
    if resolved != dummy_index:
        raise RuntimeError(
            f"Loch resolves the sphere reference to atom {resolved}, but the dummy "
            f"atom is {dummy_index}"
        )
    # The dummy must not perturb any audit the rest of the pipeline relies on.
    post_audit = physical_water_audit(system)
    if int(post_audit["water_molecules"]) != input_waters:
        raise RuntimeError(
            f"Adding the dummy atom changed the water count from {input_waters} to "
            f"{int(post_audit['water_molecules'])}"
        )
    if int(post_audit["zero_interaction_water_count"]) != input_zero_interaction:
        raise RuntimeError("Adding the dummy atom changed the zero-interaction count")
    validate_single_ligand(system, opt.ligand_resname)

    # --- build the sampler at this window's chemical potential -------------
    sampler = make_sampler(
        system,
        attempts=opt.attempts,
        batch_size=opt.batch_size,
        seed=seed,
        log_file=output / f"{opt.prefix}-gcmc.log",
        ghost_file=ghost_file,
        ligand_resname=opt.ligand_resname,
        platform=opt.gcmc_platform,
        reference=reference,
        radius=f"{radius:.12g} A",
        excess_chemical_potential=f"{mu:.12g} kcal/mol",
        standard_volume=f"{standard_volume:.12g} A^3",
        num_ghost_waters=num_ghosts,
        adams_shift=adams_shift,
        # Sphere-only sampling: see the note in options().
        bulk_sampling_probability=0.0,
    )
    adams_value = assert_loch_adams_matches(sampler, target_b)
    print(f"Loch is sampling at B = {adams_value:.12g}", flush=True)

    gcmc_system = sampler.system()
    raw_audit = physical_water_audit(gcmc_system)
    if int(raw_audit["water_molecules"]) != input_waters + num_ghosts:
        raise ValueError(
            f"GCI sampler did not append exactly {num_ghosts} waters"
        )
    if int(raw_audit["zero_interaction_water_count"]) != num_ghosts:
        raise ValueError(
            f"Fresh GCI topology does not contain exactly {num_ghosts} inactive ghosts"
        )
    sampler_dummy_audit = validate_dummy_atom(
        gcmc_system,
        centre=centre,
        resname=opt.dummy_resname,
        atomname=opt.dummy_atomname,
        label="GCI dummy atom (sampler topology)",
    )
    if int(sampler_dummy_audit["atom_index"]) != dummy_index:
        raise RuntimeError(
            f"Appending the ghost buffer moved the dummy atom from index "
            f"{dummy_index} to {sampler_dummy_audit['atom_index']}; the OpenMM "
            "restraint index would be wrong"
        )
    topology_paths = save_system(gcmc_system, raw_prefix)
    # Reload from disk: an AMBER round-trip that lost the dummy's zero mass
    # would silently unfreeze the sphere centre, so prove it survived. The index
    # is checked too, because this topology is the DCD's atom map -- a writer that
    # reordered molecules would leave coordinate-correct frames labelled wrongly.
    reloaded = sr.load(topology_paths["prmtop"], topology_paths["rst7"])
    reloaded_dummy_audit = validate_dummy_atom(
        reloaded,
        centre=centre,
        resname=opt.dummy_resname,
        atomname=opt.dummy_atomname,
        label="GCI dummy atom (reloaded from AMBER)",
    )
    if int(reloaded_dummy_audit["atom_index"]) != dummy_index:
        raise RuntimeError(
            f"The saved AMBER topology places the dummy atom at index "
            f"{reloaded_dummy_audit['atom_index']} instead of {dummy_index}; atom "
            "order changed on write, so the trajectory topology is unreliable"
        )
    topology = app.AmberPrmtopFile(topology_paths["prmtop"]).topology

    # --- dynamics ---------------------------------------------------------
    # No pressure: GCI is muVT. Loch rejects a barostat without a pressure, and
    # a barostat would also rescale the dummy atom's coordinate.
    restraint_indices = list(ca_restraints(gcmc_system)) + [dummy_index]
    dynamics = make_dynamics(
        gcmc_system,
        restraints=restraint_indices,
        platform=opt.md_platform,
        precision=opt.precision,
    )
    sampler.bind_dynamics(dynamics)
    context = dynamics.context()
    particle_audit = validate_dummy_particle(context, dummy_index)
    randomise_velocities(context, seed)

    # Confirm the sphere centre Loch will compute is the point we asked for.
    # No enforcePeriodicBox here: wrapping would report a lattice-shifted image.
    live = context.getState(getPositions=True)
    live_position = live.getPositions(asNumpy=True)[dummy_index].value_in_unit(
        unit.angstrom
    )
    realised_centre = [float(value) for value in live_position]
    drift = max(abs(a - b) for a, b in zip(realised_centre, centre))
    if drift > 1.0e-4:
        raise RuntimeError(
            f"Live dummy position {realised_centre} differs from the requested "
            f"centre {centre} by {drift:.6g} A"
        )

    csv = CsvStateWriter(csv_file, context)
    titration = TitrationWriter(titration_csv)
    dcd_handle = raw_dcd.open("wb")
    dcd = app.DCDFile(
        dcd_handle,
        topology,
        TIMESTEP_FS * unit.femtoseconds,
        firstStep=0,
        interval=opt.md_steps * opt.trajectory_stride,
    )
    started = time.time()
    completed = 0
    minimum_ghost_pool = num_ghosts
    try:
        for cycle in range(opt.cycles):
            completed = run_with_csv_reports(
                dynamics,
                context,
                opt.md_steps,
                completed,
                opt.report_interval,
                csv,
            )
            sampler.move(context)
            # Pass the context explicitly: without it Loch returns a count
            # cached at the start of the last GCMC batch, which is blind to
            # waters that diffused across the sphere boundary during MD. This
            # also refreshes the value print_sampler reports.
            sphere_waters = int(sampler.num_waters(context))
            pool = ghost_pool_size(sampler)
            minimum_ghost_pool = min(minimum_ghost_pool, pool)
            if pool < 1:
                raise RuntimeError(
                    f"Ghost reservoir exhausted at cycle {cycle + 1} of "
                    f"{opt.cycles} (B={target_b:+.4f}). Sphere occupancy is "
                    f"clipped at {sphere_waters}; re-run this window with a "
                    f"larger --num-ghosts (currently {num_ghosts})."
                )
            titration.write(
                step=(cycle + 1) * opt.attempts,
                cycle=cycle + 1,
                md_steps_completed=completed,
                sphere_waters=sphere_waters,
                accepted_moves=int(sampler.num_accepted_moves()),
                accepted_attempts=int(sampler.num_accepted_attempts()),
                insertions=int(sampler.num_insertions()),
                deletions=int(sampler.num_deletions()),
                ghost_pool=pool,
            )
            sampler.write_ghost_residues()
            if (cycle + 1) % opt.trajectory_stride == 0:
                state = context.getState(getPositions=True, enforcePeriodicBox=True)
                dcd.writeModel(
                    state.getPositions(),
                    periodicBoxVectors=state.getPeriodicBoxVectors(),
                )
            print_sampler(
                f"GCI B={target_b:+.4f} {cycle + 1}/{opt.cycles}", sampler
            )
    finally:
        csv.close()
        titration.close()
        dcd_handle.close()

    final_system = finalise_sampler_system(sampler, context)
    handoff = validate_gcmc_handoff(
        final_system,
        sampler,
        input_water_count=input_waters,
        label="GCI finalized handoff",
        num_ghost_waters=num_ghosts,
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
    titration_audit = finite_csv(
        titration_csv,
        total_steps=opt.cycles * opt.attempts,
        report_interval=opt.attempts,
    )
    if ghosts["lines"] != opt.cycles or frames != expected_frames:
        raise ValueError(
            f"GCI window emitted {frames} frames and {ghosts['lines']} ghost lines "
            f"for {opt.cycles} cycles at stride {opt.trajectory_stride}"
        )

    metadata = {
        "stage": "gci_window",
        "window_index": opt.window_index,
        "prefix": opt.prefix,
        "target_b": target_b,
        "adams_value_from_loch": adams_value,
        "mu_kcal_per_mol": mu,
        "adams_shift": adams_shift,
        "radius_angstrom": radius,
        "sphere_volume_angstrom3": sphere_volume_angstrom3(radius),
        "standard_volume_angstrom3": standard_volume,
        "bulk_sampling_probability": 0.0,
        "temperature_K": TEMPERATURE_K,
        "kt_kcal_per_mol": kt,
        "mu_hydration_kcal_per_mol": float(opt.mu_hydration),
        "equilibrium_b": equilibrium_b(
            radius_angstrom=radius,
            mu_hydration_kcal_per_mol=float(opt.mu_hydration),
            standard_volume_angstrom3=standard_volume,
            adams_shift=adams_shift,
            temperature_K=TEMPERATURE_K,
        ),
        "sphere_centre_angstrom": centre,
        "realised_sphere_centre_angstrom": realised_centre,
        "sphere_centre_drift_angstrom": drift,
        "dummy_resname": opt.dummy_resname,
        "dummy_atomname": opt.dummy_atomname,
        "dummy_resnum": int(opt.dummy_resnum),
        "dummy_atom_index": dummy_index,
        "dummy_atom_audit": dummy_audit,
        "dummy_particle_audit": particle_audit,
        "box_audit": box_audit,
        "sphere_environment": environment,
        "reference": reference,
        "num_ghost_waters": num_ghosts,
        "minimum_ghost_pool": minimum_ghost_pool,
        "input_physical_waters": input_waters,
        "cycles": opt.cycles,
        # Recorded because it is the titration checkpoint interval in GCMC
        # attempts; analysis must not assume a value for it.
        "attempts_per_cycle": opt.attempts,
        "md_steps_per_cycle": opt.md_steps,
        "report_interval": opt.report_interval,
        "trajectory_stride": opt.trajectory_stride,
        "total_gcmc_attempts": opt.cycles * opt.attempts,
        "total_md_steps": opt.cycles * opt.md_steps,
        "total_md_ps": opt.cycles * opt.md_steps * TIMESTEP_FS / 1000.0,
        "seed": seed,
        "batch_size": opt.batch_size,
        "ligand_resname": opt.ligand_resname,
        "gcmc_platform": opt.gcmc_platform,
        "md_platform": opt.md_platform,
        "precision": opt.precision,
        "input_prmtop_sha256": sha256(input_top),
        "input_rst7_sha256": sha256(input_rst),
        "titration_csv": titration_csv.name,
        "handoff": handoff,
        "ghost_history": ghosts,
        "trajectory_frames": frames,
        "energy_csv": csv_audit,
        "titration_csv_audit": titration_audit,
        # The shared ev71 MD protocol, recorded for provenance. Five of its
        # entries describe the ligand-centred GCMC/MD stages and do NOT apply
        # here; the authoritative GCI values are the top-level keys above.
        "md_protocol_signature": physical_protocol_signature(),
        "md_protocol_superseded_keys": [
            "sphere_radius",
            "excess_chemical_potential",
            "num_ghost_waters",
            "pressure",
            "barostat_frequency",
        ],
        "implementation": signature["implementation"],
    }
    write_json_atomic(titration_json, metadata)

    complete_checkpoint(
        marker,
        signature=signature,
        outputs=outputs,
        details={
            "raw_topology": {
                "physical_plus_buffer_waters": int(raw_audit["water_molecules"]),
                "zero_interaction_waters": int(
                    raw_audit["zero_interaction_water_count"]
                ),
            },
            "window": {
                "target_b": target_b,
                "adams_value_from_loch": adams_value,
                "mu_kcal_per_mol": mu,
                "radius_angstrom": radius,
                "sphere_centre_angstrom": centre,
                "num_ghost_waters": num_ghosts,
                "minimum_ghost_pool": minimum_ghost_pool,
            },
            "handoff": handoff,
            "trajectory_frames": frames,
            "ghost_history": ghosts,
            "csv": csv_audit,
            "titration_csv": titration_audit,
            "wall_seconds": time.time() - started,
        },
    )
    print(
        json.dumps(
            {
                "target_b": target_b,
                "adams_value_from_loch": adams_value,
                "mu_kcal_per_mol": mu,
                "radius_angstrom": radius,
                "sphere_centre_angstrom": centre,
                "num_ghost_waters": num_ghosts,
                "minimum_ghost_pool": minimum_ghost_pool,
                "handoff": handoff,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"GCI window elapsed: {time.time() - started:.1f} s", flush=True)
    print(
        f"GCI_WINDOW_TOTAL_WALL_SECONDS={time.time() - total_started:.3f}", flush=True
    )


if __name__ == "__main__":
    main()
