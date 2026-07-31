"""Shared, system-independent helpers for the Ludovic-style Loch protocol."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import openmm
from openmm import unit
import sire as sr
from loch import GCMCSampler


TEMPERATURE_K = 300.0
TEMPERATURE = f"{TEMPERATURE_K:g} K"
FRICTION_PER_PS = 1.0
CUTOFF = "12 A"
SWITCH_DISTANCE_NM = 1.0
EWALD_ERROR_TOLERANCE = 5.0e-4
SPHERE_RADIUS = "10 A"
MU_EX = "-6.09 kcal/mol"
STANDARD_VOLUME = "30.345 A^3"
NUM_GHOSTS = 45
TIMESTEP_FS = 2.0
TIMESTEP = f"{TIMESTEP_FS:g} fs"
PRESSURE = "1 bar"
CA_RESTRAINT_K = 100.0
MINIMIZATION_TOLERANCE_KJ_MOL_NM = 10.0
COM_RESET_FREQUENCY = 1
BAROSTAT_FREQUENCY = 25
DEFAULT_BATCH_SIZE = 50

UVT1_INITIAL_ATTEMPTS = 10_000
UVT1_CYCLES = 100
UVT1_ATTEMPTS = 1_000
UVT1_MD_STEPS = 5
UVT1_REPORT_INTERVAL = 100
NPT_STEPS = 1_000_000
NPT_REPORT_INTERVAL = 2_500
UVT2_CYCLES = 125
UVT2_ATTEMPTS = 800
UVT2_MD_STEPS = 2_000
UVT2_REPORT_INTERVAL = 500
PRODUCTION_CYCLES = 2_500
PRODUCTION_ATTEMPTS = 200
PRODUCTION_MD_STEPS = 2_000
PRODUCTION_REPORT_INTERVAL = 500
SEED = 20260714


def physical_protocol_signature() -> dict[str, object]:
    """Return every fixed Ludovic physical setting used by MD or GCMC."""
    return {
        "temperature_K": TEMPERATURE_K,
        "friction_per_ps": FRICTION_PER_PS,
        "cutoff": CUTOFF,
        "switch_distance_nm": SWITCH_DISTANCE_NM,
        "ewald_error_tolerance": EWALD_ERROR_TOLERANCE,
        "sphere_radius": SPHERE_RADIUS,
        "excess_chemical_potential": MU_EX,
        "standard_volume": STANDARD_VOLUME,
        "num_ghost_waters": NUM_GHOSTS,
        "timestep_fs": TIMESTEP_FS,
        "pressure": PRESSURE,
        "ca_restraint_k_kj_mol_nm2": CA_RESTRAINT_K,
        "minimization_tolerance_kj_mol_nm": MINIMIZATION_TOLERANCE_KJ_MOL_NM,
        "com_reset_frequency": COM_RESET_FREQUENCY,
        "barostat_frequency": BAROSTAT_FREQUENCY,
        "cutoff_type": "pme",
        "constraint": "h_bonds",
        "integrator": "langevin_middle",
        "bulk_sampling_probability": 0.0,
    }


def nvcc_path() -> str:
    path = Path(sys.executable).with_name("nvcc")
    if not path.is_file():
        raise FileNotFoundError(f"Loch requires nvcc; expected {path}")
    return str(path)


def validate_single_ligand(system, resname: str = "LIG") -> dict[str, int]:
    """Require exactly one ligand molecule for the GCMC sphere reference."""
    selection = system[f"resname {resname}"]
    molecules = list(selection.molecules())
    residues = list(selection.residues())
    if len(molecules) != 1 or len(residues) != 1:
        raise ValueError(
            f"Expected one {resname} molecule/residue; found "
            f"{len(molecules)} molecule(s) and {len(residues)} residue(s)"
        )
    return {"molecules": 1, "residues": 1, "atoms": int(molecules[0].num_atoms())}


def ca_restraints(system):
    """Return the OpenMM particle indices of protein C-alpha atoms."""
    all_atoms = system.atoms()
    indices = []
    for atom in system["protein and atomname CA"].atoms():
        found = all_atoms.find(atom)
        indices.append(found if isinstance(found, int) else found[0])
    if not indices:
        raise ValueError("No protein C-alpha atoms found for restraints")
    return indices


def add_ca_restraints(context: openmm.Context, indices: list[int]) -> None:
    """Add Ludovic's particle-free E=100*r^2 periodic restraint force."""
    positions = context.getState(getPositions=True).getPositions()
    force = openmm.CustomExternalForce(
        "k*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    force.addGlobalParameter(
        "k", CA_RESTRAINT_K * unit.kilojoule_per_mole / unit.nanometer**2
    )
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for index in indices:
        xyz = positions[index].value_in_unit(unit.nanometer)
        force.addParticle(index, [xyz.x, xyz.y, xyz.z])
    context.getSystem().addForce(force)


def configure_ludovic_nonbonded(context: openmm.Context) -> None:
    """Apply the live OpenMM settings used by GRAND in Ludovic's scripts."""
    changed = False
    for force in context.getSystem().getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(SWITCH_DISTANCE_NM * unit.nanometer)
            force.setUseDispersionCorrection(False)
            force.setEwaldErrorTolerance(EWALD_ERROR_TOLERANCE)
            changed = True
    if not changed:
        raise RuntimeError("No OpenMM NonbondedForce was found")
    context.reinitialize(preserveState=True)


def make_dynamics(
    system,
    *,
    restraints=None,
    pressure=None,
    barostat_frequency=None,
    timestep=TIMESTEP,
    platform="cuda",
    precision="mixed",
):
    kwargs = dict(
        integrator="langevin_middle",
        temperature=TEMPERATURE,
        pressure=pressure,
        cutoff_type="pme",
        cutoff=CUTOFF,
        constraint="h_bonds",
        timestep=timestep,
        com_reset_frequency=COM_RESET_FREQUENCY,
        platform=platform,
        save_frequency=0,
        # Sire 2025.4 exposes friction only through its property map. Passing
        # it as a direct _dynamics() keyword raises TypeError before context
        # construction. Keep Ludovic's explicit 1 ps^-1 value here rather
        # than relying on Sire's currently identical default.
        map={"friction": FRICTION_PER_PS / sr.units.picosecond},
    )
    if platform.lower() in {"cuda", "opencl", "hip"}:
        kwargs["precision"] = precision
    if barostat_frequency is not None:
        kwargs["barostat_frequency"] = barostat_frequency
    dynamics = system.dynamics(**kwargs)
    if restraints is not None:
        add_ca_restraints(dynamics.context(), restraints)
    configure_ludovic_nonbonded(dynamics.context())
    return dynamics


def make_sampler(
    system,
    *,
    attempts: int,
    batch_size: int,
    seed: int,
    log_file: Path,
    ghost_file: Path,
    ligand_resname: str = "LIG",
    platform: str = "cuda",
    reference: str | None = None,
    radius: str = SPHERE_RADIUS,
    excess_chemical_potential: str = MU_EX,
    standard_volume: str = STANDARD_VOLUME,
    num_ghost_waters: int = NUM_GHOSTS,
    adams_shift: float = 0.0,
    bulk_sampling_probability: float = 0.0,
):
    """Build a Loch sampler.

    Every keyword defaults to the fixed Ludovic value, so callers that omit them
    are unaffected. They are exposed because Grand Canonical Integration needs a
    per-window chemical potential and a sphere anchored on a dummy atom rather
    than on the ligand.
    """
    if reference is None:
        reference = f"resname {ligand_resname}"
    # Loch validates these only after cloning the whole system, and its
    # **kwargs silently swallows unrecognised keywords (so a mistyped
    # sphere_centre= would be accepted and ignored). Check here, where the
    # failure is cheap and attributable.
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("'reference' must be a non-empty Sire selection string")
    if not isinstance(num_ghost_waters, int) or num_ghost_waters < 1:
        raise ValueError("'num_ghost_waters' must be a positive integer")
    if attempts < batch_size:
        raise ValueError(
            f"'attempts' ({attempts}) must be greater than or equal to "
            f"'batch_size' ({batch_size}); Loch evaluates one batch of trials at "
            "a time and cannot exceed the attempt budget"
        )
    if not 0.0 <= float(bulk_sampling_probability) <= 1.0:
        raise ValueError("'bulk_sampling_probability' must be between 0 and 1")
    kwargs = dict(
        system=system,
        reference=reference,
        radius=radius,
        cutoff_type="pme",
        cutoff=CUTOFF,
        excess_chemical_potential=excess_chemical_potential,
        standard_volume=standard_volume,
        temperature=TEMPERATURE,
        num_ghost_waters=num_ghost_waters,
        batch_size=batch_size,
        num_attempts=attempts,
        adams_shift=float(adams_shift),
        bulk_sampling_probability=float(bulk_sampling_probability),
        platform=platform,
        seed=seed,
        log_file=str(log_file),
        ghost_file=str(ghost_file),
        log_level="info",
        overwrite=True,
    )
    if platform.lower() == "cuda":
        kwargs["nvcc"] = nvcc_path()
    return GCMCSampler(**kwargs)


def randomise_velocities(context: openmm.Context, seed: int) -> None:
    integrator = context.getIntegrator()
    if hasattr(integrator, "setRandomNumberSeed"):
        integrator.setRandomNumberSeed(seed)
    context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, seed)


def image_context(context: openmm.Context) -> None:
    state = context.getState(getPositions=True, enforcePeriodicBox=True)
    context.setPositions(state.getPositions())


def update_system_from_context(system, context: openmm.Context):
    """Return a clone with imaged live coordinates and live periodic box."""
    from sire.legacy.IO import setCoordinates

    state = context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = (state.getPositions(asNumpy=True) / unit.angstrom).tolist()
    updated = system.clone()
    updated._system = setCoordinates(updated._system, positions)
    box = state.getPeriodicBoxVectors()
    v0 = [10 * box[0].x, 10 * box[0].y, 10 * box[0].z]
    v1 = [10 * box[1].x, 10 * box[1].y, 10 * box[1].z]
    v2 = [10 * box[2].x, 10 * box[2].y, 10 * box[2].z]
    updated.set_property(
        "space",
        sr.vol.TriclinicBox(
            sr.maths.Vector(*v0), sr.maths.Vector(*v1), sr.maths.Vector(*v2)
        ),
    )
    return updated


def finalise_sampler_system(sampler, context: openmm.Context):
    """Return the live physical system with inactive ghost waters removed.

    Loch 2025.2 exposes the water state but has no ``finalise_system`` helper.
    The appended buffer waters have zero charge and epsilon in the stored Sire
    topology. Loch only changes their parameters in the live OpenMM Context,
    so accepted buffer waters must be made interacting in the Sire topology
    before it is saved for the next simulation stage.
    """
    updated = update_system_from_context(sampler.system(), context)
    water_state = sampler.water_state()
    water_indices = getattr(sampler, "_water_indices", None)
    if water_indices is None or len(water_indices) != len(water_state):
        raise RuntimeError("Loch water-state and water-index arrays are inconsistent")

    num_buffer = getattr(sampler, "_num_ghost_waters", None)
    if not isinstance(num_buffer, int) or not 0 <= num_buffer <= len(water_state):
        raise RuntimeError("Loch ghost-buffer size is unavailable or inconsistent")
    first_buffer = len(water_state) - num_buffer
    active_buffer_count = sum(
        int(water_state[index] != 0)
        for index in range(first_buffer, len(water_state))
    )
    expected_active_waters = sum(int(state != 0) for state in water_state)

    ghost_oxygens = [
        int(water_indices[index])
        for index, state in enumerate(water_state)
        if state == 0
    ]
    if ghost_oxygens:
        # Capture the molecules before deleting any: each removal changes the
        # subsequent Sire atom indices. Removing the stable molecule objects
        # one at a time mirrors newer Loch topology handling.
        ghost_molecules = [
            updated[updated.atoms()[oxygen].molecule()]
            for oxygen in ghost_oxygens
        ]
        if len({molecule.number() for molecule in ghost_molecules}) != len(
            ghost_oxygens
        ):
            raise RuntimeError(
                "Ghost oxygen indices do not map one-to-one to water molecules"
            )
        if any(molecule.num_atoms() != 3 for molecule in ghost_molecules):
            raise RuntimeError("A Loch ghost oxygen mapped to a non-TIP3P molecule")
        for molecule in ghost_molecules:
            updated.remove(molecule)

    # Accepted appended waters remain zero-interaction molecules in Loch
    # 2025.2's stored Sire topology. After all logical ghosts are removed,
    # they are the only zero-interaction waters that may remain. Requiring the
    # count to match the logical active-buffer count prevents us from silently
    # activating an unrelated malformed water.
    zero_waters = []
    for molecule in updated["water and not property is_perturbable"].molecules():
        atoms = list(molecule.atoms())
        if molecule.num_atoms() != 3:
            continue
        charges = [abs(float(atom.charge().value())) for atom in atoms]
        epsilons = [
            abs(float(atom.property("LJ").epsilon().value())) for atom in atoms
        ]
        if all(value < 1.0e-8 for value in charges + epsilons):
            zero_waters.append(molecule)
    if len(zero_waters) != active_buffer_count:
        raise RuntimeError(
            "Loch handoff retained "
            f"{len(zero_waters)} zero-interaction waters but logical state has "
            f"{active_buffer_count} active buffer waters"
        )

    template = getattr(sampler, "_water_template", None)
    if template is None or template.num_atoms() != 3:
        raise RuntimeError("Loch physical water template is unavailable")
    template_atoms = list(template.atoms())
    water_charge = [float(atom.charge().value()) for atom in template_atoms]
    water_sigma = [
        float(atom.property("LJ").sigma().value()) for atom in template_atoms
    ]
    water_epsilon = [
        float(atom.property("LJ").epsilon().value()) for atom in template_atoms
    ]
    if all(
        abs(value) < 1.0e-8 for value in water_charge + water_epsilon
    ):
        raise RuntimeError("Loch water template is itself non-interacting")

    activated_molecule_numbers = []
    for molecule in zero_waters:
        cursor = molecule.cursor()
        atoms = list(cursor.atoms())
        if [str(atom.name) for atom in atoms] != [
            str(atom.name().value()) for atom in template_atoms
        ]:
            raise RuntimeError("A retained zero-interaction water mismatches the template")
        for atom_index, atom in enumerate(atoms):
            atom["charge"] = water_charge[atom_index] * sr.units.mod_electron
            atom["LJ"] = sr.legacy.MM.LJParameter(
                water_sigma[atom_index] * sr.units.angstrom,
                water_epsilon[atom_index] * sr.units.kcal_per_mol,
            )
        committed = cursor.commit()
        updated.update(committed)
        activated_molecule_numbers.append(committed.number())

    for molecule_number in activated_molecule_numbers:
        committed = updated[molecule_number]
        for atom_index, atom in enumerate(committed.atoms()):
            lj = atom.property("LJ")
            actual = (
                float(atom.charge().value()),
                float(lj.sigma().value()),
                float(lj.epsilon().value()),
            )
            expected = (
                water_charge[atom_index],
                water_sigma[atom_index],
                water_epsilon[atom_index],
            )
            if not all(
                math.isclose(value, target, rel_tol=1.0e-7, abs_tol=1.0e-8)
                for value, target in zip(actual, expected)
            ):
                raise RuntimeError("Failed to activate a Loch buffer water")

    validate_physical_water_topology(
        updated,
        expected_water_count=expected_active_waters,
        label="Loch finalized handoff",
    )

    print(
        f"Loch handoff: activated {len(zero_waters)} accepted buffer "
        f"waters and removed {len(ghost_oxygens)} inactive ghosts",
        flush=True,
    )
    return updated


def physical_water_audit(system) -> dict[str, object]:
    """Summarize physical and all-zero three-site waters in a Sire system."""
    waters = []
    zero_interaction = []
    for molecule in system["water and not property is_perturbable"].molecules():
        if molecule.num_atoms() != 3:
            continue
        waters.append(molecule)
        charges = [abs(float(atom.charge().value())) for atom in molecule.atoms()]
        epsilons = [
            abs(float(atom.property("LJ").epsilon().value()))
            for atom in molecule.atoms()
        ]
        if all(value < 1.0e-8 for value in charges + epsilons):
            residue = molecule.residues()[0]
            zero_interaction.append(
                {
                    "molecule_number": int(molecule.number().value()),
                    "residue": str(residue.name().value()),
                    "residue_number": int(residue.number().value()),
                }
            )
    return {
        "water_molecules": len(waters),
        "zero_interaction_water_count": len(zero_interaction),
        "zero_interaction_waters": zero_interaction,
    }


def validate_physical_water_topology(
    system,
    *,
    expected_water_count: int | None = None,
    label: str = "physical topology",
) -> dict[str, object]:
    """Reject a physical stage boundary containing missing or all-zero waters."""
    audit = physical_water_audit(system)
    count = int(audit["water_molecules"])
    zero_count = int(audit["zero_interaction_water_count"])
    if expected_water_count is not None and count != expected_water_count:
        raise RuntimeError(
            f"{label} contains {count} waters; expected {expected_water_count}"
        )
    if zero_count:
        examples = audit["zero_interaction_waters"]
        assert isinstance(examples, list)
        raise RuntimeError(
            f"{label} contains {zero_count} all-zero charge/LJ waters: "
            f"{examples[:6]}"
        )
    return audit


def validate_gcmc_handoff(
    system,
    sampler,
    *,
    input_water_count: int,
    label: str,
    num_ghost_waters: int = NUM_GHOSTS,
) -> dict[str, int]:
    """Cross-check a finalized handoff against the sampler's logical state.

    ``num_ghost_waters`` must match the buffer the sampler was built with;
    the water arithmetic is silently wrong otherwise.
    """
    state = list(sampler.water_state())
    state_zero = sum(int(value == 0) for value in state)
    expected = input_water_count + num_ghost_waters - state_zero
    audit = validate_physical_water_topology(
        system,
        expected_water_count=expected,
        label=label,
    )
    return {
        "input_physical_waters": int(input_water_count),
        "buffer_waters": int(num_ghost_waters),
        "final_state_zero_waters": state_zero,
        "expected_physical_waters": expected,
        "saved_physical_waters": int(audit["water_molecules"]),
        "zero_interaction_waters": int(audit["zero_interaction_water_count"]),
    }


def with_extension(prefix: Path, extension: str) -> Path:
    """Return ``prefix`` with ``extension`` appended.

    ``Path.with_suffix`` replaces everything after the last dot in the basename,
    so a prefix containing a decimal or a dotted identifier would be truncated
    and two different stages could collide on one filename. Appending is always
    what is meant here, because the prefix is a caller-supplied name and not a
    filename with a suffix to replace.
    """
    return prefix.with_name(f"{prefix.name}.{extension.lstrip('.')}")


def validate_output_prefix(prefix: Path) -> Path:
    """Reject a prefix Sire cannot write to.

    Sire chooses the output format from the file extension and refuses any
    basename containing another dot, so a prefix such as ``run_mu-8.45`` fails
    deep inside the writer with "Cannot find parsers that support the following
    format". Catching it here makes the cause obvious.
    """
    name = prefix.name
    if not name:
        raise ValueError("Output prefix must not be empty")
    if "." in name:
        raise ValueError(
            f"Output prefix {name!r} contains a '.', which Sire cannot write: it "
            "infers the file format from the extension and rejects any further dot "
            "in the basename. Use a dot-free prefix (put varying values such as a "
            "chemical potential in the directory name instead)."
        )
    return prefix


def save_system(system, prefix: Path) -> dict[str, str]:
    validate_output_prefix(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "prmtop": str(with_extension(prefix, "prmtop")),
        "rst7": str(with_extension(prefix, "rst7")),
        "pdb": str(with_extension(prefix, "pdb")),
    }
    for path in paths.values():
        sr.save(system, path, parallel=False, save_velocities=False)
    return paths


def save_physical_system(
    system,
    prefix: Path,
    *,
    expected_water_count: int | None = None,
) -> dict[str, str]:
    """Save and reload-audit a ghost-free physical stage boundary."""
    before = validate_physical_water_topology(
        system,
        expected_water_count=expected_water_count,
        label=f"{prefix} in memory",
    )
    paths = save_system(system, prefix)
    reloaded = sr.load(paths["prmtop"], paths["rst7"])
    after = validate_physical_water_topology(
        reloaded,
        expected_water_count=int(before["water_molecules"]),
        label=f"{prefix} reloaded from AMBER",
    )
    print(
        f"Physical handoff validated: {prefix} "
        f"waters={after['water_molecules']} zero_interaction=0",
        flush=True,
    )
    return paths


class CsvStateWriter:
    def __init__(self, path: Path, context: openmm.Context):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(
            ["step", "potential_kj_mol", "kinetic_kj_mol", "temperature_K", "volume_nm3", "density_g_ml"]
        )
        system = context.getSystem()
        self._mass_da = sum(
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        )
        self._dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3

    def write(self, step: int, context: openmm.Context) -> None:
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        volume = abs(float(vectors[0].dot(__import__("numpy").cross(vectors[1], vectors[2]))))
        temperature = 2.0 * kinetic / (self._dof * 0.00831446261815324)
        density = self._mass_da * 0.00166053906660 / volume
        self._writer.writerow([step, potential, kinetic, temperature, volume, density])
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def run_with_csv_reports(
    dynamics,
    context: openmm.Context,
    num_steps: int,
    completed_steps: int,
    report_interval: int,
    writer: CsvStateWriter,
) -> int:
    """Run MD and write states at OpenMM StateDataReporter-style intervals."""
    if num_steps < 0 or completed_steps < 0:
        raise ValueError("MD step counts cannot be negative")
    if report_interval < 1:
        raise ValueError("Report interval must be positive")
    remaining = num_steps
    while remaining:
        until_report = report_interval - (completed_steps % report_interval)
        chunk = min(remaining, until_report)
        dynamics.run(chunk, save_frequency=0, auto_fix_minimise=False)
        completed_steps += chunk
        remaining -= chunk
        if completed_steps % report_interval == 0:
            writer.write(completed_steps, context)
    return completed_steps


def print_sampler(label: str, sampler) -> None:
    moves = getattr(sampler, "_num_moves", 0)
    accepted = sampler.num_accepted_moves()
    print(
        f"{label}: sphere_waters={sampler.num_waters()} "
        f"move_calls={moves} accepted_transitions={accepted} "
        f"insertions={sampler.num_insertions()} deletions={sampler.num_deletions()}",
        flush=True,
    )
