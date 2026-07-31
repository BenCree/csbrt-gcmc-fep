"""Shared helpers for Grand Canonical Integration (GCI) titration with Loch.

GCI titrates the Adams value ``B`` across many independent windows and
integrates the resulting occupancy curve to obtain the binding free energy of a
water network (Ross et al., JACS 2015). It differs from ordinary GCMC/MD in two
ways that drive everything in this module:

1. **The GCMC sphere is anchored on an arbitrary fixed point** in the binding
   site (a hydration site found by clustering crystallographic waters), not on
   the ligand centroid. Loch's ``reference`` argument must be a Sire selection
   *string* and it has no coordinate API, so a massless dummy atom is inserted
   into the topology and selected by name. Mass 0 freezes the particle in
   OpenMM, so the centre Loch recomputes from live positions on every move is a
   genuinely fixed lab-frame point (there is no barostat in the muVT ensemble to
   rescale it).

2. **The excess chemical potential is the swept variable.** ``mu`` is derived
   from a target ``B`` for the radius actually in use, rather than read from a
   precomputed table. The historical workflow hardcoded 36 ``mu`` values that
   were only valid for its 4 A sphere; reusing them for the 7 A sphere shifted
   every ``B`` by about 1.7. Deriving ``mu`` here makes that class of error
   impossible.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import openmm
from openmm import unit
import sire as sr
import sire.system  # noqa: F401  (registers sr.system.System)
from loch import GCMCSampler

from ev71_loch_common import TEMPERATURE_K


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
# The schedule that produced the published titration data: 625 cycles of 400
# GCMC attempts interleaved with 8 ps of MD = 5.0 ns and 250,000 attempts per
# window. (The copy of the original script left on disk carried a much shorter
# smoke-test schedule whose per-250-cycle branch was unreachable.)
GCI_CYCLES = 625
GCI_ATTEMPTS = 400
GCI_MD_STEPS = 4_000
GCI_REPORT_INTERVAL = 500

# Offset applied to the base seed so GCI windows can never collide with the
# equilibration/production stages, which use base + 0..3.
GCI_SEED_OFFSET = 100

# Numeric mirror of ev71_loch_common.STANDARD_VOLUME ("30.345 A^3"), needed
# because the B/mu algebra works in floats.
GCI_STANDARD_VOLUME_ANGSTROM3 = 30.345

# exp(+-B) must stay representable, and Loch casts it to float32 before handing
# it to the acceptance kernel, so the limit is float32's exponent range (~88),
# not float64's. Above it, exp(B) becomes inf and every insertion is accepted.
GCI_MAX_ABS_B = 80.0

GCI_DUMMY_RESNAME = "DUM"
GCI_DUMMY_ATOMNAME = "SPH"
GCI_DUMMY_RESNUM = 9991
# Matches dum.xml's sigma of 0.1 nm. Physically irrelevant at epsilon 0, and it
# is not recoverable from an AMBER prmtop (LJ is stored as A/B coefficients,
# both of which vanish when epsilon is 0), so it is never asserted on reload.
GCI_DUMMY_SIGMA_ANGSTROM = 1.0

_ZERO_TOLERANCE = 1.0e-12


# ---------------------------------------------------------------------------
# B <-> mu algebra (single source of truth)
# ---------------------------------------------------------------------------
def kt_kcal_per_mol(temperature_K: float = TEMPERATURE_K) -> float:
    """Return kT in kcal/mol.

    This is Loch's own ``_beta`` expression inverted, so a B computed here and
    the B Loch computes internally agree to machine precision.
    """
    return float(sr.units.k_boltz.to("kcal/(mol*kelvin)") * float(temperature_K))


def sphere_volume_angstrom3(radius_angstrom: float) -> float:
    """Return 4*pi*r^3/3, identical to Loch's internal ``gcmc_volume``."""
    radius = float(radius_angstrom)
    if not radius > 0.0:
        raise ValueError(f"Sphere radius must be positive; got {radius}")
    return 4.0 * math.pi * radius**3 / 3.0


def mu_from_target_b(
    target_b: float,
    *,
    radius_angstrom: float,
    standard_volume_angstrom3: float = GCI_STANDARD_VOLUME_ANGSTROM3,
    adams_shift: float = 0.0,
    temperature_K: float = TEMPERATURE_K,
) -> float:
    """Return the excess chemical potential (kcal/mol) giving ``target_b``."""
    volume_ratio = sphere_volume_angstrom3(radius_angstrom) / float(
        standard_volume_angstrom3
    )
    return kt_kcal_per_mol(temperature_K) * (
        float(target_b) - float(adams_shift) - math.log(volume_ratio)
    )


def b_from_mu(
    mu_kcal_per_mol: float,
    *,
    radius_angstrom: float,
    standard_volume_angstrom3: float = GCI_STANDARD_VOLUME_ANGSTROM3,
    adams_shift: float = 0.0,
    temperature_K: float = TEMPERATURE_K,
) -> float:
    """Return the Adams value B implied by an excess chemical potential."""
    volume_ratio = sphere_volume_angstrom3(radius_angstrom) / float(
        standard_volume_angstrom3
    )
    return (
        float(mu_kcal_per_mol) / kt_kcal_per_mol(temperature_K)
        + math.log(volume_ratio)
        + float(adams_shift)
    )


def equilibrium_b(
    *,
    radius_angstrom: float,
    mu_hydration_kcal_per_mol: float,
    standard_volume_angstrom3: float = GCI_STANDARD_VOLUME_ANGSTROM3,
    adams_shift: float = 0.0,
    temperature_K: float = TEMPERATURE_K,
) -> float:
    """Return B at equilibrium with bulk water for this sphere."""
    return b_from_mu(
        mu_hydration_kcal_per_mol,
        radius_angstrom=radius_angstrom,
        standard_volume_angstrom3=standard_volume_angstrom3,
        adams_shift=adams_shift,
        temperature_K=temperature_K,
    )


def suggested_num_ghosts(
    radius_angstrom: float,
    standard_volume_angstrom3: float = GCI_STANDARD_VOLUME_ANGSTROM3,
) -> int:
    """Return a ghost-buffer size comfortably above bulk-equivalent occupancy.

    The buffer is the ceiling on how many waters can be inserted, so a buffer
    close to the expected occupancy clips the high-B tail of the titration
    curve and biases the integral.
    """
    bulk_equivalent = sphere_volume_angstrom3(radius_angstrom) / float(
        standard_volume_angstrom3
    )
    return max(20, 3 * math.ceil(bulk_equivalent))


# ---------------------------------------------------------------------------
# The dummy atom
# ---------------------------------------------------------------------------
def build_dummy_molecule(
    system,
    centre: Sequence[float],
    *,
    resname: str = GCI_DUMMY_RESNAME,
    atomname: str = GCI_DUMMY_ATOMNAME,
    resnum: int = GCI_DUMMY_RESNUM,
):
    """Return a one-atom, zero-mass, zero-charge, zero-epsilon Sire molecule.

    The typed property constructors matter: Sire's AMBER conversion requires
    ``AtomCharges``/``AtomLJs``/``AtomMasses`` (not generic per-atom values),
    plus ``forcefield``, ``connectivity`` and ``intrascale``. Assigning through
    a per-atom cursor yields the wrong property types and the molecule then
    fails to convert.
    """
    from sire.legacy.Maths import Vector
    from sire.legacy.MM import AtomLJs, CLJNBPairs, CLJScaleFactor, LJParameter
    from sire.legacy.Mol import (
        AtomCharges,
        AtomCoords,
        AtomElements,
        AtomMasses,
        AtomName,
        AtomNum,
        AtomStringProperty,
        CGName,
        Connectivity,
        Element,
        Molecule as LegacyMolecule,
        ResName,
        ResNum,
    )
    import sire.legacy.Units as legacy_units

    if len(tuple(centre)) != 3:
        raise ValueError(f"Sphere centre must have three components; got {centre!r}")
    coordinates = [float(value) for value in centre]
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"Sphere centre must be finite; got {centre!r}")

    waters = system["water and not property is_perturbable"].molecules()
    if not len(waters):
        raise ValueError("Cannot build the GCI dummy atom: system contains no water")
    forcefield = waters[0].molecule().property("forcefield")

    editor = LegacyMolecule(resname).edit()
    editor = editor.add(CGName(resname)).molecule()
    residue = editor.add(ResName(resname))
    residue.renumber(ResNum(int(resnum)))
    editor = residue.molecule()
    atom = editor.add(AtomName(atomname))
    atom.renumber(AtomNum(1))
    atom.reparent(ResName(resname))
    atom.reparent(CGName(resname))

    molecule = sr.mol.Molecule(atom.molecule().commit())
    info = molecule.info()
    atom_coordinates = AtomCoords(info)
    atom_coordinates.set(0, Vector(*coordinates))

    cursor = molecule.cursor()
    cursor["charge"] = AtomCharges(info, 0.0 * legacy_units.mod_electron)
    cursor["LJ"] = AtomLJs(
        info,
        LJParameter(
            GCI_DUMMY_SIGMA_ANGSTROM * legacy_units.angstrom,
            0.0 * legacy_units.kcal_per_mol,
        ),
    )
    cursor["mass"] = AtomMasses(info, 0.0 * legacy_units.g_per_mol)
    # Element 0 ("Xx") is the Sire equivalent of the original script forcing
    # atom.element = None so an elementless template would match.
    cursor["element"] = AtomElements(info, Element(0))
    cursor["ambertype"] = AtomStringProperty(info, atomname)
    cursor["atomtype"] = AtomStringProperty(info, atomname)
    cursor["coordinates"] = atom_coordinates
    cursor["forcefield"] = forcefield
    cursor["connectivity"] = Connectivity(info)
    cursor["intrascale"] = CLJNBPairs(info, CLJScaleFactor(0, 0))
    return cursor.commit()


def _select(system, selection: str):
    """Return the selection result, or None when nothing matched.

    Sire raises ``KeyError`` for a search that matches nothing rather than
    returning an empty selection, so probing for absence needs this wrapper.
    """
    try:
        return system[selection]
    except KeyError:
        return None


def count_molecules(system, selection: str) -> int:
    """Return how many molecules match ``selection`` (0 when nothing matches)."""
    result = _select(system, selection)
    return 0 if result is None else len(list(result.molecules()))


def dummy_selection(
    resname: str = GCI_DUMMY_RESNAME, atomname: str = GCI_DUMMY_ATOMNAME
) -> str:
    """Return the Loch reference selection for the dummy atom.

    Deliberately excludes ``resid``: an AMBER prmtop stores residue labels but
    no residue numbers, so a resid-based selection would not survive a save and
    reload.
    """
    return f"resname {resname} and atomname {atomname}"


def add_dummy_atom(
    system,
    centre: Sequence[float],
    *,
    resname: str = GCI_DUMMY_RESNAME,
    atomname: str = GCI_DUMMY_ATOMNAME,
    resnum: int = GCI_DUMMY_RESNUM,
) -> tuple[Any, int, str]:
    """Return ``(system_with_dummy, atom_index, reference_selection)``."""
    if count_molecules(system, f"resname {resname}"):
        raise ValueError(
            f"Input system already contains a {resname} residue; refusing to add "
            "a second GCI dummy atom"
        )
    reference = dummy_selection(resname, atomname)
    updated = system.clone()
    updated.add(
        build_dummy_molecule(
            system, centre, resname=resname, atomname=atomname, resnum=resnum
        )
    )
    return updated, atom_index(updated, reference), reference


# ---------------------------------------------------------------------------
# Validators. Every one raises; nothing here returns a soft failure.
# ---------------------------------------------------------------------------
def atom_index(system, selection: str) -> int:
    """Return the single atom index matching ``selection``, else raise."""
    result = _select(system, selection)
    atoms = [] if result is None else result.atoms()
    if len(atoms) != 1:
        raise ValueError(
            f"Selection {selection!r} matched {len(atoms)} atoms; the GCMC sphere "
            "centre requires exactly one"
        )
    found = system.atoms().find(atoms[0])
    return int(found if isinstance(found, int) else found[0])


def validate_dummy_atom(
    system,
    *,
    centre: Sequence[float],
    resname: str = GCI_DUMMY_RESNAME,
    atomname: str = GCI_DUMMY_ATOMNAME,
    tolerance_angstrom: float = 1.0e-3,
    label: str = "GCI dummy atom",
) -> dict[str, Any]:
    """Require exactly one frozen, non-interacting dummy atom at ``centre``.

    Sigma is deliberately not checked: it is meaningless at epsilon 0 and is
    lost when the topology round-trips through an AMBER prmtop.
    """
    selection = dummy_selection(resname, atomname)
    residue_result = _select(system, f"resname {resname}")
    molecules = [] if residue_result is None else list(residue_result.molecules())
    if len(molecules) != 1:
        raise RuntimeError(
            f"{label}: expected one {resname} molecule; found {len(molecules)}"
        )
    if molecules[0].num_atoms() != 1:
        raise RuntimeError(
            f"{label}: {resname} molecule has {molecules[0].num_atoms()} atoms; "
            "expected exactly one"
        )
    index = atom_index(system, selection)
    atom = system[selection].atoms()[0]

    mass = float(atom.property("mass").value())
    charge = float(atom.charge().value())
    epsilon = float(atom.property("LJ").epsilon().value())
    for name, value in (("mass", mass), ("charge", charge), ("epsilon", epsilon)):
        if abs(value) > _ZERO_TOLERANCE:
            raise RuntimeError(
                f"{label}: {name} is {value!r}; the dummy atom must be frozen and "
                "non-interacting so the GCMC sphere centre stays fixed"
            )

    expected = [float(value) for value in centre]
    position = atom.property("coordinates")
    actual = [
        float(position.x().value()),
        float(position.y().value()),
        float(position.z().value()),
    ]
    deviation = max(abs(a - b) for a, b in zip(actual, expected))
    if deviation > float(tolerance_angstrom):
        raise RuntimeError(
            f"{label}: dummy atom is at {actual} but the requested sphere centre "
            f"is {expected} (max deviation {deviation:.6g} A)"
        )
    return {
        "atom_index": index,
        "centre_angstrom": actual,
        "max_deviation_angstrom": deviation,
        "mass": mass,
        "charge": charge,
        "epsilon": epsilon,
    }


def validate_dummy_particle(context: openmm.Context, index: int) -> dict[str, float]:
    """Require the dummy to be frozen and non-interacting in the live context."""
    system = context.getSystem()
    if not 0 <= int(index) < system.getNumParticles():
        raise RuntimeError(
            f"Dummy atom index {index} is outside the OpenMM system "
            f"({system.getNumParticles()} particles)"
        )
    mass = system.getParticleMass(int(index)).value_in_unit(unit.dalton)
    nonbonded = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    if not nonbonded:
        raise RuntimeError("No OpenMM NonbondedForce was found")
    charge, sigma, epsilon = nonbonded[0].getParticleParameters(int(index))
    charge_e = charge.value_in_unit(unit.elementary_charge)
    epsilon_kj = epsilon.value_in_unit(unit.kilojoule_per_mole)
    for name, value in (
        ("mass", mass),
        ("charge", charge_e),
        ("epsilon", epsilon_kj),
    ):
        if abs(value) > _ZERO_TOLERANCE:
            raise RuntimeError(
                f"OpenMM dummy particle {index} has {name} = {value!r}; it must be "
                "zero so the GCMC sphere centre cannot move"
            )
    return {
        "mass_dalton": float(mass),
        "charge_e": float(charge_e),
        "sigma_nm": float(sigma.value_in_unit(unit.nanometer)),
        "epsilon_kj_mol": float(epsilon_kj),
    }


def box_perpendicular_widths(system) -> list[float]:
    """Return the three perpendicular widths of the periodic cell, in Angstrom.

    Works for triclinic cells, where the vector lengths overestimate the
    minimum-image limit.
    """
    space = system.property("space")
    try:
        matrix = space.box_matrix()
    except AttributeError as error:
        raise RuntimeError(
            f"Cannot determine box vectors from space of type "
            f"{type(space).__name__}"
        ) from error
    # box_matrix() holds the cell vectors as its COLUMNS, and is available on
    # both PeriodicBox (orthorhombic) and TriclinicBox. Reading the columns keeps
    # this correct for rectangular systems, which have no vector0/1/2 accessors.
    vectors = np.array(
        [
            [matrix.xx(), matrix.yx(), matrix.zx()],
            [matrix.xy(), matrix.yy(), matrix.zy()],
            [matrix.xz(), matrix.yz(), matrix.zz()],
        ],
        dtype=float,
    )
    volume = abs(float(np.dot(vectors[0], np.cross(vectors[1], vectors[2]))))
    if not volume > 0.0:
        raise RuntimeError("Periodic cell has non-positive volume")
    widths = []
    for first, second in ((1, 2), (2, 0), (0, 1)):
        area = float(np.linalg.norm(np.cross(vectors[first], vectors[second])))
        if not area > 0.0:
            raise RuntimeError("Periodic cell is degenerate")
        widths.append(volume / area)
    return widths


def validate_sphere_fits_box(system, radius_angstrom: float) -> dict[str, Any]:
    """Require the GCMC sphere to fit inside the minimum image convention.

    Loch performs no such check, and a sphere wider than half the cell would
    double-count waters through the periodic images.
    """
    radius = float(radius_angstrom)
    if not radius > 0.0:
        raise ValueError(f"Sphere radius must be positive; got {radius}")
    widths = box_perpendicular_widths(system)
    minimum = min(widths)
    if 2.0 * radius >= minimum:
        raise RuntimeError(
            f"GCMC sphere diameter {2.0 * radius:.3f} A does not fit the periodic "
            f"cell (minimum perpendicular width {minimum:.3f} A)"
        )
    return {
        "radius_angstrom": radius,
        "box_perpendicular_widths_angstrom": widths,
        "minimum_width_angstrom": minimum,
    }


_SOLVENT_RESNAMES = frozenset({"WAT", "HOH", "TIP3", "SOL", "T3P"})
_ION_RESNAMES = frozenset({"Na+", "Cl-", "K+", "Br-", "Mg+", "Ca+", "Zn+", "F-", "I-"})


def sphere_environment(
    system,
    reference: str,
    radius_angstrom: float,
    *,
    probe_radius_angstrom: float,
) -> dict[str, Any]:
    """Describe what surrounds the GCMC sphere centre.

    ``reference`` must select the dummy atom already placed at the centre, so the
    distance search runs inside Sire rather than over every atom in Python. Only
    the small candidate set it returns is classified here.
    """
    radius = float(radius_angstrom)
    probe = float(probe_radius_angstrom)
    if probe < radius:
        probe = radius
    candidates = _select(system, f"atoms within {probe:.6g} of ({reference})")
    atoms = [] if candidates is None else list(candidates.atoms())

    centre_atom = system[reference].atoms()[0]
    centre_position = centre_atom.property("coordinates")
    target = np.array(
        [
            float(centre_position.x().value()),
            float(centre_position.y().value()),
            float(centre_position.z().value()),
        ]
    )

    solute: list[float] = []
    water_oxygen: list[float] = []
    for atom in atoms:
        element = str(atom.element().symbol())
        residue = str(atom.residue().name().value())
        position = atom.property("coordinates")
        distance = float(
            np.linalg.norm(
                np.array(
                    [
                        float(position.x().value()),
                        float(position.y().value()),
                        float(position.z().value()),
                    ]
                )
                - target
            )
        )
        if residue in _SOLVENT_RESNAMES:
            if element == "O":
                water_oxygen.append(distance)
        elif residue in _ION_RESNAMES or element in ("H", "Xx", ""):
            continue
        else:
            solute.append(distance)

    return {
        "probe_radius_angstrom": probe,
        "candidate_atoms": len(atoms),
        "nearest_solute_heavy_atom_angstrom": float(min(solute)) if solute else None,
        "solute_heavy_atoms_within_radius": int(
            sum(1 for value in solute if value < radius)
        ),
        "water_oxygens_within_radius": int(
            sum(1 for value in water_oxygen if value < radius)
        ),
        "nearest_water_oxygen_angstrom": float(min(water_oxygen))
        if water_oxygen
        else None,
    }


def validate_sphere_environment(
    system,
    reference: str,
    radius_angstrom: float,
    *,
    min_solute_clearance_angstrom: float = 2.0,
    max_solute_distance_angstrom: float | None = None,
    require_waters_in_sphere: int = 0,
) -> dict[str, Any]:
    """Require the sphere centre to be a plausible hydration site in *this* input.

    Every other check on the sphere is self-consistent -- the dummy sits exactly
    where it was asked to, is frozen, and Loch resolves it. None of them can tell
    whether the coordinate means anything in the structure being sampled, and a
    centre carried over from a different equilibration run silently produces a
    titration curve for the wrong region. Two impossible cases are rejected:

    * buried inside a solute atom, where no water can ever be inserted;
    * far from any solute, i.e. bulk solvent rather than a pocket.
    """
    radius = float(radius_angstrom)
    if max_solute_distance_angstrom is None:
        # The sphere must at least touch the solute to be a pocket.
        max_solute_distance_angstrom = radius + 4.0
    maximum = float(max_solute_distance_angstrom)
    # Probe a little beyond the rejection threshold so "no solute nearby" is a
    # conclusion rather than an artefact of too small a search.
    environment = sphere_environment(
        system, reference, radius, probe_radius_angstrom=maximum + 2.0
    )
    nearest = environment["nearest_solute_heavy_atom_angstrom"]

    if nearest is None:
        raise RuntimeError(
            f"No solute heavy atom lies within "
            f"{environment['probe_radius_angstrom']:.2f} A of the GCMC sphere "
            "centre. The sphere is in bulk solvent rather than a binding-site "
            "pocket, so the titration would not describe a hydration site. Check "
            "--sphere-centre against this --rst7."
        )
    if nearest < float(min_solute_clearance_angstrom):
        raise RuntimeError(
            f"GCMC sphere centre is {nearest:.2f} A from the nearest solute heavy "
            f"atom, closer than the {float(min_solute_clearance_angstrom):.2f} A "
            "clearance a water oxygen needs. The centre is buried inside the "
            "solute, which usually means it was taken from a different structure "
            "or frame than --rst7."
        )
    if nearest > maximum:
        raise RuntimeError(
            f"GCMC sphere centre is {nearest:.2f} A from the nearest solute heavy "
            f"atom, beyond {maximum:.2f} A. The sphere lies in bulk solvent rather "
            "than a binding-site pocket, so the titration would not describe a "
            "hydration site."
        )
    waters = int(environment["water_oxygens_within_radius"])
    if waters < int(require_waters_in_sphere):
        raise RuntimeError(
            f"GCMC sphere contains {waters} water oxygens but at least "
            f"{int(require_waters_in_sphere)} were required"
        )
    environment["min_solute_clearance_angstrom"] = float(min_solute_clearance_angstrom)
    environment["max_solute_distance_angstrom"] = float(max_solute_distance_angstrom)
    return environment


def validate_reference_selection(system, reference: str) -> int:
    """Resolve ``reference`` exactly as Loch will, before building a sampler.

    Uses Loch's own static resolver, so a selection that would silently average
    over many atoms is caught here rather than after a full system clone.
    """
    indices = GCMCSampler._get_reference_indices(system, reference)
    values = [int(value) for value in np.atleast_1d(indices)]
    if len(values) != 1:
        raise RuntimeError(
            f"Loch resolved reference {reference!r} to {len(values)} atoms; the "
            "GCMC sphere centre must be a single fixed atom"
        )
    return values[0]


def assert_loch_adams_matches(
    sampler, expected_b: float, *, tolerance: float = 1.0e-9
) -> float:
    """Verify the B value Loch actually built against the intended one.

    Loch never exposes B publicly (it stores only exp(+-B) and emits a debug
    line), and its constructor accepts arbitrary unknown keywords silently. This
    is therefore the assertion that proves the sampler received the sphere and
    chemical potential we asked for.
    """
    exp_b = float(getattr(sampler, "_exp_B"))
    if not exp_b > 0.0 or not math.isfinite(exp_b):
        raise RuntimeError(
            f"Loch stored exp(B) = {exp_b!r}, so B cannot be verified; |B| is too "
            "large to represent"
        )
    actual = math.log(exp_b)
    if abs(actual - float(expected_b)) > float(tolerance):
        raise RuntimeError(
            f"Loch is sampling at B = {actual:.12g} but B = {float(expected_b):.12g} "
            "was requested; the radius, standard volume, chemical potential or "
            "Adams shift did not reach the sampler"
        )
    return actual


def validate_target_b(target_b: float) -> float:
    value = float(target_b)
    if not math.isfinite(value):
        raise ValueError(f"Target B must be finite; got {target_b!r}")
    if abs(value) > GCI_MAX_ABS_B:
        raise ValueError(
            f"|B| = {abs(value):.6g} exceeds {GCI_MAX_ABS_B:g}; exp(+-B) would "
            "overflow or underflow inside Loch"
        )
    return value


# ---------------------------------------------------------------------------
# Titration record
# ---------------------------------------------------------------------------
class TitrationWriter:
    """Write the per-checkpoint occupancy record that defines the B window.

    The original workflow left the occupancy only in log prose, so downstream
    analysis had to regex-parse scheduler output. All columns here are numeric
    so ``pipeline_utils.finite_csv`` can validate the schedule exactly.
    """

    COLUMNS = (
        "step",
        "cycle",
        "md_steps_completed",
        "sphere_waters",
        "accepted_moves",
        "accepted_attempts",
        "insertions",
        "deletions",
        "ghost_pool",
    )

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(self.COLUMNS)
        self._handle.flush()

    def write(
        self,
        *,
        step: int,
        cycle: int,
        md_steps_completed: int,
        sphere_waters: int,
        accepted_moves: int,
        accepted_attempts: int,
        insertions: int,
        deletions: int,
        ghost_pool: int,
    ) -> None:
        self._writer.writerow(
            [
                int(step),
                int(cycle),
                int(md_steps_completed),
                int(sphere_waters),
                int(accepted_moves),
                int(accepted_attempts),
                int(insertions),
                int(deletions),
                int(ghost_pool),
            ]
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def ghost_pool_size(sampler) -> int:
    """Return how many buffer waters are currently available for insertion."""
    return int(sum(1 for state in sampler.water_state() if int(state) == 0))
