#!/usr/bin/env python3
"""Materialize strict protein receptor inputs from PDB or macromolecular CIF."""

from __future__ import annotations

from pathlib import Path
import math
import shutil
import subprocess
import tempfile

from pipeline_utils import require_file, sha256


RECEPTOR_CONVERSION_VERSION = 1
SUPPORTED_RECEPTOR_SUFFIXES = {".pdb": "pdb", ".cif": "mmcif", ".mmcif": "mmcif"}


def receptor_format(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        return SUPPORTED_RECEPTOR_SUFFIXES[suffix]
    except KeyError as error:
        raise ValueError(
            f"Unsupported receptor format {path.suffix!r}; expected .pdb, .cif, or .mmcif"
        ) from error


def openbabel_provenance(source: Path) -> dict[str, object] | None:
    """Return the converter fingerprint needed for this receptor input."""
    if receptor_format(source) == "pdb":
        return None
    executable = shutil.which("obabel")
    if executable is None:
        raise FileNotFoundError(
            "A .cif/.mmcif receptor requires Open Babel 'obabel' in the active Mamba environment"
        )
    executable_path = Path(executable).resolve()
    version = subprocess.run(
        [str(executable_path), "-V"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "conversion_version": RECEPTOR_CONVERSION_VERSION,
        "executable": str(executable_path),
        "executable_sha256": sha256(executable_path),
        "version": (version.stdout or version.stderr).strip(),
        "input_format": "mmcif",
        "output_format": "pdb",
    }


def _canonicalize_openbabel_pdb(raw: Path, output: Path) -> None:
    """Remove generated bond tables and supply the TER record Open Babel omits."""
    lines = raw.read_text().splitlines(keepends=True)
    lines = [line for line in lines if not line.startswith(("CONECT", "MASTER"))]
    atom_indices = [index for index, line in enumerate(lines) if line.startswith("ATOM  ")]
    if not atom_indices:
        raise ValueError("Open Babel produced no receptor ATOM records from the mmCIF input")
    if not any(line.startswith("TER") for line in lines):
        last_index = atom_indices[-1]
        last = lines[last_index]
        serial = int(last[6:11]) + 1
        ter = (
            f"TER   {serial:5d}      {last[17:20]} {last[21]}"
            f"{last[22:26]}{last[26]}\n"
        )
        lines.insert(last_index + 1, ter)
    output.write_text("".join(lines))


def _mmcif_values(payload: dict[str, object], *names: str) -> list[str] | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            return [str(item) for item in value]
    return None


def audit_mmcif_conversion(source: Path, output: Path) -> dict[str, object]:
    """Independently compare mmCIF atom identity/order and coordinates to PDB."""
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as error:
        raise ImportError(
            "Audited mmCIF conversion requires Biopython in the Loch Mamba environment"
        ) from error

    payload = MMCIF2Dict(str(source))
    names = _mmcif_values(payload, "_atom_site.auth_atom_id", "_atom_site.label_atom_id")
    residues = _mmcif_values(payload, "_atom_site.auth_comp_id", "_atom_site.label_comp_id")
    sequences = _mmcif_values(payload, "_atom_site.auth_seq_id", "_atom_site.label_seq_id")
    elements = _mmcif_values(payload, "_atom_site.type_symbol")
    xs = _mmcif_values(payload, "_atom_site.Cartn_x")
    ys = _mmcif_values(payload, "_atom_site.Cartn_y")
    zs = _mmcif_values(payload, "_atom_site.Cartn_z")
    required = (names, residues, sequences, elements, xs, ys, zs)
    if any(values is None for values in required):
        raise ValueError("mmCIF receptor lacks required atom identity or Cartesian fields")
    assert all(values is not None for values in required)
    count = len(names)
    if count == 0 or any(len(values) != count for values in required):
        raise ValueError("mmCIF receptor atom-site columns have inconsistent lengths")

    groups = _mmcif_values(payload, "_atom_site.group_PDB")
    if groups is not None and any(group.upper() != "ATOM" for group in groups):
        raise ValueError("mmCIF receptor contains HETATM records unsupported by this path")
    models = _mmcif_values(payload, "_atom_site.pdbx_PDB_model_num")
    if models is not None and len(set(models)) != 1:
        raise ValueError("mmCIF receptor must contain exactly one coordinate model")
    altlocs = _mmcif_values(payload, "_atom_site.label_alt_id")
    if altlocs is not None and any(value not in {".", "?"} for value in altlocs):
        raise ValueError("mmCIF receptor contains unsupported alternate locations")
    insertions = _mmcif_values(payload, "_atom_site.pdbx_PDB_ins_code")
    if insertions is not None and any(value not in {".", "?"} for value in insertions):
        raise ValueError("mmCIF receptor contains unsupported insertion codes")

    chains = _mmcif_values(payload, "_atom_site.auth_asym_id", "_atom_site.label_asym_id")
    if chains is not None and any(len(chain) != 1 for chain in chains):
        raise ValueError("mmCIF chain identifiers must fit the one-character PDB boundary")

    pdb_lines = [
        line for line in output.read_text().splitlines() if line.startswith("ATOM  ")
    ]
    if len(pdb_lines) != count:
        raise ValueError(
            f"mmCIF-to-PDB atom count changed: source={count}, materialized={len(pdb_lines)}"
        )
    max_coordinate_delta = 0.0
    for index, line in enumerate(pdb_lines):
        expected_identity = (
            names[index], residues[index], int(float(sequences[index])), elements[index].upper()
        )
        actual_identity = (
            line[12:16].strip(), line[17:20].strip(), int(line[22:26]), line[76:78].strip().upper()
        )
        if actual_identity != expected_identity:
            raise ValueError(
                f"mmCIF-to-PDB atom identity changed at atom {index + 1}: "
                f"source={expected_identity}, materialized={actual_identity}"
            )
        if chains is not None and line[21] != chains[index]:
            raise ValueError(f"mmCIF-to-PDB chain changed at atom {index + 1}")
        source_xyz = (float(xs[index]), float(ys[index]), float(zs[index]))
        pdb_xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if not all(math.isfinite(value) for value in (*source_xyz, *pdb_xyz)):
            raise ValueError("mmCIF-to-PDB conversion produced non-finite coordinates")
        max_coordinate_delta = max(
            max_coordinate_delta,
            max(abs(before - after) for before, after in zip(source_xyz, pdb_xyz)),
        )
    if max_coordinate_delta > 1.0e-3:
        raise ValueError(
            f"mmCIF-to-PDB coordinates changed by {max_coordinate_delta:.6f} A"
        )
    return {
        "atoms": count,
        "explicit_hydrogens": sum(element.upper() == "H" for element in elements),
        "maximum_coordinate_delta_angstrom": max_coordinate_delta,
        "coordinate_tolerance_angstrom": 1.0e-3,
    }


def materialize_receptor_pdb(source: Path, output: Path, log: Path) -> dict[str, object]:
    """Copy PDB verbatim or convert macromolecular CIF to a deterministic PDB boundary."""
    source = require_file(source)
    source_format = receptor_format(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source_format == "pdb":
        if source.resolve() != output.resolve():
            shutil.copy2(source, output)
        log.write_text("[pipeline] receptor is PDB; copied without format conversion\n")
        return {
            "conversion_version": RECEPTOR_CONVERSION_VERSION,
            "source_format": "pdb",
            "converted": False,
            "source_sha256": sha256(source),
            "pdb_sha256": sha256(output),
        }

    provenance = openbabel_provenance(source)
    assert provenance is not None
    with tempfile.TemporaryDirectory(prefix="receptor-cif-", dir=output.parent) as temporary:
        raw = Path(temporary) / "openbabel.pdb"
        command = [
            str(provenance["executable"]),
            "-immcif",
            str(source),
            "-opdb",
            "-O",
            str(raw),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        log.write_text(
            f"[pipeline] command: {' '.join(command)}\n"
            f"{result.stdout}{result.stderr}"
        )
        _canonicalize_openbabel_pdb(require_file(raw), output)
    conversion_audit = audit_mmcif_conversion(source, output)
    return {
        **provenance,
        "source_format": source.suffix.lower().lstrip("."),
        "converted": True,
        "source_sha256": sha256(source),
        "pdb_sha256": sha256(output),
        "atom_conversion_audit": conversion_audit,
    }
