"""Small checkpoint and validation helpers for the EV71 Loch pipeline."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
from typing import Any


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def resolve_scripts_dir(project: Path, sentinel: str) -> Path:
    """Locate the directory holding the stage scripts.

    In a source checkout they live in ``<project>/scripts/``. Once the package is
    pip-installed they sit beside this module in ``site-packages/csbrt/`` and
    ``<project>/scripts/`` does not exist -- ``Path(__file__).parents[1]`` then
    points at ``site-packages`` itself, so the naive ``project / "scripts"``
    resolves to a directory that was never installed.

    ``sentinel`` is a stage script the caller actually invokes, so an unrelated
    ``scripts/`` directory cannot satisfy the check.
    """
    candidate = project / "scripts"
    if (candidate / sentinel).is_file():
        return candidate
    packaged = Path(__file__).resolve().parent
    if (packaged / sentinel).is_file():
        return packaged
    raise FileNotFoundError(
        f"cannot locate {sentinel}: looked in {candidate} and {packaged}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Hash versioned pipeline sources without baking checkout paths into markers."""
    return {name: sha256(path.resolve()) for name, path in sorted(paths.items())}


def implementation_signature(
    *,
    sources: dict[str, Path],
    distributions: tuple[str, ...] = (),
    modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Fingerprint pipeline code and the third-party implementation it executes.

    Module hashes matter for local validation overlays: a patched Loch module can
    retain the same package version, so version strings alone are insufficient.
    """
    versions: dict[str, str] = {}
    for name in sorted(set(distributions)):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"

    module_hashes: dict[str, str] = {}
    module_versions: dict[str, str] = {}
    for name in sorted(set(modules)):
        spec = importlib.util.find_spec(name)
        origin = None if spec is None else spec.origin
        if origin is None or not Path(origin).is_file():
            module_hashes[name] = "no-python-source"
        else:
            module_hashes[name] = sha256(Path(origin))
        root_name = name.split(".", 1)[0]
        if root_name not in module_versions:
            module = importlib.import_module(root_name)
            module_versions[root_name] = str(getattr(module, "__version__", "unknown"))

    return {
        "python": platform.python_version(),
        "source_sha256": source_hashes(sources),
        "dependency_versions": versions,
        "module_versions": module_versions,
        "module_sha256": module_hashes,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(require_file(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def checkpoint_output_path(marker: Path, output: Path) -> tuple[str, Path]:
    """Return a marker-relative key and its resolved, contained output path.

    Checkpoints are intended to move with their stage directory.  Absolute
    output names silently made a copied run validate files in the original run,
    so every recorded artifact is now required to live beneath the marker.
    """
    root = marker.resolve().parent
    resolved = output.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Checkpoint output {resolved} is outside marker directory {root}"
        ) from error
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe checkpoint output path: {relative}")
    return relative.as_posix(), resolved


def output_hashes(marker: Path, outputs: list[Path]) -> dict[str, str]:
    return {
        key: sha256(resolved)
        for key, resolved in (checkpoint_output_path(marker, path) for path in outputs)
    }


def complete_checkpoint(
    marker: Path,
    *,
    signature: dict[str, Any],
    outputs: list[Path],
    details: dict[str, Any] | None = None,
) -> None:
    """Atomically publish a completed marker only after hashing every output."""
    payload: dict[str, Any] = {
        "status": "completed",
        "signature": signature,
        "output_sha256": output_hashes(marker, outputs),
    }
    if details:
        payload.update(details)
    write_json_atomic(marker, payload)


def invalidate_checkpoint(marker: Path) -> None:
    """Ensure a failed forced/rebuild attempt cannot leave an old valid marker."""
    marker.unlink(missing_ok=True)


def checkpoint_matches(
    marker: Path,
    *,
    signature: dict[str, Any],
    outputs: list[Path],
) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = read_json(marker)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if payload.get("status") != "completed" or payload.get("signature") != signature:
        return False
    recorded = payload.get("output_sha256")
    if not isinstance(recorded, dict):
        return False
    expected_keys: set[str] = set()
    for path in outputs:
        try:
            key, resolved = checkpoint_output_path(marker, path)
        except ValueError:
            return False
        expected_keys.add(key)
        if not resolved.is_file() or resolved.stat().st_size == 0:
            return False
        if recorded.get(key) != sha256(resolved):
            return False
    return set(recorded) == expected_keys


def validate_recorded_outputs(marker: Path) -> dict[str, Any]:
    """Independently verify status and every output hash recorded by a marker."""
    payload = read_json(marker)
    if payload.get("status") != "completed":
        raise ValueError(f"Checkpoint is not completed: {marker}")
    recorded = payload.get("output_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError(f"Checkpoint has no output hashes: {marker}")
    root = marker.resolve().parent
    for filename, expected in recorded.items():
        relative = Path(filename)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"Unsafe checkpoint output path in {marker}: {filename}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Checkpoint output escapes marker directory: {filename}"
            ) from error
        path = require_file(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checkpoint output hash mismatch: {path}")
    return payload


def finite_csv(
    path: Path,
    *,
    total_steps: int | None = None,
    report_interval: int | None = None,
) -> dict[str, Any]:
    import csv

    with require_file(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    for row_number, row in enumerate(rows, start=2):
        for key, value in row.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Non-numeric {key} at {path}:{row_number}") from error
            if not math.isfinite(numeric):
                raise ValueError(f"Non-finite {key} at {path}:{row_number}")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise ValueError(f"Steps are not strictly increasing in {path}")
    if (total_steps is None) != (report_interval is None):
        raise ValueError("total_steps and report_interval must be supplied together")
    if total_steps is not None and report_interval is not None:
        expected = list(range(report_interval, total_steps + 1, report_interval))
        if steps != expected:
            raise ValueError(
                f"CSV step schedule in {path} is {steps[:5]}...{steps[-5:]}; "
                f"expected {expected[:5]}...{expected[-5:]}"
            )
    digest = hashlib.sha256(",".join(str(step) for step in steps).encode()).hexdigest()
    return {
        "rows": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "step_sequence_sha256": digest,
    }


def read_ghost_records(path: Path) -> list[list[int]]:
    records: list[list[int]] = []
    for line_number, line in enumerate(require_file(path).read_text().splitlines(), start=1):
        try:
            values = [int(value.strip()) for value in line.split(",") if value.strip()]
        except ValueError as error:
            raise ValueError(f"Invalid ghost index at {path}:{line_number}") from error
        if any(value < 0 for value in values):
            raise ValueError(f"Negative ghost index at {path}:{line_number}")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate ghost index at {path}:{line_number}")
        records.append(values)
    if not records:
        raise ValueError(f"No ghost records in {path}")
    return records


def ghost_history(path: Path) -> dict[str, Any]:
    records = read_ghost_records(path)
    counts = [len(record) for record in records]
    return {
        "lines": len(records),
        "minimum_state_zero": min(counts),
        "maximum_state_zero": max(counts),
        "final_state_zero": counts[-1],
    }
