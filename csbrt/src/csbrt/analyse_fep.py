#!/usr/bin/env python3
"""Analyse bound/free SOMD2 legs and report the relative binding free energy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import BioSimSpace as BSS
import numpy as np

from pipeline_utils import validate_recorded_outputs, write_json_atomic


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-output", type=Path, required=True)
    parser.add_argument("--free-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge-id", required=True)
    return parser.parse_args()


def serialise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise(item) for item in value]
    return str(value)


def adjacent_overlaps(matrix: Any) -> list[float]:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 2:
        return []
    return [float(min(array[index, index + 1], array[index + 1, index])) for index in range(array.shape[0] - 1)]


def main() -> None:
    opt = options()
    bound = opt.bound_output.resolve()
    free = opt.free_output.resolve()
    validate_recorded_outputs(bound / "fep_leg.complete.json")
    validate_recorded_outputs(free / "fep_leg.complete.json")
    pmf_bound, overlap_bound = BSS.FreeEnergy.Relative.analyse(str(bound))
    pmf_free, overlap_free = BSS.FreeEnergy.Relative.analyse(str(free))
    difference = BSS.FreeEnergy.Relative.difference(pmf_bound, pmf_free)
    bound_adjacent = adjacent_overlaps(overlap_bound)
    free_adjacent = adjacent_overlaps(overlap_free)
    payload = {
        "status": "completed",
        "edge_id": opt.edge_id,
        "definition": "delta_delta_g_binding = delta_g_bound - delta_g_free",
        "relative_binding_free_energy": serialise(difference),
        "bound_pmf": serialise(pmf_bound),
        "free_pmf": serialise(pmf_free),
        "bound_overlap_matrix": serialise(overlap_bound),
        "free_overlap_matrix": serialise(overlap_free),
        "bound_adjacent_overlap_minimum": min(bound_adjacent) if bound_adjacent else None,
        "free_adjacent_overlap_minimum": min(free_adjacent) if free_adjacent else None,
        "bound_output": str(bound),
        "free_output": str(free),
    }
    output = opt.output.resolve()
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"FEP_ANALYSIS={output}", flush=True)


if __name__ == "__main__":
    main()
