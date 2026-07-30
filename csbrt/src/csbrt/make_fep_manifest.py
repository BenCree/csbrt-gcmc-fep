#!/usr/bin/env python3
"""Resolve a reviewed ligand-edge TSV to endpoint preparation directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

from pipeline_utils import require_file, validate_recorded_outputs


SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True, help="TSV: state_a, state_b[, edge_id]")
    parser.add_argument("--endpoint-run-root", type=Path, required=True)
    parser.add_argument("--replica", type=int, default=1)
    parser.add_argument(
        "--bound-frame",
        choices=("production", "preparation"),
        default="production",
        help="production: seed the FEP bound leg from state A's water-equilibrated "
        "'-production-final' restart (the point of the Loch pipeline); preparation: "
        "use the freshly-solvated prep system (legacy behaviour, empty bound columns)",
    )
    parser.add_argument(
        "--bound-frame-root",
        type=Path,
        help="Directory of precomputed per-ligand bound frames "
        "(LIGAND/production-final.{prmtop,rst7}), e.g. from select_bound_frames.py. "
        "Overrides --bound-frame; each edge reuses state A's representative placed "
        "waters so FEP recomputes no GCMC.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def precomputed_bound_frame(root: Path, ligand: str) -> tuple[Path, Path]:
    directory = root / ligand
    return (
        require_file(directory / "production-final.prmtop"),
        require_file(directory / "production-final.rst7"),
    )


def production_bound_frame(run_dir: Path) -> tuple[Path, Path]:
    """Resolve state A's water-equilibrated production restart as the bound frame."""
    production = run_dir / "production"
    validate_recorded_outputs(production / "production.complete.json")
    topologies = sorted(production.glob("*-production-final.prmtop"))
    coordinates = sorted(production.glob("*-production-final.rst7"))
    if len(topologies) != 1 or len(coordinates) != 1:
        raise ValueError(
            f"Expected one *-production-final.prmtop/rst7 pair in {production}; found "
            f"{len(topologies)} and {len(coordinates)}"
        )
    if topologies[0].stem.removesuffix("-production-final") != coordinates[0].stem.removesuffix(
        "-production-final"
    ):
        raise ValueError("Production-final topology and restart prefixes differ")
    return require_file(topologies[0]), require_file(coordinates[0])


def main() -> None:
    opt = options()
    if opt.replica < 1:
        raise ValueError("--replica must be positive")
    with require_file(opt.edges).open(newline="") as handle:
        edges = list(csv.DictReader(handle, delimiter="\t"))
    if not edges or not {"state_a", "state_b"}.issubset(edges[0]):
        raise ValueError("Edge TSV must contain state_a and state_b columns")
    root = opt.endpoint_run_root.resolve()
    rows = []
    seen = set()
    for index, edge in enumerate(edges):
        state_a, state_b = edge["state_a"], edge["state_b"]
        edge_id = edge.get("edge_id") or f"{state_a}_to_{state_b}"
        if any(not SAFE.fullmatch(value) for value in (state_a, state_b, edge_id)):
            raise ValueError(f"Unsafe edge identifiers at row {index + 2}")
        if edge_id in seen or state_a == state_b:
            raise ValueError(f"Duplicate/identity edge at row {index + 2}: {edge_id}")
        seen.add(edge_id)
        prep_a = root / state_a / f"rep{opt.replica}" / "preparation"
        prep_b = root / state_b / f"rep{opt.replica}" / "preparation"
        validate_recorded_outputs(prep_a / "preparation.complete.json")
        validate_recorded_outputs(prep_b / "preparation.complete.json")
        if opt.bound_frame_root is not None:
            bound_prmtop, bound_rst7 = precomputed_bound_frame(
                opt.bound_frame_root.resolve(), state_a
            )
            bound_prmtop_column, bound_rst7_column = str(bound_prmtop), str(bound_rst7)
        elif opt.bound_frame == "production":
            bound_prmtop, bound_rst7 = production_bound_frame(
                root / state_a / f"rep{opt.replica}"
            )
            bound_prmtop_column, bound_rst7_column = str(bound_prmtop), str(bound_rst7)
        else:
            bound_prmtop_column, bound_rst7_column = "", ""
        rows.append(
            {
                "edge_index": index,
                "edge_id": edge_id,
                "state_a": state_a,
                "state_b": state_b,
                "state_a_preparation": str(prep_a),
                "state_b_preparation": str(prep_b),
                "state_a_bound_prmtop": bound_prmtop_column,
                "state_a_bound_rst7": bound_rst7_column,
            }
        )
    output = opt.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"FEP_MANIFEST={output} EDGES={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
