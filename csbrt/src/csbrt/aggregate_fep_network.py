#!/usr/bin/env python3
"""Fit ligand relative free energies and edge residuals from FEP analyses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from pipeline_utils import require_file, write_json_atomic


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def numeric(value: object) -> float:
    match = NUMBER.search(str(value))
    if match is None:
        raise ValueError(f"No numeric value in {value!r}")
    return float(match.group())


def components(nodes: list[str], pairs: list[tuple[str, str]]) -> list[list[str]]:
    neighbours = {node: set() for node in nodes}
    for left, right in pairs:
        neighbours[left].add(right)
        neighbours[right].add(left)
    groups = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen)
        stack, group = [start], []
        unseen.remove(start)
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbour in sorted(neighbours[node]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        groups.append(sorted(group))
    return groups


def main() -> None:
    opt = options()
    with require_file(opt.manifest).open(newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    if not manifest or not {"edge_id", "state_a", "state_b"}.issubset(manifest[0]):
        raise ValueError("FEP manifest lacks edge_id/state_a/state_b columns")
    root = opt.fep_root.resolve()
    edges = []
    for row in manifest:
        analysis_path = require_file(root / row["edge_id"] / "analysis.json")
        analysis = json.loads(analysis_path.read_text())
        estimate = analysis.get("relative_binding_free_energy")
        if not isinstance(estimate, list) or len(estimate) < 2:
            raise ValueError(f"No estimate/uncertainty pair in {analysis_path}")
        uncertainty = numeric(estimate[1])
        if uncertainty <= 0:
            uncertainty = 1.0e-6
        edges.append(
            {
                "edge_id": row["edge_id"],
                "state_a": row["state_a"],
                "state_b": row["state_b"],
                "ddg_kcal_mol": numeric(estimate[0]),
                "uncertainty_kcal_mol": uncertainty,
                "bound_adjacent_overlap_minimum": analysis.get("bound_adjacent_overlap_minimum"),
                "free_adjacent_overlap_minimum": analysis.get("free_adjacent_overlap_minimum"),
            }
        )
    nodes = sorted({edge[key] for edge in edges for key in ("state_a", "state_b")})
    groups = components(nodes, [(edge["state_a"], edge["state_b"]) for edge in edges])
    fitted: dict[str, dict[str, float | str]] = {}
    edge_rows = []
    for component_index, group in enumerate(groups):
        group_edges = [edge for edge in edges if edge["state_a"] in group]
        anchor = group[0]
        free_nodes = [node for node in group if node != anchor]
        columns = {node: index for index, node in enumerate(free_nodes)}
        design = np.zeros((len(group_edges), len(free_nodes)))
        values = np.zeros(len(group_edges))
        weights = np.zeros(len(group_edges))
        for row_index, edge in enumerate(group_edges):
            if edge["state_a"] != anchor:
                design[row_index, columns[edge["state_a"]]] = -1.0
            if edge["state_b"] != anchor:
                design[row_index, columns[edge["state_b"]]] = 1.0
            values[row_index] = edge["ddg_kcal_mol"]
            weights[row_index] = 1.0 / edge["uncertainty_kcal_mol"]
        weighted_design = design * weights[:, None]
        weighted_values = values * weights
        solution, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        if rank != len(free_nodes):
            raise ValueError(f"FEP component is underdetermined: {group}")
        energies = {anchor: 0.0, **{node: float(solution[index]) for node, index in columns.items()}}
        covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
        fitted[anchor] = {"ddg_kcal_mol": 0.0, "uncertainty_kcal_mol": 0.0, "anchor": anchor}
        for node, index in columns.items():
            fitted[node] = {
                "ddg_kcal_mol": energies[node],
                "uncertainty_kcal_mol": float(np.sqrt(max(covariance[index, index], 0.0))),
                "anchor": anchor,
            }
        for edge in group_edges:
            predicted = energies[edge["state_b"]] - energies[edge["state_a"]]
            residual = edge["ddg_kcal_mol"] - predicted
            edge_rows.append(
                {
                    **edge,
                    "component": component_index,
                    "fitted_ddg_kcal_mol": predicted,
                    "residual_kcal_mol": residual,
                    "standardised_residual": residual / edge["uncertainty_kcal_mol"],
                }
            )

    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "fep_network_edges.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)
    payload = {
        "status": "completed",
        "definition": "ligand values satisfy G(state_b)-G(state_a)=edge DDG",
        "components": groups,
        "ligands": fitted,
        "edges": edge_rows,
        "largest_absolute_standardised_residual": max(abs(row["standardised_residual"]) for row in edge_rows),
        "edge_table": str(csv_path),
    }
    json_path = output / "fep_network_analysis.json"
    write_json_atomic(json_path, payload)
    print(f"FEP_NETWORK_ANALYSIS={json_path}", flush=True)


if __name__ == "__main__":
    main()
