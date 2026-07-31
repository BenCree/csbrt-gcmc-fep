#!/usr/bin/env python3
"""Select a subset of allowable FEP edges by A-optimal experimental design.

`screen_fep_edges.py` says which edges *could* be run. Running all of them is
wasteful: a network of L ligands needs only L-1 edges to be determined at all, and
every edge beyond that buys a diminishing reduction in the free-energy uncertainties.

The network fit in `aggregate_fep_network.py` is a weighted linear model -- each edge
is a row with -1 on state_a and +1 on state_b, and the ligand free energies come from
`lstsq` with covariance `pinv(design.T @ design)`. Choosing edges to minimise the trace
of that covariance is textbook A-optimal design, and it is Bayesian in the exact sense
that matters here: the trace is the posterior variance summed over ligands, under the
Gaussian noise model the aggregator already assumes.

Selection starts from a maximum-reliability spanning tree, so every component is
connected and full-rank by construction -- `aggregate_fep_network.py` raises
"FEP component is underdetermined" otherwise. Extra edges are then added greedily,
each time taking the one that most reduces the trace.

Edge noise is predicted from the mapped heavy-atom fraction: a poorly mapped edge
perturbs more atoms, which historically means worse lambda-window overlap and a noisier
DDG. That is a prior, not a fitted model. Every candidate and its features are written
to an edge ledger so real outcomes can replace the prior once FEP results exist.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# A mapped heavy fraction of 1.0 is the best case and gets the reference sigma; the
# noise is modelled as growing inversely with the mapped fraction. prepare_fep.py
# refuses to build an edge below 0.5 by default, so the realistic range is [0.5, 1.0]
# and the implied sigma ratio across it is 2x.
REFERENCE_SIGMA_KCAL_MOL = 0.3
MINIMUM_MAPPED_FRACTION = 1.0e-3


def edge_sigma(mapped_heavy_fraction: float, reference_sigma: float) -> float:
    """Prior standard deviation for an edge's DDG, from how well its ligands map."""
    fraction = max(float(mapped_heavy_fraction), MINIMUM_MAPPED_FRACTION)
    return reference_sigma / fraction


def read_edges(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No edges in {path}")
    edges = []
    for index, row in enumerate(rows):
        state_a = (row.get("state_a") or "").strip()
        state_b = (row.get("state_b") or "").strip()
        if not state_a or not state_b:
            raise ValueError(f"Row {index} of {path} is missing state_a/state_b")
        if state_a == state_b:
            raise ValueError(f"Row {index} of {path} is an identity edge {state_a}")
        allowable = (row.get("allowable") or "true").strip().lower()
        if allowable in {"false", "0", "no"}:
            continue
        raw_fraction = (row.get("mapped_heavy_fraction") or "").strip()
        fraction = float(raw_fraction) if raw_fraction else 1.0
        edges.append(
            {
                "state_a": state_a,
                "state_b": state_b,
                "edge_id": (row.get("edge_id") or f"{state_a}_to_{state_b}").strip(),
                "mapped_heavy_fraction": fraction,
            }
        )
    if not edges:
        raise ValueError(f"Every edge in {path} was marked not allowable")
    seen: set[str] = set()
    for edge in edges:
        edge_id = str(edge["edge_id"])
        if edge_id in seen:
            raise ValueError(f"Duplicate edge_id {edge_id!r} in {path}")
        seen.add(edge_id)
    return edges


def components(nodes: list[str], pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components, matching aggregate_fep_network.components()."""
    neighbours: dict[str, set[str]] = {node: set() for node in nodes}
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


def information_matrix(
    edges: list[dict[str, object]],
    columns: dict[str, int],
    anchor: str,
    reference_sigma: float,
) -> np.ndarray:
    """design.T @ design for the weighted least-squares fit of one component."""
    size = len(columns)
    design = np.zeros((len(edges), size))
    weights = np.zeros(len(edges))
    for row_index, edge in enumerate(edges):
        state_a = str(edge["state_a"])
        state_b = str(edge["state_b"])
        if state_a != anchor:
            design[row_index, columns[state_a]] = -1.0
        if state_b != anchor:
            design[row_index, columns[state_b]] = 1.0
        weights[row_index] = 1.0 / edge_sigma(
            float(edge["mapped_heavy_fraction"]), reference_sigma
        )
    weighted = design * weights[:, None]
    return weighted.T @ weighted


def posterior_trace(
    edges: list[dict[str, object]],
    columns: dict[str, int],
    anchor: str,
    reference_sigma: float,
) -> float:
    """Sum of posterior variances over the free ligands. Infinite if underdetermined."""
    if not columns:
        return 0.0
    matrix = information_matrix(edges, columns, anchor, reference_sigma)
    rank = np.linalg.matrix_rank(matrix)
    if rank < len(columns):
        return float("inf")
    return float(np.trace(np.linalg.pinv(matrix)))


def spanning_tree(
    group: list[str], candidates: list[dict[str, object]], reference_sigma: float
) -> list[dict[str, object]]:
    """Maximum-reliability spanning tree (Kruskal, lowest-sigma edges first)."""
    parent = {node: node for node in group}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    ordered = sorted(
        candidates,
        key=lambda edge: (
            edge_sigma(float(edge["mapped_heavy_fraction"]), reference_sigma),
            str(edge["edge_id"]),
        ),
    )
    chosen = []
    for edge in ordered:
        root_a = find(str(edge["state_a"]))
        root_b = find(str(edge["state_b"]))
        if root_a == root_b:
            continue
        parent[root_a] = root_b
        chosen.append(edge)
        if len(chosen) == len(group) - 1:
            break
    return chosen


def select_for_component(
    group: list[str],
    candidates: list[dict[str, object]],
    *,
    max_edges: int | None,
    variance_target: float | None,
    minimum_relative_gain: float,
    reference_sigma: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Greedy A-optimal selection within one connected component."""
    anchor = group[0]
    columns = {node: index for index, node in enumerate(group[1:])}
    history: list[dict[str, object]] = []

    selected = spanning_tree(group, candidates, reference_sigma)
    remaining = [edge for edge in candidates if edge not in selected]
    trace = posterior_trace(selected, columns, anchor, reference_sigma)
    history.append(
        {"step": 0, "added": "spanning_tree", "edges": len(selected), "trace": trace}
    )

    budget = max_edges if max_edges is not None else len(candidates)
    while remaining and len(selected) < budget:
        if variance_target is not None and trace <= variance_target:
            break
        best_edge = None
        best_trace = trace
        for edge in remaining:
            trial = posterior_trace(
                selected + [edge], columns, anchor, reference_sigma
            )
            if trial < best_trace:
                best_trace = trial
                best_edge = edge
        if best_edge is None:
            break
        # Diminishing returns: past the spanning tree every extra edge still lowers the
        # trace a little, so without this the greedy loop selects the entire candidate
        # set and the whole point of selecting is lost. Stop at the knee instead.
        if np.isfinite(trace) and trace > 0.0:
            relative_gain = (trace - best_trace) / trace
            if relative_gain < minimum_relative_gain:
                break
        selected.append(best_edge)
        remaining.remove(best_edge)
        history.append(
            {
                "step": len(history),
                "added": str(best_edge["edge_id"]),
                "edges": len(selected),
                "trace": best_trace,
            }
        )
        trace = best_trace
    return selected, history


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True,
                        help="Allowable-edge TSV from screen_fep_edges.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Selected-edge TSV, in the format make_fep_manifest.py reads")
    parser.add_argument("--ledger", type=Path, default=None,
                        help="Edge ledger JSON (default: <output>.ledger.json)")
    parser.add_argument("--selection", choices=("a-optimal", "spanning-tree", "all"),
                        default="a-optimal")
    parser.add_argument("--max-edges", type=int, default=None,
                        help="Cap on selected edges, summed over components")
    parser.add_argument("--variance-target", type=float, default=None,
                        help="Stop adding edges once the summed posterior variance "
                             "falls to this value")
    parser.add_argument("--reference-sigma", type=float, default=REFERENCE_SIGMA_KCAL_MOL,
                        help="Prior DDG sigma for a perfectly mapped edge (kcal/mol)")
    parser.add_argument("--min-relative-gain", type=float, default=0.02,
                        help="Stop adding edges once the best remaining one reduces the "
                             "posterior variance trace by less than this fraction "
                             "(default 0.02). Set 0 to add every edge that helps at all.")
    opt = parser.parse_args()
    if opt.max_edges is not None and opt.max_edges < 1:
        raise SystemExit("--max-edges must be positive")
    if not 0.0 <= opt.min_relative_gain < 1.0:
        raise SystemExit("--min-relative-gain must be in [0, 1)")
    if opt.reference_sigma <= 0:
        raise SystemExit("--reference-sigma must be positive")
    return opt


def main() -> None:
    opt = options()
    candidates = read_edges(opt.edges)
    nodes = sorted({str(edge[key]) for edge in candidates for key in ("state_a", "state_b")})
    pairs = [(str(edge["state_a"]), str(edge["state_b"])) for edge in candidates]
    groups = components(nodes, pairs)

    selected: list[dict[str, object]] = []
    component_reports = []
    for index, group in enumerate(groups):
        group_set = set(group)
        # Both endpoints must be inside the component. aggregate_fep_network.py filters
        # on state_a alone, which mis-partitions if an edge spans two groups; it cannot
        # here because groups come from the same edge list, but be explicit anyway.
        group_edges = [
            edge for edge in candidates
            if str(edge["state_a"]) in group_set and str(edge["state_b"]) in group_set
        ]
        if opt.selection == "all":
            chosen, history = list(group_edges), []
        elif opt.selection == "spanning-tree":
            chosen = spanning_tree(group, group_edges, opt.reference_sigma)
            history = []
        else:
            budget = None
            if opt.max_edges is not None:
                # Split the cap across components in proportion to their size.
                budget = max(len(group) - 1,
                             round(opt.max_edges * len(group) / len(nodes)))
            chosen, history = select_for_component(
                group, group_edges,
                max_edges=budget,
                variance_target=opt.variance_target,
                minimum_relative_gain=opt.min_relative_gain,
                reference_sigma=opt.reference_sigma,
            )
        anchor = group[0]
        columns = {node: position for position, node in enumerate(group[1:])}
        final_trace = posterior_trace(chosen, columns, anchor, opt.reference_sigma)
        full_trace = posterior_trace(group_edges, columns, anchor, opt.reference_sigma)
        component_reports.append(
            {
                "component": index,
                "ligands": len(group),
                "candidate_edges": len(group_edges),
                "selected_edges": len(chosen),
                "posterior_variance_trace": final_trace,
                "posterior_variance_trace_all_edges": full_trace,
                "history": history,
            }
        )
        selected.extend(chosen)

    if any(not np.isfinite(report["posterior_variance_trace"]) for report in component_reports):
        raise SystemExit(
            "Selection left a component underdetermined; aggregate_fep_network.py "
            "would reject this network"
        )

    # edge_index must renumber contiguously from 0: fep_edge.slurm maps it directly onto
    # SLURM_ARRAY_TASK_ID and hard-fails on a mismatch.
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    with opt.output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["edge_index", "edge_id", "state_a", "state_b",
                         "mapped_heavy_fraction", "prior_sigma_kcal_mol"])
        for index, edge in enumerate(selected):
            writer.writerow([
                index,
                edge["edge_id"],
                edge["state_a"],
                edge["state_b"],
                f"{float(edge['mapped_heavy_fraction']):.4f}",
                f"{edge_sigma(float(edge['mapped_heavy_fraction']), opt.reference_sigma):.4f}",
            ])

    selected_ids = {str(edge["edge_id"]) for edge in selected}
    ledger_path = opt.ledger or opt.output.with_suffix(".ledger.json")
    write_ledger(ledger_path, opt, candidates, selected_ids, component_reports)

    print(f"candidates : {len(candidates)}")
    print(f"selected   : {len(selected)}  ({opt.selection})")
    print(f"components : {len(groups)}  ligands: {len(nodes)}")
    for report in component_reports:
        print(
            f"  component {report['component']}: {report['ligands']} ligands, "
            f"{report['selected_edges']}/{report['candidate_edges']} edges, "
            f"variance trace {report['posterior_variance_trace']:.4f} "
            f"(all edges: {report['posterior_variance_trace_all_edges']:.4f})"
        )
    print(f"SELECTED_EDGES={opt.output}")
    print(f"EDGE_LEDGER={ledger_path}")


def write_ledger(
    path: Path,
    opt: argparse.Namespace,
    candidates: list[dict[str, object]],
    selected_ids: set[str],
    component_reports: list[dict[str, object]],
) -> None:
    """Record every candidate and its features so outcomes can be folded in later."""
    payload = {
        "source_edges": str(opt.edges),
        "selection": opt.selection,
        "reference_sigma_kcal_mol": opt.reference_sigma,
        "max_edges": opt.max_edges,
        "variance_target": opt.variance_target,
        "components": component_reports,
        "edges": [
            {
                "edge_id": edge["edge_id"],
                "state_a": edge["state_a"],
                "state_b": edge["state_b"],
                "mapped_heavy_fraction": edge["mapped_heavy_fraction"],
                "prior_sigma_kcal_mol": edge_sigma(
                    float(edge["mapped_heavy_fraction"]), opt.reference_sigma
                ),
                "selected": str(edge["edge_id"]) in selected_ids,
                # Filled in from analysis.json once the edge has actually run.
                "observed_ddg_kcal_mol": None,
                "observed_uncertainty_kcal_mol": None,
                "observed_minimum_overlap": None,
            }
            for edge in candidates
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
