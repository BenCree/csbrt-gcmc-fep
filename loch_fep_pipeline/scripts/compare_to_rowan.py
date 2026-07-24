#!/usr/bin/env python3
"""Compare this pipeline's FEP network against the Rowan OpenBind EV-A71 benchmark.

Joins ``aggregate_fep_network.py``'s ``fep_network_analysis.json`` against Rowan's
``rowan_results_per_edge_wide.csv`` (per-edge reference DDG) and, when available,
``pyrrolidine_32_subset.csv`` / ``rowan_results_per_compound_wide.csv`` (per-ligand
experimental dG).  Reports edge-level agreement (our DDG vs Rowan's) and, if the
experimental data is supplied, ligand-level agreement of the network-fitted dG.

Everything is relative, so edge DDGs are compared directly (orientation is matched
per unordered pair) and ligand dGs are mean-centred over the common ligand set
before scoring.  Statistics use numpy only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True,
                        help="fep_network_analysis.json from aggregate_fep_network.py")
    parser.add_argument("--rowan-edges", type=Path, required=True,
                        help="rowan_results_per_edge_wide.csv")
    parser.add_argument("--rowan-edge-column", default="xtal_nagl_ddg_kcal_mol",
                        help="Reference DDG column to compare against")
    parser.add_argument("--experimental", type=Path,
                        help="Optional per-compound CSV with experimental dG "
                        "(pyrrolidine_32_subset.csv or rowan_results_per_compound_wide.csv)")
    parser.add_argument("--experimental-column", default="experimental_delta_g_kcal_mol",
                        help="Experimental dG column (kcal/mol); falls back to pKD*-1.364")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        # average ties
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts)); np.add.at(sums, inverse, ranks)
        return (sums / counts)[inverse]
    return pearson(rank(x), rank(y))


def metrics(ours: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = ours - reference
    return {
        "n": int(len(ours)),
        "pearson_r": pearson(ours, reference),
        "spearman_r": spearman(ours, reference),
        "mue_kcal_mol": float(np.mean(np.abs(diff))),
        "rmse_kcal_mol": float(np.sqrt(np.mean(diff ** 2))),
        "mean_signed_error_kcal_mol": float(np.mean(diff)),
    }


def load_network_edges(path: Path) -> tuple[dict[tuple[str, str], dict], dict[str, float]]:
    payload = json.loads(path.read_text())
    edges = {}
    for edge in payload.get("edges", []):
        key = (edge["state_a"], edge["state_b"])
        edges[key] = {"ddg": float(edge["ddg_kcal_mol"]),
                      "err": float(edge.get("uncertainty_kcal_mol", float("nan")))}
    ligands = {name: float(value["ddg_kcal_mol"])
               for name, value in payload.get("ligands", {}).items()}
    return edges, ligands


def load_rowan_edges(path: Path, column: str) -> dict[frozenset, tuple[str, str, float]]:
    table = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            a, b = row["reference_ligand_a"], row["reference_ligand_b"]
            value = row.get(column, "").strip()
            if value in ("", "nan", "NaN"):
                continue
            table[frozenset((a, b))] = (a, b, float(value))
    return table


def load_experimental(path: Path, column: str) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    header = rows[0]
    key = "ligand_name" if "ligand_name" in header else ("compound" if "compound" in header else None)
    if key is None:
        raise ValueError(f"{path} has no ligand_name/compound column")
    values: dict[str, float] = {}
    for row in rows:
        raw = row.get(column, "").strip()
        if raw not in ("", "nan"):
            values[row[key]] = float(raw)
        elif row.get("experimental_pKD", "").strip():
            # dG = -RT ln(10) * pKD ~= -1.364 * pKD kcal/mol at 298 K
            values[row[key]] = -1.364 * float(row["experimental_pKD"])
    return values


def main() -> None:
    opt = options()
    opt.output_dir.mkdir(parents=True, exist_ok=True)
    our_edges, our_ligands = load_network_edges(opt.network)
    rowan = load_rowan_edges(opt.rowan_edges, opt.rowan_edge_column)

    edge_rows = []
    ours, ref = [], []
    for (a, b), data in sorted(our_edges.items()):
        match = rowan.get(frozenset((a, b)))
        if match is None:
            edge_rows.append({"state_a": a, "state_b": b, "our_ddg": data["ddg"],
                              "rowan_ddg": "", "difference": "", "note": "no_rowan_edge"})
            continue
        ra, rb, rvalue = match
        # Orient Rowan's DDG to our (a->b) direction.
        oriented = rvalue if (ra, rb) == (a, b) else -rvalue
        edge_rows.append({"state_a": a, "state_b": b, "our_ddg": round(data["ddg"], 4),
                          "our_err": round(data["err"], 4), "rowan_ddg": round(oriented, 4),
                          "difference": round(data["ddg"] - oriented, 4), "note": "matched"})
        ours.append(data["ddg"]); ref.append(oriented)

    summary = {"rowan_edge_column": opt.rowan_edge_column,
               "edges_compared": len(ours),
               "edges_in_our_network": len(our_edges),
               "edge_metrics": metrics(np.array(ours), np.array(ref)) if ours else None}

    ligand_rows = []
    if opt.experimental is not None:
        experimental = load_experimental(opt.experimental, opt.experimental_column)
        common = sorted(set(our_ligands) & set(experimental))
        if len(common) >= 2:
            ours_dg = np.array([our_ligands[l] for l in common])
            exp_dg = np.array([experimental[l] for l in common])
            # Mean-centre both (RBFE gives relative dG only).
            ours_c = ours_dg - ours_dg.mean()
            exp_c = exp_dg - exp_dg.mean()
            for l, o, e in zip(common, ours_c, exp_c):
                ligand_rows.append({"ligand": l, "our_dg_centered": round(float(o), 4),
                                    "experimental_dg_centered": round(float(e), 4),
                                    "difference": round(float(o - e), 4)})
            summary["ligand_metrics"] = metrics(ours_c, exp_c)
            summary["ligands_compared"] = len(common)
        else:
            summary["ligand_metrics"] = None
            summary["ligands_compared"] = len(common)

    edge_csv = opt.output_dir / "comparison_edges.csv"
    with edge_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state_a", "state_b", "our_ddg", "our_err",
                                                    "rowan_ddg", "difference", "note"])
        writer.writeheader()
        writer.writerows(edge_rows)
    if ligand_rows:
        with (opt.output_dir / "comparison_ligands.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ligand_rows[0]))
            writer.writeheader(); writer.writerows(ligand_rows)
    (opt.output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"COMPARISON={opt.output_dir}", flush=True)


if __name__ == "__main__":
    main()
