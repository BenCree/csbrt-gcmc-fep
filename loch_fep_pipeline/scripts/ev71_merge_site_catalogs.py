#!/usr/bin/env python3
"""Merge receptor-aligned provisional hydration sites into a common catalog."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from pipeline_utils import require_file, write_json_atomic


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--merge-cutoff", type=float, default=2.4)
    parser.add_argument("--minimum-support", type=int, default=1)
    return parser.parse_args()


def read_catalog(path: Path) -> tuple[list[str], np.ndarray]:
    with require_file(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"site_id", "x_angstrom", "y_angstrom", "z_angstrom"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}")
    identifiers = [row["site_id"] for row in rows]
    coordinates = np.asarray(
        [[float(row[key]) for key in ("x_angstrom", "y_angstrom", "z_angstrom")] for row in rows],
        dtype=np.float64,
    )
    if len(set(identifiers)) != len(identifiers) or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid site identifiers or coordinates in {path}")
    return identifiers, coordinates


def main() -> None:
    opt = options()
    if opt.merge_cutoff <= 0 or opt.minimum_support < 1:
        raise ValueError("Merge cutoff and minimum support must be positive")
    resolved = [require_file(path) for path in opt.catalogs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same catalog was supplied more than once")

    coordinates: list[np.ndarray] = []
    sources: list[int] = []
    for source_index, path in enumerate(resolved):
        _, xyz = read_catalog(path)
        coordinates.extend(xyz)
        sources.extend([source_index] * len(xyz))
    xyz = np.asarray(coordinates, dtype=np.float64)
    source_array = np.asarray(sources, dtype=np.int32)
    if not len(xyz):
        raise ValueError("No sites were read")

    if len(xyz) == 1:
        cluster_ids = np.ones(1, dtype=np.int32)
    else:
        tree = hierarchy.linkage(pdist(xyz), method="complete")
        cluster_ids = hierarchy.fcluster(tree, t=opt.merge_cutoff, criterion="distance")

    clusters: list[dict[str, object]] = []
    for cluster_id in np.unique(cluster_ids):
        members = np.flatnonzero(cluster_ids == cluster_id)
        source_ids = sorted(set(int(value) for value in source_array[members]))
        if len(source_ids) < opt.minimum_support:
            continue
        centre = xyz[members].mean(axis=0)
        clusters.append(
            {
                "coordinate": centre,
                "support": len(source_ids),
                "observations": len(members),
                "sources": [
                    f"{resolved[index].parent.parent.name}/{resolved[index].name}"
                    for index in source_ids
                ],
            }
        )
    clusters.sort(
        key=lambda item: (
            -int(item["support"]),
            -int(item["observations"]),
            *np.asarray(item["coordinate"]).tolist(),
        )
    )

    output = opt.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "site_id",
                "x_angstrom",
                "y_angstrom",
                "z_angstrom",
                "catalog_support",
                "catalog_fraction",
                "source_site_observations",
                "source_catalogs",
            )
        )
        for index, cluster in enumerate(clusters, start=1):
            coordinate = np.asarray(cluster["coordinate"])
            writer.writerow(
                (
                    f"HS{index:03d}",
                    *[f"{value:.6f}" for value in coordinate],
                    cluster["support"],
                    f"{int(cluster['support']) / len(resolved):.8f}",
                    cluster["observations"],
                    ";".join(cluster["sources"]),
                )
            )

    summary_path = (opt.summary or output.with_suffix(".json")).resolve()
    summary = {
        "status": "completed",
        "input_catalogs": [str(path) for path in resolved],
        "input_catalog_count": len(resolved),
        "input_sites": len(xyz),
        "merge_method": "complete_linkage",
        "merge_cutoff_angstrom": opt.merge_cutoff,
        "minimum_catalog_support": opt.minimum_support,
        "common_sites": len(clusters),
        "output_catalog": str(output),
        "instruction": (
            "Pass this exact CSV as --site-catalog to ev71_density_sites.py for "
            "every ligand and replica. Inputs must share one alignment reference."
        ),
    }
    write_json_atomic(summary_path, summary)
    print(f"Wrote {len(clusters)} common sites to {output}", flush=True)
    print(f"RESULT={summary_path}", flush=True)


if __name__ == "__main__":
    main()
