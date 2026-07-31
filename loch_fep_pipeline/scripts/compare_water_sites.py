#!/usr/bin/env python3
"""Compare hydration-site sets produced by GCMC/MD clustering.

Consumes the ``<prefix>-lig-clusts.pdb`` files written by ``ev71_postprocess.py``
(or by GRAND's ``cluster_waters``): one record per site, occupancy in the
occupancy column, ordered by descending occupancy. The clustering itself is not
reimplemented here -- this only compares the resulting site sets.

Two sites are considered the same hydration site when they lie within a distance
cutoff of one another; matching is one-to-one and greedy from the closest pair
outwards, so a single site cannot absorb several of its counterparts.

Reported per pair, following the metrics used in the original study:

    recall     = matched / |reference|      does it find the known sites?
    precision  = matched / |test|           how much else does it predict?
    Tanimoto   = matched / (|ref| + |test| - matched)

Site coordinates are only comparable inside one frame. A set derived from a
different structure must be brought into the target frame first (see
``gci_map_centre.py``); this script checks for the mismatch rather than assuming
it away.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from pipeline_utils import require_file, sha256, write_json_atomic

DEFAULT_MATCH_CUTOFF = 2.0     # Angstrom, the study's matching distance
DEFAULT_MIN_OCCUPANCY = 0.2    # the study's threshold for a persistent site
FRAME_WARNING_ANGSTROM = 25.0


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites",
        nargs=2,
        action="append",
        metavar=("LABEL", "PDB"),
        required=True,
        help="A labelled site set. Give at least two; repeatable.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Label to treat as the reference in recall/precision. Defaults to "
        "the first --sites entry.",
    )
    parser.add_argument(
        "--match-cutoff", type=float, default=DEFAULT_MATCH_CUTOFF,
        help=f"Distance for two sites to be the same (default {DEFAULT_MATCH_CUTOFF} A).",
    )
    parser.add_argument(
        "--min-occupancy", type=float, default=DEFAULT_MIN_OCCUPANCY,
        help=f"Discard sites below this occupancy (default {DEFAULT_MIN_OCCUPANCY}). "
        "Use 0 to keep every cluster.",
    )
    parser.add_argument(
        "--centre", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
        help="Restrict to sites within --radius of this point, e.g. a binding site.",
    )
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="water_sites")
    parser.add_argument("--plots", action="store_true", help="Write diagnostic figures.")
    parser.add_argument(
        "--ligand-reference", type=Path, default=None,
        help="PDB containing the ligand; enables the distance-to-ligand panel.",
    )
    parser.add_argument("--ligand-resname", default="LIG")
    return parser.parse_args()


def read_ligand(path: Path, resname: str) -> np.ndarray:
    coordinates = [
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        for line in require_file(path).read_text().splitlines()
        if line[:6].strip() in ("ATOM", "HETATM") and line[17:20].strip() == resname
    ]
    if not coordinates:
        raise ValueError(f"No {resname} atoms in {path}")
    return np.asarray(coordinates, dtype=float)


def threshold_sweep(
    a_xyz: np.ndarray, a_occ: np.ndarray,
    b_xyz: np.ndarray, b_occ: np.ndarray,
    cutoff: float, thresholds: np.ndarray,
) -> list[dict]:
    """Agreement as a function of the occupancy cut.

    Transient one-frame clusters dominate raw cluster counts, so a single
    threshold can hide how much of the disagreement is confined to them.
    """
    rows = []
    for threshold in thresholds:
        a = a_xyz[a_occ >= threshold]
        b = b_xyz[b_occ >= threshold]
        if a.size == 0 or b.size == 0:
            continue
        pairs, _ = match_sites(a, b, cutoff)
        matched = len(pairs)
        union = len(a) + len(b) - matched
        rows.append({
            "threshold": float(threshold),
            "n_a": int(len(a)), "n_b": int(len(b)), "matched": matched,
            "recall": matched / len(a), "precision": matched / len(b),
            "tanimoto": matched / union if union else float("nan"),
        })
    return rows


def make_plots(
    output: Path, prefix: str, sets: dict, labels: list[str],
    cutoff: float, min_occupancy: float,
    raw: dict, sweep: list[dict], ligand: np.ndarray | None,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_label, b_label = labels[0], labels[1]
    paths: list[Path] = []

    # 1. occupancy rank curves over ALL clusters, before any threshold
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for label in labels:
        occ = np.sort(raw[label]["occupancies"])[::-1]
        axes[0].plot(np.arange(1, occ.size + 1), occ, lw=1.6, label=f"{label} (n={occ.size})")
    axes[0].axhline(min_occupancy, ls="--", lw=1, color="grey")
    axes[0].annotate(f"occupancy cut = {min_occupancy:g}", xy=(1, min_occupancy),
                     xytext=(4, 4), textcoords="offset points", fontsize=8, color="grey")
    axes[0].set_xlabel("site rank"); axes[0].set_ylabel("occupancy")
    axes[0].set_title("All clusters, ranked"); axes[0].legend(fontsize=8)

    bins = np.linspace(0, 1, 21)
    for label in labels:
        axes[1].hist(raw[label]["occupancies"], bins=bins, alpha=0.55, label=label)
    axes[1].axvline(min_occupancy, ls="--", lw=1, color="grey")
    axes[1].set_xlabel("occupancy"); axes[1].set_ylabel("clusters")
    axes[1].set_yscale("log"); axes[1].set_title("Occupancy distribution")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    path = output / f"{prefix}_occupancy.png"; figure.savefig(path, dpi=160); plt.close(figure)
    paths.append(path)

    # 2. matched-distance histogram + threshold sweep
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    pairs, _ = match_sites(sets[a_label]["coordinates"], sets[b_label]["coordinates"], cutoff)
    distances = [pair[2] for pair in pairs]
    if distances:
        axes[0].hist(distances, bins=np.linspace(0, cutoff, 21), color="steelblue")
        axes[0].axvline(float(np.mean(distances)), color="crimson", lw=1.5,
                        label=f"mean {np.mean(distances):.2f} A")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("distance between matched sites (A)")
    axes[0].set_ylabel("pairs")
    axes[0].set_title(f"Matched sites ({len(pairs)} pairs, occ >= {min_occupancy:g})")

    if sweep:
        t = [row["threshold"] for row in sweep]
        axes[1].plot(t, [r["recall"] for r in sweep], "o-", ms=3, label="recall")
        axes[1].plot(t, [r["precision"] for r in sweep], "s-", ms=3, label="precision")
        axes[1].plot(t, [r["tanimoto"] for r in sweep], "^-", ms=3, label="Tanimoto")
        axes[1].set_xlabel("occupancy threshold"); axes[1].set_ylabel("metric")
        axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=8)
        axes[1].set_title("Agreement vs occupancy cut")
    figure.tight_layout()
    path = output / f"{prefix}_agreement.png"; figure.savefig(path, dpi=160); plt.close(figure)
    paths.append(path)

    # 3. where the disagreeing clusters are, relative to the ligand
    if ligand is not None:
        a_all, b_all = raw[a_label], raw[b_label]
        pairs_all, _ = match_sites(a_all["coordinates"], b_all["coordinates"], cutoff)
        matched_a = {p[0] for p in pairs_all}
        matched_b = {p[1] for p in pairs_all}
        only_a = [i for i in range(len(a_all["coordinates"])) if i not in matched_a]
        only_b = [i for i in range(len(b_all["coordinates"])) if i not in matched_b]

        def to_ligand(xyz):
            return np.linalg.norm(xyz[:, None, :] - ligand[None, :, :], axis=2).min(axis=1)

        groups = [
            (f"matched ({len(matched_a)})", a_all["coordinates"][sorted(matched_a)],
             a_all["occupancies"][sorted(matched_a)], "tab:green"),
            (f"{a_label} only ({len(only_a)})", a_all["coordinates"][only_a],
             a_all["occupancies"][only_a], "tab:red"),
            (f"{b_label} only ({len(only_b)})", b_all["coordinates"][only_b],
             b_all["occupancies"][only_b], "tab:blue"),
        ]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for name, xyz, occ, colour in groups:
            if len(xyz) == 0:
                continue
            axes[0].scatter(to_ligand(xyz), occ, s=14, alpha=0.6, color=colour, label=name)
        axes[0].axhline(min_occupancy, ls="--", lw=1, color="grey")
        axes[0].set_xlabel("distance to nearest ligand atom (A)")
        axes[0].set_ylabel("occupancy")
        axes[0].set_title("Where the two methods disagree")
        axes[0].legend(fontsize=8)

        data = [occ for _, xyz, occ, _ in groups if len(xyz)]
        names = [name for name, xyz, _, _ in groups if len(xyz)]
        axes[1].boxplot(data, tick_labels=names, showfliers=False)
        axes[1].axhline(min_occupancy, ls="--", lw=1, color="grey")
        axes[1].set_ylabel("occupancy")
        axes[1].set_title("Occupancy by agreement group")
        axes[1].tick_params(axis="x", labelsize=8)
        figure.tight_layout()
        path = output / f"{prefix}_disagreement.png"
        figure.savefig(path, dpi=160); plt.close(figure)
        paths.append(path)
    return paths


def read_sites(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (coordinates in Angstrom, occupancies) from a cluster PDB."""
    coordinates, occupancies = [], []
    for line in require_file(path).read_text().splitlines():
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        coordinates.append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
        field = line[54:60].strip()
        occupancies.append(float(field) if field else 1.0)
    if not coordinates:
        raise ValueError(f"No site records in {path}")
    return np.asarray(coordinates, dtype=float), np.asarray(occupancies, dtype=float)


def filter_sites(
    coordinates: np.ndarray,
    occupancies: np.ndarray,
    *,
    min_occupancy: float,
    centre: list[float] | None,
    radius: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mask = occupancies >= float(min_occupancy)
    dropped_occupancy = int((~mask).sum())
    dropped_region = 0
    if centre is not None:
        if radius is None:
            raise ValueError("--centre requires --radius")
        distance = np.linalg.norm(coordinates - np.asarray(centre, dtype=float), axis=1)
        region = distance <= float(radius)
        dropped_region = int((mask & ~region).sum())
        mask = mask & region
    return (
        coordinates[mask],
        occupancies[mask],
        {
            "input_sites": int(coordinates.shape[0]),
            "dropped_below_occupancy": dropped_occupancy,
            "dropped_outside_region": dropped_region,
            "retained_sites": int(mask.sum()),
        },
    )


def match_sites(
    a: np.ndarray, b: np.ndarray, cutoff: float
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Greedy one-to-one matching, closest pair first.

    Greedy-closest-first is used rather than nearest-neighbour-per-site because
    the latter is not symmetric and lets one site claim several counterparts,
    which inflates the match count.
    """
    if a.size == 0 or b.size == 0:
        return [], np.empty(0)
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    pairs = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(distances, axis=None), distances.shape))[0]
    for index_a, index_b in order:
        distance = float(distances[index_a, index_b])
        if distance > cutoff:
            break
        if int(index_a) in used_a or int(index_b) in used_b:
            continue
        used_a.add(int(index_a))
        used_b.add(int(index_b))
        pairs.append((int(index_a), int(index_b), distance))
    return pairs, distances


def compare(label_a: str, set_a: dict, label_b: str, set_b: dict, cutoff: float) -> dict:
    a, b = set_a["coordinates"], set_b["coordinates"]
    separation = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))) if a.size and b.size else float("nan")
    pairs, _ = match_sites(a, b, cutoff)
    matched = len(pairs)
    n_a, n_b = int(a.shape[0]), int(b.shape[0])
    union = n_a + n_b - matched
    distances = [pair[2] for pair in pairs]
    return {
        "reference": label_a,
        "test": label_b,
        "reference_sites": n_a,
        "test_sites": n_b,
        "matched": matched,
        "recall": matched / n_a if n_a else float("nan"),
        "precision": matched / n_b if n_b else float("nan"),
        "tanimoto": matched / union if union else float("nan"),
        "mean_matched_distance_angstrom": float(np.mean(distances)) if distances else None,
        "max_matched_distance_angstrom": float(np.max(distances)) if distances else None,
        "centroid_separation_angstrom": separation,
    }


def main() -> None:
    opt = options()
    if len(opt.sites) < 2:
        raise ValueError("Give at least two --sites entries to compare")
    labels = [label for label, _ in opt.sites]
    if len(set(labels)) != len(labels):
        raise ValueError("Site labels must be unique")

    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sets: dict[str, dict] = {}
    raw_sets: dict[str, dict] = {}
    print(f"match cutoff {opt.match_cutoff:g} A | min occupancy {opt.min_occupancy:g}")
    for label, path_string in opt.sites:
        path = Path(path_string)
        coordinates, occupancies = read_sites(path)
        kept_xyz, kept_occ, audit = filter_sites(
            coordinates,
            occupancies,
            min_occupancy=opt.min_occupancy,
            centre=opt.centre,
            radius=opt.radius,
        )
        if kept_xyz.size == 0:
            raise ValueError(f"{label}: no sites survived filtering")
        raw_sets[label] = {"coordinates": coordinates, "occupancies": occupancies}
        sets[label] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "coordinates": kept_xyz,
            "occupancies": kept_occ,
            "audit": audit,
            "mean_occupancy": float(kept_occ.mean()),
        }
        print(
            f"  {label:14s} {audit['retained_sites']:4d} sites "
            f"(of {audit['input_sites']}) mean occupancy {kept_occ.mean():.2f}"
        )

    reference_label = opt.reference or labels[0]
    if reference_label not in sets:
        raise ValueError(f"--reference {reference_label!r} is not one of {labels}")

    comparisons = []
    for label_a, label_b in combinations(labels, 2):
        # Report each pair in both directions: recall and precision swap, and
        # which set is "reference" is a choice, not a property of the data.
        comparisons.append(
            compare(label_a, sets[label_a], label_b, sets[label_b], opt.match_cutoff)
        )
        comparisons.append(
            compare(label_b, sets[label_b], label_a, sets[label_a], opt.match_cutoff)
        )

    frame_warnings = []
    print()
    print(
        f"{'reference':>14} {'test':>14} {'matched':>8} {'recall':>7} "
        f"{'prec.':>7} {'tanim.':>7} {'mean d':>7}"
    )
    for row in comparisons:
        mean_d = row["mean_matched_distance_angstrom"]
        print(
            f"{row['reference']:>14} {row['test']:>14} {row['matched']:>8} "
            f"{row['recall']:>7.3f} {row['precision']:>7.3f} {row['tanimoto']:>7.3f} "
            f"{(f'{mean_d:.2f}' if mean_d is not None else '-'):>7}"
        )
        if row["centroid_separation_angstrom"] > FRAME_WARNING_ANGSTROM:
            frame_warnings.append((row["reference"], row["test"], row["centroid_separation_angstrom"]))

    if frame_warnings:
        print()
        for label_a, label_b, separation in sorted(set(frame_warnings)):
            print(
                f"WARNING: {label_a} and {label_b} have site centroids {separation:.1f} A "
                "apart. Site coordinates are only comparable within one frame -- these "
                "sets are almost certainly from different structures. Map them first "
                "(gci_map_centre.py) or the comparison is meaningless."
            )

    sweep = []
    plot_paths = []
    if len(labels) == 2:
        a, b = labels
        sweep = threshold_sweep(
            raw_sets[a]["coordinates"], raw_sets[a]["occupancies"],
            raw_sets[b]["coordinates"], raw_sets[b]["occupancies"],
            opt.match_cutoff, np.arange(0.0, 0.95, 0.05),
        )
    if opt.plots:
        ligand = (read_ligand(opt.ligand_reference, opt.ligand_resname)
                  if opt.ligand_reference else None)
        plot_paths = make_plots(output, opt.prefix, sets, labels, opt.match_cutoff,
                                opt.min_occupancy, raw_sets, sweep, ligand)
        for path in plot_paths:
            print(f"  figure {path.name}")

    csv_path = output / f"{opt.prefix}_site_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    summary = {
        "stage": "compare_water_sites",
        "match_cutoff_angstrom": opt.match_cutoff,
        "min_occupancy": opt.min_occupancy,
        "region_centre": opt.centre,
        "region_radius_angstrom": opt.radius,
        "reference_label": reference_label,
        "site_sets": {
            label: {
                "path": data["path"],
                "sha256": data["sha256"],
                "mean_occupancy": data["mean_occupancy"],
                **data["audit"],
            }
            for label, data in sets.items()
        },
        "comparisons": comparisons,
        "threshold_sweep": sweep,
        "frame_warnings": [
            {"a": a, "b": b, "centroid_separation_angstrom": d}
            for a, b, d in sorted(set(frame_warnings))
        ],
    }
    json_path = output / f"{opt.prefix}_site_comparison.json"
    write_json_atomic(json_path, summary)
    print(f"\nwrote {csv_path.name} and {json_path.name}")


if __name__ == "__main__":
    main()
