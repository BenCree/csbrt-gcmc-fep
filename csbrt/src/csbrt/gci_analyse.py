#!/usr/bin/env python3
"""Pool GCI titration windows and integrate them into a binding free energy.

Implements the discrete Grand Canonical Integration of Ross et al. (JACS 2015,
eq. 19), following the original CRY1 analysis: no sigmoid is fitted, the
integration runs directly on the measured points.

    dF_bind(N) = kT * [ N*B_N - ln(N!) - integral(-inf .. B_N) <N(B)> dB ]
                 - N * mu_hyd

with B_N obtained by inverse interpolation of the titration curve and
dF_bind(0) = 0 by convention.

Everything that describes the simulation -- sphere radius, standard volume, kT,
the checkpoint interval in GCMC attempts -- is read from each window's
``_titration.json`` rather than hardcoded here. The original notebook hardcoded
the radius (7 A while its data was 4 A) and the checkpoint interval (200 while
the real value was 400); reading them removes both errors by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from pipeline_utils import (
    checkpoint_matches,
    complete_checkpoint,
    implementation_signature,
    invalidate_checkpoint,
    read_json,
    require_file,
    sha256,
    validate_recorded_outputs,
    write_json_atomic,
)

# Fields that must agree across windows: they define the physical experiment, so
# a difference means the windows are not the same titration.
CONSISTENT_FIELDS = (
    "input_prmtop_sha256",
    "input_rst7_sha256",
    "radius_angstrom",
    "standard_volume_angstrom3",
    "adams_shift",
    "bulk_sampling_probability",
    "temperature_K",
    "sphere_centre_angstrom",
    "reference",
    "num_ghost_waters",
    "cycles",
    "attempts_per_cycle",
    "md_steps_per_cycle",
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--equilibrated-fraction",
        type=float,
        default=0.5,
        help="Fraction of each window used for the occupancy average (default 0.5).",
    )
    group.add_argument(
        "--equilibrated-moves",
        type=int,
        default=None,
        help="Use the last N GCMC attempts of each window instead of a fraction.",
    )
    parser.add_argument(
        "--mu-hydration",
        type=float,
        default=None,
        help="Excess chemical potential of bulk water (kcal/mol). Defaults to the "
        "value recorded by the windows.",
    )
    parser.add_argument(
        "--exclude-window-index",
        type=int,
        action="append",
        default=[],
        help="Drop a window from the curve. Repeatable.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Largest N to evaluate. Defaults to round(<N> at B_equil) + 1.",
    )
    parser.add_argument(
        "--monotonic-fit",
        choices=("none", "isotonic"),
        default="none",
        help="'none' (default) rejects a non-monotonic curve. 'isotonic' fits the "
        "least-squares non-decreasing curve first; use for under-converged data "
        "and label the result as preliminary.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reading windows
# ---------------------------------------------------------------------------
def read_windows(run_root: Path, prefix: str) -> list[dict]:
    """Return one record per window, refusing to read unverified output."""
    directories = sorted(p for p in run_root.iterdir() if p.is_dir())
    windows = []
    for directory in directories:
        marker = directory / "gci_window.complete.json"
        if not marker.is_file():
            print(f"  skipping {directory.name}: no completed checkpoint", flush=True)
            continue
        # Re-hash every recorded output before trusting the numbers.
        validate_recorded_outputs(marker)
        metadata = read_json(directory / f"{prefix}_titration.json")
        with require_file(directory / f"{prefix}_titration.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No titration rows in {directory}")
        windows.append(
            {
                "directory": directory,
                "marker_sha256": sha256(marker),
                "metadata": metadata,
                "steps": [int(row["step"]) for row in rows],
                "occupancy": [int(row["sphere_waters"]) for row in rows],
                "ghost_pool": [int(row["ghost_pool"]) for row in rows],
            }
        )
    if not windows:
        raise ValueError(f"No completed GCI windows found under {run_root}")
    return windows


def require_consistent_windows(windows: list[dict]) -> dict:
    """Require every window to be the same experiment at a different B."""
    reference = windows[0]["metadata"]
    for window in windows[1:]:
        for field in CONSISTENT_FIELDS:
            if window["metadata"].get(field) != reference.get(field):
                raise ValueError(
                    f"Window {window['directory'].name} disagrees with "
                    f"{windows[0]['directory'].name} on {field!r}: "
                    f"{window['metadata'].get(field)!r} vs {reference.get(field)!r}. "
                    "These windows are not the same titration."
                )
    targets = [window["metadata"]["target_b"] for window in windows]
    if len(set(targets)) != len(targets):
        raise ValueError("Two windows share the same target B")
    return {field: reference.get(field) for field in CONSISTENT_FIELDS}


def tail_statistics(window: dict, *, n_tail: int) -> dict:
    occupancy = np.asarray(window["occupancy"], dtype=float)
    if n_tail < 1 or n_tail > occupancy.size:
        raise ValueError(
            f"{window['directory'].name}: cannot average the last {n_tail} of "
            f"{occupancy.size} checkpoints"
        )
    tail = occupancy[-n_tail:]
    mean = float(tail.mean())
    # Population standard deviation, matching the original analysis. This is a
    # spread, not a converged statistical error: successive checkpoints are
    # correlated, so the true uncertainty is larger.
    std = float(tail.std())
    return {
        "mean_N": mean,
        "std_N": std,
        "sem_N": float(std / math.sqrt(n_tail)),
        "n_checkpoints": int(n_tail),
    }


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------
def isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit (pool adjacent violators).

    <N(B)> is non-decreasing in B on physical grounds, so when noise inverts a
    pair this returns the closest curve that is not inverted. It is the
    maximum-likelihood monotone fit under Gaussian error, not a smoothing hack,
    but it cannot manufacture information: on badly under-sampled data it will
    flatten whole stretches of the curve.
    """
    blocks: list[list[float]] = []  # [value, weight, count]
    for value in values:
        blocks.append([float(value), 1.0, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            value_b, weight_b, count_b = blocks.pop()
            value_a, weight_a, count_a = blocks.pop()
            merged = (value_a * weight_a + value_b * weight_b) / (weight_a + weight_b)
            blocks.append([merged, weight_a + weight_b, count_a + count_b])
    fitted: list[float] = []
    for value, _, count in blocks:
        fitted.extend([value] * count)
    return np.asarray(fitted, dtype=float)


def monotonicity_violations(b: np.ndarray, n: np.ndarray) -> list[tuple]:
    return [
        (float(b[i]), float(n[i]), float(b[i + 1]), float(n[i + 1]))
        for i in range(len(n) - 1)
        if n[i + 1] < n[i] - 1.0e-12
    ]


# ---------------------------------------------------------------------------
# Ross et al. 2015 integration
# ---------------------------------------------------------------------------
def integrate_titration(
    b_points: np.ndarray, n_points: np.ndarray, b_upper: float
) -> tuple[float, bool]:
    """Integrate <N(B)> dB from -inf to ``b_upper`` by the trapezoid rule.

    Below the lowest sampled B the integrand is taken as zero; above the highest
    it is linearly extrapolated from the last two points and clipped at zero.
    """
    b_min, b_max = float(b_points[0]), float(b_points[-1])
    extrapolated = False
    if b_upper <= b_min:
        return 0.0, extrapolated
    if b_upper <= b_max:
        n_at_upper = float(np.interp(b_upper, b_points, n_points))
        mask = b_points < b_upper
        b_int = np.concatenate([b_points[mask], [b_upper]])
        n_int = np.concatenate([n_points[mask], [n_at_upper]])
    else:
        extrapolated = True
        slope = (
            (n_points[-1] - n_points[-2]) / (b_points[-1] - b_points[-2])
            if len(b_points) >= 2
            else 0.0
        )
        n_at_upper = max(float(n_points[-1] + slope * (b_upper - b_max)), 0.0)
        b_int = np.concatenate([b_points, [b_upper]])
        n_int = np.concatenate([n_points, [n_at_upper]])
    return float(np.trapezoid(np.clip(n_int, 0.0, None), b_int)), extrapolated


def compute_df_bind(
    n: int,
    b_points: np.ndarray,
    n_points: np.ndarray,
    kt: float,
    mu_hydration: float,
) -> tuple[float, bool]:
    """Return (dF_bind(N), extrapolated) from Ross et al. 2015 eq. 19."""
    if n == 0:
        return 0.0, False
    n_min, n_max = float(n_points[0]), float(n_points[-1])
    extrapolated = False
    if n_min <= n <= n_max:
        b_n = float(np.interp(float(n), n_points, b_points))
    else:
        extrapolated = True
        if len(b_points) < 2:
            b_n = float(b_points[-1])
        elif n > n_max:
            denominator = n_points[-1] - n_points[-2]
            slope = (b_points[-1] - b_points[-2]) / denominator if denominator else 0.0
            b_n = float(b_points[-1] + slope * (n - n_max))
        else:
            denominator = n_points[1] - n_points[0]
            slope = (b_points[1] - b_points[0]) / denominator if denominator else 0.0
            b_n = float(b_points[0] + slope * (n - n_min))
    integral, integral_extrapolated = integrate_titration(b_points, n_points, b_n)
    df = kt * (n * b_n - math.lgamma(n + 1) - integral) - n * mu_hydration
    return float(df), bool(extrapolated or integral_extrapolated)


# ---------------------------------------------------------------------------
def make_plots(
    output_dir: Path,
    prefix: str,
    curve: list[dict],
    fitted_n: np.ndarray,
    b_equil: float,
    free_energy: list[dict],
    n_star: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = [point for point in curve if not point["excluded"]]
    b = [point["target_b"] for point in kept]
    n = [point["mean_N"] for point in kept]
    err = [point["sem_N"] for point in kept]

    paths = []
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.errorbar(b, n, yerr=err, fmt="o", ms=4, lw=1, capsize=2, label="measured")
    if fitted_n is not None:
        axis.plot(b, fitted_n, "-", lw=1.5, label="monotonic fit")
    axis.axvline(b_equil, ls="--", lw=1, color="grey")
    axis.annotate(
        f"$B_{{equil}}$ = {b_equil:.2f}",
        xy=(b_equil, max(n) if n else 1),
        xytext=(4, -4),
        textcoords="offset points",
        fontsize=8,
        color="grey",
    )
    axis.set_xlabel("Adams value $B$")
    axis.set_ylabel(r"$\langle N \rangle$")
    axis.set_title("GCI titration curve")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = output_dir / f"{prefix}_titration_curve.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    ns = [point["n"] for point in free_energy]
    dfs = [point["df_bind_kcal_per_mol"] for point in free_energy]
    axis.plot(ns, dfs, "o-", ms=4, lw=1.2)
    axis.axhline(0.0, lw=0.8, color="grey")
    axis.plot([n_star], [dfs[ns.index(n_star)]], "s", ms=9, mfc="none", mec="crimson")
    axis.annotate(
        f"$N^*$ = {n_star}",
        xy=(n_star, dfs[ns.index(n_star)]),
        xytext=(6, 6),
        textcoords="offset points",
        color="crimson",
        fontsize=9,
    )
    axis.set_xlabel("$N$ waters")
    axis.set_ylabel(r"$\Delta F_{bind}$ (kcal/mol)")
    axis.set_title("Binding free energy of the water network")
    figure.tight_layout()
    path = output_dir / f"{prefix}_gci_free_energy.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def main() -> None:
    opt = options()
    run_root = opt.run_root.resolve()
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    windows = read_windows(run_root, opt.prefix)
    shared = require_consistent_windows(windows)
    reference = windows[0]["metadata"]

    radius = float(reference["radius_angstrom"])
    standard_volume = float(reference["standard_volume_angstrom3"])
    kt = float(reference["kt_kcal_per_mol"])
    checkpoint_interval = int(reference["attempts_per_cycle"])
    mu_hydration = (
        float(opt.mu_hydration)
        if opt.mu_hydration is not None
        else float(reference["mu_hydration_kcal_per_mol"])
    )
    b_equil = mu_hydration / kt + math.log(
        (4.0 * math.pi * radius**3 / 3.0) / standard_volume
    ) + float(reference["adams_shift"])

    # How much of each window counts as equilibrated, expressed in checkpoints.
    n_checkpoints = len(windows[0]["occupancy"])
    if opt.equilibrated_moves is not None:
        n_tail = int(opt.equilibrated_moves) // checkpoint_interval
        if n_tail < 1:
            raise ValueError(
                f"--equilibrated-moves {opt.equilibrated_moves} is less than one "
                f"checkpoint interval ({checkpoint_interval} attempts)"
            )
    else:
        if not 0.0 < opt.equilibrated_fraction <= 1.0:
            raise ValueError("--equilibrated-fraction must be in (0, 1]")
        n_tail = max(1, int(round(n_checkpoints * opt.equilibrated_fraction)))
    n_tail = min(n_tail, n_checkpoints)

    print(
        f"{len(windows)} windows | radius {radius:g} A | kT {kt:.6f} kcal/mol | "
        f"B_equil {b_equil:.4f}",
        flush=True,
    )
    print(
        f"averaging the last {n_tail} of {n_checkpoints} checkpoints "
        f"({n_tail * checkpoint_interval} GCMC attempts)",
        flush=True,
    )

    excluded = set(int(index) for index in opt.exclude_window_index)
    curve = []
    for window in windows:
        metadata = window["metadata"]
        statistics = tail_statistics(window, n_tail=n_tail)
        curve.append(
            {
                "window_index": int(metadata["window_index"]),
                "directory": window["directory"].name,
                "target_b": float(metadata["target_b"]),
                "mu_kcal_per_mol": float(metadata["mu_kcal_per_mol"]),
                "minimum_ghost_pool": int(metadata["minimum_ghost_pool"]),
                "excluded": int(metadata["window_index"]) in excluded,
                **statistics,
            }
        )
    curve.sort(key=lambda point: point["target_b"])

    kept = [point for point in curve if not point["excluded"]]
    if len(kept) < 3:
        raise ValueError(f"Only {len(kept)} usable windows; need at least 3 to integrate")
    b_points = np.array([point["target_b"] for point in kept], dtype=float)
    measured_n = np.array([point["mean_N"] for point in kept], dtype=float)

    violations = monotonicity_violations(b_points, measured_n)
    fitted_n = measured_n
    if violations:
        if opt.monotonic_fit == "none":
            worst = violations[:4]
            raise ValueError(
                f"The titration curve is not monotonic in B: {len(violations)} "
                f"inversions, e.g. "
                + "; ".join(
                    f"<N>({a:.3f})={na:.2f} > <N>({bb:.3f})={nb:.2f}"
                    for a, na, bb, nb in worst
                )
                + ". <N> must be non-decreasing in B for the inverse interpolation "
                "that gives B_N to be meaningful. Sample longer, drop outliers with "
                "--exclude-window-index, or use --monotonic-fit isotonic and treat "
                "the result as preliminary."
            )
        fitted_n = isotonic_non_decreasing(measured_n)
        print(
            f"WARNING: {len(violations)} monotonicity inversions; applied an "
            "isotonic fit. Results are preliminary.",
            flush=True,
        )

    n_at_equil = float(np.interp(b_equil, b_points, fitted_n))
    max_n = int(opt.max_n) if opt.max_n is not None else max(1, int(round(n_at_equil))) + 1

    free_energy = []
    for n in range(0, max_n + 1):
        df, extrapolated = compute_df_bind(n, b_points, fitted_n, kt, mu_hydration)
        free_energy.append(
            {"n": n, "df_bind_kcal_per_mol": df, "extrapolated": bool(extrapolated)}
        )
    values = [point["df_bind_kcal_per_mol"] for point in free_energy]
    best = int(np.argmin(values))
    n_star = free_energy[best]["n"]
    df_star = values[best]

    print(f"<N> at B_equil = {n_at_equil:.3f}")
    print(f"N* = {n_star}   dF_bind(N*) = {df_star:+.3f} kcal/mol", flush=True)

    curve_csv = output / f"{opt.prefix}_titration_curve.csv"
    with curve_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_index",
                "target_b",
                "mu_kcal_per_mol",
                "mean_N",
                "std_N",
                "sem_N",
                "n_checkpoints",
                "minimum_ghost_pool",
                "excluded",
            ],
        )
        writer.writeheader()
        for point in curve:
            writer.writerow({key: point[key] for key in writer.fieldnames})

    energy_csv = output / f"{opt.prefix}_gci_free_energy.csv"
    with energy_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["n", "df_bind_kcal_per_mol", "extrapolated"]
        )
        writer.writeheader()
        writer.writerows(free_energy)

    outputs = [curve_csv, energy_csv]
    plots = []
    if not opt.no_plots:
        plots = make_plots(
            output, opt.prefix, curve, fitted_n if violations else None,
            b_equil, free_energy, n_star,
        )
        outputs.extend(plots)

    summary = {
        "stage": "gci_analysis",
        "run_root": str(run_root),
        "prefix": opt.prefix,
        "windows_used": len(kept),
        "windows_found": len(windows),
        "excluded_window_indices": sorted(excluded),
        "radius_angstrom": radius,
        "standard_volume_angstrom3": standard_volume,
        "kt_kcal_per_mol": kt,
        "mu_hydration_kcal_per_mol": mu_hydration,
        "equilibrium_b": b_equil,
        "checkpoint_interval_moves": checkpoint_interval,
        "checkpoints_per_window": n_checkpoints,
        "checkpoints_averaged": n_tail,
        "attempts_averaged": n_tail * checkpoint_interval,
        "monotonic_fit": opt.monotonic_fit,
        "monotonicity_inversions": len(violations),
        "n_at_equilibrium_b": n_at_equil,
        "max_n_evaluated": max_n,
        "n_star": n_star,
        "df_bind_star_kcal_per_mol": df_star,
        "df_bind_star_extrapolated": free_energy[best]["extrapolated"],
        "shared_window_metadata": shared,
        "window_markers_sha256": {
            window["directory"].name: window["marker_sha256"] for window in windows
        },
    }
    summary_json = output / f"{opt.prefix}_gci_analysis.json"
    write_json_atomic(summary_json, summary)
    outputs.append(summary_json)

    signature = {
        "stage": "gci_analysis",
        "prefix": opt.prefix,
        "windows": summary["window_markers_sha256"],
        "excluded_window_indices": summary["excluded_window_indices"],
        "equilibrated_fraction": opt.equilibrated_fraction,
        "equilibrated_moves": opt.equilibrated_moves,
        "mu_hydration_kcal_per_mol": mu_hydration,
        "max_n": opt.max_n,
        "monotonic_fit": opt.monotonic_fit,
        "plots": not opt.no_plots,
        "implementation": implementation_signature(
            sources={
                "gci_analyse.py": Path(__file__),
                "pipeline_utils.py": Path(__file__).with_name("pipeline_utils.py"),
            },
            distributions=("numpy", "matplotlib"),
            modules=("numpy",),
        ),
    }
    marker = output / "gci_analysis.complete.json"
    if not opt.force and checkpoint_matches(marker, signature=signature, outputs=outputs):
        print(f"Analysis checkpoint is valid: {marker}", flush=True)
        return
    invalidate_checkpoint(marker)
    complete_checkpoint(
        marker,
        signature=signature,
        outputs=outputs,
        details={"summary": {k: summary[k] for k in (
            "windows_used", "equilibrium_b", "n_at_equilibrium_b",
            "n_star", "df_bind_star_kcal_per_mol", "monotonicity_inversions")}},
    )
    for path in outputs:
        print(f"  wrote {path.name}", flush=True)


if __name__ == "__main__":
    main()
