#!/usr/bin/env python3
"""Create presentation-ready EV71 results and non-specialist speaker notes."""

from __future__ import annotations

import textwrap
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "replicate_average_report"
TECHNICAL = ROOT / "report"
OUT = ROOT / "colleague_summary"
OUT.mkdir(exist_ok=True)

NAVY = "#17365D"
BLUE = "#2B6CA3"
SKY = "#7CB7D8"
ORANGE = "#E07A3F"
GREEN = "#3A8D6D"
RED = "#B54545"
GREY = "#566573"
LIGHT = "#F3F6F8"


def wrapped(text: str, width: int) -> str:
    return "\n".join(
        line
        for paragraph in text.splitlines()
        for line in (textwrap.wrap(paragraph, width=width) or [""])
    )


def title(fig, heading: str, subheading: str | None = None) -> None:
    fig.text(.06, .955, heading, fontsize=24, fontweight="bold", color=NAVY, va="top")
    if subheading:
        fig.text(.06, .905, subheading, fontsize=11.5, color=GREY, va="top")


def save(fig, pdf, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight", facecolor="white")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, width, height, heading, body, color=BLUE) -> None:
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.02",
                           facecolor="white", edgecolor=color, linewidth=2,
                           transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(xy[0] + .025, xy[1] + height - .055, heading, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=color, va="top")
    ax.text(xy[0] + .025, xy[1] + height - .115, wrapped(body, 35), transform=ax.transAxes,
            fontsize=9.5, color="#263238", va="top", linespacing=1.25)


def main() -> None:
    cell = pd.read_csv(SOURCE / "ligand_neighborhood_replica_summary.csv")
    ligand = pd.read_csv(SOURCE / "ligand_replica_range_summary.csv")
    feature = pd.read_csv(SOURCE / "neighborhood_between_ligand_vs_replica_noise.csv")
    means = pd.read_csv(SOURCE / "ligand_mean_profiles.csv", index_col=0)
    pairs = pd.read_csv(SOURCE / "ligand_pairwise_mean_profile_comparison.csv")
    site = pd.read_csv(TECHNICAL / "site_statistics.csv").set_index("site_id")
    informative = feature.loc[feature.informative_ligand_difference.astype(bool),
                              "neighborhood_id"].tolist()
    label_map = {
        "NG16": "HS019\nSer87",
        "NG15": "HS018\nAla86",
        "NG08": "HS008/13/15\nGly128",
        "NG21": "HS027\nAla86",
        "NG03": "HS003/23\nGly108",
    }
    feature = feature.set_index("neighborhood_id")

    sns.set_theme(style="whitegrid", context="talk")
    with nullcontext(None) as pdf:
        # 1: Study design and result in one sentence.
        fig = plt.figure(figsize=(13.33, 7.5))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        title(fig, "EV71 ligand-series hydration mapping",
              "32 ligands × 6 independent replicas | Loch GCMC/MD | shared spatial water-site analysis")
        fig.text(.06, .79, wrapped(
            "The binding pocket retains a strongly conserved hydration pattern across the chemical series, "
            "with five local regions showing reproducible ligand-dependent water rearrangements.", 92),
            fontsize=21, color=NAVY, fontweight="bold", va="top", linespacing=1.25)
        box(ax, (.06, .37), .25, .25, "Simulation", "192 full production runs\n10 ns MD per run\n96 million GCMC attempts", BLUE)
        box(ax, (.375, .37), .25, .25, "Replication", "Six independent runs per ligand\nMean = ligand estimate\nRange and SD = repeatability", GREEN)
        box(ax, (.69, .37), .25, .25, "Comparison", "17 supported water regions\n32 ligand-average profiles\nSignal judged against replica noise", ORANGE)
        fig.text(.06, .19, "Headline numbers", fontsize=15, fontweight="bold", color=NAVY)
        fig.text(.06, .12, "0.917", fontsize=31, fontweight="bold", color=BLUE)
        fig.text(.17, .125, "median correlation between ligand-average hydration profiles", fontsize=12, color=GREY)
        fig.text(.61, .12, "5", fontsize=31, fontweight="bold", color=ORANGE)
        fig.text(.66, .125, "regions with differences larger than replica noise", fontsize=12, color=GREY)
        save(fig, pdf, "01_study_overview.png")

        # 2: Broad similarity.
        fig = plt.figure(figsize=(13.33, 7.5))
        title(fig, "The ligand-average hydration profiles are broadly similar",
              "Correlation measures pattern similarity; RMSE measures the absolute size of differences")
        ax1 = fig.add_axes([.07, .25, .38, .52])
        ax2 = fig.add_axes([.56, .25, .38, .52])
        sns.histplot(pairs.mean_profile_pearson_correlation, bins=18, color=BLUE, ax=ax1)
        median_corr = pairs.mean_profile_pearson_correlation.median()
        ax1.axvline(median_corr, color=ORANGE, lw=3)
        ax1.text(median_corr - .01, ax1.get_ylim()[1] * .91, f"median = {median_corr:.3f}",
                 color=ORANGE, ha="right", fontweight="bold")
        ax1.set(xlabel="Pearson correlation", ylabel="ligand pairs",
                title="Same wet/dry pattern")
        sns.histplot(pairs.mean_profile_rmse, bins=18, color=GREEN, ax=ax2)
        median_rmse = pairs.mean_profile_rmse.median()
        ax2.axvline(median_rmse, color=ORANGE, lw=3)
        ax2.text(median_rmse + .004, ax2.get_ylim()[1] * .91, f"median = {median_rmse:.3f}",
                 color=ORANGE, ha="left", fontweight="bold")
        ax2.set(xlabel="RMSE in regional occupancy", ylabel="ligand pairs",
                title="Small absolute differences")
        fig.text(.07, .065, wrapped(
            "Interpretation: the compounds mostly preserve the same water network, as expected for a related series "
            "binding the same pocket. High correlation does not mean every site is identical; local deviations are examined next.", 125),
            fontsize=12, color=GREY)
        save(fig, pdf, "02_ligand_profile_agreement.png")

        # 3: What the six replicas tell us.
        fig = plt.figure(figsize=(13.33, 7.5))
        title(fig, "Six replicas quantify how repeatable each ligand result is",
              "Each box summarizes the max−min occupancy ranges across all supported water regions for one ligand")
        ax = fig.add_axes([.10, .11, .82, .73])
        order = ligand.sort_values("median_replica_range").ligand
        sns.boxplot(data=cell, y="ligand", x="replica_range", order=order,
                    color=SKY, fliersize=2, linewidth=1, ax=ax)
        ax.axvline(cell.replica_range.median(), color=ORANGE, ls="--", lw=2,
                   label=f"overall median = {cell.replica_range.median():.3f}")
        ax.set(xlabel="occupancy range among six replicas", ylabel="ligand")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.legend(loc="lower right", fontsize=10)
        fig.text(.67, .76, "A range is deliberately conservative:\nit uses the most extreme pair of six runs.",
                 fontsize=10.5, color=GREY, ha="right",
                 bbox=dict(facecolor="white", edgecolor="#D5D8DC", boxstyle="round,pad=.4"))
        save(fig, pdf, "03_replica_range_distributions.png")

        # 4: Signal versus replica noise.
        fig = plt.figure(figsize=(13.33, 7.5))
        title(fig, "Five water regions differ across ligands beyond typical replica variation",
              "Points above the diagonal vary more across ligand means than within a ligand's six replicas")
        ax = fig.add_axes([.07, .15, .55, .69])
        plot = feature.reset_index()
        colors = np.where(plot.informative_ligand_difference.astype(bool), ORANGE, BLUE)
        sizes = 80 + 500 * np.clip(plot.icc_1_1, 0, None)
        ax.scatter(plot.median_within_ligand_replica_range, plot.between_ligand_mean_range,
                   s=sizes, c=colors, alpha=.9, edgecolor="white", linewidth=1)
        maximum = max(plot.median_within_ligand_replica_range.max(), plot.between_ligand_mean_range.max()) * 1.06
        ax.plot([0, maximum], [0, maximum], ls="--", color="#89939A")
        for row in plot.itertuples():
            ax.annotate(row.neighborhood_id,
                        (row.median_within_ligand_replica_range, row.between_ligand_mean_range),
                        fontsize=8, xytext=(3, 2), textcoords="offset points")
        ax.set(xlabel="median within-ligand replica range",
               ylabel="range among 32 ligand means", xlim=(-.02, maximum), ylim=(-.02, maximum))
        ax2 = fig.add_axes([.66, .16, .31, .66]); ax2.axis("off")
        ax2.text(0, 1, "Reproducible ligand-sensitive regions", fontsize=13,
                 fontweight="bold", color=NAVY, va="top")
        y = .90
        for gid in informative:
            row = feature.loc[gid]
            ax2.text(0, y, f"{gid}  {row.member_sites}", fontsize=11.5,
                     fontweight="bold", color=ORANGE, va="top")
            ax2.text(.04, y - .055,
                     f"ligand span {row.between_ligand_mean_range:.3f}  |  typical replica range {row.median_within_ligand_replica_range:.3f}",
                     fontsize=9.5, color=GREY, va="top")
            y -= .17
        save(fig, pdf, "04_signal_vs_replica_noise.png")

        # 5: Which ligands drive those five regions.
        fig = plt.figure(figsize=(13.33, 7.5))
        title(fig, "Local differences are superimposed on the conserved network",
              "Values are six-replica mean regional occupancies; ligands are ordered by profile similarity")
        scaled = means[informative]
        order = hierarchy.leaves_list(hierarchy.linkage(scaled, method="average", metric="euclidean"))
        heat = scaled.iloc[order].rename(columns=label_map)
        ax = fig.add_axes([.09, .12, .68, .74])
        sns.heatmap(heat, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.3,
                    annot_kws={"fontsize": 8},
                    cbar_kws={"label": "mean regional occupancy"}, ax=ax)
        ax.set(xlabel="water region / nearest pocket residue", ylabel="ligand")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=9, rotation=0)
        ax2 = fig.add_axes([.80, .17, .18, .62]); ax2.axis("off")
        ax2.text(0, 1, "Examples", fontsize=14, fontweight="bold", color=NAVY, va="top")
        examples = [
            ("HS019", "x7247a: 0.129\nx6832a: 0.608"),
            ("HS027", "x6738a: 0.077\nx7247a: 0.473"),
            ("HS003/23", "x7589a: 0.494\nx7309a: 0.902"),
        ]
        y = .86
        for heading, body in examples:
            ax2.text(0, y, heading, fontsize=12, fontweight="bold", color=ORANGE, va="top")
            ax2.text(.04, y - .07, body, fontsize=10.5, color=GREY, va="top", linespacing=1.3)
            y -= .27
        save(fig, pdf, "05_key_region_heatmap.png")

        # 6: What can and cannot be claimed.
        fig = plt.figure(figsize=(13.33, 7.5))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        title(fig, "How to explain the result")
        box(ax, (.06, .58), .40, .24, "What the data support",
            "The hydration architecture is conserved across the series. Five local regions respond reproducibly to ligand identity beyond six-replica sampling variation.", GREEN)
        box(ax, (.54, .58), .40, .24, "Interesting conserved feature",
            f"HS009 is occupied in {site.loc['HS009', 'mean_occupancy']:.1%} of frames and is a geometry-based "
            f"protein–water–ligand bridge candidate in {site.loc['HS009', 'mean_water_bridge_fraction']:.1%}. It is conserved rather than ligand-discriminating.", BLUE)
        box(ax, (.06, .25), .40, .24, "What not to overclaim",
            "This does not prove identical ligand poses, binding free energies, or kinetic water residence times. Bridge assignments are geometry-based.", RED)
        box(ax, (.54, .25), .40, .24, "Overall conclusion",
            "The workflow behaves sensibly across 32 related complexes. Broad agreement dominates, while specific substitutions reorganize a few pocket-water regions.", ORANGE)
        fig.text(.06, .11, "Suggested one-sentence conclusion", fontsize=13, fontweight="bold", color=NAVY)
        fig.text(.06, .055, wrapped(
            "Across 32 EV71 ligands, replicate-averaged GCMC/MD maps recover a conserved hydration network and identify five "
            "localized, reproducible ligand-dependent water rearrangements.", 125), fontsize=13, color=GREY)
        save(fig, pdf, "06_interpretation_and_caveats.png")

    print(OUT)
    return

    notes = f"""# EV71 hydration results: speaking notes

## Thirty-second explanation

We simulated 32 related EV71 ligands, with six independent GCMC/MD replicas for each ligand. We mapped the same spatial water regions in every run, averaged the six replicas to obtain one hydration profile per ligand, and used the replica spread to measure uncertainty. The ligand profiles are strongly similar overall—the median pairwise correlation is {pairs.mean_profile_pearson_correlation.median():.3f}—which is consistent with a related chemical series binding the same pocket. Against that conserved background, five local water regions show ligand-dependent changes larger than normal replica variation.

## What was measured

1. A **hydration site** is a spatial region where water oxygen atoms repeatedly occur.
2. For each replica, **occupancy** is the fraction of 2,500 production frames in which that region contains a water.
3. The six occupancies for one ligand are averaged to give its best estimate.
4. Their range, SD, SEM, and confidence interval describe repeatability.
5. The 32 ligand means are compared only after this averaging; the 192 runs are not treated as independent ligands.

Nearby site labels closer than the assignment radius were combined into non-overlapping neighborhoods before the comparison. For combined neighborhoods, “regional occupancy” is the summed occupancy of the component sites and should not be described as a single-site probability.

## What the headline numbers mean

- **Median profile correlation = {pairs.mean_profile_pearson_correlation.median():.3f}.** The same regions generally remain relatively wet or dry across ligands.
- **Median profile RMSE = {pairs.mean_profile_rmse.median():.3f}.** The typical absolute difference between two ligand means is about nine percentage points of regional occupancy.
- **Median six-replica range = {cell.replica_range.median():.3f}.** For one ligand and one region, the most extreme pair among six simulations commonly differs by about 0.22. This is a conservative spread, not the uncertainty of the average.
- **Median RMS SEM = {ligand.rms_standard_error_of_mean_profile.median():.3f}.** Averaging six replicas reduces the typical uncertainty of the ligand profile to about 0.056.

## The five ligand-sensitive regions

| Region | Spatial sites | Nearest residue | Ligand-mean span | Typical replica range | Interpretation |
|---|---|---|---:|---:|---|
| NG16 | HS019 | Ser87 | 0.479 | 0.183 | Strongest ligand-dependent occupancy difference |
| NG15 | HS018 | Ala86 | 0.164 | 0.080 | Smaller but highly repeatable shift |
| NG08 | HS008/13/15 | Gly128 | 0.264 | 0.142 | Extended/alternative water positions near one pocket region |
| NG21 | HS027 | Ala86 | 0.396 | 0.240 | Ligand-dependent site near Ala86 |
| NG03 | HS003/23 | Gly108 | 0.408 | 0.282 | Reorganization near Gly108 |

The key criterion is not merely that ligand means differ. Their difference must also exceed the variation seen when the same ligand is simulated six times.

## Conserved bridge candidate

HS009 is occupied in {site.loc['HS009', 'mean_occupancy']:.1%} of frames. It meets the geometry criteria for a protein–water–ligand bridge in {site.loc['HS009', 'mean_water_bridge_fraction']:.1%} of frames and lies nearest {site.loc['HS009', 'nearest_residue']}. Its occupancy does not vary reproducibly by ligand, so it is a candidate conserved structural water rather than an explanation of ligand differences. The H-bond typing is geometry-only and needs visual chemical inspection.

## How to interpret the lack of strong clusters

The ligand averages do not separate into several clean hydration classes. The best mathematical split isolates two ligands from the other 30, but its bootstrap support is only moderate. This is not a negative result. A continuous family of similar hydration profiles is what one might expect for related compounds sharing a binding mode, with gradual local changes as substituents change.

## Defensible conclusion

The simulations are technically complete and the replicate-average analysis behaves sensibly. The dominant result is conservation of the pocket hydration network across the series. Five localized water regions show reproducible ligand dependence beyond replica noise, providing concrete sites for structural interpretation or future structure–activity analysis.

## Caveats to state explicitly

- These are simulation repeatability estimates, not experimental uncertainties.
- High profile correlation supports similar hydration patterns but does not prove identical ligand poses.
- Occupancy is an equilibrium population-like measure; it is not a kinetic residence time.
- The GCMC insertion/deletion sequence is not physical time, so kinetic claims require a different analysis.
- The affinity association at NG05 is exploratory and should not be presented as a validated predictive model.
"""
    (OUT / "EV71_results_speaking_notes.md").write_text(notes)
    print(OUT)


if __name__ == "__main__":
    main()
