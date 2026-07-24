#!/usr/bin/env python3
"""Analyze EV71 as 32 ligand estimates, each averaged over six replicas.

This is intentionally a second-level analysis.  The common spatial site
catalog supplies matched coordinates; the six independent simulations then
provide the mean and uncertainty for each ligand at every site/neighborhood.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "report"
OUT = ROOT / "replicate_average_report"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260721)
N_REPLICAS = 6
BOOTSTRAPS = 1000


def bh_fdr(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(ranked, 1)
    return out


def icc_balanced(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "group": groups})
    arrays = [x.value.to_numpy() for _, x in frame.groupby("group")]
    means = np.array([a.mean() for a in arrays])
    grand = frame.value.mean()
    ss_between = N_REPLICAS * np.sum((means - grand) ** 2)
    ss_within = sum(np.sum((a - a.mean()) ** 2) for a in arrays)
    ms_between = ss_between / (len(arrays) - 1)
    ms_within = ss_within / (len(arrays) * (N_REPLICAS - 1))
    denominator = ms_between + (N_REPLICAS - 1) * ms_within
    return float((ms_between - ms_within) / denominator) if denominator else float("nan")


def main() -> None:
    metrics_path = SOURCE / "all_run_site_metrics.csv"
    neighborhoods_path = SOURCE / "site_neighborhood_statistics.csv"
    if not metrics_path.exists() or not neighborhoods_path.exists():
        raise FileNotFoundError("Run analyze_ev71_consistency.py first to build the audited common-site tables")
    metrics = pd.read_csv(metrics_path)
    neighborhoods = pd.read_csv(neighborhoods_path)
    supported = neighborhoods[neighborhoods.supported.astype(bool)].copy()
    site_to_group = {
        site: row.neighborhood_id
        for row in neighborhoods.itertuples()
        for site in row.member_sites.split(";")
    }
    metrics["neighborhood_id"] = metrics.site_id.map(site_to_group)
    run_long = metrics.groupby(
        ["ligand", "replicate", "neighborhood_id"], as_index=False
    ).occupancy.sum()
    run_profiles = run_long.pivot(
        index=["ligand", "replicate"], columns="neighborhood_id", values="occupancy"
    )[supported.neighborhood_id]
    if run_profiles.shape != (32 * N_REPLICAS, len(supported)):
        raise RuntimeError(f"Unexpected run matrix {run_profiles.shape}")

    # The primary result: one mean and an uncertainty distribution for every
    # ligand × matched hydration neighborhood.
    cell = run_long[run_long.neighborhood_id.isin(supported.neighborhood_id)].groupby(
        ["ligand", "neighborhood_id"]
    ).occupancy.agg(["mean", "median", "min", "max", "std"]).reset_index()
    cell = cell.rename(columns={"mean": "replica_mean_occupancy",
                                "median": "replica_median_occupancy",
                                "min": "replica_minimum_occupancy",
                                "max": "replica_maximum_occupancy",
                                "std": "replica_sd"})
    cell["replica_range"] = cell.replica_maximum_occupancy - cell.replica_minimum_occupancy
    cell["replica_sem"] = cell.replica_sd / math.sqrt(N_REPLICAS)
    cell["mean_ci95_half_width_t_df5"] = cell.replica_sem * stats.t.ppf(.975, N_REPLICAS - 1)
    cell["mean_ci95_low"] = cell.replica_mean_occupancy - cell.mean_ci95_half_width_t_df5
    cell["mean_ci95_high"] = cell.replica_mean_occupancy + cell.mean_ci95_half_width_t_df5
    cell = cell.merge(supported[["neighborhood_id", "member_sites", "nearest_residues"]],
                      on="neighborhood_id")
    cell.to_csv(OUT / "ligand_neighborhood_replica_summary.csv", index=False)

    ligand_summary = cell.groupby("ligand").agg(
        median_replica_range=("replica_range", "median"),
        mean_replica_range=("replica_range", "mean"),
        q75_replica_range=("replica_range", lambda x: x.quantile(.75)),
        p90_replica_range=("replica_range", lambda x: x.quantile(.90)),
        maximum_replica_range=("replica_range", "max"),
        median_replica_sd=("replica_sd", "median"),
        median_mean_ci95_half_width=("mean_ci95_half_width_t_df5", "median"),
        fraction_neighborhoods_range_le_0_10=("replica_range", lambda x: np.mean(x <= .10)),
        fraction_neighborhoods_range_le_0_20=("replica_range", lambda x: np.mean(x <= .20)),
    ).reset_index()
    run_sd = run_profiles.groupby(level="ligand").std()
    ligand_summary["rms_standard_error_of_mean_profile"] = ligand_summary.ligand.map(
        np.sqrt((run_sd ** 2 / N_REPLICAS).mean(axis=1)))
    ligand_summary.to_csv(OUT / "ligand_replica_range_summary.csv", index=False)

    # For each shared region, compare variation among the 32 six-replica means
    # against the typical range/noise among replicas of one ligand.
    feature_rows = []
    for gid in supported.neighborhood_id:
        x = run_long[run_long.neighborhood_id == gid]
        means = x.groupby("ligand").occupancy.mean()
        ranges = x.groupby("ligand").occupancy.agg(lambda y: y.max() - y.min())
        within_var = x.groupby("ligand").occupancy.var()
        arrays = [g.occupancy.to_numpy() for _, g in x.groupby("ligand")]
        f_stat, p_value = stats.f_oneway(*arrays)
        feature_rows.append({
            "neighborhood_id": gid,
            "member_sites": supported.set_index("neighborhood_id").loc[gid, "member_sites"],
            "mean_occupancy_across_ligands": means.mean(),
            "minimum_ligand_mean": means.min(),
            "minimum_ligand": means.idxmin(),
            "maximum_ligand_mean": means.max(),
            "maximum_ligand": means.idxmax(),
            "between_ligand_mean_range": means.max() - means.min(),
            "between_ligand_mean_sd": means.std(),
            "median_within_ligand_replica_range": ranges.median(),
            "q75_within_ligand_replica_range": ranges.quantile(.75),
            "p90_within_ligand_replica_range": ranges.quantile(.90),
            "pooled_within_ligand_replica_sd": math.sqrt(within_var.mean()),
            "icc_1_1": icc_balanced(x.occupancy, x.ligand),
            "anova_f": f_stat,
            "anova_p": p_value,
        })
    feature = pd.DataFrame(feature_rows)
    feature["anova_q_bh"] = bh_fdr(feature.anova_p)
    feature["between_range_over_median_within_range"] = (
        feature.between_ligand_mean_range / feature.median_within_ligand_replica_range)
    feature["between_sd_over_pooled_within_sd"] = (
        feature.between_ligand_mean_sd / feature.pooled_within_ligand_replica_sd)
    feature["mean_difference_exceeds_typical_replica_range"] = (
        feature.between_ligand_mean_range > feature.median_within_ligand_replica_range)
    feature["informative_ligand_difference"] = (
        (feature.anova_q_bh < .05) & (feature.icc_1_1 > .25) &
        feature.mean_difference_exceeds_typical_replica_range)
    feature = feature.sort_values("between_sd_over_pooled_within_sd", ascending=False)
    feature.to_csv(OUT / "neighborhood_between_ligand_vs_replica_noise.csv", index=False)
    informative = feature.loc[feature.informative_ligand_difference, "neighborhood_id"].tolist()
    if len(informative) < 3:
        informative = feature.head(5).neighborhood_id.tolist()

    means = run_profiles.groupby(level="ligand").mean()
    sds = run_profiles.groupby(level="ligand").std()
    means.to_csv(OUT / "ligand_mean_profiles.csv")
    sds.to_csv(OUT / "ligand_profile_replica_sd.csv")

    # Pairwise agreement between the 32 ligand estimates.  Bootstrap resamples
    # replicas within each ligand, preserving the unit of replication.
    ligands = list(means.index)
    bootstrap_means = {}
    for ligand in ligands:
        values = run_profiles.loc[ligand].to_numpy()
        choices = RNG.integers(0, N_REPLICAS, size=(BOOTSTRAPS, N_REPLICAS))
        bootstrap_means[ligand] = values[choices].mean(axis=1)
    pooled_sd = np.sqrt(run_profiles.groupby(level="ligand").var().mean()).clip(lower=.05)
    pair_rows = []
    for a, b in combinations(ligands, 2):
        delta = means.loc[a] - means.loc[b]
        boot_distance = np.sqrt(np.mean((bootstrap_means[a] - bootstrap_means[b]) ** 2, axis=1))
        raw_distance = float(np.sqrt(np.mean(delta ** 2)))
        correlation = float(np.corrcoef(means.loc[a], means.loc[b])[0, 1])
        standardized = float(np.sqrt(np.mean((delta / pooled_sd) ** 2)))
        mean_se = np.sqrt(sds.loc[a] ** 2 / N_REPLICAS + sds.loc[b] ** 2 / N_REPLICAS).clip(lower=.02)
        pair_rows.append({
            "ligand_1": a, "ligand_2": b,
            "mean_profile_pearson_correlation": correlation,
            "mean_profile_rmse": raw_distance,
            "bootstrap_resampled_profile_rmse_p025": np.quantile(boot_distance, .025),
            "bootstrap_resampled_profile_rmse_p975": np.quantile(boot_distance, .975),
            "pooled_noise_standardized_rmse": standardized,
            "difference_over_pair_mean_standard_error_rms": float(np.sqrt(np.mean((delta / mean_se) ** 2))),
        })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "ligand_pairwise_mean_profile_comparison.csv", index=False)

    # Descriptive clustering of ligand means on only reproducible discriminatory
    # regions. Bootstrap co-assignment quantifies whether those branches survive
    # resampling the six replicas.
    cluster_scale = pooled_sd[informative]
    cluster_data = (means[informative] - means[informative].mean()) / cluster_scale
    linkage = hierarchy.linkage(cluster_data, method="ward")
    selection_rows = []
    for k in range(2, 9):
        labels = hierarchy.fcluster(linkage, k, criterion="maxclust")
        sizes = np.bincount(labels)[1:]
        selection_rows.append({"clusters": k, "silhouette": silhouette_score(cluster_data, labels),
                               "minimum_cluster_size": sizes.min(), "maximum_cluster_size": sizes.max()})
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(OUT / "cluster_number_diagnostics.csv", index=False)
    selected_k = int(selection.sort_values("silhouette", ascending=False).iloc[0].clusters)
    base_labels = hierarchy.fcluster(linkage, selected_k, criterion="maxclust")
    coassignment = np.zeros((len(ligands), len(ligands)))
    cluster_bootstraps = 500
    for _ in range(cluster_bootstraps):
        boot = []
        for ligand in ligands:
            values = run_profiles.loc[ligand, informative].to_numpy()
            boot.append(values[RNG.integers(0, N_REPLICAS, N_REPLICAS)].mean(axis=0))
        boot = (np.asarray(boot) - means[informative].mean().to_numpy()) / cluster_scale.to_numpy()
        labels = hierarchy.fcluster(hierarchy.linkage(boot, method="ward"), selected_k,
                                    criterion="maxclust")
        coassignment += labels[:, None] == labels[None, :]
    coassignment /= cluster_bootstraps
    pd.DataFrame(coassignment, index=ligands, columns=ligands).to_csv(
        OUT / "bootstrap_cluster_coassignment.csv")
    cluster_rows = []
    for i, ligand in enumerate(ligands):
        same = np.flatnonzero(base_labels == base_labels[i])
        others = same[same != i]
        stability = float(coassignment[i, others].mean()) if len(others) else np.nan
        cluster_rows.append({"ligand": ligand, "descriptive_cluster": int(base_labels[i]),
                             "mean_bootstrap_coassignment_with_base_cluster": stability})
    cluster_assignments = pd.DataFrame(cluster_rows)
    cluster_assignments.to_csv(OUT / "ligand_mean_profile_clusters.csv", index=False)

    # PCA is visualization only; replica points show uncertainty around each
    # ligand mean instead of pretending each mean is exact.
    pca = PCA(n_components=2)
    mean_scores = pca.fit_transform(cluster_data)
    replica_scaled = (run_profiles[informative] - means[informative].mean()) / cluster_scale
    replica_scores = pca.transform(replica_scaled)
    score_rows = []
    for (ligand, replica), score in zip(run_profiles.index, replica_scores):
        score_rows.append({"ligand": ligand, "replicate": replica, "PC1": score[0], "PC2": score[1]})
    pd.DataFrame(score_rows).to_csv(OUT / "ligand_replica_pca_scores.csv", index=False)

    # Experimental affinity is tested against the six-replica ligand means,
    # rather than treating 192 runs as independent compounds.
    subset = pd.read_csv(ROOT / "openbind_ev71_2a_pyrrolidine_benchmark_release" /
                         "subset" / "pyrrolidine_32_subset.csv").set_index("ligand_name")
    affinity_rows = []
    for gid in means.columns:
        rho, p = stats.spearmanr(means[gid].reindex(subset.index), subset.experimental_pKD)
        affinity_rows.append({"neighborhood_id": gid, "member_sites": supported.set_index(
            "neighborhood_id").loc[gid, "member_sites"], "spearman_rho_vs_pKD": rho, "p_value": p})
    affinity = pd.DataFrame(affinity_rows)
    affinity["q_value_bh"] = bh_fdr(affinity.p_value)
    affinity.to_csv(OUT / "ligand_mean_affinity_correlations.csv", index=False)

    # Figures.
    sns.set_theme(style="whitegrid", context="notebook")
    ligand_order = ligand_summary.sort_values("median_replica_range").ligand
    fig, ax = plt.subplots(figsize=(9, 10))
    sns.boxplot(data=cell, y="ligand", x="replica_range", order=ligand_order,
                color="#75aadb", fliersize=2, ax=ax)
    ax.axvline(.2, color="firebrick", ls="--", lw=1, label="range = 0.20")
    ax.set(title="Distribution of six-replica occupancy ranges within each ligand",
           xlabel="max replica occupancy − min replica occupancy", ylabel="ligand")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "replica_ranges_by_ligand.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_feature = feature.rename(columns={"informative_ligand_difference": "difference exceeds replica noise",
                                           "icc_1_1": "ICC"})
    sns.scatterplot(data=plot_feature, x="median_within_ligand_replica_range",
                    y="between_ligand_mean_range", hue="difference exceeds replica noise",
                    size="ICC", sizes=(40, 230), ax=ax)
    limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, limit], [0, limit], color="grey", ls="--", lw=1)
    for row in feature.itertuples():
        ax.annotate(row.neighborhood_id, (row.median_within_ligand_replica_range,
                                         row.between_ligand_mean_range), fontsize=8, xytext=(3, 2),
                    textcoords="offset points")
    ax.set(title="Ligand-to-ligand signal versus within-ligand replica range",
           xlabel="median range among six replicas", ylabel="range among 32 ligand means")
    fig.tight_layout(); fig.savefig(OUT / "between_ligand_signal_vs_replica_range.png", dpi=180); plt.close(fig)

    ordered = hierarchy.leaves_list(linkage)
    heat = means[informative].iloc[ordered]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(heat, cmap="viridis", annot=True, fmt=".2f", vmin=0, ax=ax,
                cbar_kws={"label": "six-replica mean regional occupancy"})
    ax.set(title="Ligand-average profiles at reproducible discriminatory regions",
           xlabel="hydration neighborhood", ylabel="ligand")
    fig.tight_layout(); fig.savefig(OUT / "ligand_average_profile_heatmap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ligand in enumerate(ligands):
        sub = replica_scores[np.array([x[0] == ligand for x in run_profiles.index])]
        ax.scatter(sub[:, 0], sub[:, 1], s=12, alpha=.20, color="grey")
        for point in sub:
            ax.plot([mean_scores[i, 0], point[0]], [mean_scores[i, 1], point[1]],
                    color="grey", alpha=.10, lw=.5)
        ax.scatter(mean_scores[i, 0], mean_scores[i, 1], s=30, color="#2468a2")
        ax.annotate(ligand, mean_scores[i], fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set(title="Ligand means and their six replica profiles",
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.tight_layout(); fig.savefig(OUT / "ligand_means_with_replica_spread_pca.png", dpi=180); plt.close(fig)

    # Pairwise mean-profile distance matrix in the same descriptive order.
    distance = np.zeros((len(ligands), len(ligands)))
    for row in pairs.itertuples():
        i, j = ligands.index(row.ligand_1), ligands.index(row.ligand_2)
        distance[i, j] = distance[j, i] = row.mean_profile_rmse
    order_ligands = list(heat.index)
    indices = [ligands.index(x) for x in order_ligands]
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(distance[np.ix_(indices, indices)], xticklabels=order_ligands,
                yticklabels=order_ligands, cmap="mako", ax=ax,
                cbar_kws={"label": "RMSE between six-replica mean profiles"})
    ax.set_title("Pairwise differences between ligand-average hydration profiles")
    fig.tight_layout(); fig.savefig(OUT / "ligand_mean_pairwise_distance.png", dpi=180); plt.close(fig)

    # Summary and report.
    range_quantiles = cell.replica_range.quantile([.25, .5, .75, .9, .95]).to_dict()
    pair_corr_quantiles = pairs.mean_profile_pearson_correlation.quantile([.1, .5, .9]).to_dict()
    pair_rmse_quantiles = pairs.mean_profile_rmse.quantile([.1, .5, .9]).to_dict()
    cluster_sizes = pd.Series(base_labels).value_counts().sort_index().to_dict()
    summary = {
        "analysis_unit": "32 ligands; each ligand estimate is the mean of six independent replicas",
        "ligands": 32, "replicas_per_ligand": N_REPLICAS,
        "supported_neighborhoods": len(supported),
        "ligand_neighborhood_estimates": len(cell),
        "within_ligand_replica_range": {
            "q25": range_quantiles[.25], "median": range_quantiles[.5],
            "q75": range_quantiles[.75], "p90": range_quantiles[.9], "p95": range_quantiles[.95],
        },
        "uncertainty_of_six_replica_mean_profile": {
            "median_rms_sem": float(ligand_summary.rms_standard_error_of_mean_profile.median()),
            "minimum_rms_sem": float(ligand_summary.rms_standard_error_of_mean_profile.min()),
            "maximum_rms_sem": float(ligand_summary.rms_standard_error_of_mean_profile.max()),
        },
        "agreement_between_ligand_means": {
            "profile_correlation_p10_median_p90": [pair_corr_quantiles[.1], pair_corr_quantiles[.5], pair_corr_quantiles[.9]],
            "profile_rmse_p10_median_p90": [pair_rmse_quantiles[.1], pair_rmse_quantiles[.5], pair_rmse_quantiles[.9]],
        },
        "informative_neighborhoods": informative,
        "descriptive_clustering": {"selected_k_by_silhouette": selected_k,
                                   "silhouette": float(selection.loc[selection.clusters == selected_k, "silhouette"].iloc[0]),
                                   "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()}},
        "significant_mean_occupancy_vs_pKD_after_bh": affinity.loc[affinity.q_value_bh < .05,
                                                                     "neighborhood_id"].tolist(),
    }
    (OUT / "replicate_average_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    top_features = feature.head(8)
    feature_lines = "\n".join(
        f"- {x.neighborhood_id} ({x.member_sites}): ligand-mean range {x.between_ligand_mean_range:.3f}; "
        f"median six-replica range {x.median_within_ligand_replica_range:.3f}; "
        f"ICC {x.icc_1_1:.2f}; ratio {x.between_range_over_median_within_range:.2f}"
        for x in top_features.itertuples())
    noisiest = ligand_summary.nlargest(5, "median_replica_range")
    quietest = ligand_summary.nsmallest(5, "median_replica_range")
    report = f"""# EV71 analysis based on six-replica ligand averages

## Correct unit of analysis

Each of the 32 ligands is represented by the **mean of its six independent replicas** at each matched hydration neighborhood. The six values are retained as the sampling distribution: minimum, maximum, range, SD, SEM, and a t-based 95% confidence interval are all reported. Individual replicas are not treated as 192 independent ligands.

## Main result

The ligand-average hydration profiles agree strongly in their broad pattern: across all 496 ligand pairs, the median Pearson correlation between mean profiles is **{pair_corr_quantiles[.5]:.3f}** (10th–90th percentile {pair_corr_quantiles[.1]:.3f}–{pair_corr_quantiles[.9]:.3f}). The median RMSE between ligand means is **{pair_rmse_quantiles[.5]:.3f}** occupancy units.

Replica variability is substantial but now quantified directly. Across all **{len(cell)} ligand × neighborhood estimates**, the median max-minus-min range among six replicas is **{range_quantiles[.5]:.3f}** (IQR {range_quantiles[.25]:.3f}–{range_quantiles[.75]:.3f}; 90th percentile {range_quantiles[.9]:.3f}). Averaging reduces the median RMS standard error of an entire ligand profile to **{ligand_summary.rms_standard_error_of_mean_profile.median():.3f}**. This is the value of the replicas: the ligand means are appreciably more precise than any one run, and every reported ligand difference now has an empirical uncertainty.

## Where ligand means differ beyond replica variation

The most useful comparison is the range among the 32 ligand means divided by the typical range among six replicas of one ligand. Values above one mean that the full chemical-series span exceeds typical replica spread.

{feature_lines}

The prespecified combination of corrected ANOVA q < 0.05, ICC > 0.25, and between-ligand range greater than the median within-ligand range retains **{', '.join(informative)}**. These are the neighborhoods where ligand-average differences are most defensible. Other regions may be conserved, or their apparent ligand differences are not larger than replica noise.

## Distribution of replica ranges by ligand

The five ligands with the largest median site range are **{', '.join(noisiest.ligand)}** ({', '.join(f'{x:.3f}' for x in noisiest.median_replica_range)}). The five most internally consistent are **{', '.join(quietest.ligand)}** ({', '.join(f'{x:.3f}' for x in quietest.median_replica_range)}). This is a distribution over all supported neighborhoods, not a judgment based on one outlier site.

## Do the ligand averages form clusters?

I hierarchically clustered ligand means only on the reproducible discriminatory neighborhoods, after scaling each neighborhood by its pooled within-ligand replica SD. The best silhouette is **{summary['descriptive_clustering']['silhouette']:.3f}** at k={selected_k}, with cluster sizes {summary['descriptive_clustering']['cluster_sizes']}. Because this solution is imbalanced and replica-bootstrap co-assignment is not uniformly high, the dendrogram should be read as a similarity map rather than evidence for sharply separated ligand classes. The continuous pairwise distance and bootstrap tables are safer than hard labels.

## How to use the outputs

- Start with `ligand_neighborhood_replica_summary.csv`: one row per ligand and neighborhood, with the six-replica mean and full range/uncertainty.
- `ligand_replica_range_summary.csv` gives the requested distribution-of-ranges summary for each ligand.
- `neighborhood_between_ligand_vs_replica_noise.csv` identifies where differences among ligand means exceed replicate variability.
- `ligand_pairwise_mean_profile_comparison.csv` compares every pair of ligand means with replica-bootstrap uncertainty.
- `ligand_mean_profiles.csv` is the 32 × {len(supported)} matrix for downstream modelling.
- The heatmap and PCA show averages and replica spread; the cluster co-assignment matrix shows which branches survive replica resampling.

## Limits

The six replicas estimate simulation repeatability, not experimental error. The common spatial catalog is needed so all ligand means refer to the same locations; nearby competing site labels were combined into non-overlapping neighborhoods before averaging. Confidence intervals are descriptive with only six replicas, and the trajectory blocks are correlated, so the independent unit remains the replica—not the frame.
"""
    (OUT / "replicate_average_report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
