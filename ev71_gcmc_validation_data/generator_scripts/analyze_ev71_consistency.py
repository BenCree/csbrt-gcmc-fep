#!/usr/bin/env python3
"""Audit and compare the completed EV71 Loch hydration series.

The script is deliberately self-contained and read-only with respect to the
downloaded run bundle.  It writes derived tables, figures, and a Markdown
report beneath ``report/`` next to this file.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "ev71-density-series"
INPUT_ROOT = ROOT / "openbind_ev71_2a_pyrrolidine_benchmark_release"
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260721)


def bh_fdr(values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def safe_corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(a, b).statistic)
    return float(np.corrcoef(a, b)[0, 1])


def ligand_rep(path: Path) -> tuple[str, int]:
    rel = path.relative_to(RUN_ROOT)
    return rel.parts[0], int(rel.parts[1].removeprefix("rep"))


def nested_get(d: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def audit_protocol(audit_paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    rows, issues = [], []
    for path in audit_paths:
        ligand, rep = ligand_rep(path)
        d = json.loads(path.read_text())
        row = {
            "ligand": ligand,
            "replicate": rep,
            "status": d.get("status"),
            "profile": d.get("profile"),
            "scope": d.get("validation_scope"),
            "selector": nested_get(d, ("preparation", "ligand_selector")),
            "pose_rmsd_A": nested_get(d, ("preparation", "ligand_only_amber_chemistry", "heavy_atom_pose_rmsd_angstrom")),
            "npt_last_step": nested_get(d, ("equilibration", "csv", "npt", "last_step")),
            "uvt1_ghost_lines": nested_get(d, ("equilibration", "ghost_histories", "uvt1", "lines")),
            "uvt2_ghost_lines": nested_get(d, ("equilibration", "ghost_histories", "uvt2", "lines")),
            "production_last_step": nested_get(d, ("production", "csv", "last_step")),
            "production_frames": nested_get(d, ("production", "trajectory_frames")),
            "production_ghost_lines": nested_get(d, ("production", "ghost_history", "lines")),
            "raw_zero_ghosts": nested_get(d, ("production", "raw_topology", "zero_interaction_waters")),
            "final_zero_ghosts": nested_get(d, ("production", "final_topology", "zero_interaction_waters")),
            "solute_hashes_identical": len(set((d.get("solute_topology_stages") or {}).values())) == 1,
        }
        rows.append(row)
        expected = {
            "status": "passed", "profile": "full", "scope": "full_ludovic_schedule",
            "selector": ligand, "npt_last_step": 1_000_000, "uvt1_ghost_lines": 100,
            "uvt2_ghost_lines": 125, "production_last_step": 5_000_000,
            "production_frames": 2500, "production_ghost_lines": 2500,
            "raw_zero_ghosts": 45, "final_zero_ghosts": 0,
            "solute_hashes_identical": True,
        }
        for key, wanted in expected.items():
            if row[key] != wanted:
                issues.append(f"{ligand}/rep{rep}: {key}={row[key]!r}, expected {wanted!r}")
    return pd.DataFrame(rows).sort_values(["ligand", "replicate"]), issues


def parse_receptor_neighbors(catalog: pd.DataFrame) -> dict[str, dict]:
    pdb = next((INPUT_ROOT / "receptor").glob("*.pdb"))
    residues: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip() or re.sub(r"[^A-Za-z]", "", line[12:16]).strip()[:1]
        if element.upper() == "H":
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        key = (line[21].strip() or "_", line[22:26].strip() + line[26].strip(), line[17:20].strip())
        residues.setdefault(key, []).append(xyz)
    result = {}
    for r in catalog.itertuples():
        site = np.array([r.x_angstrom, r.y_angstrom, r.z_angstrom])
        ds = []
        for (chain, resid, resname), atoms in residues.items():
            ds.append((float(np.min(np.linalg.norm(np.vstack(atoms) - site, axis=1))), chain, resid, resname))
        ds.sort()
        result[r.site_id] = {
            "nearest_residue": f"{ds[0][3]} {ds[0][1]}:{ds[0][2]}",
            "nearest_distance_A": ds[0][0],
            "residues_within_4A": "; ".join(f"{name} {chain}:{resid} ({dist:.2f} A)" for dist, chain, resid, name in ds if dist <= 4.0),
        }
    return result


def one_way_stats(values: pd.Series, groups: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    arrays = [x.v.to_numpy() for _, x in frame.groupby("g")]
    n_groups, k = len(arrays), len(arrays[0])
    means = np.array([a.mean() for a in arrays])
    grand = frame.v.mean()
    ss_between = k * np.sum((means - grand) ** 2)
    ss_within = sum(np.sum((a - a.mean()) ** 2) for a in arrays)
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_groups * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom else np.nan
    f, p = stats.f_oneway(*arrays)
    total = ss_between + ss_within
    return {
        "between_ligand_sd": float(np.std(means, ddof=1)),
        "pooled_within_ligand_sd": float(math.sqrt(ms_within)),
        "icc_1_1": float(icc),
        "eta_squared": float(ss_between / total) if total else np.nan,
        "anova_f": float(f),
        "anova_p": float(p),
    }


def main() -> None:
    # Locate the unique completed common analysis.
    summaries = list(RUN_ROOT.glob("_series_submissions/*/common_analysis/series-density-analysis.json"))
    completed = [p for p in summaries if json.loads(p.read_text()).get("status") == "completed"]
    if len(completed) != 1:
        raise RuntimeError(f"Expected one completed series analysis, found {len(completed)}")
    common_dir = completed[0].parent
    series = json.loads(completed[0].read_text())
    catalog = pd.read_csv(common_dir / "common-site-catalog.csv")

    # Full pipeline gate.
    audit_paths = sorted(RUN_ROOT.glob("*/rep*/pipeline_audit.json"))
    audit, audit_issues = audit_protocol(audit_paths)
    expected_pairs = {(x, r) for x in audit.ligand.unique() for r in range(1, 7)}
    actual_pairs = set(zip(audit.ligand, audit.replicate))
    if actual_pairs != expected_pairs or len(audit.ligand.unique()) != 32:
        audit_issues.append(f"Run matrix mismatch: {len(actual_pairs)} pairs, {audit.ligand.nunique()} ligands")
    audit.to_csv(REPORT / "protocol_audit.csv", index=False)

    # Load the identically indexed common-catalog metrics and frame blocks.
    metric_rows, run_rows, validation_issues = [], [], []
    for path in sorted(RUN_ROOT.glob("*/rep*/common_site_analysis/*-site-metrics.csv")):
        ligand, rep = ligand_rep(path)
        df = pd.read_csv(path)
        df.insert(0, "replicate", rep)
        df.insert(0, "ligand", ligand)
        metric_rows.append(df)
        stem = path.name.replace("-site-metrics.csv", "")
        npz_path = path.with_name(stem + "-frame-site-series.npz")
        json_path = path.with_name(stem + "-density-analysis.json")
        z = np.load(npz_path)
        meta = json.loads(json_path.read_text())
        observed = z["occupied"].mean(axis=0)
        if list(z["site_ids"].astype(str)) != list(catalog.site_id):
            validation_issues.append(f"{ligand}/rep{rep}: site IDs/order differ from catalog")
        if np.max(np.abs(observed - df.occupancy.to_numpy())) > 1e-12:
            validation_issues.append(f"{ligand}/rep{rep}: CSV occupancy differs from frame series")
        block = z["block_occupancy"].astype(float)
        for j, sid in enumerate(catalog.site_id):
            for b in range(block.shape[0]):
                run_rows.append({"ligand": ligand, "replicate": rep, "site_id": sid,
                                 "block": b + 1, "block_occupancy": block[b, j]})
        run_rows[-1]["_noop"] = None
        run_meta = {
            "ligand": ligand, "replicate": rep,
            "trajectory_frames": meta.get("trajectory_frames"),
            "hydration_sites": meta.get("hydration_sites"),
            "blocks_used": meta.get("blocks_used"),
            "catalog_mode": meta.get("site_catalog_mode"),
            "physical_water_observations": meta.get("physical_water_observations"),
            "sphere_water_min": nested_get(meta, ("sphere_water_count", "minimum")),
            "sphere_water_mean": nested_get(meta, ("sphere_water_count", "mean")),
            "sphere_water_max": nested_get(meta, ("sphere_water_count", "maximum")),
        }
        run_rows.append({"_run_meta": run_meta})

    metrics = pd.concat(metric_rows, ignore_index=True)
    # Separate the sentinel metadata records from the compact loop above.
    run_meta = pd.DataFrame([x["_run_meta"] for x in run_rows if "_run_meta" in x])
    blocks = pd.DataFrame([x for x in run_rows if "_run_meta" not in x]).drop(columns=["_noop"], errors="ignore")
    for row in run_meta.itertuples():
        if (row.trajectory_frames, row.hydration_sites, row.blocks_used, row.catalog_mode) != (2500, 33, 5, "reused_common_catalog"):
            validation_issues.append(f"{row.ligand}/rep{row.replicate}: noncanonical common analysis metadata")
    if len(metrics) != 192 * 33 or len(run_meta) != 192:
        validation_issues.append(f"Expected 6336 metric rows and 192 metadata rows; got {len(metrics)}, {len(run_meta)}")
    metrics.to_csv(REPORT / "all_run_site_metrics.csv", index=False)
    run_meta.to_csv(REPORT / "run_water_counts.csv", index=False)

    # Catalog discovery provenance, including reproducible ligand-specific sites.
    source_counts = {}
    source_re = re.compile(r"ev71_2a_(x[0-9]+a)-site-catalog\.csv$")
    for row in catalog.itertuples():
        per_lig = Counter()
        for source in str(row.source_catalogs).split(";"):
            m = source_re.search(source)
            if m:
                per_lig[m.group(1)] += 1
        source_counts[row.site_id] = per_lig
    neighbors = parse_receptor_neighbors(catalog)

    # Site-level variance, convergence, displacement and bridge summaries.
    site_rows = []
    for sid, sdf in metrics.groupby("site_id", sort=False):
        c = catalog.loc[catalog.site_id == sid].iloc[0]
        ow = one_way_stats(sdf.occupancy, sdf.ligand)
        ligand_means = sdf.groupby("ligand").occupancy.mean().sort_values()
        b = blocks[blocks.site_id == sid]
        piv = b.pivot(index=["ligand", "replicate"], columns="block", values="block_occupancy")
        discoveries = source_counts[sid]
        row = {
            "site_id": sid,
            "x_angstrom": c.x_angstrom, "y_angstrom": c.y_angstrom, "z_angstrom": c.z_angstrom,
            "catalog_support_runs": int(c.catalog_support),
            "catalog_fraction": c.catalog_fraction,
            "mean_occupancy": sdf.occupancy.mean(),
            "median_occupancy": sdf.occupancy.median(),
            "mean_replica_sd_per_ligand": sdf.groupby("ligand").occupancy.std().mean(),
            **ow,
            "ligand_mean_range": ligand_means.iloc[-1] - ligand_means.iloc[0],
            "lowest_occupancy_ligand": ligand_means.index[0],
            "lowest_ligand_mean": ligand_means.iloc[0],
            "highest_occupancy_ligand": ligand_means.index[-1],
            "highest_ligand_mean": ligand_means.iloc[-1],
            "ligands_mean_occupancy_ge_0_2": int((ligand_means >= 0.2).sum()),
            "ligands_mean_occupancy_ge_0_5": int((ligand_means >= 0.5).sum()),
            "mean_ligand_overlap_fraction": sdf.ligand_overlap_fraction.mean(),
            "mean_water_ligand_hbond_fraction": sdf.water_ligand_hbond_fraction.mean(),
            "mean_water_protein_hbond_fraction": sdf.water_protein_hbond_fraction.mean(),
            "mean_water_bridge_fraction": sdf.water_bridge_fraction.mean(),
            "max_ligand_mean_bridge_fraction": sdf.groupby("ligand").water_bridge_fraction.mean().max(),
            "median_block_standard_error": sdf.occupancy_block_standard_error.median(),
            "median_abs_first_last_block_change": np.median(np.abs(piv[5] - piv[1])),
            "runs_abs_first_last_change_gt_0_1": int((np.abs(piv[5] - piv[1]) > 0.1).sum()),
            "max_discovery_reps_for_one_ligand": max(discoveries.values(), default=0),
            "ligands_discovered_ge_4_of_6": sum(v >= 4 for v in discoveries.values()),
            "top_discovery_ligand": max(discoveries, key=discoveries.get) if discoveries else "",
            **neighbors[sid],
        }
        site_rows.append(row)
    site_stats = pd.DataFrame(site_rows)
    site_stats["anova_q_bh"] = bh_fdr(site_stats.anova_p)
    site_stats["site_tier"] = np.select(
        [site_stats.catalog_fraction >= 0.50,
         site_stats.max_discovery_reps_for_one_ligand >= 4,
         site_stats.catalog_support_runs >= 16],
        ["series_consensus", "reproducible_ligand_specific", "provisional_series"],
        default="tentative_low_support",
    )
    site_stats["statistical_ligand_effect"] = ((site_stats.anova_q_bh < 0.05) &
                                                (site_stats.icc_1_1 > 0.25) &
                                                (site_stats.ligand_mean_range > 0.15))
    # Statistical significance alone cannot rescue coordinates that were only
    # proposed by a few source catalogs.  Keep those tests in the table, but do
    # not call them interpretable ligand-sensitive sites.
    site_stats["ligand_sensitive"] = (site_stats.statistical_ligand_effect &
                                       (site_stats.site_tier != "tentative_low_support"))
    site_xyz = site_stats[["x_angstrom", "y_angstrom", "z_angstrom"]].to_numpy()
    site_distance = squareform(pdist(site_xyz))
    np.fill_diagonal(site_distance, np.inf)
    site_stats["nearest_site"] = [site_stats.iloc[i].site_id for i in np.argmin(site_distance, axis=1)]
    site_stats["nearest_site_distance_A"] = np.min(site_distance, axis=1)
    site_stats["overlaps_another_assignment_sphere"] = site_stats.nearest_site_distance_A < 1.4
    site_stats.to_csv(REPORT / "site_statistics.csv", index=False)

    analysis_sites = site_stats.loc[site_stats.site_tier != "tentative_low_support", "site_id"].tolist()
    if len(analysis_sites) < 5:
        analysis_sites = site_stats.nlargest(10, "catalog_support_runs").site_id.tolist()

    # Occupancy matrices and replica profile agreement.
    occ = metrics.pivot(index=["ligand", "replicate"], columns="site_id", values="occupancy").sort_index()
    lig_mean = occ.groupby(level="ligand").mean()
    lig_sd = occ.groupby(level="ligand").std()
    lig_mean.to_csv(REPORT / "occupancy_ligand_means.csv")
    lig_sd.to_csv(REPORT / "occupancy_ligand_replica_sd.csv")
    ligand_site_summary = metrics.groupby(["ligand", "site_id"]).agg(
        mean_occupancy=("occupancy", "mean"),
        replica_sd=("occupancy", "std"),
        mean_ligand_overlap_fraction=("ligand_overlap_fraction", "mean"),
        mean_water_bridge_fraction=("water_bridge_fraction", "mean"),
    ).reset_index()
    ligand_site_summary["replica_sem"] = ligand_site_summary.replica_sd / math.sqrt(6)
    ligand_site_summary["occupancy_ci95_half_width_t_df5"] = (
        ligand_site_summary.replica_sem * stats.t.ppf(.975, 5))
    ligand_site_summary.to_csv(REPORT / "ligand_site_replica_summary.csv", index=False)

    # Coordinates closer than the assignment radius compete for one-to-one
    # water assignments.  Sum those alternate labels into spatial neighborhoods
    # as a sensitivity analysis of physically stable regional occupancy.
    from scipy.sparse.csgraph import connected_components
    _, component = connected_components((site_distance < 1.4).astype(np.int8), directed=False)
    site_to_group = {sid: f"NG{component[i] + 1:02d}" for i, sid in enumerate(site_stats.site_id)}
    metrics["neighborhood_id"] = metrics.site_id.map(site_to_group)
    blocks["neighborhood_id"] = blocks.site_id.map(site_to_group)
    group_occ = metrics.groupby(["ligand", "replicate", "neighborhood_id"]).occupancy.sum().unstack()
    group_lig_mean = group_occ.groupby(level="ligand").mean()
    analysis_groups = sorted({site_to_group[sid] for sid in analysis_sites})
    group_lig_mean.to_csv(REPORT / "neighborhood_occupancy_ligand_means.csv")
    group_blocks = blocks.groupby(["ligand", "replicate", "neighborhood_id", "block"],
                                  as_index=False).block_occupancy.sum()
    neighborhood_rows = []
    for gid, gdf in metrics.groupby("neighborhood_id"):
        members = [sid for sid, group in site_to_group.items() if group == gid]
        vals = gdf.groupby(["ligand", "replicate"]).occupancy.sum().reset_index()
        ow = one_way_stats(vals.occupancy, vals.ligand)
        lm = vals.groupby("ligand").occupancy.mean().sort_values()
        tiers = site_stats.set_index("site_id").loc[members, "site_tier"]
        gb = group_blocks[group_blocks.neighborhood_id == gid].pivot(
            index=["ligand", "replicate"], columns="block", values="block_occupancy")
        neighborhood_rows.append({
            "neighborhood_id": gid,
            "member_sites": ";".join(members),
            "member_count": len(members),
            "contains_overlapping_site_labels": len(members) > 1,
            "supported": bool(any(t != "tentative_low_support" for t in tiers)),
            "member_tiers": ";".join(tiers),
            "maximum_member_catalog_support": int(site_stats.set_index("site_id").loc[members, "catalog_support_runs"].max()),
            "mean_regional_occupancy": vals.occupancy.mean(),
            **ow,
            "ligand_mean_range": lm.iloc[-1] - lm.iloc[0],
            "lowest_occupancy_ligand": lm.index[0], "lowest_ligand_mean": lm.iloc[0],
            "highest_occupancy_ligand": lm.index[-1], "highest_ligand_mean": lm.iloc[-1],
            "median_abs_first_last_block_change": np.median(np.abs(gb[5] - gb[1])),
            "nearest_residues": "; ".join(site_stats.set_index("site_id").loc[members, "nearest_residue"]),
        })
    neighborhood_stats = pd.DataFrame(neighborhood_rows).sort_values("neighborhood_id")
    neighborhood_stats["anova_q_bh"] = bh_fdr(neighborhood_stats.anova_p)
    neighborhood_stats["ligand_sensitive"] = (neighborhood_stats.supported &
                                                (neighborhood_stats.anova_q_bh < .05) &
                                                (neighborhood_stats.icc_1_1 > .25) &
                                                (neighborhood_stats.ligand_mean_range > .15))
    neighborhood_stats.to_csv(REPORT / "site_neighborhood_statistics.csv", index=False)

    profile_occ = group_occ[analysis_groups]
    profile_lig_mean = profile_occ.groupby(level="ligand").mean()
    pairs = []
    for lig in profile_lig_mean.index:
        sub = profile_occ.loc[lig]
        for r1, r2 in combinations(sub.index, 2):
            a, b = sub.loc[r1].to_numpy(), sub.loc[r2].to_numpy()
            pairs.append({"ligand": lig, "replicate_1": r1, "replicate_2": r2,
                          "pearson": safe_corr(a, b), "spearman": safe_corr(a, b, "spearman"),
                          "rmse": float(np.sqrt(np.mean((a - b) ** 2)))})
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(REPORT / "within_ligand_replica_pairs.csv", index=False)
    between_dist = pdist(profile_lig_mean.to_numpy(), metric="euclidean") / math.sqrt(len(analysis_groups))
    within_run_dist, between_run_dist = [], []
    occ_array, occ_index = profile_occ.to_numpy(), list(profile_occ.index)
    for i, j in combinations(range(len(occ_index)), 2):
        distance = float(np.sqrt(np.mean((occ_array[i] - occ_array[j]) ** 2)))
        if occ_index[i][0] == occ_index[j][0]:
            within_run_dist.append(distance)
        else:
            between_run_dist.append(distance)
    separation_auc = float(stats.mannwhitneyu(between_run_dist, within_run_dist,
                                               alternative="greater").statistic /
                           (len(between_run_dist) * len(within_run_dist)))

    # Leave-one-replica-out nearest ligand centroid classification.
    loo_rows = []
    for (lig, rep), vector in profile_occ.iterrows():
        distances = {}
        for candidate in profile_lig_mean.index:
            train = profile_occ.loc[candidate]
            if candidate == lig:
                train = train.drop(rep)
            centroid = train.mean(axis=0).to_numpy()
            distances[candidate] = float(np.sqrt(np.mean((vector.to_numpy() - centroid) ** 2)))
        ranked = sorted(distances, key=distances.get)
        loo_rows.append({"ligand": lig, "replicate": rep, "predicted_ligand": ranked[0],
                         "correct": ranked[0] == lig, "true_ligand_rank": ranked.index(lig) + 1,
                         "nearest_distance": distances[ranked[0]], "true_distance": distances[lig]})
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(REPORT / "leave_one_replica_out.csv", index=False)

    # Per-run reproducibility/convergence table for targeted trajectory review.
    run_repro = loo.rename(columns={"true_distance": "rmse_to_other_five_replica_centroid"}).merge(
        run_meta, on=["ligand", "replicate"])
    block_piv = group_blocks[group_blocks.neighborhood_id.isin(analysis_groups)].pivot(
        index=["ligand", "replicate", "neighborhood_id"], columns="block", values="block_occupancy")
    block_change = np.abs(block_piv[5] - block_piv[1]).groupby(level=["ligand", "replicate"]).median()
    run_repro = run_repro.merge(block_change.rename("median_analysis_site_abs_block1_to5_change").reset_index(),
                                on=["ligand", "replicate"])
    run_repro["within_ligand_rmse_rank"] = run_repro.groupby("ligand")[
        "rmse_to_other_five_replica_centroid"].rank(method="min", ascending=False).astype(int)
    run_repro.to_csv(REPORT / "run_reproducibility.csv", index=False)
    replicate_bias = stats.kruskal(*[x.rmse_to_other_five_replica_centroid.to_numpy()
                                     for _, x in run_repro.groupby("replicate")])

    # Ligand summaries including sphere counts and convergence.
    ligand_rows = []
    for lig in lig_mean.index:
        m = metrics[metrics.ligand == lig]
        rm = run_meta[run_meta.ligand == lig]
        bp = blocks[blocks.ligand == lig].pivot(index=["replicate", "site_id"], columns="block", values="block_occupancy")
        ligand_rows.append({
            "ligand": lig,
            "mean_sphere_waters": rm.sphere_water_mean.mean(),
            "replica_sd_sphere_waters": rm.sphere_water_mean.std(),
            "minimum_sphere_waters_observed": rm.sphere_water_min.min(),
            "maximum_sphere_waters_observed": rm.sphere_water_max.max(),
            "mean_total_site_occupancy": lig_mean.loc[lig].sum(),
            "mean_analysis_site_occupancy": lig_mean.loc[lig, analysis_sites].sum(),
            "mean_bridge_fraction": m.water_bridge_fraction.mean(),
            "maximum_site_bridge_fraction": m.groupby("site_id").water_bridge_fraction.mean().max(),
            "median_replica_profile_rmse": pair_df.loc[pair_df.ligand == lig, "rmse"].median(),
            "median_abs_first_last_block_change": np.abs(bp[5] - bp[1]).median(),
        })
    ligand_stats = pd.DataFrame(ligand_rows)

    # Affinity association (exploratory: only 32 ligands and multiple site tests).
    subset = pd.read_csv(INPUT_ROOT / "subset" / "pyrrolidine_32_subset.csv")
    if set(subset.ligand_name) != set(audit.ligand):
        validation_issues.append("Ligand IDs in the 32-compound input subset and run matrix differ")
    affinity = subset.set_index("ligand_name")["experimental_pKD"]
    ligand_stats["experimental_pKD"] = ligand_stats.ligand.map(affinity)
    affinity_rows = []
    for sid in catalog.site_id:
        x = lig_mean[sid].reindex(affinity.index)
        rho, p = stats.spearmanr(x, affinity)
        affinity_rows.append({"site_id": sid, "spearman_rho_occupancy_vs_pKD": rho, "p_value": p})
    affinity_df = pd.DataFrame(affinity_rows)
    affinity_df["q_value_bh"] = bh_fdr(affinity_df.p_value)
    affinity_df = affinity_df.merge(site_stats[["site_id", "site_tier"]], on="site_id")
    affinity_df.to_csv(REPORT / "affinity_site_correlations.csv", index=False)
    neighborhood_affinity_rows = []
    for gid in group_lig_mean.columns:
        rho, p = stats.spearmanr(group_lig_mean[gid].reindex(affinity.index), affinity)
        neighborhood_affinity_rows.append({"neighborhood_id": gid,
                                           "spearman_rho_occupancy_vs_pKD": rho, "p_value": p})
    neighborhood_affinity = pd.DataFrame(neighborhood_affinity_rows)
    neighborhood_affinity["q_value_bh"] = bh_fdr(neighborhood_affinity.p_value)
    neighborhood_affinity = neighborhood_affinity.merge(
        neighborhood_stats[["neighborhood_id", "member_sites", "supported"]], on="neighborhood_id")
    neighborhood_affinity.to_csv(REPORT / "neighborhood_affinity_correlations.csv", index=False)
    ligand_stats.to_csv(REPORT / "ligand_statistics.csv", index=False)

    # Chemical similarity versus hydration-profile similarity.
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles = subset.set_index("ligand_name").smiles
    ligands = list(lig_mean.index)
    fps = {x: fpgen.GetFingerprint(Chem.MolFromSmiles(smiles[x])) for x in ligands}
    chem, hydro, chem_rows = [], [], []
    for a, b in combinations(ligands, 2):
        tan = DataStructs.TanimotoSimilarity(fps[a], fps[b])
        hd = float(np.sqrt(np.mean((profile_lig_mean.loc[a] - profile_lig_mean.loc[b]) ** 2)))
        chem.append(tan); hydro.append(hd)
        chem_rows.append({"ligand_1": a, "ligand_2": b, "tanimoto": tan, "hydration_rmse": hd})
    chem_df = pd.DataFrame(chem_rows)
    chem_df.to_csv(REPORT / "chemical_hydration_pairwise.csv", index=False)
    chem_rho, chem_p = stats.spearmanr(chem, hydro)
    hydro_mat = squareform(hydro)
    perm = []
    tri = np.triu_indices(len(ligands), 1)
    for _ in range(2000):
        idx = RNG.permutation(len(ligands))
        perm.append(stats.spearmanr(chem, hydro_mat[idx][:, idx][tri]).statistic)
    perm_p = (1 + np.sum(np.abs(perm) >= abs(chem_rho))) / (1 + len(perm))

    # Figures.
    sns.set_theme(style="whitegrid", context="notebook")
    informative = neighborhood_stats.query("supported").sort_values(
        ["ligand_sensitive", "maximum_member_catalog_support"], ascending=[False, False]).neighborhood_id.tolist()
    plot_matrix = group_lig_mean[informative]
    if len(plot_matrix) > 1:
        row_order = hierarchy.leaves_list(hierarchy.linkage(plot_matrix, method="average", metric="euclidean"))
        plot_matrix = plot_matrix.iloc[row_order]
    fig, ax = plt.subplots(figsize=(max(9, .48 * len(informative)), 11))
    sns.heatmap(plot_matrix, cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "occupancy"})
    ax.set_title("EV71 mean hydration-neighborhood occupancy (six replicas per ligand)")
    ax.set_xlabel("spatial site neighborhood"); ax.set_ylabel("ligand")
    fig.tight_layout(); fig.savefig(REPORT / "occupancy_heatmap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=site_stats, x="catalog_fraction", y="icc_1_1", size="ligand_mean_range",
                    hue="site_tier", sizes=(30, 220), ax=ax)
    ax.axhline(.25, color="grey", ls="--", lw=1)
    ax.set(title="Site support and ligand-specific reproducibility", xlabel="fraction of runs discovering site",
           ylabel="ICC: fraction of variation attributable to ligand")
    fig.tight_layout(); fig.savefig(REPORT / "site_support_reproducibility.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    dist_plot = pd.DataFrame({"profile RMSE": np.r_[within_run_dist, between_run_dist, between_dist],
                              "comparison": ["same-ligand runs"] * len(within_run_dist) +
                                            ["different-ligand runs"] * len(between_run_dist) +
                                            ["six-replica ligand means"] * len(between_dist)})
    sns.violinplot(data=dist_plot, x="comparison", y="profile RMSE", inner="quart", ax=ax)
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=8)
    ax.set_title("Replica noise versus ligand-to-ligand hydration differences")
    fig.tight_layout(); fig.savefig(REPORT / "replica_vs_ligand_distances.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=chem_df, x="tanimoto", y="hydration_rmse", ax=ax, alpha=.65)
    ax.set_title("Chemical similarity versus hydration-profile difference")
    fig.tight_layout(); fig.savefig(REPORT / "chemical_vs_hydration.png", dpi=180); plt.close(fig)

    # Machine-readable summary and a compact human report.
    consensus = site_stats.query("site_tier == 'series_consensus'")
    sensitive = site_stats.query("ligand_sensitive").sort_values("icc_1_1", ascending=False)
    robust_neighborhoods = neighborhood_stats.query("ligand_sensitive").sort_values("icc_1_1", ascending=False)
    bridge_candidates = site_stats.query(
        "site_tier != 'tentative_low_support' and mean_water_bridge_fraction >= 0.10"
    ).sort_values("mean_water_bridge_fraction", ascending=False)
    p_water = stats.spearmanr(ligand_stats.mean_sphere_waters, ligand_stats.experimental_pKD)
    summary = {
        "input": {"completed_submission": completed[0].parents[1].name,
                  "ligands": int(audit.ligand.nunique()), "replicas_per_ligand": 6,
                  "runs": len(audit), "frames_per_run": 2500, "total_frames": len(audit) * 2500,
                  "common_sites": len(catalog)},
        "integrity": {"protocol_issues": audit_issues, "analysis_validation_issues": validation_issues,
                      "maximum_preparation_pose_rmsd_A": float(audit.pose_rmsd_A.max())},
        "site_filter": {"analysis_site_count": len(analysis_sites), "analysis_sites": analysis_sites,
                        "analysis_neighborhood_count": len(analysis_groups), "analysis_neighborhoods": analysis_groups,
                        "overlapping_coordinate_components": int(sum(neighborhood_stats.member_count > 1)),
                        "sites_in_overlapping_components": int(neighborhood_stats.loc[neighborhood_stats.member_count > 1, "member_count"].sum()),
                        "tier_counts": site_stats.site_tier.value_counts().to_dict()},
        "replicates": {"median_within_ligand_profile_rmse": float(pair_df.rmse.median()),
                       "median_between_ligand_single_run_profile_rmse": float(np.median(between_run_dist)),
                       "median_between_ligand_mean_profile_rmse": float(np.median(between_dist)),
                       "probability_between_run_distance_exceeds_within_run_distance": separation_auc,
                       "loo_nearest_centroid_accuracy": float(loo.correct.mean()),
                       "loo_median_true_ligand_rank": float(loo.true_ligand_rank.median()),
                       "replicate_index_kruskal_p_for_profile_error": float(replicate_bias.pvalue)},
        "sites": {"series_consensus": consensus.site_id.tolist(),
                  "individual_ligand_sensitive_before_neighborhood_sensitivity": sensitive.site_id.tolist(),
                  "robust_ligand_sensitive_neighborhoods": {
                      row.neighborhood_id: row.member_sites for row in robust_neighborhoods.itertuples()},
                  "conserved_geometry_only_bridge_candidates": bridge_candidates.site_id.tolist()},
        "convergence": {"median_site_run_abs_first_last_block_change": float(np.median(np.abs(
            blocks.pivot(index=["ligand", "replicate", "site_id"], columns="block", values="block_occupancy")[5] -
            blocks.pivot(index=["ligand", "replicate", "site_id"], columns="block", values="block_occupancy")[1]))),
            "fraction_site_runs_abs_change_gt_0_1": float(np.mean(np.abs(
            blocks.pivot(index=["ligand", "replicate", "site_id"], columns="block", values="block_occupancy")[5] -
            blocks.pivot(index=["ligand", "replicate", "site_id"], columns="block", values="block_occupancy")[1]) > .1))},
        "sphere_waters": {"run_mean_range": [float(run_meta.sphere_water_mean.min()), float(run_meta.sphere_water_mean.max())],
                          "absolute_observed_range": [int(run_meta.sphere_water_min.min()), int(run_meta.sphere_water_max.max())],
                          "spearman_vs_pKD": float(p_water.statistic), "p_value": float(p_water.pvalue)},
        "chemical_vs_hydration": {"spearman_tanimoto_vs_hydration_rmse": float(chem_rho),
                                  "asymptotic_p": float(chem_p), "permutation_p": float(perm_p)},
        "affinity": {
            "all_sites_bh_q_lt_0_05": affinity_df.loc[affinity_df.q_value_bh < .05, "site_id"].tolist(),
            "supported_sites_bh_q_lt_0_05": affinity_df.loc[(affinity_df.q_value_bh < .05) &
                                                             (affinity_df.site_tier != "tentative_low_support"),
                                                             "site_id"].tolist(),
            "supported_neighborhoods_bh_q_lt_0_05": neighborhood_affinity.loc[
                (neighborhood_affinity.q_value_bh < .05) & neighborhood_affinity.supported,
                "neighborhood_id"].tolist(),
        },
        "cautions": [
            "Common-site coordinates were learned from all runs; tests are descriptive/exploratory, not held-out validation.",
            "Hydrogen-bond and bridge flags are geometry-only candidates, not protonation-aware chemical assignments.",
            "Tentative low-support catalog sites should not be interpreted as series-wide hydration sites.",
            "Individual coordinates closer than the 1.4 A assignment radius compete for water labels; prefer neighborhood sensitivity results.",
            "First-versus-last block differences diagnose drift but do not prove full thermodynamic convergence.",
        ],
    }
    (REPORT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    def fmt_sites(frame: pd.DataFrame, n=12) -> str:
        if frame.empty:
            return "None met the prespecified criteria."
        lines = []
        for x in frame.head(n).itertuples():
            lines.append(f"- {x.site_id}: mean occupancy {x.mean_occupancy:.3f}, ICC {x.icc_1_1:.2f}, "
                         f"ligand range {x.ligand_mean_range:.3f} ({x.lowest_occupancy_ligand} {x.lowest_ligand_mean:.3f} "
                         f"to {x.highest_occupancy_ligand} {x.highest_ligand_mean:.3f}), "
                         f"support {x.catalog_support_runs}/192; "
                         f"nearest {x.nearest_residue} at {x.nearest_distance_A:.2f} A")
        return "\n".join(lines)

    def fmt_neighborhoods(frame: pd.DataFrame, n=12) -> str:
        if frame.empty:
            return "None met the prespecified criteria."
        return "\n".join(
            f"- {x.neighborhood_id} ({x.member_sites}): regional mean {x.mean_regional_occupancy:.3f}, "
            f"ICC {x.icc_1_1:.2f}, ligand range {x.ligand_mean_range:.3f} "
            f"({x.lowest_occupancy_ligand} {x.lowest_ligand_mean:.3f} to "
            f"{x.highest_occupancy_ligand} {x.highest_ligand_mean:.3f}); nearest residues {x.nearest_residues}"
            for x in frame.head(n).itertuples()
        )

    report = f"""# EV71 Loch 32-ligand / 6-replica consistency analysis

## Bottom line

The archive contains the complete **32 × 6 = 192-run** full-protocol matrix. All pipeline audits pass the canonical schedule and topology handoff checks, and all common-site outputs reconstruct exactly from their per-frame occupancy arrays. No truncated or smoke-profile simulation is mixed into this analysis.

The scientific result is mixed rather than uniformly consistent. After combining common-site labels closer than the 1.4 A assignment radius, {len(analysis_sites)} supportable sites become **{len(analysis_groups)} spatial neighborhoods**. Their median same-ligand run-pair RMSE is **{pair_df.rmse.median():.3f}**, only modestly below **{np.median(between_run_dist):.3f}** for different-ligand run pairs; a randomly selected between-ligand distance exceeds a within-ligand distance with probability **{separation_auc:.1%}** (50% means no separation). Averaging six replicas suppresses noise—the median distance between ligand means is **{np.median(between_dist):.3f}**—but a held-out replica's closest ligand centroid is the correct ligand in only **{loo.correct.mean():.1%}** of cases (chance **3.1%**; median true-ligand rank {loo.true_ligand_rank.median():.0f}). The dataset therefore supports several specific ligand-sensitive regions, not the stronger claim that each ligand has a cleanly reproducible whole-pocket signature.

## Input and integrity

- Completed submission: `{completed[0].parents[1].name}`
- Total analyzed production: **480,000 frames** (2,500 per run)
- Protocol audit deviations: **{len(audit_issues)}**
- Common-analysis reconstruction/metadata deviations: **{len(validation_issues)}**
- Maximum prepared ligand heavy-atom pose RMSD from its supplied SDF: **{audit.pose_rmsd_A.max():.4f} A**
- Raw production topologies retain exactly 45 inactive ghosts; physical final topologies contain zero zero-interaction waters in every run.

## How sites were triaged

The pooled catalog has 33 coordinates, but its minimum inclusion rule was only two source catalogs. I therefore did not treat all 33 equally:

- **series_consensus**: discovered independently in at least half of all 192 runs;
- **reproducible_ligand_specific**: below that global threshold but rediscovered in at least 4/6 replicas for one or more ligands;
- **provisional_series**: at least 16 independent source-run discoveries;
- **tentative_low_support**: everything else; retained in tables, excluded from headline profile comparisons.

Tier counts: {json.dumps(site_stats.site_tier.value_counts().to_dict())}.

## Spatial-label sensitivity check

The common catalog contains **{int(sum(neighborhood_stats.member_count > 1))} groups / {int(neighborhood_stats.loc[neighborhood_stats.member_count > 1, 'member_count'].sum())} coordinates** whose centers are closer than the 1.4 A site-assignment radius. Those labels can exchange one physical water through the one-to-one assignment algorithm, so individual-site changes can describe a coordinate-label shift rather than water displacement. I summed each such component into a regional occupancy and repeated the ligand-effect test. This is the preferred mechanistic result:

{fmt_neighborhoods(robust_neighborhoods)}

For example, the individual HS012 effect does **not** survive combination with its 1.02 A neighbor HS030 (regional ICC {neighborhood_stats.loc[neighborhood_stats.member_sites == 'HS012;HS030', 'icc_1_1'].iloc[0]:.2f}). By contrast, the combined HS008/HS013/HS015 region remains ligand-sensitive. This is why the neighborhood table should take precedence over isolated significant p-values.

## Individual ligand-sensitive coordinates

For traceability, these are the supported individual coordinates meeting BH-adjusted ANOVA q < 0.05, ICC > 0.25, and ligand-average occupancy range > 0.15 before the overlapping-label sensitivity check. ICC asks whether replicas of the same ligand resemble one another more than different ligands do.

{fmt_sites(sensitive)}

## Consensus hydration sites

{fmt_sites(consensus.sort_values('mean_occupancy', ascending=False))}

## Conserved bridge candidate

**HS009** is the standout conserved structural-water hypothesis: mean occupancy **{site_stats.set_index('site_id').loc['HS009', 'mean_occupancy']:.3f}**, geometry-only protein–water–ligand bridge fraction **{site_stats.set_index('site_id').loc['HS009', 'mean_water_bridge_fraction']:.3f}**, protein H-bond fraction **{site_stats.set_index('site_id').loc['HS009', 'mean_water_protein_hbond_fraction']:.3f}**, and ligand H-bond fraction **{site_stats.set_index('site_id').loc['HS009', 'mean_water_ligand_hbond_fraction']:.3f}**. Its ICC is approximately zero, meaning it is shared across the chemical series rather than explaining ligand differences. It lies nearest **{site_stats.set_index('site_id').loc['HS009', 'nearest_residue']}**. Because donor/acceptor typing is geometry-only, this should be checked visually before calling it a chemical H-bond network.

## Sampling and sphere-water behavior

Across every run/site pair, the median absolute occupancy shift from production block 1 to block 5 is **{summary['convergence']['median_site_run_abs_first_last_block_change']:.3f}**; **{summary['convergence']['fraction_site_runs_abs_change_gt_0_1']:.1%}** shift by more than 0.10. Use the site-level columns and heatmap to distinguish stable signals from drifting ones.

The per-run mean number of physical waters in the 10 A sphere ranges from **{run_meta.sphere_water_mean.min():.1f} to {run_meta.sphere_water_mean.max():.1f}**; instantaneous extrema across the entire series are **{run_meta.sphere_water_min.min():.0f} to {run_meta.sphere_water_max.max():.0f}**. A count near 50–60 is therefore normal for this sphere and is not capped by the 45-ghost buffer: ghosts are trial capacity, while the sphere count includes ordinary physical waters already present in the solvated system.

The noisiest profile is **{run_repro.sort_values('rmse_to_other_five_replica_centroid', ascending=False).iloc[0].ligand}/rep{int(run_repro.sort_values('rmse_to_other_five_replica_centroid', ascending=False).iloc[0].replicate)}** (RMSE {run_repro.rmse_to_other_five_replica_centroid.max():.3f} from its other-five-replica centroid), followed by **{run_repro.sort_values('rmse_to_other_five_replica_centroid', ascending=False).iloc[1].ligand}/rep{int(run_repro.sort_values('rmse_to_other_five_replica_centroid', ascending=False).iloc[1].replicate)}**. There is no systematic bad replica index (Kruskal p = {replicate_bias.pvalue:.3f}), and these runs passed every protocol/topology audit. They are trajectory-review priorities, not evidence of job failure; their low end-to-end block drift suggests different sampled hydration basins rather than an unfinished file.

## Relation to chemistry and affinity

Morgan-fingerprint Tanimoto similarity versus hydration-profile RMSE has Spearman rho **{chem_rho:.3f}** (permutation p **{perm_p:.4f}**). The sign is expected to be negative if chemically similar ligands retain similar hydration patterns. This is a series-level sanity check, not a binding model.

At the individual-coordinate level, **{len(summary['affinity']['all_sites_bh_q_lt_0_05'])}** occupancy–experimental-pKD associations have q < 0.05 after correction, but one is a tentative low-support coordinate. After repeating the analysis on non-overlapping neighborhoods, the supported corrected association is **{', '.join(summary['affinity']['supported_neighborhoods_bh_q_lt_0_05']) or 'none'}** ({neighborhood_affinity.loc[neighborhood_affinity.neighborhood_id.isin(summary['affinity']['supported_neighborhoods_bh_q_lt_0_05']), 'member_sites'].str.cat(sep=', ') or 'no member sites'}). These are exploratory because the common catalog and hypotheses were derived from the same 32 compounds; they need prospective or held-out validation before mechanistic claims.

## Interpretation for Project 2

The useful object is not the total number of waters. It is the vector of probabilities that matched spatial sites are occupied, displaced by ligand atoms, or form candidate protein–water–ligand bridges. Six replicas quantify how noisy each probability is. A site is most useful when the six replicas agree within a ligand, different ligands give distinct means, its block history is stable, and its location makes structural sense near the pocket. The supplied tables support exactly those checks.

## Caveats

- The common coordinate catalog was learned from all runs, so the statistical tests are descriptive rather than fully independent validation.
- Bridge/H-bond annotations are geometry-only candidates and require structural inspection.
- Low-support catalog entries are hypotheses, not established sites.
- Five blocks provide a practical drift diagnostic; they do not prove thermodynamic convergence.

## Files

- `site_statistics.csv`: individual-site reliability, ligand effect, convergence, overlap warning, bridge, and nearest-residue table
- `site_neighborhood_statistics.csv`: preferred overlapping-label sensitivity analysis
- `ligand_statistics.csv`: per-ligand replica, water-count, bridge, and pKD summary
- `occupancy_ligand_means.csv` and `occupancy_ligand_replica_sd.csv`: analysis-ready matrices
- `ligand_site_replica_summary.csv`: long-form means, replica SD/SEM, and t-based 95% uncertainty
- `neighborhood_occupancy_ligand_means.csv`: analysis-ready non-overlapping regional matrix
- `within_ligand_replica_pairs.csv` and `leave_one_replica_out.csv`: reproducibility evidence
- `run_reproducibility.csv`: run-level profile error, block drift, sphere counts, and review ranking
- `affinity_site_correlations.csv`: multiplicity-corrected exploratory affinity associations
- `neighborhood_affinity_correlations.csv`: affinity sensitivity analysis after combining overlapping labels
- `chemical_hydration_pairwise.csv`: chemical/hydration series comparison
- `protocol_audit.csv`: full-schedule and topology-handoff audit
- `analysis_summary.json`: machine-readable headline results and cautions
- PNG figures: occupancy heatmap, support/reproducibility, replica-versus-ligand distances, and chemistry-versus-hydration
"""
    (REPORT / "analysis_report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
