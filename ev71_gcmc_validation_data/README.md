# EV-A71 2A Loch GCMC — hydration consistency validation data

Analysis outputs from the **32-ligand × 6-replica** Loch endpoint run on the
EV-A71 2A protease pyrrolidine series (192 full-protocol GCMC simulations,
480,000 production frames). These are the validation graphs and raw data for the
Loch hydration-mapping pipeline — they characterise how **reproducible** the
per-ligand hydration signal is, not binding free energies.

## Bottom line

All 192 runs pass the protocol/topology audits, and every common-site output
reconstructs exactly from its per-frame occupancy arrays. The scientific signal is
**mixed**: hydration profiles are broadly conserved across ligands, and averaging
six replicas sharpens each ligand's estimate, but only a handful of pocket regions
differ across ligands beyond replicate noise. The data support several specific
ligand-sensitive water regions, not a clean whole-pocket fingerprint per ligand.
Full prose is in `report/analysis_report.md` and
`replicate_average_report/replicate_average_report.md`.

## Contents

- `report/` — per-run consistency analysis (all 192 runs as the unit).
- `replicate_average_report/` — the six-replica **ligand-average** analysis (the
  statistically correct unit: 32 ligand means with replicate uncertainty).
- `colleague_summary/` — six narrative slides summarising the result for a
  non-specialist reader.
- `single_ligand_density_example/` — raw hydration-density + site outputs for one
  ligand (x7259a): 3D water density (`.npz`/`.dx`), the discovered site catalog,
  hydration-site PDB, and per-frame site occupancy (`-frame-site-series.npz`).
- `generator_scripts/` — the scripts that produced these reports (traceability).

Raw tables (`.csv`/`.json`) accompany every figure in each report dir; the report
`.md` files name the key ones.

## What each graph shows

### report/ (per-run reproducibility)
- **occupancy_heatmap.png** — hydration-site occupancy for every site × ligand;
  the overall wet/dry pattern of the pocket across the series.
- **site_support_reproducibility.png** — how consistently each hydration site is
  independently rediscovered across the 192 runs (site support / triage tiers).
- **replica_vs_ligand_distances.png** — distribution of hydration-profile distances
  for same-ligand run pairs vs different-ligand run pairs; the two overlap heavily
  (a random between-ligand distance exceeds a within-ligand one only ~55% of the
  time), i.e. per-run separation between ligands is weak.
- **chemical_vs_hydration.png** — ligand chemical similarity vs hydration-profile
  similarity, testing whether chemically similar ligands hydrate similarly.

### replicate_average_report/ (six-replica ligand averages)
- **ligand_average_profile_heatmap.png** — the 32-ligand × 17-neighborhood matrix
  of mean occupancies (the averaged pocket fingerprint per ligand).
- **replica_ranges_by_ligand.png** — per-ligand distribution of the max−min spread
  across its six replicas; which ligands are internally consistent vs variable.
- **between_ligand_signal_vs_replica_range.png** — for each pocket region, the
  spread among ligand means vs the typical spread among six replicas of one ligand;
  regions with ratio > 1 (signal exceeds replicate noise) are the defensible ones.
- **ligand_mean_pairwise_distance.png** — pairwise distances between the 32 ligand
  mean profiles (a similarity map of ligands by hydration).
- **ligand_means_with_replica_spread_pca.png** — PCA of the ligand means with each
  ligand's replicate spread drawn in, showing separation relative to noise.

### colleague_summary/ (narrative, read in order)
- **01_study_overview.png** — study design and the one-sentence result.
- **02_ligand_profile_agreement.png** — ligand-average profiles are broadly similar
  (same wet/dry pattern; small absolute differences).
- **03_replica_range_distributions.png** — what the six replicas say about how
  repeatable each ligand result is.
- **04_signal_vs_replica_noise.png** — the five water regions that differ across
  ligands beyond typical replicate variation.
- **05_key_region_heatmap.png** — those local differences shown on top of the
  otherwise conserved hydration network.
- **06_interpretation_and_caveats.png** — what can and cannot be claimed.

## Caveats

The six replicas estimate **simulation repeatability, not experimental error**. A
shared common-site catalog is required so all ligand means refer to the same
locations; nearby competing site labels were merged into non-overlapping
neighborhoods before averaging. With only six replicas the confidence intervals
are descriptive, and trajectory blocks are correlated — the independent unit is the
replica, not the frame.
