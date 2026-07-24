# EV71 Loch 32-ligand / 6-replica consistency analysis

## Bottom line

The archive contains the complete **32 × 6 = 192-run** full-protocol matrix. All pipeline audits pass the canonical schedule and topology handoff checks, and all common-site outputs reconstruct exactly from their per-frame occupancy arrays. No truncated or smoke-profile simulation is mixed into this analysis.

The scientific result is mixed rather than uniformly consistent. After combining common-site labels closer than the 1.4 A assignment radius, 20 supportable sites become **17 spatial neighborhoods**. Their median same-ligand run-pair RMSE is **0.175**, only modestly below **0.182** for different-ligand run pairs; a randomly selected between-ligand distance exceeds a within-ligand distance with probability **54.7%** (50% means no separation). Averaging six replicas suppresses noise—the median distance between ligand means is **0.094**—but a held-out replica's closest ligand centroid is the correct ligand in only **12.5%** of cases (chance **3.1%**; median true-ligand rank 12). The dataset therefore supports several specific ligand-sensitive regions, not the stronger claim that each ligand has a cleanly reproducible whole-pocket signature.

## Input and integrity

- Completed submission: `20260716T053049Z-1509066`
- Total analyzed production: **480,000 frames** (2,500 per run)
- Protocol audit deviations: **0**
- Common-analysis reconstruction/metadata deviations: **0**
- Maximum prepared ligand heavy-atom pose RMSD from its supplied SDF: **0.0006 A**
- Raw production topologies retain exactly 45 inactive ghosts; physical final topologies contain zero zero-interaction waters in every run.

## How sites were triaged

The pooled catalog has 33 coordinates, but its minimum inclusion rule was only two source catalogs. I therefore did not treat all 33 equally:

- **series_consensus**: discovered independently in at least half of all 192 runs;
- **reproducible_ligand_specific**: below that global threshold but rediscovered in at least 4/6 replicas for one or more ligands;
- **provisional_series**: at least 16 independent source-run discoveries;
- **tentative_low_support**: everything else; retained in tables, excluded from headline profile comparisons.

Tier counts: {"reproducible_ligand_specific": 13, "tentative_low_support": 13, "series_consensus": 5, "provisional_series": 2}.

## Spatial-label sensitivity check

The common catalog contains **7 groups / 15 coordinates** whose centers are closer than the 1.4 A site-assignment radius. Those labels can exchange one physical water through the one-to-one assignment algorithm, so individual-site changes can describe a coordinate-label shift rather than water displacement. I summed each such component into a regional occupancy and repeated the ligand-effect test. This is the preferred mechanistic result:

- NG16 (HS019): regional mean 0.487, ICC 0.50, ligand range 0.479 (x7247a 0.129 to x6832a 0.608); nearest residues SER A:87
- NG15 (HS018): regional mean 0.386, ICC 0.47, ligand range 0.164 (x7247a 0.280 to x7491a 0.444); nearest residues ALA A:86
- NG08 (HS008;HS013;HS015): regional mean 0.832, ICC 0.40, ligand range 0.264 (x7309a 0.670 to x7257a 0.934); nearest residues GLY A:128; GLY A:128; GLY A:128
- NG21 (HS027): regional mean 0.278, ICC 0.33, ligand range 0.396 (x6738a 0.077 to x7247a 0.473); nearest residues ALA A:86
- NG03 (HS003;HS023): regional mean 0.686, ICC 0.27, ligand range 0.408 (x7589a 0.494 to x7309a 0.902); nearest residues GLY A:108; GLY A:108

For example, the individual HS012 effect does **not** survive combination with its 1.02 A neighbor HS030 (regional ICC 0.04). By contrast, the combined HS008/HS013/HS015 region remains ligand-sensitive. This is why the neighborhood table should take precedence over isolated significant p-values.

## Individual ligand-sensitive coordinates

For traceability, these are the supported individual coordinates meeting BH-adjusted ANOVA q < 0.05, ICC > 0.25, and ligand-average occupancy range > 0.15 before the overlapping-label sensitivity check. ICC asks whether replicas of the same ligand resemble one another more than different ligands do.

- HS015: mean occupancy 0.235, ICC 0.63, ligand range 0.360 (x7247a 0.068 to x7093a 0.429), support 26/192; nearest GLY A:128 at 3.74 A
- HS019: mean occupancy 0.487, ICC 0.50, ligand range 0.479 (x7247a 0.129 to x6832a 0.608), support 12/192; nearest SER A:87 at 3.56 A
- HS012: mean occupancy 0.404, ICC 0.47, ligand range 0.426 (x7247a 0.111 to x7309a 0.537), support 46/192; nearest GLY A:127 at 2.84 A
- HS018: mean occupancy 0.386, ICC 0.47, ligand range 0.164 (x7247a 0.280 to x7491a 0.444), support 15/192; nearest ALA A:86 at 2.83 A
- HS008: mean occupancy 0.264, ICC 0.38, ligand range 0.271 (x6832a 0.161 to x7510a 0.432), support 69/192; nearest GLY A:128 at 4.46 A
- HS027: mean occupancy 0.278, ICC 0.33, ligand range 0.396 (x6738a 0.077 to x7247a 0.473), support 5/192; nearest ALA A:86 at 2.78 A

## Consensus hydration sites

- HS001: mean occupancy 0.880, ICC -0.01, ligand range 0.063 (x7491a 0.851 to x7093a 0.914), support 190/192; nearest HIS A:21 at 2.80 A
- HS004: mean occupancy 0.820, ICC 0.04, ligand range 0.665 (x7247a 0.335 to x7024a 1.000), support 127/192; nearest PRO A:91 at 2.60 A
- HS002: mean occupancy 0.795, ICC 0.05, ligand range 0.288 (x7175a 0.624 to x7475a 0.912), support 178/192; nearest GLY A:103 at 2.83 A
- HS005: mean occupancy 0.509, ICC 0.07, ligand range 0.135 (x7147a 0.433 to x7491a 0.568), support 101/192; nearest SER A:105 at 2.90 A
- HS003: mean occupancy 0.293, ICC 0.14, ligand range 0.217 (x7056a 0.220 to x7132a 0.438), support 145/192; nearest GLY A:108 at 3.21 A

## Conserved bridge candidate

**HS009** is the standout conserved structural-water hypothesis: mean occupancy **0.967**, geometry-only protein–water–ligand bridge fraction **0.468**, protein H-bond fraction **0.943**, and ligand H-bond fraction **0.484**. Its ICC is approximately zero, meaning it is shared across the chemical series rather than explaining ligand differences. It lies nearest **GLU A:85**. Because donor/acceptor typing is geometry-only, this should be checked visually before calling it a chemical H-bond network.

## Sampling and sphere-water behavior

Across every run/site pair, the median absolute occupancy shift from production block 1 to block 5 is **0.048**; **24.8%** shift by more than 0.10. Use the site-level columns and heatmap to distinguish stable signals from drifting ones.

The per-run mean number of physical waters in the 10 A sphere ranges from **46.6 to 66.5**; instantaneous extrema across the entire series are **36 to 83**. A count near 50–60 is therefore normal for this sphere and is not capped by the 45-ghost buffer: ghosts are trial capacity, while the sphere count includes ordinary physical waters already present in the solvated system.

The noisiest profile is **x7101a/rep2** (RMSE 0.392 from its other-five-replica centroid), followed by **x7259a/rep2**. There is no systematic bad replica index (Kruskal p = 0.715), and these runs passed every protocol/topology audit. They are trajectory-review priorities, not evidence of job failure; their low end-to-end block drift suggests different sampled hydration basins rather than an unfinished file.

## Relation to chemistry and affinity

Morgan-fingerprint Tanimoto similarity versus hydration-profile RMSE has Spearman rho **-0.177** (permutation p **0.0300**). The sign is expected to be negative if chemically similar ligands retain similar hydration patterns. This is a series-level sanity check, not a binding model.

At the individual-coordinate level, **4** occupancy–experimental-pKD associations have q < 0.05 after correction, but one is a tentative low-support coordinate. After repeating the analysis on non-overlapping neighborhoods, the supported corrected association is **NG05** (HS005;HS028). These are exploratory because the common catalog and hypotheses were derived from the same 32 compounds; they need prospective or held-out validation before mechanistic claims.

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
