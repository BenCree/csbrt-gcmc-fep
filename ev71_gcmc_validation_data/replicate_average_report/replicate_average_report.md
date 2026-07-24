# EV71 analysis based on six-replica ligand averages

## Correct unit of analysis

Each of the 32 ligands is represented by the **mean of its six independent replicas** at each matched hydration neighborhood. The six values are retained as the sampling distribution: minimum, maximum, range, SD, SEM, and a t-based 95% confidence interval are all reported. Individual replicas are not treated as 192 independent ligands.

## Main result

The ligand-average hydration profiles agree strongly in their broad pattern: across all 496 ligand pairs, the median Pearson correlation between mean profiles is **0.917** (10th–90th percentile 0.836–0.962). The median RMSE between ligand means is **0.094** occupancy units.

Replica variability is substantial but now quantified directly. Across all **544 ligand × neighborhood estimates**, the median max-minus-min range among six replicas is **0.216** (IQR 0.114–0.360; 90th percentile 0.530). Averaging reduces the median RMS standard error of an entire ligand profile to **0.056**. This is the value of the replicas: the ligand means are appreciably more precise than any one run, and every reported ligand difference now has an empirical uncertainty.

## Where ligand means differ beyond replica variation

The most useful comparison is the range among the 32 ligand means divided by the typical range among six replicas of one ligand. Values above one mean that the full chemical-series span exceeds typical replica spread.

- NG16 (HS019): ligand-mean range 0.479; median six-replica range 0.183; ICC 0.50; ratio 2.62
- NG15 (HS018): ligand-mean range 0.164; median six-replica range 0.080; ICC 0.47; ratio 2.04
- NG08 (HS008;HS013;HS015): ligand-mean range 0.264; median six-replica range 0.142; ICC 0.40; ratio 1.85
- NG21 (HS027): ligand-mean range 0.396; median six-replica range 0.240; ICC 0.33; ratio 1.65
- NG03 (HS003;HS023): ligand-mean range 0.408; median six-replica range 0.282; ICC 0.27; ratio 1.45
- NG13 (HS014;HS016): ligand-mean range 0.146; median six-replica range 0.150; ICC 0.17; ratio 0.97
- NG14 (HS017): ligand-mean range 0.095; median six-replica range 0.103; ICC 0.15; ratio 0.92
- NG02 (HS002): ligand-mean range 0.288; median six-replica range 0.377; ICC 0.05; ratio 0.76

The prespecified combination of corrected ANOVA q < 0.05, ICC > 0.25, and between-ligand range greater than the median within-ligand range retains **NG16, NG15, NG08, NG21, NG03**. These are the neighborhoods where ligand-average differences are most defensible. Other regions may be conserved, or their apparent ligand differences are not larger than replica noise.

## Distribution of replica ranges by ligand

The five ligands with the largest median site range are **x7101a, x7259a, x7570a, x7031a, x7132a** (0.400, 0.295, 0.289, 0.279, 0.272). The five most internally consistent are **x7135a, x7024a, x7316a, x7427a, x7161a** (0.128, 0.143, 0.145, 0.152, 0.153). This is a distribution over all supported neighborhoods, not a judgment based on one outlier site.

## Do the ligand averages form clusters?

I hierarchically clustered ligand means only on the reproducible discriminatory neighborhoods, after scaling each neighborhood by its pooled within-ligand replica SD. The best silhouette is **0.471** at k=2, with cluster sizes {'1': 2, '2': 30}. Because this solution is imbalanced and replica-bootstrap co-assignment is not uniformly high, the dendrogram should be read as a similarity map rather than evidence for sharply separated ligand classes. The continuous pairwise distance and bootstrap tables are safer than hard labels.

## How to use the outputs

- Start with `ligand_neighborhood_replica_summary.csv`: one row per ligand and neighborhood, with the six-replica mean and full range/uncertainty.
- `ligand_replica_range_summary.csv` gives the requested distribution-of-ranges summary for each ligand.
- `neighborhood_between_ligand_vs_replica_noise.csv` identifies where differences among ligand means exceed replicate variability.
- `ligand_pairwise_mean_profile_comparison.csv` compares every pair of ligand means with replica-bootstrap uncertainty.
- `ligand_mean_profiles.csv` is the 32 × 17 matrix for downstream modelling.
- The heatmap and PCA show averages and replica spread; the cluster co-assignment matrix shows which branches survive replica resampling.

## Limits

The six replicas estimate simulation repeatability, not experimental error. The common spatial catalog is needed so all ligand means refer to the same locations; nearby competing site labels were combined into non-overlapping neighborhoods before averaging. Confidence intervals are descriptive with only six replicas, and the trajectory blocks are correlated, so the independent unit remains the replica—not the frame.
