# Loch endpoint + FEP pipeline

A checkpointed pipeline for hydration-aware relative binding free energy (RBFE)
on protein/ligand systems, built around **Loch GCMC** water sampling. It has two
branches that share one idea: use Loch to place the pocket waters, then reuse that
equilibrated water state everywhere downstream.

1. **Endpoint MD/GCMC** — preparation → Loch equilibration (UVT1/NPT/UVT2) → Loch
   GCMC production → hydration-density/site analysis. This is where the waters are
   placed. It is **not** FEP.
2. **Relative FEP** — map/merge a reviewed ligand pair, run bound+free legs with
   SOMD2, and report `DDG_bind = DG_bound − DG_free`, then fit the edge network
   and compare against a reference benchmark.

The two branches connect through a small **averaging** step: because each ligand
has several endpoint replicates, one representative ("medoid") equilibrated frame
is selected per ligand and fed into FEP as the bound starting structure, so FEP
reuses the placed waters and **recomputes no GCMC** (bound-leg GCMC is off by
default).

## Environments (Mamba)

- Endpoint MD/GCMC + averaging: `cry-loch-babel` (Loch 2025.2 / Sire 2025.4 /
  OpenMM 8.4). Do not add SOMD2 to this env.
- FEP: `automated-fep`, from `environment-fep.yml` (SOMD2 2026.1 + `cuda-nvvm`).

```bash
mamba env create -f environment-fep.yml
```

## Quick start (FEP against an existing density series)

```bash
# 1. medoid bound frame per ligand (inspect the report before continuing)
mamba activate cry-loch-babel
python scripts/select_bound_frames.py \
  --series-root  /path/to/ev71-density-series \
  --output-root  /path/to/ev71-density-series/bound_frames

# 2. resolve reviewed edges to a manifest seeded from those frames
python scripts/make_fep_manifest.py \
  --edges fep_edges/rowan_xtal_edges_full.tsv \
  --endpoint-run-root /path/to/ev71-density-series \
  --bound-frame-root  /path/to/ev71-density-series/bound_frames \
  --output fep_manifest.tsv

# 3. submit all edges: one GPU each, N at a time; then network fit + benchmark compare
mamba activate automated-fep
./scripts/submit_fep_edges.sh --manifest fep_manifest.tsv --batch 24 \
  --run-root fep-runs \
  --rowan-edges  <benchmark>/rowan_results_per_edge_wide.csv \
  --experimental <benchmark>/pyrrolidine_32_subset.csv
```

`fep_benchmark.py` wraps stages 1–3 (and optionally the endpoint production) as a
single entry point — the intended console-script for a future Mamba package:

```bash
python scripts/fep_benchmark.py --series-root /path/to/ev71-density-series \
  --edges fep_edges/rowan_xtal_edges_full.tsv --run-root fep-runs --batch 24 \
  --rowan-edges <benchmark>/...per_edge_wide.csv --experimental <benchmark>/...subset.csv
```

## Edge network

`fep_edges/` holds ready-to-use edge TSVs derived from the Rowan OpenBind EV-A71
crystal (XTAL) network:

- `rowan_xtal_edges_full.tsv` — all 52 edges (32 ligands, connected, 21 cycles).
- `rowan_xtal_edges_starter_spanning.tsv` — 32-edge spanning tree + 1 cycle.
- `rowan_xtal_edges_demo8.tsv` — 7-edge strong→weak validation subset.

## Script map

**End-to-end**
- `fep_benchmark.py` — single driver: production → averaging → manifest → FEP.

**Endpoint MD/GCMC (branch 1)**
- `run_ev71_pipeline.py`, `ev71_full_pipeline.slurm` — per-ligand endpoint driver.
- `prepare_ev71_system.py`, `receptor_io.py`, `extract_ligands.py` — preparation.
- `ev71_equilibrate.py`, `ev71_production.py`, `ev71_loch_common.py` — Loch stages.
- `ev71_density_sites.py`, `ev71_postprocess.py`, `ev71_merge_site_catalogs.py` —
  hydration-density/site analysis.
- `ev71_make_series_manifest.py`, `ev71_density_series_task.slurm`,
  `submit_ev71_density_series.sh`, `ev71_finalize_density_series.*` — the
  32-ligand × 6-replica series.
- `audit_ev71_pipeline.py`, `pipeline_utils.py` — invariants and checkpoints.

**Averaging (the bridge)**
- `select_bound_frames.py` — pick one medoid equilibrated bound frame per ligand
  (replicate closest to the mean hydration-site occupancy; validates ghost-free).

**Relative FEP (branch 2)**
- `make_fep_manifest.py` — reviewed edges + bound frames → FEP manifest.
- `prepare_fep.py` — BioSimSpace atom mapping/merge; bound/free perturbable streams;
  `--align-to-bound-pose` places the merged ligand in the equilibrated pocket.
- `run_fep_leg.py`, `somd2_config.yaml` — resumable SOMD2 leg execution.
- `analyse_fep.py` — bound/free PMFs, overlap matrices, `DDG_bind`.
- `aggregate_fep_network.py` — weighted network fit; per-ligand values + residuals.
- `compare_to_rowan.py` — join the network against a reference benchmark
  (edge- and ligand-level Pearson/Spearman/MUE/RMSE).

**FEP submission**
- `fep_edge.slurm` + `submit_fep_edges.sh` — one GPU per edge, `--batch N`
  concurrency; each task runs prepare→bound→free→analyse, then a dependent
  network-fit and (if `--rowan-edges`) benchmark-compare job.
- `fep_prepare/leg/analyse/aggregate/compare.slurm` + `submit_fep_series.sh` — the
  older per-stage dependency-graph submitter (equivalent, finer-grained).

## Notes

- Bound-leg GCMC is **off by default** in FEP (the waters are already placed); pass
  `--with-gcmc` to run SOMD2 Loch GCMC during the transformation.
- `somd2_config.yaml` (11 windows, 5 ns/leg) is a conservative starting schedule,
  not a convergence guarantee. Check per-window `ns day⁻¹` and adjacent-window
  overlap on one edge before committing a full network.
- Every stage is checkpointed; reruns resume rather than restart.
