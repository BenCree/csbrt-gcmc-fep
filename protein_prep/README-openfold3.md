# OpenFold3 track — a fork of the loop-modelling steps

This is a **parallel implementation** of the receptor-prep steps described in
[`README.md`](README.md), ported from Boltz-2 to **OpenFold3**. It does not
replace anything: Ben's Boltz-2 scripts and their three-environment workflow are
unchanged and remain the reference implementation. Use whichever track matches
the environment you are running in.

| Boltz-2 (original) | OpenFold3 port | Step |
| --- | --- | --- |
| `scripts/1_analyze_gap.py` | `scripts/of3_analyze_gap.py` | identify the missing loop from sequence |
| `scripts/2_boltz_prep.py`  | `scripts/of3_prep.py`       | write the predictor query for the holo complex |
| `scripts/3_graft_loop.py`  | `scripts/of3_graft_loop.py` | superpose + graft the loop into the receptor |
| *(inside `4_run_md.py`)*   | `scripts/of3_protonate.py`  | PDB2PQR/PROPKA protonation as a standalone step |
| —                          | `scripts/prepare_ligands.py` | explicit-hydrogen ligand prep for the endpoint/FEP stages |

There is no OpenFold3 equivalent of `4_run_md.py`. The OpenFold3 track hands the
protonated receptor to the Loch GCMC + SOMD2 FEP pipeline in
[`../csbrt/`](../csbrt/) instead of running OpenFE MD.

## Why this fork exists

The two tracks cannot share one conda environment. `boltz` requires `numpy<2.0`
(and Chai-1 `numpy~=1.21`), while `sire`, `somd2`, `loch` and `mdtraj` are built
against numpy 2.x — installing Boltz-2 alongside them downgrades numpy and breaks
somd2 with `No module named 'numpy.core.multiarray'`.

OpenFold3 leaves numpy unpinned and still supports explicit mmCIF templates
(`template_cif_paths`, "CIF-direct" mode) plus ColabFold MSAs, so it does the same
job inside the single unified environment that the GCMC and FEP stages need. That
is the only reason for the port; the Boltz-2 track is not deprecated.

- Running the three-environment workflow in this directory → use the numbered
  Boltz-2 scripts.
- Running the unified `csbrt` package → use the `of3_*` scripts.

## Two behavioural differences before you switch

1. **No pocket constraint.** Boltz accepted an explicit `pocket` constraint tying
   the ligand to named binding residues. OpenFold3's query schema has no
   equivalent, so `of3_prep.py` still computes and reports the pocket residues for
   the record, but only the template localises the pose.
2. **mmCIF, not PDB.** OpenFold3 writes mmCIF, so `of3_graft_loop.py` locates the
   model by glob and parses with whichever biopython parser matches the extension.

Because of (1), **re-validate the grafted loop against the crystal** before
trusting downstream results from this track. Superpose on the *non-modelled*
scaffold, then inspect the modelled region together with the model's own
per-residue confidence. Reporting the superposition RMSD of the anchor atoms
proves nothing about the loop itself.

## GPU requirement

OpenFold3 needs Ampere or newer (SM >= 8.0). On Turing (e.g. RTX 2080 Ti) its
Triton triangle kernels fail with
`LLVM ERROR: Unsupported rounding mode for conversion`; pass a runner yaml setting
`settings.memory.eval.use_triton_triangle_kernels: false` — see
[`../csbrt/runner_turing.yml`](../csbrt/runner_turing.yml). Loch and SOMD2 have no
such constraint.
