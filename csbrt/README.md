# csbrt — hydration-aware GCMC + FEP workflow, in one environment

End-to-end workflow from a crystal structure with a missing loop to relative
binding free energies, with grand-canonical water sampling in between:

```
preprocess  →  equilibrate  →  gcmc  →  fep  →  analysis
```

| stage | what it does | key tools |
|---|---|---|
| `preprocess` | model the missing loop (explicit mmCIF template, optional ColabFold MSAs), graft it into the crystal receptor, protonate at pH | OpenFold3, biopython, pdb2pqr/PROPKA |
| `equilibrate` | AM1-BCC/GAFF2 + ff14SB/TIP3P preparation, Loch UVT1 → NPT → UVT2 | AmberTools, Loch, Sire/OpenMM |
| `gcmc` | Loch MD/GCMC production + hydration-site density analysis | Loch, MDTraj |
| `fep` | pick equilibrated bound frames, resolve reviewed edges, run SOMD2 RBFE | BioSimSpace, SOMD2 |
| `analysis` | network fit, benchmark/experiment comparison, sampling diagnostics | numpy, slow-rotations |

The whole thing runs in **one conda environment**. There is no per-stage
interpreter indirection.

## Install

```bash
./install.sh                 # or: ./install.sh <env-name>
mamba activate csbrt
```

`install.sh` performs exactly the three steps below; run them by hand if you
prefer.

**Conda is mandatory.** OpenBioSim distributes `loch`, `somd2` and `BioSimSpace`
through their own conda channel only, and AmberTools is conda-only as well. That
rules out a conda-forge recipe (no cross-channel or pip dependencies allowed) and
equally rules out a pure-PyPI install: `loch`, `somd2`, `biosimspace` and
`ambertools` are simply not on PyPI. The `csbrt` package itself is pure Python and
installs with pip; the engines underneath it cannot.

```bash
mamba env create -f environment.yml
mamba activate csbrt

# CUDA torch for OpenFold3 (self-contained wheels; leaves the conda CUDA 12.8
# toolchain used by OpenMM/loch untouched)
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
pip install openfold3 pdb2pqr==3.7.1 propka==3.5.1 \
            git+https://github.com/MobleyLab/slow-rotations.git
pip install .                # the csbrt CLI
```

## Commands

Five per-stage commands plus one that runs any range:

```bash
csbrt-preprocess   --config run.yaml     # OpenFold3 -> graft -> protonate -> ligand prep
csbrt-equilibrate  --config run.yaml     # Loch UVT1/NPT/UVT2
csbrt-gcmc         --config run.yaml     # GCMC production + hydration-site density
csbrt-fep          --config run.yaml     # bound frames + edge manifest
csbrt-analysis     --config run.yaml     # network fit + benchmark comparison

csbrt --all                              --config run.yaml
csbrt --from equilibrate --through gcmc  --config run.yaml
csbrt gcmc                               --config run.yaml
csbrt --all --config run.yaml --dry-run  # print commands without running
```

Two constraints in `environment.yml` are load-bearing — read the comments there
before changing them:

1. **CUDA is pinned whole to 12.8**, not just `cuda-version`. An unpinned
   `cuda-nvcc` resolves to 12.9 and OpenMM then fails on 12.8-capped cluster
   drivers with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)`, producing *zero*
   output while appearing to run.
2. **The structure predictor is OpenFold3, not Boltz-2.** `boltz` requires
   `numpy<2.0` and Chai-1 requires `numpy~=1.21`, while sire/somd2/loch/mdtraj
   are built against numpy 2.x. Neither can share this environment; OpenFold3
   leaves numpy unpinned and still supports explicit mmCIF templates.

### GPU requirement (preprocess stage only)

OpenFold3's Triton triangle kernels need **Ampere or newer (SM ≥ 8.0)** — they
require hardware bf16. On Turing (SM 7.5, e.g. RTX 2080 Ti) kernel compilation
fails with:

```
LLVM ERROR: Unsupported rounding mode for conversion
  openfold3/core/kernels/triton/evoformer.py — ConvertTritonGPUToLLVM failed
```

Fixes, in order of preference:

1. Run `preprocess` on an Ampere/Ada GPU (RTX A4000, 3090, 4090, RTX 4000 Ada …).
2. On Turing, pass the bundled fallback runner config, which disables the Triton
   kernels:
   ```bash
   run_openfold predict ... --runner-yaml runner_turing.yml
   ```

The other stages (Loch GCMC, SOMD2 FEP) have no such constraint and run fine on
Turing — they are OpenMM/CUDA, not Triton.

Verify the install (GPU node, ~1 min):

```bash
python -c "
import numpy, torch, somd2, sire, loch, openfold3, openmm
print('numpy', numpy.__version__, '| torch cuda', torch.cuda.is_available())
print('sire', sire.__version__, 'loch', loch.__version__, 'somd2', somd2.__version__)"
src/csbrt/preflight_fep.sh          # OpenMM CUDA context + a real short SOMD2 leg
```

## Run

```bash
cp config.example.yaml run.yaml    # edit paths, ligand, loop, pH
csbrt --all --config run.yaml
```

Set `profile: smoke` in the config for a fast plumbing run (reduced counts —
**not** a scientific trajectory). See **Commands** above for per-stage invocation.

## Cluster / scale-out

The endpoint series and the FEP network are embarrassingly parallel and ship with
SLURM submitters:

```bash
./src/csbrt/submit_ev71_density_series.sh --dataset dataset.tsv --run-root runs/ --replicates 6
./src/csbrt/submit_fep_edges.sh --manifest fep_manifest.tsv --batch 20 --run-root runs/fep-runs \
    --rowan-edges <benchmark>.csv --experimental <affinities>.csv
```

`submit_fep_edges.sh` runs one edge per GPU (`--batch N` concurrent), then a
dependent network fit and benchmark comparison. Every stage is checkpointed, so
re-submitting completed work is a no-op — but note that editing a pipeline source
file changes its hash and invalidates downstream stage markers, so when retrying
one failed array task after a code change, submit only that array index.

## Preprocessing notes

`of3_prep.py` writes an OpenFold3 query that co-folds the receptor (with the
missing loop inserted) together with the ligand, using the deposited mmCIF as an
explicit template so the model reproduces the crystal conformation.

One difference from the earlier Boltz-2 version: Boltz accepted a `pocket`
constraint tying the ligand to named binding residues, and OpenFold3's query
schema has no equivalent. The pocket residues are still computed and reported for
the record, but only the template localises the pose. **If you are migrating from
the Boltz workflow, re-validate the grafted loop against the crystal** before
trusting downstream results.

## Diagnostics worth running

- **Window overlap** (in each edge's `analysis.json`): near-zero adjacent-window
  overlap means the ΔΔG diverged rather than converged — add λ windows.
- **Cycle closure** (`fep_network_analysis.json`): standardized residuals flag
  edges inconsistent with the rest of the network. This and overlap are *internal*
  criteria, so filtering on them is legitimate; filtering on disagreement with a
  reference benchmark is circular.
- **Torsional sampling** (`slow-rotations`): an edge can have healthy overlap and
  still be wrong if a ligand torsion never transitions. The tool counts dihedral
  state transitions and flags those with fewer than ~10.
