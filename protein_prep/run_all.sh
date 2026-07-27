#!/usr/bin/env bash
# Reproducible loop-modelling + MD workflow for the CRY1 (7DLI) monomer.
# Runs from this folder; all inputs/outputs are local.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Point these at the environments created from the environment-*.yml files.
MD_PY="${MD_PY:-$HOME/miniforge3/envs/openfe/bin/python}"
BOLTZ_PY="${BOLTZ_PY:-$HOME/miniforge3/envs/boltz/bin/python}"
BOLTZ="${BOLTZ:-$HOME/miniforge3/envs/boltz/bin/boltz}"

echo "== Step 1: identify the missing loop (sequence alignment) =="
"$MD_PY" scripts/1_analyze_gap.py

echo "== Step 2: write the Boltz-2 input YAML (monomer) =="
"$BOLTZ_PY" scripts/2_boltz_prep.py

echo "== Step 2b: Boltz-2 loop modelling (template = 7dli.cif, single-sequence) =="
# First run downloads the Boltz-2 weights (~6 GB) to \$BOLTZ_CACHE or ~/.boltz.
# Drop --accelerator to auto-select the GPU; use cpu only if the GPU is too small.
"$BOLTZ" predict outputs/boltz_7dli.yaml --out_dir outputs/boltz_out \
    --output_format pdb --override --no_kernels

echo "== Step 3: graft the modelled loop into the crystal receptor =="
"$BOLTZ_PY" scripts/3_graft_loop.py

echo "== Step 4: protein preparation + MD (OpenFE PlainMDProtocol) =="
"$MD_PY" scripts/4_run_md.py

echo
echo "Done. Key outputs:"
echo "  outputs/recfinal_7dli_water_loopmodelled.pdb        loop-completed receptor"
echo "  outputs/md/shared_*/  (system.pdb, equil_npt.pdb, simulation.xtc)"
