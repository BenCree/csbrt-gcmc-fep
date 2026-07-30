#!/usr/bin/env bash
# Reproduce the documented install exactly as the README describes it.
#   ./install.sh [env_name]
set -eo pipefail
ENV="${1:-csbrt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA="$HOME/miniforge3/bin/mamba"

echo "== 1/3 conda environment from environment.yml =="
"$MAMBA" env create -y -n "$ENV" -f "$HERE/environment.yml"

PY="$HOME/miniforge3/envs/$ENV/bin/python"

# sire hard-depends on conda `pytorch`/`libtorch` (>=2.10,<2.11), which resolves to
# a CPU-only build. Two torches therefore coexist here BY DESIGN and both are
# required:
#
#   $ENV/lib/libc10.so, libtorch*.so   conda 2.10, the C++ layer sire/loch link
#                                      against. loch dlopens libc10.so lazily,
#                                      inside the GCMC sampler.
#   site-packages/torch/               pip 2.7.1+cu126, what OpenFold3 imports.
#
# They never collide because OpenFold3 (preprocess) and loch GCMC run in separate
# processes and never co-load. Do NOT "clean up" the conda pytorch: removing it,
# or deleting $ENV/lib/libtorch*.so / libc10*.so, leaves every import passing and
# then fails mid-GCMC with `libc10.so: cannot open shared object file`.
echo "== 2/3 CUDA torch + OpenFold3 + protonation + diagnostics (pip) =="
"$PY" -m pip install --no-input --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1'
"$PY" -m pip install --no-input openfold3 pdb2pqr==3.7.1 propka==3.5.1

echo "== 3/3 the csbrt package itself =="
"$PY" -m pip install --no-input "$HERE"

echo "== verifying =="
# Imports alone are NOT sufficient: loch dlopens libc10.so lazily inside the GCMC
# sampler, so a broken torch layout imports cleanly and only fails mid-run.
# Check the conda C++ libraries on disk as well.
ENV_LIB="$HOME/miniforge3/envs/$ENV/lib"
for so in libc10.so libtorch.so; do
    [ -f "$ENV_LIB/$so" ] || {
        echo "FAIL: $ENV_LIB/$so is missing -- sire/loch will fail mid-GCMC." >&2
        echo "      Do not remove the conda pytorch/libtorch packages." >&2
        exit 1
    }
done
"$PY" - <<'CHECK'
import numpy, torch, sire, loch, somd2, BioSimSpace, openmm, openfold3
if not torch.cuda.is_available():
    raise SystemExit("FAIL: pip torch reports no CUDA device")
print(f"  numpy {numpy.__version__} | torch {torch.__version__} cuda={torch.cuda.is_available()}")
print(f"  sire {sire.__version__} loch {loch.__version__} somd2 {somd2.__version__}")
CHECK
echo "  conda C++ torch libs present in $ENV_LIB (required by loch GCMC)"

echo "== done: mamba activate $ENV =="
