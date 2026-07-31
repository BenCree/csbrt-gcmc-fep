#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -eo pipefail
: "${PIPELINE_DIR:?}" "${FEP_ROOT:?}" "${EDGE_ID:?}"
FEP_ENV="${FEP_ENV:-automated-fep}"
set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u
export PYMBAR_DISABLE_JAX="${PYMBAR_DISABLE_JAX:-1}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
python -u "$PIPELINE_DIR/analyse_fep.py" --bound-output "$FEP_ROOT/$EDGE_ID/bound" \
  --free-output "$FEP_ROOT/$EDGE_ID/free" --edge-id "$EDGE_ID" \
  --output "$FEP_ROOT/$EDGE_ID/analysis.json"
