#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -eo pipefail
: "${PIPELINE_DIR:?}" "${FEP_ROOT:?}" "${FEP_MANIFEST:?}"
FEP_ENV="${FEP_ENV:-automated-fep}"
set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u
python -u "$PIPELINE_DIR/aggregate_fep_network.py" --manifest "$FEP_MANIFEST" \
  --fep-root "$FEP_ROOT" --output-dir "$FEP_ROOT/network_analysis"
