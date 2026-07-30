#!/bin/bash
#SBATCH --job-name=ev71-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -eo pipefail

: "${PROJECT_DIR:?PROJECT_DIR was not exported by the series submitter}"
: "${SERIES_MANIFEST:?SERIES_MANIFEST was not exported by the series submitter}"
: "${RECEPTOR:?RECEPTOR was not exported by the series submitter}"
: "${SERIES_OUTPUT_DIR:?SERIES_OUTPUT_DIR was not exported by the series submitter}"

CONDA_ENV="${CONDA_ENV:-cry-loch-babel}"
MINIMUM_CATALOG_SUPPORT="${MINIMUM_CATALOG_SUPPORT:-2}"

cd "$PROJECT_DIR"
set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -m py_compile \
    scripts/pipeline_utils.py \
    scripts/receptor_io.py \
    scripts/ev71_density_sites.py \
    scripts/ev71_merge_site_catalogs.py \
    scripts/ev71_finalize_density_series.py

python -u scripts/ev71_finalize_density_series.py \
    --manifest "$SERIES_MANIFEST" \
    --project-dir "$PROJECT_DIR" \
    --receptor "$RECEPTOR" \
    --output-dir "$SERIES_OUTPUT_DIR" \
    --minimum-catalog-support "$MINIMUM_CATALOG_SUPPORT" \
    --profile "${PROFILE:-full}"
