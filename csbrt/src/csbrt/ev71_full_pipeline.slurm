#!/bin/bash
#SBATCH --job-name=ev71-loch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=ev71-loch-%j.out
#SBATCH --error=ev71-loch-%j.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project_2}"
LIGAND_ID="${LIGAND_ID:-x7259a}"
REPLICA="${REPLICA:-1}"
if ! [[ "$REPLICA" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPLICA must be a positive integer: $REPLICA" >&2
    exit 2
fi
BASE_SEED="${BASE_SEED:-20260714}"
if ! [[ "$BASE_SEED" =~ ^[0-9]+$ ]]; then
    echo "BASE_SEED must be a nonnegative integer: $BASE_SEED" >&2
    exit 2
fi
SEED="${SEED:-$((BASE_SEED + (REPLICA - 1) * 1000))}"
PROFILE="${PROFILE:-full}"
THROUGH="${THROUGH:-postprocessing}"
CLUSTER_STRIDE="${CLUSTER_STRIDE:-1}"
MAX_DISTANCE_MEMORY_GB="${MAX_DISTANCE_MEMORY_GB:-32}"
CONDA_ENV="${CONDA_ENV:-cry-loch-babel}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/${LIGAND_ID}-rep${REPLICA}-${PROFILE}}"
RECEPTOR="${RECEPTOR:-$PROJECT_DIR/openbind_ev71_2a_pyrrolidine_benchmark_release/receptor/ev71_2a_x7339a_template_prepared.pdb}"
LIGAND_LIBRARY="${LIGAND_LIBRARY:-$PROJECT_DIR/openbind_ev71_2a_pyrrolidine_benchmark_release/ligands/pyrrolidine_32_rowan_docked_poses.sdf}"

cd "$PROJECT_DIR"
for input in "$RECEPTOR" "$LIGAND_LIBRARY"; do
    if [[ ! -s "$input" ]]; then
        echo "Missing required input: $input" >&2
        exit 2
    fi
done

set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENMM_CPU_THREADS="${SLURM_CPUS_PER_TASK:-16}"

command -v nvcc
python -m py_compile \
    scripts/pipeline_utils.py \
    scripts/receptor_io.py \
    scripts/extract_ligands.py \
    scripts/prepare_ev71_system.py \
    scripts/ev71_loch_common.py \
    scripts/ev71_equilibrate.py \
    scripts/ev71_production.py \
    scripts/ev71_postprocess.py \
    scripts/audit_ev71_pipeline.py \
    scripts/run_ev71_pipeline.py

python -u scripts/run_ev71_pipeline.py \
    --project-dir "$PROJECT_DIR" \
    --receptor "$RECEPTOR" \
    --ligand-library "$LIGAND_LIBRARY" \
    --ligand-id "$LIGAND_ID" \
    --run-dir "$RUN_DIR" \
    --seed "$SEED" \
    --profile "$PROFILE" \
    --through "$THROUGH" \
    --cluster-stride "$CLUSTER_STRIDE" \
    --max-distance-memory-gb "$MAX_DISTANCE_MEMORY_GB" \
    --gcmc-platform cuda \
    --md-platform cuda

echo "Complete EV71 Loch pipeline written to $RUN_DIR"
