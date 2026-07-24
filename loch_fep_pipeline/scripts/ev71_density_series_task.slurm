#!/bin/bash
#SBATCH --job-name=ev71-series
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -eo pipefail

: "${PROJECT_DIR:?PROJECT_DIR was not exported by the series submitter}"
: "${SERIES_MANIFEST:?SERIES_MANIFEST was not exported by the series submitter}"
: "${RECEPTOR:?RECEPTOR was not exported by the series submitter}"
: "${LIGAND_LIBRARY:?LIGAND_LIBRARY was not exported by the series submitter}"
: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task}"

CONDA_ENV="${CONDA_ENV:-cry-loch-babel}"
PROFILE="${PROFILE:-full}"
if [[ "$PROFILE" != "full" && "$PROFILE" != "smoke" ]]; then
    echo "PROFILE must be full or smoke: $PROFILE" >&2
    exit 2
fi

line_number="$((SLURM_ARRAY_TASK_ID + 2))"
task_line="$(awk -F '\t' -v line="$line_number" 'NR == line {print; exit}' "$SERIES_MANIFEST")"
if [[ -z "$task_line" ]]; then
    echo "No manifest row for array task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
IFS=$'\t' read -r task_index ligand_id replica seed prefix run_dir <<< "$task_line"
# Be defensive toward manifests created by older csv.DictWriter defaults.
run_dir="${run_dir%$'\r'}"
if [[ "$task_index" != "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Manifest task $task_index does not match array task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
if ! [[ "$replica" =~ ^[1-9][0-9]*$ && "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid replica or seed in manifest row: $task_line" >&2
    exit 2
fi

cd "$PROJECT_DIR"
for input in "$RECEPTOR" "$LIGAND_LIBRARY" "$SERIES_MANIFEST"; do
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

python -m py_compile \
    scripts/pipeline_utils.py \
    scripts/receptor_io.py \
    scripts/extract_ligands.py \
    scripts/prepare_ev71_system.py \
    scripts/ev71_loch_common.py \
    scripts/ev71_equilibrate.py \
    scripts/ev71_production.py \
    scripts/audit_ev71_pipeline.py \
    scripts/run_ev71_pipeline.py \
    scripts/ev71_density_sites.py

mkdir -p "$run_dir"
rm -f "$run_dir/series_task.complete.json"

echo "EV71 series task=$task_index ligand=$ligand_id replica=$replica seed=$seed"
echo "Run directory: $run_dir"

python -u scripts/run_ev71_pipeline.py \
    --project-dir "$PROJECT_DIR" \
    --receptor "$RECEPTOR" \
    --ligand-library "$LIGAND_LIBRARY" \
    --ligand-id "$ligand_id" \
    --run-dir "$run_dir" \
    --prefix "$prefix" \
    --seed "$seed" \
    --profile "$PROFILE" \
    --through production \
    --gcmc-platform cuda \
    --md-platform cuda

python -u scripts/ev71_density_sites.py \
    --topology "$run_dir/production/${prefix}-loch-ghosts.pdb" \
    --trajectory "$run_dir/production/${prefix}-raw.dcd" \
    --ghost-file "$run_dir/production/${prefix}-gcmc-ghosts.txt" \
    --alignment-reference "$run_dir/preparation/receptor_input.pdb" \
    --output-dir "$run_dir/density_analysis" \
    --prefix "$prefix"

RUN_DIR="$run_dir" TASK_INDEX="$task_index" LIGAND_ID="$ligand_id" \
REPLICA="$replica" SEED="$seed" PREFIX="$prefix" PROFILE="$PROFILE" \
PYTHONPATH="$PROJECT_DIR/scripts" python - <<'PY'
import os
from pathlib import Path

from pipeline_utils import sha256, validate_recorded_outputs, write_json_atomic

run = Path(os.environ["RUN_DIR"]).resolve()
production_marker = run / "production" / "production.complete.json"
density_marker = run / "density_analysis" / "density_analysis.complete.json"
validate_recorded_outputs(production_marker)
validate_recorded_outputs(density_marker)
write_json_atomic(
    run / "series_task.complete.json",
    {
        "status": "completed",
        "validation_scope": (
            "full_simulation_density_analysis"
            if os.environ["PROFILE"] == "full"
            else "smoke_plumbing_density_analysis"
        ),
        "profile": os.environ["PROFILE"],
        "task_index": int(os.environ["TASK_INDEX"]),
        "ligand_id": os.environ["LIGAND_ID"],
        "replica": int(os.environ["REPLICA"]),
        "seed": int(os.environ["SEED"]),
        "prefix": os.environ["PREFIX"],
        "production_marker_sha256": sha256(production_marker),
        "density_marker_sha256": sha256(density_marker),
    },
)
PY

echo "SERIES_TASK_RESULT=$run_dir/series_task.complete.json"
