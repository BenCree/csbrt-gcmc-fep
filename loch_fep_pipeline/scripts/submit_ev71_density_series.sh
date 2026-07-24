#!/usr/bin/env bash
# Submit a reusable EV71 ligand-by-replica Loch/density Slurm series.

set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$script_dir/.." && pwd)}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/openbind_ev71_2a_pyrrolidine_benchmark_release}"
RECEPTOR="${RECEPTOR:-}"
LIGAND_LIBRARY="${LIGAND_LIBRARY:-}"
REPLICATES="${REPLICATES:-6}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/ev71-density-series}"
BASE_SEED="${BASE_SEED:-20260714}"
PREFIX_TEMPLATE="${PREFIX_TEMPLATE:-ev71_2a_{ligand_id}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
CONDA_ENV="${CONDA_ENV:-cry-loch-babel}"
PROFILE="${PROFILE:-full}"
MINIMUM_CATALOG_SUPPORT="${MINIMUM_CATALOG_SUPPORT:-2}"
GPU_MEMORY="${GPU_MEMORY:-32G}"
GPU_TIME="${GPU_TIME:-72:00:00}"
FINALIZER_MEMORY="${FINALIZER_MEMORY:-32G}"
FINALIZER_TIME="${FINALIZER_TIME:-04:00:00}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: scripts/submit_ev71_density_series.sh [options]

Core options:
  --input-folder PATH       Folder containing receptor/ and ligands/ (default: bundled release)
  --receptor PATH           Explicit receptor PDB/CIF/mmCIF (otherwise auto-detect under input/receptor)
  --ligand-library PATH     Explicit multi-record SDF (otherwise auto-detect one under input/ligands)
  --replicates N            Independent replicas per ligand (default: 6)
  --run-root PATH           Series output root
  --max-concurrent N        Maximum simultaneous GPU array tasks (default: 8)
  --base-seed N             First deterministic seed block
  --prefix-template TEXT    Filename prefix containing {ligand_id}
  --minimum-support N       Catalogs required to retain a common site (default: 2)
  --profile full|smoke      Full canonical schedule or plumbing test (default: full)

Scheduler options:
  --partition NAME          Slurm partition
  --account NAME            Slurm account
  --qos NAME                Slurm QoS
  --gpu-memory SIZE         Memory per GPU task (default: 32G)
  --gpu-time HH:MM:SS       Time per GPU task (default: 72:00:00)
  --finalizer-memory SIZE   Memory for common-catalog finalizer (default: 32G)
  --finalizer-time HH:MM:SS Time for finalizer (default: 04:00:00)
  --dry-run                 Build and inspect the manifest without submitting
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-folder) INPUT_DIR="$2"; shift 2 ;;
        --receptor) RECEPTOR="$2"; shift 2 ;;
        --ligand-library) LIGAND_LIBRARY="$2"; shift 2 ;;
        --replicates) REPLICATES="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --base-seed) BASE_SEED="$2"; shift 2 ;;
        --prefix-template) PREFIX_TEMPLATE="$2"; shift 2 ;;
        --minimum-support) MINIMUM_CATALOG_SUPPORT="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
        --gpu-time) GPU_TIME="$2"; shift 2 ;;
        --finalizer-memory) FINALIZER_MEMORY="$2"; shift 2 ;;
        --finalizer-time) FINALIZER_TIME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$REPLICATES" "$MAX_CONCURRENT" "$MINIMUM_CATALOG_SUPPORT"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Replica, concurrency, and support values must be positive integers" >&2
        exit 2
    fi
done
if ! [[ "$BASE_SEED" =~ ^[0-9]+$ ]]; then
    echo "--base-seed must be a nonnegative integer" >&2
    exit 2
fi
if [[ "$PROFILE" != "full" && "$PROFILE" != "smoke" ]]; then
    echo "--profile must be full or smoke" >&2
    exit 2
fi

discover_one() {
    local folder="$1"
    local pattern="$2"
    local label="$3"
    local matches=()
    mapfile -t matches < <(find "$folder" -maxdepth 1 -type f -name "$pattern" | sort)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "Expected exactly one $label under $folder; found ${#matches[@]}. Use the explicit option." >&2
        exit 2
    fi
    printf '%s\n' "${matches[0]}"
}

if [[ -z "$RECEPTOR" ]]; then
    mapfile -t receptor_candidates < <(
        find "$INPUT_DIR/receptor" -maxdepth 1 -type f \
            \( -iname '*.pdb' -o -iname '*.cif' -o -iname '*.mmcif' \) | sort
    )
    if [[ ${#receptor_candidates[@]} -ne 1 ]]; then
        echo "Expected exactly one receptor PDB/CIF/mmCIF under $INPUT_DIR/receptor; found ${#receptor_candidates[@]}. Use --receptor." >&2
        exit 2
    fi
    RECEPTOR="${receptor_candidates[0]}"
fi
if [[ -z "$LIGAND_LIBRARY" ]]; then
    LIGAND_LIBRARY="$(discover_one "$INPUT_DIR/ligands" '*.sdf' 'ligand SDF')"
fi
for input in "$RECEPTOR" "$LIGAND_LIBRARY"; do
    if [[ ! -s "$input" ]]; then
        echo "Missing required input: $input" >&2
        exit 2
    fi
done

python_bin="$HOME/miniforge3/envs/$CONDA_ENV/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing Mamba-environment Python: $python_bin" >&2
    exit 2
fi
if ! command -v sbatch >/dev/null && [[ "$DRY_RUN" -eq 0 ]]; then
    echo "sbatch is unavailable; use --dry-run to validate locally" >&2
    exit 2
fi

submission_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
submission_dir="$RUN_ROOT/_series_submissions/$submission_id"
log_dir="$submission_dir/logs"
mkdir -p "$log_dir"
SERIES_MANIFEST="$submission_dir/tasks.tsv"
SERIES_OUTPUT_DIR="$submission_dir/common_analysis"

"$python_bin" -u "$PROJECT_DIR/scripts/ev71_make_series_manifest.py" \
    --ligand-library "$LIGAND_LIBRARY" \
    --run-root "$RUN_ROOT" \
    --replicates "$REPLICATES" \
    --base-seed "$BASE_SEED" \
    --prefix-template "$PREFIX_TEMPLATE" \
    --output "$SERIES_MANIFEST"

task_count="$(( $(wc -l < "$SERIES_MANIFEST") - 1 ))"
if [[ "$task_count" -lt 1 ]]; then
    echo "Manifest contains no tasks" >&2
    exit 2
fi
array_spec="0-$((task_count - 1))%$MAX_CONCURRENT"

export PROJECT_DIR INPUT_DIR RECEPTOR LIGAND_LIBRARY RUN_ROOT CONDA_ENV PROFILE
export SERIES_MANIFEST SERIES_OUTPUT_DIR MINIMUM_CATALOG_SUPPORT

common_sbatch=(
    --export=ALL
)
if [[ -n "$PARTITION" ]]; then common_sbatch+=(--partition "$PARTITION"); fi
if [[ -n "$ACCOUNT" ]]; then common_sbatch+=(--account "$ACCOUNT"); fi
if [[ -n "$QOS" ]]; then common_sbatch+=(--qos "$QOS"); fi

echo "Input folder: $INPUT_DIR"
echo "Ligand library: $LIGAND_LIBRARY"
echo "Receptor: $RECEPTOR"
echo "Manifest: $SERIES_MANIFEST"
echo "Tasks: $task_count (${REPLICATES} replicas per ligand)"
echo "GPU array: $array_spec"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN: no jobs submitted"
    echo "Worker: $PROJECT_DIR/scripts/ev71_density_series_task.slurm"
    echo "Finalizer: $PROJECT_DIR/scripts/ev71_finalize_density_series.slurm"
    exit 0
fi

array_job="$(sbatch --parsable \
    "${common_sbatch[@]}" \
    --array "$array_spec" \
    --job-name "ev71-series" \
    --cpus-per-task 16 \
    --mem "$GPU_MEMORY" \
    --time "$GPU_TIME" \
    --output "$log_dir/%A_%a.out" \
    --error "$log_dir/%A_%a.err" \
    "$PROJECT_DIR/scripts/ev71_density_series_task.slurm")"
array_job="${array_job%%;*}"

finalizer_job="$(sbatch --parsable \
    "${common_sbatch[@]}" \
    --dependency "afterok:$array_job" \
    --job-name "ev71-finalize" \
    --cpus-per-task 8 \
    --mem "$FINALIZER_MEMORY" \
    --time "$FINALIZER_TIME" \
    --output "$log_dir/finalize-%j.out" \
    --error "$log_dir/finalize-%j.err" \
    "$PROJECT_DIR/scripts/ev71_finalize_density_series.slurm")"
finalizer_job="${finalizer_job%%;*}"

echo "Submitted GPU array job: $array_job"
echo "Submitted dependent finalizer job: $finalizer_job"
echo "Final series result: $SERIES_OUTPUT_DIR/series-density-analysis.json"
