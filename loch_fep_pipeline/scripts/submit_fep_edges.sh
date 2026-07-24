#!/usr/bin/env bash
# Submit a whole FEP edge network as ONE throttled job array: one GPU per edge,
# at most --batch edges running at once. Each array task runs
# prepare -> bound -> free -> analyse for its edge (fep_edge.slurm). A dependent
# CPU job then fits the network (aggregate_fep_network.py).
#
#   ./submit_fep_edges.sh --manifest fep_manifest.tsv --batch 24 [--run-root DIR]
#
# --batch N is the Slurm array concurrency (the "%N" in 0-(E-1)%N): with 24 GPUs
# use --batch 24 to keep them all busy.
: "${BASH_VERSION:?Run this script with Bash}"
set -eo pipefail
pipeline_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FEP_MANIFEST="${FEP_MANIFEST:-}"
FEP_ROOT="${FEP_ROOT:-$PWD/fep-runs}"
FEP_CONFIG="${FEP_CONFIG:-$pipeline_dir/somd2_config.yaml}"
FEP_ENV="${FEP_ENV:-automated-fep}"
FEP_GCMC="${FEP_GCMC:-0}"
BATCH="${BATCH:-8}"
PARTITION=""; ACCOUNT=""; QOS=""
AGGREGATE=1
ROWAN_EDGES="${ROWAN_EDGES:-}"
ROWAN_EXPERIMENTAL="${ROWAN_EXPERIMENTAL:-}"
ROWAN_EDGE_COLUMN="${ROWAN_EDGE_COLUMN:-}"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 --manifest FEP.tsv [options]
  --manifest FILE     FEP manifest from make_fep_manifest.py (required)
  --run-root DIR      Output root (default: \$PWD/fep-runs)
  --config FILE       SOMD2 config yaml (default: somd2_config.yaml)
  --batch N           Max concurrent edges = array %N (default: 8; use 24 for 24 GPUs)
  --with-gcmc         Run bound-leg GCMC during FEP (default: off)
  --without-gcmc      Force bound-leg GCMC off (default)
  --fep-env NAME      Mamba env (default: automated-fep)
  --partition NAME    Slurm partition
  --account NAME      Slurm account
  --qos NAME          Slurm QOS
  --no-aggregate      Skip the dependent network-fit job
  --rowan-edges FILE      rowan_results_per_edge_wide.csv; adds a final compare job
  --experimental FILE     per-compound CSV with experimental dG (optional)
  --rowan-edge-column COL Rowan DDG column (default: xtal_nagl_ddg_kcal_mol)
  --dry-run           Print the plan without submitting
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) FEP_MANIFEST="$2"; shift 2 ;;
    --run-root) FEP_ROOT="$2"; shift 2 ;;
    --config) FEP_CONFIG="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --with-gcmc) FEP_GCMC=1; shift ;;
    --without-gcmc) FEP_GCMC=0; shift ;;
    --fep-env) FEP_ENV="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --no-aggregate) AGGREGATE=0; shift ;;
    --rowan-edges) ROWAN_EDGES="$2"; shift 2 ;;
    --experimental) ROWAN_EXPERIMENTAL="$2"; shift 2 ;;
    --rowan-edge-column) ROWAN_EDGE_COLUMN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$FEP_MANIFEST" && -s "$FEP_MANIFEST" && -s "$FEP_CONFIG" ]] || {
  echo "A valid --manifest and --config are required" >&2; exit 2; }
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "--batch must be a positive integer" >&2; exit 2; }
[[ -z "$ROWAN_EDGES" || -s "$ROWAN_EDGES" ]] || { echo "--rowan-edges file not found: $ROWAN_EDGES" >&2; exit 2; }
[[ -z "$ROWAN_EXPERIMENTAL" || -s "$ROWAN_EXPERIMENTAL" ]] || { echo "--experimental file not found: $ROWAN_EXPERIMENTAL" >&2; exit 2; }
compare=0; [[ -n "$ROWAN_EDGES" && "$AGGREGATE" -eq 1 ]] && compare=1

edge_count="$(( $(wc -l < "$FEP_MANIFEST") - 1 ))"
[[ "$edge_count" -ge 1 ]] || { echo "Manifest has no edges" >&2; exit 2; }
array_spec="0-$((edge_count - 1))%$BATCH"
log_dir="$FEP_ROOT/_logs"

echo "Manifest : $FEP_MANIFEST ($edge_count edges)"
echo "Run root : $FEP_ROOT"
echo "Array    : $array_spec  (<= $BATCH concurrent GPUs)"
echo "GCMC     : $([[ "$FEP_GCMC" == "1" ]] && echo on || echo off)"
echo "Aggregate: $([[ "$AGGREGATE" == "1" ]] && echo yes || echo no)"
echo "Compare  : $([[ "$compare" == "1" ]] && echo "yes ($ROWAN_EDGES)" || echo no)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN: sbatch --array $array_spec fep_edge.slurm" \
       "$([[ "$AGGREGATE" == "1" ]] && echo '-> aggregate')" \
       "$([[ "$compare" == "1" ]] && echo '-> compare_to_rowan')"
  exit 0
fi
command -v sbatch >/dev/null || { echo "sbatch unavailable; use --dry-run" >&2; exit 2; }
mkdir -p "$log_dir"

export PIPELINE_DIR="$pipeline_dir" FEP_MANIFEST FEP_ROOT FEP_CONFIG FEP_ENV FEP_GCMC
export ROWAN_EDGES ROWAN_EXPERIMENTAL ROWAN_EDGE_COLUMN

common_sbatch=(--parsable --export=ALL)
[[ -n "$PARTITION" ]] && common_sbatch+=(--partition "$PARTITION")
[[ -n "$ACCOUNT" ]] && common_sbatch+=(--account "$ACCOUNT")
[[ -n "$QOS" ]] && common_sbatch+=(--qos "$QOS")

array_job="$(sbatch "${common_sbatch[@]}" --array "$array_spec" --job-name fep-edges \
  --output "$log_dir/edge-%A_%a.out" --error "$log_dir/edge-%A_%a.err" \
  "$pipeline_dir/fep_edge.slurm")"
array_job="${array_job%%;*}"
echo "edge array job: $array_job"

if [[ "$AGGREGATE" -eq 1 ]]; then
  aggregate_job="$(sbatch "${common_sbatch[@]}" --dependency="afterok:$array_job" \
    --job-name fep-network --output "$log_dir/network-%j.out" \
    --error "$log_dir/network-%j.err" "$pipeline_dir/fep_aggregate.slurm")"
  aggregate_job="${aggregate_job%%;*}"
  echo "network fit job: $aggregate_job (after array $array_job)"
  if [[ "$compare" -eq 1 ]]; then
    compare_job="$(sbatch "${common_sbatch[@]}" --dependency="afterok:$aggregate_job" \
      --job-name fep-compare --output "$log_dir/compare-%j.out" \
      --error "$log_dir/compare-%j.err" "$pipeline_dir/fep_compare.slurm")"
    echo "rowan compare job: ${compare_job%%;*} (after network fit $aggregate_job)"
  fi
fi
