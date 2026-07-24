#!/usr/bin/env bash
: "${BASH_VERSION:?Run this script with Bash}"
set -eo pipefail
pipeline_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEP_MANIFEST="${FEP_MANIFEST:-}"
FEP_ROOT="${FEP_ROOT:-$PWD/fep-runs}"
FEP_CONFIG="${FEP_CONFIG:-$pipeline_dir/somd2_config.yaml}"
FEP_ENV="${FEP_ENV:-automated-fep}"
# The Loch endpoint pipeline exists to place the waters; the default FEP path now
# consumes that equilibrated frame with GCMC OFF. Pass --with-gcmc to also run
# grand-canonical water sampling during the alchemical transformation.
FEP_GCMC="${FEP_GCMC:-0}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) FEP_MANIFEST="$2"; shift 2 ;;
    --run-root) FEP_ROOT="$2"; shift 2 ;;
    --config) FEP_CONFIG="$2"; shift 2 ;;
    --fep-env) FEP_ENV="$2"; shift 2 ;;
    --without-gcmc) FEP_GCMC=0; shift ;;
    --with-gcmc) FEP_GCMC=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: $0 --manifest FEP.tsv [--run-root DIR] [--with-gcmc|--without-gcmc]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$FEP_MANIFEST" && -s "$FEP_MANIFEST" && -s "$FEP_CONFIG" ]] || {
  echo "A valid --manifest and --config are required" >&2; exit 2; }
mkdir -p "$FEP_ROOT/_logs"
export PIPELINE_DIR="$pipeline_dir" FEP_MANIFEST FEP_ROOT FEP_CONFIG FEP_ENV FEP_GCMC
edge_count="$(( $(wc -l < "$FEP_MANIFEST") - 1 ))"
analysis_jobs=()
for ((index=0; index<edge_count; index++)); do
  edge_id="$(awk -F '\t' -v line="$((index + 2))" 'NR == line {print $2; exit}' "$FEP_MANIFEST")"
  [[ -n "$edge_id" ]] || { echo "Missing edge ID at index $index" >&2; exit 2; }
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN edge=$edge_id: prepare -> bound+free -> analyse"
    continue
  fi
  prep="$(sbatch --parsable --export=ALL,FEP_EDGE_INDEX="$index" --job-name="fep-prep-$edge_id" \
    --output="$FEP_ROOT/_logs/$edge_id-prep-%j.out" --error="$FEP_ROOT/_logs/$edge_id-prep-%j.err" \
    "$pipeline_dir/fep_prepare.slurm")"; prep="${prep%%;*}"
  bound="$(sbatch --parsable --dependency="afterok:$prep" --export=ALL,EDGE_ID="$edge_id",LEG=bound \
    --job-name="fep-b-$edge_id" --output="$FEP_ROOT/_logs/$edge_id-bound-%j.out" \
    --error="$FEP_ROOT/_logs/$edge_id-bound-%j.err" "$pipeline_dir/fep_leg.slurm")"; bound="${bound%%;*}"
  free="$(sbatch --parsable --dependency="afterok:$prep" --export=ALL,EDGE_ID="$edge_id",LEG=free \
    --job-name="fep-f-$edge_id" --output="$FEP_ROOT/_logs/$edge_id-free-%j.out" \
    --error="$FEP_ROOT/_logs/$edge_id-free-%j.err" "$pipeline_dir/fep_leg.slurm")"; free="${free%%;*}"
  analysis="$(sbatch --parsable --dependency="afterok:$bound:$free" --export=ALL,EDGE_ID="$edge_id" \
    --job-name="fep-a-$edge_id" --output="$FEP_ROOT/_logs/$edge_id-analysis-%j.out" \
    --error="$FEP_ROOT/_logs/$edge_id-analysis-%j.err" "$pipeline_dir/fep_analyse.slurm")"; analysis="${analysis%%;*}"
  echo "$edge_id prepare=$prep bound=$bound free=$free analysis=$analysis"
  analysis_jobs+=("$analysis")
done
if [[ "$DRY_RUN" -eq 0 && ${#analysis_jobs[@]} -gt 0 ]]; then
  dependency="$(IFS=:; echo "${analysis_jobs[*]}")"
  aggregate="$(sbatch --parsable --dependency="afterok:$dependency" --export=ALL \
    --job-name=fep-network --output="$FEP_ROOT/_logs/network-%j.out" \
    --error="$FEP_ROOT/_logs/network-%j.err" "$pipeline_dir/fep_aggregate.slurm")"
  echo "network analysis=${aggregate%%;*}"
fi
