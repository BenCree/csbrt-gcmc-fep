#!/bin/bash
# One complete FEP edge on one GPU: prepare -> bound leg -> free leg -> analyse.
# Submitted as a job array (one task per manifest row) by submit_fep_edges.sh.
# Each step checkpoints, so a requeued/rerun task resumes rather than restarts.
#SBATCH --job-name=fep-edge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
set -eo pipefail
: "${PIPELINE_DIR:?}" "${FEP_MANIFEST:?}" "${FEP_ROOT:?}" "${FEP_CONFIG:?}" "${SLURM_ARRAY_TASK_ID:?}"
FEP_ENV="${FEP_ENV:-automated-fep}"
FEP_GCMC="${FEP_GCMC:-0}"

line_number="$((SLURM_ARRAY_TASK_ID + 2))"
row="$(awk -F '\t' -v line="$line_number" 'NR == line {print; exit}' "$FEP_MANIFEST")"
[[ -n "$row" ]] || { echo "No manifest row for array task $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
IFS=$'\t' read -r index edge_id state_a state_b prep_a prep_b bound_prmtop bound_rst7 <<< "$row"
[[ "$index" == "$SLURM_ARRAY_TASK_ID" ]] || { echo "FEP manifest index mismatch" >&2; exit 2; }

set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u

edge_root="$FEP_ROOT/$edge_id"
echo "[fep-edge] $edge_id : $state_a -> $state_b (gcmc=$FEP_GCMC)"

# 1. Prepare the perturbable bound/free streams.
prepare_args=(--state-a-preparation "$prep_a" --state-b-preparation "$prep_b"
  --output-dir "$edge_root/setup" --edge-id "$edge_id")
if [[ -n "${bound_prmtop:-}" && -n "${bound_rst7:-}" ]]; then
  prepare_args+=(--bound-prmtop "$bound_prmtop" --bound-rst7 "$bound_rst7" --align-to-bound-pose)
fi
python -u "$PIPELINE_DIR/prepare_fep.py" "${prepare_args[@]}"

# 2. Bound leg. GCMC is off by default (waters already placed); --with-gcmc re-enables.
bound_args=(--stream "$edge_root/setup/${edge_id}_bound.bss" --config "$FEP_CONFIG"
  --output-dir "$edge_root/bound" --leg bound --platform cuda --max-gpus 1)
if [[ "$FEP_GCMC" == "1" ]]; then
  bound_args+=(--gcmc-bound --gcmc-selection "${GCMC_SELECTION:-resname LIG}"
    --gcmc-radius "${GCMC_RADIUS:-10 A}" --gcmc-num-waters "${GCMC_NUM_WATERS:-45}"
    --gcmc-standard-volume "${GCMC_STANDARD_VOLUME:-30.345 A^3}"
    --gcmc-excess-chemical-potential "${GCMC_EXCESS_CHEMICAL_POTENTIAL:--6.09 kcal/mol}"
    --gcmc-bulk-sampling-probability "${GCMC_BULK_SAMPLING_PROBABILITY:-0.0}")
fi
python -u "$PIPELINE_DIR/run_fep_leg.py" "${bound_args[@]}"

# 3. Free leg (never uses GCMC).
python -u "$PIPELINE_DIR/run_fep_leg.py" --stream "$edge_root/setup/${edge_id}_free.bss" \
  --config "$FEP_CONFIG" --output-dir "$edge_root/free" --leg free --platform cuda --max-gpus 1

# 4. Analyse this edge (DDG_bind = DG_bound - DG_free).
export PYMBAR_DISABLE_JAX="${PYMBAR_DISABLE_JAX:-1}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
python -u "$PIPELINE_DIR/analyse_fep.py" --bound-output "$edge_root/bound" \
  --free-output "$edge_root/free" --edge-id "$edge_id" --output "$edge_root/analysis.json"
echo "[fep-edge] $edge_id COMPLETE"
