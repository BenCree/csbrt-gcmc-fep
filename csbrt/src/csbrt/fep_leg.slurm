#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
set -eo pipefail
: "${PIPELINE_DIR:?}" "${FEP_ROOT:?}" "${EDGE_ID:?}" "${LEG:?}" "${FEP_CONFIG:?}"
FEP_ENV="${FEP_ENV:-automated-fep}"
set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u
args=(--stream "$FEP_ROOT/$EDGE_ID/setup/${EDGE_ID}_${LEG}.bss" --config "$FEP_CONFIG"
  --output-dir "$FEP_ROOT/$EDGE_ID/$LEG" --leg "$LEG" --platform cuda --max-gpus 1)
if [[ "$LEG" == "bound" && "${FEP_GCMC:-1}" == "1" ]]; then
  args+=(--gcmc-bound --gcmc-selection "${GCMC_SELECTION:-resname LIG}" --gcmc-radius "${GCMC_RADIUS:-10 A}")
  args+=(--gcmc-num-waters "${GCMC_NUM_WATERS:-45}" \
    --gcmc-standard-volume "${GCMC_STANDARD_VOLUME:-30.345 A^3}" \
    --gcmc-excess-chemical-potential "${GCMC_EXCESS_CHEMICAL_POTENTIAL:--6.09 kcal/mol}" \
    --gcmc-bulk-sampling-probability "${GCMC_BULK_SAMPLING_PROBABILITY:-0.0}")
fi
python -u "$PIPELINE_DIR/run_fep_leg.py" "${args[@]}"
