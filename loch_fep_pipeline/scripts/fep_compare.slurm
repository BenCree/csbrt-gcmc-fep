#!/bin/bash
# Compare the fitted FEP network against the Rowan OpenBind benchmark. Runs last,
# after the network fit, so the whole edge->aggregate->compare chain completes
# unattended.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -eo pipefail
: "${PIPELINE_DIR:?}" "${FEP_ROOT:?}" "${ROWAN_EDGES:?}"
FEP_ENV="${FEP_ENV:-automated-fep}"
set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u
args=(--network "$FEP_ROOT/network_analysis/fep_network_analysis.json"
  --rowan-edges "$ROWAN_EDGES" --output-dir "$FEP_ROOT/rowan_comparison")
[[ -n "${ROWAN_EXPERIMENTAL:-}" ]] && args+=(--experimental "$ROWAN_EXPERIMENTAL")
[[ -n "${ROWAN_EDGE_COLUMN:-}" ]] && args+=(--rowan-edge-column "$ROWAN_EDGE_COLUMN")
python -u "$PIPELINE_DIR/compare_to_rowan.py" "${args[@]}"
