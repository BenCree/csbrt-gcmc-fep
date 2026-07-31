#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -eo pipefail
: "${PIPELINE_DIR:?}" "${FEP_MANIFEST:?}" "${FEP_ROOT:?}" "${FEP_EDGE_INDEX:?}"
FEP_ENV="${FEP_ENV:-automated-fep}"
row="$(awk -F '\t' -v line="$((FEP_EDGE_INDEX + 2))" 'NR == line {print; exit}' "$FEP_MANIFEST")"
IFS=$'\t' read -r index edge_id state_a state_b prep_a prep_b bound_prmtop bound_rst7 <<< "$row"
[[ "$index" == "$FEP_EDGE_INDEX" ]] || { echo "FEP manifest index mismatch" >&2; exit 2; }
set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u
prepare_args=(--state-a-preparation "$prep_a" --state-b-preparation "$prep_b"
  --output-dir "$FEP_ROOT/$edge_id/setup" --edge-id "$edge_id")
# When the manifest carries a water-equilibrated production frame, seed the bound
# leg from it and reposition the merged ligand onto that equilibrated pose.
if [[ -n "${bound_prmtop:-}" && -n "${bound_rst7:-}" ]]; then
  prepare_args+=(--bound-prmtop "$bound_prmtop" --bound-rst7 "$bound_rst7" --align-to-bound-pose)
fi
python -u "$PIPELINE_DIR/prepare_fep.py" "${prepare_args[@]}"
