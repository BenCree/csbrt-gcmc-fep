#!/bin/bash
# Rerun ONE free leg with perturbable-bond constraints removed, to test whether
# that clears the lambda windows that fail minimisation.
#
#   sbatch --partition=<P> --account=<A> \
#          --export=ALL,EDGE=x7317a_to_x7427a,SRC_ROOT=$HOME/cry/project_2/runs/fep-runs-rep2 \
#          constraint_test.slurm
#
# Writes to a NEW directory; the source run is read-only and untouched.
#SBATCH --job-name=constraint-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=constraint-test-%j.out
set -eo pipefail

EDGE="${EDGE:?set EDGE=<edge_id>}"
PIPELINE_DIR="${PIPELINE_DIR:-$HOME/cry/project_2/scripts}"
RUNS="${RUNS:-$HOME/cry/project_2/runs}"
SRC_ROOT="${SRC_ROOT:-$RUNS/fep-runs-rep2}"
OUT_ROOT="${OUT_ROOT:-$RUNS/constraint-test}"
FEP_ENV="${FEP_ENV:-automated-fep}"

SRC_CONFIG="$SRC_ROOT/$EDGE/free/config.yaml"
STREAM="$SRC_ROOT/$EDGE/setup/${EDGE}_free.bss"
CONFIG="$OUT_ROOT/$EDGE/somd2_no_perturbable_constraints.yaml"
OUTDIR="$OUT_ROOT/$EDGE/free"

for f in "$SRC_CONFIG" "$STREAM"; do
    [ -f "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done
mkdir -p "$(dirname "$CONFIG")"

# The failure being tested:
#   SireError::invalid_state: Despite repeated attempts, the minimiser could not
#   minimise the system while simultaneously satisfying the constraints.
# Removing constraints on perturbable bonds addresses it directly. The timestep
# must drop to 2 fs once X-H bonds on those atoms are unconstrained. A larger
# shift_delta softens the LJ singularity in the intermediate-lambda region where
# every failure sits. The last two make a failure report itself instead of
# leaving SOMD2 to exit 0 with dead windows.
sed -e 's/^perturbable_constraint:.*/perturbable_constraint: none/' \
    -e 's/^timestep:.*/timestep: 2 fs/' \
    -e 's/^shift_delta:.*/shift_delta: 2.0 A/' \
    -e 's/^minimisation_errors:.*/minimisation_errors: true/' \
    -e 's/^save_crash_report:.*/save_crash_report: true/' \
    -e 's|^output_directory:.*|output_directory: null|' \
    "$SRC_CONFIG" > "$CONFIG"

echo "[constraint-test] edge   : $EDGE"
echo "[constraint-test] stream : $STREAM"
echo "[constraint-test] config : $CONFIG"
grep -E "^(perturbable_constraint|timestep|shift_delta|num_lambda|runtime|minimisation_errors|save_crash_report):" \
     "$CONFIG" | sed 's/^/[constraint-test]   /'

set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u

command -v somd2 >/dev/null || { echo "somd2 not on PATH after activating $FEP_ENV" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/[constraint-test] gpu: /'

python -u "$PIPELINE_DIR/run_fep_leg.py" \
    --leg free --platform cuda --max-gpus 1 \
    --stream "$STREAM" --config "$CONFIG" --output-dir "$OUTDIR"
RC=$?

# run_fep_leg.py already refuses to publish its marker on a partial leg, but say
# it plainly here too: SOMD2 exits 0 with dead windows, so the artefact is the
# only trustworthy signal.
EXPECTED=$(grep -E "^num_lambda:" "$CONFIG" | awk '{print $2}')
GOT=$(ls -1 "$OUTDIR"/energy_traj_*.parquet 2>/dev/null | wc -l)
FAILS=$(grep -c "Minimisation failed" "$OUTDIR/log.txt" 2>/dev/null || echo 0)

echo
echo "[constraint-test] windows written    : $GOT / $EXPECTED"
echo "[constraint-test] minimisation fails : $FAILS  (was 6 with h_bonds_not_heavy_perturbed)"
if [ "$GOT" = "$EXPECTED" ] && [ "$FAILS" = "0" ]; then
    echo "[constraint-test] RESULT: clean -- the constraint scheme was the cause."
elif [ "$GOT" = "$EXPECTED" ]; then
    echo "[constraint-test] RESULT: all windows present, but $FAILS minimisation(s) still"
    echo "[constraint-test]         failed and were rescued by auto_fix_minimise."
    echo "[constraint-test]         Survivors, not successes -- still marginal."
else
    echo "[constraint-test] RESULT: still incomplete. Try a larger shift_delta, or"
    echo "[constraint-test]         suspect the mapping (12 ghosts on rotatable sp3"
    echo "[constraint-test]         anchors) rather than the protocol."
fi
exit $RC
