#!/usr/bin/env bash
# Submit the full FEP network three times under replica exchange, with velocities
# randomised independently in each replicate.
#
#   ./submit_rex_replicates.sh --dry-run          # print sbatch commands, submit nothing
#   ./submit_rex_replicates.sh                    # submit
#
# Overridable by environment or flag:
#   MANIFEST  RUNS  PIPE  CONFIG  BATCH  REPLICATES  PARTITION  ACCOUNT  QOS
#
# Expected shape: 52 edges x 3 replicates = 156 array tasks, each doing
# prepare -> bound leg -> free leg -> analyse.
set -euo pipefail

RUNS="${RUNS:-$HOME/cry/project_2/runs}"
PIPE="${PIPE:-$HOME/cry/project_2/scripts}"
MANIFEST="${MANIFEST:-$RUNS/fep_manifest.tsv}"
CONFIG="${CONFIG:-$HOME/somd2_rex.yaml}"
BATCH="${BATCH:-16}"
REPLICATES="${REPLICATES:-3}"
RUN_PREFIX="${RUN_PREFIX:-rex-rep}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)    DRY=1; shift ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --replicates) REPLICATES="$2"; shift 2 ;;
        --manifest)   MANIFEST="$2"; shift 2 ;;
        --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --qos)        QOS="$2"; shift 2 ;;
        -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- preflight
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
[ -x "$PIPE/submit_fep_edges.sh" ] || [ -f "$PIPE/submit_fep_edges.sh" ] \
    || { echo "submitter not found: $PIPE/submit_fep_edges.sh" >&2; exit 2; }

EDGES=$(( $(wc -l < "$MANIFEST") - 1 ))
TOTAL=$(( EDGES * REPLICATES ))
echo "manifest    : $MANIFEST"
echo "edges       : $EDGES"
echo "replicates  : $REPLICATES"
echo "batch       : $BATCH  (concurrent array tasks per replicate)"
echo "TOTAL TASKS : $TOTAL   (each = prepare + bound leg + free leg + analyse)"
echo

# ---------------------------------------------------------------- config
# Written fresh every time so what runs is never a stale file.
#
# replica_exchange       swaps configurations between neighbouring lambda windows.
#                        Raises adjacent-window overlap directly; measured cost on
#                        an RTX 2080 Ti was 64 s vs 66 s without, i.e. free, because
#                        SOMD2 keeps all replicas resident on the GPU.
# randomise_velocities   fires at the START OF EACH REX CYCLE. This is the only
#                        mechanism by which velocities differ between replicates --
#                        without replica_exchange (and with no terminal-flip moves)
#                        it is a no-op, which is why the earlier rep1/rep2/rep3 all
#                        began from identical velocities.
# energy_frequency       cycles = runtime / energy_frequency, and mixing work scales
#                        as num_lambda^2 per cycle. 10 ps gives 500 cycles and 500
#                        energy samples per window; 1 ps would give 5000 cycles for
#                        no statistical gain worth the cost.
cat > "$CONFIG" <<'EOF'
runtime: 5 ns
timestep: 4 fs
temperature: 300 K
pressure: 1 atm
num_lambda: 11
lambda_schedule: standard_morph
equilibration_time: 100 ps
energy_frequency: 10 ps
frame_frequency: 100 ps
checkpoint_frequency: 100 ps
save_trajectories: true
save_energy_components: false
replica_exchange: true
randomise_velocities: true
perturbable_constraint: h_bonds_not_heavy_perturbed
constraint: h_bonds
save_crash_report: true
overwrite: false
EOF

echo "config written: $CONFIG"
grep -E "^(replica_exchange|randomise_velocities|energy_frequency|num_lambda|runtime|timestep):" \
     "$CONFIG" | sed 's/^/  /'
echo

# ---------------------------------------------------------------- submit
EXTRA=()
[ -n "$PARTITION" ] && EXTRA+=(--partition "$PARTITION")
[ -n "$ACCOUNT" ]   && EXTRA+=(--account "$ACCOUNT")
[ -n "$QOS" ]       && EXTRA+=(--qos "$QOS")

for REP in $(seq 1 "$REPLICATES"); do
    ROOT="$RUNS/${RUN_PREFIX}${REP}"
    CMD=(bash "$PIPE/submit_fep_edges.sh"
         --manifest "$MANIFEST"
         --run-root "$ROOT"
         --config   "$CONFIG"
         --batch    "$BATCH"
         --without-gcmc
         "${EXTRA[@]}")
    echo "=== replicate $REP -> $ROOT ==="
    if [ "$DRY" = 1 ]; then
        printf '  %q' "${CMD[@]}"; echo
    else
        "${CMD[@]}"
    fi
done

echo
echo "=== after the arrays finish, verify by ARTEFACT, not exit status ==="
echo "  python $PIPE/audit_fep_network.py $RUNS --glob '${RUN_PREFIX}*'"
echo
echo "  # every leg should have num_lambda energy parquets:"
echo "  find $RUNS/${RUN_PREFIX}* -name 'energy_traj_*.parquet' | wc -l"
echo "  # expected: $TOTAL edges x 2 legs x 11 windows = $(( TOTAL * 2 * 11 ))"
echo
echo "  # confirm REX genuinely engaged (identical timings can mean it was ignored):"
echo "  grep -c 'Mixing replicas' \$(ls -d $RUNS/${RUN_PREFIX}1/*/free | head -1)/runner.stdout.log"
echo "  # expected: ~500 (runtime / energy_frequency)"
echo
echo "  # confirm the replicates are genuinely independent:"
echo "  #   audit section 2 should report 'DDGs differ', NOT bit-identical"
