#!/usr/bin/env bash
# Diagnose why specific lambda windows failed in a SOMD2 FEP leg.
#
#   ./diagnose_failed_windows.sh EDGE_ID [REPLICATE_ROOT] [RUNS_DIR]
#
#   ./diagnose_failed_windows.sh x7317a_to_x7427a
#   ./diagnose_failed_windows.sh x7317a_to_x7427a fep-runs-rep3
#
# Read-only. Reruns nothing. Everything below comes from files already on disk.
set -uo pipefail

EDGE="${1:?usage: $0 EDGE_ID [REPLICATE_ROOT] [RUNS_DIR]}"
REP="${2:-fep-runs-rep2}"
RUNS="${3:-$HOME/cry/project_2/runs}"

EDGE_DIR="$RUNS/$REP/$EDGE"
FREE="$EDGE_DIR/free"
BOUND="$EDGE_DIR/bound"
SETUP="$EDGE_DIR/setup"

[ -d "$EDGE_DIR" ] || { echo "no such edge dir: $EDGE_DIR" >&2; exit 1; }
echo "edge : $EDGE"
echo "root : $RUNS/$REP"
hr() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- 1
hr "1. WINDOWS PRESENT (energy parquets)"
for LEG in bound free; do
  D="$EDGE_DIR/$LEG"
  [ -d "$D" ] || { echo "  $LEG: missing"; continue; }
  N=$(ls -1 "$D"/energy_traj_*.parquet 2>/dev/null | wc -l)
  L=$(ls -1 "$D"/energy_traj_*.parquet 2>/dev/null \
        | sed 's|.*energy_traj_||;s|\.parquet||' | sort -n | tr '\n' ' ')
  echo "  $LEG: n=$N  [$L]"
  EXP=$(grep -E "^num_lambda:" "$D/config.yaml" 2>/dev/null | awk '{print $2}')
  [ -n "${EXP:-}" ] && echo "         expected num_lambda=$EXP"
done

# ---------------------------------------------------------------- 2
hr "2. HOW FAR EACH WINDOW GOT (construction vs minimisation)"
echo "  -- windows that reached minimisation --"
for LEG in bound free; do
  D="$EDGE_DIR/$LEG"
  [ -f "$D/log.txt" ] || continue
  M=$(grep -oE "Minimising at .*= *[0-9.]+" "$D/log.txt" 2>/dev/null \
        | grep -oE "[0-9]+\.[0-9]+" | sort -u | tr '\n' ' ')
  echo "     $LEG: ${M:-<none logged>}"
done
echo "  -- windows in BOUND but not FREE (bound is the control: same topology) --"
if [ -f "$BOUND/log.txt" ] && [ -f "$FREE/log.txt" ]; then
  comm -23 \
    <(grep -oE "Minimising at .*= *[0-9.]+" "$BOUND/log.txt" | grep -oE "[0-9]+\.[0-9]+" | sort -u) \
    <(grep -oE "Minimising at .*= *[0-9.]+" "$FREE/log.txt"  | grep -oE "[0-9]+\.[0-9]+" | sort -u) \
    | sed 's/^/     /'
fi

# ---------------------------------------------------------------- 3
hr "3. LOG LINES AT THE DEAD WINDOWS (free leg)"
MISSING=$(python3 - "$FREE" <<'PY' 2>/dev/null
import sys, glob, re, os
d = sys.argv[1]
have = set()
for f in glob.glob(os.path.join(d, "energy_traj_*.parquet")):
    m = re.search(r"energy_traj_([0-9.]+)\.parquet", f)
    if m: have.add(round(float(m.group(1)), 5))
n = 11
try:
    for line in open(os.path.join(d, "config.yaml")):
        if line.startswith("num_lambda:"):
            n = int(line.split(":")[1]); break
except OSError:
    pass
print(" ".join(f"{i/(n-1):.5f}" for i in range(n) if round(i/(n-1),5) not in have))
PY
)
echo "  missing windows: ${MISSING:-<none>}"
for LAM in $MISSING; do
  echo "  --- lambda $LAM ---"
  grep -nE "$LAM" "$FREE/log.txt" 2>/dev/null | grep -viE "finished block" | head -6 \
    | sed 's/^/     /' || echo "     (no log lines mention this window at all)"
done

# ---------------------------------------------------------------- 4
hr "4. ERROR-SHAPED LINES (free leg)"
grep -niE "error|nan|exception|traceback|fail|unstable|particle|abort|terminated|pool|overflow" \
  "$FREE/log.txt" "$FREE/runner.stdout.log" 2>/dev/null | head -25 | sed 's/^/  /' \
  || echo "  none matched"

# ---------------------------------------------------------------- 5
hr "5. SLURM LOG (run_fep_leg.py embeds SOMD2's error hints here)"
# Match the edge id in file CONTENT or in the FILENAME -- Slurm logs are often
# named after the edge without repeating it inside.
FOUND=0
CANDIDATES=$( { grep -rl "$EDGE" "$RUNS/$REP/_logs/" 2>/dev/null
                find "$RUNS/$REP/_logs/" -name "*$EDGE*" 2>/dev/null; } | sort -u | head -4 )
for f in $CANDIDATES; do
  echo "  --- $f ---"; FOUND=1
  grep -A22 "Expected .* lambda energy trajectories" "$f" 2>/dev/null | head -26 | sed 's/^/     /'
done
[ "$FOUND" = 0 ] && echo "  no _logs entries mention this edge"

# ---------------------------------------------------------------- 6
hr "6. ARTEFACTS ON DISK FOR DEAD WINDOWS (started-and-died vs never-started)"
for LAM in $MISSING; do
  HITS=$(ls -1 "$FREE" 2>/dev/null | grep -F "$LAM" | tr '\n' ' ')
  echo "  lambda $LAM: ${HITS:-<nothing>}"
done
echo "  checkpoints present: $(ls -1 "$FREE"/checkpoint_*.npz 2>/dev/null \
      | sed 's|.*checkpoint_||;s|\.npz||' | sort -n | tr '\n' ' ')"

# ---------------------------------------------------------------- 7
hr "7. SYSTEM CONSTRUCTION WARNINGS (setup + ghost handling)"
grep -iE "warning|error|failed|could not" "$SETUP/state_a_free_tleap.log" 2>/dev/null \
  | head -10 | sed 's/^/  tleap: /' || echo "  tleap: clean or log absent"
grep -iE "rotamer anchor|ghost|constraint.*not the same|zero sigma" "$FREE/log.txt" 2>/dev/null \
  | head -10 | sed 's/^/  ghostly: /' || echo "  ghostly: nothing"

# ---------------------------------------------------------------- 8
hr "8. PERTURBATION SIZE"
python3 - "$SETUP/fep_preparation.complete.json" <<'PY' 2>/dev/null || echo "  prep json unreadable"
import json, sys
p = json.load(open(sys.argv[1]))
m = p.get("mapping") or []
if m:
    na = max(x["state_a_atom"] for x in m) + 1
    nb = max(x["state_b_atom"] for x in m) + 1
    print(f"  mapped atoms       : {len(m)}")
    print(f"  state A atoms      : {na}   dummies in A: {na-len(m)}")
    print(f"  state B atoms      : {nb}   dummies in B: {nb-len(m)}")
print(f"  mapped_heavy_fraction: {p.get('mapped_heavy_fraction')}")
print(f"  charge change        : {p.get('charge_change')}")
PY

echo
echo "=== READING THIS ==="
echo "  Compare sections 1 and 2. If a window appears in section 2 (it reached"
echo "  minimisation) but is absent from section 1 (no energy parquet), it"
echo "  CONSTRUCTED fine and then failed to minimise or equilibrate -- look for"
echo "  \"could not minimise the system while simultaneously satisfying the"
echo "  constraints\" in section 4."
echo
echo "  That failure is about CONSTRAINTS on perturbable atoms, not lambda spacing."
echo "  Adding lambda windows does NOT help: each window minimises the same"
echo "  starting coordinates at its own lambda, so an intermediate value is no"
echo "  easier than its neighbours. Replica exchange does not help either -- this"
echo "  happens before any dynamics, which is why it reproduces exactly across"
echo "  replicates while velocities vary."
echo
echo "  Try instead: perturbable_constraint: none (with timestep: 2 fs), a larger"
echo "  shift_delta to soften the intermediate-lambda LJ singularity, or a mapping"
echo "  with fewer ghost atoms on rotatable sp3 anchors (see sections 7 and 8)."
echo
echo "  Note that auto_fix_minimise can rescue a failed minimisation -- check"
echo "  section 4 for windows that failed and still produced a parquet. Those are"
echo "  survivors, not successes, and they mean the protocol is marginal."
