#!/usr/bin/env bash
#
# Run a GCI titration sweep: one gci_window.py invocation per Adams value.
#
# Deliberately hardcoded and relative-path only, for development. Everything it
# needs must sit in the SAME directory as this script:
#
#   run_gci_sweep.sh          <- this file
#   gci_window.py             gci_common.py
#   ev71_loch_common.py       pipeline_utils.py
#   <PRMTOP>                  <RST7>          <- your two input files
#
# Run it from that directory:   ./run_gci_sweep.sh
#
# Windows run one after another. Each writes into its own directory under
# $RUN_ROOT, so nothing collides. Re-running is safe: gci_window.py sees its own
# checkpoint, re-validates the outputs, and skips the work. A window that failed
# has no checkpoint and is simply redone.

set -euo pipefail


# ===================== EDIT THIS BLOCK =====================

PREFIX="CRY1KL101"                    # no dots: Sire reads the file format from the extension
PRMTOP="CRY1KL101uvt2.prmtop"
RST7="CRY1KL101uvt2.rst7"

# GCMC sphere. These MUST be in the frame of $RST7 -- a coordinate from a paper
# or from another equilibration run will land somewhere arbitrary and the guard
# in gci_window.py will reject it. Get them from gci_map_centre.py, or from
# clustering your own production waters.
#
# The value below is a pocket water in this KL101 uvt2 frame (2.75 A from the
# ligand, 3.09 A from the protein) -- a working placeholder, not a chosen site.
CENTRE_X="48.331"
CENTRE_Y="57.025"
CENTRE_Z="56.001"
RADIUS="4.0"
NUM_GHOSTS="27"                       # gci_map_centre.py suggests this per radius

RUN_ROOT="gci_runs"

# 1 = ~1 min per window, for checking the setup works.
# 0 = the real schedule: 625 x (400 attempts + 4000 MD steps) = 5 ns per window.
SMOKE=1

# ===========================================================


# The historical ladder: 36 values of B. Non-uniform on purpose -- fine spacing
# through the occupancy transition near B = -17, coarse on the flat tails.
B_LADDER=(
  -30    -28.5  -27    -25.5  -24    -22.5
  -22    -21.5  -21    -20.5  -20    -19.5  -19    -18.5  -18
  -17.75 -17.5  -17.25 -17.125 -17
  -16.75 -16.5  -16.25 -16
  -15.5  -15    -14.5  -14    -13.5  -13
  -11.5  -10    -8.5   -7     -5.5   -4
)

if [ "$SMOKE" -eq 1 ]; then
  SCHEDULE=(--cycles 10 --attempts 400 --md-steps 200 --report-interval 100)
  echo "### SMOKE MODE: short windows. Set SMOKE=0 for the real 5 ns schedule."
else
  SCHEDULE=()          # no flags => gci_window.py uses the published defaults
  echo "### FULL MODE: 5 ns per window, ${#B_LADDER[@]} windows."
fi

# Fail before the loop if anything is missing, rather than 36 times inside it.
for f in gci_window.py gci_common.py ev71_loch_common.py pipeline_utils.py \
         "$PRMTOP" "$RST7"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: ./$f -- see the header of this script" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT"
echo "prefix=$PREFIX  radius=$RADIUS  ghosts=$NUM_GHOSTS"
echo "centre=($CENTRE_X, $CENTRE_Y, $CENTRE_Z)"
echo "output=$RUN_ROOT/"
echo

FAILED=()

for i in "${!B_LADDER[@]}"; do
  B="${B_LADDER[$i]}"
  DIR="$RUN_ROOT/window_$(printf '%02d' "$i")_B${B}"
  mkdir -p "$DIR"

  echo "=== window $i of ${#B_LADDER[@]}: B = $B -> $DIR"

  # '|| true' so one bad window does not abort the sweep; the exit status is
  # recovered from PIPESTATUS because the pipe through tee would mask it.
  set +e
  python -u gci_window.py \
    --prmtop "$PRMTOP" \
    --rst7 "$RST7" \
    --output-dir "$DIR" \
    --prefix "$PREFIX" \
    --window-index "$i" \
    --sphere-centre "$CENTRE_X" "$CENTRE_Y" "$CENTRE_Z" \
    --sphere-radius "$RADIUS" \
    --num-ghosts "$NUM_GHOSTS" \
    --target-b "$B" \
    "${SCHEDULE[@]}" \
    2>&1 | tee "$DIR/window.log"
  STATUS="${PIPESTATUS[0]}"
  set -e

  if [ "$STATUS" -ne 0 ]; then
    echo "    FAILED (exit $STATUS) -- see $DIR/window.log"
    FAILED+=("$B")
  fi
  echo
done


echo "==================== SUMMARY ===================="
echo "windows attempted : ${#B_LADDER[@]}"
echo "checkpoints found : $(find "$RUN_ROOT" -name gci_window.complete.json | wc -l)"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED at B = ${FAILED[*]}"
  echo "Fix the cause and re-run; completed windows will be skipped."
  exit 1
fi

echo "all windows complete"
echo
echo "Titration points (tail-mean occupancy is the observable):"
# Last row of each window's titration CSV: step,cycle,md,sphere_waters,...
for d in "$RUN_ROOT"/window_*/; do
  csv="$d$PREFIX"_titration.csv
  [ -f "$csv" ] || continue
  b=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['target_b'])" \
        "$d$PREFIX"_titration.json)
  n=$(tail -1 "$csv" | cut -d, -f4)
  printf "  B = %9s   final sphere_waters = %s\n" "$b" "$n"
done
