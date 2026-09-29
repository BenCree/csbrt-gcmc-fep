#!/usr/bin/env bash
# Hyperparameter sweep for the FEP simulation, crossed with the water arms
# (bound-leg GCMC off / on). One submit_fep_edges.sh call per grid point, each with
# its own somd2 config and run root, so runs never share checkpoints.
#
#   ./submit_fep_hparam_sweep.sh --manifest one_edge.tsv --replicates 3 --dry-run
#   ./submit_fep_hparam_sweep.sh --manifest one_edge.tsv --sweep-root $RUNS/hp \
#       --partition P --account A
#   python summarise_fep_hparam_sweep.py $RUNS/hp
#
# Use a SMALL manifest (1-3 edges you already know are hard, e.g. low overlap).
# Grid = every combination of the axes below x the arms. Each axis is a
# space-separated list; override by flag or environment.
#
# somd2 axes (written into the per-point config):
#   RUNTIMES     "2 ns 5 ns"  -> use --runtimes "2 ns,5 ns" (comma separated)
#   NUM_LAMBDAS  "11,16,21"
#   TIMESTEPS    "2 fs,4 fs"
#   REX          "false,true"   replica_exchange
#   ENERGY_FREQS "1 ps,10 ps"   energy_frequency (REX cycle length)
# GCMC axes (only used by arm=gcmc):
#   --gcmc-radii "6 A,10 A"  --gcmc-num-waters "20,45"  --gcmc-bulk-probs "0.0"
# Arms: nowater = --without-gcmc, gcmc = --with-gcmc (default: both).
: "${BASH_VERSION:?Run this script with Bash}"
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MANIFEST=""; SWEEP_ROOT="$PWD/hp-sweep"; BATCH=4; DRY=0; REPLICATES=3
PARTITION=""; ACCOUNT=""; QOS=""
RUNTIMES="5 ns"; NUM_LAMBDAS="11"; TIMESTEPS="4 fs"; REX="false,true"; ENERGY_FREQS="10 ps"
SWEEP_GCMC_RADII="10 A"; SWEEP_GCMC_NWAT="45"; SWEEP_GCMC_BULK="0.0"
ARMS="nowater,gcmc"
BASE_CONFIG="$here/somd2_config.yaml"

usage() { sed -n '2,24p' "$0"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --sweep-root) SWEEP_ROOT="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --replicates) REPLICATES="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --runtimes) RUNTIMES="$2"; shift 2 ;;
    --num-lambdas) NUM_LAMBDAS="$2"; shift 2 ;;
    --timesteps) TIMESTEPS="$2"; shift 2 ;;
    --rex) REX="$2"; shift 2 ;;
    --energy-freqs) ENERGY_FREQS="$2"; shift 2 ;;
    --gcmc-radii) SWEEP_GCMC_RADII="$2"; shift 2 ;;
    --gcmc-num-waters) SWEEP_GCMC_NWAT="$2"; shift 2 ;;
    --gcmc-bulk-probs) SWEEP_GCMC_BULK="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -s "$MANIFEST" && -s "$BASE_CONFIG" ]] || { echo "--manifest and --base-config must exist" >&2; exit 2; }

slug() { echo "$1" | tr -d ' ' | tr '/' '-' | tr -c 'A-Za-z0-9._\n-' '_'; }
split() { IFS=',' read -ra _out <<< "$1"; printf '%s\n' "${_out[@]}"; }

mkdir -p "$SWEEP_ROOT/_configs"
n=0
while IFS= read -r arm; do
 while IFS= read -r rt; do while IFS= read -r nl; do while IFS= read -r ts; do
 while IFS= read -r rex; do while IFS= read -r ef; do
  if [[ "$arm" == "gcmc" ]]; then
    radii="$SWEEP_GCMC_RADII"; nwat="$SWEEP_GCMC_NWAT"; bulk="$SWEEP_GCMC_BULK"
  else
    radii="-"; nwat="-"; bulk="-"   # GCMC axes are meaningless without GCMC
  fi
  while IFS= read -r rad; do while IFS= read -r nw; do while IFS= read -r bp; do
    name="${arm}_rt$(slug "$rt")_nl${nl}_ts$(slug "$ts")_rex${rex}_ef$(slug "$ef")"
    [[ "$arm" == "gcmc" ]] && name+="_r$(slug "$rad")_w${nw}_b${bp}"
    cfg="$SWEEP_ROOT/_configs/$name.yaml"
    # Override the base config's keys; everything else (constraints, lambda_schedule,
    # ...) is inherited so only the swept hparams differ between points.
    python - "$BASE_CONFIG" "$cfg" "$rt" "$nl" "$ts" "$rex" "$ef" <<'PY'
import sys, yaml
base, out, rt, nl, ts, rex, ef = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg.update(runtime=rt, num_lambda=int(nl), timestep=ts,
           replica_exchange=(rex == "true"), energy_frequency=ef)
if rex == "true":
    cfg["randomise_velocities"] = True
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
    # Replicates are mandatory in practice: on this network the seed alone moves an
    # edge by a median 0.33 kcal/mol (four edges by >2), which is larger than any
    # single-run difference between settings. One run per point cannot be read.
    for rep in $(seq 1 "$REPLICATES"); do
      cmd=(bash "$here/submit_fep_edges.sh" --manifest "$MANIFEST"
           --run-root "$SWEEP_ROOT/${name}__rep${rep}" --config "$cfg" --batch "$BATCH" --no-aggregate)
      if [[ "$arm" == "gcmc" ]]; then
        cmd+=(--with-gcmc)
        export GCMC_RADIUS="$rad" GCMC_NUM_WATERS="$nw" GCMC_BULK_SAMPLING_PROBABILITY="$bp"
      else
        cmd+=(--without-gcmc)
      fi
      [[ -n "$PARTITION" ]] && cmd+=(--partition "$PARTITION")
      [[ -n "$ACCOUNT" ]] && cmd+=(--account "$ACCOUNT")
      [[ -n "$QOS" ]] && cmd+=(--qos "$QOS")
      n=$((n + 1))
      echo "=== [$n] $name rep$rep ==="
      if [[ "$DRY" -eq 1 ]]; then
        printf '  %q' "${cmd[@]}"; echo
      else
        CSBRT_SYNC="${CSBRT_SYNC:-0}" "${cmd[@]}"
      fi
    done
    unset GCMC_RADIUS GCMC_NUM_WATERS GCMC_BULK_SAMPLING_PROBABILITY
  done < <(split "$bulk"); done < <(split "$nwat"); done < <(split "$radii")
 done < <(split "$ENERGY_FREQS"); done < <(split "$REX")
 done < <(split "$TIMESTEPS"); done < <(split "$NUM_LAMBDAS"); done < <(split "$RUNTIMES")
done < <(split "$ARMS")
echo "runs submitted: $n  (grid points x $REPLICATES replicates; x $(( $(wc -l < "$MANIFEST") - 1 )) edge(s) each)"
echo "summarise : python $here/summarise_fep_hparam_sweep.py $SWEEP_ROOT"
