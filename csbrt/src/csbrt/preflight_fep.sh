#!/usr/bin/env bash
# Pre-flight the FEP env ON A GPU COMPUTE NODE before submitting a network.
# Must run on a GPU node (srun/sbatch), NOT the login node and NOT a dev box:
#
#   srun --gres=gpu:1 --time=00:20:00 scripts/preflight_fep.sh \
#        runs/fep-runs/x6738a_to_x7024a/setup/x6738a_to_x7024a_free.bss
#
# Gate 3 (OpenMM CUDA context) fails in ~2 s if the env's CUDA build is newer
# than the node driver (the CUDA_ERROR_UNSUPPORTED_PTX_VERSION that wasted the
# weekend). Gate 4 runs a tiny real SOMD2 leg to confirm end-to-end execution.
set -eo pipefail
PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEP_ENV="${FEP_ENV:-automated-fep}"
MAX_CUDA="${MAX_CUDA:-12.8}"   # openbiosim/SOMD2 is incompatible with 12.9; cluster driver caps here too
STREAM="${1:-}"
set +u
eval "$("$HOME/miniforge3/bin/mamba" shell hook --shell bash)"
mamba activate "$FEP_ENV"
set -u

echo "== 1. node GPU + driver =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' || true

echo "== 2. env CUDA build (must be <= the driver's CUDA above) =="
mamba list -n "$FEP_ENV" 2>/dev/null | grep -iE 'cuda-version|^openmm ' || true

echo "== 2b. assert env CUDA <= $MAX_CUDA (openbiosim/SOMD2 breaks on 12.9) =="
# NB: no early `awk exit` here — under `set -o pipefail` it SIGPIPEs `mamba list`
# (exit 141) and aborts the script. Read the whole stream, print the last match.
cv="$(mamba list -n "$FEP_ENV" 2>/dev/null | awk '$1=="cuda-version"{v=$2} END{print v}')"
if [[ -n "$cv" ]]; then
  highest="$(printf '%s\n%s\n' "$cv" "$MAX_CUDA" | sort -V | tail -1)"
  if [[ "$cv" != "$MAX_CUDA" && "$highest" == "$cv" ]]; then
    echo "PREFLIGHT FAIL: env cuda-version=$cv > $MAX_CUDA. openbiosim/SOMD2 is incompatible" >&2
    echo "  with 12.9 and the cluster driver caps at $MAX_CUDA. Rebuild the env pinned to" >&2
    echo "  cuda-version=$MAX_CUDA (see environment-fep.yml)." >&2
    exit 1
  fi
  echo "   env cuda-version=$cv (<= $MAX_CUDA) OK"
else
  echo "   (cuda-version not listed; relying on the CUDA context test below)"
fi

echo "== 3. OpenMM CUDA context + kernel (THE check that catches PTX 222) =="
# Must use a real particle and actually run a kernel: an empty System errors with
# "no particles" before CUDA is touched, and merely creating a context may not
# load the PTX. getState() forces kernel execution, which is what fails on 222.
python - <<'PY'
import openmm as mm, openmm.unit as u
s = mm.System(); s.addParticle(1.0)
nb = mm.NonbondedForce(); nb.addParticle(0.0, 0.1, 0.0); s.addForce(nb)
ctx = mm.Context(s, mm.VerletIntegrator(1.0 * u.femtosecond),
                 mm.Platform.getPlatformByName("CUDA"))
ctx.setPositions([[0, 0, 0]])
ctx.getState(getEnergy=True)
print("   OK — CUDA context created and kernel ran on", ctx.getPlatform().getName())
PY

if [[ -z "$STREAM" ]]; then
  echo "== 4. skipped (no stream given) — pass a *_bound.bss or *_free.bss to test a real leg =="
  echo "PREFLIGHT: CUDA context OK. Env/driver are compatible."
  exit 0
fi

echo "== 4. tiny real SOMD2 leg on $STREAM (2 windows, 20 ps) =="
cfg="$(mktemp --suffix=.yaml)"
cat > "$cfg" <<YAML
runtime: 20 ps
timestep: 4 fs
temperature: 300 K
num_lambda: 2
equilibration_time: 5 ps
energy_frequency: 5 ps
YAML
out="$(mktemp -d)"
python -u "$PIPE/run_fep_leg.py" --stream "$STREAM" --config "$cfg" \
  --output-dir "$out" --leg free --platform cuda --max-gpus 1
n="$(ls "$out"/energy_traj_*.parquet 2>/dev/null | wc -l)"
[[ "$n" -eq 2 ]] || { echo "PREFLIGHT FAIL: expected 2 parquet windows, got $n" >&2; exit 1; }
echo "PREFLIGHT PASS: SOMD2 produced $n/2 energy windows on this node. Safe to submit."
