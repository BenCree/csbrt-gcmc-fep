# Full REX FEP network on HPC — 52 edges × 3 replicates

Commands to run the whole edge network with Hamiltonian replica exchange, three
times. Written 2026-07-30. Paths assume `$HOME/cry/project_2`; adjust `PIPE` if
your checkout differs.

Replica exchange is enabled through the **somd2 config**, not a submitter flag:
`fep_edge.slurm` passes `--config "$FEP_CONFIG"` to `run_fep_leg.py` for both
legs, so whatever the config says is what runs.

```bash
PIPE=$HOME/cry/project_2/scripts
RUNS=$HOME/cry/project_2/runs
```

---

## 0. Before anything: does REX fit in GPU memory?

This is the binding constraint, not wall time. somd2 keeps **every replica
resident on the GPU simultaneously** — that is why REX costs almost nothing in
time (measured: 64 s vs 66 s for 5 windows on an RTX 2080 Ti), but it means
memory scales with `num_lambda`.

Run **one bound leg** and read the estimate somd2 prints before it starts:

```bash
grep -E "Memory per replica|Estimated memory usage" \
     $RUNS/rex-probe/<edge>/bound/runner.stdout.log
```

Expect lines like:

```
Memory per replica on device 0: first = 174 MiB, marginal = 174 MiB
Estimated memory usage on device 0 after creating all replicas: 1.92 GB, Available: 11.00 GB
```

174 MiB/replica was a ~3,000-atom free leg. A solvated protein bound leg is far
larger, and 11 of them may not fit. If the estimate exceeds the card, reduce
`num_lambda` or request a bigger GPU — do not just launch 52 edges and hope.
Note `--batch` is irrelevant here: every array task gets its own GPU, so the
11-replica footprint is per-task regardless of how many tasks run concurrently.

**If using more than one GPU:** `num_lambda` must be divisible by `num_gpus`
(`somd2/runner/_repex.py:95` raises otherwise). 11 windows on 2 GPUs fails;
11 on 1 GPU, or 12 on 2, is fine.

---

## 1. Build the REX config

```bash
cat > $HOME/somd2_rex.yaml <<'EOF'
runtime: 5 ns
timestep: 4 fs
temperature: 300 K
pressure: 1 atm
num_lambda: 11
lambda_schedule: standard_morph
equilibration_time: 100 ps

# Mixing work scales as num_lambda^2 per cycle, and
#   cycles = runtime / energy_frequency
# At the old 1 ps that is 5000 cycles; 10 ps gives 500, still 500 energy
# samples per window. Raise this first if REX turns out to be slow.
energy_frequency: 10 ps

frame_frequency: 100 ps
checkpoint_frequency: 100 ps
save_trajectories: true
save_energy_components: false

replica_exchange: true

# Only fires during REX cycles or terminal-flip MC moves. With replica_exchange
# false and no terminal flips it is a no-op -- which is why the earlier
# replicates started from identical velocities.
randomise_velocities: true

# Make failures speak instead of exiting 0 with dead windows.
save_crash_report: true
minimisation_errors: true

overwrite: false
EOF
```

---

## 2. Submit the three replicates

Same manifest each time (bound frames stay the medoid, so the receptor
conformation is held fixed); only the run root changes. With REX plus
`randomise_velocities`, the three runs genuinely diverge.

```bash
for REP in 1 2 3; do
  bash $PIPE/submit_fep_edges.sh \
    --manifest $RUNS/fep_manifest.tsv \
    --run-root $RUNS/rex-rep$REP \
    --config   $HOME/somd2_rex.yaml \
    --batch    8 \
    --without-gcmc \
    --partition <PARTITION> --account <ACCOUNT>
done
```

- `--batch 8` is a Slurm array throttle: up to 8 tasks at once, **each with its
  own dedicated GPU** (`fep_edge.slurm` requests `--gres=gpu:1` per task).
  Nothing is shared, so lowering it does NOT reduce REX memory pressure -- that
  is per-task. It only controls how many GPUs you occupy at once.
- `--without-gcmc` is correct: bound-leg waters are already placed by the Loch
  endpoint, and the free leg never uses GCMC.
- Add `--dry-run` first to print the sbatch commands without submitting.

**Do one edge before all 52 × 3.** A single edge costs ~2 GPU-hours and catches
a memory or config error that would otherwise waste 156 jobs:

```bash
head -2 $RUNS/fep_manifest.tsv > $RUNS/one_edge.tsv
bash $PIPE/submit_fep_edges.sh --manifest $RUNS/one_edge.tsv \
  --run-root $RUNS/rex-probe --config $HOME/somd2_rex.yaml --batch 1 \
  --without-gcmc --partition <PARTITION> --account <ACCOUNT>
```

---

## 3. Verify by artefact, never by exit status

somd2 **exits 0 with every lambda window dead** — reproduced locally: 0 of 2
windows written, exit code 0, errors only inside its own log. `run_fep_leg.py`
catches this per leg (it refuses to publish `fep_leg.complete.json` unless the
parquet count matches `num_lambda`), but across 156 leg-runs one missing marker
is easy to overlook.

```bash
python -m csbrt.audit_fep_network $RUNS --glob 'rex-rep*'
```

Reports incomplete edges (including which exact lambda values to rerun), whether
the replicates are genuinely independent, and DDG spread stratified by overlap.

---

## 4. Did REX actually help?

The whole point is adjacent-window overlap. Compare against the pre-REX runs:

```bash
python - <<'PY'
import json, pathlib
runs = pathlib.Path.home()/"cry/project_2/runs"
for edge in ("x7161a_to_x7257a", "x7190a_to_x7317a", "x7026a_to_x7247a"):
    row = [edge]
    for root in ("fep-runs-rep2", "rex-rep1"):
        f = runs/root/edge/"analysis.json"
        if f.is_file():
            a = json.loads(f.read_text())
            row.append(f"{min(a['bound_adjacent_overlap_minimum'],
                               a['free_adjacent_overlap_minimum']):.3f}")
        else:
            row.append("  --  ")
    print(f"{row[0]:26s} pre-REX={row[1]}  REX={row[2]}")
PY
```

Those three had minima of 0.001, 0.004 and 0.008. Anything above ~0.1 means REX
solved the systematic problem and applies to all 29 low-overlap edges.

Confirm REX genuinely engaged — identical timings can mean the flag was ignored:

```bash
grep -c "Mixing replicas" $RUNS/rex-rep1/<edge>/free/runner.stdout.log   # expect ~500
```

---

## What REX will NOT fix

`x7317a_to_x7427a` loses 4 of 11 free-leg windows (λ = 0.2, 0.3, 0.4, 0.8),
identically in every replicate. The cause is now known, from the logs:

```
Minimisation failed for λ = 0.20000: SireError::invalid_state: Despite repeated
attempts, the minimiser could not minimise the system while simultaneously
satisfying the constraints.
```

The windows construct fine and then fail to minimise. That is **before any
dynamics**, so REX cannot touch it -- and it is why the same windows die in every
replicate while velocities vary. Under REX it may fail harder, since REX builds
every replica up front and one bad replica can take the run down.

Minimisation actually failed at **six** windows (0.2, 0.3, 0.4, 0.6, 0.7, 0.8);
`auto_fix_minimise` rescued 0.6 and 0.7. Those two are survivors, not successes --
the protocol is marginal across the whole intermediate-lambda range.

Mechanism: this edge grows **12 dummy atoms**, attached through rotatable sp3
bonds. SOMD2 warns about it directly:

```
Potential rotamer anchor at λ = 0: bond 10-9 is a rotatable sp3 bond.
Surviving anchor dihedrals may allow rotameric transitions of ghost atoms.
```

Ghost atoms on freely rotating bonds can swing into overlaps that the minimiser
cannot resolve while holding `h_bonds` constraints fixed.

**Adding lambda windows does not help.** Each window minimises the same starting
coordinates at its own lambda, so a window at 0.25 is no easier than 0.2 or 0.3.
Try instead, on this one edge's free leg first (~2 GPU-hours):

```yaml
perturbable_constraint: none   # the error is about constraints on perturbable atoms
timestep: 2 fs                 # required once X-H bonds are unconstrained
shift_delta: 2.0 A             # softer LJ singularity at intermediate lambda (was 1.5)
minimisation_errors: true      # fail loudly instead of exiting 0
save_crash_report: true
```

If that still fails, the mapping is the problem rather than the protocol:
`x7317a` is marginal across all three of its edges, which points at a badly posed
perturbation rather than bad luck.
