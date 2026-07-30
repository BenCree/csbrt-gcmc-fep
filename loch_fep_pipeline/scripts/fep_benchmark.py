#!/usr/bin/env python3
"""End-to-end driver for the Loch endpoint -> averaging -> FEP -> comparison benchmark.

One entry point that runs the stages in order, shelling out to the individual
tools in this directory. Intended as the skeleton for a future Mamba package
console-script.

Stages:
  production  submit the endpoint density series (preparation, Loch equilibration,
              MD/GCMC production, density analysis) for every ligand/replicate
              via submit_md_gcmc_series.sh                          [async Slurm]
  frames      pick one equilibrated bound frame per ligand: the medoid replicate
              (select_bound_frames.py) or a consensus-hydration frame
              (build_consensus_frames.py)
  manifest    resolve the reviewed edges to a FEP manifest seeded from those frames
  fep         submit the throttled per-edge FEP array, then network fit and the
              optional Rowan/experimental comparison                [async Slurm]

`production` and `fep` submit Slurm work and return immediately; the frame/manifest
stages need `production` to have finished first. Default runs frames -> fep, which
assumes the density series is already complete. Use --dry-run to print commands.

Environments: the endpoint and FEP stages run in separate Mamba envs by default
(`cry-loch-babel` and `automated-fep`). To use the unified environment.yml stack
instead, pass `--loch-env loch-fep --fep-env loch-fep`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

STAGES = ("production", "frames", "manifest", "fep")
HERE = Path(__file__).resolve().parent


def options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series-root", type=Path, required=True,
                   help="Endpoint density-series root (LIGAND/repN/...)")
    p.add_argument("--edges", type=Path, help="Reviewed edge TSV (required for manifest/fep)")
    p.add_argument("--bound-frames", type=Path, help="Bound-frame output dir "
                   "(default: <series-root>/bound_frames)")
    p.add_argument("--manifest", type=Path, help="FEP manifest path "
                   "(default: <run-root>/fep_manifest.tsv)")
    p.add_argument("--run-root", type=Path, default=Path("fep-runs"))
    p.add_argument("--batch", type=int, default=8, help="Concurrent FEP edges (array %%N)")
    p.add_argument("--start", choices=STAGES, default="frames")
    p.add_argument("--through", choices=STAGES, default="fep")
    p.add_argument("--fep-env", default="automated-fep")
    p.add_argument("--loch-env", default="cry-loch-babel")
    p.add_argument("--dry-run", action="store_true")

    endpoint = p.add_argument_group("production stage (only needed if --start production)")
    endpoint.add_argument("--dataset", type=Path,
                          help="Endpoint dataset TSV from prepare_openbind_dataset.py")
    endpoint.add_argument("--replicates", type=int, default=6)
    endpoint.add_argument("--base-seed", type=int, default=20260714)
    endpoint.add_argument("--profile", default="full", choices=("full", "smoke"))
    endpoint.add_argument("--max-concurrent", type=int,
                          help="Concurrent endpoint tasks (default: --batch)")

    frames = p.add_argument_group("frames stage")
    frames.add_argument("--frames-method", choices=("medoid", "consensus"), default="medoid",
                        help="medoid: select_bound_frames.py (default). "
                             "consensus: build_consensus_frames.py (needs --common-catalog "
                             "and --alignment-reference)")
    frames.add_argument("--common-catalog", type=Path, help="Frozen common site catalog "
                        "(consensus frames only)")
    frames.add_argument("--alignment-reference", type=Path, help="Shared receptor reference "
                        "(consensus frames only)")

    manifest = p.add_argument_group("manifest stage")
    manifest.add_argument("--replica", type=int, default=1,
                          help="Endpoint replica to resolve preparations from")

    fep = p.add_argument_group("fep stage")
    fep.add_argument("--fep-config", type=Path, help="SOMD2 config yaml "
                     "(default: somd2_config.yaml)")
    fep.add_argument("--with-gcmc", action="store_true",
                     help="Run bound-leg GCMC during FEP (off by default: the seeded "
                          "frame already carries the equilibrated waters)")
    fep.add_argument("--no-aggregate", action="store_true",
                     help="Skip the dependent network-fit job")
    fep.add_argument("--rowan-edges", type=Path, help="rowan_results_per_edge_wide.csv (compare)")
    fep.add_argument("--rowan-edge-column", help="Rowan DDG column")
    fep.add_argument("--experimental", type=Path, help="per-compound experimental dG CSV")
    fep.add_argument("--partition"); fep.add_argument("--account"); fep.add_argument("--qos")
    return p.parse_args()


def run(cmd: list, *, dry: bool) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True)


def slurm_flags(opt: argparse.Namespace) -> list[str]:
    """Slurm passthrough. Only submit_fep_edges.sh accepts these."""
    flags: list[str] = []
    for name in ("partition", "account", "qos"):
        value = getattr(opt, name)
        if value:
            flags += [f"--{name}", value]
    return flags


def env_python(env: str, *, dry: bool) -> str:
    """Path to a named Mamba env's python, so each stage uses the right stack."""
    candidate = Path.home() / "miniforge3" / "envs" / env / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    if dry:
        return str(candidate)
    raise SystemExit(f"No python in Mamba env '{env}' (expected {candidate}). "
                     "Create it or pass --loch-env/--fep-env.")


def main() -> None:
    opt = options()
    if STAGES.index(opt.start) > STAGES.index(opt.through):
        raise SystemExit("--start must not come after --through")
    selected = STAGES[STAGES.index(opt.start): STAGES.index(opt.through) + 1]
    bound_frames = opt.bound_frames or (opt.series_root / "bound_frames")
    manifest = opt.manifest or (opt.run_root / "fep_manifest.tsv")

    if "production" in selected:
        if not opt.dataset:
            raise SystemExit("production stage needs --dataset (see prepare_openbind_dataset.py)")
        if slurm_flags(opt):
            print("[fep_benchmark] note: submit_md_gcmc_series.sh has no partition/account/qos "
                  "passthrough; set them in md_gcmc_task.slurm or via SBATCH_* env vars.",
                  flush=True)
        cmd = [HERE / "submit_md_gcmc_series.sh",
               "--dataset", opt.dataset,
               "--run-root", opt.series_root,
               "--replicates", opt.replicates,
               "--base-seed", opt.base_seed,
               "--max-concurrent", opt.max_concurrent or opt.batch,
               "--conda-env", opt.loch_env,
               "--profile", opt.profile]
        run(cmd, dry=opt.dry_run)
        print("[fep_benchmark] production submitted; wait for the array AND its "
              "common-catalog finalizer, then rerun --start frames.", flush=True)
        if opt.through == "production":
            return
        if not opt.dry_run:
            return  # do not run downstream stages against an incomplete series

    if "frames" in selected:
        python = env_python(opt.loch_env, dry=opt.dry_run)
        if opt.frames_method == "consensus":
            if not (opt.common_catalog and opt.alignment_reference):
                raise SystemExit("--frames-method consensus needs --common-catalog "
                                 "and --alignment-reference")
            run([python, HERE / "build_consensus_frames.py",
                 "--series-root", opt.series_root, "--output-root", bound_frames,
                 "--common-catalog", opt.common_catalog,
                 "--alignment-reference", opt.alignment_reference], dry=opt.dry_run)
        else:
            run([python, HERE / "select_bound_frames.py",
                 "--series-root", opt.series_root, "--output-root", bound_frames],
                dry=opt.dry_run)

    if "manifest" in selected:
        if not opt.edges:
            raise SystemExit("manifest stage needs --edges")
        run([env_python(opt.loch_env, dry=opt.dry_run), HERE / "make_fep_manifest.py",
             "--edges", opt.edges, "--endpoint-run-root", opt.series_root,
             "--replica", opt.replica, "--bound-frame-root", bound_frames,
             "--output", manifest], dry=opt.dry_run)

    if "fep" in selected:
        cmd = [HERE / "submit_fep_edges.sh", "--manifest", manifest, "--batch", opt.batch,
               "--run-root", opt.run_root, "--fep-env", opt.fep_env] + slurm_flags(opt)
        if opt.fep_config:
            cmd += ["--config", opt.fep_config]
        cmd += ["--with-gcmc"] if opt.with_gcmc else ["--without-gcmc"]
        if opt.no_aggregate:
            cmd += ["--no-aggregate"]
        if opt.rowan_edges:
            cmd += ["--rowan-edges", opt.rowan_edges]
        if opt.rowan_edge_column:
            cmd += ["--rowan-edge-column", opt.rowan_edge_column]
        if opt.experimental:
            cmd += ["--experimental", opt.experimental]
        run(cmd, dry=opt.dry_run)


if __name__ == "__main__":
    main()
