#!/usr/bin/env python3
"""End-to-end driver for the Loch endpoint -> averaging -> FEP -> comparison benchmark.

One entry point that runs the stages in order, shelling out to the individual
tools. Intended as the skeleton for a future Mamba package console-script.

Stages:
  production  submit the endpoint density series (Loch equilibration + GCMC
              production) for every ligand/replicate                [async Slurm]
  frames      select one medoid equilibrated bound frame per ligand (averaging)
  manifest    resolve the reviewed edges to a FEP manifest seeded from those frames
  fep         submit the throttled per-edge FEP array, then network fit + Rowan
              compare                                               [async Slurm]

`production` and `fep` submit Slurm work and return immediately; the frame/manifest
stages need `production` to have finished first. Default runs frames -> fep, which
assumes the density series is already complete. Use --dry-run to print commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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
    p.add_argument("--rowan-edges", type=Path, help="rowan_results_per_edge_wide.csv (compare)")
    p.add_argument("--experimental", type=Path, help="per-compound experimental dG CSV")
    p.add_argument("--fep-env", default="automated-fep")
    p.add_argument("--loch-env", default="cry-loch-babel")
    # production-stage inputs (only needed if --start production)
    p.add_argument("--receptor", type=Path)
    p.add_argument("--ligand-library", type=Path)
    p.add_argument("--replicates", type=int, default=6)
    p.add_argument("--partition"); p.add_argument("--account"); p.add_argument("--qos")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(cmd: list[str], *, dry: bool) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True)


def slurm_flags(opt: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for name in ("partition", "account", "qos"):
        value = getattr(opt, name)
        if value:
            flags += [f"--{name}", value]
    return flags


def main() -> None:
    opt = options()
    if STAGES.index(opt.start) > STAGES.index(opt.through):
        raise SystemExit("--start must not come after --through")
    selected = STAGES[STAGES.index(opt.start): STAGES.index(opt.through) + 1]
    bound_frames = opt.bound_frames or (opt.series_root / "bound_frames")
    manifest = opt.manifest or (opt.run_root / "fep_manifest.tsv")

    if "production" in selected:
        if not (opt.receptor and opt.ligand_library):
            raise SystemExit("production stage needs --receptor and --ligand-library")
        cmd = [HERE / "submit_ev71_density_series.sh", "--run-root", opt.series_root,
               "--receptor", opt.receptor, "--ligand-library", opt.ligand_library,
               "--replicates", opt.replicates, "--max-concurrent", opt.batch] + slurm_flags(opt)
        run(cmd, dry=opt.dry_run)
        print("[fep_benchmark] production submitted; wait for it, then rerun "
              "--start frames.", flush=True)
        if opt.through == "production":
            return
        if not opt.dry_run:
            return  # do not run downstream stages against an incomplete series

    if "frames" in selected:
        run([opt_python(opt.loch_env), HERE / "select_bound_frames.py",
             "--series-root", opt.series_root, "--output-root", bound_frames], dry=opt.dry_run)

    if "manifest" in selected:
        if not opt.edges:
            raise SystemExit("manifest stage needs --edges")
        run([opt_python(opt.loch_env), HERE / "make_fep_manifest.py", "--edges", opt.edges,
             "--endpoint-run-root", opt.series_root, "--bound-frame-root", bound_frames,
             "--output", manifest], dry=opt.dry_run)

    if "fep" in selected:
        cmd = [HERE / "submit_fep_edges.sh", "--manifest", manifest, "--batch", opt.batch,
               "--run-root", opt.run_root, "--fep-env", opt.fep_env] + slurm_flags(opt)
        if opt.rowan_edges:
            cmd += ["--rowan-edges", opt.rowan_edges]
        if opt.experimental:
            cmd += ["--experimental", opt.experimental]
        run(cmd, dry=opt.dry_run)


def opt_python(env: str) -> str:
    """Path to a named Mamba env's python, so each stage uses the right stack."""
    candidate = Path.home() / "miniforge3" / "envs" / env / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


if __name__ == "__main__":
    main()
