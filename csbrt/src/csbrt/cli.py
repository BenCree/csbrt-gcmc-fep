"""Command-line entry points for the csbrt workflow.

Each stage is its own command, and ``csbrt`` runs any range of them:

    csbrt-preprocess   --config run.yaml
    csbrt-equilibrate  --config run.yaml
    csbrt-gcmc         --config run.yaml
    csbrt-fep          --config run.yaml
    csbrt-analysis     --config run.yaml

    csbrt --all                          --config run.yaml
    csbrt --from equilibrate --through gcmc --config run.yaml
    csbrt gcmc                           --config run.yaml

The stage implementations shell out to the validated stage scripts that ship
inside this package. They are invoked as subprocesses rather than imported so
their sibling imports keep resolving exactly as they do when run by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STAGES = ("preprocess", "equilibrate", "gcmc", "fep", "analysis")


# --------------------------------------------------------------------------- helpers

def load_config(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def need(cfg: dict, key: str, stage: str):
    if key not in cfg or cfg[key] in (None, ""):
        raise SystemExit(f"config key '{key}' is required for the '{stage}' stage")
    return cfg[key]


def run(cmd: list, *, dry: bool) -> None:
    print("\n+ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True)


def script(name: str) -> list:
    return [sys.executable, "-u", str(HERE / name)]


def output_dir(cfg: dict) -> Path:
    out = Path(cfg.get("output_dir", "csbrt-run")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepared_ligands(cfg: dict, out: Path) -> Path:
    """Explicit-hydrogen ligand SDF produced by the preprocess stage."""
    return Path(cfg.get("prepared_ligands", out / "ligands_h.sdf"))


def ligand_library(cfg: dict, out: Path, stage: str) -> Path:
    prepared = prepared_ligands(cfg, out)
    return prepared if prepared.exists() else Path(need(cfg, "ligand_sdf", stage))


# --------------------------------------------------------------------------- stages

def stage_preprocess(cfg: dict, out: Path, dry: bool) -> None:
    receptor = Path(need(cfg, "receptor", "preprocess"))
    loop = cfg.get("loop", {})
    query, of3_out = out / "of3_query.json", out / "of3_out"
    grafted = out / "receptor_loopmodelled.pdb"
    protonated = out / "receptor_protonated.pdb"

    run(script("of3_prep.py") + [
        "--receptor", receptor,
        "--template-cif", need(cfg, "template_cif", "preprocess"),
        "--ligand-sdf", need(cfg, "ligand_sdf", "preprocess"),
        "--ligand-name", need(cfg, "ligand_name", "preprocess"),
        "--receptor-chain", cfg.get("receptor_chain", "B"),
        "--template-chain", cfg.get("template_chain", "B"),
        "--loop-after-resid", need(loop, "after_resid", "preprocess"),
        "--loop-seq", need(loop, "sequence", "preprocess"),
        "--output", query,
    ] + (["--use-msas"] if cfg.get("use_msa_server") else []), dry=dry)

    predict = ["run_openfold", "predict", "--query-json", query,
               "--output-dir", of3_out, "--use-templates", "true",
               "--num-diffusion-samples", cfg.get("diffusion_samples", 1),
               "--num-model-seeds", cfg.get("model_seeds", 1),
               "--use-msa-server", "true" if cfg.get("use_msa_server") else "false"]
    # OpenFold3's Triton kernels need Ampere or newer; on Turing point
    # runner_yaml at the bundled fallback config.
    if cfg.get("runner_yaml"):
        predict += ["--runner-yaml", cfg["runner_yaml"]]
    run(predict, dry=dry)

    run(script("of3_graft_loop.py") + [
        "--receptor", receptor, "--model-dir", of3_out, "--output", grafted,
        "--receptor-chain", cfg.get("receptor_chain", "B"),
        "--loop-after-resid", need(loop, "after_resid", "preprocess"),
        "--loop-length", len(str(need(loop, "sequence", "preprocess"))),
    ], dry=dry)

    run(script("of3_protonate.py") + [
        "--input", grafted, "--output", protonated, "--ph", cfg.get("ph", 7.4),
    ] + (["--keep-waters"] if cfg.get("keep_waters") else []), dry=dry)

    run(script("prepare_ligands.py") + [
        "--input", need(cfg, "ligand_sdf", "preprocess"),
        "--output", prepared_ligands(cfg, out),
    ], dry=dry)
    print(f"\npreprocess complete -> {protonated}")


def _endpoint(cfg: dict, out: Path, dry: bool, through: str, stage: str) -> Path:
    ligand = need(cfg, "ligand_name", stage)
    run_dir = out / "endpoint" / ligand / "rep1"
    run(script("run_ev71_pipeline.py") + [
        "--receptor", cfg.get("prepared_receptor", out / "receptor_protonated.pdb"),
        "--ligand-library", ligand_library(cfg, out, stage),
        "--ligand-id", ligand,
        "--run-dir", run_dir,
        "--seed", cfg.get("seed", 20260714),
        "--profile", cfg.get("profile", "full"),
        "--through", through,
    ], dry=dry)
    return run_dir


def stage_equilibrate(cfg: dict, out: Path, dry: bool) -> None:
    _endpoint(cfg, out, dry, "equilibration", "equilibrate")


def stage_gcmc(cfg: dict, out: Path, dry: bool) -> None:
    ligand = need(cfg, "ligand_name", "gcmc")
    # Generic default: no dataset name baked into output filenames.
    prefix = cfg.get("prefix", str(ligand))
    # run_ev71_pipeline stages are preparation/equilibration/production/
    # postprocessing -- there is no "analysis" stage in the EV71 driver.
    run_dir = _endpoint(cfg, out, dry, "production", "gcmc")
    production = run_dir / "production"
    density = script("ev71_density_sites.py") + [
        "--topology", production / f"{prefix}-loch-ghosts.pdb",
        "--trajectory", production / f"{prefix}-raw.dcd",
        "--ghost-file", production / f"{prefix}-gcmc-ghosts.txt",
        "--output-dir", run_dir / "density_analysis",
        "--prefix", prefix,
    ]
    if cfg.get("site_catalog"):
        density += ["--site-catalog", cfg["site_catalog"]]
    run(density, dry=dry)


def stage_fep(cfg: dict, out: Path, dry: bool) -> None:
    series = Path(cfg.get("series_root", out / "endpoint"))
    frames, manifest = out / "bound_frames", out / "fep_manifest.tsv"
    run(script("select_bound_frames.py") + [
        "--series-root", series, "--output-root", frames,
        # A single-ligand run writes 'density_analysis'; 'common_site_analysis'
        # only exists after the cross-run finalizer.
        "--site-metrics-subdir", cfg.get("site_metrics_subdir", "common_site_analysis"),
    ], dry=dry)
    run(script("make_fep_manifest.py") + [
        "--edges", need(cfg, "edges", "fep"), "--endpoint-run-root", series,
        "--bound-frame-root", frames, "--output", manifest,
    ], dry=dry)
    print("\nFEP edges resolved. Submit the network with submit_fep_edges.sh, "
          "or run one edge locally with run_fep_leg.py (see README).")


def stage_analysis(cfg: dict, out: Path, dry: bool) -> None:
    fep_root = Path(cfg.get("fep_root", out / "fep-runs"))

    # Sweep for incomplete edges BEFORE fitting. run_fep_leg.py catches a single
    # leg missing lambda windows, but across a whole network one absent marker is
    # easy to miss -- and SOMD2 can exit 0 with every window dead.
    run(script("audit_fep_network.py") + [
        fep_root.parent,
        "--glob", cfg.get("replicate_glob", fep_root.name),
        "--overlap-threshold", cfg.get("overlap_threshold", 0.15),
    ], dry=dry)

    run(script("aggregate_fep_network.py") + [
        "--manifest", cfg.get("fep_manifest", out / "fep_manifest.tsv"),
        "--fep-root", fep_root, "--output-dir", fep_root / "network_analysis",
    ], dry=dry)
    if cfg.get("rowan_edges"):
        cmd = script("compare_to_rowan.py") + [
            "--network", fep_root / "network_analysis" / "fep_network_analysis.json",
            "--rowan-edges", cfg["rowan_edges"],
            "--output-dir", fep_root / "rowan_comparison",
        ]
        if cfg.get("experimental"):
            cmd += ["--experimental", cfg["experimental"]]
        run(cmd, dry=dry)

    # Torsional sampling. An edge can have healthy window overlap and still be
    # wrong if a rotatable bond never crossed its barrier, so this runs as part
    # of the stage rather than being left as a suggestion in the output.
    ligands = ligand_library(cfg, out, "analysis")
    for edge_dir in sorted(p for p in fep_root.glob("*") if p.is_dir()):
        if edge_dir.name == "network_analysis":
            continue
        for leg in ("free", "bound"):
            topology = edge_dir / leg / "system0.prm7"
            trajectories = sorted((edge_dir / leg).glob("traj_*.dcd"))
            if not topology.is_file() or not trajectories:
                continue
            run(script("torsion_diagnostics.py") + [
                "--topology", topology,
                "--trajectory", trajectories[0],
                "--ligand-sdf", ligands,
                "--ligand-resname", cfg.get("ligand_resname", "LIG"),
                "--output", edge_dir / f"torsion_diagnostics_{leg}.json",
            ], dry=dry)


RUNNERS = {
    "preprocess": stage_preprocess,
    "equilibrate": stage_equilibrate,
    "gcmc": stage_gcmc,
    "fep": stage_fep,
    "analysis": stage_analysis,
}


# --------------------------------------------------------------------------- CLI

def _base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True,
                   help="YAML/JSON config; see config.example.yaml")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    return p


def _single_stage(stage: str):
    """Build a console-script entry point for one stage."""

    def entry(argv: list | None = None) -> None:
        opt = _base_parser(f"Run the '{stage}' stage of the csbrt workflow.").parse_args(argv)
        cfg = load_config(opt.config)
        out = output_dir(cfg)
        print(f"csbrt: stage {stage}   output {out}")
        RUNNERS[stage](cfg, out, opt.dry_run)
        print("\ncsbrt: done")

    entry.__name__ = f"{stage}_entry"
    return entry


preprocess = _single_stage("preprocess")
equilibrate = _single_stage("equilibrate")
gcmc = _single_stage("gcmc")
fep = _single_stage("fep")
analysis = _single_stage("analysis")


def main(argv: list | None = None) -> None:
    p = _base_parser(__doc__)
    p.add_argument("stage", nargs="?", choices=STAGES,
                   help="Run exactly this stage (omit with --all/--from/--through)")
    p.add_argument("--all", action="store_true", help="Run every stage in order")
    p.add_argument("--from", dest="start", choices=STAGES)
    p.add_argument("--through", dest="through", choices=STAGES)
    opt = p.parse_args(argv)

    if not (opt.stage or opt.all or opt.start or opt.through):
        raise SystemExit("give a stage, or --all, or --from/--through")
    if opt.stage:
        selected = [opt.stage]
    else:
        start = opt.start or STAGES[0]
        through = opt.through or STAGES[-1]
        if STAGES.index(start) > STAGES.index(through):
            raise SystemExit("--from must not come after --through")
        selected = list(STAGES[STAGES.index(start): STAGES.index(through) + 1])

    cfg = load_config(opt.config)
    out = output_dir(cfg)
    print(f"csbrt: stages {' -> '.join(selected)}   output {out}")
    for stage in selected:
        print(f"\n{'=' * 70}\n== {stage}\n{'=' * 70}")
        RUNNERS[stage](cfg, out, opt.dry_run)
    print("\ncsbrt: done")


if __name__ == "__main__":
    main()
