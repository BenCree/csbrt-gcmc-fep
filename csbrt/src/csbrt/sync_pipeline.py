#!/usr/bin/env python3
"""Compare the local stage scripts against a pinned GitHub ref and update them.

The cluster checkout is not a git clone -- it is a flat `scripts/` directory that also
holds files the repo does not have (submit_rex_replicates.sh). Keeping it in step has
meant hand-copying individual files, which is how a run ends up half-patched: on
2026-07-31 the free-leg alignment fix was on the cluster but the checkpoint-restart fix
was not, and nothing reported the mismatch.

This diffs every tracked file by content hash and fetches only what differs.

Deliberately conservative, because silently swapping code under a running campaign is
worse than being out of date:

* reports by default; `--apply` is required to write anything;
* pins to a commit SHA by default, since a branch can move between submitting the
  manifest and the array tasks starting;
* backs up whatever it replaces to `<file>.bak-<sha>`;
* compiles every Python file before installing it, so a truncated download cannot
  brick the pipeline;
* refuses to run while Slurm jobs are active unless forced -- changing a stage script
  mid-array gives different tasks different code, and `implementation_signature()`
  will invalidate checkpoints in ways that are painful to unpick.

Run it on the frontend BEFORE submitting, never from inside a job script.

Only the standard library is used so it works in any environment on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = "BenCree/csbrt-gcmc-fep"
SOURCE_PREFIX = "csbrt/src/csbrt/"
API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"


def fetch(url: str, *, token: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "csbrt-sync"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"GitHub returned {error.code} for {url}") from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Could not reach {url}: {error.reason}. If the cluster has no outbound "
            "HTTPS, copy the files across with scp instead."
        ) from error


def git_blob_sha(data: bytes) -> str:
    """Git's object id for a blob: sha1 of the header plus content.

    Computing this locally means the tree listing alone tells us which files differ,
    so nothing is downloaded unless it actually changed.
    """
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def resolve_ref(ref: str, token: str | None) -> str:
    """Turn a branch name or short sha into a full commit sha."""
    payload = json.loads(fetch(f"{API}/repos/{REPO}/commits/{ref}", token=token))
    return payload["sha"]


def remote_tree(commit: str, token: str | None) -> dict[str, str]:
    """Map filename -> blob sha for every tracked source file at this commit."""
    payload = json.loads(
        fetch(f"{API}/repos/{REPO}/git/trees/{commit}?recursive=1", token=token)
    )
    if payload.get("truncated"):
        raise SystemExit("GitHub truncated the tree listing; cannot diff reliably")
    return {
        entry["path"][len(SOURCE_PREFIX):]: entry["sha"]
        for entry in payload.get("tree", [])
        if entry["type"] == "blob"
        and entry["path"].startswith(SOURCE_PREFIX)
        and "/" not in entry["path"][len(SOURCE_PREFIX):]
    }


def slurm_jobs_active(user: str | None) -> int:
    """Count the caller's queued or running jobs. Zero if squeue is unavailable."""
    if shutil.which("squeue") is None:
        return 0
    command = ["squeue", "-h", "-o", "%i"]
    if user:
        command += ["-u", user]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def install(path: Path, data: bytes, commit: str, *, backup: bool) -> None:
    """Validate then atomically replace one file."""
    if path.suffix == ".py":
        with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as handle:
            handle.write(data)
            candidate = Path(handle.name)
        try:
            py_compile.compile(str(candidate), doraise=True)
        except py_compile.PyCompileError as error:
            candidate.unlink(missing_ok=True)
            raise SystemExit(f"Refusing to install {path.name}: it does not compile\n{error}")
        candidate.unlink(missing_ok=True)
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak-{commit[:8]}"))
    # Write beside the target then rename, so an interrupted sync cannot leave a
    # half-written stage script that a job would then try to run.
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_bytes(data)
    if path.exists():
        staging.chmod(path.stat().st_mode)
    staging.replace(path)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripts-dir", type=Path, required=True,
                        help="The flat directory the pipeline actually invokes, "
                             "e.g. ~/cry/project_2/scripts")
    parser.add_argument("--ref", default="main",
                        help="Branch, tag or commit to sync against (default: main). "
                             "Resolved to a commit sha before anything is compared.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write the changed files (default: report only)")
    parser.add_argument("--include-new", action="store_true",
                        help="Also install files that exist upstream but not locally")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Sync even while Slurm jobs are queued or running")
    parser.add_argument("--user", default=None, help="Slurm user for the active-job check")
    parser.add_argument("--token", default=None, help="GitHub token, if the repo is private")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Restrict to these filenames")
    opt = parser.parse_args()
    if not opt.scripts_dir.is_dir():
        raise SystemExit(f"{opt.scripts_dir} is not a directory")
    return opt


def main() -> None:
    opt = options()
    commit = resolve_ref(opt.ref, opt.token)
    tree = remote_tree(commit, opt.token)
    if opt.only:
        tree = {name: sha for name, sha in tree.items() if name in set(opt.only)}
        if not tree:
            raise SystemExit(f"None of {opt.only} exist under {SOURCE_PREFIX} at {commit[:8]}")

    changed, missing, same = [], [], 0
    for name, remote_sha in sorted(tree.items()):
        local = opt.scripts_dir / name
        if not local.exists():
            missing.append(name)
            continue
        if git_blob_sha(local.read_bytes()) == remote_sha:
            same += 1
        else:
            changed.append(name)

    untracked = sorted(
        path.name for path in opt.scripts_dir.iterdir()
        if path.is_file() and path.name not in tree
        and path.suffix in {".py", ".sh", ".slurm", ".yaml", ".yml"}
        and not path.name.endswith(".partial")
        and ".bak" not in path.name
    )

    print(f"ref        : {opt.ref} -> {commit}")
    print(f"scripts dir: {opt.scripts_dir}")
    print(f"up to date : {same}")
    print(f"changed    : {len(changed)}")
    for name in changed:
        print(f"    {name}")
    if missing:
        print(f"absent locally: {len(missing)}"
              f"{'  (use --include-new to install)' if not opt.include_new else ''}")
        for name in missing:
            print(f"    {name}")
    if untracked:
        # Not an error: the cluster legitimately holds scripts the repo does not,
        # such as submit_rex_replicates.sh. Listed so they are never a surprise.
        print(f"local only, left alone: {', '.join(untracked)}")

    to_install = list(changed) + (missing if opt.include_new else [])
    if not to_install:
        print("\nnothing to do")
        return
    if not opt.apply:
        print(f"\n{len(to_install)} file(s) would change. Re-run with --apply to write them.")
        return

    active = 0 if opt.force else slurm_jobs_active(opt.user)
    if active:
        raise SystemExit(
            f"{active} Slurm job(s) queued or running. Changing a stage script now would "
            "give different array tasks different code, and would invalidate downstream "
            "checkpoints via implementation_signature(). Wait for the queue to drain, or "
            "pass --force if you are certain."
        )

    for name in to_install:
        data = fetch(f"{RAW}/{REPO}/{commit}/{SOURCE_PREFIX}{name}", token=opt.token)
        if git_blob_sha(data) != tree[name]:
            raise SystemExit(f"Downloaded {name} does not match its expected hash")
        install(opt.scripts_dir / name, data, commit, backup=not opt.no_backup)
        print(f"  installed {name}")

    record = opt.scripts_dir / ".csbrt_sync.json"
    record.write_text(json.dumps(
        {"ref": opt.ref, "commit": commit, "installed": to_install}, indent=2) + "\n")
    print(f"\n{len(to_install)} file(s) installed at {commit[:8]}")
    print(f"SYNC_RECORD={record}")


if __name__ == "__main__":
    sys.exit(main())
