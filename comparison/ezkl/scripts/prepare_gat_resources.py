#!/usr/bin/env python3
"""Prepare and authenticate all baseline-local ezkl GAT resources.

The command is deliberately explicit: without ``--commit-manifest`` it only
prints the operator command and does not modify resources.  With the flag it
regenerates the deterministic exact tables/ledgers from this baseline's
canonical cache/checkpoint bundles, updates their local registry fields, and
then writes and verifies a SHA-256 resource manifest.  It never creates an
SRS; the formal entry authenticates trusted official ceremony material
through ``provision_srs.py`` using the pinned native ``get-srs`` source identity.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from process_guardian import check_guardian_health, finish_guarded_process, spawn_guarded_process, terminate_guarded_process_locally

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--commit-manifest",
        action="store_true",
        help="generate all five resources and atomically refresh resource_manifest.json",
    )
    result.add_argument(
        "--formal-auto",
        action="store_true",
        help="internal formal-runner mode: authenticate base inputs and repair generated tables/ledgers",
    )
    return result


def run(command: list[str], label: str, *, timeout_seconds: int = 21600) -> None:
    print(f"[ezkl-prepare] {label}: {' '.join(command)} (timeout={timeout_seconds}s)", flush=True)
    process = spawn_guarded_process(command, cwd=ROOT)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                returncode = process.wait(timeout=min(5.0, max(0.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                check_guardian_health()
    except subprocess.TimeoutExpired as exc:
        _stop_process(process)
        raise RuntimeError(f"{label} timed out after {timeout_seconds}s; child process group was terminated") from exc
    except KeyboardInterrupt:
        print(f"[ezkl-prepare] interrupted during {label}; terminating child process group", file=sys.stderr, flush=True)
        _stop_process(process)
        raise
    terminated_leftovers = finish_guarded_process(process)
    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")
    if terminated_leftovers:
        raise RuntimeError(f"{label} left descendants after its leader exited; process group was cleaned")


def _stop_process(process: subprocess.Popen[object]) -> None:
    if getattr(process, "_ezkl_process_group", None) is not None:
        terminate_guarded_process_locally(process)
        return
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _mutable_identity_files() -> list[Path]:
    paths: list[Path] = []
    for dataset in DATASETS:
        paths.append(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg")
        paths.append(ROOT / "resources/canonical/checkpoints" / f"{'ogbn_arxiv_gat' if dataset == 'ogbn_arxiv' else dataset + '_gat'}" / "manifest.json")
        if dataset in {"ppi", "ogbn_arxiv"}:
            paths.append(ROOT / "resources/canonical/cache" / dataset / "meta.cfg")
    return paths


def _transaction_paths() -> list[Path]:
    paths = _mutable_identity_files()
    for dataset in DATASETS:
        paths.extend((
            ROOT / "resources/canonical/tables" / dataset / "gat_static_tables.bin",
            ROOT / "resources/canonical/tables" / dataset / "gat_static_tables.json",
            ROOT / "resources/canonical/ledgers" / dataset / "gat_integer_ledger.json",
        ))
    paths.append(ROOT / "resources/resource_manifest.json")
    return paths


def _backup_transaction(paths: list[Path], backup_root: Path) -> tuple[dict[Path, Path | None], dict[Path, bool]]:
    """Copy generated outputs and identity docs to disk before a resource refresh."""
    files: dict[Path, Path | None] = {}
    directories: dict[Path, bool] = {}
    for path in paths:
        path = path.absolute()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise RuntimeError(f"resource transaction path escapes baseline: {path}") from exc
        parent = path.parent
        while parent != ROOT and parent.is_relative_to(ROOT):
            if parent.is_symlink():
                raise RuntimeError(f"resource transaction directory is a symlink: {parent}")
            directories.setdefault(parent, parent.is_dir())
            parent = parent.parent
        if path.is_symlink():
            raise RuntimeError(f"resource transaction file is a symlink: {path}")
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"resource transaction target is not a regular file: {path}")
            relative = path.relative_to(ROOT)
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            files[path] = backup
        else:
            files[path] = None
    return files, directories


def _restore_transaction(files: dict[Path, Path | None], directories: dict[Path, bool]) -> None:
    for path, backup in files.items():
        if backup is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.restore.", dir=path.parent)
        os.close(fd)
        try:
            shutil.copyfile(backup, temporary)
            shutil.copystat(backup, temporary)
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    for path, existed in sorted(directories.items(), key=lambda item: len(item[0].parts), reverse=True):
        if not existed:
            try:
                path.rmdir()
            except (FileNotFoundError, OSError):
                pass


def main(args: argparse.Namespace) -> int:
    if args.commit_manifest and args.formal_auto:
        raise SystemExit("choose either --commit-manifest or --formal-auto")
    generator = ROOT / "scripts" / "generate_gat_resources.py"
    validator = ROOT / "scripts" / "validate_resources.py"
    if not args.commit_manifest and not args.formal_auto:
        print("PREPARE/BLOCKED: exact five-dataset resource preparation was not run")
        print("rerun with: python3 scripts/prepare_gat_resources.py --commit-manifest")
        print("native official SRS is provisioned by python3 scripts/run_formal_matrix.py --execute-formal")
        return 2

    from validate_resources import (
        authenticate_preparation_inputs,
        preparation_resources_are_current,
        refresh_manifest_after_authenticated_derivation,
    )

    # Authenticate the old manifest before deciding whether outputs are stale.
    # A failed raw-input authentication is never converted into a regeneration.
    snapshot = authenticate_preparation_inputs(ROOT)
    if args.formal_auto and preparation_resources_are_current(snapshot):
        print("EZKL_RESOURCE_PREPARE_READY: all five datasets have authenticated complete resources")
        return 0

    with tempfile.TemporaryDirectory(prefix="ezkl-resource-rollback-") as temporary:
        backups, directories = _backup_transaction(_transaction_paths(), Path(temporary))
        try:
            for dataset in DATASETS:
                run([sys.executable, str(generator), "--root", str(ROOT), "--dataset", dataset], f"generate authenticated {dataset} exact tables/ledger")
            refresh_manifest_after_authenticated_derivation(snapshot)
        except BaseException as failure:
            try:
                _restore_transaction(backups, directories)
            except BaseException as rollback_failure:
                raise RuntimeError(
                    f"resource preparation failed ({failure}); rollback also failed ({rollback_failure})"
                ) from rollback_failure
            raise
    print("authenticated five-dataset ezkl GAT tables/ledgers and resource manifest")
    print("native official SRS will be provisioned by python3 scripts/run_formal_matrix.py --execute-formal")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (OSError, RuntimeError) as exc:
        print(f"formal resource preparation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
