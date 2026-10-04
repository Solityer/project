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
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

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
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=(os.name == "posix"))
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _stop_process(process)
        raise RuntimeError(f"{label} timed out after {timeout_seconds}s; child process group was terminated") from exc
    except KeyboardInterrupt:
        print(f"[ezkl-prepare] interrupted during {label}; terminating child process group", file=sys.stderr, flush=True)
        _stop_process(process)
        raise
    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")


def _stop_process(process: subprocess.Popen[object]) -> None:
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


def _input_resource_paths() -> list[Path]:
    manifest_path = ROOT / "resources" / "resource_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot snapshot canonical inputs before preparation: {exc}") from exc
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("canonical resource manifest has no input entries to snapshot")
    controlled_outputs = {path.resolve() for path in _mutable_identity_files()}
    paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError("canonical resource manifest has a malformed input entry")
        relative = entry["path"]
        if relative.startswith("resources/canonical/tables/") or relative.startswith("resources/canonical/ledgers/"):
            continue
        local = Path(relative)
        local_path = local if local.is_absolute() else ROOT / local
        if local_path.resolve() in controlled_outputs:
            continue
        for key in ("path", "source_path"):
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"canonical resource manifest input entry lacks {key}")
            candidate = Path(value)
            paths.append(candidate if candidate.is_absolute() else ROOT / candidate)
    return paths


def _snapshot_files(paths: list[Path]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths}


def _restore_files(snapshot: dict[Path, bytes]) -> None:
    for path, payload in snapshot.items():
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.restore.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def main(args: argparse.Namespace) -> int:
    if args.commit_manifest and args.formal_auto:
        raise SystemExit("choose either --commit-manifest or --formal-auto")
    generator = ROOT / "scripts" / "generate_gat_resources.py"
    validator = ROOT / "scripts" / "validate_resources.py"
    if not args.commit_manifest and not args.formal_auto:
        print("PREPARE/BLOCKED: exact five-dataset resource preparation was not run")
        print("rerun with: python3 scripts/prepare_gat_resources.py --commit-manifest")
        print("native official SRS is provisioned by run_formal_matrix.py --execute-formal")
        return 2

    from formal_common import resource_file_identities_for_paths

    authenticated_inputs = _input_resource_paths()
    input_identities_before = resource_file_identities_for_paths(authenticated_inputs)
    run([sys.executable, str(validator), "--validate-inputs-only"], "authenticate baseline canonical inputs")
    if resource_file_identities_for_paths(authenticated_inputs) != input_identities_before:
        raise RuntimeError("canonical input files changed during their authenticated preflight")
    backups = _snapshot_files(_mutable_identity_files() + [ROOT / "resources/resource_manifest.json"])
    try:
        for dataset in DATASETS:
            run([sys.executable, str(generator), "--root", str(ROOT), "--dataset", dataset], f"generate authenticated {dataset} exact tables/ledger")

        run([
            sys.executable,
            str(validator),
            "--write-manifest",
            "--require-complete",
        ], "refresh and authenticate canonical resource manifest")
        if resource_file_identities_for_paths(authenticated_inputs) != input_identities_before:
            raise RuntimeError("canonical input files changed during official resource preparation/authentication")
    except BaseException:
        _restore_files(backups)
        raise
    print("authenticated five-dataset ezkl GAT tables/ledgers and resource manifest")
    print("native official SRS will be provisioned by scripts/run_formal_matrix.py --execute-formal")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (OSError, RuntimeError) as exc:
        print(f"formal resource preparation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
