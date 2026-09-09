#!/usr/bin/env python3
"""Formal single-dataset/backend and matrix runner for the zkGPT baseline."""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _lexical(path: pathlib.Path | str) -> pathlib.Path:
    value = pathlib.Path(path).expanduser()
    if not value.is_absolute():
        value = pathlib.Path.cwd() / value
    return pathlib.Path(os.path.abspath(os.path.normpath(os.fspath(value))))


def _strict_project_root(raw: pathlib.Path | str) -> pathlib.Path:
    """Accept only the canonical, real baseline root spelling."""
    supplied = pathlib.Path(raw).expanduser()
    if not supplied.is_absolute() or any(part in {".", ".."} for part in supplied.parts):
        raise ValueError("--project-root must be an absolute canonical path without aliases")
    lexical = _lexical(supplied)
    if lexical != ROOT:
        raise ValueError(f"--project-root must equal canonical zkGPT baseline root: {ROOT}")
    current = pathlib.Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"--project-root contains a symlink component: {current}")
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"--project-root must be a real directory: {lexical}")
    return lexical


def _bootstrap_local_python() -> None:
    """Ensure formal execution imports only the baseline-local environment."""
    helper = ROOT / "scripts" / "ensure_environment.py"
    completed = subprocess.run(
        [sys.executable, str(helper), "--print-python"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    local = pathlib.Path(os.path.abspath(completed.stdout.strip()))
    if local.parent.parent != ROOT / ".venv" or local != ROOT / ".venv" / "bin" / "python":
        raise SystemExit("ensure_environment returned a non-baseline-local interpreter")
    current = pathlib.Path(os.path.abspath(sys.executable))
    if current != local:
        os.execv(str(local), [str(local), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])


_bootstrap_local_python()
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.runner import run_many, selected_backends, selected_datasets  # noqa: E402
from zkgpt_gat.result_validation import ResultValidationError, resolve_results_dir  # noqa: E402


def _run_step(label: str, command: list[str]) -> None:
    print(f"[zkGPT] {label}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _prepare_formal_chain(datasets: list[str], backends: list[str], *, setup_only: bool) -> None:
    """Run the one authenticated build/setup/resource chain before proving."""
    python = str(pathlib.Path(sys.executable))
    selected = ",".join(datasets)
    # Build is first so a stale/failed native implementation cannot publish
    # model/Y/registry resources that would later be used by a run.
    for backend in backends:
        _run_step(
            f"build {backend}",
            [str(ROOT / "scripts" / "build.sh"), f"--{backend}"],
        )
    _run_step(
        "prepare baseline-local resource layout",
        [python, str(ROOT / "scripts" / "prepare_resources.py"), "--datasets", selected],
    )
    _run_step(
        "validate installed canonical checkpoints",
        [python, str(ROOT / "scripts" / "validate_canonical_checkpoints.py"), "--datasets", selected],
    )
    if not setup_only:
        _run_step(
            "generate canonical public Y with independent reference",
            [python, str(ROOT / "scripts" / "generate_public_y.py"), "--datasets", selected],
        )
    for dataset in datasets:
        _run_step(
            f"register canonical model {dataset}",
            [
                python,
                str(ROOT / "scripts" / "register_model.py"),
                "--dataset", dataset,
                "--project-root", str(ROOT),
                "--zk-root", str(ROOT),
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the native GAT-ZKML-zkGPT proof path")
    parser.add_argument("--project-root", default=str(ROOT), help="baseline root; external project roots are rejected")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--backends", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="register canonical model commitments without proving")
    args = parser.parse_args()
    try:
        project_root = _strict_project_root(args.project_root)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        results_dir = resolve_results_dir(ROOT, args.results_dir)
    except ResultValidationError as exc:
        parser.error(str(exc))
    datasets = selected_datasets(args.datasets)
    backends = selected_backends(args.backends)
    _prepare_formal_chain(datasets, backends, setup_only=args.setup_only)
    if args.setup_only:
        return 0
    run_many(
        datasets,
        backends,
        project_root,
        ROOT,
        results_dir,
        args.force,
    )
    selection = ",".join(datasets)
    backend_selection = ",".join(backends)
    python = str(pathlib.Path(sys.executable))
    _run_step(
        "collect verifier-backed results",
        [
            python,
            str(ROOT / "scripts" / "collect_results.py"),
            "--results-dir", str(results_dir),
            "--datasets", selection,
            "--backends", backend_selection,
        ],
    )
    _run_step(
        "validate collected results",
        [
            python,
            str(ROOT / "scripts" / "validate_final_results.py"),
            "--results-dir", str(results_dir),
            "--datasets", selection,
            "--backends", backend_selection,
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
