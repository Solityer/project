#!/usr/bin/env python3
"""Create and verify the baseline-local zkGPT Python environment.

This helper uses only the standard library so it can be called by the public
entry point before importing NumPy.  It never falls back to the main project
environment or to another baseline's virtual environment.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / ".venv"
PYTHON = ENV_ROOT / "bin" / "python"
LOCK = ROOT / "requirements.lock"
EXPECTED_NUMPY = "2.4.4"


def _version(python: pathlib.Path) -> str | None:
    probe = subprocess.run(
        [str(python), "-c", "import numpy; print(numpy.__version__, end='')"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.stdout.strip() if probe.returncode == 0 else None


def ensure() -> pathlib.Path:
    if not LOCK.is_file():
        raise RuntimeError(f"baseline dependency lock is missing: {LOCK}")
    if not PYTHON.is_file() or not PYTHON.stat().st_mode & 0o111:
        subprocess.run([sys.executable, "-m", "venv", str(ENV_ROOT)], cwd=ROOT, check=True)
    if _version(PYTHON) != EXPECTED_NUMPY:
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "--requirement", str(LOCK)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            check=True,
        )
    if _version(PYTHON) != EXPECTED_NUMPY:
        raise RuntimeError("baseline-local NumPy version does not match requirements.lock")
    # A venv's ``bin/python`` commonly is a symlink to the system interpreter;
    # the authenticated entry point is the baseline-local venv path itself,
    # not the symlink target.  Never substitute that target in a command.
    lexical_python = pathlib.Path(PYTHON)
    if lexical_python.parent.parent != ENV_ROOT:
        raise RuntimeError(f"baseline Python path is not local to zkGPT: {lexical_python}")
    return lexical_python


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure baseline-local zkGPT dependencies")
    parser.add_argument("--print-python", action="store_true")
    args = parser.parse_args()
    python = ensure()
    if args.print_python:
        print(python)
    else:
        print(f"baseline-local Python ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
