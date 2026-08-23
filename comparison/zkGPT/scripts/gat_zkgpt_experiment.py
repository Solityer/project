#!/usr/bin/env python3
"""Official entry point; refuses the non-formal retained research graft."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run formal GAT-ZKML-zkGPT experiments")
    parser.add_argument("--project-root", default=str(ROOT.parents[1]))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--backends", default="all")
    parser.add_argument("--force", action="store_true")
    parser.parse_args()
    return subprocess.run(
        [sys.executable, ROOT / "scripts" / "formal_preflight.py"], cwd=ROOT, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
