#!/usr/bin/env python3
"""Compatibility wrapper for the authoritative fail-closed preflight."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    arguments = [sys.executable, ROOT / "scripts" / "formal_preflight.py"]
    if "--source-only" in sys.argv[1:]:
        arguments.append("--source-only")
    raise SystemExit(subprocess.run(arguments, cwd=ROOT, check=False).returncode)
