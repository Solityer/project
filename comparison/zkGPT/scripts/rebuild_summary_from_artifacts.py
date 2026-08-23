#!/usr/bin/env python3
"""Compatibility wrapper for the fail-closed formal collector."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    raise SystemExit(subprocess.run([sys.executable, ROOT / "scripts" / "collect_results.py"], cwd=ROOT, check=False).returncode)
