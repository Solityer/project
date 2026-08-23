#!/usr/bin/env python3
"""Formal negative tests remain unavailable until a formal prover exists."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    raise SystemExit(subprocess.run([sys.executable, ROOT / "scripts" / "formal_preflight.py", "--source-only"], cwd=ROOT, check=False).returncode)
