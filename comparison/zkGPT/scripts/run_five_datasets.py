#!/usr/bin/env python3
"""唯一的五数据集正式入口; execution remains in gat_zkgpt_experiment.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from common import BACKENDS, baseline_root


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("all",) + BACKENDS, default="cpu",
        help="formal backend (default: cpu; gpu is fail-closed unsupported)",
    )
    args = parser.parse_args(argv)
    root = baseline_root()
    command = [sys.executable, str(root / "scripts" / "gat_zkgpt_experiment.py"),
               "--datasets", "all", "--backends", args.backend]
    completed = subprocess.run(command, cwd=str(root), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
