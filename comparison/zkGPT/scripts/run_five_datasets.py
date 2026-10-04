#!/usr/bin/env python3
"""Run the fixed five-dataset formal path inside an active official session."""

from __future__ import annotations

import argparse
import sys
from typing import List

from common import ChainError, baseline_root, require_formal_session
from gat_zkgpt_experiment import main as run_formal_cells


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        require_formal_session(baseline_root())
    except ChainError as exc:
        print(f"EXPERIMENT_BLOCKED: {exc}", file=sys.stderr)
        return 2
    return run_formal_cells([])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
