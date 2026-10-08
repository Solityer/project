#!/usr/bin/env python3
"""Compatibility command redirected to the sole zkGPT formal entrypoint."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from run_formal import main as run_official


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv))
    return run_official([])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
