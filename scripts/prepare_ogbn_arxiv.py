#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

from ogbn_arxiv_utils import ensure_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    cache_path = ensure_cache(project_root, data_root)
    print(cache_path)


if __name__ == "__main__":
    main()
