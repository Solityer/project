#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import collect_gat_ezkl_results


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def main() -> int:
    rc = collect_gat_ezkl_results.main()
    copy_if_exists(RESULTS / "gat_ezkl_full_results.json", RESULTS / "final_ezkl_results.json")
    copy_if_exists(RESULTS / "gat_ezkl_full_results.csv", RESULTS / "final_ezkl_results.csv")
    copy_if_exists(RESULTS / "summary.md", RESULTS / "final_ezkl_report.md")
    return rc


if __name__ == "__main__":
    sys.exit(main())
