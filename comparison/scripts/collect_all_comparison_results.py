#!/usr/bin/env python3
"""Collect paper-ready comparison tables after full validation passes."""

from __future__ import annotations

import argparse

from pathlib import Path

from comparison_schema import (
    RESULTS_DIR,
    paper_ready_rows,
    validate_all,
    write_csv,
    write_json,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-path",
        default=str(RESULTS_DIR / "final_comparison_audit.json"),
    )
    args = parser.parse_args()

    audit = validate_all(mode="full", audit_path=Path(args.audit_path))
    if audit["status"] != "passed":
        print(
            "comparison_collection=REFUSED "
            f"errors={len(audit['errors'])} audit={args.audit_path}"
        )
        return 1

    rows = paper_ready_rows(audit)
    write_json(
        RESULTS_DIR / "final_comparison_results.json",
        {
            "generated_at": audit["generated_at"],
            "git_head": audit["git_head"],
            "rows": rows,
        },
    )
    write_csv(RESULTS_DIR / "final_comparison_results.csv", rows)
    write_report(RESULTS_DIR / "final_comparison_report.md", rows, audit)
    print(
        "comparison_collection=OK "
        f"rows={len(rows)} results_dir={RESULTS_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
