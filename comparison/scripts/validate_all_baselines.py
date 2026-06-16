#!/usr/bin/env python3
"""Validate all independent comparison baselines against the common schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from comparison_schema import RESULTS_DIR, validate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "partial"], default="full")
    parser.add_argument(
        "--through",
        choices=["ezkl", "zkdl", "zkGPT"],
        help="Required last baseline in partial mode.",
    )
    parser.add_argument(
        "--audit-path",
        default=str(RESULTS_DIR / "final_comparison_audit.json"),
    )
    args = parser.parse_args()

    audit = validate_all(
        mode=args.mode,
        through=args.through,
        audit_path=Path(args.audit_path),
    )
    if audit["status"] == "passed":
        print(
            f"comparison_validation=OK mode={args.mode} "
            f"required_rows={audit['required_row_count']}"
        )
        return 0
    print(
        f"comparison_validation=FAILED mode={args.mode} "
        f"errors={len(audit['errors'])} audit={args.audit_path}"
    )
    for error in audit["errors"][:25]:
        print(f"- {error}")
    if len(audit["errors"]) > 25:
        print(f"- ... {len(audit['errors']) - 25} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
