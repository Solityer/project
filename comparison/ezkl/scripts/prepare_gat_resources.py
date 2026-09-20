#!/usr/bin/env python3
"""Prepare and authenticate all baseline-local ezkl GAT resources.

The command is deliberately explicit: without ``--commit-manifest`` it only
prints the operator command and does not modify resources.  With the flag it
regenerates the deterministic exact tables/ledgers from this baseline's
canonical cache/checkpoint bundles, updates their local registry fields, and
then writes and verifies a SHA-256 resource manifest.  It never creates an
SRS; the formal runner stages trusted ceremony material through the native
``get-srs`` command in ``provision_srs.py`` after the CPU binary is built.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--commit-manifest",
        action="store_true",
        help="generate all five resources and atomically refresh resource_manifest.json",
    )
    result.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT,
        help="read-only provenance root used for manifest source metadata",
    )
    return result


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(args: argparse.Namespace) -> int:
    generator = ROOT / "scripts" / "generate_gat_resources.py"
    validator = ROOT / "scripts" / "validate_resources.py"
    if not args.commit_manifest:
        print("PREPARE/BLOCKED: exact five-dataset resource preparation was not run")
        print("rerun with: python3 scripts/prepare_gat_resources.py --commit-manifest")
        print("native official SRS is provisioned by run_cpu_formal.py")
        return 2

    for dataset in DATASETS:
        run([sys.executable, str(generator), "--root", str(ROOT), "--dataset", dataset])

    source_root = args.source_root.resolve()
    run(
        [
            sys.executable,
            str(validator),
            "--write-manifest",
            "--require-complete",
            "--source-root",
            str(source_root),
        ]
    )
    print("authenticated five-dataset ezkl GAT tables/ledgers and resource manifest")
    print("native official SRS will be provisioned by scripts/run_cpu_formal.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
