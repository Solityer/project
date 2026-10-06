#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("zkdl_prepare_form_resources", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native ledger producer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--ledger-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    ledger_paths = [args.ledger_dir / f"{dataset}_gat_integer_ledger.json" for dataset in DATASETS]
    if args.native.is_symlink() or not args.native.is_file():
        raise SystemExit("native integer-bound ledger contract executable is missing or symlinked")
    if any(path.is_symlink() or not path.is_file() for path in ledger_paths):
        raise SystemExit("one or more canonical integer-bound ledger files are missing or symlinked")

    completed = subprocess.run([str(args.native.resolve()), *(str(path.resolve()) for path in ledger_paths)], cwd=root,
                               text=True, capture_output=True, check=True)
    native_digests = completed.stdout.splitlines()
    if len(native_digests) != len(DATASETS):
        raise SystemExit("native ledger contract returned the wrong number of digests")
    producer = load_generator(root / "scripts" / "prepare_form_resources.py")
    for dataset, path, native_digest in zip(DATASETS, ledger_paths, native_digests):
        ledger = json.loads(path.read_text(encoding="utf-8"))
        python_digest = producer.ledger_digest(ledger)
        if native_digest != python_digest or ledger.get("digest") != python_digest:
            raise SystemExit(f"Python/native authenticated ledger digest mismatch: {dataset}")
        print(f"integer_bound_ledger_contract=PASS dataset={dataset} digest={native_digest}")
    print("integer_bound_ledger_contract=PASS datasets=5 parser=production-native digest=production-python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
