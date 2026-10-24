#!/usr/bin/env python3
"""Run the production native integer-bound ledger parser on all five generator outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


def canonical_table_file_digest(table: dict) -> str:
    encoded = (json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()

    session = args.session.resolve()
    session.relative_to(ROOT / "runs")
    if args.session.is_symlink() or not session.is_dir():
        raise SystemExit("ledger-contract-preflight: session must be an existing baseline-local directory")
    native = ROOT / "gat_resource_setup"

    sys.path.insert(0, str(ROOT / "scripts"))
    producer = importlib.import_module("prepare_form_resources")
    quant_path = ROOT / "configs" / "quantization_scale_chain.cfg"
    native_candidates: list[tuple[str, dict, Path]] = []
    with tempfile.TemporaryDirectory(prefix=".ledger-contract-", dir=session) as temporary:
        temporary_dir = Path(temporary)
        for dataset in DATASETS:
            config = ROOT / "configs" / f"{session.name}_{dataset}.cfg"
            if config.is_symlink() or not config.is_file():
                raise SystemExit(f"ledger-contract-preflight: session snapshot config is missing: {dataset}")
            values = producer.parse_config(config)
            if values.get("dataset") != dataset:
                raise SystemExit(f"ledger-contract-preflight: snapshot config dataset mismatch: {dataset}")
            producer.validated_scale_chain(quant_path, values.get("quant_cfg_id", ""))
            quant_digest = producer.sha256_file(quant_path)
            _nodes, _edges, degree = producer.graph_degree(ROOT, dataset, values)
            table = producer.make_tables(values, quant_digest)
            table_file_digest = canonical_table_file_digest(table)
            tensors = producer.load_tensors(ROOT / values["checkpoint_bundle"] / "tensors.txt")
            model_max = max((abs(producer.round_away(value * (16 if name.endswith("output_bias") else 4)))
                             for name, _shape, data in tensors for value in data), default=1)
            ledger = producer.make_ledger(ROOT, values, table, table_file_digest, quant_digest, degree, model_max)
            expected = producer.ledger_digest(ledger)
            if ledger.get("digest") != expected:
                raise SystemExit(f"ledger-contract-preflight: Python producer returned an inconsistent digest: {dataset}")
            ledger_path = temporary_dir / f"{dataset}_gat_integer_ledger.json"
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            native_candidates.append((dataset, ledger, ledger_path))

        for dataset, ledger, ledger_path in native_candidates:
            result = subprocess.run([str(native), "verify-ledger", "--path", str(ledger_path), "--baseline-root", str(ROOT)],
                                    cwd=ROOT, text=True, capture_output=True, check=True)
            match = re.search(r"native_integer_bound_ledger=PASS digest=([0-9a-f]{64})", result.stdout)
            if match is None or match.group(1) != ledger["digest"]:
                raise SystemExit(f"ledger-contract-preflight: Python/native digest disagreement: {dataset}\n{result.stdout}{result.stderr}")
            print(f"ledger_contract_preflight=PASS dataset={dataset} digest={match.group(1)}", flush=True)

    print("ledger_contract_preflight=PASS datasets=5 parser=native digest=canonical-python", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
