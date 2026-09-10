#!/usr/bin/env python3
"""Authenticate the baseline-local zkGPT inputs before a formal run.

This command deliberately does not copy, download, or discover resources
outside comparison/zkGPT.  A missing or modified sidecar is an error; the
command never repairs a digest from the payload it is supposed to trust.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import (
    BACKENDS,
    DATASETS,
    ChainError,
    atomic_write_json,
    baseline_root,
    config_with_auth,
    sha256_file,
    verify_cache,
    verify_checkpoint,
    verify_sidecar,
)


def inspect_dataset(root: Path, dataset: str) -> Dict[str, Any]:
    config_path, config = config_with_auth(root, dataset)
    if config.get("dataset") != dataset:
        raise ChainError(f"config dataset identity mismatch: {dataset}")
    if config.get("setup_mode") != "hyrax_native_local" or config.get("setup_resource_scope") != "baseline-local":
        raise ChainError(f"{dataset} does not declare baseline-local Hyrax setup")
    if "srs_path" in config or "srs_id" in config or "srs_type" in config:
        raise ChainError(f"{dataset} still declares an external SRS dependency")
    cache_entries = verify_cache(root, dataset)
    checkpoint_entries = verify_checkpoint(root, dataset)
    cache_dir = root / "baseline-local" / "data" / "cache" / dataset
    checkpoint_dir = root / "baseline-local" / "checkpoints" / f"{dataset}_gat"
    return {
        "config_sha256": sha256_file(config_path),
        "cache_sidecar_sha256": sha256_file(cache_dir / "integrity.sha256"),
        "cache_entries": cache_entries,
        "checkpoint_sidecar_sha256": sha256_file(checkpoint_dir / "integrity.sha256"),
        "checkpoint_entries": checkpoint_entries,
        # This is the digest of the canonical manifest itself, not a digest
        # inferred from model tensor contents at runtime.
        "checkpoint_manifest_sha256": checkpoint_entries["manifest.json"],
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="authenticate and rewrite the local resource manifest")
    args = parser.parse_args(argv)
    del args
    root = baseline_root()
    try:
        datasets: Dict[str, Any] = {}
        for dataset in DATASETS:
            datasets[dataset] = inspect_dataset(root, dataset)
        quant = root / "baseline-local" / "configs" / "quantization_scale_chain.cfg"
        quant_entries = verify_sidecar(quant.parent, quant.name, quant.with_name(quant.name + ".sha256"))
        payload = {
            "schema": "zkgpt-gat-resource-manifest-v1",
            "status": "AUTHENTICATED",
            "baseline": "zkGPT",
            "resource_scope": "baseline-local",
            "setup_mode": "hyrax_native_local",
            "static_tables_source": "authenticated_quantization_scale_chain",
            "quantization_scale_chain_sha256": sha256_file(quant),
            "quantization_entries": quant_entries,
            "datasets": datasets,
        }
        atomic_write_json(root / "results" / "resource_manifest.json", payload)
    except ChainError as exc:
        print(f"RESOURCE_CHECK_FAILED: {exc}", file=sys.stderr)
        return 2
    print("AUTHENTICATED baseline-local resources for " + ",".join(DATASETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
