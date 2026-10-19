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
    RESOURCE_MANIFEST_SCHEMA,
    SCOPED_RESOURCE_MANIFEST_SCHEMA,
    ChainError,
    atomic_write_json,
    baseline_root,
    canonical_dataset_scope,
    config_with_auth,
    formal_resource_manifest_path,
    require_formal_session,
    resource_manifest_dataset_entry,
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
    return resource_manifest_dataset_entry(root, dataset, config_path, cache_entries, checkpoint_entries)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="authenticate and rewrite the local resource manifest")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    args = parser.parse_args(argv)
    root = baseline_root()
    try:
        dataset_scope = canonical_dataset_scope(args.datasets)
        quant = root / "baseline-local" / "configs" / "quantization_scale_chain.cfg"
        quant_entries = verify_sidecar(quant.parent, quant.name, quant.with_name(quant.name + ".sha256"))
        quant_sha256 = sha256_file(quant)
        if dataset_scope != DATASETS:
            session_id, session, session_manifest_sha256 = require_formal_session(root)
            if canonical_dataset_scope(session.get("dataset_scope")) != dataset_scope:
                raise ChainError("selected resources differ from the immutable formal session scope")
            manifest_path = formal_resource_manifest_path(root, session_id, session)
            if manifest_path.exists() or manifest_path.is_symlink():
                raise ChainError("scoped resource manifest already exists; refusing stale-result reuse")
            datasets = {dataset: inspect_dataset(root, dataset) for dataset in dataset_scope}
            payload = {
                "schema": SCOPED_RESOURCE_MANIFEST_SCHEMA,
                "status": "AUTHENTICATED",
                "baseline": "zkGPT",
                "resource_scope": "baseline-local",
                "setup_mode": "hyrax_native_local",
                "static_tables_source": "authenticated_quantization_scale_chain",
                "session_id": session_id,
                "session_manifest_sha256": session_manifest_sha256,
                "dataset_scope": list(dataset_scope),
                "quantization_scale_chain_sha256": quant_sha256,
                "quantization_entries": quant_entries,
                "datasets": datasets,
            }
            atomic_write_json(manifest_path, payload, root)
            print("AUTHENTICATED baseline-local resources for " + ",".join(dataset_scope) + " (session-scoped manifest)")
            return 0
        datasets: Dict[str, Any] = {}
        for dataset in dataset_scope:
            datasets[dataset] = inspect_dataset(root, dataset)
        payload = {
            "schema": RESOURCE_MANIFEST_SCHEMA,
            "status": "AUTHENTICATED",
            "baseline": "zkGPT",
            "resource_scope": "baseline-local",
            "setup_mode": "hyrax_native_local",
            "static_tables_source": "authenticated_quantization_scale_chain",
            "quantization_scale_chain_sha256": quant_sha256,
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
