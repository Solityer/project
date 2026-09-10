#!/usr/bin/env python3
"""Create a baseline-local identity manifest for one canonical dataset.

The manifest is deliberately content-addressed.  It never follows a path
outside the zkDL baseline root and it records every file in the copied cache,
checkpoint bundle, and local configuration.  The production verifier can
reject a run when this identity is absent or stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def fail(message: str) -> None:
    raise SystemExit(f"asset-manifest: {message}")


def local_path(root: Path, raw: str, field: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        fail(f"{field} must be relative to baseline root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        fail(f"{field} escapes baseline root")
    return resolved


def reject_source_path_values(values: dict[str, str]) -> None:
    """Reject absolute/traversal path references before materialization.

    Configs are copied artifacts, not a mechanism for reaching the main
    experiment.  Scan every value (not only the three paths used below) so a
    future source/data/checkpoint override cannot smuggle an external path
    into the manifest.
    """
    path_keys = ("path", "root", "dir", "file", "source", "data", "cache", "checkpoint", "export")
    for key, value in values.items():
        candidate = value.strip()
        if key == "dataset" or key.endswith("_id") or key in {"task_type", "report_unit", "round", "rescale", "output_average"}:
            continue
        looks_like_path = any(token in key.lower() for token in path_keys)
        if looks_like_path and Path(candidate).is_absolute():
            fail(f"absolute source path is forbidden in config key {key}")
        if looks_like_path and any(part == ".." for part in Path(candidate).parts):
            fail(f"parent traversal is forbidden in config key {key}")


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            fail(f"malformed config line {number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in values:
            fail(f"invalid or duplicate config key at line {number}")
        values[key] = value
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for directory in paths:
        if directory.is_symlink() or not directory.is_dir():
            fail(f"asset directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not path.is_file():
                fail(f"asset tree contains a symlink/non-file: {path}")
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    if config_path.parent.name != "configs":
        fail("config must live in baseline-local configs/")
    root = config_path.parent.parent.resolve()
    values = read_config(config_path)
    reject_source_path_values(values)
    for key in ("dataset", "cache_root", "checkpoint_bundle", "export_dir", "quant_cfg_id"):
        if key not in values:
            fail(f"config key missing: {key}")
    relation_keys = ("static_table_descriptor", "integer_bound_ledger")
    if values["dataset"] in {"cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"}:
        for key in relation_keys:
            if key not in values:
                fail(f"BLOCKED: config key missing: {key}")
    cache = local_path(root, values["cache_root"], "cache_root") / values["dataset"]
    checkpoint = local_path(root, values["checkpoint_bundle"], "checkpoint_bundle")
    quant = root / "configs" / "quantization_scale_chain.cfg"
    if quant.is_symlink() or not quant.is_file():
        fail("quantization_scale_chain.cfg is missing")
    selected = [config_path, quant]
    relation_digests: dict[str, str] = {}
    for key in relation_keys:
        if key not in values:
            continue
        relation_path = local_path(root, values[key], key)
        if relation_path.is_symlink() or not relation_path.is_file():
            fail(f"BLOCKED: authenticated relation resource is missing: {relation_path}")
        selected.append(relation_path)
        relation_digests[key] = sha256(relation_path)
    asset_entries = files_under((cache, checkpoint), root)
    for path in selected:
        if path.is_symlink():
            fail(f"configuration asset is a symlink: {path}")
        asset_entries.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    asset_entries.sort(key=lambda entry: str(entry["path"]))
    payload = {
        "schema": "zkDL_gat_asset_manifest_v1",
        "dataset": values["dataset"],
        "quant_cfg_id": values["quant_cfg_id"],
        "baseline_root": ".",
        "config": config_path.relative_to(root).as_posix(),
        "scale_chain_digest": sha256(quant),
        "files": asset_entries,
    }
    if relation_keys[0] in relation_digests:
        payload["static_table_descriptor"] = values[relation_keys[0]]
        payload["static_table_descriptor_digest"] = relation_digests[relation_keys[0]]
    if relation_keys[1] in relation_digests:
        payload["integer_bound_ledger"] = values[relation_keys[1]]
        payload["integer_bound_ledger_descriptor_digest"] = relation_digests[relation_keys[1]]
    output = args.output or (local_path(root, values["export_dir"], "export_dir") / "asset_manifest.json")
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        fail("output escapes baseline root")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"asset_manifest={output} files={len(asset_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
