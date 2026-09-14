#!/usr/bin/env python3
"""Generate and verify the ezkl-local canonical resource manifest.

The manifest records both local copy identity and read-only provenance of the
source file.  Formal runners use only the local path, size and SHA-256 fields;
``source_path`` is metadata and is never opened unless ``--verify-source`` is
explicitly requested during staging/audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn


SCHEMA_VERSION = 1
MANIFEST_REL = Path("resources/resource_manifest.json")
CHECKPOINT_DATASETS = {
    "cora_gat": "cora",
    "citeseer_gat": "citeseer",
    "pubmed_gat": "pubmed",
    "ppi_gat": "ppi",
    "ogbn_arxiv_gat": "ogbn_arxiv",
}
REQUIRED_GENERATED_DATASETS = frozenset(CHECKPOINT_DATASETS.values())
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "baseline",
    "resource_root",
    "source_snapshot",
    "files",
}
REQUIRED_FILE_KEYS = {
    "path",
    "size",
    "sha256",
    "source_path",
    "source_sha256",
    "dataset",
    "role",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(message: str) -> "NoReturn":
    print(f"resource validation failed: {message}", file=sys.stderr)
    raise SystemExit(2)


def regular_files(root: Path) -> list[Path]:
    canonical = root / "resources" / "canonical"
    if not canonical.is_dir():
        fail(f"missing canonical resource directory: {canonical}")
    result: list[Path] = []
    for path in sorted(canonical.rglob("*")):
        if path.is_symlink():
            fail(f"symlink is not allowed in canonical resources: {path}")
        if path.is_file():
            lowered = "/".join(path.relative_to(root).parts).lower()
            forbidden = ("dummy", "gen-srs", "reduced-srs", "reduced_srs", "placeholder", "fake")
            if any(marker in lowered for marker in forbidden):
                fail(f"forbidden dummy/generated/reduced/placeholder resource: {path}")
            if "tests" in path.parts or "assets" in path.parts:
                fail(f"test-only resource cannot be a formal canonical resource: {path}")
            result.append(path)
    if not result:
        fail("canonical resource directory is empty")
    return result


def source_for(local_rel: Path, source_root: Path) -> tuple[Path, str, str]:
    parts = local_rel.parts
    if parts[:3] == ("resources", "canonical", "cache") and len(parts) >= 5:
        dataset = parts[3]
        source = source_root / "data" / "cache" / dataset / Path(*parts[4:])
        role = "canonical_dataset_cache"
    elif parts[:3] == ("resources", "canonical", "checkpoints") and len(parts) >= 5:
        bundle = parts[3]
        dataset = CHECKPOINT_DATASETS.get(bundle)
        if dataset is None:
            fail(f"unrecognized checkpoint bundle: {bundle}")
        source = source_root / "artifacts" / "checkpoints" / bundle / Path(*parts[4:])
        role = "canonical_checkpoint_bundle"
    elif parts == ("resources", "canonical", "configs", "quantization_scale_chain.cfg"):
        dataset = "all"
        source = source_root / "configs" / "quantization_scale_chain.cfg"
        role = "canonical_quantization_config"
    elif parts[:3] == ("resources", "canonical", "configs") and len(parts) == 4:
        name = parts[3]
        if not name.endswith("_secure.cfg"):
            fail(f"unexpected canonical config: {local_rel}")
        dataset = name.removesuffix("_secure.cfg")
        if dataset not in CHECKPOINT_DATASETS.values():
            fail(f"unrecognized secure config dataset: {dataset}")
        source = source_root / "configs" / name
        role = "canonical_secure_config"
    elif parts[:3] == ("resources", "canonical", "tables") and len(parts) == 5:
        dataset = parts[3]
        if dataset not in REQUIRED_GENERATED_DATASETS:
            fail(f"unrecognized generated table dataset: {dataset}")
        source = source_root / "comparison" / "ezkl" / "scripts" / "generate_gat_resources.py"
        role = "baseline_local_exact_static_tables"
    elif parts[:3] == ("resources", "canonical", "ledgers") and len(parts) == 5:
        dataset = parts[3]
        if dataset not in REQUIRED_GENERATED_DATASETS:
            fail(f"unrecognized generated ledger dataset: {dataset}")
        source = source_root / "comparison" / "ezkl" / "scripts" / "generate_gat_resources.py"
        role = "baseline_local_integer_bound_ledger"
    else:
        fail(f"unexpected canonical resource path: {local_rel}")
    return source, dataset, role


def source_snapshot(source_root: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"root": str(source_root.resolve())}
    try:
        snapshot["git_commit"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        snapshot["git_commit"] = "unknown"
    try:
        snapshot["git_tree"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD^{tree}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        snapshot["git_tree"] = "unknown"
    return snapshot


def build_manifest(root: Path, source_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for local in regular_files(root):
        local_rel = local.relative_to(root)
        source, dataset, role = source_for(local_rel, source_root)
        if not source.is_file():
            fail(f"source file for {local_rel} is missing: {source}")
        entries.append(
            {
                "path": local_rel.as_posix(),
                "size": local.stat().st_size,
                "sha256": digest(local),
                "source_path": str(source.resolve()),
                "source_sha256": digest(source),
                "dataset": dataset,
                "role": role,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": "ezkl",
        "resource_root": "resources/canonical",
        "source_snapshot": source_snapshot(source_root),
        "files": entries,
    }


def safe_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail(f"manifest path escapes baseline: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        fail(f"manifest path escapes baseline: {relative}")
    return resolved


def validate_manifest(root: Path, manifest: dict[str, Any], verify_source: bool) -> None:
    if set(manifest) != REQUIRED_TOP_LEVEL:
        fail("manifest top-level schema is not exact")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["baseline"] != "ezkl":
        fail("unsupported manifest version or baseline")
    if manifest["resource_root"] != "resources/canonical":
        fail("resource_root is not baseline-local")
    if not isinstance(manifest["source_snapshot"], dict):
        fail("source_snapshot must be an object")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        fail("files must contain at least one canonical entry")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != REQUIRED_FILE_KEYS:
            fail("file entry schema is not exact")
        path_text = entry["path"]
        if not isinstance(path_text, str) or path_text in seen:
            fail(f"duplicate or malformed manifest path: {path_text!r}")
        seen.add(path_text)
        local = safe_relative(root, path_text)
        if not local.is_file() or local.is_symlink():
            fail(f"manifest file is missing or symlink: {path_text}")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            fail(f"invalid size for {path_text}")
        if local.stat().st_size != entry["size"]:
            fail(f"size mismatch for {path_text}")
        for key in ("sha256", "source_sha256"):
            value = entry[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                fail(f"invalid {key} for {path_text}")
        if digest(local) != entry["sha256"]:
            fail(f"local SHA-256 mismatch for {path_text}")
        if not isinstance(entry["source_path"], str) or not os.path.isabs(entry["source_path"]):
            fail(f"source_path is not absolute metadata for {path_text}")
        if not isinstance(entry["dataset"], str) or not entry["dataset"]:
            fail(f"dataset metadata missing for {path_text}")
        if not isinstance(entry["role"], str) or not entry["role"]:
            fail(f"role metadata missing for {path_text}")
        if "qat_best_logits" in path_text:
            fail("qat_best_logits is not a formal resource")
        if path_text.endswith("/manifest.json"):
            try:
                checkpoint_manifest = json.loads(local.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"invalid checkpoint manifest {path_text}: {exc}")
            prefix = checkpoint_manifest.get("checkpoint_prefix")
            if prefix != "best_model.pt":
                fail(f"checkpoint_prefix is not bundle-local in {path_text}")
        if verify_source:
            source = Path(entry["source_path"])
            if not source.is_file():
                fail(f"source file is missing for {path_text}: {source}")
            if digest(source) != entry["source_sha256"]:
                fail(f"source SHA-256 mismatch for {path_text}: {source}")
    expected = {p.relative_to(root).as_posix() for p in regular_files(root)}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        fail(f"manifest file set mismatch; missing={missing}, extra={extra}")


def require_complete_resource_set(root: Path, manifest: dict[str, Any]) -> None:
    """Require all five datasets to have authenticated exact resources.

    The ordinary manifest check intentionally permits a staged package while
    the operator is preparing large exact tables.  Formal execution calls
    this stricter gate so an absent table/ledger can never become a runtime
    fallback or an implicitly trusted sidecar.
    """

    paths = {
        entry["path"]
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing = []
    for dataset in sorted(REQUIRED_GENERATED_DATASETS):
        for relative in (
            f"resources/canonical/tables/{dataset}/gat_static_tables.json",
            f"resources/canonical/ledgers/{dataset}/gat_integer_ledger.json",
        ):
            if relative not in paths:
                missing.append(relative)
    if missing:
        fail(
            "PREPARE/BLOCKED: authenticated exact table/ledger resources are missing; "
            "run scripts/prepare_gat_resources.py first: "
            + ", ".join(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require authenticated exact resources for all five datasets",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / MANIFEST_REL
    if args.write_manifest:
        manifest = build_manifest(root, args.source_root.resolve())
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse manifest: {exc}")
    validate_manifest(root, manifest, args.verify_source)
    if args.require_complete:
        require_complete_resource_set(root, manifest)
    print(f"resource manifest valid: {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
