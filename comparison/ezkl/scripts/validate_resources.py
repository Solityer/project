#!/usr/bin/env python3
"""Generate and verify the ezkl-local canonical resource manifest.

The manifest records local copy identity and baseline-local provenance. Formal
runners use only the local path, size and SHA-256 fields; ``source_path`` is
metadata and is never opened unless ``--verify-source`` is explicitly
requested during staging/audit.  A source entry outside this checkout is
rejected so formal execution cannot silently fall back to the main experiment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, NoReturn

SCHEMA_VERSION = 1
TABLE_MAGIC = b"EZKL-GAT-TABLES\x00"
TABLE_FORMAT_VERSION = 1
TABLE_METADATA_LIMIT = 16 * 1024 * 1024
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
            if path.name == "gat_static_tables.json":
                fail(f"legacy monolithic lookup table is forbidden; use {path.with_suffix('.bin')}")
            forbidden = ("dummy", "gen-srs", "reduced-srs", "reduced_srs", "placeholder", "fake")
            if any(marker in lowered for marker in forbidden):
                fail(f"forbidden dummy/generated/reduced/placeholder resource: {path}")
            if "tests" in path.parts or "assets" in path.parts:
                fail(f"test-only resource cannot be a formal canonical resource: {path}")
            result.append(path)
    if not result:
        fail("canonical resource directory is empty")
    return result


def source_for(local_rel: Path, baseline_root: Path) -> tuple[Path, str, str]:
    """Map a canonical file to a source copy inside this baseline only."""
    parts = local_rel.parts
    if parts[:3] == ("resources", "canonical", "cache") and len(parts) >= 5:
        dataset = parts[3]
        source = baseline_root / "resources" / "canonical" / "cache" / dataset / Path(*parts[4:])
        role = "canonical_dataset_cache"
    elif parts[:3] == ("resources", "canonical", "checkpoints") and len(parts) >= 5:
        bundle = parts[3]
        dataset = CHECKPOINT_DATASETS.get(bundle)
        if dataset is None:
            fail(f"unrecognized checkpoint bundle: {bundle}")
        source = baseline_root / "resources" / "canonical" / "checkpoints" / bundle / Path(*parts[4:])
        role = "canonical_checkpoint_bundle"
    elif parts == ("resources", "canonical", "configs", "quantization_scale_chain.cfg"):
        dataset = "all"
        source = baseline_root / "resources" / "canonical" / "configs" / "quantization_scale_chain.cfg"
        role = "canonical_quantization_config"
    elif parts[:3] == ("resources", "canonical", "configs") and len(parts) == 4:
        name = parts[3]
        if not name.endswith("_secure.cfg"):
            fail(f"unexpected canonical config: {local_rel}")
        dataset = name.removesuffix("_secure.cfg")
        if dataset not in CHECKPOINT_DATASETS.values():
            fail(f"unrecognized secure config dataset: {dataset}")
        source = baseline_root / "resources" / "canonical" / "configs" / name
        role = "canonical_secure_config"
    elif parts[:3] == ("resources", "canonical", "tables") and len(parts) == 5:
        dataset = parts[3]
        if dataset not in REQUIRED_GENERATED_DATASETS:
            fail(f"unrecognized generated table dataset: {dataset}")
        source = baseline_root / "scripts" / "generate_gat_resources.py"
        role = "baseline_local_exact_static_tables"
    elif parts[:3] == ("resources", "canonical", "ledgers") and len(parts) == 5:
        dataset = parts[3]
        if dataset not in REQUIRED_GENERATED_DATASETS:
            fail(f"unrecognized generated ledger dataset: {dataset}")
        source = baseline_root / "scripts" / "generate_gat_resources.py"
        role = "baseline_local_integer_bound_ledger"
    else:
        fail(f"unexpected canonical resource path: {local_rel}")
    return source, dataset, role


def canonical_archive_key(logical_name: str) -> str:
    """Return the baseline-local NPZ key for a logical tensor name.

    The Rust loader intentionally keeps slash-separated logical names in the
    checkpoint manifest and applies this exact encoding only at the NPZ
    boundary.  Keep the Python resource gate aligned with that contract so a
    stale underscore/slash alias cannot silently select a different tensor.
    """

    if not logical_name or "\x00" in logical_name or "\\" in logical_name:
        fail(f"invalid logical tensor name: {logical_name!r}")
    encoded = logical_name.replace("/", "__").replace(":", "__")
    if not encoded or "/" in encoded or ":" in encoded:
        fail(f"logical tensor name is not encodable: {logical_name!r}")
    return encoded


def validate_checkpoint_archive(root: Path, manifest_path: Path, checkpoint: dict[str, Any]) -> None:
    """Check manifest logical names, archive keys, metadata and NPZ members."""

    tensor_index = checkpoint.get("tensor_index")
    if not isinstance(tensor_index, dict) or not tensor_index:
        fail(f"checkpoint tensor_index is missing or empty: {manifest_path}")
    expected: dict[str, dict[str, Any]] = {}
    for logical_name, metadata in tensor_index.items():
        if not isinstance(logical_name, str) or not isinstance(metadata, dict):
            fail(f"invalid tensor_index metadata in {manifest_path}")
        required = {"archive_key", "dtype", "shape"}
        if set(metadata) != required:
            fail(f"tensor metadata schema mismatch for {logical_name} in {manifest_path}")
        archive_key = metadata["archive_key"]
        if not isinstance(archive_key, str) or archive_key != canonical_archive_key(logical_name):
            fail(
                f"tensor archive key mismatch for {logical_name} in {manifest_path}: "
                f"expected {canonical_archive_key(logical_name)!r}, got {archive_key!r}"
            )
        if archive_key in expected:
            fail(f"tensor archive-key collision in {manifest_path}: {archive_key}")
        if not isinstance(metadata["dtype"], str) or not metadata["dtype"]:
            fail(f"tensor dtype is missing for {logical_name} in {manifest_path}")
        shape = metadata["shape"]
        if (
            not isinstance(shape, list) or not shape or
            any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 for dimension in shape)
        ):
            fail(f"tensor shape is invalid for {logical_name} in {manifest_path}")
        expected[archive_key] = metadata

    if checkpoint.get("tensor_count") != len(expected):
        fail(f"tensor_count mismatch in {manifest_path}: "
             f"declared {checkpoint.get('tensor_count')}, indexed {len(expected)}")

    archive_path = manifest_path.parent / "tensors.npz"
    if not archive_path.is_file() or archive_path.is_symlink():
        fail(f"checkpoint tensor archive is missing or symlinked: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
    except (OSError, zipfile.BadZipFile) as exc:
        fail(f"invalid checkpoint tensor archive {archive_path}: {exc}")
    actual = set()
    for member in members:
        if not member.endswith(".npy") or "/" in member[:-4]:
            fail(f"unexpected NPZ member in {archive_path}: {member}")
        key = member[:-4]
        if key in actual:
            fail(f"duplicate NPZ member in {archive_path}: {member}")
        actual.add(key)
    expected_keys = set(expected)
    if actual != expected_keys:
        fail(f"checkpoint archive/manifest mismatch in {archive_path}: "
             f"missing={sorted(expected_keys - actual)}, extra={sorted(actual - expected_keys)}")


def _npy_contract(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(10)
        if len(prefix) < 10 or prefix[:6] != b"\x93NUMPY":
            fail(f"invalid NPY magic: {path}")
        major = prefix[6]
        if major == 1:
            header_len = int.from_bytes(prefix[8:10], "little")
            data_start = 10
        elif major == 2:
            version_tail = stream.read(2)
            if len(version_tail) != 2:
                fail(f"truncated NPY header: {path}")
            header_len = int.from_bytes(prefix[8:10] + version_tail, "little")
            data_start = 12
        else:
            fail(f"unsupported NPY version {major}: {path}")
        if header_len > TABLE_METADATA_LIMIT:
            fail(f"NPY header is too large: {path}")
        header_bytes = stream.read(header_len)
        if len(header_bytes) != header_len:
            fail(f"truncated NPY header body: {path}")
    try:
        header = ast.literal_eval(header_bytes.decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        fail(f"invalid NPY header {path}: {exc}")
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        fail(f"unsupported NPY header schema: {path}")
    descr = header["descr"]
    dtype = {"<f4": "float32", "<f8": "float64"}.get(descr)
    if dtype is None or header["fortran_order"] is not False:
        fail(f"NPY dtype/order is not canonical: {path}")
    shape = header["shape"]
    if not isinstance(shape, tuple) or not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
        fail(f"invalid NPY shape: {path}")
    item_size = 4 if dtype == "float32" else 8
    length = 1
    for dimension in shape:
        length *= dimension
    data_offset = data_start + header_len
    expected_size = data_offset + length * item_size
    if path.stat().st_size != expected_size:
        fail(f"NPY payload length mismatch: {path}")
    return {
        "dtype": dtype,
        "shape": list(shape),
        "order": "C",
        "length": length,
        "sha256": digest(path),
    }


def _parse_cfg(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            fail(f"malformed config line {path}:{number}")
        key, value = (item.strip() for item in line.split("=", 1))
        if not key or key in values:
            fail(f"duplicate/empty config key {path}:{number}")
        values[key] = value
    return values


def validate_npy_contract(root: Path, feature_path: Path) -> None:
    contract = _npy_contract(feature_path)
    meta_path = feature_path.with_name("meta.cfg")
    if not meta_path.is_file() or meta_path.is_symlink():
        fail(f"NPY feature metadata is missing: {meta_path}")
    meta = _parse_cfg(meta_path)
    expected = {
        "feature_format": "npy",
        "feature_dtype": contract["dtype"],
        "feature_shape": ",".join(str(x) for x in contract["shape"]),
        "feature_order": "C",
        "feature_length": str(contract["length"]),
        "feature_sha256": contract["sha256"],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            fail(f"NPY metadata mismatch for {feature_path}: {key}")


def _hash_string(hasher: Any, value: str) -> None:
    encoded = value.encode()
    hasher.update(struct.pack("<Q", len(encoded)))
    hasher.update(encoded)


def _hash_i128(hasher: Any, value: int) -> None:
    try:
        hasher.update(value.to_bytes(16, "little", signed=True))
    except OverflowError as exc:
        fail(f"lookup row is outside signed i128: {value} ({exc})")


def _read_exact(stream: Any, length: int, label: str, path: Path) -> bytes:
    data = stream.read(length)
    if len(data) != length:
        fail(f"truncated {label} in {path}")
    return data


def _read_u64(stream: Any, label: str, path: Path) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8, label, path))[0]


def _read_metadata(stream: Any, label: str, path: Path) -> dict[str, Any]:
    length = _read_u64(stream, f"{label} length", path)
    if length > TABLE_METADATA_LIMIT:
        fail(f"{label} is too large in {path}")
    try:
        value = json.loads(_read_exact(stream, length, label, path))
    except (TypeError, json.JSONDecodeError) as exc:
        fail(f"invalid {label} JSON in {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} is not an object in {path}")
    return value


def _table_canonical_length(metadata: dict[str, Any]) -> int:
    try:
        row_count = int(metadata["row_count"])
        lengths = [len(str(metadata[key]).encode()) for key in ("table_id", "function_id", "domain_id", "generator_id")]
        return len(b"ezkl-gat-exact-table-v1") + sum(8 + item for item in lengths) + 4 + 4 + 16 + 16 + row_count * 32
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        fail(f"invalid static-table metadata: {exc}")


def validate_table_archive(path: Path) -> None:
    """Validate the versioned binary registry without materializing its rows."""
    with path.open("rb") as stream:
        if _read_exact(stream, len(TABLE_MAGIC), "table archive magic", path) != TABLE_MAGIC:
            fail(f"invalid table archive magic: {path}")
        version = struct.unpack("<I", _read_exact(stream, 4, "table archive version", path))[0]
        count = struct.unpack("<I", _read_exact(stream, 4, "table archive count", path))[0]
        if version != TABLE_FORMAT_VERSION or count != 4:
            fail(f"unsupported table archive version/count in {path}")
        registry = _read_metadata(stream, "registry metadata", path)
        if set(registry) != {"format", "version", "static_tables_id", "backend_aux_tables_id", "static_tables_digest", "table_order"}:
            fail(f"registry metadata schema mismatch: {path}")
        if registry["format"] != "ezkl-gat-static-tables-bin" or registry["version"] != TABLE_FORMAT_VERSION:
            fail(f"registry metadata identity mismatch: {path}")
        if (
            not isinstance(registry["static_tables_id"], str) or not isinstance(registry["backend_aux_tables_id"], str) or
            not isinstance(registry["table_order"], list)
        ):
            fail(f"registry metadata types are invalid: {path}")
        table_order = ["lrelu", "exp", "elu", "max_relu"]
        if registry["table_order"] != table_order:
            fail(f"registry table order mismatch: {path}")
        if (
            not isinstance(registry["static_tables_digest"], str) or len(registry["static_tables_digest"]) != 64 or
            any(char not in "0123456789abcdef" for char in registry["static_tables_digest"])
        ):
            fail(f"registry digest is malformed: {path}")
        static_hasher = hashlib.sha256(b"ezkl-gat-static-tables-v1")
        _hash_string(static_hasher, registry["static_tables_id"])
        _hash_string(static_hasher, registry["backend_aux_tables_id"])
        functions = ["lrelu", "exp_neg", "elu", "max_relu"]
        for expected_function in functions:
            metadata = _read_metadata(stream, "table metadata", path)
            required = {
                "table_id", "input_scale", "output_scale", "function_id", "domain_id", "generator_id", "input_min", "input_max", "row_count", "digest",
                "canonical_len"
            }
            if set(metadata) != required:
                fail(f"table metadata schema mismatch: {path}")
            if any(not isinstance(metadata[key], str) for key in ("table_id", "function_id", "domain_id", "generator_id", "digest")) or any(
                isinstance(metadata[key], bool) or not isinstance(metadata[key], int) for key in (
                    "input_scale",
                    "output_scale",
                    "input_min",
                    "input_max",
                    "row_count",
                    "canonical_len",
                )
            ):
                fail(f"table metadata types are invalid: {path}")
            if not -(1 << 31) <= metadata["input_scale"] <= (1 << 31) - 1 or not -(1 << 31) <= metadata["output_scale"] <= (1 << 31) - 1:
                fail(f"table scale is outside signed 32-bit range: {path}")
            row_count = _read_u64(stream, "table row count", path)
            if row_count != metadata["row_count"]:
                fail(f"table row count mismatch: {path}")
            input_min = metadata["input_min"]
            input_max = metadata["input_max"]
            if not isinstance(input_min, int) or not isinstance(input_max, int) or input_max < input_min:
                fail(f"table domain is invalid: {path}")
            if row_count != input_max - input_min + 1:
                fail(f"table domain is not contiguous: {path}")
            if metadata["function_id"] != expected_function:
                fail(f"table function identity mismatch: {path}")
            if (not isinstance(metadata["digest"], str) or len(metadata["digest"]) != 64 or any(char not in "0123456789abcdef" for char in metadata["digest"])):
                fail(f"table digest is malformed: {path}")
            canonical_length = _table_canonical_length(metadata)
            if metadata["canonical_len"] != canonical_length:
                fail(f"table canonical length mismatch: {path}")
            table_hasher = hashlib.sha256(b"ezkl-gat-exact-table-v1")
            _hash_string(table_hasher, metadata["table_id"])
            table_hasher.update(struct.pack("<i", metadata["input_scale"]))
            table_hasher.update(struct.pack("<i", metadata["output_scale"]))
            _hash_string(table_hasher, metadata["function_id"])
            _hash_string(table_hasher, metadata["domain_id"])
            _hash_string(table_hasher, metadata["generator_id"])
            _hash_i128(table_hasher, input_min)
            _hash_i128(table_hasher, input_max)
            static_hasher.update(struct.pack("<Q", canonical_length))
            static_hasher.update(b"ezkl-gat-exact-table-v1")
            _hash_string(static_hasher, metadata["table_id"])
            static_hasher.update(struct.pack("<i", metadata["input_scale"]))
            static_hasher.update(struct.pack("<i", metadata["output_scale"]))
            _hash_string(static_hasher, metadata["function_id"])
            _hash_string(static_hasher, metadata["domain_id"])
            _hash_string(static_hasher, metadata["generator_id"])
            _hash_i128(static_hasher, input_min)
            _hash_i128(static_hasher, input_max)
            for expected_input in range(input_min, input_max + 1):
                row = _read_exact(stream, 32, "table row", path)
                actual_input = int.from_bytes(row[:16], "little", signed=True)
                if actual_input != expected_input:
                    fail(f"table row input is not contiguous: {path}")
                table_hasher.update(row)
                static_hasher.update(row)
            table_digest = table_hasher.hexdigest()
            if metadata["digest"] != table_digest:
                fail(f"table row digest mismatch: {path}")
            _hash_string(static_hasher, metadata["digest"])
        if static_hasher.hexdigest() != registry["static_tables_digest"]:
            fail(f"static-table registry digest mismatch: {path}")
        if stream.read(1):
            fail(f"static-table archive has trailing bytes: {path}")


def source_snapshot(baseline_root: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"root": str(baseline_root.resolve())}
    try:
        snapshot["git_commit"] = subprocess.check_output(
            ["git", "-C", str(baseline_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        snapshot["git_commit"] = "unknown"
    try:
        snapshot["git_tree"] = subprocess.check_output(
            ["git", "-C", str(baseline_root), "rev-parse", "HEAD^{tree}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        snapshot["git_tree"] = "unknown"
    return snapshot


def build_manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for local in regular_files(root):
        local_rel = local.relative_to(root)
        source, dataset, role = source_for(local_rel, root)
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
        "source_snapshot": source_snapshot(root),
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
    snapshot_root = manifest["source_snapshot"].get("root")
    if not isinstance(snapshot_root, str):
        fail("source_snapshot.root must be a string")
    try:
        Path(snapshot_root).resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        fail(f"source_snapshot.root escapes baseline: {snapshot_root} ({exc})")
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
        source = Path(entry["source_path"])
        try:
            source_resolved = source.resolve(strict=True)
            source_resolved.relative_to(root.resolve())
        except (OSError, ValueError) as exc:
            fail(f"source_path escapes baseline or is missing for {path_text}: {source} ({exc})")
        if source.is_symlink() or not source.is_file():
            fail(f"source_path is not a baseline-local regular file for {path_text}: {source}")
        expected_source, expected_dataset, expected_role = source_for(Path(path_text), root)
        if source.resolve() != expected_source.resolve():
            fail(f"source_path does not match baseline provenance for {path_text}")
        if entry["dataset"] != expected_dataset or entry["role"] != expected_role:
            fail(f"manifest dataset/role provenance mismatch for {path_text}")
        if digest(source) != entry["source_sha256"]:
            fail(f"source SHA-256 mismatch for {path_text}: {source}")
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
            expected_dataset = CHECKPOINT_DATASETS.get(Path(path_text).parts[3])
            if checkpoint_manifest.get("dataset") != expected_dataset:
                fail(f"checkpoint dataset identity mismatch in {path_text}")
            validate_checkpoint_archive(root, local, checkpoint_manifest)
        if path_text.endswith("/features.npy"):
            validate_npy_contract(root, local)
        if path_text.endswith("/gat_static_tables.bin"):
            validate_table_archive(local)
        if verify_source:
            pass
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

    paths = {entry["path"] for entry in manifest.get("files", []) if isinstance(entry, dict) and isinstance(entry.get("path"), str)}
    missing = []
    for dataset in sorted(REQUIRED_GENERATED_DATASETS):
        for relative in (
            f"resources/canonical/tables/{dataset}/gat_static_tables.bin",
            f"resources/canonical/ledgers/{dataset}/gat_integer_ledger.json",
        ):
            if relative not in paths:
                missing.append(relative)
    if missing:
        fail("PREPARE/BLOCKED: authenticated exact table/ledger resources are missing; "
             "run scripts/prepare_gat_resources.py first: " + ", ".join(missing))


def atomic_manifest_write(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


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
        "--require-native-srs",
        action="store_true",
        help="also require the source-matched native get-srs manifest/config binding",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / MANIFEST_REL
    if args.write_manifest:
        manifest = build_manifest(root)
        atomic_manifest_write(manifest_path, manifest)
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse manifest: {exc}")
    validate_manifest(root, manifest, args.verify_source)
    if args.require_complete:
        require_complete_resource_set(root, manifest)
    if args.require_native_srs:
        try:
            from formal_common import validate_native_srs_manifest

            validate_native_srs_manifest()
        except Exception as exc:
            fail(f"native SRS provenance validation failed: {exc}")
    print(f"resource manifest valid: {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
