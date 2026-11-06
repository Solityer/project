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
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

SCHEMA_VERSION = 1
TABLE_MAGIC = b"EZKL-GAT-TABLES\x00"
TABLE_FORMAT_VERSION = 1
TABLE_METADATA_LIMIT = 16 * 1024 * 1024
TABLE_CHUNK_ROWS = 1 << 20
MANIFEST_REL = Path("resources/resource_manifest.json")
CHECKPOINT_DATASETS = {
    "cora_gat": "cora",
    "citeseer_gat": "citeseer",
    "pubmed_gat": "pubmed",
    "ppi_gat": "ppi",
    "ogbn_arxiv_gat": "ogbn_arxiv",
}
REQUIRED_GENERATED_DATASETS = frozenset(CHECKPOINT_DATASETS.values())
REQUIRED_GENERATED_PATHS = frozenset(
    f"resources/canonical/{kind}/{dataset}/{name}"
    for kind, name in (("tables", "gat_static_tables.bin"), ("ledgers", "gat_integer_ledger.json"))
    for dataset in REQUIRED_GENERATED_DATASETS
)
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
DERIVED_IDENTITY_KEYS = frozenset(("static_table_id", "static_table_digest", "degree_bound_id", "degree_bound_digest"))
DERIVED_FEATURE_KEYS = frozenset(("feature_format", "feature_dtype", "feature_shape", "feature_order", "feature_length", "feature_sha256"))
SRS_CONFIG_BINDING_KEYS = frozenset(("srs_id", "srs_path", "trusted_srs_sha256"))
_PREPARATION_AUTH_TOKEN = object()


class AuthenticatedPreparationSnapshot:
    """Pre-generation identities authenticated by this process for one refresh."""

    __slots__ = ("root", "manifest_path", "manifest_bytes", "manifest", "source_snapshot", "input_paths", "input_identities", "identity_files", "_token")

    def __init__(self, token: object, root: Path, manifest_path: Path, manifest_bytes: bytes, manifest: dict[str, Any], source_snapshot: dict[str, str], input_paths: list[Path], input_identities: tuple[Any, ...], identity_files: dict[Path, bytes]) -> None:
        if token is not _PREPARATION_AUTH_TOKEN:
            raise TypeError("authenticated preparation snapshots can only be issued by input authentication")
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest_bytes = manifest_bytes
        self.manifest = manifest
        self.source_snapshot = source_snapshot
        self.input_paths = input_paths
        self.input_identities = input_identities
        self.identity_files = identity_files
        self._token = token


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


def _npy_contract(
    path: Path,
    content_sha256: str | None = None,
    *,
    include_sha256: bool = True,
) -> dict[str, Any]:
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
    contract = {
        "dtype": dtype,
        "shape": list(shape),
        "order": "C",
        "length": length,
    }
    if include_sha256:
        contract["sha256"] = content_sha256 or digest(path)
    return contract


def _parse_cfg(path: Path) -> dict[str, str]:
    return _parse_cfg_bytes(path.read_bytes(), path)


def validate_npy_contract(
    root: Path,
    feature_path: Path,
    content_sha256: str | None = None,
    *,
    check_sha256: bool = True,
) -> None:
    contract = _npy_contract(feature_path, content_sha256, include_sha256=check_sha256)
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
    }
    if check_sha256:
        expected["feature_sha256"] = contract["sha256"]
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


class DigestReader:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, length: int = -1) -> bytes:
        data = self.stream.read(length)
        self.hasher.update(data)
        self.bytes_read += len(data)
        return data


def validate_table_archive(path: Path) -> str:
    """Validate the table schema and raw file digest during the same bounded scan."""
    print(f"[resource-validation] table schema/content scan: {path} ({path.stat().st_size:,} bytes)", flush=True)
    with path.open("rb") as raw_stream:
        stream = DigestReader(raw_stream)
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
            input_fits_i64 = -(1 << 63) <= input_min <= input_max <= (1 << 63) - 1
            expected_input = input_min
            row_dtype = np.dtype([("low", "<i8"), ("high", "<i8"), ("value", "V16")])
            for row_offset in range(0, row_count, TABLE_CHUNK_ROWS):
                chunk_rows = min(TABLE_CHUNK_ROWS, row_count - row_offset)
                rows = _read_exact(stream, chunk_rows * 32, "table rows", path)
                records = np.frombuffer(rows, dtype=row_dtype)
                if input_fits_i64:
                    # Avoid constructing an exclusive stop of INT64_MAX + 1
                    # for a valid final key at the signed upper boundary.
                    expected = np.int64(expected_input) + np.arange(chunk_rows, dtype=np.int64)
                    if not np.array_equal(records["low"], expected):
                        fail(f"table row input is not contiguous: {path}")
                    negative = expected < 0
                    if np.any(records["high"][negative] != -1) or np.any(records["high"][~negative] != 0):
                        fail(f"table row input is not canonical signed i128: {path}")
                else:
                    for row_index in range(chunk_rows):
                        start = row_index * 32
                        actual_input = int.from_bytes(rows[start : start + 16], "little", signed=True)
                        if actual_input != expected_input + row_index:
                            fail(f"table row input is not contiguous: {path}")
                table_hasher.update(rows)
                static_hasher.update(rows)
                expected_input += chunk_rows
                print(f"[resource-validation] table rows {row_offset + chunk_rows:,}/{row_count:,} ({stream.bytes_read:,} archive bytes read)", flush=True)
            table_digest = table_hasher.hexdigest()
            if metadata["digest"] != table_digest:
                fail(f"table row digest mismatch: {path}")
            _hash_string(static_hasher, metadata["digest"])
        if static_hasher.hexdigest() != registry["static_tables_digest"]:
            fail(f"static-table registry digest mismatch: {path}")
        if stream.read(1):
            fail(f"static-table archive has trailing bytes: {path}")
        return stream.hasher.hexdigest()


def build_manifest(root: Path, origin_snapshot: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    entries: list[dict[str, Any]] = []
    parsed_table_digests: dict[str, str] = {}
    for local in regular_files(root):
        local_rel = local.relative_to(root)
        local_text = local_rel.as_posix()
        source, dataset, role = source_for(local_rel, root)
        if not source.is_file():
            fail(f"source file for {local_rel} is missing: {source}")
        if local_text.endswith("/gat_static_tables.bin"):
            local_digest = validate_table_archive(local)
            parsed_table_digests[local_text] = local_digest
        else:
            local_digest = digest(local)
        source_digest = local_digest if source.resolve() == local.resolve() else digest(source)
        entries.append(
            {
                "path": local_text,
                "size": local.stat().st_size,
                "sha256": local_digest,
                "source_path": str(source.resolve()),
                "source_sha256": source_digest,
                "dataset": dataset,
                "role": role,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": "ezkl",
        "resource_root": "resources/canonical",
        "source_snapshot": origin_snapshot,
        "files": entries,
    }, parsed_table_digests


def preserved_source_snapshot(root: Path, old_manifest: dict[str, Any]) -> dict[str, str]:
    """Carry forward the authenticated origin snapshot without invoking Git."""
    snapshot = old_manifest.get("source_snapshot")
    if not isinstance(snapshot, dict) or set(snapshot) != {"root", "git_commit", "git_tree"}:
        fail("existing resource manifest has no complete origin source_snapshot")
    if snapshot.get("root") != str(root.resolve()):
        fail("existing resource manifest source_snapshot.root does not match this baseline")
    for key in ("git_commit", "git_tree"):
        value = snapshot.get(key)
        if not isinstance(value, str) or len(value) not in (40, 64) or any(char not in "0123456789abcdef" for char in value):
            fail(f"existing resource manifest source_snapshot.{key} is not a recorded hex identity")
    return {key: snapshot[key] for key in ("root", "git_commit", "git_tree")}


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


def validate_manifest(
    root: Path,
    manifest: dict[str, Any],
    verify_source: bool,
    *,
    inputs_only: bool = False,
    parsed_table_digests: dict[str, str] | None = None,
) -> None:
    skip_provenance_checks = os.environ.get("EZKL_SKIP_PROVENANCE_CHECKS") == "1"
    if set(manifest) != REQUIRED_TOP_LEVEL:
        fail("manifest top-level schema is not exact")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["baseline"] != "ezkl":
        fail("unsupported manifest version or baseline")
    if manifest["resource_root"] != "resources/canonical":
        fail("resource_root is not baseline-local")
    if not isinstance(manifest["source_snapshot"], dict):
        fail("source_snapshot must be an object")
    snapshot = manifest["source_snapshot"]
    if set(snapshot) != {"root", "git_commit", "git_tree"}:
        fail("source_snapshot schema mismatch")
    snapshot_root = snapshot.get("root")
    if not isinstance(snapshot_root, str):
        fail("source_snapshot.root must be a string")
    if snapshot_root != str(root.resolve()):
        fail("source_snapshot.root does not match this baseline")
    for key in ("git_commit", "git_tree"):
        value = snapshot.get(key)
        if not isinstance(value, str) or len(value) not in (40, 64) or any(char not in "0123456789abcdef" for char in value):
            fail(f"source_snapshot.{key} is not a recorded hex identity")
    try:
        Path(snapshot_root).resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        fail(f"source_snapshot.root escapes baseline: {snapshot_root} ({exc})")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        fail("files must contain at least one canonical entry")
    expected_paths = {path.relative_to(root).as_posix() for path in regular_files(root)}
    seen: set[str] = set()
    validated_entries: list[tuple[dict[str, Any], Path, Path]] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != REQUIRED_FILE_KEYS:
            fail("file entry schema is not exact")
        path_text = entry["path"]
        if not isinstance(path_text, str) or path_text in seen:
            fail(f"duplicate or malformed manifest path: {path_text!r}")
        seen.add(path_text)
        relative = Path(path_text)
        generated = relative.parts[:3] in {("resources", "canonical", "tables"), ("resources", "canonical", "ledgers")}
        if generated:
            valid_table = relative.parts[:3] == ("resources", "canonical", "tables") and len(relative.parts) == 5 and relative.name == "gat_static_tables.bin"
            valid_ledger = relative.parts[:3] == ("resources", "canonical", "ledgers") and len(relative.parts) == 5 and relative.name == "gat_integer_ledger.json"
            if not (valid_table or valid_ledger) or relative.parts[3] not in REQUIRED_GENERATED_DATASETS:
                fail(f"unexpected generated resource path: {path_text}")
        if not isinstance(entry["size"], int) or isinstance(entry["size"], bool) or entry["size"] < 0:
            fail(f"invalid size for {path_text}")
        for key in ("sha256", "source_sha256"):
            value = entry[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                fail(f"invalid {key} for {path_text}")
        if not isinstance(entry["source_path"], str) or not os.path.isabs(entry["source_path"]):
            fail(f"source_path is not absolute metadata for {path_text}")
        source = Path(entry["source_path"])
        expected_source, expected_dataset, expected_role = source_for(relative, root)
        if (
            (not skip_provenance_checks and source.resolve() != expected_source.resolve()) or
            entry["dataset"] != expected_dataset or entry["role"] != expected_role
        ):
            fail(f"manifest dataset/role/source provenance mismatch for {path_text}")
        if not isinstance(entry["dataset"], str) or not entry["dataset"] or not isinstance(entry["role"], str) or not entry["role"]:
            fail(f"manifest dataset/role metadata is missing for {path_text}")
        if "qat_best_logits" in path_text:
            fail("qat_best_logits is not a formal resource")
        if not skip_provenance_checks:
            try:
                source_resolved = source.resolve(strict=True)
                source_resolved.relative_to(root.resolve())
            except (OSError, ValueError) as exc:
                fail(f"source_path escapes baseline or is missing for {path_text}: {source} ({exc})")
            if source.is_symlink() or not source.is_file():
                fail(f"source_path is not a baseline-local regular file for {path_text}: {source}")
        if inputs_only and generated:
            continue
        local = safe_relative(root, path_text)
        if not local.is_file() or local.is_symlink():
            fail(f"manifest file is missing or symlink: {path_text}")
        if not skip_provenance_checks and local.stat().st_size != entry["size"]:
            fail(f"size mismatch for {path_text}")
        validated_entries.append((entry, local, source))
    ignored_generated = REQUIRED_GENERATED_PATHS if inputs_only else frozenset()
    validated_seen = seen - ignored_generated
    validated_expected = expected_paths - ignored_generated
    if validated_seen != validated_expected:
        missing = sorted(validated_expected - validated_seen)
        extra = sorted(validated_seen - validated_expected)
        fail(f"manifest file set mismatch; missing={missing}, extra={extra}")
    for index, (entry, local, source) in enumerate(validated_entries, 1):
        path_text = entry["path"]
        print(f"[resource-validation] content {index}/{len(validated_entries)}: {path_text} ({entry['size']:,} bytes)", flush=True)
        if path_text.endswith("/gat_static_tables.bin"):
            # A manifest refresh just parsed this exact table and bound the
            # complete raw SHA-256. Reuse that content-authenticated result in
            # this same process; ordinary validation always parses and hashes
            # in one pass below.
            cached_digest = parsed_table_digests.get(path_text) if parsed_table_digests is not None else None
            if cached_digest is not None and not skip_provenance_checks:
                file_digest = cached_digest
            else:
                file_digest = validate_table_archive(local)
        elif skip_provenance_checks:
            # The unbound recovery path still parses semantic structures below,
            # but it does not rescan every resource to compare stale manifest
            # hashes or consult the old source-copy identities.
            file_digest = None
        else:
            file_digest = digest(local)
        if not skip_provenance_checks and file_digest != entry["sha256"]:
            fail(f"local SHA-256 mismatch for {path_text}")
        if not skip_provenance_checks:
            if source.resolve() == local.resolve():
                source_digest = file_digest
            else:
                source_digest = digest(source)
            if source_digest != entry["source_sha256"]:
                fail(f"source SHA-256 mismatch for {path_text}: {source}")
        if path_text.endswith("/features.npy"):
            validate_npy_contract(
                root,
                local,
                file_digest,
                check_sha256=not skip_provenance_checks,
            )
        if path_text.endswith("/manifest.json"):
            try:
                checkpoint_manifest = json.loads(local.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"invalid checkpoint manifest {path_text}: {exc}")
            if checkpoint_manifest.get("checkpoint_prefix") != "best_model.pt":
                fail(f"checkpoint_prefix is not bundle-local in {path_text}")
            if checkpoint_manifest.get("dataset") != CHECKPOINT_DATASETS.get(Path(path_text).parts[3]):
                fail(f"checkpoint dataset identity mismatch in {path_text}")
            validate_checkpoint_archive(root, local, checkpoint_manifest)
        # ``verify_source`` remains accepted for compatibility. Baseline-local
        # source files are always hashed above; no external source is read.


def _preparation_identity_paths(root: Path) -> list[Path]:
    paths = [root / "resources/canonical/configs" / f"{dataset}_secure.cfg" for dataset in CHECKPOINT_DATASETS.values()]
    paths.extend(root / "resources/canonical/checkpoints" / bundle / "manifest.json" for bundle in CHECKPOINT_DATASETS)
    paths.extend(root / "resources/canonical/cache" / dataset / "meta.cfg" for dataset in ("ppi", "ogbn_arxiv"))
    return paths


def _preparation_input_paths(root: Path, manifest: dict[str, Any]) -> list[Path]:
    identity_paths = {path.resolve() for path in _preparation_identity_paths(root)}
    paths: list[Path] = []
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("source_path"), str):
            fail("resource manifest has a malformed entry before authenticated preparation")
        relative = entry["path"]
        if relative.startswith("resources/canonical/tables/") or relative.startswith("resources/canonical/ledgers/"):
            continue
        local = safe_relative(root, relative)
        source = Path(entry["source_path"])
        if not source.is_absolute():
            source = root / source
        if local.resolve() not in identity_paths:
            paths.append(local)
        if source.resolve() not in identity_paths:
            paths.append(source)
    paths.append(root / "scripts/generate_gat_resources.py")
    return paths


def _file_identity_snapshot(root: Path, paths: list[Path]) -> tuple[Any, ...]:
    by_relative: dict[str, Path] = {}
    for candidate in paths:
        if candidate.is_symlink():
            fail(f"authenticated resource path is a symlink: {candidate}")
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(root.resolve()).as_posix()
        except (OSError, ValueError) as exc:
            fail(f"authenticated resource path is missing or escapes baseline: {candidate} ({exc})")
        if not resolved.is_file():
            fail(f"authenticated resource path is not a regular file: {candidate}")
        by_relative[relative] = resolved
    snapshots = []
    for relative, path in sorted(by_relative.items()):
        info = path.stat()
        snapshots.append((relative, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)))
    return tuple(snapshots)


def authenticate_preparation_inputs(root: Path) -> AuthenticatedPreparationSnapshot:
    """Authenticate the old manifest before a generator may change derived fields."""
    root = root.resolve()
    manifest_path = root / MANIFEST_REL
    if manifest_path.is_symlink() or not manifest_path.is_file():
        fail(f"canonical resource manifest is missing or symlinked before preparation: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read canonical resource manifest before preparation: {exc}")
    if not isinstance(manifest, dict):
        fail("canonical resource manifest must be an object before preparation")
    identity_paths = _preparation_identity_paths(root)
    try:
        for path in identity_paths:
            if path.is_symlink() or not path.is_file():
                fail(f"derivation identity file is missing or symlinked before preparation: {path}")
        identity_files = {path: path.read_bytes() for path in identity_paths}
    except OSError as exc:
        fail(f"cannot snapshot derivation identity files before preparation: {exc}")
    input_paths = _preparation_input_paths(root, manifest)
    input_identities = _file_identity_snapshot(root, input_paths)
    print("[resource-validation] authenticate original canonical inputs before derivation", flush=True)
    validate_manifest(root, manifest, False, inputs_only=True)
    if manifest_path.read_bytes() != manifest_bytes:
        fail("resource manifest changed during authenticated preparation preflight")
    if _file_identity_snapshot(root, input_paths) != input_identities:
        fail("canonical input files changed during authenticated preparation preflight")
    for path, payload in identity_files.items():
        if path.read_bytes() != payload:
            fail(f"derivation identity file changed during authenticated preparation preflight: {path}")
    return AuthenticatedPreparationSnapshot(
        _PREPARATION_AUTH_TOKEN,
        root,
        manifest_path,
        manifest_bytes,
        manifest,
        preserved_source_snapshot(root, manifest),
        input_paths,
        input_identities,
        identity_files,
    )


def _parse_cfg_bytes(payload: bytes, path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(payload.decode().splitlines(), 1):
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


def _check_derived_identity_changes(snapshot: AuthenticatedPreparationSnapshot) -> None:
    root = snapshot.root
    for path, old_payload in snapshot.identity_files.items():
        try:
            if path.is_symlink() or not path.is_file():
                fail(f"derivation identity file is not a regular file after generation: {path}")
            new_payload = path.read_bytes()
        except OSError as exc:
            fail(f"derivation identity file is missing after generation: {path} ({exc})")
        relative = path.relative_to(root).as_posix()
        if relative.startswith("resources/canonical/configs/"):
            old_values = _parse_cfg_bytes(old_payload, path)
            new_values = _parse_cfg_bytes(new_payload, path)
            if {key: value for key, value in old_values.items() if key not in DERIVED_IDENTITY_KEYS} != {key: value for key, value in new_values.items() if key not in DERIVED_IDENTITY_KEYS}:
                fail(f"resource generator changed a non-derived config value: {relative}")
        elif relative.startswith("resources/canonical/checkpoints/"):
            try:
                old_values = json.loads(old_payload)
                new_values = json.loads(new_payload)
            except json.JSONDecodeError as exc:
                fail(f"invalid checkpoint manifest after generation: {relative} ({exc})")
            if not isinstance(old_values, dict) or not isinstance(new_values, dict):
                fail(f"checkpoint manifest is not an object after generation: {relative}")
            if {key: value for key, value in old_values.items() if key not in DERIVED_IDENTITY_KEYS} != {key: value for key, value in new_values.items() if key not in DERIVED_IDENTITY_KEYS}:
                fail(f"resource generator changed a non-derived checkpoint field: {relative}")
        elif relative in {"resources/canonical/cache/ppi/meta.cfg", "resources/canonical/cache/ogbn_arxiv/meta.cfg"}:
            old_values = _parse_cfg_bytes(old_payload, path)
            new_values = _parse_cfg_bytes(new_payload, path)
            if {key: value for key, value in old_values.items() if key not in DERIVED_FEATURE_KEYS} != {key: value for key, value in new_values.items() if key not in DERIVED_FEATURE_KEYS}:
                fail(f"resource generator changed a non-derived feature metadata value: {relative}")
        else:
            fail(f"unrecognized derivation identity file: {relative}")


def _read_table_registry(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            if _read_exact(stream, len(TABLE_MAGIC), "table archive magic", path) != TABLE_MAGIC:
                fail(f"invalid table archive magic: {path}")
            version = struct.unpack("<I", _read_exact(stream, 4, "table archive version", path))[0]
            count = struct.unpack("<I", _read_exact(stream, 4, "table archive count", path))[0]
            if version != TABLE_FORMAT_VERSION or count != 4:
                fail(f"unsupported table archive version/count in {path}")
            registry = _read_metadata(stream, "registry metadata", path)
    except OSError as exc:
        fail(f"cannot read table registry {path}: {exc}")
    if registry.get("format") != "ezkl-gat-static-tables-bin" or registry.get("version") != TABLE_FORMAT_VERSION:
        fail(f"table registry identity mismatch: {path}")
    return registry


def validate_generated_resource_links(root: Path) -> None:
    """Bind every dataset's configs and checkpoint to the actual table/ledger IDs."""
    try:
        from generate_gat_resources import ledger_digest
    except ImportError as exc:
        fail(f"cannot load canonical ledger digest implementation: {exc}")
    for bundle, dataset in CHECKPOINT_DATASETS.items():
        config_path = root / "resources/canonical/configs" / f"{dataset}_secure.cfg"
        checkpoint_path = root / "resources/canonical/checkpoints" / bundle / "manifest.json"
        table_path = root / "resources/canonical/tables" / dataset / "gat_static_tables.bin"
        ledger_path = root / "resources/canonical/ledgers" / dataset / "gat_integer_ledger.json"
        config = _parse_cfg(config_path)
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
            ledger = json.loads(ledger_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"cannot read {dataset} checkpoint/ledger identities: {exc}")
        if not isinstance(checkpoint, dict) or not isinstance(ledger, dict):
            fail(f"{dataset} checkpoint manifest and integer ledger must be JSON objects")
        registry = _read_table_registry(table_path)
        expected_table_id = f"tables:gat:{dataset}:v1"
        expected_ledger_id = f"ledger:gat:{dataset}:v1"
        if registry.get("static_tables_id") != expected_table_id or registry.get("static_tables_digest") != checkpoint.get("static_table_digest"):
            fail(f"{dataset} table header identity does not match the canonical dataset/checkpoint")
        if registry.get("static_tables_id") != checkpoint.get("static_table_id"):
            fail(f"{dataset} table header ID does not match checkpoint manifest")
        if registry.get("static_tables_digest") != config.get("static_table_digest") or registry.get("static_tables_id") != config.get("static_table_id"):
            fail(f"{dataset} table header identity does not match secure config")
        if ledger.get("ledger_id") != expected_ledger_id or ledger.get("ledger_digest") != checkpoint.get("degree_bound_digest"):
            fail(f"{dataset} integer ledger identity does not match the canonical dataset/checkpoint")
        if ledger.get("ledger_id") != checkpoint.get("degree_bound_id"):
            fail(f"{dataset} integer ledger ID does not match checkpoint manifest")
        if ledger.get("ledger_id") != config.get("degree_bound_id") or ledger.get("ledger_digest") != config.get("degree_bound_digest"):
            fail(f"{dataset} integer ledger identity does not match secure config")
        try:
            calculated_digest = ledger_digest(ledger)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            fail(f"invalid {dataset} integer ledger canonical fields: {exc}")
        if ledger.get("ledger_digest") != calculated_digest:
            fail(f"{dataset} integer ledger digest does not match canonical bounds")


def _verify_preparation_snapshot(snapshot: AuthenticatedPreparationSnapshot, *, manifest_unchanged: bool) -> None:
    if not isinstance(snapshot, AuthenticatedPreparationSnapshot) or snapshot._token is not _PREPARATION_AUTH_TOKEN:
        fail("resource refresh requires the current authenticated preparation snapshot")
    if snapshot.root != snapshot.manifest_path.parents[1]:
        fail("authenticated preparation snapshot belongs to a different baseline")
    if manifest_unchanged and snapshot.manifest_path.read_bytes() != snapshot.manifest_bytes:
        fail("resource manifest changed during authenticated derivation")
    if _file_identity_snapshot(snapshot.root, snapshot.input_paths) != snapshot.input_identities:
        fail("an authenticated canonical input changed during derivation")
    if manifest_unchanged:
        _check_derived_identity_changes(snapshot)


def preparation_resources_are_current(snapshot: AuthenticatedPreparationSnapshot) -> bool:
    _verify_preparation_snapshot(snapshot, manifest_unchanged=True)
    try:
        from formal_common import resource_file_identities

        identities_before = resource_file_identities()
        require_complete_resource_set(snapshot.root, snapshot.manifest)
        validate_manifest(snapshot.root, snapshot.manifest, False)
        validate_generated_resource_links(snapshot.root)
        identities_after = resource_file_identities()
        if identities_before != identities_after:
            fail("manifest-listed resource files changed during complete current-resource authentication")
    except SystemExit as exc:
        if exc.code not in (None, 2):
            raise
        _verify_preparation_snapshot(snapshot, manifest_unchanged=True)
        return False
    _verify_preparation_snapshot(snapshot, manifest_unchanged=True)
    _emit_formal_resource_auth_marker(snapshot.root, identities_after)
    return True


def _compare_fresh_manifest_to_authenticated_inputs(
    snapshot: AuthenticatedPreparationSnapshot,
    fresh_manifest: dict[str, Any],
) -> None:
    """Permit new hashes only for regenerated outputs and allowlisted local identities."""
    old_entries = {entry["path"]: entry for entry in snapshot.manifest["files"]}
    fresh_entries = {entry["path"]: entry for entry in fresh_manifest["files"]}
    mutable_identity_paths = {
        path.relative_to(snapshot.root).as_posix() for path in _preparation_identity_paths(snapshot.root)
    }
    generated_paths = set(REQUIRED_GENERATED_PATHS)
    if not mutable_identity_paths.issubset(set(old_entries) & set(fresh_entries)):
        fail("a managed derived identity file disappeared from the authenticated manifest")
    if (set(old_entries) - generated_paths) - set(fresh_entries):
        missing = sorted((set(old_entries) - generated_paths) - set(fresh_entries))
        fail(f"authenticated immutable resources disappeared during derivation: {missing}")
    if (set(fresh_entries) - generated_paths) - set(old_entries):
        added = sorted((set(fresh_entries) - generated_paths) - set(old_entries))
        fail(f"unmanifested immutable resources appeared during derivation: {added}")

    fields = ("size", "sha256", "source_path", "source_sha256", "dataset", "role")
    for path_text in sorted(set(old_entries) & set(fresh_entries)):
        old = old_entries[path_text]
        fresh = fresh_entries[path_text]
        if path_text in generated_paths:
            # Generated tables/ledgers may be rebuilt. Their source is the current
            # local generator, whose provenance is refreshed with the output.
            for key in ("source_path", "dataset", "role"):
                if old[key] != fresh[key]:
                    fail(f"generated resource provenance changed during derivation: {path_text} ({key})")
            continue
        if path_text in mutable_identity_paths:
            local = safe_relative(snapshot.root, path_text).resolve()
            for entry in (old, fresh):
                try:
                    source = Path(entry["source_path"]).resolve(strict=True)
                except OSError as exc:
                    fail(f"derived identity source is missing for {path_text}: {exc}")
                if source != local or entry["source_path"] != old["source_path"]:
                    fail(f"derived identity is not bound to its same local managed file: {path_text}")
                if entry["source_sha256"] != entry["sha256"]:
                    fail(f"derived identity source hash is not its local file hash: {path_text}")
            for key in ("dataset", "role"):
                if old[key] != fresh[key]:
                    fail(f"derived identity provenance changed during derivation: {path_text} ({key})")
            continue
        if any(old[key] != fresh[key] for key in fields):
            fail(f"authenticated immutable resource identity changed during derivation: {path_text}")

    # A newly generated table/ledger must still name the expected local producer.
    generator = (snapshot.root / "scripts/generate_gat_resources.py").resolve()
    for path_text in generated_paths & set(fresh_entries):
        if Path(fresh_entries[path_text]["source_path"]).resolve() != generator:
            fail(f"generated resource has an unexpected producer: {path_text}")


def _srs_bound_config_bytes(old_payload: bytes, path: Path, expected: dict[str, str]) -> bytes:
    if set(expected) != SRS_CONFIG_BINDING_KEYS:
        fail(f"authenticated SRS config binding has an invalid schema for {path}")
    output: list[str] = []
    seen: set[str] = set()
    for raw in old_payload.decode().splitlines():
        key = raw.split("#", 1)[0].strip().split("=", 1)[0].strip() if "=" in raw else ""
        if key in SRS_CONFIG_BINDING_KEYS:
            output.append(f"{key} = {expected[key]}")
            seen.add(key)
        else:
            output.append(raw)
    if seen != SRS_CONFIG_BINDING_KEYS:
        fail(f"{path} must contain srs_id, srs_path and trusted_srs_sha256")
    return ("\n".join(output) + "\n").encode()


def _check_authenticated_srs_binding(
    root: Path,
    srs_manifest: dict[str, Any],
    *,
    source_url: str,
    logrows: int,
    expected_sha256: str,
    actual_sha256: str,
    srs_path: str,
    srs_size: int,
    srs_file_identity: tuple[int, int, int, int, int],
) -> dict[str, dict[str, str]]:
    from formal_common import native_srs_contract
    native_url, native_logrows, native_hashes = native_srs_contract()
    if (source_url, logrows, expected_sha256) != (f"{native_url}{native_logrows}.srs", native_logrows, native_hashes[native_logrows]):
        fail("authenticated SRS derivation does not match the pinned native get-srs source/hash contract")
    if actual_sha256 != expected_sha256 or len(actual_sha256) != 64 or any(char not in "0123456789abcdef" for char in actual_sha256):
        fail("authenticated SRS derivation digest is not the pinned native trusted SRS hash")
    expected_id = f"official_ezkl_native_shared_kzg{logrows}_srs"
    expected_path = f"resources/native_srs/kzg{logrows}.srs"
    if srs_path != expected_path or Path(srs_path).is_absolute() or ".." in Path(srs_path).parts:
        fail("authenticated SRS derivation path is not the pinned native shared cache")
    unresolved_srs = root / srs_path
    if unresolved_srs.is_symlink():
        fail("authenticated native SRS path is symlinked")
    local_srs = safe_relative(root, srs_path)
    if local_srs.is_symlink() or not local_srs.is_file() or srs_size < 4096:
        fail("authenticated native SRS is missing, symlinked or too small")
    info = local_srs.stat()
    current_identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if current_identity != srs_file_identity or info.st_size != srs_size:
        fail("authenticated native SRS file identity changed during resource derivation")
    expected_top = {
        "schema_version": 2,
        "baseline": "ezkl",
        "srs_type": "halo2-kzg-shplonk",
        "shared": True,
        "max_logrows": logrows,
        "source": "official_ezkl_ceremony",
        "provisioner": "native_get_srs",
        "source_url": source_url,
        "hash_source": "src/srs_sha.rs",
        "native_hash_verified": True,
    }
    if not isinstance(srs_manifest, dict) or set(srs_manifest) != set(expected_top) | {"entries"}:
        fail("authenticated native SRS manifest schema is invalid")
    if any(srs_manifest.get(key) != value for key, value in expected_top.items()):
        fail("authenticated native SRS manifest does not match the pinned source contract")
    manifest_path = root / "resources/native_srs/srs_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        fail("authenticated native SRS manifest is missing or symlinked")
    try:
        on_disk_manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot re-read authenticated native SRS manifest: {exc}")
    if on_disk_manifest != srs_manifest:
        fail("native SRS manifest changed during resource derivation")
    entries = srs_manifest.get("entries")
    if not isinstance(entries, dict) or set(entries) != set(CHECKPOINT_DATASETS.values()):
        fail("authenticated native SRS manifest does not contain exactly five formal datasets")
    result: dict[str, dict[str, str]] = {}
    for dataset in CHECKPOINT_DATASETS.values():
        entry = entries[dataset]
        expected_entry = {
            "dataset": dataset,
            "srs_id": expected_id,
            "srs_type": "trusted_srs",
            "path": srs_path,
            "sha256": actual_sha256,
            "size": srs_size,
            "logrows": logrows,
            "source": "official_ezkl_ceremony",
            "provisioner": "native_get_srs",
            "source_url": source_url,
            "hash_source": "src/srs_sha.rs",
            "native_hash_verified": True,
            "trusted": True,
            "reduced": False,
            "generated": False,
        }
        if not isinstance(entry, dict) or entry != expected_entry:
            fail(f"native SRS manifest entry does not match authenticated source/file identity: {dataset}")
        result[dataset] = {"srs_id": expected_id, "srs_path": srs_path, "trusted_srs_sha256": actual_sha256}
    return result


def refresh_manifest_after_authenticated_srs_derivation(
    snapshot: AuthenticatedPreparationSnapshot,
    srs_manifest: dict[str, Any],
    *,
    source_url: str,
    logrows: int,
    expected_sha256: str,
    actual_sha256: str,
    srs_path: str,
    srs_size: int,
    srs_file_identity: tuple[int, int, int, int, int],
) -> None:
    """Refresh only the five exact SRS config bindings from an authenticated native SRS file."""
    if not isinstance(snapshot, AuthenticatedPreparationSnapshot) or snapshot._token is not _PREPARATION_AUTH_TOKEN:
        fail("SRS resource refresh requires the current authenticated preparation snapshot")
    if snapshot.root != snapshot.manifest_path.parents[1] or snapshot.manifest_path.read_bytes() != snapshot.manifest_bytes:
        fail("authenticated resource manifest changed before SRS derivation refresh")
    if _file_identity_snapshot(snapshot.root, snapshot.input_paths) != snapshot.input_identities:
        fail("an authenticated canonical input changed during SRS derivation")
    expected_configs = _check_authenticated_srs_binding(
        snapshot.root,
        srs_manifest,
        source_url=source_url,
        logrows=logrows,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        srs_path=srs_path,
        srs_size=srs_size,
        srs_file_identity=srs_file_identity,
    )
    config_paths = {snapshot.root / "resources/canonical/configs" / f"{dataset}_secure.cfg": values for dataset, values in expected_configs.items()}
    for path, old_payload in snapshot.identity_files.items():
        if path in config_paths:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != _srs_bound_config_bytes(old_payload, path, config_paths[path]):
                fail(f"SRS derivation changed an unauthorized config field: {path.relative_to(snapshot.root).as_posix()}")
        elif path.is_symlink() or not path.is_file() or path.read_bytes() != old_payload:
            fail(f"non-SRS derivation identity changed during SRS provisioning: {path.relative_to(snapshot.root).as_posix()}")
    resource_paths: list[Path] = []
    for local in regular_files(snapshot.root):
        source, _, _ = source_for(local.relative_to(snapshot.root), snapshot.root)
        resource_paths.extend((local, source))
    identities_before_build = _file_identity_snapshot(snapshot.root, resource_paths)
    manifest, parsed_table_digests = build_manifest(snapshot.root, snapshot.source_snapshot)
    if identities_before_build != _file_identity_snapshot(snapshot.root, resource_paths):
        fail("canonical resource files changed while the SRS-bound manifest hashes were computed")
    old_entries = {entry["path"]: entry for entry in snapshot.manifest["files"]}
    fresh_entries = {entry["path"]: entry for entry in manifest["files"]}
    if set(old_entries) != set(fresh_entries):
        fail("canonical resource paths changed during SRS binding")
    for path_text, old in old_entries.items():
        fresh = fresh_entries[path_text]
        if path_text.startswith("resources/canonical/configs/") and path_text.endswith("_secure.cfg"):
            if any(old[key] != fresh[key] for key in ("source_path", "dataset", "role")):
                fail(f"secure config provenance changed during native SRS binding: {path_text}")
            if old["source_path"] != str(safe_relative(snapshot.root, path_text).resolve()) or fresh["source_path"] != old["source_path"]:
                fail(f"secure config source is not its same authenticated local file: {path_text}")
            if fresh["source_sha256"] != fresh["sha256"]:
                fail(f"secure config source hash is not its local file hash: {path_text}")
        elif old != fresh:
            fail(f"non-SRS canonical resource identity changed during native SRS binding: {path_text}")
    require_complete_resource_set(snapshot.root, manifest)
    validate_manifest(snapshot.root, manifest, False, parsed_table_digests=parsed_table_digests)
    validate_generated_resource_links(snapshot.root)
    identities_after_validation = _file_identity_snapshot(snapshot.root, resource_paths)
    if identities_before_build != identities_after_validation:
        fail("canonical resource files changed during complete SRS-bound resource authentication")
    _check_authenticated_srs_binding(
        snapshot.root,
        srs_manifest,
        source_url=source_url,
        logrows=logrows,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        srs_path=srs_path,
        srs_size=srs_size,
        srs_file_identity=srs_file_identity,
    )
    atomic_manifest_write(snapshot.manifest_path, manifest)
    _emit_formal_resource_auth_marker(snapshot.root, identities_after_validation)


def refresh_manifest_after_authenticated_derivation(snapshot: AuthenticatedPreparationSnapshot) -> None:
    """Refresh and strictly validate resources changed only by the authenticated generator."""
    _verify_preparation_snapshot(snapshot, manifest_unchanged=True)
    _check_derived_identity_changes(snapshot)
    resource_paths: list[Path] = []
    for local in regular_files(snapshot.root):
        source, _, _ = source_for(local.relative_to(snapshot.root), snapshot.root)
        resource_paths.extend((local, source))
    identities_before_build = _file_identity_snapshot(snapshot.root, resource_paths)
    manifest, parsed_table_digests = build_manifest(snapshot.root, snapshot.source_snapshot)
    identities_after_build = _file_identity_snapshot(snapshot.root, resource_paths)
    if identities_before_build != identities_after_build:
        fail("resource files changed while the fresh manifest hashes were computed")
    _verify_preparation_snapshot(snapshot, manifest_unchanged=True)
    _compare_fresh_manifest_to_authenticated_inputs(snapshot, manifest)
    require_complete_resource_set(snapshot.root, manifest)
    validate_manifest(
        snapshot.root,
        manifest,
        False,
        parsed_table_digests=parsed_table_digests,
    )
    validate_generated_resource_links(snapshot.root)
    identities_after_validation = _file_identity_snapshot(snapshot.root, resource_paths)
    if identities_after_build != identities_after_validation:
        fail("manifest-listed resources changed during complete post-derivation authentication")
    _verify_preparation_snapshot(snapshot, manifest_unchanged=True)
    atomic_manifest_write(snapshot.manifest_path, manifest)
    _emit_formal_resource_auth_marker(snapshot.root, identities_after_validation)


def _emit_formal_resource_auth_marker(root: Path, identities: tuple[Any, ...]) -> None:
    session_id = os.environ.get("EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID")
    if not session_id:
        return
    from formal_common import DATASETS, FORMAL_SESSION_ID, resource_file_identities, resource_manifest_hash, sha256, source_tree_hash
    if not FORMAL_SESSION_ID.fullmatch(session_id):
        fail("formal resource-authentication handoff has an invalid session identity")
    current_identities = resource_file_identities()
    if current_identities != identities:
        fail("manifest-listed resource files changed before the formal authentication handoff")
    config_sha256s = [[dataset, sha256(root / "resources/canonical/configs" / f"{dataset}_secure.cfg")] for dataset in DATASETS]
    marker = {
        "schema_version": 1,
        "authenticated": "complete_resource_manifest",
        "session_id": session_id,
        "source_tree_sha256": source_tree_hash(),
        "config_sha256s": config_sha256s,
        "resource_manifest_sha256": resource_manifest_hash(),
        "resource_file_identities_before": identities,
        "resource_file_identities_after": current_identities,
    }
    print("EZKL_FORMAL_RESOURCE_AUTH_V1=" + json.dumps(marker, sort_keys=True, separators=(",", ":")), flush=True)


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
    parser.add_argument("--validate-inputs-only", action="store_true", help="authenticate canonical cache/config/checkpoint inputs while allowing generated tables/ledgers to be absent")
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
    authenticated_file_identities_before: tuple[tuple[str, tuple[int, int, int, int, int]], ...] | None = None
    if args.write_manifest:
        if not manifest_path.is_file():
            fail(f"cannot refresh manifest without an authenticated origin snapshot: {manifest_path}")
        try:
            old_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"cannot parse existing resource manifest before refresh: {exc}")
        if not isinstance(old_manifest, dict):
            fail("existing resource manifest must be a JSON object before refresh")
        print("[resource-validation] authenticate existing non-generated inputs before manifest refresh", flush=True)
        validate_manifest(root, old_manifest, args.verify_source, inputs_only=True)
        origin_snapshot = preserved_source_snapshot(root, old_manifest)
        from formal_common import resource_file_identities_for_paths

        snapshot_paths: list[Path] = []
        for local in regular_files(root):
            source, _, _ = source_for(local.relative_to(root), root)
            snapshot_paths.extend((local, source))
        authenticated_file_identities_before = resource_file_identities_for_paths(snapshot_paths)
        manifest, parsed_table_digests = build_manifest(root, origin_snapshot)
        atomic_manifest_write(manifest_path, manifest)
    else:
        parsed_table_digests = None
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse manifest: {exc}")
    if not isinstance(manifest, dict):
        fail("resource manifest must be a JSON object")
    if args.validate_inputs_only and args.require_complete:
        fail("--validate-inputs-only cannot be combined with --require-complete")
    print(f"[resource-validation] schema=1 manifest={manifest_path} entries={len(manifest.get('files', []))} inputs_only={args.validate_inputs_only}", flush=True)
    if args.require_complete:
        require_complete_resource_set(root, manifest)
        from formal_common import (
            DATASETS,
            FORMAL_SESSION_ID,
            resource_file_identities,
            resource_manifest_hash,
            sha256,
            source_tree_hash,
        )

        if authenticated_file_identities_before is None:
            authenticated_file_identities_before = resource_file_identities()
    validate_manifest(
        root,
        manifest,
        args.verify_source,
        inputs_only=args.validate_inputs_only,
        parsed_table_digests=parsed_table_digests,
    )
    if args.require_complete:
        validate_generated_resource_links(root)
    if args.require_native_srs:
        try:
            from formal_common import validate_native_srs_manifest

            validate_native_srs_manifest()
        except Exception as exc:
            fail(f"native SRS provenance validation failed: {exc}")
    print(f"resource manifest valid: {len(manifest['files'])} files")
    if args.require_complete:
        authenticated_file_identities_after = resource_file_identities()
        if authenticated_file_identities_before != authenticated_file_identities_after:
            fail("manifest-listed resource files changed during complete authentication")
        _emit_formal_resource_auth_marker(root, authenticated_file_identities_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
