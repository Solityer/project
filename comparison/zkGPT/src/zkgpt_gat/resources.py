"""Baseline-local resource policy for the formal zkGPT runner.

The comparison baseline is deliberately self-contained.  This module resolves
and checks only baseline-local paths.  Resource *contents* are produced by the
official preparation entry point from the baseline's authenticated raw inputs;
the path checker itself never downloads, copies, or silently substitutes data.
"""
from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict, Iterable

from .util import assert_contained_path, lexical_path, require_dir
from .quantization import CANONICAL_QUANT_CFG_ID


RESOURCE_MANIFEST_SCHEMA = "zkGPT_baseline_resource_manifest_v1"
MODEL_REGISTRY_ROOT_RELATIVE_PATH = "results/model_registry"
PRIVATE_MODEL_SETUP_ROOT_RELATIVE_PATH = "artifacts/private_setup/model_registry"
DATASET_NAMES = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
CANONICAL_RANGE_BITS = {
    "cora": 12,
    "citeseer": 12,
    "pubmed": 12,
    "ppi": 14,
    "ogbn-arxiv": 21,
}
DATASET_ALIASES = {"ogbn_arxiv": "ogbn-arxiv"}
RESOURCE_BASENAME = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "ppi": "ppi",
    # The public dataset identity uses a hyphen; on-disk resources use the
    # one canonical underscore slug everywhere.
    "ogbn-arxiv": "ogbn_arxiv",
}
STATIC_TABLE_DESCRIPTOR_SCHEMA = "zkGPT_static_tables_descriptor_v1"
STATIC_TABLE_FAMILIES = ("T_LReLU", "T_ELU", "T_exp", "T_range")
STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH = "resources/static_tables/exact_integer_gat.json"

# This is the small, baseline-local protocol descriptor for the fixed tables.
# It deliberately contains no table rows, dataset values, checkpoint material,
# or generated proof data.  Per-instance integer domains remain certified by
# the public bound ledger; this descriptor authenticates the function identity,
# scales, encoding, and range-table policy used to construct those rows.
STATIC_TABLE_CONTENT: Dict[str, Any] = {
    "protocol_id": "GAT-ZKML-zkGPT",
    "encoding_id": "Pack_f_offset_base_v1",
    "canonical_row_order": "ascending signed x; row=(x,y,Pack_f(x,y))",
    "function_evaluators": {
        "T_LReLU": {
            "table_id": "T_LReLU",
            "function": "LeakyReLU",
            "evaluation_rule": "y=x if x>=0 else round(0.2*x)",
            "evaluator_id": "signed_fixed_point_leaky_relu_alpha_0p2_nearest_q",
            "input_scale": 256,
            "output_scale": 256,
            "domain_policy": "signed fixed-point representatives certified by integer_bound_ledger",
        },
        "T_ELU": {
            "table_id": "T_ELU",
            "function": "ELU",
            "evaluation_rule": "y=x if x>=0 else round(exp(x)-1)",
            "evaluator_id": "signed_fixed_point_elu_expm1_nearest_q",
            "input_scale": 16,
            "output_scale": 16,
            "domain_policy": "signed fixed-point representatives certified by integer_bound_ledger",
        },
        "T_exp": {
            "table_id": "T_exp",
            "function": "ExpMap",
            "evaluation_rule": "y=round(exp(-x)) for nonnegative x",
            "evaluator_id": "signed_fixed_point_exp_neg_nearest_q",
            "input_scale": 256,
            "output_scale": 16,
            "domain_policy": "nonnegative stable-softmax differences certified by integer_bound_ledger",
        },
    },
    "range_table": {
        "table_id": "T_range",
        "table_kind": "materialized_fixed_range_table",
        "entry_policy": "all limbs in [0,2^range_bits-1]",
        # Range width is a dataset-bound protocol input.  OGBN-Arxiv uses
        # 21 bits; retaining the complete map in the authenticated descriptor
        # prevents a stale global 12-bit default from being accepted.
        "range_bits_by_dataset": {
            "cora": 12,
            "citeseer": 12,
            "pubmed": 12,
            "ppi": 14,
            "ogbn-arxiv": 21,
        },
    },
}


def baseline_root(path: pathlib.Path) -> pathlib.Path:
    return require_dir(path, label="baseline root")


def inside(root: pathlib.Path, path: pathlib.Path) -> bool:
    try:
        assert_contained_path(root, path, "resource path", allow_missing=True)
    except (OSError, ValueError):
        return False
    return True


def canonical_dataset_name(dataset: str) -> str:
    value = str(dataset)
    value = DATASET_ALIASES.get(value, value)
    if value not in DATASET_NAMES:
        raise ValueError(f"unsupported formal dataset: {dataset}")
    return value


def canonical_range_bits(dataset: str) -> int:
    return int(CANONICAL_RANGE_BITS[canonical_dataset_name(dataset)])


def resolve_local(root: pathlib.Path, raw: str | pathlib.Path, label: str) -> pathlib.Path:
    """Resolve a config/CLI path and fail if it escapes ``root``.

    Absolute paths are accepted only when they already point inside the
    baseline.  Generated formal configs use relative paths, which keeps the
    execution chain relocatable and makes accidental project-root coupling
    visible during review.
    """
    root = baseline_root(root)
    value = pathlib.Path(raw).expanduser()
    candidate = value if value.is_absolute() else root / value
    return assert_contained_path(root, candidate, label, allow_missing=True)


def _owned_regular(path: pathlib.Path, label: str) -> pathlib.Path:
    """Authenticate one baseline-local resource file before opening it."""
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise OSError(f"cannot inspect {label}: {path}") from exc
    import stat

    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if info.st_nlink != 1:
        raise ValueError(f"{label} hardlink count must be one: {path}")
    return path


def required_resource_paths(root: pathlib.Path, dataset: str) -> Dict[str, pathlib.Path]:
    root = baseline_root(root)
    dataset = canonical_dataset_name(dataset)
    slug = RESOURCE_BASENAME[dataset]
    return {
        "data_root": resolve_local(root, "data", "data_root"),
        "cache_root": resolve_local(root, "data/cache", "cache_root"),
        "cache_dir": resolve_local(root, pathlib.Path("data/cache") / slug, "dataset cache"),
        # Historical ``*_gat`` bundles contain forbidden hidden biases.  The
        # formal chain uses a separately generated canonical no-bias bundle.
        "checkpoint_bundle": resolve_local(root, pathlib.Path("artifacts/checkpoints") / f"{slug}_gat_canonical", "checkpoint_bundle"),
        "static_table_path": resolve_local(root, STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH, "static_table_path"),
        "public_y_path": resolve_local(root, pathlib.Path("resources/public_y/canonical") / f"{slug}.json", "public_y_path"),
        "resource_manifest": resolve_local(root, "resources/resource_manifest.json", "resource_manifest"),
        "model_registry_root": resolve_local(root, MODEL_REGISTRY_ROOT_RELATIVE_PATH, "model_registry_root"),
        "private_model_setup_root": resolve_local(root, PRIVATE_MODEL_SETUP_ROOT_RELATIVE_PATH, "private_model_setup_root"),
    }


def resource_manifest_payload(root: pathlib.Path) -> Dict[str, Any]:
    root = baseline_root(root)
    static_descriptor = validate_static_table_descriptor(root)
    datasets: Dict[str, Any] = {}
    for dataset in DATASET_NAMES:
        paths = required_resource_paths(root, dataset)
        datasets[dataset] = {
            "cache_dir": str(paths["cache_dir"].relative_to(root)),
            "required_files": ([
                "meta.cfg",
                "edges.txt",
                "features.txt or features.npy",
                "node_ptr.txt",
                "labels_multi.npy or labels_multi.txt",
                "train_mask.npy",
                "val_mask.npy",
                "test_mask.npy",
            ] if dataset == "ppi" else [
                "meta.cfg",
                "edges.txt",
                "features.txt or features.npy",
                "labels.txt",
            ]),
            "checkpoint_bundle": str(paths["checkpoint_bundle"].relative_to(root)),
            "checkpoint_required_files": ["manifest.json", "tensors.npz", "tensors.txt"],
            "public_y": str(paths["public_y_path"].relative_to(root)),
            "status": "runtime_generated_from_baseline_local_inputs",
        }
    return {
        "schema": RESOURCE_MANIFEST_SCHEMA,
        "baseline": "zkGPT",
        "root_policy": "all_project_resources_must_be_baseline_local",
            "dataset_execution_policy": "full_datasets_are_baseline_local_inputs; expensive materialization occurs only in the formal runner",
        "model_registry": {
            "root": MODEL_REGISTRY_ROOT_RELATIVE_PATH,
            "private_root": PRIVATE_MODEL_SETUP_ROOT_RELATIVE_PATH,
            "public_name": "registry.json",
            "secret_name": "registry.secret.json",
            "registration_policy": "one canonical public registry per model_param_id; private blindings stay under private_root",
        },
        "static_table": {
            "id": "tables:lrelu+elu+exp+range",
            "quant_cfg_id": CANONICAL_QUANT_CFG_ID,
            "path": STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH,
            "schema": STATIC_TABLE_DESCRIPTOR_SCHEMA,
            "content_digest": static_descriptor["content_digest"],
            "descriptor_digest": static_descriptor["descriptor_digest"],
        },
        "datasets": datasets,
    }


def read_resource_manifest(root: pathlib.Path) -> Dict[str, Any]:
    root = baseline_root(root)
    path = _owned_regular(
        resolve_local(root, "resources/resource_manifest.json", "resource manifest"),
        "resource manifest",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read baseline resource manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != RESOURCE_MANIFEST_SCHEMA:
        raise ValueError("baseline resource manifest schema mismatch")
    if payload.get("baseline") != "zkGPT":
        raise ValueError("baseline resource manifest identity mismatch")
    if set(payload.get("datasets", {})) != set(DATASET_NAMES):
        raise ValueError("baseline resource manifest dataset coverage mismatch")
    static = payload.get("static_table")
    descriptor = validate_static_table_descriptor(root)
    if (
        not isinstance(static, dict)
        or static.get("id") != "tables:lrelu+elu+exp+range"
        or static.get("quant_cfg_id") != CANONICAL_QUANT_CFG_ID
        or static.get("path") != STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH
        or static.get("schema") != STATIC_TABLE_DESCRIPTOR_SCHEMA
        or static.get("content_digest") != descriptor["content_digest"]
        or static.get("descriptor_digest") != descriptor["descriptor_digest"]
    ):
        raise ValueError("baseline resource manifest static-table identity/path mismatch")
    if payload.get("model_registry") != {
        "root": MODEL_REGISTRY_ROOT_RELATIVE_PATH,
        "private_root": PRIVATE_MODEL_SETUP_ROOT_RELATIVE_PATH,
        "public_name": "registry.json",
        "secret_name": "registry.secret.json",
        "registration_policy": "one canonical public registry per model_param_id; private blindings stay under private_root",
    }:
        raise ValueError("baseline resource manifest model-registry policy mismatch")
    for dataset in DATASET_NAMES:
        entry = payload["datasets"].get(dataset)
        paths = required_resource_paths(root, dataset)
        expected = {
            "cache_dir": str(paths["cache_dir"].relative_to(root)),
            "checkpoint_bundle": str(paths["checkpoint_bundle"].relative_to(root)),
            "public_y": str(paths["public_y_path"].relative_to(root)),
        }
        if not isinstance(entry, dict) or any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError(f"baseline resource manifest path mismatch: {dataset}")
        expected_required_files = ([
            "meta.cfg",
            "edges.txt",
            "features.txt or features.npy",
            "node_ptr.txt",
            "labels_multi.npy or labels_multi.txt",
            "train_mask.npy",
            "val_mask.npy",
            "test_mask.npy",
        ] if dataset == "ppi" else [
            "meta.cfg",
            "edges.txt",
            "features.txt or features.npy",
            "labels.txt",
        ])
        if entry.get("required_files") != expected_required_files:
            raise ValueError(f"baseline resource manifest file requirements mismatch: {dataset}")
        if entry.get("checkpoint_required_files") != ["manifest.json", "tensors.npz", "tensors.txt"]:
            raise ValueError(f"baseline resource manifest checkpoint requirements mismatch: {dataset}")
    return payload


def validate_static_table_descriptor(root: pathlib.Path) -> Dict[str, Any]:
    root = baseline_root(root)
    path = _owned_regular(
        resolve_local(root, STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH, "static table"),
        "static table descriptor",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read static-table descriptor: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != STATIC_TABLE_DESCRIPTOR_SCHEMA:
        raise ValueError("static-table descriptor schema mismatch")
    if payload.get("static_tables_id") != "tables:lrelu+elu+exp+range" or payload.get("quant_cfg_id") != CANONICAL_QUANT_CFG_ID:
        raise ValueError("static-table descriptor identity mismatch")
    if tuple(payload.get("families", ())) != STATIC_TABLE_FAMILIES:
        raise ValueError("static-table descriptor family order mismatch")
    if payload.get("content") != STATIC_TABLE_CONTENT:
        raise ValueError("static-table descriptor content mismatch")
    if payload.get("content_digest") != _digest_json(STATIC_TABLE_CONTENT):
        raise ValueError("static-table descriptor content digest mismatch")
    descriptor = dict(payload)
    digest = descriptor.pop("descriptor_digest", None)
    if digest != _digest_json(descriptor):
        raise ValueError("static-table descriptor digest mismatch")
    return payload


def _digest_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def check_resource_layout(
    root: pathlib.Path,
    datasets: Iterable[str],
    *,
    require_staged: bool = False,
    require_public_y: bool = True,
) -> Dict[str, Any]:
    """Check only paths and required file names, never dataset contents.

    ``require_staged`` is intended for the real runner.  It checks presence of
    the files needed by the native loader but does not open them; the proof
    path itself performs the authoritative content/schema checks.  Registration
    may set ``require_public_y=False`` because setup commits only the model
    weights and does not consume the terminal public output.  Formal preflight
    and proof execution retain the default strict public-Y requirement.
    """
    root = baseline_root(root)
    manifest = read_resource_manifest(root)
    checked: Dict[str, Any] = {}
    static_path = resolve_local(root, manifest.get("static_table", {}).get("path", ""), "static table")
    if require_staged and not static_path.is_file():
        raise FileNotFoundError(f"baseline-local static table is not staged: {static_path}")
    if require_staged:
        validate_static_table_descriptor(root)
    for raw_dataset in datasets:
        dataset = canonical_dataset_name(raw_dataset)
        paths = required_resource_paths(root, dataset)
        entry = manifest.get("datasets", {}).get(dataset)
        if not isinstance(entry, dict):
            raise ValueError(f"resource manifest has no dataset entry: {dataset}")
        if require_staged:
            missing = []
            cache = paths["cache_dir"]
            if not cache.is_dir() or cache.is_symlink():
                raise ValueError(f"dataset cache must be a real directory: {cache}")
            for name in ("meta.cfg", "edges.txt"):
                try:
                    _owned_regular(cache / name, f"dataset {dataset} {name}")
                except (FileNotFoundError, OSError, ValueError):
                    missing.append(str(cache / name))
            feature_paths = [cache / "features.txt", cache / "features.npy"]
            feature_present = False
            for feature_path in feature_paths:
                try:
                    _owned_regular(feature_path, f"dataset {dataset} features")
                except (FileNotFoundError, OSError, ValueError):
                    continue
                feature_present = True
                break
            if not feature_present:
                missing.append(f"{cache}/features.txt or features.npy")
            if dataset == "ppi":
                for name in ("node_ptr.txt", "train_mask.npy", "val_mask.npy", "test_mask.npy"):
                    try:
                        _owned_regular(cache / name, f"dataset {dataset} {name}")
                    except (FileNotFoundError, OSError, ValueError):
                        missing.append(str(cache / name))
                label_present = False
                for name in ("labels_multi.npy", "labels_multi.txt"):
                    try:
                        _owned_regular(cache / name, f"dataset {dataset} {name}")
                    except (FileNotFoundError, OSError, ValueError):
                        continue
                    label_present = True
                    break
                if not label_present:
                    missing.append(f"{cache}/labels_multi.npy or labels_multi.txt")
            else:
                try:
                    _owned_regular(cache / "labels.txt", f"dataset {dataset} labels.txt")
                except (FileNotFoundError, OSError, ValueError):
                    missing.append(str(cache / "labels.txt"))
            for name in ("manifest.json", "tensors.npz", "tensors.txt"):
                try:
                    _owned_regular(paths["checkpoint_bundle"] / name, f"checkpoint {name}")
                except (FileNotFoundError, OSError, ValueError):
                    missing.append(str(paths["checkpoint_bundle"] / name))
            if require_public_y:
                try:
                    _owned_regular(paths["public_y_path"], f"dataset {dataset} public_y")
                except (FileNotFoundError, OSError, ValueError):
                    missing.append(str(paths["public_y_path"]))
            if missing:
                raise FileNotFoundError("baseline-local resources are not staged: " + ", ".join(missing))
        checked[dataset] = {
            "cache_dir": str(paths["cache_dir"]),
            "checkpoint_bundle": str(paths["checkpoint_bundle"]),
            "public_y_path": str(paths["public_y_path"]),
            "staged": bool(
                paths["cache_dir"].is_dir()
                and paths["checkpoint_bundle"].is_dir()
                and (not require_public_y or paths["public_y_path"].is_file())
            ),
        }
    return {"schema": RESOURCE_MANIFEST_SCHEMA, "baseline": "zkGPT", "datasets": checked}
