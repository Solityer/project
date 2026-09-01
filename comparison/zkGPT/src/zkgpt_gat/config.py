from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List

from .constants import CONFIG_BY_DATASET, CONFIG_DATASET_VALUE
from .rounding import configured_rounding_mode
from .resources import (
    STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH,
    read_resource_manifest,
    resolve_local,
    required_resource_paths,
    validate_static_table_descriptor,
)
from .quantization import canonical_input_preprocess_id
from .util import assert_contained_path, require_dir, write_text_atomic


@dataclass(frozen=True)
class HiddenShape:
    head_count: int
    head_dim: int


@dataclass(frozen=True)
class AppConfig:
    entries: Dict[str, str]
    path: pathlib.Path
    project_root: pathlib.Path

    @property
    def dataset(self) -> str:
        return self.entries["dataset"]

    @property
    def data_root(self) -> pathlib.Path:
        return resolve_local(self.project_root, self.entries.get("data_root", "data"), "data_root")

    @property
    def cache_root(self) -> pathlib.Path:
        return resolve_local(self.project_root, self.entries.get("cache_root", "data/cache"), "cache_root")

    @property
    def cache_dir(self) -> pathlib.Path:
        path = (self.cache_root / self.dataset).resolve()
        if self.project_root not in path.parents and path != self.project_root:
            raise ValueError(f"dataset cache escapes baseline root: {path}")
        return path

    @property
    def checkpoint_bundle(self) -> pathlib.Path:
        return resolve_local(self.project_root, self.entries["checkpoint_bundle"], "checkpoint_bundle")

    @property
    def reference_output_dir(self) -> pathlib.Path:
        return resolve_local(self.project_root, self.entries.get("reference_output_dir", "runs/reference"), "reference_output_dir")

    @property
    def public_y_path(self) -> pathlib.Path:
        return resolve_local(self.project_root, self.entries.get("public_y_path", "resources/public_y"), "public_y_path")

    @property
    def static_table_path(self) -> pathlib.Path:
        return resolve_local(self.project_root, self.entries.get("static_table_path", STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH), "static_table_path")

    @property
    def static_table_descriptor_digest(self) -> str:
        return self.entries["static_table_descriptor_digest"]

    @property
    def range_bits(self) -> int:
        return int(self.entries.get("range_bits", "12"))

    @property
    def batch_graphs(self) -> int:
        return int(self.entries.get("batch_graphs", "1"))

    @property
    def hidden_profile(self) -> List[HiddenShape]:
        profile = parse_hidden_profile(self.entries.get("hidden_profile", ""))
        if not profile:
            raise ValueError("formal config hidden_profile must contain at least one hidden layer")
        if any(item.head_count <= 0 or item.head_dim <= 0 for item in profile):
            raise ValueError("formal config hidden_profile dimensions must be positive")
        return profile

    @property
    def d_in_profile(self) -> List[int]:
        profile = parse_int_list(self.entries.get("d_in_profile", ""))
        hidden = self.hidden_profile
        if len(profile) != len(hidden):
            raise ValueError(
                "formal config d_in_profile must provide one input dimension per hidden layer"
            )
        if any(value <= 0 for value in profile):
            raise ValueError("formal config d_in_profile dimensions must be positive")
        expected = [hidden[index - 1].head_count * hidden[index - 1].head_dim for index in range(1, len(hidden))]
        if profile[1:] != expected:
            raise ValueError("formal config d_in_profile does not follow hidden concat dimensions")
        return profile

    @property
    def layer_count(self) -> int:
        value = int(self.entries.get("L", str(len(self.hidden_profile) + 1)))
        if value != len(self.hidden_profile) + 1 or value < 2:
            raise ValueError("formal config L must equal hidden layer count plus output layer")
        return value

    @property
    def k_out(self) -> int:
        value = int(self.entries.get("K_out", "0"))
        if value <= 0:
            raise ValueError("formal config K_out must be positive")
        return value

    @property
    def num_classes(self) -> int:
        value = int(self.entries.get("num_classes", "0"))
        if value <= 0:
            raise ValueError("formal config num_classes must be positive")
        return value

    @property
    def rounding_mode(self) -> str:
        """Authenticated SignedRoundDiv mode; missing values fail closed."""
        return configured_rounding_mode(self.entries)

    @property
    def input_preprocess_id(self) -> str:
        # The production loader accepts only the five registered datasets.
        # Keep a clearly test-scoped identity for the in-memory synthetic
        # fixture so existing unit tests do not acquire a production fallback.
        if self.dataset.startswith("synthetic-"):
            expected = "synthetic_fixture_v1"
        else:
            expected = canonical_input_preprocess_id(self.dataset)
        value = str(self.entries.get("input_preprocess_id", ""))
        if self.dataset.startswith("synthetic-") and not value:
            return expected
        if expected is None or value != expected:
            raise ValueError(
                f"formal config input_preprocess_id mismatch: {value!r} != {expected!r}"
            )
        return value

    @property
    def symmetrize_edges(self) -> bool:
        return self.entries.get("symmetrize_edges", "false").lower() == "true"

    @property
    def deduplicate_edges(self) -> bool:
        return self.entries.get("deduplicate_edges", "true").lower() != "false"


def read_config(path: pathlib.Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"invalid config line in {path}: {raw}")
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def write_config(
    path: pathlib.Path,
    entries: Dict[str, str],
    *,
    root: pathlib.Path | None = None,
) -> None:
    order = [
        "dataset",
        "resource_root",
        "data_root",
        "cache_root",
        "checkpoint_bundle",
        "reference_output_dir",
        "public_y_path",
        "static_table_path",
        "static_table_descriptor_path",
        "static_table_descriptor_digest",
        "task_type",
        "report_unit",
        "batching_rule",
        "self_loop_rule",
        "edge_sort_rule",
        "chunking_rule",
        "L",
        "hidden_profile",
        "d_in_profile",
        "K_out",
        "batch_graphs",
        "num_classes",
        "range_bits",
        "seed",
        "allow_synthetic_model",
        "dump_trace",
        "auto_prepare_dataset",
        "model_arch_id",
        "model_param_id",
        "quant_cfg_id",
        "input_preprocess_id",
        "rounding_mode",
        "static_table_id",
        "static_tables_id",
        "edge_rule_id",
        "resource_manifest",
        "degree_bound_id",
        "symmetrize_edges",
        "deduplicate_edges",
    ]
    lines = ["# Generated by the independent zkGPT baseline runner (baseline-local)."]
    for key in order:
        if key in entries:
            lines.append(f"{key} = {entries[key]}")
    for key in sorted(entries):
        if key not in order:
            lines.append(f"{key} = {entries[key]}")
    # Generated configs are production inputs to both the prover and the
    # independent verifier.  Publish them through the same authenticated,
    # atomic writer as proof/result artifacts.  ``root`` is supplied by the
    # formal generator so a malicious path cannot redirect config creation
    # outside the baseline.
    if root is not None:
        path = assert_contained_path(root, path, "generated config", allow_missing=True)
    write_text_atomic(path, "\n".join(lines) + "\n", label="generated config", root=root)


def parse_hidden_profile(value: str) -> List[HiddenShape]:
    out: List[HiddenShape] = []
    for token in re.sub(r"[\[\]\s]", "", value).split(","):
        if not token:
            continue
        left, right = re.split("[xX]", token, maxsplit=1)
        out.append(HiddenShape(int(left), int(right)))
    return out


def parse_int_list(value: str) -> List[int]:
    text = re.sub(r"[\[\]\s]", "", str(value))
    return [int(part) for part in text.split(",") if part]


def generated_config(dataset: str, backend: str, project_root: pathlib.Path, zk_root: pathlib.Path) -> pathlib.Path:
    # Formal comparison configurations are immutable baseline-local templates.
    # Runtime must not import the main experiment's configuration files.
    project_root = require_dir(project_root, label="formal project root")
    zk_root = require_dir(zk_root, label="formal zkGPT root")
    if project_root != zk_root:
        raise ValueError(
            "formal zkGPT execution is baseline-local; project_root must equal comparison/zkGPT"
        )
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"unsupported backend: {backend}")
    source = assert_contained_path(
        zk_root,
        zk_root / "configs" / "formal" / CONFIG_BY_DATASET[dataset],
        "formal config template",
        allow_missing=False,
    )
    try:
        source_info = source.lstat()
    except OSError as exc:
        raise FileNotFoundError(f"cannot inspect baseline-local formal config: {source}") from exc
    if not source_info or not source.is_file() or source_info.st_nlink != 1:
        raise FileNotFoundError(f"missing baseline-local formal config: {source}")
    entries = read_config(source)
    entries["dataset"] = CONFIG_DATASET_VALUE[dataset]
    # Keep all project-level paths relative to the baseline.  The loader
    # resolves and re-checks them against ``project_root`` before use.
    paths = required_resource_paths(zk_root, dataset)
    entries["resource_root"] = "resources"
    entries["data_root"] = "data"
    entries["cache_root"] = "data/cache"
    entries["checkpoint_bundle"] = str(paths["checkpoint_bundle"].relative_to(zk_root))
    entries["static_table_path"] = str(paths["static_table_path"].relative_to(zk_root))
    descriptor = validate_static_table_descriptor(zk_root)
    entries["static_table_descriptor_path"] = STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH
    entries["static_table_descriptor_digest"] = descriptor["descriptor_digest"]
    entries["public_y_path"] = str(paths["public_y_path"].relative_to(zk_root))
    entries["reference_output_dir"] = str((zk_root / "runs" / f"{dataset}_full" / "reference").relative_to(zk_root))
    entries["resource_manifest"] = "resources/resource_manifest.json"
    entries["input_preprocess_id"] = canonical_input_preprocess_id(entries["dataset"])
    entries["static_tables_id"] = entries.get("static_table_id", "tables:lrelu+elu+exp+range")
    # The immutable formal template must carry an explicit mode.  Do not
    # infer a mode from backend or from a proof artifact.
    entries["rounding_mode"] = configured_rounding_mode(entries)
    entries["edge_rule_id"] = entries.get("edge_rule_id", "per_node_self_loop+edge_gid_then_dst_stable")
    entries["auto_prepare_dataset"] = "false"
    out = assert_contained_path(
        zk_root,
        zk_root / "configs" / "generated" / backend / CONFIG_BY_DATASET[dataset],
        "generated config",
        allow_missing=True,
    )
    write_config(out, entries, root=zk_root)
    return out


def load_app_config(path: pathlib.Path, project_root: pathlib.Path) -> AppConfig:
    root = require_dir(project_root, label="config baseline root")
    config_path = assert_contained_path(root, path, "config", allow_missing=False)
    try:
        info = config_path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect config: {config_path}") from exc
    if not config_path.is_file() or info.st_nlink != 1:
        raise ValueError(f"config must be an owned regular file: {config_path}")
    entries = read_config(config_path)
    if "hidden_dim" in entries:
        raise ValueError("legacy hidden_dim is not supported; use hidden_profile")
    dataset = entries.get("dataset", "")
    if dataset not in CONFIG_DATASET_VALUE.values():
        raise ValueError(f"unsupported config dataset: {dataset}")
    canonical_resources = required_resource_paths(root, dataset)
    # Resolve every path-bearing field now so a malformed config fails before
    # any dataset/model loader can open an external project resource.
    for key, default in (
        ("resource_root", "resources"),
        ("data_root", "data"),
        ("cache_root", "data/cache"),
        ("checkpoint_bundle", str(canonical_resources["checkpoint_bundle"].relative_to(root))),
        ("reference_output_dir", "runs/reference"),
        ("public_y_path", str(canonical_resources["public_y_path"].relative_to(root))),
        ("static_table_path", STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH),
        ("static_table_descriptor_path", STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH),
        ("resource_manifest", "resources/resource_manifest.json"),
    ):
        resolve_local(root, entries.get(key, default), key)
    descriptor = validate_static_table_descriptor(root)
    manifest = read_resource_manifest(root)
    expected_descriptor_path = resolve_local(root, STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH, "static_table_descriptor_path")
    configured_descriptor_path = resolve_local(root, entries.get("static_table_descriptor_path", ""), "static_table_descriptor_path")
    if configured_descriptor_path != expected_descriptor_path:
        raise ValueError("static_table_descriptor_path must be the canonical baseline-local descriptor")
    if entries.get("static_table_descriptor_digest") != descriptor["descriptor_digest"]:
        raise ValueError("static_table_descriptor_digest does not match canonical descriptor")
    configured_static_path = resolve_local(root, entries.get("static_table_path", ""), "static_table_path")
    if configured_static_path != expected_descriptor_path:
        raise ValueError("static_table_path must be the canonical baseline-local descriptor")
    if entries.get("static_tables_id", entries.get("static_table_id", "")) != entries.get("static_table_id", ""):
        raise ValueError("static_tables_id and static_table_id must match")
    if entries.get("static_table_id") != descriptor["static_tables_id"] or entries.get("quant_cfg_id") != descriptor["quant_cfg_id"]:
        raise ValueError("formal config static-table identity does not match canonical descriptor")
    _ = AppConfig(entries=entries, path=config_path, project_root=root).input_preprocess_id
    if manifest.get("static_table", {}).get("descriptor_digest") != descriptor["descriptor_digest"]:
        raise ValueError("resource manifest is not bound to canonical static-table descriptor")
    if not entries.get("model_arch_id") or not entries.get("model_param_id") or not entries.get("quant_cfg_id") or not entries.get("static_table_id"):
        raise ValueError("formal config is missing model/static-table identity")
    # Force structural validation before any dataset/checkpoint execution.
    # The formal schema stores one d_in entry per hidden layer (the internal
    # proof dimensions append each layer's concatenated width separately).
    _ = AppConfig(entries=entries, path=config_path, project_root=root).hidden_profile
    _ = AppConfig(entries=entries, path=config_path, project_root=root).d_in_profile
    _ = AppConfig(entries=entries, path=config_path, project_root=root).layer_count
    _ = AppConfig(entries=entries, path=config_path, project_root=root).k_out
    _ = AppConfig(entries=entries, path=config_path, project_root=root).num_classes
    return AppConfig(entries=entries, path=config_path, project_root=root)
