#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pathlib
import zipfile
from typing import Dict, List, Mapping, Sequence

import numpy as np

from quantization_scale_chain import (
    CANONICAL_QUANT_CFG_ID,
    OGBN_ARXIV_INTERNAL_NAME,
    canonical_input_preprocess_id,
    canonical_quantization_scale_chain,
)


# The JSON manifest is emitted by ``build_family_manifest`` and the PPI
# importer.  Keep this list synchronized with the C++ checkpoint reader: a
# manifest is one canonical object, so an unrecognized top-level key is never
# silently ignored.  Dataset-specific fields are part of the same format and
# are allowed for every dataset; their semantic validators decide when they
# are required.
CHECKPOINT_MANIFEST_FIELDS = frozenset(
    {
        "checkpoint_prefix",
        "dataset",
        "family_schema",
        "output_average_rule",
        "L",
        "d_in_profile",
        "hidden_profile",
        "hidden_layers",
        "hidden_head_specs",
        "K_out",
        "output_head_specs",
        "C",
        "task_type",
        "report_unit",
        "selection_metric",
        "tensor_count",
        "model_arch_id",
        "model_param_id",
        "input_preprocess_id",
        "quant_cfg_id",
        "quantization_scale_chain",
        "static_table_id",
        "degree_bound_id",
        "batching_rule",
        "subgraph_rule",
        "self_loop_rule",
        "edge_sort_rule",
        "label_format",
        "loss",
        "split_format",
        "tensor_index",
    }
)

CHECKPOINT_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "checkpoint_prefix",
        "dataset",
        "family_schema",
        "output_average_rule",
        "L",
        "d_in_profile",
        "hidden_profile",
        "hidden_layers",
        "hidden_head_specs",
        "K_out",
        "output_head_specs",
        "C",
        "task_type",
        "report_unit",
        "selection_metric",
        "tensor_count",
        "model_arch_id",
        "model_param_id",
        "input_preprocess_id",
        "quant_cfg_id",
        "quantization_scale_chain",
        "static_table_id",
        "degree_bound_id",
        "batching_rule",
        "subgraph_rule",
        "self_loop_rule",
        "edge_sort_rule",
        "label_format",
        "loss",
        "split_format",
        "tensor_index",
    }
)

# The torch checkpoint carries training provenance in addition to the fields
# copied into the JSON manifest.  It has its own explicit canonical format;
# keeping it strict prevents an ignored metadata alias from entering export or
# replay through the Python path.
CHECKPOINT_METADATA_FIELDS = frozenset(
    {
        "dataset",
        "family_schema",
        "L",
        "d_in_profile",
        "hidden_profile",
        "K_out",
        "C",
        "num_features",
        "num_classes",
        "seed",
        "dropout",
        "learning_rate",
        "weight_decay",
        "task_type",
        "report_unit",
        "selection_metric",
        "model_arch_id",
        "model_param_id",
        "quant_cfg_id",
        "input_preprocess_id",
        "quantization_scale_chain",
        "static_table_id",
        "degree_bound_id",
        "output_average_rule",
        "batching_rule",
        "subgraph_rule",
        "self_loop_rule",
        "edge_sort_rule",
        "label_format",
        "loss",
        "split_format",
    }
)

CHECKPOINT_METADATA_REQUIRED_FIELDS = frozenset(
    {
        "dataset",
        "family_schema",
        "L",
        "d_in_profile",
        "hidden_profile",
        "K_out",
        "C",
        "num_features",
        "num_classes",
        "seed",
        "dropout",
        "learning_rate",
        "weight_decay",
        "task_type",
        "report_unit",
        "selection_metric",
        "model_arch_id",
        "model_param_id",
        "quant_cfg_id",
        "input_preprocess_id",
        "quantization_scale_chain",
        "static_table_id",
        "degree_bound_id",
        "output_average_rule",
        "batching_rule",
        "subgraph_rule",
        "self_loop_rule",
        "edge_sort_rule",
        "label_format",
        "loss",
        "split_format",
    }
)

_U64_MAX = (1 << 64) - 1

_HIDDEN_PROFILE_FIELDS = frozenset({"head_count", "head_dim"})
_HIDDEN_LAYER_FIELDS = frozenset(
    {"layer_index", "input_dim", "head_count", "head_dim"}
)
_HIDDEN_HEAD_FIELDS = frozenset(
    {
        "layer_index",
        "local_head_index",
        "seq_kernel",
        "attn_dst_kernel",
        "attn_src_kernel",
    }
)
_OUTPUT_HEAD_FIELDS = frozenset(
    {"head_index", "seq_kernel", "attn_dst_kernel", "attn_src_kernel", "output_bias"}
)
_TENSOR_INDEX_FIELDS = frozenset({"archive_key", "dtype", "shape"})


def archive_key(name: str) -> str:
    """Encode one logical tensor name using the canonical NPZ key rule.

    The returned value is only an NPZ archive encoding key.  The logical name
    remains the identity used by the manifest and text dump, and the two
    encodings are cross-checked by ``validate_tensor_encodings``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("logical tensor name must be a non-empty string")
    return name.replace("/", "__").replace(":", "__")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build JSON objects without silently applying last-key-wins semantics."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not accepted: {value}")


def load_checkpoint_manifest(path: pathlib.Path) -> Dict[str, object]:
    """Load and validate the one canonical checkpoint manifest format.

    ``object_pairs_hook`` is recursive, so duplicate keys are rejected at
    every nesting level before any semantic consumer sees the object.
    """
    manifest_path = pathlib.Path(path)
    try:
        value = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load canonical checkpoint manifest {manifest_path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("checkpoint manifest must be an object")
    validate_checkpoint_manifest(value)
    return value


def _require_string(value: object, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_unsigned(value: object, label: str, *, positive: bool = False) -> int:
    # ``bool`` is an ``int`` subclass, but is never a valid manifest integer.
    if type(value) is not int or value < 0 or value > _U64_MAX:
        raise ValueError(f"{label} must be an unsigned integer")
    if positive and value == 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_float(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be a finite number")
    return result


def _require_exact_object(
    value: object,
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _validate_field_set(value, allowed, required, label)
    return value


def _require_unsigned_array(
    value: object,
    label: str,
    *,
    nonempty: bool = True,
    positive: bool = False,
) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    if nonempty and not value:
        raise ValueError(f"{label} must be a non-empty array")
    return [
        _require_unsigned(element, f"{label}[{index}]", positive=positive)
        for index, element in enumerate(value)
    ]


def _validate_profile_and_dimension_chain(
    *,
    layer_count: int,
    d_in_profile: list[int],
    hidden_profile: list[dict[str, object]],
    hidden_layers: list[dict[str, object]] | None,
    output_head_count: int,
    class_count: int,
    label: str,
) -> None:
    if layer_count < 2:
        raise ValueError(f"{label}.L must be at least 2")
    if layer_count != len(hidden_profile) + 1:
        raise ValueError(f"{label}.L conflicts with hidden_profile length")
    if len(d_in_profile) != len(hidden_profile):
        raise ValueError(f"{label}.d_in_profile conflicts with hidden_profile length")
    if output_head_count == 0:
        raise ValueError(f"{label}.K_out must be positive")
    if class_count == 0:
        raise ValueError(f"{label}.C must be positive")

    previous_concat = None
    for layer_index, profile_entry in enumerate(hidden_profile):
        head_count = _require_unsigned(
            profile_entry["head_count"],
            f"{label}.hidden_profile[{layer_index}].head_count",
            positive=True,
        )
        head_dim = _require_unsigned(
            profile_entry["head_dim"],
            f"{label}.hidden_profile[{layer_index}].head_dim",
            positive=True,
        )
        input_dim = d_in_profile[layer_index]
        if input_dim == 0:
            raise ValueError(f"{label}.d_in_profile[{layer_index}] must be positive")
        if previous_concat is not None and input_dim != previous_concat:
            raise ValueError(
                f"{label}.d_in_profile[{layer_index}] conflicts with previous concat width"
            )
        previous_concat = head_count * head_dim

    if layer_count != 3 or output_head_count != 2 or [
        {
            "head_count": profile["head_count"],
            "head_dim": profile["head_dim"],
        }
        for profile in hidden_profile
    ] != [
        {"head_count": 8, "head_dim": 8},
        {"head_count": 8, "head_dim": 8},
    ]:
        raise ValueError(f"{label} must use the canonical L=3, hidden 8x8/8x8, K_out=2 shape")

    if hidden_layers is not None:
        if len(hidden_layers) != len(hidden_profile):
            raise ValueError(f"{label}.hidden_layers conflicts with hidden_profile length")
        for layer_index, layer in enumerate(hidden_layers):
            if _require_unsigned(
                layer["layer_index"], f"{label}.hidden_layers[{layer_index}].layer_index"
            ) != layer_index:
                raise ValueError(f"{label}.hidden_layers must use contiguous layer_index values")
            if _require_unsigned(
                layer["input_dim"], f"{label}.hidden_layers[{layer_index}].input_dim", positive=True
            ) != d_in_profile[layer_index]:
                raise ValueError(f"{label}.hidden_layers[{layer_index}].input_dim conflicts with d_in_profile")
            if _require_unsigned(
                layer["head_count"], f"{label}.hidden_layers[{layer_index}].head_count", positive=True
            ) != _require_unsigned(
                hidden_profile[layer_index]["head_count"],
                f"{label}.hidden_profile[{layer_index}].head_count",
                positive=True,
            ) or _require_unsigned(
                layer["head_dim"], f"{label}.hidden_layers[{layer_index}].head_dim", positive=True
            ) != _require_unsigned(
                hidden_profile[layer_index]["head_dim"],
                f"{label}.hidden_profile[{layer_index}].head_dim",
                positive=True,
            ):
                raise ValueError(f"{label}.hidden_layers[{layer_index}] conflicts with hidden_profile")


def _validate_tensor_index(
    value: object,
    label: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    result: dict[str, dict[str, object]] = {}
    archive_keys: set[str] = set()
    for name, raw_entry in value.items():
        _require_string(name, f"{label} logical tensor name")
        entry = _require_exact_object(
            raw_entry,
            _TENSOR_INDEX_FIELDS,
            _TENSOR_INDEX_FIELDS,
            f"{label}[{name!r}]",
        )
        archive_key_value = _require_string(entry["archive_key"], f"{label}[{name!r}].archive_key")
        expected_archive_key = archive_key(name)
        if archive_key_value != expected_archive_key:
            raise ValueError(
                f"{label}[{name!r}].archive_key is not the canonical NPZ encoding key"
            )
        if archive_key_value in archive_keys:
            raise ValueError(f"{label} archive_key is not unique: {archive_key_value}")
        archive_keys.add(archive_key_value)
        dtype = _require_string(entry["dtype"], f"{label}[{name!r}].dtype")
        if dtype != "float32":
            raise ValueError(f"{label}[{name!r}].dtype must be float32")
        shape = _require_unsigned_array(
            entry["shape"], f"{label}[{name!r}].shape", positive=True
        )
        result[name] = {"archive_key": archive_key_value, "dtype": dtype, "shape": shape}
    return result


def _tensor_shape(
    tensor_index: dict[str, dict[str, object]],
    name: object,
    expected: tuple[int, ...],
    label: str,
) -> None:
    tensor_name = _require_string(name, f"{label} tensor reference")
    if tensor_name not in tensor_index:
        raise ValueError(f"{label} references missing tensor: {tensor_name}")
    shape = tensor_index[tensor_name]["shape"]
    if tuple(shape) != expected:
        raise ValueError(f"{label} tensor shape mismatch for {tensor_name}")


def _validate_head_specs_and_tensor_shapes(
    *,
    tensor_index: dict[str, dict[str, object]],
    hidden_layers: list[dict[str, object]],
    hidden_head_specs: list[dict[str, object]],
    output_head_specs: list[dict[str, object]],
    class_count: int,
    label: str,
) -> None:
    expected_pairs = [
        (layer_index, local_head_index)
        for layer_index, layer in enumerate(hidden_layers)
        for local_head_index in range(
            _require_unsigned(layer["head_count"], f"{label}.hidden_layers[{layer_index}].head_count", positive=True)
        )
    ]
    actual_pairs = [
        (
            _require_unsigned(spec["layer_index"], f"{label}.hidden_head_specs.layer_index"),
            _require_unsigned(spec["local_head_index"], f"{label}.hidden_head_specs.local_head_index"),
        )
        for spec in hidden_head_specs
    ]
    if actual_pairs != expected_pairs:
        raise ValueError(f"{label}.hidden_head_specs must use canonical layer/local order")

    referenced: list[str] = []
    for spec_index, spec in enumerate(hidden_head_specs):
        layer_index, local_head_index = actual_pairs[spec_index]
        layer = hidden_layers[layer_index]
        input_dim = _require_unsigned(layer["input_dim"], f"{label}.hidden_layers[{layer_index}].input_dim", positive=True)
        output_dim = _require_unsigned(layer["head_dim"], f"{label}.hidden_layers[{layer_index}].head_dim", positive=True)
        for field, shape in (
            ("seq_kernel", (1, input_dim, output_dim)),
            ("attn_dst_kernel", (1, output_dim, 1)),
            ("attn_src_kernel", (1, output_dim, 1)),
        ):
            name = _require_string(spec[field], f"{label}.hidden_head_specs[{spec_index}].{field}")
            referenced.append(name)
            _tensor_shape(tensor_index, name, shape, f"{label}.hidden_head_specs[{spec_index}].{field}")

    output_input_dim = _require_unsigned(hidden_layers[-1]["head_count"], f"{label}.hidden_layers[-1].head_count", positive=True) * _require_unsigned(hidden_layers[-1]["head_dim"], f"{label}.hidden_layers[-1].head_dim", positive=True)
    expected_output_indices = list(range(len(output_head_specs)))
    actual_output_indices = [
        _require_unsigned(spec["head_index"], f"{label}.output_head_specs.head_index")
        for spec in output_head_specs
    ]
    if actual_output_indices != expected_output_indices:
        raise ValueError(f"{label}.output_head_specs must use contiguous head_index values")
    for spec_index, spec in enumerate(output_head_specs):
        for field, shape in (
            ("seq_kernel", (1, output_input_dim, class_count)),
            ("attn_dst_kernel", (1, class_count, 1)),
            ("attn_src_kernel", (1, class_count, 1)),
            ("output_bias", (class_count,)),
        ):
            name = _require_string(spec[field], f"{label}.output_head_specs[{spec_index}].{field}")
            referenced.append(name)
            _tensor_shape(tensor_index, name, shape, f"{label}.output_head_specs[{spec_index}].{field}")

    if len(referenced) != len(set(referenced)):
        raise ValueError(f"{label} references the same tensor more than once")
    if set(referenced) != set(tensor_index):
        extras = sorted(set(tensor_index).difference(referenced))
        missing = sorted(set(referenced).difference(tensor_index))
        detail = extras[0] if extras else missing[0]
        raise ValueError(f"{label}.tensor_index does not match referenced tensors: {detail}")


def _validate_manifest_types_and_shape(manifest: Mapping[str, object]) -> None:
    """Validate the complete nested manifest schema and all shape identities."""
    for field in (
        "checkpoint_prefix", "dataset", "family_schema", "output_average_rule",
        "task_type", "report_unit", "selection_metric", "model_arch_id",
        "model_param_id", "input_preprocess_id", "quant_cfg_id",
        "quantization_scale_chain", "static_table_id", "degree_bound_id",
        "batching_rule", "subgraph_rule", "self_loop_rule", "edge_sort_rule",
        "label_format", "loss", "split_format",
    ):
        _require_string(manifest[field], f"checkpoint manifest.{field}")
    layer_count = _require_unsigned(manifest["L"], "checkpoint manifest.L", positive=True)
    output_head_count = _require_unsigned(manifest["K_out"], "checkpoint manifest.K_out", positive=True)
    class_count = _require_unsigned(manifest["C"], "checkpoint manifest.C", positive=True)
    tensor_count = _require_unsigned(manifest["tensor_count"], "checkpoint manifest.tensor_count")
    d_in_profile = _require_unsigned_array(
        manifest["d_in_profile"], "checkpoint manifest.d_in_profile", positive=True
    )

    hidden_profile_raw = manifest["hidden_profile"]
    if not isinstance(hidden_profile_raw, list) or not hidden_profile_raw:
        raise ValueError("checkpoint manifest.hidden_profile must be a non-empty array")
    hidden_profile: list[dict[str, object]] = []
    for index, raw_entry in enumerate(hidden_profile_raw):
        entry = _require_exact_object(
            raw_entry,
            _HIDDEN_PROFILE_FIELDS,
            _HIDDEN_PROFILE_FIELDS,
            f"checkpoint manifest.hidden_profile[{index}]",
        )
        _require_unsigned(entry["head_count"], f"checkpoint manifest.hidden_profile[{index}].head_count", positive=True)
        _require_unsigned(entry["head_dim"], f"checkpoint manifest.hidden_profile[{index}].head_dim", positive=True)
        hidden_profile.append(entry)

    hidden_layers_raw = manifest["hidden_layers"]
    if not isinstance(hidden_layers_raw, list) or not hidden_layers_raw:
        raise ValueError("checkpoint manifest.hidden_layers must be a non-empty array")
    hidden_layers: list[dict[str, object]] = []
    for index, raw_entry in enumerate(hidden_layers_raw):
        entry = _require_exact_object(
            raw_entry,
            _HIDDEN_LAYER_FIELDS,
            _HIDDEN_LAYER_FIELDS,
            f"checkpoint manifest.hidden_layers[{index}]",
        )
        for field in ("layer_index", "input_dim", "head_count", "head_dim"):
            _require_unsigned(
                entry[field],
                f"checkpoint manifest.hidden_layers[{index}].{field}",
                positive=field != "layer_index",
            )
        hidden_layers.append(entry)

    hidden_head_specs_raw = manifest["hidden_head_specs"]
    if not isinstance(hidden_head_specs_raw, list):
        raise ValueError("checkpoint manifest.hidden_head_specs must be an array")
    hidden_head_specs: list[dict[str, object]] = []
    for index, raw_entry in enumerate(hidden_head_specs_raw):
        entry = _require_exact_object(
            raw_entry,
            _HIDDEN_HEAD_FIELDS,
            _HIDDEN_HEAD_FIELDS,
            f"checkpoint manifest.hidden_head_specs[{index}]",
        )
        for field in ("layer_index", "local_head_index"):
            _require_unsigned(entry[field], f"checkpoint manifest.hidden_head_specs[{index}].{field}")
        for field in ("seq_kernel", "attn_dst_kernel", "attn_src_kernel"):
            _require_string(entry[field], f"checkpoint manifest.hidden_head_specs[{index}].{field}")
        hidden_head_specs.append(entry)

    output_head_specs_raw = manifest["output_head_specs"]
    if not isinstance(output_head_specs_raw, list) or not output_head_specs_raw:
        raise ValueError("checkpoint manifest.output_head_specs must be a non-empty array")
    output_head_specs: list[dict[str, object]] = []
    for index, raw_entry in enumerate(output_head_specs_raw):
        entry = _require_exact_object(
            raw_entry,
            _OUTPUT_HEAD_FIELDS,
            _OUTPUT_HEAD_FIELDS,
            f"checkpoint manifest.output_head_specs[{index}]",
        )
        _require_unsigned(entry["head_index"], f"checkpoint manifest.output_head_specs[{index}].head_index")
        for field in ("seq_kernel", "attn_dst_kernel", "attn_src_kernel", "output_bias"):
            _require_string(entry[field], f"checkpoint manifest.output_head_specs[{index}].{field}")
        output_head_specs.append(entry)

    tensor_index = _validate_tensor_index(manifest["tensor_index"], "checkpoint manifest.tensor_index")
    if tensor_count != len(tensor_index):
        raise ValueError("checkpoint manifest.tensor_count conflicts with tensor_index")
    if output_head_count != len(output_head_specs):
        raise ValueError("checkpoint manifest.K_out conflicts with output_head_specs")
    _validate_profile_and_dimension_chain(
        layer_count=layer_count,
        d_in_profile=d_in_profile,
        hidden_profile=hidden_profile,
        hidden_layers=hidden_layers,
        output_head_count=output_head_count,
        class_count=class_count,
        label="checkpoint manifest",
    )
    if len(hidden_head_specs) != sum(
        _require_unsigned(layer["head_count"], "checkpoint manifest.hidden_layers.head_count", positive=True)
        for layer in hidden_layers
    ):
        raise ValueError("checkpoint manifest.hidden_head_specs count conflicts with hidden_layers")
    _validate_head_specs_and_tensor_shapes(
        tensor_index=tensor_index,
        hidden_layers=hidden_layers,
        hidden_head_specs=hidden_head_specs,
        output_head_specs=output_head_specs,
        class_count=class_count,
        label="checkpoint manifest",
    )


def canonical_checkpoint_semantics(dataset: str) -> Dict[str, str]:
    """Return the one producer contract for the formal dataset split.

    These fields are part of the checkpoint manifest identity.  Keep their
    values in one place so the torch checkpoint metadata and exported JSON
    manifest cannot silently drift apart.
    """
    if dataset == "ppi":
        return {
            "batching_rule": "multi_graph_batch",
            "subgraph_rule": "whole_graph",
            "self_loop_rule": "per_node",
            "edge_sort_rule": "edge_gid_then_dst_stable",
            "label_format": "multi_hot",
            "loss": "BCEWithLogitsLoss",
            "split_format": "node_masks",
        }
    if dataset == OGBN_ARXIV_INTERNAL_NAME:
        split_format = "index_split"
    elif dataset in {"cora", "citeseer", "pubmed"}:
        split_format = "node_masks"
    else:
        raise ValueError(f"unsupported formal dataset: {dataset}")
    return {
        "batching_rule": "whole_graph_single",
        "subgraph_rule": "whole_graph",
        "self_loop_rule": "per_node",
        "edge_sort_rule": "edge_gid_then_dst_stable",
        "label_format": "class_index",
        "loss": "CrossEntropyLoss",
        "split_format": split_format,
    }


def _validate_field_set(
    values: Mapping[str, object],
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    unexpected = sorted(set(values).difference(allowed))
    if unexpected:
        raise ValueError(
            f"{label} contains unknown/non-canonical manifest field: {unexpected[0]}"
        )
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"{label} is missing canonical field: {missing[0]}")


def validate_checkpoint_manifest(manifest: Mapping[str, object]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be an object")
    _validate_field_set(
        manifest,
        CHECKPOINT_MANIFEST_FIELDS,
        CHECKPOINT_MANIFEST_REQUIRED_FIELDS,
        "checkpoint manifest",
    )
    _validate_manifest_types_and_shape(manifest)
    if manifest["family_schema"] != "multi_layer_multi_head":
        raise ValueError("checkpoint manifest family_schema is not canonical")
    semantics = canonical_checkpoint_semantics(manifest["dataset"])
    for key, expected in semantics.items():
        if manifest[key] != expected:
            raise ValueError(
                f"checkpoint manifest {key} is not canonical for {manifest['dataset']!r}"
            )
    if manifest["input_preprocess_id"] != canonical_input_preprocess_id(manifest["dataset"]):
        raise ValueError("checkpoint manifest input_preprocess_id does not match dataset")
    expected_chain = canonical_quantization_scale_chain(
        manifest["quant_cfg_id"], manifest["hidden_profile"], manifest["K_out"]
    ).canonical_serialization()
    if manifest["quantization_scale_chain"] != expected_chain:
        raise ValueError("checkpoint manifest quantization_scale_chain is not canonical")


def validate_checkpoint_metadata(metadata: Mapping[str, object]) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    _validate_field_set(
        metadata,
        CHECKPOINT_METADATA_FIELDS,
        CHECKPOINT_METADATA_REQUIRED_FIELDS,
        "checkpoint metadata",
    )
    for field in (
        "dataset", "family_schema", "task_type", "report_unit", "selection_metric",
        "model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id",
        "quantization_scale_chain", "static_table_id", "degree_bound_id",
        "output_average_rule", "batching_rule", "subgraph_rule", "self_loop_rule",
        "edge_sort_rule", "label_format", "loss", "split_format",
    ):
        _require_string(metadata[field], f"checkpoint metadata.{field}")
    layer_count = _require_unsigned(metadata["L"], "checkpoint metadata.L", positive=True)
    output_head_count = _require_unsigned(metadata["K_out"], "checkpoint metadata.K_out", positive=True)
    class_count = _require_unsigned(metadata["C"], "checkpoint metadata.C", positive=True)
    num_features = _require_unsigned(metadata["num_features"], "checkpoint metadata.num_features", positive=True)
    num_classes = _require_unsigned(metadata["num_classes"], "checkpoint metadata.num_classes", positive=True)
    if class_count != num_classes:
        raise ValueError("checkpoint metadata.C conflicts with num_classes")
    d_in_profile = _require_unsigned_array(
        metadata["d_in_profile"], "checkpoint metadata.d_in_profile", positive=True
    )
    if d_in_profile[0] != num_features:
        raise ValueError("checkpoint metadata.num_features conflicts with d_in_profile")
    raw_profile = metadata["hidden_profile"]
    if not isinstance(raw_profile, list) or not raw_profile:
        raise ValueError("checkpoint metadata.hidden_profile must be a non-empty array")
    profile: list[dict[str, object]] = []
    for index, raw_entry in enumerate(raw_profile):
        entry = _require_exact_object(
            raw_entry,
            _HIDDEN_PROFILE_FIELDS,
            _HIDDEN_PROFILE_FIELDS,
            f"checkpoint metadata.hidden_profile[{index}]",
        )
        _require_unsigned(entry["head_count"], f"checkpoint metadata.hidden_profile[{index}].head_count", positive=True)
        _require_unsigned(entry["head_dim"], f"checkpoint metadata.hidden_profile[{index}].head_dim", positive=True)
        profile.append(entry)
    _validate_profile_and_dimension_chain(
        layer_count=layer_count,
        d_in_profile=d_in_profile,
        hidden_profile=profile,
        hidden_layers=None,
        output_head_count=output_head_count,
        class_count=class_count,
        label="checkpoint metadata",
    )
    _require_unsigned(metadata["seed"], "checkpoint metadata.seed")
    _require_float(metadata["dropout"], "checkpoint metadata.dropout")
    _require_float(metadata["learning_rate"], "checkpoint metadata.learning_rate", positive=True)
    _require_float(metadata["weight_decay"], "checkpoint metadata.weight_decay")
    if not 0.0 <= float(metadata["dropout"]) < 1.0:
        raise ValueError("checkpoint metadata.dropout must be in [0, 1)")
    if metadata["family_schema"] != "multi_layer_multi_head":
        raise ValueError("checkpoint metadata family_schema is not canonical")
    semantics = canonical_checkpoint_semantics(metadata["dataset"])
    for key, expected in semantics.items():
        if metadata[key] != expected:
            raise ValueError(
                f"checkpoint metadata {key} is not canonical for {metadata['dataset']!r}"
            )
    if metadata["input_preprocess_id"] != canonical_input_preprocess_id(metadata["dataset"]):
        raise ValueError("checkpoint metadata input_preprocess_id does not match dataset")
    expected_chain = canonical_quantization_scale_chain(
        metadata["quant_cfg_id"], metadata["hidden_profile"], metadata["K_out"]
    ).canonical_serialization()
    if metadata["quantization_scale_chain"] != expected_chain:
        raise ValueError("checkpoint metadata quantization_scale_chain is not canonical")


def _parse_decimal_unsigned(token: str, label: str, *, positive: bool = False) -> int:
    if not token or any(character not in "0123456789" for character in token):
        raise ValueError(f"{label} must be an unsigned decimal integer")
    return _require_unsigned(int(token), label, positive=positive)


def _shape_element_count(shape: Sequence[int], label: str) -> int:
    count = 1
    for index, dimension in enumerate(shape):
        count *= _require_unsigned(
            dimension,
            f"{label}[{index}]",
            positive=True,
        )
    return count


def _parse_text_float32(token: str, label: str) -> np.float32:
    try:
        parsed = float(token)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint tensor dump value is not numeric: {label}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"checkpoint tensor dump value is not finite: {label}")
    # The producer writes float32 values as float64 ``.17g`` decimals.  Parse
    # the decimal once and round only to float32, then compare the resulting
    # bytes with the NPZ value.  A float32 overflow is not an admissible
    # encoding even when the source decimal itself is finite.
    if abs(parsed) > float(np.finfo(np.float32).max):
        raise ValueError(f"checkpoint tensor dump value overflows float32: {label}")
    with np.errstate(all="ignore"):
        result = np.float32(parsed)
    if not bool(np.isfinite(result)):
        raise ValueError(f"checkpoint tensor dump value is not finite float32: {label}")
    return result


def _load_text_tensor_map(
    manifest: Mapping[str, object],
    path: pathlib.Path,
) -> Dict[str, np.ndarray]:
    tensor_index = manifest["tensor_index"]
    if not isinstance(tensor_index, dict):
        raise ValueError("checkpoint manifest.tensor_index must be an object")
    try:
        tokens = pathlib.Path(path).read_text(encoding="utf-8").split()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"unable to load checkpoint tensor dump {path}: {error}") from error

    tensors: Dict[str, np.ndarray] = {}
    cursor = 0
    while cursor < len(tokens):
        if tokens[cursor] != "TENSOR":
            raise ValueError("malformed checkpoint tensor dump marker")
        cursor += 1
        if cursor + 2 > len(tokens):
            raise ValueError("truncated checkpoint tensor dump header")
        name = tokens[cursor]
        cursor += 1
        if not name:
            raise ValueError("checkpoint tensor dump logical tensor name is empty")
        rank = _parse_decimal_unsigned(tokens[cursor], "checkpoint tensor dump rank")
        cursor += 1
        if rank == 0:
            raise ValueError("checkpoint tensor dump rank must be positive")
        if cursor + rank + 1 > len(tokens):
            raise ValueError("truncated checkpoint tensor dump shape")
        shape = tuple(
            _parse_decimal_unsigned(
                tokens[cursor + index],
                f"checkpoint tensor dump dimension {name}[{index}]",
                positive=True,
            )
            for index in range(rank)
        )
        cursor += rank
        count = _parse_decimal_unsigned(tokens[cursor], "checkpoint tensor dump count")
        cursor += 1
        expected_count = _shape_element_count(shape, f"checkpoint tensor dump shape {name}")
        if count != expected_count:
            raise ValueError(f"checkpoint tensor dump count mismatch: {name}")
        if cursor + count > len(tokens):
            raise ValueError("truncated checkpoint tensor dump values")
        try:
            values = np.empty(count, dtype=np.float32)
        except MemoryError as error:
            raise ValueError(f"checkpoint tensor dump is too large: {name}") from error
        for index, token in enumerate(tokens[cursor : cursor + count]):
            values[index] = _parse_text_float32(token, name)
        cursor += count

        if name in tensors:
            raise ValueError(f"checkpoint tensor dump contains duplicate tensor record: {name}")
        if name not in tensor_index:
            raise ValueError(f"checkpoint tensor dump contains unindexed tensor: {name}")
        expected_shape = tuple(tensor_index[name]["shape"])
        if shape != expected_shape:
            raise ValueError(f"checkpoint tensor dump shape mismatch: {name}")
        tensors[name] = values.reshape(shape, order="C").copy()

    if set(tensors) != set(tensor_index):
        raise ValueError("checkpoint tensor dump names do not match tensor_index")
    return tensors


def validate_tensor_encodings(
    manifest: Mapping[str, object],
    npz_path: pathlib.Path,
    text_path: pathlib.Path,
) -> Dict[str, np.ndarray]:
    """Validate and cross-check the one canonical NPZ/TXT tensor map.

    ``tensor_index`` names are the sole logical identities.  Each manifest
    archive key must be the deterministic ``archive_key(name)`` encoding, and
    both required files must encode exactly the same finite float32 tensors in
    C-order.  The returned map is the validated NPZ data keyed by logical name
    so consumers cannot use an unchecked archive after this function returns.
    """
    # Keep this boundary fail-closed for direct callers as well as callers that
    # already loaded a validated manifest.
    validate_checkpoint_manifest(manifest)
    tensor_index = manifest["tensor_index"]
    assert isinstance(tensor_index, dict)

    expected_archive_keys = {
        archive_key(name)
        for name in tensor_index
    }
    npz_by_name: Dict[str, np.ndarray] = {}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            archive_names = list(archive.files)
            if len(archive_names) != len(set(archive_names)):
                raise ValueError("checkpoint NPZ archive contains duplicate keys")
            if set(archive_names) != expected_archive_keys:
                raise ValueError("checkpoint NPZ keys do not match canonical tensor_index")
            for logical_name, raw_entry in tensor_index.items():
                assert isinstance(raw_entry, dict)
                key = raw_entry["archive_key"]
                array = np.asarray(archive[key])
                if array.dtype != np.dtype("float32"):
                    raise ValueError(f"checkpoint NPZ dtype mismatch: {logical_name}")
                expected_shape = tuple(raw_entry["shape"])
                if tuple(array.shape) != expected_shape:
                    raise ValueError(f"checkpoint NPZ shape mismatch: {logical_name}")
                expected_count = _shape_element_count(
                    expected_shape,
                    f"checkpoint manifest tensor_index[{logical_name!r}].shape",
                )
                if int(array.size) != expected_count:
                    raise ValueError(f"checkpoint NPZ element count mismatch: {logical_name}")
                if not bool(np.all(np.isfinite(array))):
                    raise ValueError(f"checkpoint NPZ value is not finite: {logical_name}")
                npz_by_name[logical_name] = np.ascontiguousarray(array).copy()
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"unable to load checkpoint NPZ archive {npz_path}: {error}") from error

    text_by_name = _load_text_tensor_map(manifest, text_path)
    if set(npz_by_name) != set(text_by_name):
        raise ValueError("checkpoint NPZ/TXT logical tensor names do not match")
    for logical_name in sorted(npz_by_name):
        npz_tensor = npz_by_name[logical_name]
        text_tensor = text_by_name[logical_name]
        if npz_tensor.dtype != np.dtype("float32") or text_tensor.dtype != np.dtype("float32"):
            raise ValueError(f"checkpoint NPZ/TXT dtype mismatch: {logical_name}")
        if npz_tensor.shape != text_tensor.shape:
            raise ValueError(f"checkpoint NPZ/TXT shape mismatch: {logical_name}")
        if npz_tensor.size != text_tensor.size:
            raise ValueError(f"checkpoint NPZ/TXT element count mismatch: {logical_name}")
        if npz_tensor.tobytes(order="C") != text_tensor.tobytes(order="C"):
            raise ValueError(
                f"checkpoint NPZ/TXT value mismatch for canonical tensor: {logical_name}"
            )
    return npz_by_name


def write_text_tensor_dump(path: pathlib.Path, tensors: Dict[str, np.ndarray]) -> None:
    encoded_keys: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for name, tensor in sorted(tensors.items()):
            if not isinstance(tensor, np.ndarray) or tensor.dtype != np.dtype("float32"):
                raise ValueError(f"checkpoint tensor must be float32: {name}")
            key = archive_key(name)
            if key in encoded_keys:
                raise ValueError(f"checkpoint tensor archive_key is not unique: {key}")
            encoded_keys.add(key)
            if not bool(np.all(np.isfinite(tensor))):
                raise ValueError(f"checkpoint tensor contains a non-finite value: {name}")
            array = np.ascontiguousarray(tensor)
            flat = array.astype(np.float64, copy=False).reshape(-1, order="C")
            shape = " ".join(str(dim) for dim in array.shape)
            handle.write(f"TENSOR {name} {array.ndim} {shape} {flat.size}\n")
            handle.write(" ".join(format(float(value), ".17g") for value in flat))
            handle.write("\n")


def build_family_manifest(
    tensors: Dict[str, np.ndarray],
    checkpoint_prefix: str,
    *,
    d_in_profile: Sequence[int],
    hidden_layers: Sequence[Dict[str, int]],
    hidden_head_specs: Sequence[Dict[str, object]],
    output_head_specs: Sequence[Dict[str, object]],
    num_classes: int,
    task_type: str,
    report_unit: str,
    quant_cfg_id: str,
    quantization_scale_chain: str,
    static_table_id: str,
    degree_bound_id: str,
    model_arch_id: str,
    model_param_id: str,
    dataset: str,
    input_preprocess_id: str,
) -> Dict[str, object]:
    expected_preprocess_id = canonical_input_preprocess_id(dataset)
    if input_preprocess_id != expected_preprocess_id:
        raise ValueError(
            "formal checkpoint manifest input_preprocess_id does not match dataset"
        )
    if quant_cfg_id != CANONICAL_QUANT_CFG_ID:
        raise ValueError("formal checkpoint manifest quant_cfg_id is not canonical")
    _require_string(quantization_scale_chain, "formal checkpoint manifest quantization_scale_chain")
    hidden_profile = [
        {
            "head_count": _require_unsigned(
                layer["head_count"], "formal checkpoint manifest hidden head_count", positive=True
            ),
            "head_dim": _require_unsigned(
                layer["head_dim"], "formal checkpoint manifest hidden head_dim", positive=True
            ),
        }
        for layer in hidden_layers
    ]
    k_out = len(output_head_specs)
    if len(hidden_layers) != 2 or hidden_profile != [
        {"head_count": 8, "head_dim": 8},
        {"head_count": 8, "head_dim": 8},
    ]:
        raise ValueError("formal checkpoint manifest requires hidden_profile=[8x8,8x8]")
    if len(d_in_profile) != 2:
        raise ValueError("formal checkpoint manifest requires two input dimensions")
    if k_out != 2:
        raise ValueError("formal checkpoint manifest requires K_out=2")
    expected_chain = canonical_quantization_scale_chain(
        quant_cfg_id,
        hidden_profile,
        k_out,
    ).canonical_serialization()
    if quantization_scale_chain != expected_chain:
        raise ValueError("formal checkpoint manifest quantization_scale_chain is not canonical")
    _require_string(model_arch_id, "formal checkpoint manifest model_arch_id")
    _require_string(model_param_id, "formal checkpoint manifest model_param_id")

    manifest: Dict[str, object] = {
        "checkpoint_prefix": checkpoint_prefix,
        "dataset": dataset,
        "family_schema": "multi_layer_multi_head",
        "output_average_rule": "per_head_bias_then_arithmetic_mean",
        "L": len(hidden_layers) + 1,
        "d_in_profile": list(d_in_profile),
        "hidden_profile": hidden_profile,
        "hidden_layers": list(hidden_layers),
        "hidden_head_specs": list(hidden_head_specs),
        "K_out": k_out,
        "output_head_specs": list(output_head_specs),
        "C": num_classes,
        "task_type": task_type,
        "report_unit": report_unit,
        "selection_metric": "validation_loss",
        "tensor_count": len(tensors),
        "model_arch_id": model_arch_id,
        "model_param_id": model_param_id,
        "input_preprocess_id": input_preprocess_id,
        "quant_cfg_id": quant_cfg_id,
        "static_table_id": static_table_id,
        "degree_bound_id": degree_bound_id,
        **canonical_checkpoint_semantics(dataset),
        "tensor_index": {},
    }
    manifest["quantization_scale_chain"] = quantization_scale_chain
    tensor_index = manifest["tensor_index"]
    assert isinstance(tensor_index, dict)
    for name, tensor in sorted(tensors.items()):
        tensor_index[name] = {
            "archive_key": archive_key(name),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    validate_checkpoint_manifest(manifest)
    return manifest
