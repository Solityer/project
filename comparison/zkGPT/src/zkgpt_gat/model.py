from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .rounding import validate_rounding_mode
from .quantization import (
    CANONICAL_QUANT_CFG_ID,
    canonical_input_preprocess_id,
    canonical_scale_chain_descriptor,
    canonical_scale_profile,
)


QUANT_SCALE_PROFILES: Dict[str, Dict[str, int]] = {
    # The formal comparison profile is the same explicit chain used by the
    # current main GAT experiment.  Source checkpoint formats are handled by
    # the conversion entry point and are never accepted by the formal loader.
    CANONICAL_QUANT_CFG_ID: canonical_scale_profile(),
}


@dataclass
class AttentionHead:
    seq_kernel: np.ndarray
    attn_dst_kernel: np.ndarray
    attn_src_kernel: np.ndarray
    output_bias: np.ndarray


@dataclass
class HiddenLayer:
    layer_index: int
    input_dim: int
    head_count: int
    head_dim: int
    heads: List[AttentionHead]


@dataclass
class Model:
    manifest: Dict
    hidden_layers: List[HiddenLayer]
    output_heads: List[AttentionHead]
    class_count: int
    k_out: int
    model_arch_id: str
    model_param_id: str
    quant_cfg_id: str
    static_table_id: str
    output_average_rule: str
    quant_scales: Dict[str, int]
    # Manual/synthetic models retain the canonical nearest mode for backwards
    # compatible construction; formal loaders always replace it with the
    # explicit config value and validate it.
    rounding_mode: str = "nearest_ties_away"
    input_preprocess_id: str = ""


CANONICAL_CHECKPOINT_SCHEMA = "zkGPT_canonical_no_bias_gat_checkpoint_v1"
_CANONICAL_MANIFEST_REQUIRED = {
    "schema", "dataset", "model_arch_id", "model_param_id", "quant_cfg_id",
    "input_preprocess_id", "rounding_mode", "static_table_id", "quant_scale_chain",
    "L", "hidden_layers", "hidden_profile", "d_in_profile", "C", "K_out",
    "hidden_head_specs", "output_head_specs", "output_average_rule",
    "tensor_count", "tensor_index", "tensor_index_digest", "training_procedure_id",
    "source_data_digest",
}
_CANONICAL_MANIFEST_ALLOWED = _CANONICAL_MANIFEST_REQUIRED | {
    "family_schema_version", "task_type", "report_unit", "batching_rule",
    "self_loop_rule", "edge_sort_rule", "chunking_rule", "label_format",
    "split_format", "subgraph_rule", "loss", "selection_metric", "degree_bound_id",
    "source_bundle_path", "source_manifest_sha256", "source_tensors_sha256",
    "conversion_policy", "family_schema_version",
}


def _load_text_tensors(path: pathlib.Path) -> Dict[str, np.ndarray]:
    tokens = iter(path.read_text(encoding="utf-8").split())
    out: Dict[str, np.ndarray] = {}
    while True:
        try:
            marker = next(tokens)
        except StopIteration:
            break
        if marker != "TENSOR":
            raise ValueError(f"malformed tensor marker in {path}: {marker}")
        name = next(tokens)
        if name in out:
            raise ValueError(f"duplicate tensor name in {path}: {name}")
        rank = int(next(tokens))
        shape = [int(next(tokens)) for _ in range(rank)]
        count = int(next(tokens))
        values = [float(next(tokens)) for _ in range(count)]
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"non-finite tensor value in {name}")
        if int(np.prod(shape)) != count:
            raise ValueError(f"tensor count mismatch for {name}")
        out[name] = array.reshape(shape)
    return out


def _reshape_seq(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"expected seq kernel shape [1,in,out], got {array.shape}")
    return np.asarray(array[0], dtype=np.float64)


def _reshape_attn(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] == 1:
        return np.asarray(array[0, :, 0], dtype=np.float64)
    if array.ndim == 1:
        return np.asarray(array, dtype=np.float64)
    raise ValueError(f"expected attention kernel shape [1,d,1], got {array.shape}")


def _scalar(array: np.ndarray) -> float:
    flat = np.ravel(array)
    if flat.size != 1:
        raise ValueError(f"expected scalar tensor, got {array.shape}")
    return float(flat[0])


def _bias(array: np.ndarray) -> np.ndarray:
    return np.ravel(array).astype(np.float64)


def _round_ties_away(value: float) -> int:
    return int(np.floor(value + 0.5) if value >= 0.0 else np.ceil(value - 0.5))


def _quantize(values: np.ndarray, scale: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64)
    return np.asarray([_round_ties_away(float(item) * scale) for item in np.ravel(flat)], dtype=np.int64).reshape(flat.shape)


def _head_from_spec(spec: Dict, tensors: Dict[str, np.ndarray], scales: Dict[str, int], output_head: bool = False) -> AttentionHead:
    # These scales are the canonical baseline-local formal GAT relation.  Once
    # converted, every forward tensor is an exact integer and no
    # floating-point operation is part of the proof witness.
    return AttentionHead(
        seq_kernel=_quantize(_reshape_seq(tensors[spec["seq_kernel"]]), scales["output_weight"] if output_head else scales["hidden_weight"]),
        attn_dst_kernel=_quantize(_reshape_attn(tensors[spec["attn_dst_kernel"]]), scales["output_attention"] if output_head else scales["hidden_attention"]),
        attn_src_kernel=_quantize(_reshape_attn(tensors[spec["attn_src_kernel"]]), scales["output_attention"] if output_head else scales["hidden_attention"]),
        output_bias=(
            _quantize(_bias(tensors[spec["output_bias"]]), scales["output_bias"])
            if output_head
            else np.zeros(int(_reshape_seq(tensors[spec["seq_kernel"]]).shape[1]), dtype=np.int64)
        ),
    )


def load_model(
    bundle: pathlib.Path,
    *,
    model_arch_id: str | None = None,
    model_param_id: str | None = None,
    quant_cfg_id: str | None = None,
    static_table_id: str | None = None,
    rounding_mode: str | None = None,
) -> Model:
    """Load and validate the raw checkpoint, with optional formal identities.

    The manifest remains the authenticated checkpoint source.  Formal runs
    may bind baseline-local parameter/quantization/table identities from their
    immutable config while retaining the raw manifest IDs in ``manifest``;
    default arguments preserve the historical raw-bundle behavior for tools
    that do not have a formal config.
    """
    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical checkpoint manifest: {manifest_path}") from exc
    _validate_canonical_manifest(manifest)
    if manifest.get("output_average_rule") != "per_head_bias_then_arithmetic_mean":
        raise ValueError("output layer must average per-head post-bias values")
    tensors_path = bundle / "tensors.txt"
    tensors = _load_text_tensors(tensors_path)
    _validate_tensor_index(manifest, tensors)
    npz_tensors = _load_npz_tensors(bundle / "tensors.npz", manifest)
    for name, text_tensor in tensors.items():
        text_float32 = np.ascontiguousarray(np.asarray(text_tensor, dtype=np.float32))
        npz_tensor = npz_tensors[name]
        if npz_tensor.tobytes(order="C") != text_float32.tobytes(order="C"):
            raise ValueError(f"checkpoint tensors.npz/tensors.txt value mismatch: {name}")
    resolved_quant_cfg = str(manifest["quant_cfg_id"] if quant_cfg_id is None else quant_cfg_id)
    if resolved_quant_cfg != CANONICAL_QUANT_CFG_ID:
        raise ValueError("legacy/noncanonical checkpoint quant_cfg_id is rejected")
    if manifest["quant_cfg_id"] != CANONICAL_QUANT_CFG_ID:
        raise ValueError("canonical checkpoint manifest quant_cfg_id is invalid")
    quant_scales = dict(QUANT_SCALE_PROFILES[resolved_quant_cfg])
    resolved_rounding_mode = validate_rounding_mode(
        manifest.get("rounding_mode") if rounding_mode is None else rounding_mode
    )

    hidden_layers: List[HiddenLayer] = []
    for layer_info in manifest["hidden_layers"]:
        specs = [
            item for item in manifest["hidden_head_specs"]
            if int(item["layer_index"]) == int(layer_info["layer_index"])
        ]
        specs.sort(key=lambda item: int(item["local_head_index"]))
        heads = [_head_from_spec(spec, tensors, quant_scales, output_head=False) for spec in specs]
        hidden_layers.append(
            HiddenLayer(
                layer_index=int(layer_info["layer_index"]),
                input_dim=int(layer_info["input_dim"]),
                head_count=int(layer_info["head_count"]),
                head_dim=int(layer_info["head_dim"]),
                heads=heads,
            )
        )
    hidden_layers.sort(key=lambda layer: layer.layer_index)

    output_specs = sorted(manifest["output_head_specs"], key=lambda item: int(item["head_index"]))
    output_heads = [_head_from_spec(spec, tensors, quant_scales, output_head=True) for spec in output_specs]

    if model_arch_id is not None and str(model_arch_id) != str(manifest["model_arch_id"]):
        raise ValueError("checkpoint model_arch_id does not match formal identity")
    if model_param_id is not None and str(model_param_id) != str(manifest["model_param_id"]):
        raise ValueError("checkpoint model_param_id does not match formal identity")
    if static_table_id is not None and str(static_table_id) != str(manifest["static_table_id"]):
        raise ValueError("checkpoint static_table_id does not match formal identity")
    model = Model(
        manifest=manifest,
        hidden_layers=hidden_layers,
        output_heads=output_heads,
        class_count=int(manifest["C"]),
        k_out=int(manifest["K_out"]),
        model_arch_id=str(manifest["model_arch_id"] if model_arch_id is None else model_arch_id),
        model_param_id=str(manifest["model_param_id"] if model_param_id is None else model_param_id),
        quant_cfg_id=resolved_quant_cfg,
        static_table_id=str(manifest["static_table_id"] if static_table_id is None else static_table_id),
        output_average_rule=str(manifest["output_average_rule"]),
        quant_scales=quant_scales,
        rounding_mode=resolved_rounding_mode,
        input_preprocess_id=str(manifest["input_preprocess_id"]),
    )
    _validate_model(model)
    return model


def _validate_canonical_manifest(manifest: Dict) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != CANONICAL_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a canonical no-bias zkGPT bundle")
    missing = sorted(_CANONICAL_MANIFEST_REQUIRED - set(manifest))
    extra = sorted(set(manifest) - _CANONICAL_MANIFEST_ALLOWED)
    if missing:
        raise ValueError(f"canonical checkpoint manifest is missing fields: {missing}")
    if extra:
        raise ValueError(f"canonical checkpoint manifest has unsupported fields: {extra}")
    if manifest.get("quant_cfg_id") != CANONICAL_QUANT_CFG_ID:
        raise ValueError("checkpoint quantization identity is not the registered scale chain")
    dataset = str(manifest.get("dataset", ""))
    expected_preprocess = canonical_input_preprocess_id(dataset)
    if manifest.get("input_preprocess_id") != expected_preprocess:
        raise ValueError("checkpoint input_preprocess_id does not match dataset")
    if validate_rounding_mode(manifest.get("rounding_mode")) != "nearest_ties_away":
        raise ValueError("canonical checkpoint rounding mode is invalid")
    if manifest.get("static_table_id") != "tables:lrelu+elu+exp+range":
        raise ValueError("canonical checkpoint static-table identity is invalid")
    for provenance_key in ("source_data_digest", "source_manifest_sha256", "source_tensors_sha256"):
        value = str(manifest.get(provenance_key, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError(f"canonical checkpoint {provenance_key} is missing or malformed")
    if manifest.get("training_procedure_id") != "baseline_local_source_checkpoint_no_bias_projection_v1":
        raise ValueError("canonical checkpoint training/conversion procedure is not authenticated")
    if manifest.get("conversion_policy") != "drop_hidden_and_attention_bias_terms; retain_output_bias_only":
        raise ValueError("canonical checkpoint bias conversion policy is not authenticated")
    if manifest.get("quant_scale_chain") != canonical_scale_chain_descriptor(
        hidden_profile=manifest.get("hidden_profile"), k_out=int(manifest.get("K_out", -1))
    ):
        raise ValueError("checkpoint quant_scale_chain is not the complete canonical chain")
    hidden_specs = manifest.get("hidden_head_specs")
    output_specs = manifest.get("output_head_specs")
    if not isinstance(hidden_specs, list) or not isinstance(output_specs, list):
        raise ValueError("checkpoint head specs must be arrays")
    for spec in hidden_specs:
        if not isinstance(spec, dict) or any("bias" in str(key).lower() for key in spec):
            raise ValueError("hidden attention/output bias is forbidden in canonical checkpoint")
        if set(spec) != {"layer_index", "local_head_index", "seq_kernel", "attn_src_kernel", "attn_dst_kernel"}:
            raise ValueError("hidden head spec schema is not canonical")
    for spec in output_specs:
        if not isinstance(spec, dict):
            raise ValueError("output head spec must be an object")
        if any("bias" in str(key).lower() and key != "output_bias" for key in spec):
            raise ValueError("output attention bias is forbidden in canonical checkpoint")
        if set(spec) != {"head_index", "seq_kernel", "attn_src_kernel", "attn_dst_kernel", "output_bias"}:
            raise ValueError("output head spec schema is not canonical")
    index = manifest.get("tensor_index")
    if not isinstance(index, dict) or not index:
        raise ValueError("canonical checkpoint tensor_index is missing")
    if any("bias" in str(name).lower() and "/output_bias" not in str(name).lower() for name in index):
        raise ValueError("tensor_index contains a hidden/attention bias")
    if int(manifest.get("tensor_count", -1)) != len(index):
        raise ValueError("checkpoint tensor_count does not match tensor_index")
    if manifest.get("tensor_index_digest") != _digest_json(index):
        raise ValueError("checkpoint tensor_index_digest is invalid")
    expected_names = set()
    for spec in hidden_specs:
        expected_names.update(spec[key] for key in ("seq_kernel", "attn_src_kernel", "attn_dst_kernel"))
    for spec in output_specs:
        expected_names.update(spec[key] for key in ("seq_kernel", "attn_src_kernel", "attn_dst_kernel", "output_bias"))
    if set(index) != expected_names:
        raise ValueError("checkpoint tensor_index does not exactly match canonical head specs")
    for name, entry in index.items():
        if not isinstance(entry, dict) or set(entry) != {"archive_key", "dtype", "shape", "count", "sha256"}:
            raise ValueError(f"canonical checkpoint tensor index entry is invalid: {name}")
        if entry.get("archive_key") != _archive_key(name) or entry.get("dtype") != "float32":
            raise ValueError(f"canonical checkpoint tensor encoding is invalid: {name}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not shape or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError(f"canonical checkpoint tensor shape is invalid: {name}")
        if type(entry.get("count")) is not int or int(entry["count"]) <= 0:
            raise ValueError(f"canonical checkpoint tensor count is invalid: {name}")
        if int(entry["count"]) != int(np.prod(shape, dtype=np.int64)):
            raise ValueError(f"canonical checkpoint tensor count/shape mismatch: {name}")
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"canonical checkpoint tensor digest is invalid: {name}")


def _digest_json(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _archive_key(name: str) -> str:
    return str(name).replace("/", "__").replace(":", "__")


def _tensor_digest(array: np.ndarray) -> str:
    import hashlib

    payload = np.ascontiguousarray(np.asarray(array, dtype=np.float32)).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _validate_tensor_index(manifest: Dict, tensors: Dict[str, np.ndarray]) -> None:
    index = manifest["tensor_index"]
    if set(index) != set(tensors):
        raise ValueError("checkpoint tensor file set differs from authenticated tensor_index")
    for name, entry in index.items():
        if not isinstance(entry, dict) or set(entry) != {"archive_key", "dtype", "shape", "count", "sha256"}:
            raise ValueError(f"tensor_index entry schema is invalid: {name}")
        array = tensors[name]
        if entry["archive_key"] != _archive_key(name) or entry["dtype"] != "float32" or [int(v) for v in entry["shape"]] != list(array.shape) or int(entry["count"]) != int(array.size):
            raise ValueError(f"tensor_index shape/dtype mismatch: {name}")
        canonical = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(canonical)):
            raise ValueError(f"tensor contains a non-finite float32 value: {name}")
        if np.asarray(canonical, dtype=np.float64).tobytes(order="C") != np.asarray(array, dtype=np.float64).tobytes(order="C"):
            raise ValueError(f"tensor text is not an exact float32 encoding: {name}")
        if entry["sha256"] != _tensor_digest(canonical):
            raise ValueError(f"tensor digest mismatch: {name}")


def _load_npz_tensors(path: pathlib.Path, manifest: Dict) -> Dict[str, np.ndarray]:
    index = manifest.get("tensor_index")
    if not isinstance(index, dict):
        raise ValueError("canonical checkpoint tensor_index is missing")
    expected = {str(entry["archive_key"]) for entry in index.values()}
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("checkpoint tensors.npz keys do not match canonical tensor_index")
            out: Dict[str, np.ndarray] = {}
            for logical_name, entry in index.items():
                array = np.asarray(archive[str(entry["archive_key"])])
                if array.dtype != np.dtype("float32") or list(array.shape) != [int(v) for v in entry["shape"]]:
                    raise ValueError(f"checkpoint tensors.npz dtype/shape mismatch: {logical_name}")
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"checkpoint tensors.npz contains a non-finite tensor: {logical_name}")
                out[logical_name] = np.ascontiguousarray(array).copy()
            return out
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"unable to load canonical checkpoint tensors.npz: {path}") from exc


def _validate_model(model: Model) -> None:
    manifest = model.manifest
    manifest_hidden = manifest.get("hidden_layers")
    if not isinstance(manifest_hidden, list) or len(manifest_hidden) != len(model.hidden_layers):
        raise ValueError("model manifest hidden-layer cardinality mismatch")
    if int(manifest.get("L", -1)) != len(model.hidden_layers) + 1:
        raise ValueError("model manifest L does not equal hidden layers plus output layer")
    if [int(item.get("layer_index", -1)) for item in manifest_hidden] != list(range(len(model.hidden_layers))):
        raise ValueError("model manifest hidden layer indices are not contiguous")
    manifest_profile = manifest.get("hidden_profile")
    if manifest_profile is not None:
        expected_profile = [
            {"head_count": int(layer.head_count), "head_dim": int(layer.head_dim)}
            for layer in model.hidden_layers
        ]
        if manifest_profile != expected_profile:
            raise ValueError("model manifest hidden_profile differs from hidden_layers")
    d_in_profile = manifest.get("d_in_profile")
    if not isinstance(d_in_profile, (list, tuple)) or len(d_in_profile) != len(model.hidden_layers):
        raise ValueError("model manifest d_in_profile must have one entry per hidden layer")
    try:
        canonical_d_in = [int(value) for value in d_in_profile]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("model manifest d_in_profile is not integral") from exc
    if any(value <= 0 for value in canonical_d_in):
        raise ValueError("model manifest d_in_profile contains a non-positive dimension")
    if canonical_d_in != [int(layer.input_dim) for layer in model.hidden_layers]:
        raise ValueError("model manifest d_in_profile does not match hidden input dimensions")
    if int(manifest.get("C", -1)) != model.class_count or int(manifest.get("K_out", -1)) != model.k_out:
        raise ValueError("model manifest output dimensions differ from loaded model")
    if model.quant_cfg_id != CANONICAL_QUANT_CFG_ID or model.quant_scales != canonical_scale_profile():
        raise ValueError("loaded model does not use the canonical scale chain")
    if model.input_preprocess_id != canonical_input_preprocess_id(str(manifest.get("dataset", ""))):
        raise ValueError("loaded model input preprocessing identity is not canonical")
    if [(int(item.get("head_count", -1)), int(item.get("head_dim", -1))) for item in manifest_hidden] != [(8, 8), (8, 8)]:
        raise ValueError("canonical model hidden profile must be [8x8,8x8]")
    if model.k_out != 2:
        raise ValueError("canonical model K_out must be 2")
    output_specs = manifest.get("output_head_specs")
    if not isinstance(output_specs, list) or len(output_specs) != model.k_out:
        raise ValueError("model manifest output head cardinality mismatch")
    if [int(item.get("head_index", -1)) for item in output_specs] != list(range(model.k_out)):
        raise ValueError("model manifest output head indices are not contiguous")
    if model.k_out != len(model.output_heads):
        raise ValueError("K_out conflicts with output_head_specs")
    hidden_specs = manifest.get("hidden_head_specs", [])
    if len(hidden_specs) != 16:
        raise ValueError("canonical model hidden head cardinality must be 16")
    for layer_index in range(len(model.hidden_layers)):
        layer_specs = [item for item in hidden_specs if int(item["layer_index"]) == layer_index]
        if len(layer_specs) != 8 or [int(item["local_head_index"]) for item in layer_specs] != list(range(8)):
            raise ValueError("canonical model hidden head indices are not contiguous")
    prev_dim = canonical_d_in[0]
    for layer in model.hidden_layers:
        if layer.input_dim != prev_dim:
            raise ValueError("hidden layer input dimension chain mismatch")
        if layer.head_count != len(layer.heads):
            raise ValueError("hidden layer head count mismatch")
        for head in layer.heads:
            if head.seq_kernel.shape != (layer.input_dim, layer.head_dim):
                raise ValueError(f"bad hidden seq shape: {head.seq_kernel.shape}")
            if head.attn_src_kernel.shape[0] != layer.head_dim or head.attn_dst_kernel.shape[0] != layer.head_dim:
                raise ValueError("hidden attention vector dimension mismatch")
        prev_dim = layer.head_count * layer.head_dim
    for head in model.output_heads:
        if head.seq_kernel.shape != (prev_dim, model.class_count):
            raise ValueError(f"bad output seq shape: {head.seq_kernel.shape}")
        if head.output_bias.shape[0] != model.class_count:
            raise ValueError("output bias dimension mismatch")
