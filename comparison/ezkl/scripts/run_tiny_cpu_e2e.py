#!/usr/bin/env python3
"""Run one bounded, native ezkl/Halo2 tiny CPU E2E cell.

The script is intentionally self-contained at the orchestration boundary.  It
stages a temporary baseline-local checkout, creates a two-node/four-edge
canonical resource package, and then invokes the already built native binary
for the real ``gat-prepare``/``prove``/``verify`` chain.  The temporary
package is not a formal dataset replacement: it is an explicit fixture used
only to exercise the authenticated call chain end to end.

No proof is mocked or synthesized.  The external producer is the checked-in
``scripts/produce_public_statement.py`` and the verifier is the native
Halo2/KZG/SHPLONK verifier.  Every child process has a hard timeout and its
stdout is streamed to a phase log and the terminal.  By default successful
temporary directories are removed while failed directories are retained for
diagnosis; use ``--cleanup never`` to retain a successful package too.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
TINY_DATASET = "cora"
TINY_NODE_COUNT = 2
TINY_FEATURE_DIM = 1
TINY_HIDDEN_HEAD_COUNTS = (1, 1)
TINY_HEAD_DIM = 1
TINY_OUTPUT_HEADS = 1
TINY_CLASSES = 1
TINY_HIDDEN_LAYERS = len(TINY_HIDDEN_HEAD_COUNTS)
TINY_LAYER_COUNT = TINY_HIDDEN_LAYERS + 1
TINY_HIDDEN_INPUT_DIMS = (TINY_FEATURE_DIM,) + tuple(TINY_HIDDEN_HEAD_COUNTS[layer] * TINY_HEAD_DIM for layer in range(TINY_HIDDEN_LAYERS - 1))
TINY_OUTPUT_INPUT_DIM = TINY_HIDDEN_HEAD_COUNTS[-1] * TINY_HEAD_DIM
TINY_MODEL_ARCH = (
    f"gat_family_L{TINY_LAYER_COUNT}_hid" + "-".join(f"{heads}x{TINY_HEAD_DIM}"
                                                     for heads in TINY_HIDDEN_HEAD_COUNTS) + f"_kout{TINY_OUTPUT_HEADS}_c{TINY_CLASSES}"
)
TINY_EDGES = ((0, 0), (1, 0), (0, 1), (1, 1))
DEFAULT_STAGE_TIMEOUT = 45.0
DEFAULT_OVERALL_TIMEOUT = 300.0
DEFAULT_MAX_SRS_LOGROWS = 18
SRS_HASH_RE = "0123456789abcdef"
FORMAL_PATH_FLAGS = frozenset(
    {
        "--resource-manifest",
        "--output-dir",
        "--srs-path",
        "--public-statement",
        "--public-output",
        "--producer-manifest",
        "--setup-context",
        "--config",
        "--compiled-circuit",
        "--statement",
        "--output",
        "--witness",
        "--pk-path",
        "--proof-path",
        "--settings-path",
        "--vk-path",
        "--artifact-manifest",
        "--runs-root",
        "--matrix",
        "--formal-contract",
    }
)


class TinyE2EError(RuntimeError):
    """A bounded tiny run failed at a named phase."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_cfg(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise TinyE2EError(f"malformed config line {path}:{number}")
        key, value = (piece.strip() for piece in line.split("=", 1))
        if not key or key in result:
            raise TinyE2EError(f"duplicate/empty config key {path}:{number}")
        result[key] = value
    return result


def update_cfg(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text().splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for raw in lines:
        stripped = raw.split("#", 1)[0].strip()
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                output.append(f"{key} = {updates[key]}")
                seen.add(key)
                continue
        output.append(raw)
    output.extend(f"{key} = {value}" for key, value in updates.items() if key not in seen)
    path.write_text("\n".join(output) + "\n")


def archive_key(name: str) -> str:
    return name.replace("/", "__").replace(":", "__")


def npy_bytes(shape: tuple[int, ...], values: Iterable[float] | None = None) -> bytes:
    """Build a canonical little-endian float32 C-order NPY v2 array.

    The staged resource generator reads a twelve-byte prefix before consuming
    the v2 header.  Using v2 here keeps that reader, the formal validator, and
    the Rust loader on the same header boundary.  The header length includes
    the terminating newline and is padded so the data payload starts at a
    sixteen-byte boundary, as required by the NPY format.
    """

    if not shape or any(dimension <= 0 for dimension in shape):
        raise TinyE2EError(f"invalid tiny NPY shape: {shape}")
    tuple_text = "(" + ", ".join(str(dimension) for dimension in shape)
    tuple_text += ",)" if len(shape) == 1 else ")"
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + tuple_text + ", }"
    header_bytes = header.encode("latin1")
    prefix_size = 12
    padding = (-(prefix_size + len(header_bytes) + 1)) % 16
    header_bytes += b" " * padding + b"\n"
    prefix = b"\x93NUMPY" + bytes((2, 0)) + struct.pack("<I", len(header_bytes))
    count = 1
    for dimension in shape:
        count *= dimension
    if values is None:
        payload = b"\x00" * (count * 4)
    else:
        materialized = [float(value) for value in values]
        if len(materialized) != count:
            raise TinyE2EError(f"NPY value count {len(materialized)} disagrees with shape {shape} ({count})")
        payload = b"".join(struct.pack("<f", value) for value in materialized)
    return prefix + header_bytes + payload


def scale_exponent(multiplier: str, label: str) -> int:
    """Decode one formal source multiplier without rounding."""

    value = int(multiplier)
    if value <= 0 or value & (value - 1):
        raise TinyE2EError(f"{label} is not a positive power-of-two multiplier: {multiplier}")
    return value.bit_length() - 1


def scale_multiplier(exponent: int, label: str) -> str:
    """Encode one canonical exponent using the source-config representation."""

    if exponent < 0:
        raise TinyE2EError(f"{label} has a negative canonical exponent: {exponent}")
    return str(1 << exponent)


def matrix_shape(value: str, label: str) -> tuple[int, int]:
    """Parse the canonical ``[[row];[row]]`` source shape only."""

    if not (value.startswith("[[") and value.endswith("]]")):
        raise TinyE2EError(f"{label} is not a canonical nested matrix")
    rows = value[2:-2].split("];[")
    if not rows or any("[" in row or "]" in row for row in rows):
        raise TinyE2EError(f"{label} has malformed canonical matrix rows")
    widths = [len(row.split(",")) for row in rows]
    if not widths or any(width != widths[0] for width in widths):
        raise TinyE2EError(f"{label} has ragged canonical matrix rows")
    return len(rows), widths[0]


def matrix_widths(value: str, label: str) -> tuple[int, ...]:
    """Return each canonical matrix row width without assuming uniform heads."""

    if not (value.startswith("[[") and value.endswith("]]")):
        raise TinyE2EError(f"{label} is not a canonical nested matrix")
    rows = value[2:-2].split("];[")
    if not rows or any("[" in row or "]" in row for row in rows):
        raise TinyE2EError(f"{label} has malformed canonical matrix rows")
    return tuple(len(row.split(",")) for row in rows)


def assert_formal_matrix_shapes(chain: str, source_scale: Path) -> None:
    """Keep tiny layer count aligned with the formal scale-chain schema.

    The tiny fixture intentionally uses one head per hidden layer, so its
    matrix widths are smaller than the formal eight-head checkpoint.  Only
    the layer-family count is shared here; the per-head widths are closed by
    the tiny architecture checks below.
    """

    formal_manifest = source_scale.parent.parent / "checkpoints/cora_gat/manifest.json"
    formal = json.loads(formal_manifest.read_text())
    formal_fields = {key: value for piece in formal["quantization_scale_chain"].split("|") for key, value in ([piece.split("=", 1)] if "=" in piece else [])}
    tiny_fields = {key: value for piece in chain.split("|") for key, value in ([piece.split("=", 1)] if "=" in piece else [])}
    matrix_fields = (
        "S_W_hidden",
        "S_H_prime_hidden",
        "S_a_src_hidden",
        "S_a_dst_hidden",
    )
    for field in matrix_fields:
        tiny_rows = matrix_widths(tiny_fields[field], field)
        formal_rows = matrix_widths(formal_fields[field], f"formal {field}")
        if len(tiny_rows) != len(formal_rows):
            raise TinyE2EError(f"tiny {field} layer count disagrees with formal checkpoint manifest")


def chain_fields(chain: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for piece in chain.split("|"):
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        fields[key] = value
    return fields


def assert_tiny_scale_shape_closure(chain: str) -> None:
    """Bind every scale vector/matrix width to the tiny model architecture."""

    fields = chain_fields(chain)
    hidden_matrix_fields = (
        "S_W_hidden",
        "S_H_prime_hidden",
        "S_a_src_hidden",
        "S_a_dst_hidden",
    )
    for field in hidden_matrix_fields:
        if field not in fields or matrix_widths(fields[field], field) != TINY_HIDDEN_HEAD_COUNTS:
            raise TinyE2EError(f"tiny {field} scale shape does not close to {TINY_HIDDEN_HEAD_COUNTS}")
    output_vector_fields = (
        "S_W_output",
        "S_Y_prime_output",
        "S_a_src_output",
        "S_a_dst_output",
        "S_b_output",
    )
    for field in output_vector_fields:
        value = fields.get(field)
        if value is None or not (value.startswith("[") and value.endswith("]")):
            raise TinyE2EError(f"tiny {field} scale vector is malformed")
        width = len([piece for piece in value[1:-1].split(",") if piece.strip()])
        if width != TINY_OUTPUT_HEADS:
            raise TinyE2EError(f"tiny {field} scale width {width} != {TINY_OUTPUT_HEADS}")
    for field, expected_width in (
        ("S_H", TINY_LAYER_COUNT),
        ("S_agg_pre", TINY_HIDDEN_LAYERS),
        ("S_agg", TINY_HIDDEN_LAYERS),
        ("S_cat", TINY_HIDDEN_LAYERS),
    ):
        value = fields.get(field)
        if value is None or not (value.startswith("[") and value.endswith("]")):
            raise TinyE2EError(f"tiny {field} scale vector is malformed")
        width = len([piece for piece in value[1:-1].split(",") if piece.strip()])
        if width != expected_width:
            raise TinyE2EError(f"tiny {field} scale width {width} != {expected_width}")


def tiny_scale_chain(source_scale: Path) -> tuple[dict[str, str], str]:
    """Derive a tiny fixture chain from the formal scale-chain resource.

    The native graph's ValTensor constructor starts public feature inputs at
    exponent one during setup.  Keep that single fixture boundary explicit,
    then derive every dependent scale through the authenticated schema
    closure.  Table-domain scales come directly from the formal config and
    are not duplicated here.
    """

    source = parse_cfg(source_scale)

    def scalar(key: str) -> int:
        if key not in source:
            raise TinyE2EError(f"formal scale chain is missing {key}")
        return scale_exponent(source[key], key)

    def vector(key: str, width: int, *, prefix: bool = False) -> list[int]:
        values = [scale_exponent(piece.strip(), key) for piece in source[key].split(",")]
        if (len(values) < width) or (not prefix and len(values) != width):
            comparator = "<" if prefix else "!="
            raise TinyE2EError(f"formal scale chain {key} width {len(values)} {comparator} {width}")
        return values[:width]

    s_delta = scalar("s_delta")
    s_exp = scalar("s_exp")
    s_agg_pre = vector("s_agg_pre", TINY_HIDDEN_LAYERS)
    s_agg = vector("s_agg", TINY_HIDDEN_LAYERS)
    s_cat = vector("s_cat", TINY_HIDDEN_LAYERS)
    if s_agg_pre != s_agg or s_cat != s_agg:
        raise TinyE2EError("formal scale chain does not share the required ELU/concat scales")

    s_input = 1
    s_h = [s_input, *s_cat]
    w_hidden = [vector(f"s_w_hidden_{layer}", TINY_HIDDEN_HEAD_COUNTS[layer], prefix=True) for layer in range(TINY_HIDDEN_LAYERS)]
    h_prime = [[s_h[layer] + value for value in w_hidden[layer]] for layer in range(TINY_HIDDEN_LAYERS)]
    a_src_hidden = [[s_delta - value for value in row] for row in h_prime]
    a_dst_hidden = [row[:] for row in a_src_hidden]
    if any(value < 0 for row in a_src_hidden for value in row):
        raise TinyE2EError("derived hidden attention scale is negative")

    w_output = vector("s_w_output", TINY_OUTPUT_HEADS, prefix=True)
    y_prime = [s_h[-1] + value for value in w_output]
    a_src_output = [s_delta - value for value in y_prime]
    a_dst_output = a_src_output[:]
    if any(value < 0 for value in a_src_output):
        raise TinyE2EError("derived output attention scale is negative")
    s_out_pre = scalar("s_out_pre")
    s_out_head = scalar("s_out_head")
    s_out = scalar("s_out")
    b_output = [s_out_head] * TINY_OUTPUT_HEADS
    if s_out != s_out_head:
        raise TinyE2EError("formal output scale closure is not shared")

    def vector_text(values: list[int]) -> str:
        return ",".join(scale_multiplier(value, "scale") for value in values)

    def matrix_text(values: list[list[int]]) -> str:
        return "[" + ";".join(f"[{vector_text(row)}]" for row in values) + "]"

    updates = {
        "s_input": scale_multiplier(s_input, "s_input"),
        "s_delta": scale_multiplier(s_delta, "s_delta"),
        "s_exp": scale_multiplier(s_exp, "s_exp"),
        "s_h": vector_text(s_h),
        "s_agg_pre": vector_text(s_agg_pre),
        "s_agg": vector_text(s_agg),
        "s_cat": vector_text(s_cat),
        "s_out_pre": scale_multiplier(s_out_pre, "s_out_pre"),
        "s_out_head": scale_multiplier(s_out_head, "s_out_head"),
        "s_out": scale_multiplier(s_out, "s_out"),
        "s_w_output": vector_text(w_output),
        "s_y_prime": vector_text(y_prime),
        "s_a_src_output": vector_text(a_src_output),
        "s_a_dst_output": vector_text(a_dst_output),
        "s_b_output": vector_text(b_output),
    }
    for layer in range(TINY_HIDDEN_LAYERS):
        updates.update(
            {
                f"s_w_hidden_{layer}": vector_text(w_hidden[layer]),
                f"s_h_prime_hidden_{layer}": vector_text(h_prime[layer]),
                f"s_a_src_hidden_{layer}": vector_text(a_src_hidden[layer]),
                f"s_a_dst_hidden_{layer}": vector_text(a_dst_hidden[layer]),
            }
        )
    checkpoint = (
        "schema=gat_l3_scale_chain|"
        f"S_input={updates['s_input']}|S_Delta={updates['s_delta']}|S_exp={updates['s_exp']}|"
        f"S_H=[{updates['s_h']}]|S_W_hidden={matrix_text(w_hidden)}|"
        f"S_H_prime_hidden={matrix_text(h_prime)}|"
        f"S_a_src_hidden={matrix_text(a_src_hidden)}|"
        f"S_a_dst_hidden={matrix_text(a_dst_hidden)}|"
        f"S_agg_pre=[{updates['s_agg_pre']}]|S_agg=[{updates['s_agg']}]|S_cat=[{updates['s_cat']}]|"
        f"S_out_pre={updates['s_out_pre']}|S_out_head={updates['s_out_head']}|S_out={updates['s_out']}|"
        f"S_W_output=[{updates['s_w_output']}]|S_Y_prime_output=[{updates['s_y_prime']}]|"
        f"S_a_src_output=[{updates['s_a_src_output']}]|S_a_dst_output=[{updates['s_a_dst_output']}]|"
        f"S_b_output=[{updates['s_b_output']}]|S_inv=field_exact|S_alpha=field_exact|"
        "round=nearest_ties_away|rescale=none|output_average=signed_round_div"
    )
    assert_formal_matrix_shapes(checkpoint, source_scale)
    assert_tiny_scale_shape_closure(checkpoint)
    return updates, checkpoint


def assert_tiny_tensor_shape_closure(names: list[tuple[str, tuple[int, ...]]]) -> None:
    """Check tensor shapes against the same closure used by the producer."""

    expected: list[tuple[str, tuple[int, ...]]] = []
    for layer, input_dim in enumerate(TINY_HIDDEN_INPUT_DIMS):
        for head in range(TINY_HIDDEN_HEAD_COUNTS[layer]):
            prefix = f"hidden/layer{layer}/head{head}"
            expected.extend(
                [
                    (f"{prefix}/attn_dst/kernel", (1, TINY_HEAD_DIM, 1)),
                    (f"{prefix}/attn_src/kernel", (1, TINY_HEAD_DIM, 1)),
                    (f"{prefix}/seq/kernel", (1, input_dim, TINY_HEAD_DIM)),
                ]
            )
    for head in range(TINY_OUTPUT_HEADS):
        prefix = f"output/head{head}"
        expected.extend(
            [
                (f"{prefix}/attn_dst/kernel", (1, TINY_CLASSES, 1)),
                (f"{prefix}/attn_src/kernel", (1, TINY_CLASSES, 1)),
                (f"{prefix}/seq/kernel", (1, TINY_OUTPUT_INPUT_DIM, TINY_CLASSES)),
                (f"{prefix}/output_bias", (1, TINY_CLASSES, 1)),
            ]
        )
    if len(names) != len(expected):
        raise TinyE2EError(f"tiny tensor count closure is {len(names)}/{len(expected)}")
    if names != expected:
        raise TinyE2EError(f"tiny tensor shape closure mismatch: {names!r} != {expected!r}")


def tensor_names() -> list[tuple[str, tuple[int, ...]]]:
    names: list[tuple[str, tuple[int, ...]]] = []
    for layer, input_dim in enumerate(TINY_HIDDEN_INPUT_DIMS):
        for head in range(TINY_HIDDEN_HEAD_COUNTS[layer]):
            prefix = f"hidden/layer{layer}/head{head}"
            names.extend(
                [
                    (f"{prefix}/attn_dst/kernel", (1, TINY_HEAD_DIM, 1)),
                    (f"{prefix}/attn_src/kernel", (1, TINY_HEAD_DIM, 1)),
                    (f"{prefix}/seq/kernel", (1, input_dim, TINY_HEAD_DIM)),
                ]
            )
    for head in range(TINY_OUTPUT_HEADS):
        prefix = f"output/head{head}"
        names.extend(
            [
                (f"{prefix}/attn_dst/kernel", (1, TINY_CLASSES, 1)),
                (f"{prefix}/attn_src/kernel", (1, TINY_CLASSES, 1)),
                (f"{prefix}/seq/kernel", (1, TINY_OUTPUT_INPUT_DIM, TINY_CLASSES)),
                (f"{prefix}/output_bias", (1, TINY_CLASSES, 1)),
            ]
        )
    assert_tiny_tensor_shape_closure(names)
    return names


def write_tiny_checkpoint(root: Path, dataset: str, preprocess: str, quantization_scale_chain: str) -> None:
    checkpoint = root / "resources/canonical/checkpoints" / f"{dataset}_gat"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_arch = TINY_MODEL_ARCH
    specs_hidden = [
        {
            "attn_dst_kernel": f"hidden/layer{layer}/head{head}/attn_dst/kernel",
            "attn_src_kernel": f"hidden/layer{layer}/head{head}/attn_src/kernel",
            "layer_index": layer,
            "local_head_index": head,
            "seq_kernel": f"hidden/layer{layer}/head{head}/seq/kernel",
        } for layer in range(TINY_HIDDEN_LAYERS) for head in range(TINY_HIDDEN_HEAD_COUNTS[layer])
    ]
    specs_output = [
        {
            "attn_dst_kernel": f"output/head{head}/attn_dst/kernel",
            "attn_src_kernel": f"output/head{head}/attn_src/kernel",
            "head_index": head,
            "output_bias": f"output/head{head}/output_bias",
            "seq_kernel": f"output/head{head}/seq/kernel",
        } for head in range(TINY_OUTPUT_HEADS)
    ]
    names = tensor_names()
    tensor_index = {name: {"archive_key": archive_key(name), "dtype": "float32", "shape": list(shape)} for name, shape in names}
    template = {
        "C": TINY_CLASSES,
        "K_out": TINY_OUTPUT_HEADS,
        "L": TINY_LAYER_COUNT,
        "batching_rule": "whole_graph_single",
        "checkpoint_prefix": "best_model.pt",
        "d_in_profile": list(TINY_HIDDEN_INPUT_DIMS),
        "dataset": dataset,
        "degree_bound_id": f"ledger:gat:{dataset}:v1",
        "degree_bound_digest": "0" * 64,
        "edge_sort_rule": "edge_gid_then_dst_stable",
        "family_schema": "multi_layer_multi_head",
        "hidden_head_specs": specs_hidden,
        "hidden_layers": [
            {
                "head_count": TINY_HIDDEN_HEAD_COUNTS[layer],
                "head_dim": TINY_HEAD_DIM,
                "input_dim": input_dim,
                "layer_index": layer,
            } for layer, input_dim in enumerate(TINY_HIDDEN_INPUT_DIMS)
        ],
        "hidden_profile": [{
            "head_count": TINY_HIDDEN_HEAD_COUNTS[layer],
            "head_dim": TINY_HEAD_DIM,
        } for layer in range(TINY_HIDDEN_LAYERS)],
        "input_preprocess_id": preprocess,
        "label_format": "class_index",
        "loss": "CrossEntropyLoss",
        "model_arch_id": model_arch,
        "model_param_id": "tiny_nondegenerate_seed11",
        "output_average_rule": "per_head_bias_then_arithmetic_mean",
        "output_head_specs": specs_output,
        "quant_cfg_id": "gat_l3_scale_chain",
        "quantization_scale_chain": quantization_scale_chain,
        "report_unit": "node",
        "selection_metric": "validation_loss",
        "self_loop_rule": "per_node",
        "split_format": "node_masks",
        "static_table_id": f"tables:gat:{dataset}:v1",
        "static_table_digest": "0" * 64,
        "subgraph_rule": "whole_graph",
        "task_type": "transductive_node_classification",
        "tensor_count": len(names),
        "tensor_index": tensor_index,
    }
    assert_tiny_tensor_shape_closure(names)
    if template["d_in_profile"] != list(TINY_HIDDEN_INPUT_DIMS):
        raise TinyE2EError("tiny checkpoint d_in_profile closure mismatch")
    if [(layer["head_count"], layer["head_dim"], layer["input_dim"]) for layer in template["hidden_layers"]] != [
        (TINY_HIDDEN_HEAD_COUNTS[layer], TINY_HEAD_DIM, input_dim) for layer, input_dim in enumerate(TINY_HIDDEN_INPUT_DIMS)
    ]:
        raise TinyE2EError("tiny checkpoint hidden dimension closure mismatch")
    if len(template["output_head_specs"]) != TINY_OUTPUT_HEADS:
        raise TinyE2EError("tiny checkpoint output head closure mismatch")
    write_json(checkpoint / "manifest.json", template)
    (checkpoint / "best_model.pt").write_bytes(b"tiny canonical nondegenerate checkpoint\n")
    tensor_lines: list[str] = []
    with zipfile.ZipFile(checkpoint / "tensors.npz", "w", compression=zipfile.ZIP_STORED) as archive:
        for name, shape in names:
            count = 1
            for dimension in shape:
                count *= dimension
            tensor_lines.append("TENSOR " + name + " " + str(len(shape)) + " " + " ".join(map(str, shape)) + " " + str(count))
            if name.startswith("hidden/") and "/attn_" in name:
                value = 0.03125
            elif name.endswith("/output_bias"):
                head = int(name.split("/head", 1)[1].split("/", 1)[0])
                value = 0.0625 * (head + 1)
            else:
                value = 0.25
            values = [value] * count
            tensor_lines.append(" ".join(f"{item:.8f}" for item in values))
            archive.writestr(archive_key(name) + ".npy", npy_bytes(shape, values))
    (checkpoint / "tensors.txt").write_text("\n".join(tensor_lines) + "\n")


def write_tiny_dataset(root: Path, dataset: str, preprocess: str) -> None:
    if len(TINY_EDGES) != 4:
        raise TinyE2EError(f"tiny fixture must retain exactly four edges, got {len(TINY_EDGES)}")
    if any(node < 0 or node >= TINY_NODE_COUNT for edge in TINY_EDGES for node in edge):
        raise TinyE2EError("tiny fixture edge endpoint is outside the two-node graph")
    cache = root / "resources/canonical/cache" / dataset
    cache.mkdir(parents=True, exist_ok=True)
    meta = [
        f"name = {dataset}",
        f"num_nodes = {TINY_NODE_COUNT}",
        f"num_features = {TINY_FEATURE_DIM}",
        f"num_classes = {TINY_CLASSES}",
        "graph_count = 1",
        "test_count = 1",
        "input_preprocess_id = " + preprocess,
    ]
    if dataset in {"ppi", "ogbn_arxiv"}:
        feature = cache / "features.npy"
        feature.write_bytes(npy_bytes((TINY_NODE_COUNT, TINY_FEATURE_DIM), [0.25, 0.75]))
        meta.extend(
            [
                "feature_format = npy",
                "feature_dtype = float32",
                f"feature_shape = {TINY_NODE_COUNT},{TINY_FEATURE_DIM}",
                "feature_order = C",
                f"feature_length = {TINY_NODE_COUNT * TINY_FEATURE_DIM}",
                f"feature_sha256 = {sha256(feature)}",
            ]
        )
    else:
        (cache / "features.txt").write_text(f"{TINY_NODE_COUNT} {TINY_FEATURE_DIM}\n0:0.25\n0:0.75\n")
    if dataset == "ppi":
        (cache / "edges.txt").write_text("".join(f"0 {src} {dst}\n" for src, dst in TINY_EDGES))
        (cache / "node_ptr.txt").write_text(f"0 {TINY_NODE_COUNT}\n")
    else:
        (cache / "edges.txt").write_text("".join(f"{src} {dst}\n" for src, dst in TINY_EDGES))
    (cache / "meta.cfg").write_text("\n".join(meta) + "\n")


def write_tiny_config(root: Path, dataset: str, preprocess: str) -> None:
    batching = "multi_graph_batch" if dataset == "ppi" else "whole_graph_single"
    task = "inductive_multi_graph_node_classification" if dataset == "ppi" else "transductive_node_classification"
    bundle = "ogbn_arxiv_gat" if dataset == "ogbn_arxiv" else f"{dataset}_gat"
    config = [
        f"dataset = {dataset}",
        "data_root = resources/canonical/cache",
        "cache_root = resources/canonical/cache",
        f"checkpoint_bundle = resources/canonical/checkpoints/{bundle}",
        f"task_type = {task}",
        "report_unit = node",
        f"batching_rule = {batching}",
        "self_loop_rule = per_node",
        "edge_sort_rule = edge_gid_then_dst_stable",
        "chunking_rule = none",
        f"L = {TINY_LAYER_COUNT}",
        "hidden_profile = [" + ",".join(f"{TINY_HIDDEN_HEAD_COUNTS[layer]}x{TINY_HEAD_DIM}" for layer in range(TINY_HIDDEN_LAYERS)) + "]",
        "d_in_profile = [" + ",".join(map(str, TINY_HIDDEN_INPUT_DIMS)) + "]",
        f"K_out = {TINY_OUTPUT_HEADS}",
        f"num_classes = {TINY_CLASSES}",
        "range_bits = 7",
        "seed = 11",
        "dump_trace = false",
        "auto_prepare_dataset = false",
        f"model_arch_id = {TINY_MODEL_ARCH}",
        "model_param_id = tiny_nondegenerate_seed11",
        "quant_cfg_id = gat_l3_scale_chain",
        f"input_preprocess_id = {preprocess}",
        f"static_table_id = tables:gat:{dataset}:v1",
        f"degree_bound_id = ledger:gat:{dataset}:v1",
        "srs_type = trusted_srs",
        "srs_id = official_ezkl_native_tiny_srs",
        "srs_path = resources/native_srs/tiny_pending.srs",
        "trusted_srs_sha256 = " + "0" * 64,
        "privacy_policy = model_and_trace_hidden",
        "input_privacy = public_graph_public_features",
        "output_mode = public_logits",
        "opening_mode = point_grouped_batch_opening",
        "binding_mode = zk_subproof_for_crpc",
        "proof_task = node_classification_full_gat",
        "ezkl_security_profile_id = halo2-kzg-shplonk",
        "static_table_digest = " + "0" * 64,
        "degree_bound_digest = " + "0" * 64,
    ]
    path = root / "resources/canonical/configs" / f"{dataset}_secure.cfg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(config) + "\n")


def create_tiny_resources(root: Path) -> None:
    canonical = root / "resources/canonical"
    (canonical / "configs").mkdir(parents=True, exist_ok=True)
    source_scale = Path(__file__).resolve().parents[1] / "resources/canonical/configs/quantization_scale_chain.cfg"
    scale_config = canonical / "configs/quantization_scale_chain.cfg"
    shutil.copy2(source_scale, scale_config)
    scale_updates, quantization_scale_chain = tiny_scale_chain(source_scale)
    update_cfg(scale_config, scale_updates)
    for dataset in DATASETS:
        preprocess = "none"
        write_tiny_dataset(root, dataset, preprocess)
        write_tiny_checkpoint(root, dataset, preprocess, quantization_scale_chain)
        write_tiny_config(root, dataset, preprocess)


def source_tree_hash(root: Path) -> str:
    paths: list[Path] = []
    for relative in ("src", "gat_adapter", ".cargo"):
        directory = root / relative
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink())
    for relative in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain"):
        path = root / relative
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: tuple(item.relative_to(root).parts)):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def copy_tree_item(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def stage_baseline(root: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for relative in ("src", "gat_adapter", ".cargo"):
        source = source_root / relative
        if source.is_dir():
            copy_tree_item(source, root / relative)
    for relative in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain"):
        source = source_root / relative
        if source.is_file():
            copy_tree_item(source, root / relative)
    binary = source_root / "target/release/ezkl"
    if not binary.is_file() or binary.is_symlink():
        raise TinyE2EError(f"missing already-built official binary: {binary}")
    copy_tree_item(binary, root / "target/release/ezkl")
    (root / "target/release/ezkl").chmod(0o755)
    for relative in (
        "formal_common.py",
        "validate_resources.py",
        "generate_gat_resources.py",
        "produce_public_statement.py",
        "run_single_dataset.py",
        "collect_results.py",
        "validate_final.py",
    ):
        copy_tree_item(source_root / "scripts" / relative, root / "scripts" / relative)
    copy_tree_item(
        source_root / "artifacts/gat_adapter/formal_cli.json",
        root / "artifacts/gat_adapter/formal_cli.json",
    )
    copy_tree_item(
        source_root / "artifacts/build/cpu/build_receipt.json",
        root / "artifacts/build/cpu/build_receipt.json",
    )
    create_tiny_resources(root)


def run_stage(
    name: str,
    command: list[str],
    *,
    root: Path,
    log_dir: Path,
    env: dict[str, str],
    timeout: float,
    overall_deadline: float,
    expect_success: bool = True,
) -> tuple[int, float, str]:
    remaining = overall_deadline - time.monotonic()
    if remaining <= 0:
        raise TinyE2EError(f"{name}: overall timeout expired before phase start")
    assert_formal_argv(root, command)
    hard_timeout = min(timeout, remaining)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    command_path = log_dir / f"{name}.command.json"
    command_path.write_text(json.dumps(command, indent=2) + "\n")
    print(f"[tiny-e2e] {name}: {' '.join(command)}", flush=True)
    started = time.monotonic()
    child_env = dict(env)
    child_env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=0,
        )
    except OSError as exc:
        raise TinyE2EError(f"{name}: cannot start command: {exc}") from exc

    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    chunks: list[bytes] = []
    with log_path.open("wb") as log:
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= hard_timeout:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                selector.close()
                raise TinyE2EError(f"{name}: hard timeout after {hard_timeout:.1f}s")
            events = selector.select(min(0.25, hard_timeout - elapsed))
            if events:
                for key, _ in events:
                    data = os.read(key.fd, 8192)
                    if data:
                        chunks.append(data)
                        log.write(data)
                        log.flush()
                        sys.stdout.write(data.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
                    else:
                        selector.unregister(key.fileobj)
            if process.poll() is not None and not selector.get_map():
                break
        returncode = process.wait()
    selector.close()
    elapsed = time.monotonic() - started
    output = b"".join(chunks).decode("utf-8", errors="replace")
    if expect_success and returncode != 0:
        tail = output[-1200:].replace("\n", " ")
        raise TinyE2EError(f"{name}: exit {returncode}; tail={tail}")
    if not expect_success and returncode == 0:
        raise TinyE2EError(f"{name}: tamper/rejection command unexpectedly succeeded")
    state = "PASS" if (expect_success and returncode == 0) or (not expect_success and returncode != 0) else "FAIL"
    print(f"[tiny-e2e] {name}: {state} ({elapsed:.3f}s, rc={returncode})", flush=True)
    return returncode, elapsed, output


def relative(root: Path, path: Path) -> str:
    """Return a strictly baseline-local, canonical path for formal commands."""

    canonical_root = root.resolve()
    canonical_path = path.resolve(strict=False)
    try:
        local = canonical_path.relative_to(canonical_root)
    except ValueError as exc:
        raise TinyE2EError(f"formal path escapes baseline root: {path} -> {canonical_path} (root {canonical_root})") from exc
    if any(part == ".." for part in local.parts):
        raise TinyE2EError(f"formal path contains parent traversal: {path}")
    rendered = local.as_posix()
    if ".." in Path(rendered).parts or rendered.startswith("/"):
        raise TinyE2EError(f"formal path is not a safe baseline-local string: {rendered}")
    return rendered


def assert_formal_argv(root: Path, command: list[str]) -> None:
    """Static-plan guard for every path argument sent to ezkl/formal scripts."""

    if not command:
        raise TinyE2EError("formal command argv is empty")
    local_executable = command[0]
    if local_executable.startswith(("target/", "scripts/")):
        if local_executable != relative(root, root / local_executable):
            raise TinyE2EError(f"formal executable is not baseline-local: {local_executable}")
    for index, argument in enumerate(command):
        if argument not in FORMAL_PATH_FLAGS:
            continue
        if index + 1 >= len(command):
            raise TinyE2EError(f"formal path flag has no value: {argument}")
        value = command[index + 1]
        path = Path(value)
        if path.is_absolute() or "\\" in value or any(part == ".." for part in path.parts):
            raise TinyE2EError(f"formal argv path must be baseline-local POSIX without '..': {argument} {value}")
        relative(root, root / value)


def create_tiny_workspace() -> Path:
    """Create the fixture root below this baseline, never in a system temp dir."""

    source_root = Path(__file__).resolve().parents[1]
    workspace_parent = source_root / "artifacts" / "tiny-e2e"
    workspace_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=workspace_parent))
    relative(source_root, root)
    return root


def copy_manifest_and_update_receipt(root: Path) -> None:
    manifest = root / "resources/resource_manifest.json"
    receipt_path = root / "artifacts/build/cpu/build_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["binary_sha256"] = sha256(root / "target/release/ezkl")
    receipt["source_tree_sha256"] = source_tree_hash(root)
    receipt["resource_manifest_sha256"] = sha256(manifest)
    write_json(receipt_path, receipt)


def formal_native_srs_hashes() -> tuple[int, dict[int, str]]:
    """Return the smallest usable k and hashes from the formal native table.

    ``formal_common.native_srs_contract`` parses the embedded Rust hash table
    and rejects gaps or malformed entries.  k=1/k=2 are intentionally below
    the formal SRS floor for this fixture, so they are never attempted; every
    candidate that remains is still required to be present in that table.
    """

    from formal_common import native_srs_contract

    _url, _max_logrows, hashes = native_srs_contract()
    candidates = sorted(logrows for logrows in hashes if logrows > 2)
    if not candidates:
        raise TinyE2EError("formal native SRS hash table has no usable k above 2")
    return candidates[0], hashes


def update_tiny_srs_config(root: Path, path: Path) -> None:
    digest = sha256(path)
    for dataset in DATASETS:
        config = root / "resources/canonical/configs" / f"{dataset}_secure.cfg"
        update_cfg(
            config,
            {
                "srs_id": "official_ezkl_native_tiny_srs",
                "srs_path": relative(root, path),
                "trusted_srs_sha256": digest,
            },
        )


def build_chain_commands(root: Path, output: Path, binary: Path, srs: Path) -> dict[str, list[str]]:
    """Mirror run_single_dataset.py's official command construction exactly."""

    py = sys.executable
    resource_manifest = "resources/resource_manifest.json"
    config = f"resources/canonical/configs/{TINY_DATASET}_secure.cfg"
    b = relative(root, binary)
    out = relative(root, output)
    srs_rel = relative(root, srs)
    statement = relative(root, output / "statement.json")
    public_output = relative(root, output / "public_output.json")
    producer_manifest = relative(root, output / "producer_manifest.json")
    context = relative(root, output / "producer_context.json")
    compiled = relative(root, output / "compiled.circuit")
    witness = relative(root, output / "witness.json")
    proof = relative(root, output / "proof.json")
    settings = relative(root, output / "settings.json")
    vk = relative(root, output / "vk.key")
    commands = {
        "initial_prepare": [
            b,
            "gat-prepare",
            "--dataset",
            TINY_DATASET,
            "--phase",
            "setup",
            "--resource-manifest",
            resource_manifest,
            "--output-dir",
            out,
            "--srs-path",
            srs_rel,
        ],
        "producer": [
            py,
            "scripts/produce_public_statement.py",
            "--dataset",
            TINY_DATASET,
            "--resource-manifest",
            resource_manifest,
            "--setup-context",
            context,
            "--output-dir",
            out,
            "--config",
            config,
            "--statement",
            statement,
            "--public-output",
            public_output,
            "--producer-manifest",
            producer_manifest,
            "--tiny-fixture",
        ],
        "final_prepare": [
            b,
            "gat-prepare",
            "--dataset",
            TINY_DATASET,
            "--phase",
            "final",
            "--resource-manifest",
            resource_manifest,
            "--output-dir",
            out,
            "--srs-path",
            srs_rel,
            "--public-statement",
            statement,
            "--public-output",
            public_output,
            "--producer-manifest",
            producer_manifest,
        ],
        "witness": [
            b,
            "gat-witness",
            "--dataset",
            TINY_DATASET,
            "--resource-manifest",
            resource_manifest,
            "--compiled-circuit",
            compiled,
            "--statement",
            statement,
            "--output",
            witness,
        ],
        "prove": [
            b,
            "prove",
            "--witness",
            witness,
            "--compiled-circuit",
            compiled,
            "--pk-path",
            relative(root, output / "pk.key"),
            "--proof-path",
            proof,
            "--srs-path",
            srs_rel,
            "--check-mode",
            "safe",
        ],
        "verify": [
            b,
            "verify",
            "--settings-path",
            settings,
            "--proof-path",
            proof,
            "--vk-path",
            vk,
            "--srs-path",
            srs_rel,
            "--statement",
            statement,
            "--public-output",
            public_output,
            "--artifact-manifest",
            relative(root, output / "artifact_manifest.json"),
            "--resource-manifest",
            resource_manifest,
        ],
    }
    for command in commands.values():
        assert_formal_argv(root, command)
    return commands


def make_result(root: Path, output: Path, binary: Path, srs: Path, timings: dict[str, float]) -> None:
    from formal_common import canonical_json_digest as formal_canonical_json_digest
    from formal_common import load_json, validate_gpu_evidence

    statement = output / "statement.json"
    public_output = output / "public_output.json"
    proof = output / "proof.json"
    settings = output / "settings.json"
    compiled = output / "compiled.circuit"
    vk = output / "vk.key"
    pk = output / "pk.key"
    artifact = load_json(output / "artifact_manifest.json", "artifact")
    core = load_json(output / "core_run_manifest.json", "core")
    evidence = validate_gpu_evidence(output / "gpu_evidence.json", backend="cpu")
    proof_doc = load_json(proof, "proof")
    instances = proof_doc.get("instances")
    if not isinstance(instances, list) or not instances:
        raise TinyE2EError("native proof has no instances")
    output_doc = load_json(public_output, "public output")
    result = {
        "schema_version": 1,
        "status": "VERIFY_OK",
        "dataset": TINY_DATASET,
        "backend": "cpu",
        "proof_scope": "full_gat",
        "privacy_policy": "model_and_trace_hidden",
        "public_statement_mode": "external",
        "output_mode": "public_logits",
        "relation_id": "R_GAT-ezkl",
        "quant_cfg_id": artifact["quant_cfg_id"],
        "static_tables_id": artifact["static_tables_id"],
        "public_statement_sha256": sha256(statement),
        "public_output_sha256": sha256(public_output),
        "srs_sha256": sha256(srs),
        "srs_provenance": artifact["srs_provenance"],
        "settings_sha256": sha256(settings),
        "compiled_circuit_sha256": sha256(compiled),
        "vk_sha256": sha256(vk),
        "pk_sha256": sha256(pk),
        "proof_sha256": sha256(proof),
        "proof_size_bytes": proof.stat().st_size,
        "binary_sha256": sha256(binary),
        "source_tree_sha256": source_tree_hash(root),
        "resource_manifest_sha256": sha256(root / "resources/resource_manifest.json"),
        "verifier_reads_witness": False,
        "proof_contains_private_witness": False,
        "native_verify_result": core["native_verify_result"],
        "native_public_instance_binding": core["native_public_instance_binding"],
        "phase_timings_ms": {
            "prepare_compile_setup": max(0.001, timings["initial_prepare"] + timings["final_prepare"]),
            "witness_generation": max(0.001, timings["witness"]),
            "prove": max(0.001, timings["prove"]),
            "verify": max(0.001, timings["verify"]),
        },
        "gpu_evidence": evidence,
        "circuit_ref": artifact["circuit_ref"],
        "proof_instances": {
            "proof_field": "instances",
            "encoding": "canonical-json",
            "sha256": formal_canonical_json_digest(instances),
            "count": sum(len(item) for item in instances if isinstance(item, list)),
            "public_statement_sha256": sha256(statement),
            "public_output_sha256": sha256(public_output),
            "native_binding": True,
        },
        "public_statement_path": relative(root, statement),
        "public_output_path": relative(root, public_output),
        "srs_path": relative(root, srs),
        "settings_path": relative(root, settings),
        "compiled_circuit_path": relative(root, compiled),
        "vk_path": relative(root, vk),
        "pk_path": relative(root, pk),
        "binary_path": relative(root, binary),
        "artifact_manifest_path": relative(root, output / "artifact_manifest.json"),
        "artifact_manifest_sha256": sha256(output / "artifact_manifest.json"),
        "proof_path": relative(root, proof),
        "core_run_manifest_sha256": sha256(output / "core_run_manifest.json"),
    }
    if output_doc.get("dataset") != TINY_DATASET:
        raise TinyE2EError("public output dataset does not match tiny result")
    write_json(output / "result.json", result)


def mutate_json(path: Path, key: str) -> bytes:
    original = path.read_bytes()
    value = json.loads(original)
    if key == "Y":
        value["Y"][0] = int(value["Y"][0]) + 1
    elif key == "statement_digest":
        value[key] = "0" * 64
    else:
        value[key] = "tampered"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return original


def negative_collector_cell(root: Path, output: Path, mutation: str, timeout: float, deadline: float, env: dict[str, str]) -> None:
    result_path = output / "result.json"
    original = result_path.read_bytes()
    try:
        value = json.loads(original)
        if mutation == "gpu":
            value["backend"] = "gpu"
        elif mutation == "fallback":
            value["gpu_evidence"]["cpu_fallback"] = True
        elif mutation == "stale":
            value["resource_manifest_sha256"] = "0" * 64
        elif mutation == "missing":
            del value["proof_sha256"]
        else:
            raise TinyE2EError(f"unknown collector mutation {mutation}")
        result_path.write_text(json.dumps(value, indent=2) + "\n")
        run_stage(
            f"collector-reject-{mutation}",
            [
                sys.executable,
                "scripts/collect_results.py",
                "--runs-root",
                "runs/tiny",
                "--output",
                "results/tiny_matrix.json",
                "--datasets",
                TINY_DATASET,
                "--backends",
                "cpu",
            ],
            root=root,
            log_dir=output / "logs",
            env=env,
            timeout=timeout,
            overall_deadline=deadline,
            expect_success=False,
        )
    finally:
        result_path.write_bytes(original)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-timeout", type=float, default=DEFAULT_STAGE_TIMEOUT)
    parser.add_argument("--overall-timeout", type=float, default=DEFAULT_OVERALL_TIMEOUT)
    parser.add_argument("--max-srs-logrows", type=int, default=DEFAULT_MAX_SRS_LOGROWS)
    parser.add_argument(
        "--cleanup",
        choices=("success", "always", "never"),
        default="success",
        help="remove temporary root on success, always, or never (failed roots are kept by default)",
    )
    parser.add_argument("--plan", action="store_true", help="print the official chain shape without creating a fixture")
    return parser


def print_plan() -> None:
    print("tiny fixture: mktemp baseline root; two nodes, four edges, nonzero checkpoint tensors")
    print("official chain: get-srs(k<=18) -> gat-prepare setup -> external producer -> gat-prepare final -> gat-witness -> prove -> verify")
    print("post-chain: statement/Y/producer-manifest tamper rejection -> CPU collector/final -> GPU/fallback/stale/missing rejection")
    print("collector/final axis: --datasets cora --backends cpu (default five-dataset behavior remains unchanged)")


def main(args: argparse.Namespace) -> int:
    if args.plan:
        print_plan()
        return 0
    if args.stage_timeout <= 0 or args.overall_timeout <= 0:
        raise TinyE2EError("timeouts must be positive")
    if args.max_srs_logrows < 1 or args.max_srs_logrows >= 26:
        raise TinyE2EError("--max-srs-logrows must be in [1,25]; k26 is forbidden")

    started = time.monotonic()
    deadline = started + args.overall_timeout
    temp_root = create_tiny_workspace()
    success = False
    print(f"[tiny-e2e] temporary baseline: {temp_root}", flush=True)
    try:
        stage_baseline(temp_root)
        create_log = temp_root / "logs/tiny-bootstrap"
        create_log.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "EZKL_WORKING_DIR": ".",
            "EZKL_GAT_BACKEND": "cpu",
            "EZKL_ALLOW_TINY_FIXTURE": "1",
        })
        binary = temp_root / "target/release/ezkl"
        generate = [sys.executable, "scripts/generate_gat_resources.py"]
        for dataset in DATASETS:
            generate.extend(("--dataset", dataset))
        run_stage(
            "generate-tiny-resources",
            generate,
            root=temp_root,
            log_dir=create_log,
            env=env,
            timeout=args.stage_timeout,
            overall_deadline=deadline,
        )
        run_stage(
            "validate-tiny-resources",
            [sys.executable, "scripts/validate_resources.py", "--write-manifest", "--require-complete"],
            root=temp_root,
            log_dir=create_log,
            env=env,
            timeout=args.stage_timeout,
            overall_deadline=deadline,
        )
        copy_manifest_and_update_receipt(temp_root)

        min_srs_logrows, native_srs_hashes = formal_native_srs_hashes()
        max_srs_logrows = min(args.max_srs_logrows, max(native_srs_hashes))
        if max_srs_logrows < min_srs_logrows:
            raise TinyE2EError(f"no authenticated native SRS candidate in {min_srs_logrows}..{args.max_srs_logrows}")
        chosen_srs: Path | None = None
        chosen_srs_sha256: str | None = None
        setup_output: Path | None = None
        for logrows in range(min_srs_logrows, max_srs_logrows + 1):
            if time.monotonic() >= deadline:
                raise TinyE2EError("get-srs/setup candidate search: overall timeout expired")
            candidate = temp_root / f"resources/native_srs/tiny_k{logrows}.srs"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            setup = temp_root / f"runs/tiny/cora/cpu-k{logrows}"
            command = [
                relative(temp_root, binary),
                "get-srs",
                "--srs-path",
                relative(temp_root, candidate),
                "--logrows",
                str(logrows),
            ]
            try:
                run_stage(
                    f"get-srs-k{logrows}",
                    command,
                    root=temp_root,
                    log_dir=create_log,
                    env=env,
                    timeout=args.stage_timeout,
                    overall_deadline=deadline,
                )
            except TinyE2EError as exc:
                print(f"[tiny-e2e] get-srs-k{logrows}: candidate rejected: {exc}", flush=True)
                continue
            expected_srs_sha256 = native_srs_hashes.get(logrows)
            if expected_srs_sha256 is None:
                raise TinyE2EError(f"get-srs-k{logrows}: candidate is not in the formal native hash table")
            if sha256(candidate) != expected_srs_sha256:
                raise TinyE2EError(f"get-srs-k{logrows}: bytes do not match the formal native embedded SHA-256")
            setup.mkdir(parents=True, exist_ok=False)
            commands = build_chain_commands(temp_root, setup, binary, candidate)
            try:
                _, setup_elapsed, _ = run_stage(
                    f"initial-prepare-k{logrows}",
                    commands["initial_prepare"],
                    root=temp_root,
                    log_dir=setup / "logs",
                    env={
                        **env, "EZKL_GAT_RUN_DIR": str(setup)
                    },
                    timeout=args.stage_timeout,
                    overall_deadline=deadline,
                )
            except TinyE2EError as exc:
                print(f"[tiny-e2e] initial-prepare-k{logrows}: candidate rejected: {exc}", flush=True)
                shutil.rmtree(setup, ignore_errors=True)
                continue
            chosen_srs, chosen_srs_sha256, setup_output = candidate, expected_srs_sha256, setup
            initial_setup_ms = setup_elapsed * 1000.0
            break
        if chosen_srs is None or chosen_srs_sha256 is None or setup_output is None:
            raise TinyE2EError(
                f"initial prepare blocked: no feasible authenticated native SRS logrows in {min_srs_logrows}..{max_srs_logrows}; k26 was not attempted"
            )

        update_tiny_srs_config(temp_root, chosen_srs)
        run_stage(
            "refresh-tiny-resource-manifest",
            [sys.executable, "scripts/validate_resources.py", "--write-manifest", "--require-complete"],
            root=temp_root,
            log_dir=create_log,
            env=env,
            timeout=args.stage_timeout,
            overall_deadline=deadline,
        )
        if sha256(chosen_srs) != chosen_srs_sha256:
            raise TinyE2EError("get-srs-cache-reuse: cached bytes no longer match the formal native SHA-256")
        copy_manifest_and_update_receipt(temp_root)
        run_stage(
            "get-srs-cache-reuse",
            [
                relative(temp_root, binary),
                "get-srs",
                "--srs-path",
                relative(temp_root, chosen_srs),
                "--logrows",
                str(int(chosen_srs.stem.removeprefix("tiny_k"))),
            ],
            root=temp_root,
            log_dir=create_log,
            env=env,
            timeout=args.stage_timeout,
            overall_deadline=deadline,
        )
        commands = build_chain_commands(temp_root, setup_output, binary, chosen_srs)
        timings: dict[str, float] = {}
        for phase in ("producer", "final_prepare", "witness", "prove", "verify"):
            _, elapsed, _ = run_stage(
                phase,
                commands[phase],
                root=temp_root,
                log_dir=setup_output / "logs",
                env={
                    **env, "EZKL_GAT_RUN_DIR": str(setup_output)
                },
                timeout=args.stage_timeout,
                overall_deadline=deadline,
            )
            timings[phase] = elapsed * 1000.0
        public_output_doc = json.loads((setup_output / "public_output.json").read_text())
        public_y = public_output_doc.get("Y")
        if not isinstance(public_y, list) or not public_y or not any(int(value) != 0 for value in public_y):
            raise TinyE2EError("tiny producer emitted an all-zero public Y")
        if not (setup_output / "artifact_manifest.json").is_file():
            raise TinyE2EError("final prepare did not emit artifact_manifest.json")
        final_output = temp_root / "runs/tiny/cora/cpu"
        final_output.parent.mkdir(parents=True, exist_ok=True)
        setup_output.rename(final_output)
        setup_output = final_output
        commands = build_chain_commands(temp_root, setup_output, binary, chosen_srs)
        make_result(
            temp_root, setup_output, binary, chosen_srs, {
                "initial_prepare": initial_setup_ms,
                "final_prepare": timings["final_prepare"],
                "witness": timings["witness"],
                "prove": timings["prove"],
                "verify": timings["verify"],
            }
        )

        original_statement = mutate_json(setup_output / "statement.json", "statement_digest")
        try:
            run_stage(
                "verify-reject-statement-tamper",
                commands["verify"],
                root=temp_root,
                log_dir=setup_output / "logs",
                env={
                    **env, "EZKL_GAT_RUN_DIR": str(setup_output)
                },
                timeout=args.stage_timeout,
                overall_deadline=deadline,
                expect_success=False,
            )
        finally:
            (setup_output / "statement.json").write_bytes(original_statement)
        original_output = mutate_json(setup_output / "public_output.json", "Y")
        try:
            run_stage(
                "verify-reject-public-output-tamper",
                commands["verify"],
                root=temp_root,
                log_dir=setup_output / "logs",
                env={
                    **env, "EZKL_GAT_RUN_DIR": str(setup_output)
                },
                timeout=args.stage_timeout,
                overall_deadline=deadline,
                expect_success=False,
            )
        finally:
            (setup_output / "public_output.json").write_bytes(original_output)
        original_producer = mutate_json(setup_output / "producer_manifest.json", "producer_id")
        try:
            run_stage(
                "verify-reject-producer-manifest-tamper",
                commands["verify"],
                root=temp_root,
                log_dir=setup_output / "logs",
                env={
                    **env, "EZKL_GAT_RUN_DIR": str(setup_output)
                },
                timeout=args.stage_timeout,
                overall_deadline=deadline,
                expect_success=False,
            )
        finally:
            (setup_output / "producer_manifest.json").write_bytes(original_producer)

        run_stage(
            "collect-tiny-cpu",
            [
                sys.executable,
                "scripts/collect_results.py",
                "--runs-root",
                "runs/tiny",
                "--output",
                "results/tiny_matrix.json",
                "--datasets",
                TINY_DATASET,
                "--backends",
                "cpu",
            ],
            root=temp_root,
            log_dir=setup_output / "logs",
            env=env,
            timeout=args.stage_timeout,
            overall_deadline=deadline,
        )
        run_stage(
            "validate-tiny-cpu",
            [
                sys.executable,
                "scripts/validate_final.py",
                "--matrix",
                "results/tiny_matrix.json",
                "--datasets",
                TINY_DATASET,
                "--backends",
                "cpu",
            ],
            root=temp_root,
            log_dir=setup_output / "logs",
            env=env,
            timeout=args.stage_timeout,
            overall_deadline=deadline,
        )
        for mutation in ("gpu", "fallback", "stale", "missing"):
            negative_collector_cell(temp_root, setup_output, mutation, args.stage_timeout, deadline, env)
        success = True
        print(f"[tiny-e2e] PASS: native tiny CPU E2E completed in {time.monotonic() - started:.3f}s", flush=True)
        return 0
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TinyE2EError) as exc:
        print(f"[tiny-e2e] BLOCKED/FAIL: {exc}", file=sys.stderr, flush=True)
        print(f"[tiny-e2e] retained temporary root: {temp_root}", file=sys.stderr, flush=True)
        return 2
    finally:
        if args.cleanup == "always" or (args.cleanup == "success" and success):
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args().parse_args()))
    except TinyE2EError as exc:
        print(f"run_tiny_cpu_e2e: {exc}", file=sys.stderr)
        raise SystemExit(2)
