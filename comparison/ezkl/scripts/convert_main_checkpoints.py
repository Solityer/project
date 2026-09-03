#!/usr/bin/env python3
"""Convert the current canonical main checkpoint bundles once, locally.

The converter is a preparation tool only.  It validates the source bundle's
float32 logical payload against its NPZ archive, removes source-project path
metadata, and materializes only explicit zero structural bias tensors required
by the ezkl loader.  It never retrains, rounds, or silently drops a non-zero
hidden bias.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import struct
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def framed_digest(files: list[tuple[str, bytes]]) -> str:
    payload = bytearray()
    for name, data in files:
        encoded = name.encode()
        payload.extend(struct.pack("<Q", len(encoded))); payload.extend(encoded)
        payload.extend(struct.pack("<Q", len(data))); payload.extend(data)
    return digest(bytes(payload))


def npy_values(data: bytes) -> tuple[tuple[int, ...], bytes]:
    if data[:6] != b"\x93NUMPY" or data[6:8] != b"\x01\x00":
        raise ValueError("only little-endian NPY v1 float32 is accepted")
    length = struct.unpack_from("<H", data, 8)[0]
    offset = 10 + length
    info = ast.literal_eval(data[10:offset].decode("latin1").strip())
    if info.get("descr") not in ("<f4", "|f4"):
        raise ValueError("checkpoint NPZ tensor is not little-endian float32")
    shape = tuple(int(item) for item in info["shape"])
    count = math.prod(shape)
    payload = data[offset:]
    if len(payload) != count * 4:
        raise ValueError("checkpoint NPY payload length mismatch")
    if info.get("fortran_order"):
        # The main exporter stores matrix kernels in Fortran order while the
        # logical tensors.txt contract is C-order.  Canonicalize the payload
        # before comparing it; this is a layout conversion, not a numerical
        # transformation.  Keep the implementation stdlib-only so this
        # one-time resource step has no runtime dependency on NumPy.
        raw = struct.unpack("<" + "f" * count, payload)
        logical: list[float] = []
        for linear in range(count):
            remainder = linear
            coordinates: list[int] = []
            for dimension in reversed(shape):
                coordinates.append(remainder % dimension)
                remainder //= dimension
            coordinates.reverse()
            fortran_index = 0
            stride = 1
            for coordinate, dimension in zip(coordinates, shape):
                fortran_index += coordinate * stride
                stride *= dimension
            logical.append(raw[fortran_index])
        payload = struct.pack("<" + "f" * count, *logical)
    elif info.get("fortran_order") is not False:
        raise ValueError("checkpoint NPY order flag is malformed")
    return shape, payload


def npy_zero(shape: tuple[int, ...]) -> bytes:
    header = ("{'descr': '<f4', 'fortran_order': False, 'shape': "
              + repr(shape) + ", }").encode("latin1")
    header += b" " * ((16 - ((10 + len(header) + 1) % 16)) % 16) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\x00" * (4 * math.prod(shape))


def parse_dump(data: bytes) -> dict[str, tuple[tuple[int, ...], list[str]]]:
    tokens = data.decode("utf-8").split()
    index = 0
    result: dict[str, tuple[tuple[int, ...], list[str]]] = {}
    while index < len(tokens):
        if tokens[index] != "TENSOR" or index + 2 >= len(tokens):
            raise ValueError("malformed tensors.txt marker")
        name, rank = tokens[index + 1], int(tokens[index + 2]); index += 3
        shape = tuple(int(tokens[index + i]) for i in range(rank)); index += rank
        count = int(tokens[index]); index += 1
        if count != math.prod(shape) or name in result:
            raise ValueError(f"malformed or duplicate tensor: {name}")
        values = tokens[index:index + count]; index += count
        if len(values) != count:
            raise ValueError(f"truncated tensor: {name}")
        result[name] = (shape, values)
    return result


def _pathless_manifest(manifest: object) -> None:
    """Reject a copied source pointer anywhere in the rewritten manifest."""

    if isinstance(manifest, dict):
        values = manifest.values()
    elif isinstance(manifest, list):
        values = manifest
    else:
        values = (manifest,)
    for value in values:
        if isinstance(value, (dict, list)):
            _pathless_manifest(value)
        elif isinstance(value, str) and (
            "/home/pzh/project/" in value or value.startswith("/home/pzh/")
        ):
            raise ValueError("converted checkpoint manifest retains an external absolute path")


def _head_base(item: dict[str, object]) -> str:
    kernel = item.get("seq_kernel")
    if not isinstance(kernel, str) or kernel.count("/") < 2:
        raise ValueError("checkpoint head seq_kernel is malformed")
    return kernel.rsplit("/", 2)[0]


def _complete_head(
    item: dict[str, object],
    tensors: dict[str, tuple[tuple[int, ...], list[str]]],
    extra: list[tuple[str, tuple[int, ...]]],
    *,
    hidden: bool,
    class_count: int,
) -> dict[str, object]:
    """Add only structural zero attention biases to a source head registry.

    Main checkpoints intentionally omit the unused attention biases on both
    hidden and output heads.  Hidden output bias is also structurally absent;
    an output projection bias, however, is a learned parameter and must be
    present in the source bundle.  A pre-existing hidden bias is rejected
    instead of being silently converted.
    """

    result = dict(item)
    base = _head_base(result)
    bias_names = {
        "attn_src_bias": base + "/attn_src/bias",
        "attn_dst_bias": base + "/attn_dst/bias",
        "output_bias": base + "/output_bias",
    }
    result.update(bias_names)
    if hidden:
        expected = {
            bias_names["attn_src_bias"]: (1,),
            bias_names["attn_dst_bias"]: (1,),
            bias_names["output_bias"]: (8,),
        }
        for name, shape in expected.items():
            if name in tensors:
                raise ValueError(f"source checkpoint contains a hidden bias tensor: {name}")
            extra.append((name, shape))
    else:
        # Output output_bias is part of the canonical formal relation.  It is
        # not legal to synthesize or drop this learned parameter.
        output_bias = bias_names["output_bias"]
        if output_bias not in tensors:
            raise ValueError(f"source checkpoint is missing learned output bias: {output_bias}")
        if tensors[output_bias][0] != (class_count,):
            raise ValueError(f"source checkpoint output bias shape mismatch: {output_bias}")
        for field in ("attn_src_bias", "attn_dst_bias"):
            name = bias_names[field]
            if name in tensors:
                raise ValueError(f"source checkpoint contains unsupported output attention bias: {name}")
            extra.append((name, (1,)))
    return result


def convert_one(source: Path, destination: Path) -> dict[str, object]:
    manifest_bytes = (source / "manifest.json").read_bytes()
    npz_bytes = (source / "tensors.npz").read_bytes()
    text_bytes = (source / "tensors.txt").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("L") != 3 or manifest.get("K_out") != 2 or manifest.get("hidden_profile") != [
        {"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}
    ]:
        raise ValueError(f"{source}: source checkpoint is not canonical L3/[8x8,8x8]/K2")
    if manifest.get("input_preprocess_id") not in ("row_normalize", "ogbn_arxiv_raw_float32"):
        raise ValueError(f"{source}: source preprocessing identity is not canonical")
    if "checkpoint_prefix" not in manifest:
        raise ValueError(f"{source}: expected source-project checkpoint marker is absent")
    tensors = parse_dump(text_bytes)
    with zipfile.ZipFile(source / "tensors.npz") as archive:
        archive_data = {name: archive.read(name) for name in archive.namelist()}
    index = manifest.get("tensor_index")
    if not isinstance(index, dict) or set(index) != set(tensors):
        raise ValueError(f"{source}: tensors.txt/manifest tensor index mismatch")
    if manifest.get("tensor_count") != len(tensors) or len(archive_data) != len(tensors):
        raise ValueError(f"{source}: source tensor count/archive is not exact")
    for name, (shape, values) in tensors.items():
        item = index[name]
        if not isinstance(item, dict) or not isinstance(item.get("archive_key"), str):
            raise ValueError(f"{source}: malformed tensor index entry: {name}")
        key = item["archive_key"] + ".npy"
        npy_shape, payload = npy_values(archive_data.get(key, b""))
        if tuple(item.get("shape", [])) != shape or npy_shape != shape:
            raise ValueError(f"{source}: logical tensor shape mismatch: {name}")
        if b"".join(struct.pack("<f", float(value)) for value in values) != payload:
            raise ValueError(f"{source}: tensors.txt/NPZ float32 logical payload mismatch: {name}")
    expected_archive = {str(index[name]["archive_key"]) + ".npy" for name in tensors}
    if set(archive_data) != expected_archive:
        raise ValueError(f"{source}: source NPZ contains an unknown or missing tensor archive entry")
    hidden = []
    extra = []
    for item in manifest["hidden_head_specs"]:
        hidden.append(_complete_head(dict(item), tensors, extra, hidden=True, class_count=int(manifest["C"])))
    outputs = [
        _complete_head(dict(item), tensors, extra, hidden=False, class_count=int(manifest["C"]))
        for item in manifest["output_head_specs"]
    ]
    output_text = text_bytes.rstrip() + b"\n"
    output_npz = dict(archive_data)
    if len({name for name, _ in extra}) != len(extra):
        raise ValueError(f"{source}: structural bias conversion generated duplicate tensor names")
    for name, shape in extra:
        archive_key = name.replace("/", "__") + ".npy"
        if archive_key in output_npz:
            raise ValueError(f"{source}: structural bias archive key collides: {archive_key}")
        output_text += (f"TENSOR {name} {len(shape)} " + " ".join(map(str, shape))
                        + f" {math.prod(shape)} " + " ".join("0.0" for _ in range(math.prod(shape))) + "\n").encode()
        output_npz[archive_key] = npy_zero(shape)
    converted = dict(manifest)
    converted.pop("checkpoint_prefix", None)
    converted.pop("family_schema", None)
    converted.update({
        "manifest_kind": "ezkl_canonical_checkpoint_bundle_v1",
        "family_schema_version": "multi_layer_multi_head_v2",
        "hidden_head_specs": hidden,
        "output_head_specs": outputs,
        "source_bundle_digest": framed_digest([
            ("manifest.json", manifest_bytes), ("tensors.npz", npz_bytes), ("tensors.txt", text_bytes)
        ]),
        "source_manifest_sha256": digest(manifest_bytes),
        "source_tensors_npz_sha256": digest(npz_bytes),
        "source_tensors_txt_sha256": digest(text_bytes),
        "hidden_bias_policy": "explicit_zero_structural_tensors",
        "tensor_count": len(output_npz),
        "tensor_index": {
            name: {
                "archive_key": name.replace("/", "__")[:-4]
                if name.replace("/", "__") + ".npy" in output_npz else None,
                "dtype": "float32",
                "shape": list(shape),
            }
            for name, (shape, _) in tensors.items()
        },
    })
    # Replace the placeholder index above with the exact source archive key;
    # source logical names need not follow the slash-to-double-underscore
    # naming convention even though all current canonical bundles do.
    for name, item in index.items():
        converted["tensor_index"][name]["archive_key"] = item["archive_key"]
    for name, shape in extra:
        converted["tensor_index"][name] = {
            "archive_key": name.replace("/", "__"),
            "dtype": "float32",
            "shape": list(shape),
        }
    converted["hidden_head_specs"] = hidden
    converted["output_head_specs"] = outputs
    _pathless_manifest(converted)
    output_npz_path = destination / "tensors.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.convert-", dir=destination.parent))
    output_npz_path = staging / "tensors.npz"
    with zipfile.ZipFile(output_npz_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(output_npz):
            archive.writestr(key, output_npz[key])
    (staging / "tensors.txt").write_bytes(output_text)
    converted["tensors_npz_sha256"] = digest(output_npz_path.read_bytes())
    converted["tensors_txt_sha256"] = digest(output_text)
    (staging / "manifest.json").write_text(
        json.dumps(converted, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    # Publish only after all three files have been written.  The destination
    # has no source-model file; its canonical inventory is exactly this bundle.
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "tensors.npz", "tensors.txt"):
        os.replace(staging / name, destination / name)
    staging.rmdir()
    return {"dataset": manifest["dataset"], "source_bundle_digest": converted["source_bundle_digest"],
            "tensors_npz_sha256": converted["tensors_npz_sha256"], "tensors_txt_sha256": converted["tensors_txt_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=ROOT / "artifacts" / "checkpoints")
    args = parser.parse_args()
    records = []
    for dataset in DATASETS:
        source = args.source_root / f"{dataset}_gat"
        destination = args.destination_root / f"{dataset}_gat"
        records.append(convert_one(source, destination))
    print(json.dumps({"status": "converted", "bundles": records}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
