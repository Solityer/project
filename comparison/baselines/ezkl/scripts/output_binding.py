#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FIELD_FOLD_COUNT = 4
FOLD_BLOCK_CANDIDATES = (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
HASH_LIMB_BITS = 16
HASH_LIMB_COUNT = 16
SUPPORTED_BACKENDS = {"auto", "cryptographic_hash", "field_folding", "identity_public_logits"}
DIGEST_ROUTE_ID = "ezkl_timing_private_input_digest_output_full"
IDENTITY_ROUTE_ID = "ezkl_timing_private_input_public_output_full"
LARGE_DATASETS = {"ppi_full_formal", "ogbn_arxiv_full"}


@dataclass(frozen=True)
class BindingPlan:
    route_id: str
    binding_backend: str


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def choose_binding_plan(dataset_id: str, requested_backend: str, requested_route: str | None = None) -> BindingPlan:
    if requested_backend not in SUPPORTED_BACKENDS:
        known = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"unknown binding backend '{requested_backend}', expected one of: {known}")
    if requested_backend == "cryptographic_hash":
        raise ValueError("cryptographic_hash is not available for this ONNX/ezkl backend")

    if requested_backend == "auto":
        backend = "field_folding" if dataset_id in LARGE_DATASETS else "identity_public_logits"
    else:
        backend = requested_backend

    if dataset_id in LARGE_DATASETS and backend == "identity_public_logits":
        raise ValueError(f"{dataset_id} must use digest/folding binding; identity public logits would create too many public instances")

    route_id = DIGEST_ROUTE_ID if backend == "field_folding" else IDENTITY_ROUTE_ID
    if requested_route not in (None, "auto", route_id):
        raise ValueError(f"route '{requested_route}' is incompatible with binding backend '{backend}'")
    return BindingPlan(route_id=route_id, binding_backend=backend)


def canonical_logits_payload(logits: np.ndarray, metadata: dict[str, Any]) -> bytes:
    logits32 = np.asarray(logits, dtype="<f4", order="C")
    header = {
        "magic": "gatzkml-ezkl-logits-v1",
        "dtype": "float32-le",
        "order": "row-major",
        "shape": list(logits32.shape),
        "metadata": metadata,
    }
    header_bytes = stable_json_bytes(header)
    return len(header_bytes).to_bytes(8, "little") + header_bytes + logits32.tobytes(order="C")


def write_logits_artifact(path: Path, logits: np.ndarray) -> None:
    logits32 = np.asarray(logits, dtype="<f4", order="C")
    path.write_bytes(logits32.tobytes(order="C"))


def read_logits_artifact(path: Path, shape: list[int]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"logits artifact element count mismatch: {raw.size} != {expected}")
    return raw.reshape(tuple(shape))


def hash_to_u16_limbs(digest_hex: str) -> list[int]:
    raw = bytes.fromhex(digest_hex)
    if len(raw) != 32:
        raise ValueError("expected sha256 digest hex")
    return [int.from_bytes(raw[index : index + 2], "little") for index in range(0, 32, 2)]


def shape_hash(logits_shape: list[int], dtype: str, quant_cfg_id: str) -> str:
    return sha256_hex(
        stable_json_bytes(
            {
                "dtype": dtype,
                "logits_shape": logits_shape,
                "order": "row-major",
                "quant_cfg_id": quant_cfg_id,
            }
        )
    )


def metadata_hash(metadata: dict[str, Any]) -> str:
    return sha256_hex(stable_json_bytes(metadata))


def challenge_seed(route_id: str, dataset_id: str, metadata_digest: str, artifact_digest: str, fold_index: int) -> str:
    return sha256_hex(
        stable_json_bytes(
            {
                "domain": f"fold_{fold_index}",
                "route_id": route_id,
                "dataset": dataset_id,
                "metadata_hash": metadata_digest,
                "output_artifact_hash": artifact_digest,
            }
        )
    )


def choose_fold_block_size(element_count: int) -> int:
    for candidate in FOLD_BLOCK_CANDIDATES:
        if element_count % candidate == 0:
            return candidate
    return 1


def challenge_matrix(
    *,
    element_count: int,
    route_id: str,
    dataset_id: str,
    metadata_digest: str,
    artifact_digest: str,
    fold_count: int = FIELD_FOLD_COUNT,
    block_size: int | None = None,
) -> tuple[np.ndarray, list[str], int]:
    block_size = choose_fold_block_size(element_count) if block_size is None else block_size
    if element_count % block_size != 0:
        raise ValueError(f"element_count={element_count} is not divisible by fold block size {block_size}")
    matrix = np.empty((block_size, fold_count), dtype=np.float32)
    seeds: list[str] = []
    for fold_index in range(fold_count):
        seed = challenge_seed(route_id, dataset_id, metadata_digest, artifact_digest, fold_index)
        seeds.append(seed)
        signs = np.frombuffer(hashlib.shake_256(bytes.fromhex(seed)).digest(block_size), dtype=np.uint8)
        matrix[:, fold_index] = (signs & 1).astype(np.float32) * 2.0 - 1.0
    return matrix, seeds, block_size


def public_hash_limb_values(shape_digest: str, metadata_digest: str, artifact_digest: str) -> np.ndarray:
    limbs = hash_to_u16_limbs(shape_digest) + hash_to_u16_limbs(metadata_digest) + hash_to_u16_limbs(artifact_digest)
    return np.asarray(limbs, dtype=np.float32).reshape(1, -1)


def field_fold_values(logits: np.ndarray, challenge_values: np.ndarray) -> np.ndarray:
    block_size = int(challenge_values.shape[0])
    flat = np.asarray(logits, dtype=np.float32, order="C").reshape(-1, block_size)
    column_sums = np.sum(flat, axis=0, dtype=np.float32).reshape(1, block_size)
    return (column_sums @ challenge_values.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def build_binding(
    *,
    dataset_id: str,
    route_id: str,
    binding_backend: str,
    logits: np.ndarray,
    metadata: dict[str, Any],
    output_logits_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    logits32 = np.asarray(logits, dtype=np.float32, order="C")
    logits_shape = list(logits32.shape)
    element_count = int(logits32.size)
    write_logits_artifact(output_logits_path, logits32)
    shape_digest = shape_hash(logits_shape, "float32-le", str(metadata["quant_cfg_id"]))
    metadata_digest = metadata_hash(metadata)
    artifact_digest = sha256_hex(canonical_logits_payload(logits32, metadata))

    if binding_backend == "identity_public_logits":
        public_values = logits32.reshape(1, -1)
        challenge_values = None
        challenge_seeds: list[str] = []
        fold_values: list[float] = []
        binding_security = "identity_public_logits"
        binding_note = "full logits are public outputs; used only when public instance count is small enough"
    elif binding_backend == "field_folding":
        challenge_values, challenge_seeds, fold_block_size = challenge_matrix(
            element_count=element_count,
            route_id=route_id,
            dataset_id=dataset_id,
            metadata_digest=metadata_digest,
            artifact_digest=artifact_digest,
        )
        folds = field_fold_values(logits32, challenge_values)
        public_values = np.concatenate(
            [folds, public_hash_limb_values(shape_digest, metadata_digest, artifact_digest)],
            axis=1,
        ).astype(np.float32, copy=False)
        fold_values = [float(value) for value in folds.reshape(-1)]
        binding_security = "random_linear_folding_not_collision_resistant"
        binding_note = (
            "used only as ezkl public-instance compression for timing baseline; "
            "full logits artifact remains available for external equality check"
        )
    elif binding_backend == "cryptographic_hash":
        raise ValueError("cryptographic_hash is not available for this ONNX/ezkl backend")
    else:
        raise ValueError(f"unsupported binding backend: {binding_backend}")
    if binding_backend != "field_folding":
        fold_block_size = None

    after_count = int(public_values.size)
    binding = {
        "dataset": dataset_id,
        "route_id": route_id,
        "binding_backend": binding_backend,
        "full_dataset": True,
        "logits_shape": logits_shape,
        "logits_element_count": element_count,
        "public_output_element_count_before": element_count,
        "public_output_element_count_after": after_count,
        "public_reduction_ratio": float(element_count / max(1, after_count)),
        "metadata_hash": metadata_digest,
        "output_artifact_hash": artifact_digest,
        "shape_hash": shape_digest,
        "fold_values": fold_values,
        "digest_values": [],
        "challenge_values": challenge_seeds,
        "fold_block_size": fold_block_size,
        "canonical_serialization_rule": {
            "artifact_file": "raw float32 little-endian tensor in row-major order",
            "hash_payload": "uint64_le(header_len) || stable_json(header) || raw_float32_le_row_major_logits",
            "dtype": "float32-le",
            "shape": logits_shape,
            "field_encoding_rule": "ezkl fixed-point signed field encoding with input_scale from settings; folding coefficients are deterministic block-cyclic +/-1 float32 constants",
        },
        "binding_security": binding_security,
        "binding_note": binding_note,
        "hash_limb_bits": HASH_LIMB_BITS,
        "hash_limb_count_per_hash": HASH_LIMB_COUNT,
        "public_output_layout": {
            "fold_values": [0, FIELD_FOLD_COUNT] if binding_backend == "field_folding" else None,
            "shape_hash_u16_limbs": [FIELD_FOLD_COUNT, FIELD_FOLD_COUNT + HASH_LIMB_COUNT] if binding_backend == "field_folding" else None,
            "metadata_hash_u16_limbs": [FIELD_FOLD_COUNT + HASH_LIMB_COUNT, FIELD_FOLD_COUNT + 2 * HASH_LIMB_COUNT] if binding_backend == "field_folding" else None,
            "output_artifact_hash_u16_limbs": [FIELD_FOLD_COUNT + 2 * HASH_LIMB_COUNT, FIELD_FOLD_COUNT + 3 * HASH_LIMB_COUNT] if binding_backend == "field_folding" else None,
        },
        "reads_full_logits_input": True,
        "binding_computed_inside_onnx": True,
    }
    return binding, public_values, challenge_values


def verify_binding_files(
    *,
    logits_path: Path,
    binding_path: Path,
    model_info_path: Path,
    input_json_path: Path | None = None,
) -> dict[str, Any]:
    binding = json.loads(binding_path.read_text())
    model_info = json.loads(model_info_path.read_text())
    logits = read_logits_artifact(logits_path, binding["logits_shape"])
    metadata = model_info["binding_metadata"]
    recomputed_binding, public_values, _ = build_binding(
        dataset_id=binding["dataset"],
        route_id=binding["route_id"],
        binding_backend=binding["binding_backend"],
        logits=logits,
        metadata=metadata,
        output_logits_path=logits_path,
    )
    checks = {
        "logits_file_exists": logits_path.exists(),
        "shape_matches": list(logits.shape) == binding["logits_shape"],
        "metadata_hash_matches": recomputed_binding["metadata_hash"] == binding["metadata_hash"],
        "output_artifact_hash_matches": recomputed_binding["output_artifact_hash"] == binding["output_artifact_hash"],
        "shape_hash_matches": recomputed_binding["shape_hash"] == binding["shape_hash"],
        "fold_values_match": recomputed_binding.get("fold_values") == binding.get("fold_values"),
        "challenge_values_match": recomputed_binding.get("challenge_values") == binding.get("challenge_values"),
        "public_output_count_matches": recomputed_binding["public_output_element_count_after"] == binding["public_output_element_count_after"],
        "input_json_public_output_matches": True,
    }
    if input_json_path is not None and input_json_path.exists():
        payload = json.loads(input_json_path.read_text())
        expected = np.asarray(payload.get("output_data", [[]])[0], dtype=np.float32)
        checks["input_json_public_output_matches"] = bool(
            expected.shape == public_values.reshape(-1).shape
            and np.allclose(expected, public_values.reshape(-1), rtol=0.0, atol=1e-3)
        )
    verified = all(bool(value) for value in checks.values())
    return {
        "output_binding_verified": verified,
        "checks": checks,
        "binding_backend": binding["binding_backend"],
        "route_id": binding["route_id"],
    }
