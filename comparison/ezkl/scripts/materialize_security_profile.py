#!/usr/bin/env python3
"""Resolve the pinned ezkl security profile against one ceremony SRS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from formal_source_manifest import formal_source_id

FORMAL_SOURCE_SRS_MAX_K = 28
FORMAL_SEMANTIC_VERSION = "ezkl_formal_gat_semantics_v2"
FORMAL_OPERATION_REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "formal_operation_registry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def formal_digest(domain: str, role: str, payload: bytes) -> str:
    prefix = b"GAT-ZKML/ezkl/canonical/v1"
    encoded = len(prefix).to_bytes(8, "little") + prefix
    encoded += len(domain.encode()).to_bytes(8, "little") + domain.encode()
    encoded += (1).to_bytes(8, "little")
    encoded += len(role.encode()).to_bytes(8, "little") + role.encode()
    encoded += len(payload).to_bytes(8, "little") + payload
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--ceremony-descriptor", type=Path, required=True)
    parser.add_argument("--srs-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8"))
    descriptor = json.loads(args.ceremony_descriptor.read_text(encoding="utf-8"))
    if descriptor.get("descriptor_kind") != "ezkl_trusted_ceremony_srs_v1":
        raise SystemExit("unregistered trusted ceremony descriptor")
    if not isinstance(descriptor.get("ceremony_id"), str) or not descriptor["ceremony_id"]:
        raise SystemExit("ceremony descriptor lacks a stable ceremony_id")
    source_srs_max_k = descriptor.get("source_srs_max_k")
    if source_srs_max_k != FORMAL_SOURCE_SRS_MAX_K:
        raise SystemExit("formal ezkl requires ceremony source_srs_max_k exactly 28")
    expected = descriptor.get("source_srs_sha256")
    actual = sha256(args.srs_path)
    if expected != actual:
        raise SystemExit(f"trusted ceremony SRS digest mismatch: expected {expected}, got {actual}")
    profile = {
        "profile_kind": template["profile_kind"],
        "ezkl_version": template["ezkl_version"],
        "halo2_git_commit": template["halo2_git_commit"],
        "curve": template["curve"],
        "scalar_field": template["scalar_field"],
        "polynomial_commitment": template["polynomial_commitment"],
        "multiopen": template["multiopen"],
        "transcript": template["transcript"],
        "challenge": template["challenge"],
        "setup": template["setup"],
        "native_security_level_claim": template["native_security_level_claim"],
        "ceremony_descriptor_kind": descriptor["descriptor_kind"],
        "ceremony_id": descriptor["ceremony_id"],
        "source_srs_sha256": expected,
        "source_srs_max_k": source_srs_max_k,
        "semantic_version": FORMAL_SEMANTIC_VERSION,
        "operation_registry_id": formal_digest(
            "ezkl_formal_operation_registry",
            "canonical_json",
            FORMAL_OPERATION_REGISTRY.read_bytes().rstrip(b"\r\n"),
        ),
        "semantic_source_id": formal_source_id(Path(__file__).resolve().parents[1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
