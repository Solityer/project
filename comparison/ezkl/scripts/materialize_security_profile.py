#!/usr/bin/env python3
"""Resolve the pinned ezkl security profile against one ceremony SRS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from formal_source_manifest import checked_path, formal_source_id, lexical_baseline_root
from resource_policy import (  # noqa: E402
    CEREMONY_DESCRIPTOR,
    CEREMONY_SRS,
    ROOT,
    atomic_write_json,
    validate_ceremony_material,
)

FORMAL_SEMANTIC_VERSION = "ezkl_formal_gat_semantics_v2"
FORMAL_OPERATION_REGISTRY = ROOT / "configs" / "formal_operation_registry.json"


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
    root = lexical_baseline_root(ROOT)
    try:
        template_path = checked_path(root, args.template, "security profile template")
        descriptor_path = checked_path(root, args.ceremony_descriptor, "ceremony descriptor")
        srs_path = checked_path(root, args.srs_path, "ceremony SRS")
        output_path = checked_path(root, args.output, "resolved security profile", allow_missing=True)
        canonical_template = checked_path(root, root / "configs" / "security_profile.json", "security profile template")
        canonical_descriptor = checked_path(root, root / CEREMONY_DESCRIPTOR, "ceremony descriptor")
        canonical_srs = checked_path(root, root / CEREMONY_SRS, "ceremony SRS")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if template_path != canonical_template:
        raise SystemExit("formal ezkl requires the baseline-local security profile template")
    if descriptor_path != canonical_descriptor or srs_path != canonical_srs:
        raise SystemExit(
            "formal ezkl requires baseline-local resources/ceremony/ceremony.descriptor.json "
            "and resources/ceremony/source.srs"
        )
    try:
        ceremony = validate_ceremony_material(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        template = json.loads(template_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read baseline-local ceremony/profile resource: {exc}") from exc
    source_srs_max_k = ceremony["source_srs_max_k"]
    expected = ceremony["source_srs_sha256"]
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
        "ceremony_descriptor_kind": ceremony["descriptor_kind"],
        "ceremony_id": ceremony["ceremony_id"],
        "source_srs_sha256": expected,
        "source_srs_max_k": source_srs_max_k,
        "ceremony_descriptor_path": str(CEREMONY_DESCRIPTOR),
        "source_srs_path": str(CEREMONY_SRS),
        "semantic_version": FORMAL_SEMANTIC_VERSION,
        "operation_registry_id": formal_digest(
            "ezkl_formal_operation_registry",
            "canonical_json",
            FORMAL_OPERATION_REGISTRY.read_bytes().rstrip(b"\r\n"),
        ),
        "semantic_source_id": formal_source_id(Path(__file__).resolve().parents[1]),
    }
    atomic_write_json(output_path, profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
