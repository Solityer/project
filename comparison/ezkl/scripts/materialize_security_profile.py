#!/usr/bin/env python3
"""Resolve the pinned ezkl security profile against one ceremony SRS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
    if not isinstance(source_srs_max_k, int) or source_srs_max_k <= 0:
        raise SystemExit("ceremony descriptor must provide a positive source_srs_max_k")
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
