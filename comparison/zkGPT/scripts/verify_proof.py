#!/usr/bin/env python3
"""Replay a zkGPT proof only after external statement authentication.

All model/layout/table identities are supplied through an external
``VerifierContext``.  The serialized artifact is treated as untrusted proof
messages and metadata; its self-description cannot select the context.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.proof import _verify_formal  # noqa: E402
from zkgpt_gat.verifier_context import ContextError, VerifierContext, build_verifier_context  # noqa: E402


def _read(artifact_dir: pathlib.Path, name: str) -> Dict[str, Any]:
    path = artifact_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _invoke_formal_verifier(
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    transcript: Dict[str, Any],
    lasso: Dict[str, Any],
    matrix: Dict[str, Any],
    gkr: Dict[str, Any],
    combine: Dict[str, Any],
    openings: Dict[str, Any],
    gpu_claim: Dict[str, Any] | None,
    backend: str,
    context: VerifierContext,
) -> Dict[str, Any]:
    """Call the strict proof verifier API with the rebuilt context."""
    args = (metadata, commitments, transcript, lasso, matrix, gkr, combine, openings, gpu_claim, backend)
    # This is deliberately an exact call: a verifier implementation without
    # the external_context parameter is not an accepted formal replay path.
    return {**_verify_formal(*args, external_context=context.proof_context()), "external_context_mode": "external_context"}


def verify_artifact_directory(artifact_dir: pathlib.Path, context: VerifierContext) -> Dict[str, Any]:
    metadata = _read(artifact_dir, "metadata.json")
    commitments = _read(artifact_dir, "commitments.json")
    transcript = _read(artifact_dir, "transcript.json")
    lasso = _read(artifact_dir, "lookup_proof.json")
    matrix = _read(artifact_dir, "matrix_sumcheck_proof.json")
    gkr = _read(artifact_dir, "gkr_proof.json")
    combine = _read(artifact_dir, "combine_proof.json")
    openings = _read(artifact_dir, "hyrax_mle_openings.json")
    proof_payload = _read(artifact_dir, "proof_payload.json")
    backend = str(metadata.get("backend", ""))
    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"invalid proof backend: {backend!r}")

    external_metadata = context.validate_metadata(metadata)
    if external_metadata["status"] != "VERIFY_OK":
        return {
            "status": "VERIFY_FAILED",
            "reason": "external verifier context rejected artifact metadata",
            "checks": {"external_context": False},
            "external_context": external_metadata,
        }
    external_payload = context.validate_proof_payload(metadata, proof_payload)
    external_families = context.validate_subproof_families(lasso, matrix, gkr)
    if external_payload["status"] != "VERIFY_OK" or external_families["status"] != "VERIFY_OK":
        return {
            "status": "VERIFY_FAILED",
            "reason": "external verifier context rejected public output or subproof family descriptors",
            "checks": {"external_context": True, "external_public_y": external_payload["status"] == "VERIFY_OK", "external_families": external_families["status"] == "VERIFY_OK"},
            "external_context": external_metadata,
            "external_public_y": external_payload,
            "external_families": external_families,
        }

    gpu_claim = _read(artifact_dir, "gpu_proof_hot_path.json") if backend == "gpu" else None
    result = _invoke_formal_verifier(metadata, commitments, transcript, lasso, matrix, gkr, combine, openings, gpu_claim, backend, context)
    result["external_context"] = external_metadata
    result["external_public_y"] = external_payload
    result["external_families"] = external_families
    result["context_digest"] = context.context_digest
    checks = dict(result.get("checks", {}))
    checks["external_context"] = True
    checks["external_public_y"] = True
    checks["external_families"] = True
    result["checks"] = checks
    if result.get("status") != "VERIFY_OK":
        result["status"] = "VERIFY_FAILED"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a serialized zkGPT proof with an external verifier context")
    parser.add_argument("--project-root", type=pathlib.Path, required=True, help="trusted project root used to resolve config/model/data")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="trusted canonical zkGPT config")
    parser.add_argument("--trusted-registry-public", "--registry-public", dest="registry_public", type=pathlib.Path, required=True, help="trusted model registry public JSON outside artifact")
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True, help="serialized proof artifact directory")
    parser.add_argument("--public-y", type=pathlib.Path, required=True, help="independent public Y JSON/NPY/text file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    try:
        context = build_verifier_context(args.project_root, args.config, args.registry_public, artifact_dir, args.public_y)
        result = verify_artifact_directory(artifact_dir, context)
    except (OSError, ValueError, KeyError, TypeError, ContextError) as exc:
        result = {"status": "VERIFY_FAILED", "reason": str(exc), "checks": {"external_context": False}}
    if artifact_dir.is_dir():
        output = artifact_dir / "independent_verification.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"zkgpt_proof_replay={result.get('status', 'VERIFY_FAILED')} artifact={artifact_dir} context={result.get('context_digest', 'unavailable')}")
    return 0 if result.get("status") == "VERIFY_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
