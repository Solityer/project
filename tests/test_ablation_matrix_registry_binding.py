#!/usr/bin/env python3
"""Fail-closed tests for the ablation registry/SRS manifest binding helper."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATION_ROOT = PROJECT_ROOT / "tests" / ".core_execution" / "ablation_binding_20261009"
HELPER_PATH = PROJECT_ROOT / "scripts" / "ablation_matrix_support.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("ablation_matrix_support_registry_binding_test", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import registry binding helper from {HELPER_PATH}")
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_manifest(path: pathlib.Path) -> bytes:
    prefix = (
        b'{\n'
        b'  "srs_id": "binding-test-srs",\n'
        b'  "protocol_schema_id": "' + b"a" * 64 + b'",\n'
        b'  "degree_bound": 2,\n'
    )
    raw = prefix + b'  "manifest_hash": "' + hashlib.sha256(prefix).hexdigest().encode("ascii") + b'"\n}\n'
    path.write_bytes(raw)
    return raw


def build_fixture(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict[str, str], dict[str, object], str]:
    srs_root = root / "srs"
    srs_root.mkdir()
    manifest_path = srs_root / "srs_manifest.json"
    raw_manifest = write_manifest(manifest_path)
    registry_path = root / "variant.registry"
    registry_path.write_bytes(b"isolated registry binding test bytes\n")
    digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    config_path = root / "variant.cfg"
    config_path.write_text(
        "\n".join(
            (
                "dataset = cora",
                "model_arch_id = arch-test",
                "model_param_id = params-test",
                "quant_cfg_id = quant-test",
                "static_table_id = tables-test",
                "srs_id = binding-test-srs",
                f"srs_path = {srs_root}",
                "protocol_variant = wo_shared_neighborhood",
                f"verifier_registry_path = {registry_path}",
                f"verifier_registry_digest = {digest}",
                "",
            )
        ),
        encoding="utf-8",
    )
    inspection: dict[str, object] = {
        "artifact_sha256": digest,
        "registry_digest": digest,
        "protocol_variant": "wo_shared_neighborhood",
        "security_budget_passed": True,
        "srs_id": "binding-test-srs",
        "srs_manifest_hash": hashlib.sha256(raw_manifest).hexdigest(),
        "protocol_schema_id": "a" * 64,
        "srs_degree_bound": 2,
        "dataset": "cora",
        "model_arch_id": "arch-test",
        "model_param_id": "params-test",
        "quant_cfg_id": "quant-test",
        "static_table_id": "tables-test",
        "instance_schema_hash": "b" * 64,
    }
    inspection_path = root / "inspection.json"
    inspection_path.write_text(json.dumps(inspection), encoding="utf-8")
    return config_path, registry_path, inspection_path, inspection, inspection, digest


def check(config: pathlib.Path, registry: pathlib.Path, inspection: dict[str, object], digest: str, inspection_path: pathlib.Path) -> None:
    inspection_path.write_text(json.dumps(inspection), encoding="utf-8")
    SUPPORT.check_registry(
        argparse.Namespace(
            project_root=str(PROJECT_ROOT),
            config=str(config),
            inspection=str(inspection_path),
            registry=str(registry),
            registry_digest=digest,
            variant="wo_shared_neighborhood",
        )
    )


def expect_rejected(label: str, message_fragment: str, operation) -> None:
    try:
        operation()
    except ValueError as error:
        require(message_fragment in str(error), f"{label} failed at an unexpected check: {error}")
    else:
        raise AssertionError(f"{label} was accepted")


def run() -> None:
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="registry_binding_", dir=VALIDATION_ROOT) as temp:
        root = pathlib.Path(temp)
        config, registry, inspection_path, good_inspection, _, digest = build_fixture(root)
        check(config, registry, good_inspection, digest, inspection_path)

        manifest_path = root / "srs" / "srs_manifest.json"
        original = manifest_path.read_bytes()

        broken_self_hash = original.replace(
            json.loads(original)["manifest_hash"].encode("ascii"), b"0" * 64, 1
        )
        manifest_path.write_bytes(broken_self_hash)
        self_hash_inspection = dict(good_inspection)
        self_hash_inspection["srs_manifest_hash"] = hashlib.sha256(broken_self_hash).hexdigest()
        expect_rejected(
            "damaged SRS self-hash",
            "canonical hash field is invalid",
            lambda: check(config, registry, self_hash_inspection, digest, inspection_path),
        )

        manifest_path.write_bytes(original + b"\n")
        expect_rejected(
            "manifest changed after native inspection",
            "full-file hash differs",
            lambda: check(config, registry, good_inspection, digest, inspection_path),
        )

        manifest_path.write_bytes(original)
        wrong_pin_inspection = dict(good_inspection)
        wrong_pin = "f" * 64
        if wrong_pin == good_inspection["srs_manifest_hash"]:
            wrong_pin = "e" * 64
        wrong_pin_inspection["srs_manifest_hash"] = wrong_pin
        expect_rejected(
            "registry inspection reports a wrong SRS manifest pin",
            "full-file hash differs",
            lambda: check(config, registry, wrong_pin_inspection, digest, inspection_path),
        )

    print("ABLA_REGISTRY_BINDING_TEST_OK positive=1 negative_self_hash=1 negative_full_file_bytes=1 negative_srs_pin=1")


if __name__ == "__main__":
    run()
