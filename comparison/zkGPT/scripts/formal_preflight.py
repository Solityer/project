#!/usr/bin/env python3
"""Low-cost source/configuration preflight for the formal zkGPT path."""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.constants import (  # noqa: E402
    DATASET_ORDER,
    FORMAL_GAT_CLASS_COUNTS,
    FORMAL_GAT_D_IN_PROFILE,
    FORMAL_GAT_HIDDEN_PROFILE,
    FORMAL_GAT_K_OUT,
    FORMAL_GAT_LAYER_COUNT,
    FORMAL_GAT_MODEL_ARCH_IDS,
)
from zkgpt_gat.formal_gate import formal_gate_status  # noqa: E402
from zkgpt_gat.resources import check_resource_layout, read_resource_manifest, validate_static_table_descriptor  # noqa: E402
from zkgpt_gat.config import load_app_config  # noqa: E402
from zkgpt_gat.util import assert_contained_path, lexical_path, require_dir, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the native zkGPT GAT route")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "formal_preflight.json")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    base = require_dir(ROOT, label="zkGPT baseline root")
    output = lexical_path(args.output)
    try:
        output = assert_contained_path(base, output, "preflight output", allow_missing=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    # The capability manifest is a source/configuration input.  Its formal
    # blocker is intentionally reported, but source-only mode must not turn it
    # into a source failure or inspect any staged dataset/model resource.
    gate = formal_gate_status()
    capabilities = gate.get("capabilities") or {}
    environment = {
        "python3": shutil.which("python3") is not None,
        "numpy": False,
        "nvcc": shutil.which("nvcc") is not None,
    }
    try:
        import numpy  # noqa: F401

        environment["numpy"] = True
    except ModuleNotFoundError:
        pass
    module_checks = {}
    for module_name in (
        "zkgpt_gat.proof", "zkgpt_gat.native_hyrax", "zkgpt_gat.native_lasso",
        "zkgpt_gat.native_matmul", "zkgpt_gat.native_gkr", "zkgpt_gat.gpu_execution",
    ):
        try:
            __import__(module_name)
            module_checks[module_name] = True
        except Exception:
            module_checks[module_name] = False
    source_checks = {
        "capability_manifest": gate.get("status") == "READY" and isinstance(capabilities, dict),
        "native_python_modules": all(module_checks.values()),
        "cuda_backend_source": (ROOT / "backend" / "cuda" / "zkgpt_cuda_proof_hot_path.cu").is_file(),
        "cuda_build_entrypoint": (ROOT / "scripts" / "build_cuda_backend.sh").is_file(),
        "static_table_descriptor": (ROOT / "resources" / "static_tables" / "exact_integer_gat.json").is_file(),
    }
    config_errors: list[str] = []
    for config_path in sorted((ROOT / "configs" / "formal").glob("*.cfg")) + sorted((ROOT / "configs" / "generated").glob("*/*.cfg")):
        try:
            config = load_app_config(config_path, base)
            dataset = config.dataset
            canonical_dataset = "ogbn-arxiv" if dataset == "ogbn_arxiv" else dataset
            expected_hidden = list(FORMAL_GAT_HIDDEN_PROFILE)
            actual_hidden = [(item.head_count, item.head_dim) for item in config.hidden_profile]
            if config.layer_count != FORMAL_GAT_LAYER_COUNT:
                config_errors.append(f"config_L_not_canonical:{config_path.relative_to(ROOT)}")
            if actual_hidden != expected_hidden:
                config_errors.append(f"config_hidden_profile_not_canonical:{config_path.relative_to(ROOT)}")
            if config.d_in_profile != list(FORMAL_GAT_D_IN_PROFILE[canonical_dataset]):
                config_errors.append(f"config_d_in_profile_not_canonical:{config_path.relative_to(ROOT)}")
            if config.k_out != FORMAL_GAT_K_OUT:
                config_errors.append(f"config_K_out_not_canonical:{config_path.relative_to(ROOT)}")
            if config.num_classes != FORMAL_GAT_CLASS_COUNTS[canonical_dataset]:
                config_errors.append(f"config_class_count_not_canonical:{config_path.relative_to(ROOT)}")
            if config.entries.get("model_arch_id") != FORMAL_GAT_MODEL_ARCH_IDS[canonical_dataset]:
                config_errors.append(f"config_model_arch_not_canonical:{config_path.relative_to(ROOT)}")
        except (OSError, ValueError) as exc:
            config_errors.append(f"config_invalid:{config_path.relative_to(ROOT)}:{exc}")
    source_checks["formal_configs"] = not config_errors
    try:
        read_resource_manifest(ROOT)
        validate_static_table_descriptor(ROOT)
        source_checks["resource_manifest_config"] = True
    except (OSError, ValueError, TypeError):
        source_checks["resource_manifest_config"] = False
    source_missing = [name for name, passed in source_checks.items() if not passed]
    source_missing.extend(config_errors)
    resource_errors: list[str] = []
    if args.source_only:
        # No staged-resource check, no formal gate requirement, and no dataset
        # or checkpoint load is permitted in this mode.
        resource_report = {"skipped": True, "reason": "source_only"}
        resources_ready = False
    else:
        try:
            resource_report = check_resource_layout(ROOT, DATASET_ORDER, require_staged=True)
            resources_ready = all(item.get("staged") is True for item in resource_report.get("datasets", {}).values())
            if not resources_ready:
                resource_errors.append("one or more formal dataset/model resource bundles are not staged")
        except (OSError, ValueError, TypeError) as exc:
            resource_report = {"schema": "zkGPT_baseline_resource_manifest_v1", "datasets": {}}
            resources_ready = False
            resource_errors.append(f"resource_check_failed:{exc}")
    blockers = source_missing + resource_errors
    if not args.source_only:
        blockers = list(gate.get("errors", [])) + blockers
    source_ready = not source_missing
    formal_eligible = source_ready and resources_ready and gate["status"] == "READY"
    if not args.source_only and gate["status"] != "READY" and not gate.get("errors"):
        blockers.append("formal capability gate is blocked")
    if not args.source_only and not resources_ready and not resource_errors:
        blockers.append("resources_ready is false")
    if not args.source_only and not formal_eligible and not blockers:
        blockers.append("formal experiment eligibility is false")
    report = {
        "schema": "zkGPT_gat_formal_preflight_v3",
        "status": "source_ready" if args.source_only and source_ready else ("ready" if formal_eligible else "blocked"),
        "source_ready": source_ready,
        "resources_ready": resources_ready,
        "formal_experiment_eligible": formal_eligible,
        "gate": gate,
        "privacy_profile_status": capabilities.get("privacy_profile_status"),
        "cpu_formal_path": capabilities.get("cpu_formal_path"),
        "gpu_formal_path": capabilities.get("gpu_formal_path"),
        "environment": environment,
        "source_checks": source_checks,
        "module_checks": module_checks,
        "source_incomplete": source_missing,
        "resource_check": resource_report,
        "blockers": blockers,
        "mode": "source_only" if args.source_only else "formal",
    }
    write_json(output, report, root=base, label="formal preflight report")
    print(f"zkGPT_formal_preflight={report['status']} cpu={report['cpu_formal_path']} gpu={report['gpu_formal_path']}")
    return 0 if (report["source_ready"] if args.source_only else report["formal_experiment_eligible"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
