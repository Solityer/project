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

from zkgpt_gat.constants import DATASET_ORDER  # noqa: E402
from zkgpt_gat.formal_gate import formal_gate_status  # noqa: E402
from zkgpt_gat.resources import check_resource_layout, read_resource_manifest, validate_static_table_descriptor  # noqa: E402
from zkgpt_gat.config import read_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the native zkGPT GAT route")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "formal_preflight.json")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    base = ROOT.resolve()
    if output != base and base not in output.parents:
        raise SystemExit("output must stay inside comparison/zkGPT")
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
        "static_table_descriptor": (ROOT / "resources" / "static_tables" / "exact_integer_gat_v1.json").is_file(),
    }
    config_errors: list[str] = []
    for config_path in sorted((ROOT / "configs" / "formal").glob("*.cfg")) + sorted((ROOT / "configs" / "generated").glob("*/*.cfg")):
        try:
            entries = read_config(config_path)
            if not entries.get("dataset"):
                config_errors.append(f"config_missing_dataset:{config_path.relative_to(ROOT)}")
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"zkGPT_formal_preflight={report['status']} cpu={report['cpu_formal_path']} gpu={report['gpu_formal_path']}")
    return 0 if (report["source_ready"] if args.source_only else report["formal_experiment_eligible"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
