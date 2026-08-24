#!/usr/bin/env python3
"""Low-cost source/configuration preflight for the formal zkGPT path."""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the native zkGPT GAT route")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "formal_preflight.json")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    base = ROOT.resolve()
    if output != base and base not in output.parents:
        raise SystemExit("output must stay inside comparison/zkGPT")
    capabilities = json.loads((ROOT / "configs" / "formal_capabilities.json").read_text(encoding="utf-8"))
    proof_source = (ROOT / "src" / "zkgpt_gat" / "proof.py").read_text(encoding="utf-8")
    hyrax_source = (ROOT / "src" / "zkgpt_gat" / "native_hyrax.py").read_text(encoding="utf-8")
    bn_source = (ROOT / "src" / "zkgpt_gat" / "bn254.py").read_text(encoding="utf-8")
    gpu_execution_source = (ROOT / "src" / "zkgpt_gat" / "gpu_execution.py").read_text(encoding="utf-8")
    cuda_source = (ROOT / "backend" / "cuda" / "zkgpt_cuda_proof_hot_path.cu").read_text(encoding="utf-8")
    environment = {
        "python3": shutil.which("python3") is not None,
        "numpy": False,
        "nvcc": shutil.which("nvcc") is not None,
        "mcl_optional": (ROOT / "3rd" / "mcl" / "include" / "mcl" / "bn.hpp").is_file(),
    }
    try:
        import numpy  # noqa: F401

        environment["numpy"] = True
    except ModuleNotFoundError:
        pass
    source_checks = {
        "four_segment_layout": '"X_pub", "W", "Aux", "L_qry"' in proof_source,
        "group_hyrax": "bn254-g1-pedersen-hyrax-ipa-v1" in hyrax_source and "hyrax-mle-ipa-v1" in hyrax_source,
        "bn254_group_arithmetic": "class G1Point" in bn_source and "is_on_curve" in bn_source,
        "exact_integer_tables": "_fixed_exp" in proof_source and "_integer_bound_ledger" in proof_source,
        "native_lasso": "lasso" in proof_source.lower() and "structured_table_opening" in proof_source,
        "layered_gkr": "_prove_gkr" in proof_source and "_verify_gkr" in proof_source,
        "terminal_openings": "_verify_formal" in proof_source and "verify_open_payload" in proof_source,
        "cuda_proof_backend": (
            "zkgpt_native_bn254_fr_batch_v3" in cuda_source
            and "ZKGPT_GPU_BATCH_V3" in cuda_source
            and "zkGPT_gpu_proof_messages_v3" in cuda_source
            and "product2_round_coefficients_kernel" in cuda_source
            and "product3_round_coefficients_kernel" in cuda_source
            and "lasso_grand_product" in cuda_source
            and "matmul_sumcheck" in cuda_source
            and "gkr_relation" in cuda_source
            and "fr_modulus" in cuda_source
            and "cudaDeviceSynchronize" in cuda_source
        ),
        "gpu_execution_contract": (
            'GPU_EXECUTION_SCHEMA = "zkGPT_gpu_proof_execution_v3"' in gpu_execution_source
            and 'GPU_EXECUTION_SEMANTICS = "zkgpt_cuda_authors_bn254_fr_proof_messages_v3"' in gpu_execution_source
            and '"cuda_event_elapsed_ms"' in gpu_execution_source
            and '"gpu_uuid"' in gpu_execution_source
            and "validate_gpu_execution_evidence" in gpu_execution_source
        ),
    }
    missing = [name for name, passed in source_checks.items() if not passed]
    report = {
        "schema": "zkGPT_gat_formal_preflight_v3",
        "status": "ready" if not missing else "source_incomplete",
        "formal_experiment_eligible": bool(capabilities.get("formal_experiment_eligible")) and not missing,
        "privacy_profile_status": capabilities.get("privacy_profile_status"),
        "cpu_formal_path": capabilities.get("cpu_formal_path"),
        "gpu_formal_path": capabilities.get("gpu_formal_path"),
        "environment": environment,
        "source_checks": source_checks,
        "source_incomplete": missing,
        "blockers": capabilities.get("blockers", []),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"zkGPT_formal_preflight={report['status']} cpu={report['cpu_formal_path']} gpu={report['gpu_formal_path']}")
    return 0 if report["formal_experiment_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
