#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed zkGPT formal capability preflight")
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
    cuda_source = (ROOT / "backend" / "cuda" / "zkgpt_cuda_proof_hot_path.cu").read_text(encoding="utf-8")
    environment = {
        "python3": shutil.which("python3") is not None,
        "numpy": False,
        "nvcc": shutil.which("nvcc") is not None,
        "mcl_headers": (ROOT / "3rd" / "mcl" / "include" / "mcl" / "bn.hpp").is_file(),
        "mcl_library": (ROOT / "3rd" / "mcl" / "lib" / "libmcl.a").is_file(),
    }
    try:
        import numpy  # noqa: F401

        environment["numpy"] = True
    except ModuleNotFoundError:
        pass
    source_evidence = {
        "reference_layout_is_not_four_segment": '"com_T_H"' in proof_source and '"com_psq"' in proof_source,
        "reference_hyrax_uses_deterministic_blinders": "def _blind(label: str, row: int)" in hyrax_source,
        "reference_cuda_is_u64_reducer": "proof_reduce_kernel" in cuda_source and "unsigned long long" in cuda_source,
        "native_lasso_protocol_absent": "sample_indices" in proof_source and "evaluation_samples" in proof_source,
    }
    report = {
        "schema": "zkGPT_gat_formal_preflight_v1",
        "status": "unsupported_privacy_profile",
        "formal_experiment_eligible": False,
        "cpu_formal_path": capabilities["cpu_formal_path"],
        "gpu_formal_path": capabilities["gpu_formal_path"],
        "environment": environment,
        "source_evidence": source_evidence,
        "blockers": capabilities["blockers"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("zkGPT_formal_preflight=BLOCKED privacy_profile=UNSUPPORTED")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
