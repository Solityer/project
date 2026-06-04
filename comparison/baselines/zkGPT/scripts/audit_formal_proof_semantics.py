#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import sys
from typing import Any, Dict, List


EXPECTED_DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]


def main() -> None:
    _ensure_numpy_python()
    parser = argparse.ArgumentParser(description="Audit zkGPT formal proof semantics before accepting comparison results")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()

    zk_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(zk_root / "src"))

    from zkgpt_gat import proof as proof_mod
    from zkgpt_gat.util import digest_json, read_json, write_json

    rows: List[Dict[str, Any]] = []
    rows.extend(_audit_sources(proof_mod, zk_root))

    results_dir = pathlib.Path(args.results_dir).resolve() if args.results_dir else zk_root / "results"
    if not args.source_only:
        rows.extend(_audit_results(results_dir, digest_json, read_json))

    status = "PASS" if all(row["passed"] for row in rows) else "FAIL"
    payload = {"status": status, "rows": rows}
    out_dir = results_dir if results_dir.exists() else zk_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "formal_proof_semantics_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


def _ensure_numpy_python() -> None:
    try:
        import numpy  # noqa: F401
        return
    except ModuleNotFoundError:
        venv_python = pathlib.Path("/home/pzh/project/.venv/bin/python")
        if venv_python.exists() and pathlib.Path(sys.executable) != venv_python:
            os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise


def _row(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _audit_sources(proof_mod: Any, zk_root: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    verify_lookup = inspect.getsource(proof_mod._verify_lookup_family)
    verify_gkr = inspect.getsource(proof_mod._verify_gkr_claim)
    gpu_values = inspect.getsource(proof_mod._gpu_claim_values)
    proof_py = (zk_root / "src" / "zkgpt_gat" / "proof.py").read_text(encoding="utf-8")
    backend_py = (zk_root / "src" / "zkgpt_gat" / "backend.py").read_text(encoding="utf-8")

    rows.append(_row("source.lookup_has_structured_membership_verifier", "_verify_structured_function_membership" in proof_py))
    rows.append(_row("source.lookup_no_product_equality_acceptance", "query_product\")) == to_field_int(int(family[\"table_product" not in verify_lookup))
    rows.append(_row("source.range_direct_argument_protocol_present", "direct_limb_bounded_range_argument" in proof_py and "direct_query_accumulator" in proof_py))
    rows.append(_row("source.gkr_verifier_not_status_or_residual_driven", all(token not in verify_gkr for token in ["mismatch_count", "max_abs_error", "claim.get(\"status\"", "oracle_chunks_equal and"])))
    rows.append(_row("source.gkr_committed_opening_crosscheck", "expected_committed_opening" in verify_gkr and "committed_opening_full == expected_committed_opening" in verify_gkr))
    rows.append(_row("source.gpu_input_excludes_debug_residuals", "mismatch_count" not in gpu_values and "max_abs_error" not in gpu_values))
    rows.append(_row("source.gpu_hot_path_not_digest_helper", "zkgpt_cuda_proof_hot_path" in backend_py and "zkgpt_cuda_digest" not in backend_py and "missing CUDA helper" not in backend_py))
    rows.append(_row("source.combine_verifier_present", "_verify_combine" in proof_py and "fiat-shamir-combine-openmasked-v2" in proof_py))
    return rows


def _audit_results(results_dir: pathlib.Path, digest_json, read_json) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for backend in ["cpu", "gpu"]:
        for dataset in EXPECTED_DATASETS:
            bundle = results_dir / backend / dataset
            rows.extend(_audit_bundle(bundle, backend, dataset, digest_json, read_json))
    summary = results_dir / "summary.csv"
    rows.append(_row("results.summary_exists", summary.exists(), str(summary)))
    if summary.exists():
        text = summary.read_text(encoding="utf-8")
        rows.append(_row("results.summary_has_10_verify_ok_rows", text.count("VERIFY_OK") >= 10))
        forbidden = ["fake-proof", "artifact-replay", "verify-skipped", "forward-only", "residual-only"]
        rows.append(_row("results.summary_no_forbidden_modes", not any(item in text for item in forbidden)))
    return rows


def _audit_bundle(bundle: pathlib.Path, backend: str, dataset: str, digest_json, read_json) -> List[Dict[str, Any]]:
    prefix = f"bundle.{backend}.{dataset}"
    rows: List[Dict[str, Any]] = []
    required = [
        "metadata.json",
        "commitments.json",
        "transcript.json",
        "fixed_lookup_tables.json",
        "lookup_proof.json",
        "matrix_sumcheck_proof.json",
        "gkr_proof.json",
        "combine_proof.json",
        "hyrax_mle_openings.json",
        "masked_openings.json",
        "proof_payload.json",
        "verification.json",
        "witness_manifest.json",
        "benchmark_summary.json",
        "run_manifest.json",
    ]
    if backend == "gpu":
        required.append("gpu_proof_hot_path.json")
    missing = [name for name in required if not (bundle / name).exists()]
    rows.append(_row(f"{prefix}.required_artifacts", not missing, ",".join(missing)))
    if missing:
        return rows

    metadata = read_json(bundle / "metadata.json")
    commitments = read_json(bundle / "commitments.json")
    lookup = read_json(bundle / "lookup_proof.json")
    fixed_tables = read_json(bundle / "fixed_lookup_tables.json")
    matrix = read_json(bundle / "matrix_sumcheck_proof.json")
    gkr = read_json(bundle / "gkr_proof.json")
    combine = read_json(bundle / "combine_proof.json")
    openings = read_json(bundle / "hyrax_mle_openings.json")
    payload = read_json(bundle / "proof_payload.json")
    verification = read_json(bundle / "verification.json")
    witness = read_json(bundle / "witness_manifest.json")
    manifest = read_json(bundle / "run_manifest.json")

    rows.append(_row(f"{prefix}.verified_full_dataset", verification.get("status") == "VERIFY_OK" and manifest.get("status") == "VERIFY_OK" and manifest.get("full_dataset") is True))
    rows.append(_row(f"{prefix}.payload_carries_messages", all(key in payload for key in ["Com_in", "Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Open_in", "OpenMasked"]) and payload.get("Pi_Lasso") == lookup and payload.get("Pi_GKR") == gkr))
    rows.append(_row(f"{prefix}.fixed_table_registry_bound", lookup.get("fixed_table_registry") == fixed_tables and lookup.get("fixed_table_registry_digest") == digest_json(fixed_tables)))
    rows.append(_row(f"{prefix}.commitment_order", commitments.get("order") == ["com_T_H", "com_W", "com_base", "com_comp", "com_psq", "com_qry"]))
    rows.append(_row(f"{prefix}.witness_not_verifier_input", witness.get("storage_policy", "").find("raw witness arrays are not stored as verifier input") >= 0))

    rows.extend(_audit_lookup(prefix, lookup))
    rows.extend(_audit_matrix(prefix, matrix))
    rows.extend(_audit_gkr(prefix, gkr, digest_json))
    rows.append(_row(f"{prefix}.combine_bound", combine.get("protocol") == "fiat-shamir-combine-openmasked-v2" and combine.get("combine_digest") == digest_json({key: value for key, value in combine.items() if key != "combine_digest"})))
    rows.append(_row(f"{prefix}.masked_openings_present", bool(openings.get("items")) and all(value.get("chunks") for value in openings.get("items", {}).values())))
    rows.append(_row(f"{prefix}.metadata_protocol", metadata.get("proof_object") == "pi_GAT_zkGPT=(M_pub,Y,Com_in,Pi_Lasso,Pi_MatMul,Pi_GKR,Pi_combine,Open_in)"))

    if backend == "gpu":
        gpu = read_json(bundle / "gpu_proof_hot_path.json")
        rows.append(_row(f"{prefix}.gpu_hot_path_bound", gpu.get("proof_hot_path") is True and gpu.get("cuda_kernel_executed") is True and gpu.get("cpu_fallback") is False and gpu.get("hot_path_semantics") == "cuda_segment_lookup_matrix_gkr_hot_path_v2"))
        rows.append(_row(f"{prefix}.gpu_payload_included", payload.get("GPU_proof_contribution") == gpu))
    return rows


def _audit_lookup(prefix: str, lookup: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    families = lookup.get("families", [])
    rows.append(_row(f"{prefix}.lookup_families_present", bool(families)))
    structured_ok = True
    range_ok = True
    for family in families:
        relation = family.get("relation")
        if relation == "query_multiset_membership_in_fixed_function_table":
            opening = family.get("structured_table_opening", {})
            argument = opening.get("query_membership_argument", {})
            structured_ok = structured_ok and argument.get("protocol") == "structured-fixed-function-lasso-membership-v2"
            structured_ok = structured_ok and argument.get("fixed_table_key_chunks") == family.get("selected_fixed_table_chunks")
            structured_ok = structured_ok and argument.get("query_key_chunks") == family.get("query_chunks")
            structured_ok = structured_ok and family.get("selected_fixed_table_chunks") != family.get("raw_query_chunks", [])
        if relation in {"direct_limb_bounded_range_argument", "matrix_rescale_rounding_direct_range_argument"}:
            argument = family.get("range_argument", {})
            range_ok = range_ok and not family.get("selected_fixed_table_chunks")
            range_ok = range_ok and argument.get("protocol") == "direct-fixed-range-bounded-limb-argument-v1"
            range_ok = range_ok and str(argument.get("direct_query_accumulator")) == str(family.get("query_product"))
            range_ok = range_ok and str(argument.get("fixed_table_full_product")) == str(family.get("table_product"))
    rows.append(_row(f"{prefix}.lookup_structured_function_membership", structured_ok))
    rows.append(_row(f"{prefix}.lookup_direct_range_not_fake_lasso", range_ok))
    return rows


def _audit_matrix(prefix: str, matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    claims = matrix.get("claims", [])
    rows.append(_row(f"{prefix}.matrix_claims_present", bool(claims)))
    binding_ok = True
    for claim in claims:
        binding = claim.get("rescale_rounding", {})
        binding_ok = binding_ok and binding.get("protocol") == "committed-per-entry-rescale-rounding-v2"
        binding_ok = binding_ok and binding.get("residual_oracle_binding_digest") == binding.get("residual_oracle_binding", {}).get("binding_digest")
        binding_ok = binding_ok and int(binding.get("max_abs_entry_residual_q2", "0")) <= int(binding.get("per_entry_rounding_bound_q2", "0"))
        binding_ok = binding_ok and bool(binding.get("residual_eval_sumcheck"))
    rows.append(_row(f"{prefix}.matrix_rescale_committed_and_bounded", binding_ok))
    return rows


def _audit_gkr(prefix: str, gkr: Dict[str, Any], digest_json) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    claims = gkr.get("claims", [])
    rows.append(_row(f"{prefix}.gkr_claims_present", bool(claims)))
    binding_ok = True
    for claim in claims:
        refs = claim.get("oracle_opening_references", [])
        committed = claim.get("committed_witness_opening_proofs", {})
        binding_ok = binding_ok and claim.get("oracle_opening_reference_digest") == digest_json(refs)
        binding_ok = binding_ok and committed.get("protocol") == "gkr-final-oracle-claims-bound-to-openmasked-segments-v2"
        binding_ok = binding_ok and committed.get("binding_digest") == digest_json({key: value for key, value in committed.items() if key != "binding_digest"})
        binding_ok = binding_ok and bool(claim.get("final_oracle_claims"))
        binding_ok = binding_ok and claim.get("protocol") == "gkr-layer-relation-random-linear-eval-v3"
    rows.append(_row(f"{prefix}.gkr_final_oracles_bound_to_openings", binding_ok))
    return rows


if __name__ == "__main__":
    main()
