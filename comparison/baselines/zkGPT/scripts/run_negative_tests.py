#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List


def main() -> None:
    zk_root = pathlib.Path(__file__).resolve().parents[1]
    project_root = zk_root.parents[2]
    venv_python = project_root / ".venv" / "bin" / "python"
    if pathlib.Path(sys.executable) != venv_python and venv_python.exists():
        subprocess.run([str(venv_python), str(pathlib.Path(__file__).resolve())], check=True)
        return
    sys.path.insert(0, str(zk_root / "src"))

    from zkgpt_gat.matrix_sumcheck import prove_product_sum, verify_product_sum
    from zkgpt_gat.native_hyrax import NativeHyrax, mle_point_from_u64
    from zkgpt_gat.proof import _verify_combine, _verify_formal, _verify_gkr, _verify_lasso, _verify_matrix
    from zkgpt_gat.util import digest_file, digest_json, read_json, write_json

    out_dir = zk_root / "results" / "negative_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (zk_root / "runs_tmp").mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    hyrax = NativeHyrax(zk_root)
    point = mle_point_from_u64(12345, 2)
    hyrax_work = zk_root / "runs_tmp" / "negative-hyrax"
    proof = hyrax.commit_open("negative.hyrax", [1, 2, 3, 4], point, hyrax_work)
    proof_path = hyrax_work / "negative.hyrax.hyrax.json"
    valid = hyrax.verify_open("negative.hyrax", proof_path, point, hyrax_work)
    rows.append({"case": "hyrax_valid_opening", "expected": "VERIFY_OK", "actual": valid["status"]})

    tampered = copy.deepcopy(proof)
    tampered["opening"]["masked_evaluation"] = "0100000000000000000000000000000000000000000000000000000000000000"
    tampered_path = hyrax_work / "negative.hyrax.tampered.json"
    write_json(tampered_path, tampered)
    tampered_result = hyrax.verify_open("negative.hyrax", tampered_path, point, hyrax_work)
    rows.append({"case": "hyrax_tampered_opening", "expected": "VERIFY_FAILED", "actual": tampered_result["status"]})

    sumcheck = prove_product_sum("negative.matmul", [1, 2, 3, 4], [9, 8, 7, 6], {"case": "valid"}).to_json()
    sumcheck_valid = verify_product_sum("negative.matmul", sumcheck, {"case": "valid"})
    rows.append({"case": "matrix_sumcheck_valid", "expected": "VERIFY_OK", "actual": sumcheck_valid["status"]})

    bad_sumcheck = copy.deepcopy(sumcheck)
    bad_sumcheck["rounds"][0]["coefficients"][0] = str(int(bad_sumcheck["rounds"][0]["coefficients"][0]) + 1)
    sumcheck_bad = verify_product_sum("negative.matmul", bad_sumcheck, {"case": "valid"})
    rows.append({"case": "matrix_sumcheck_tampered_payload", "expected": "VERIFY_FAILED", "actual": sumcheck_bad["status"]})

    required = [zk_root / "results" / "cpu" / "cora", zk_root / "results" / "gpu" / "cora", zk_root / "results" / "cpu" / "ppi"]
    missing = [str(path) for path in required if not (path / "run_manifest.json").exists()]
    if missing:
        raise SystemExit("missing formal result bundle(s) for negative tests: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="zkgpt-negative-bundles-", dir=str(zk_root / "runs_tmp")) as tmp:
        tmp_root = pathlib.Path(tmp)
        rows.extend(_bundle_negative_tests(zk_root, tmp_root, digest_file, digest_json, read_json, write_json, _verify_formal))
        rows.extend(_subproof_relation_negative_tests(zk_root / "results" / "cpu" / "cora", read_json, _verify_lasso, _verify_matrix, _verify_gkr, _verify_combine))

    passed = all(row["actual"] == row["expected"] for row in rows)
    payload = {"status": "PASS" if passed else "FAIL", "rows": rows}
    write_json(out_dir / "formal_negative_tests.json", payload)
    write_json(out_dir / "native_primitive_negative_tests.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def _bundle_negative_tests(zk_root: pathlib.Path, tmp_root: pathlib.Path, digest_file, digest_json, read_json, write_json, verify_formal) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cpu_cora = zk_root / "results" / "cpu" / "cora"
    gpu_cora = zk_root / "results" / "gpu" / "cora"
    cpu_ppi = zk_root / "results" / "cpu" / "ppi"

    rows.append(_expect_bundle("bundle_valid_cpu_cora", cpu_cora, True, digest_file, digest_json, read_json, verify_formal))
    rows.append(_expect_bundle("bundle_valid_gpu_cora", gpu_cora, True, digest_file, digest_json, read_json, verify_formal))

    rows.append(_tamper_case("metadata_tamper", cpu_cora, tmp_root, "metadata.json", lambda data: _set_path(data, ["dim_cfg", "N"], int(data["dim_cfg"]["N"]) + 1), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("commitment_tamper", cpu_cora, tmp_root, "commitments.json", lambda data: _set_path(data, ["items", "com_base", "chunks", 0, "row_commitments", 0], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("fixed_lookup_table_commitment_tamper", cpu_cora, tmp_root, "fixed_lookup_tables.json", lambda data: _set_path(data, ["T_LReLU", "table_commitment"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("fixed_lookup_table_id_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 1, "fixed_table", "table_id"], "T_fake"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("fixed_lookup_table_domain_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 1, "fixed_table", "input_domain", "max_exclusive"], int(data["families"][1]["fixed_table"]["input_domain"]["max_exclusive"]) - 1), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("structured_table_vk_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 1, "structured_table_opening", "structured_verification_key"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("transcript_challenge_tamper", cpu_cora, tmp_root, "transcript.json", lambda data: _set_path(data, ["log", 0, "challenge_u64"], int(data["log"][0]["challenge_u64"]) + 1), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("lookup_query_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 1, "raw_query_chunks", 0, "sha256"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("selected_table_opening_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 1, "structured_table_opening", "evaluation_samples", 0, "expected_y"], str(int(data["families"][1]["structured_table_opening"]["evaluation_samples"][0]["expected_y"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("lookup_challenge_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 1, "eta"], str(int(data["families"][1]["eta"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("formal_proof_semantics_audit_bypass_attempt", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 1, "structured_table_opening", "query_membership_argument", "protocol"], "query-generated-table-side-bypass"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("range_limb_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 0, "range_argument", "query_limb_chunks", 0, "sha256"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("range_table_product_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 0, "table_product"], str(int(data["families"][0]["table_product"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("lookup_proof_tamper", cpu_cora, tmp_root, "lookup_proof.json", lambda data: _set_path(data, ["families", 0, "query_product"], str(int(data["families"][0]["query_product"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("matrix_sumcheck_proof_tamper", cpu_cora, tmp_root, "matrix_sumcheck_proof.json", lambda data: _set_path(data, ["claims", 0, "sumcheck", "rounds", 0, "coefficients", 0], str(int(data["claims"][0]["sumcheck"]["rounds"][0]["coefficients"][0]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("matrix_output_eval_tamper", cpu_cora, tmp_root, "matrix_sumcheck_proof.json", lambda data: _set_path(data, ["claims", 0, "output_eval_q2"], str(int(data["claims"][0]["output_eval_q2"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("matrix_bias_eval_tamper", cpu_cora, tmp_root, "matrix_sumcheck_proof.json", lambda data: _set_path(data, ["claims", 1, "bias_eval_q2"], str(int(data["claims"][1]["bias_eval_q2"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("matrix_rescale_advice_tamper", cpu_cora, tmp_root, "matrix_sumcheck_proof.json", lambda data: _set_path(data, ["claims", 0, "rescale_rounding_advice"], str(int(data["claims"][0]["rescale_rounding_advice"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("matrix_rescale_limb_tamper", cpu_cora, tmp_root, "matrix_sumcheck_proof.json", lambda data: _set_path(data, ["claims", 0, "rescale_rounding", "weighted_abs_limbs", 0], str(int(data["claims"][0]["rescale_rounding"]["weighted_abs_limbs"][0]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("matrix_rounding_bound_tamper", cpu_cora, tmp_root, "matrix_sumcheck_proof.json", lambda data: _set_path(data, ["claims", 0, "rescale_rounding", "per_entry_rounding_bound_q2"], "1"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("gkr_proof_tamper", cpu_cora, tmp_root, "gkr_proof.json", lambda data: _set_path(data, ["claims", 0, "lhs_sumcheck", "rounds", 0, "coefficients", 0], str(int(data["claims"][0]["lhs_sumcheck"]["rounds"][0]["coefficients"][0]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("gkr_final_oracle_claim_tamper", cpu_cora, tmp_root, "gkr_proof.json", lambda data: _set_path(data, ["claims", 0, "final_oracle_claims", "lhs_final_product"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("gkr_seed_tamper", cpu_cora, tmp_root, "gkr_proof.json", lambda data: _set_path(data, ["claims", 0, "lhs_sumcheck", "transcript_seed", "constraint_count"], int(data["claims"][0]["lhs_sumcheck"]["transcript_seed"]["constraint_count"]) + 1), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("gkr_oracle_opening_reference_tamper", cpu_cora, tmp_root, "gkr_proof.json", lambda data: _set_path(data, ["claims", 0, "oracle_opening_references", 0, "segment_vector_sha256"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("committed_witness_opening_proof_tamper", cpu_cora, tmp_root, "gkr_proof.json", lambda data: _set_path(data, ["claims", 0, "committed_witness_opening_proofs", "segments", 0, "opening_chunks", 0, "masked_evaluation"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("combine_proof_tamper", cpu_cora, tmp_root, "combine_proof.json", lambda data: _set_path(data, ["sumcheck", "final_product"], str(int(data["sumcheck"]["final_product"]) + 1)), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("masked_opening_tamper", cpu_cora, tmp_root, "masked_openings.json", lambda data: _set_path(data, ["items", "com_base", 0, "masked_evaluation"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("output_claim_tamper", cpu_cora, tmp_root, "proof_payload.json", lambda data: _set_path(data, ["Y_digest"], "00"), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("ppi_graph_boundary_tamper", cpu_ppi, tmp_root, "run_manifest.json", lambda data: _set_path(data, ["node_ptr", 1], int(data["node_ptr"][1]) + 1), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("gpu_backend_state_tamper", gpu_cora, tmp_root, "gpu_proof_hot_path.json", lambda data: _set_path(data, ["proof_hot_path"], False), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("witness_segment_chunk_tamper", cpu_cora, tmp_root, "commitments.json", lambda data: _set_path(data, ["items", "com_qry", "chunks", 0, "vector_length"], int(data["items"]["com_qry"]["chunks"][0]["vector_length"]) - 1), digest_file, digest_json, read_json, write_json, verify_formal))
    rows.append(_tamper_case("hyrax_opening_proof_tamper", cpu_cora, tmp_root, "hyrax_mle_openings.json", lambda data: _set_path(data, ["items", "com_base", "chunks", 0, "proof", "opening", "t_prime"], "0"), digest_file, digest_json, read_json, write_json, verify_formal))
    return rows


def _subproof_relation_negative_tests(path: pathlib.Path, read_json, verify_lasso, verify_matrix, verify_gkr, verify_combine) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    lookup = read_json(path / "lookup_proof.json")
    bad_lookup = copy.deepcopy(lookup)
    bad_lookup["families"][1]["query_product"] = str(int(bad_lookup["families"][1]["query_product"]) + 1)
    rows.append({"case": "subproof_lookup_grand_product_relation_tamper", "expected": "VERIFY_FAILED", "actual": verify_lasso(bad_lookup)["status"]})

    bad_range = copy.deepcopy(lookup)
    bad_range["families"][0]["table_product"] = str(int(bad_range["families"][0]["table_product"]) + 1)
    rows.append({"case": "subproof_lookup_range_selected_table_tamper", "expected": "VERIFY_FAILED", "actual": verify_lasso(bad_range)["status"]})

    bad_challenge = copy.deepcopy(lookup)
    bad_challenge["families"][1]["beta"] = str(int(bad_challenge["families"][1]["beta"]) + 1)
    rows.append({"case": "subproof_lookup_challenge_tamper", "expected": "VERIFY_FAILED", "actual": verify_lasso(bad_challenge)["status"]})

    bad_table = copy.deepcopy(lookup)
    bad_table["families"][1]["fixed_table"]["table_commitment"] = "0"
    rows.append({"case": "subproof_fixed_lookup_table_relation_tamper", "expected": "VERIFY_FAILED", "actual": verify_lasso(bad_table)["status"]})

    matrix = read_json(path / "matrix_sumcheck_proof.json")
    bad_matrix_output = copy.deepcopy(matrix)
    bad_matrix_output["claims"][0]["output_eval_q2"] = str(int(bad_matrix_output["claims"][0]["output_eval_q2"]) + 1)
    rows.append({"case": "subproof_matrix_output_equation_tamper", "expected": "VERIFY_FAILED", "actual": verify_matrix(bad_matrix_output)["status"]})

    bad_matrix_rescale = copy.deepcopy(matrix)
    bad_matrix_rescale["claims"][0]["rescale_rounding_advice"] = str(int(bad_matrix_rescale["claims"][0]["rescale_rounding_advice"]) + 1)
    rows.append({"case": "subproof_matrix_rescale_equation_tamper", "expected": "VERIFY_FAILED", "actual": verify_matrix(bad_matrix_rescale)["status"]})

    bad_matrix_limb = copy.deepcopy(matrix)
    bad_matrix_limb["claims"][0]["rescale_rounding"]["weighted_abs_limbs"][0] = str(int(bad_matrix_limb["claims"][0]["rescale_rounding"]["weighted_abs_limbs"][0]) + 1)
    rows.append({"case": "subproof_matrix_rescale_range_binding_tamper", "expected": "VERIFY_FAILED", "actual": verify_matrix(bad_matrix_limb)["status"]})

    gkr = read_json(path / "gkr_proof.json")
    bad_gkr_round = copy.deepcopy(gkr)
    bad_gkr_round["claims"][0]["lhs_sumcheck"]["rounds"][0]["coefficients"][0] = str(int(bad_gkr_round["claims"][0]["lhs_sumcheck"]["rounds"][0]["coefficients"][0]) + 1)
    rows.append({"case": "subproof_gkr_round_relation_tamper", "expected": "VERIFY_FAILED", "actual": verify_gkr(bad_gkr_round)["status"]})

    bad_gkr_final = copy.deepcopy(gkr)
    bad_gkr_final["claims"][0]["final_oracle_claims"]["lhs_final_product"] = "0"
    rows.append({"case": "subproof_gkr_final_oracle_relation_tamper", "expected": "VERIFY_FAILED", "actual": verify_gkr(bad_gkr_final)["status"]})

    bad_gkr_seed = copy.deepcopy(gkr)
    bad_gkr_seed["claims"][0]["lhs_sumcheck"]["transcript_seed"]["constraint_count"] += 1
    rows.append({"case": "subproof_gkr_fiat_shamir_seed_tamper", "expected": "VERIFY_FAILED", "actual": verify_gkr(bad_gkr_seed)["status"]})

    bad_gkr_opening = copy.deepcopy(gkr)
    bad_gkr_opening["claims"][0]["oracle_opening_references"][0]["segment_vector_sha256"] = "0"
    rows.append({"case": "subproof_gkr_oracle_opening_reference_tamper", "expected": "VERIFY_FAILED", "actual": verify_gkr(bad_gkr_opening)["status"]})

    combine = read_json(path / "combine_proof.json")
    lasso = read_json(path / "lookup_proof.json")
    matrix = read_json(path / "matrix_sumcheck_proof.json")
    gkr = read_json(path / "gkr_proof.json")
    openings = read_json(path / "hyrax_mle_openings.json")
    transcript = read_json(path / "transcript.json")
    bad_combine = copy.deepcopy(combine)
    bad_combine["sumcheck"]["rounds"][0]["challenge"] = str(int(bad_combine["sumcheck"]["rounds"][0]["challenge"]) + 1)
    rows.append({"case": "subproof_combine_sumcheck_tamper", "expected": "VERIFY_FAILED", "actual": verify_combine(bad_combine, lasso, matrix, gkr, openings, transcript)["status"]})
    return rows


def _copy_bundle(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _tamper_case(name: str, source: pathlib.Path, tmp_root: pathlib.Path, file_name: str, mutator, digest_file, digest_json, read_json, write_json, verify_formal) -> Dict[str, Any]:
    dst = tmp_root / name
    _copy_bundle(source, dst)
    path = dst / file_name
    data = read_json(path)
    mutator(data)
    write_json(path, data)
    return _expect_bundle(name, dst, False, digest_file, digest_json, read_json, verify_formal)


def _expect_bundle(name: str, path: pathlib.Path, should_pass: bool, digest_file, digest_json, read_json, verify_formal) -> Dict[str, Any]:
    actual_bool = _verify_bundle(path, digest_file, digest_json, read_json, verify_formal)
    actual = "VERIFY_OK" if actual_bool else "VERIFY_FAILED"
    expected = "VERIFY_OK" if should_pass else "VERIFY_FAILED"
    return {"case": name, "expected": expected, "actual": actual}


def _verify_bundle(path: pathlib.Path, digest_file, digest_json, read_json, verify_formal) -> bool:
    try:
        metadata = read_json(path / "metadata.json")
        commitments = read_json(path / "commitments.json")
        transcript = read_json(path / "transcript.json")
        proof = read_json(path / "proof_payload.json")
        verification = read_json(path / "verification.json")
        witness = read_json(path / "witness_manifest.json")
        manifest = read_json(path / "run_manifest.json")
        lookup = read_json(path / "lookup_proof.json")
        fixed_tables = read_json(path / "fixed_lookup_tables.json")
        matrix = read_json(path / "matrix_sumcheck_proof.json")
        gkr = read_json(path / "gkr_proof.json")
        combine = read_json(path / "combine_proof.json")
        masked = read_json(path / "masked_openings.json")
        hyrax = read_json(path / "hyrax_mle_openings.json")
        gpu = read_json(path / "gpu_proof_hot_path.json") if manifest["backend"] == "gpu" else None
        formal = verify_formal(metadata, commitments, transcript, lookup, matrix, gkr, combine, hyrax, gpu, manifest["backend"])
        checks = [
            proof["M_pub_digest"] == digest_json(metadata),
            proof["transcript_digest"] == digest_json(transcript),
            proof["Y_digest"] == witness["output_digest"],
            proof["Com_in"] == [commitments["items"][name] for name in commitments["order"]],
            proof["Pi_Lasso"] == lookup,
            proof["FixedLookupTables"] == fixed_tables,
            lookup["fixed_table_registry"] == fixed_tables,
            proof["Pi_MatMul"] == matrix,
            proof["Pi_GKR"] == gkr,
            proof["Pi_combine"] == combine,
            proof["Open_in"] == hyrax,
            proof["OpenMasked"] == masked,
            formal["status"] == "VERIFY_OK" and all(bool(value) for value in formal["checks"].values()),
            verification["status"] == formal["status"],
            all(chunk["verify"]["status"] == "VERIFY_OK" for item in hyrax["items"].values() for chunk in item["chunks"]),
            set(masked["items"]) == set(commitments["order"]),
            manifest["node_ptr"] == metadata["dim_cfg"]["node_ptr"],
            manifest["edge_ptr"] == metadata["dim_cfg"]["edge_ptr"],
        ]
        for key in ["FixedLookupTables", "Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Open_in", "OpenMasked"]:
            ref = proof["component_artifacts"][key]
            checks.append(digest_file(path / pathlib.Path(ref["path"]).name) == ref["sha256"])
        if manifest["backend"] == "gpu":
            checks.append(gpu.get("proof_hot_path") is True and gpu.get("cuda_kernel_executed") is True)
            checks.append(digest_file(path / "gpu_proof_hot_path.json") == proof["component_artifacts"]["GPU_hot_path"]["sha256"])
        return all(checks)
    except Exception:
        return False


def _set_path(data: Dict[str, Any], keys: List[Any], value: Any) -> None:
    target: Any = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


if __name__ == "__main__":
    main()
