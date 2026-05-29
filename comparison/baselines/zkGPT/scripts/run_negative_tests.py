#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys


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
    from zkgpt_gat.util import read_json, write_json

    out_dir = zk_root / "results" / "negative_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

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

    legacy = subprocess.run(
        [str(venv_python), str(zk_root / "scripts" / "gat_zkgpt_experiment.py"), "--datasets", "cora", "--backends", "cpu"],
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    rows.append(
        {
            "case": "legacy_digest_runner_disabled",
            "expected": "REJECTED",
            "actual": "REJECTED" if legacy.returncode != 0 and "digest/replay adapter is disabled" in legacy.stderr + legacy.stdout else "UNEXPECTED_OK",
        }
    )

    passed = all(row["actual"] == row["expected"] for row in rows)
    payload = {"status": "PASS" if passed else "FAIL", "rows": rows}
    write_json(out_dir / "native_primitive_negative_tests.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
