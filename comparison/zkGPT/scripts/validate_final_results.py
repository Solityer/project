#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    capabilities = json.loads((ROOT / "configs" / "formal_capabilities.json").read_text(encoding="utf-8"))
    result_path = ROOT / "results" / "final_zkgpt_results.json"
    errors: list[str] = []
    if capabilities.get("formal_experiment_eligible") is not False:
        errors.append("capability_manifest_must_remain_fail_closed_until_all_protocol_requirements_are_implemented")
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = result.get("rows", [])
        if len(rows) != 10:
            errors.append("blocked_matrix_row_count_not_10")
        for row in rows:
            if row.get("proof_generated") is not False or row.get("verified") is not False:
                errors.append(f"invalid_formal_success_claim:{row.get('dataset')}/{row.get('backend')}")
            if row.get("status") != "unsupported_privacy_profile":
                errors.append(f"invalid_blocked_status:{row.get('dataset')}/{row.get('backend')}")
    audit = {
        "schema": "zkGPT_gat_formal_validation_v1",
        "baseline": "zkGPT",
        "status": "privacy_blocked" if not errors else "failed",
        "formal_experiment_eligible": False,
        "privacy_profile_status": "unsupported_privacy_profile",
        "errors": errors,
        "blockers": capabilities.get("blockers", []),
    }
    output = ROOT / "results" / "local_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        print(f"zkgpt_formal_validation=FAILED errors={len(errors)}")
        return 1
    print("zkgpt_formal_validation=BLOCKED privacy_profile=UNSUPPORTED")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
