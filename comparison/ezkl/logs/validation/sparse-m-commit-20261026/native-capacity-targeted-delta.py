"""Narrow check for the sparse-m prepare/commit budget delta only."""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "tests"))
import test_run_hybrid_session_schema as fixture_module  # noqa: E402
from run_hybrid_formal import _native_capacity_requirements  # noqa: E402

admission = fixture_module.LocalFormalEntryTest._capacity_admission()
requirements = _native_capacity_requirements(
    admission, {"scalar_bytes": 32, "affine_bytes": 64}
)
prepare_base = 16 * 32 + 4 * (8 + 8) + 48 + 120 + 2 * 24
sparse_pair_scratch = 32 * (8 + 32) + 24
cpu_projective_scratch = 8 * 144
expected = {
    "lookup_prepare_task_host_bytes": prepare_base + sparse_pair_scratch + cpu_projective_scratch,
    "lookup_commit_task_host_bytes": 13 * 32 + 4 * 24,
}
actual = {key: requirements[key] for key in expected}
assert actual == expected, {"actual": actual, "expected": expected}
print(json.dumps({
    "result": "PASS",
    "prepare_base_bytes": prepare_base,
    "sparse_pair_scratch_bytes": sparse_pair_scratch,
    "cpu_projective_scratch_bytes": cpu_projective_scratch,
    "actual": actual,
    "expected": expected,
}, sort_keys=True))
