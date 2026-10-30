"""Narrow producer check for sparse m evaluation admission accounting."""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "tests"))
sys.path.insert(0, str(repo / "scripts"))
import test_run_hybrid_session_schema as fixture_module  # noqa: E402
from formal_common import FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES  # noqa: E402
from run_hybrid_formal import _native_capacity_requirements  # noqa: E402

admission = fixture_module.LocalFormalEntryTest._capacity_admission()
requirements = _native_capacity_requirements(
    admission, {"scalar_bytes": 32, "affine_bytes": 64}
)
memory = admission["native_memory_model"]
fr = int(memory["size_of_fr_bytes"])
usize = int(memory["size_of_usize_bytes"])
vec_header = int(memory["size_of_vec_header_bytes"])
assert fr == 32
assert usize == 8
assert vec_header == 24

# One serial sparse m evaluation owns three bounded vectors and fixed local
# temporaries; this is deliberately not multiplied by lookup/H workers.
sparse_m_eval_scratch = (
    32 * (usize + fr)
    + vec_header
    + 32 * fr
    + vec_header
    + 32 * (fr + usize)
    + vec_header
    + 8 * fr
)
h_fft_transient = requirements["proof_evaluation_fft_transient_host_bytes"]
h_task_and_source = (
    requirements["proof_evaluation_lookup_task_host_bytes"]
    + requirements["lookup_h_source_rebuild_host_window_bytes_upper"]
)
pre_sparse_transient = max(h_fft_transient, h_task_and_source)
expected_eval_transient = max(pre_sparse_transient, sparse_m_eval_scratch)
base = (
    requirements["proof_nonprepare_heap_host_bytes"]
    + requirements["lookup_table_cache_retained_host_bytes"]
    + requirements["proof_evaluation_hot_mmap_host_bytes"]
    + requirements["proof_evaluation_fft_cache_host_bytes"]
    + int(admission["domain_size"]) * fr
    + FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES
)
expected_eval_peak = base + expected_eval_transient
assert requirements["proof_evaluation_host_peak_bytes"] == expected_eval_peak, {
    "actual": requirements["proof_evaluation_host_peak_bytes"],
    "expected": expected_eval_peak,
}
assert expected_eval_transient == max(
    h_fft_transient, h_task_and_source, sparse_m_eval_scratch
)

# Existing sparse-commit preparation accounting remains unchanged; the later
# grand-sum commit still has its separate R-heap/mmap bound.
prepare_bytes = requirements["lookup_prepare_task_host_bytes"]
commit_bytes = requirements["lookup_commit_task_host_bytes"]
assert prepare_bytes == 3248, prepare_bytes
assert commit_bytes == 512, commit_bytes

print(json.dumps({
    "result": "PASS",
    "fixture": "LocalFormalEntryTest._capacity_admission",
    "sparse_m_eval_scratch_single_task_bytes": sparse_m_eval_scratch,
    "h_fft_transient_bytes": h_fft_transient,
    "h_task_plus_source_window_bytes": h_task_and_source,
    "pre_sparse_h_transient_max_bytes": pre_sparse_transient,
    "evaluation_transient_max_bytes": expected_eval_transient,
    "evaluation_peak_bytes": requirements["proof_evaluation_host_peak_bytes"],
    "evaluation_peak_matches_producer_formula": True,
    "lookup_prepare_task_bytes_unchanged": prepare_bytes,
    "lookup_commit_task_bytes_unchanged": commit_bytes,
}, sort_keys=True))
