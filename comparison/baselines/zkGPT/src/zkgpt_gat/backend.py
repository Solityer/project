from __future__ import annotations

import pathlib
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterable

from .util import digest_json, read_json, stable_env, write_json


def gpu_proof_hot_path_binary(zk_root: pathlib.Path) -> pathlib.Path:
    return zk_root / "build" / "zkgpt_cuda_proof_hot_path"


def _run_gpu_reduce_file(zk_root: pathlib.Path, input_path: pathlib.Path, label: str) -> Dict[str, Any]:
    binary = gpu_proof_hot_path_binary(zk_root)
    if not binary.exists():
        raise FileNotFoundError(f"missing CUDA proof hot-path binary: {binary}; run scripts/build_cuda_backend.sh")
    out_path = input_path.with_suffix(".gpu_backend.json")
    env = stable_env()
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("CUDA_CACHE_DISABLE", "1")
    completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, 7):
        completed = subprocess.run(
            [str(binary), str(input_path), str(out_path), label],
            cwd=str(zk_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            break
        if "out of memory" not in completed.stderr.lower() or attempt == 6:
            break
        time.sleep(10.0 * attempt)
    if completed is None or completed.returncode != 0:
        stdout = "" if completed is None else completed.stdout
        stderr = "" if completed is None else completed.stderr
        raise RuntimeError(
            "CUDA proof hot-path execution failed\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    state = read_json(out_path)
    state["cuda_kernel_executed"] = bool(state.get("cuda_kernel_executed"))
    state["cuda_stdout"] = completed.stdout.strip()
    return state


def gpu_proof_reduce(zk_root: pathlib.Path, values: Iterable[int], label: str) -> Dict[str, Any]:
    materialized = [int(value) & ((1 << 64) - 1) for value in values]
    with tempfile.TemporaryDirectory(prefix="zkgpt-gpu-proof-", dir=str(zk_root / "runs_tmp")) as tmp:
        tmp_path = pathlib.Path(tmp)
        input_path = tmp_path / "proof_values.u64"
        input_path.write_text("\n".join(str(value) for value in materialized) + "\n", encoding="utf-8")
        state = _run_gpu_reduce_file(zk_root, input_path, label)
        state["proof_hot_path"] = True
        state["hot_path_semantics"] = "cuda_segment_lookup_matrix_gkr_hot_path_v2"
        state["hot_path_components"] = [
            "segment_encoding_and_masked_opening_claim_materialization",
            "fixed_table_lookup_query_and_selected_table_accumulators",
            "matrix_sumcheck_round_folding_and_signed_rescale_range_bindings",
            "gkr_layer_oracle_sumcheck_and_committed_segment_opening_bindings",
        ]
        state["cpu_fallback"] = False
        state["verifier_binding"] = "input_digest over current run commitments, transcript, lookup, matrix, and GKR proof hot-path field words"
        state["input_count"] = len(materialized)
        state["input_digest"] = digest_json([str(value) for value in materialized])
        return state


def run_gpu_digest(zk_root: pathlib.Path, seed_payload: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="zkgpt-gpu-", dir=str(zk_root / "runs_tmp")) as tmp:
        tmp_path = pathlib.Path(tmp)
        seed_path = tmp_path / "seed_values.u64"
        digest_words = [int(read, 16) for read in _seed_words(seed_payload)]
        seed_path.write_text("\n".join(str(value & ((1 << 64) - 1)) for value in digest_words) + "\n", encoding="utf-8")
        state = _run_gpu_reduce_file(zk_root, seed_path, "backend-probe")
        return state


def backend_state(backend: str, zk_root: pathlib.Path, seed_payload: Dict[str, Any]) -> Dict[str, Any]:
    if backend == "cpu":
        return {
            "backend": "cpu",
            "cuda_kernel_executed": False,
            "gpu_runtime_present": False,
            "note": "CPU proof path",
        }
    if backend == "gpu":
        state = run_gpu_digest(zk_root, seed_payload)
        if not state.get("cuda_kernel_executed"):
            raise RuntimeError("GPU backend did not execute CUDA kernel")
        return state
    raise ValueError(f"unknown backend: {backend}")


def _seed_words(seed_payload: Dict[str, Any]) -> list[str]:
    import hashlib
    import json

    text = json.dumps(seed_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return [digest[index:index + 16] for index in range(0, len(digest), 16)]
