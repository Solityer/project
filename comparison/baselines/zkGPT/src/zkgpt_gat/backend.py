from __future__ import annotations

import pathlib
import subprocess
import tempfile
from typing import Any, Dict

from .util import read_json, stable_env, write_json


def gpu_helper_path(zk_root: pathlib.Path) -> pathlib.Path:
    return zk_root / "build" / "zkgpt_cuda_digest"


def run_gpu_digest(zk_root: pathlib.Path, seed_payload: Dict[str, Any]) -> Dict[str, Any]:
    helper = gpu_helper_path(zk_root)
    if not helper.exists():
        raise FileNotFoundError(f"missing CUDA helper: {helper}; run scripts/build_cuda_backend.sh")
    with tempfile.TemporaryDirectory(prefix="zkgpt-gpu-", dir=str(zk_root / "runs_tmp")) as tmp:
        tmp_path = pathlib.Path(tmp)
        seed_path = tmp_path / "seed.json"
        out_path = tmp_path / "gpu_backend.json"
        write_json(seed_path, seed_payload)
        completed = subprocess.run(
            [str(helper), str(seed_path), str(out_path)],
            cwd=str(zk_root),
            env=stable_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "CUDA helper failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        state = read_json(out_path)
        state["cuda_kernel_executed"] = bool(state.get("cuda_kernel_executed"))
        state["helper_stdout"] = completed.stdout.strip()
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

