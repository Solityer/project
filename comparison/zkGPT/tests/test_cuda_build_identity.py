from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import unittest

from zkgpt_gat.build_identity import (
    CUDA_BINARY_RELATIVE_PATH,
    build_manifest_path,
    verify_build_manifest,
    write_build_manifest,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
NVCC = pathlib.Path(
    os.environ.get("ZKGPT_NVCC_BIN") or shutil.which("nvcc") or "__missing_nvcc__"
)


def _copy_build_inputs(destination: pathlib.Path) -> pathlib.Path:
    for relative in ("src", "scripts", "backend", "CMakeLists.txt", "requirements.lock"):
        source = ROOT / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    binary = destination / CUDA_BINARY_RELATIVE_PATH
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / CUDA_BINARY_RELATIVE_PATH, binary)
    return binary


class CudaBuildIdentityTests(unittest.TestCase):
    @unittest.skipUnless(NVCC.is_file(), "nvcc is not installed")
    def test_source_binary_and_manifest_staleness_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-build-identity-") as raw:
            root = pathlib.Path(raw)
            binary = _copy_build_inputs(root)
            write_build_manifest(
                root,
                binary=binary,
                nvcc=NVCC,
                target_compute_capability="12.1",
                arch_flag="121",
                compile_command=[
                    str(NVCC), "-std=c++17", "-O3", "-lineinfo",
                    "-gencode=arch=compute_121,code=sm_121",
                    str(root / "backend/cuda/zkgpt_cuda_proof_hot_path.cu"),
                    "-o", str(binary),
                ],
            )
            self.assertEqual(verify_build_manifest(root)["target"]["code"], "sm_121")

            source = root / "src/zkgpt_gat/backend.py"
            original_source = source.read_bytes()
            source.write_bytes(original_source + b"\n")
            with self.assertRaises(ValueError):
                verify_build_manifest(root)
            source.write_bytes(original_source)

            original_binary = binary.read_bytes()
            binary.write_bytes(bytes([original_binary[0] ^ 1]) + original_binary[1:])
            with self.assertRaises(ValueError):
                verify_build_manifest(root)
            binary.write_bytes(original_binary)

            manifest_path = build_manifest_path(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["binary"]["size_bytes"] = int(manifest["binary"]["size_bytes"]) + 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_build_manifest(root)


if __name__ == "__main__":
    unittest.main()
