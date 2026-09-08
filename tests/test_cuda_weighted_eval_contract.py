"""Static fail-closed contract for CUDA weighted-evaluation entry points."""

from __future__ import annotations

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/algebra/cuda_backend.cu").read_text(encoding="utf-8")


def function_body(name: str) -> str:
    match = re.search(
        rf"\bbool\s+{re.escape(name)}\s*\([^)]*\)\s*\{{",
        SOURCE,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing {name}")

    opening = SOURCE.find("{", match.start())
    depth = 0
    for index in range(opening, len(SOURCE)):
        if SOURCE[index] == "{":
            depth += 1
        elif SOURCE[index] == "}":
            depth -= 1
            if depth == 0:
                return SOURCE[opening + 1:index]
    raise AssertionError(f"unterminated {name}")


def checked_cuda_calls(body: str, name: str) -> None:
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.DOTALL)
    calls = list(re.finditer(r"\bcuda(?:Memcpy|Memset)\s*\(", body))
    if not calls:
        raise AssertionError(f"{name} has no cudaMemcpy/cudaMemset calls to check")

    for call in calls:
        opening = body.find("(", call.start())
        depth = 0
        closing = None
        for index in range(opening, len(body)):
            if body[index] == "(":
                depth += 1
            elif body[index] == ")":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            raise AssertionError(f"unterminated CUDA call in {name}")
        semicolon = body.find(";", closing)
        if semicolon < 0:
            raise AssertionError(f"missing statement terminator in {name}")
        suffix = body[closing + 1:semicolon + 1]
        if not re.search(r"(?:!=|==)\s*cudaSuccess", suffix):
            raise AssertionError(f"unchecked CUDA copy/set in {name}")


for entry in (
    "weighted_poly_eval_gpu",
    "weighted_poly_eval_gpu_columnar",
    "weighted_poly_eval_gpu_multi",
):
    checked_cuda_calls(function_body(entry), entry)

assert function_body("weighted_poly_eval_gpu").count("row_count > 65535") == 1
assert function_body("weighted_poly_eval_gpu_columnar").count("row_count > 65535") == 1
weighted_body = function_body("weighted_poly_eval_gpu")
assert "make_weighted_eval_chunk_plan" in weighted_body
assert "kMaxGpuValueBytes" in weighted_body
assert "std::max(max_cols_per_chunk" not in weighted_body
assert "checked_size_mul" in weighted_body
assert "chunk_value_bytes > kMaxGpuValueBytes" in weighted_body
assert "std::min(row_count, static_cast<std::size_t>(65535))" not in function_body(
    "weighted_poly_eval_gpu"
)
assert "std::min(row_count, static_cast<std::size_t>(65535))" not in function_body(
    "weighted_poly_eval_gpu_columnar"
)

print("CUDA weighted evaluation fail-closed contract: PASS")
