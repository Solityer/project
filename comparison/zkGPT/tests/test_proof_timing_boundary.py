from __future__ import annotations

import ast
import pathlib
import unittest


class ProofTimingBoundaryTest(unittest.TestCase):
    def test_prove_timer_starts_after_commitment_and_before_artifact_writes(self) -> None:
        path = pathlib.Path(__file__).resolve().parents[1] / "src" / "zkgpt_gat" / "proof.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_proof")
        commitment_line = next(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_commit_segments"
        )
        prove_timer_line = next(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "prove_started" for target in node.targets)
        )
        elapsed_line = next(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "proof_elapsed_ms" for target in node.targets)
        )
        first_artifact_write = min(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_json"
        )
        self.assertGreater(prove_timer_line, commitment_line)
        self.assertLess(elapsed_line, first_artifact_write)


if __name__ == "__main__":
    unittest.main()
