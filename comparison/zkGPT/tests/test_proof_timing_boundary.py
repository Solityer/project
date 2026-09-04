from __future__ import annotations

import ast
import pathlib
import unittest

from zkgpt_gat.proof import ExclusiveProofTimer


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ProofTimingBoundaryTest(unittest.TestCase):
    def test_nested_commitments_are_exclusive_and_counted_once(self) -> None:
        clock = _Clock()
        timer = ExclusiveProofTimer(clock)
        timer.start()
        clock.now = 1.0
        with timer.commitment():
            clock.now = 2.0
            with timer.commitment():
                clock.now = 4.0
            clock.now = 5.0
        clock.now = 8.0
        timer.finish()
        self.assertEqual(timer.commitment_ms, 4000.0)
        self.assertEqual(timer.prove_ms, 4000.0)

    def test_prove_includes_serialization_and_excludes_artifact_writes(self) -> None:
        path = pathlib.Path(__file__).resolve().parents[1] / "src" / "zkgpt_gat" / "proof.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_proof")
        calls = {
            node.func.id: node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"_commit_segments", "serialize_proof_payload"}
        }
        timer_calls = {
            node.func.attr: node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "timing"
        }
        first_artifact_write = min(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_json"
        )
        self.assertLess(timer_calls["start"], calls["_commit_segments"])
        self.assertLess(calls["_commit_segments"], calls["serialize_proof_payload"])
        self.assertLess(calls["serialize_proof_payload"], timer_calls["finish"])
        self.assertLess(timer_calls["finish"], first_artifact_write)

    def test_independent_verify_is_parse_through_protocol_return(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "verify_proof.py").read_text(encoding="utf-8")
        function = ast.parse(source)
        verify = next(node for node in function.body if isinstance(node, ast.FunctionDef) and node.name == "verify_artifact_directory")
        starts = [node.lineno for node in ast.walk(verify) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "protocol_verify_started" for target in node.targets)]
        parse_line = next(node.lineno for node in ast.walk(verify) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "parse_proof_payload")
        invoke_line = next(node.lineno for node in ast.walk(verify) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_invoke_formal_verifier")
        elapsed_line = next(node.lineno for node in ast.walk(verify) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "protocol_verify_elapsed_ms" for target in node.targets))
        self.assertEqual(len(starts), 1)
        self.assertLess(starts[0], parse_line)
        self.assertLess(parse_line, invoke_line)
        self.assertLess(invoke_line, elapsed_line)

    def test_runner_replaces_internal_verify_placeholder(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        source = (root / "src" / "zkgpt_gat" / "runner.py").read_text(encoding="utf-8")
        self.assertIn('token.startswith("verify_time_ms=")', source)
        self.assertIn('benchmark["verify_time_ms"] = independent_verify_ms', source)


if __name__ == "__main__":
    unittest.main()
