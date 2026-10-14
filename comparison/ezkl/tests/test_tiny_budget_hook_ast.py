#!/usr/bin/env python3
"""Static fixture for the Tiny runner's native keygen budget call binding."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "run_tiny_cpu_e2e.py"


class TinyBudgetHookAstTest(unittest.TestCase):
    def test_writer_uses_authenticated_same_session_inputs_before_setup(self) -> None:
        module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        main = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "run_hybrid_formal"
                and any(alias.name == "_write_native_keygen_budget_receipt" for alias in node.names)
                for node in ast.walk(main)
            )
        )
        calls = [
            node for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_write_native_keygen_budget_receipt"
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(ast.unparse(call.args[0]), "setup_output")
        keywords = {item.arg: item.value for item in call.keywords}
        self.assertEqual(ast.unparse(keywords["dataset"]), "TINY_DATASET")
        self.assertEqual(ast.unparse(keywords["session_id"]), "session_id")
        self.assertEqual(ast.unparse(keywords["admission"]), "admission")
        self.assertEqual(ast.unparse(keywords["capacity"]), "capacity['requirements']")
        self.assertEqual(ast.unparse(keywords["setup_check"]), "capacity['setup_check']")
        self.assertEqual(ast.unparse(keywords["workspace_query"]), "capacity['workspace_query']")

        identity = keywords["resume_identity"]
        self.assertIsInstance(identity, ast.Dict)
        bound_identity = {
            ast.literal_eval(key): ast.unparse(value)
            for key, value in zip(identity.keys, identity.values, strict=True)
        }
        self.assertEqual(
            bound_identity,
            {
                "binary_sha256": "sha256(binary)",
                "build_receipt_sha256": "env['EZKL_FORMAL_BUILD_RECEIPT_SHA256']",
                "build_environment_sha256": "env['EZKL_FORMAL_BUILD_ENVIRONMENT_SHA256']",
            },
        )

        calls_by_line = sorted(
            (
                node.lineno,
                ast.unparse(node.args[0]) if node.args else "",
            )
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_stage"
        )
        initial_setup_line = next(
            line for line, first_arg in calls_by_line if first_arg == "'initial-prepare'"
        )
        setup_dir_line = next(
            node.lineno for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "mkdir"
            and ast.unparse(node.func.value) == "setup_output"
        )
        self.assertLess(setup_dir_line, call.lineno)
        self.assertLess(call.lineno, initial_setup_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
