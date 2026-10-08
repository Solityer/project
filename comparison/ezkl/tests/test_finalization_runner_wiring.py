"""Source-level guard that final capacity checks stay on the production path."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_hybrid_formal.py"


class FinalizationRunnerWiringTest(unittest.TestCase):
    def test_capacity_gates_precede_collector_and_independent_validator(self) -> None:
        module = ast.parse(RUNNER.read_text(encoding="utf-8"))
        run_session = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_session"
        )
        calls = [node for node in ast.walk(run_session) if isinstance(node, ast.Call)]
        gate_lines: dict[str, int] = {}
        stage_lines: dict[str, int] = {}
        run_lines: dict[str, int] = {}
        for call in calls:
            if isinstance(call.func, ast.Name) and call.func.id == "check_finalization_capacity":
                phase = next(
                    (keyword.value.value for keyword in call.keywords
                     if keyword.arg == "phase" and isinstance(keyword.value, ast.Constant)),
                    None,
                )
                self.assertIn(phase, {"collector", "validator"})
                self.assertNotIn(phase, gate_lines, f"duplicate {phase} capacity gate")
                gate_lines[phase] = call.lineno
            elif isinstance(call.func, ast.Name) and call.func.id == "_set_stage" and len(call.args) >= 2:
                stage = call.args[1]
                if isinstance(stage, ast.Constant) and stage.value == "final_matrix_validation":
                    stage_lines["validator"] = call.lineno
            elif isinstance(call.func, ast.Name) and call.func.id == "run" and len(call.args) >= 2:
                label = call.args[1]
                if isinstance(label, ast.Constant):
                    if label.value == "collect five-dataset result matrix":
                        run_lines["collector"] = call.lineno
                    elif label.value == "validate five-dataset result matrix":
                        run_lines["validator"] = call.lineno

        self.assertEqual(set(gate_lines), {"collector", "validator"})
        self.assertEqual(set(stage_lines), {"validator"})
        self.assertEqual(set(run_lines), {"collector", "validator"})
        self.assertLess(gate_lines["collector"], run_lines["collector"])
        self.assertLess(run_lines["collector"], stage_lines["validator"])
        self.assertLess(stage_lines["validator"], gate_lines["validator"])
        self.assertLess(gate_lines["validator"], run_lines["validator"])

    def test_phase_receipts_are_persisted_separately(self) -> None:
        module = ast.parse(RUNNER.read_text(encoding="utf-8"))
        run_session = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_session"
        )
        receipt_keys = {
            node.targets[0].slice.value
            for node in ast.walk(run_session)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "session_document"
            and isinstance(node.targets[0].slice, ast.Constant)
            and isinstance(node.targets[0].slice.value, str)
            and node.targets[0].slice.value in {"finalization_capacity", "final_validation_capacity"}
        }
        self.assertEqual(receipt_keys, {"finalization_capacity", "final_validation_capacity"})


if __name__ == "__main__":
    unittest.main()
