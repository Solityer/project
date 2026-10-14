"""Small fixtures for session-bound result-stage reuse and exact controller identity."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ezkl_formal_control  # noqa: E402
import run_hybrid_formal  # noqa: E402


class FormalRecoveryReceiptTest(unittest.TestCase):
    def test_supervisor_exit_classification_requires_recorded_stop_and_preserves_exit_code(self) -> None:
        cases = (
            (0, False, ("success", "EXITED")),
            (0, True, ("success", "EXITED")),
            (130, True, ("stopped", "STOPPED")),
            (143, True, ("stopped", "STOPPED")),
            (-15, True, ("stopped", "STOPPED")),
            (130, False, ("failed", "EXITED")),
            (143, False, ("failed", "EXITED")),
            (-15, False, ("failed", "EXITED")),
            (1, True, ("failed", "EXITED")),
        )
        for exit_code, stop_requested, expected in cases:
            with self.subTest(exit_code=exit_code, stop_requested=stop_requested):
                self.assertEqual(
                    ezkl_formal_control._classify_supervisor_exit(
                        exit_code, stop_requested=stop_requested
                    ),
                    expected,
                )

    def test_producer_finalization_reads_its_success_footer_and_original_elapsed(self) -> None:
        class FakeParser:
            @staticmethod
            def parse_args(_argv: list[str]) -> object:
                return object()

        module = SimpleNamespace(parser=lambda: FakeParser(), main=lambda _args, **_kwargs: 0)
        with tempfile.TemporaryDirectory(prefix="ezkl-producer-log-") as temporary:
            output = Path(temporary) / "runs" / "formal" / "cora"
            producer_log = output / run_hybrid_formal._FORMAL_FINALIZATION_STAGE_LOGS["producer"]
            with mock.patch.object(run_hybrid_formal.importlib, "import_module", return_value=module):
                elapsed_ms = run_hybrid_formal.invoke_python_entrypoint(
                    "fixture_producer",
                    [],
                    "fixture public statement producer",
                    context=object(),
                    session_id="e" * 32,
                    log_path=producer_log,
                )
            logged_elapsed_ms = run_hybrid_formal._latest_successful_log_wall_ms(
                producer_log, "return_code", "fixture producer"
            )
            self.assertEqual(round(logged_elapsed_ms, 6), round(elapsed_ms, 6))

    def test_collector_receipt_reuses_only_unchanged_matrix_and_log(self) -> None:
        session_id = "a" * 32
        inputs_sha256 = "b" * 64
        with tempfile.TemporaryDirectory(prefix="ezkl-finalization-receipt-") as temporary:
            root = Path(temporary)
            session_dir = root / "runs" / "formal" / ".sessions"
            session_dir.mkdir(parents=True)
            session_file = session_dir / f"{session_id}.json"
            session_file.write_text(json.dumps({"session_id": session_id}), encoding="utf-8")
            matrix = root / "results" / "formal_matrix.json"
            matrix.parent.mkdir(parents=True)
            matrix.write_text("{}\n", encoding="utf-8")
            log = root / "runs" / "formal" / ".sessions" / "collect.log"
            log.write_text("--- command_exit_code=0 wall_ms=12.345678 outcome=success ---\n", encoding="utf-8")

            with mock.patch.object(run_hybrid_formal, "ROOT", root):
                receipt_sha256 = run_hybrid_formal._record_finalization_receipt(
                    session_file=session_file,
                    session_id=session_id,
                    stage="collector",
                    inputs_sha256=inputs_sha256,
                    matrix=matrix,
                    log_path=log,
                    status="COLLECTED_PENDING_FINAL_VALIDATION",
                    elapsed_ms=12.345678,
                )
                reused = run_hybrid_formal._try_reuse_finalization_receipt(
                    session_file=session_file,
                    session_id=session_id,
                    stage="collector",
                    inputs_sha256=inputs_sha256,
                    matrix=matrix,
                    log_path=log,
                    status="COLLECTED_PENDING_FINAL_VALIDATION",
                )
                self.assertEqual(reused, (12.345678, receipt_sha256))

                log.write_text(log.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
                self.assertIsNone(
                    run_hybrid_formal._try_reuse_finalization_receipt(
                        session_file=session_file,
                        session_id=session_id,
                        stage="collector",
                        inputs_sha256=inputs_sha256,
                        matrix=matrix,
                        log_path=log,
                        status="COLLECTED_PENDING_FINAL_VALIDATION",
                    )
                )

    def test_controller_rejects_reused_pid_or_changed_command_identity(self) -> None:
        session_id = "c" * 32
        pid, ticks = 12345, 67890
        command = [sys.executable, str(ezkl_formal_control._SCRIPT), "_supervise", "--session-id", session_id]
        identity = {"pid": pid, "start_ticks": ticks, "cmdline": command}
        with (
            mock.patch.object(ezkl_formal_control, "process_start_ticks", return_value=ticks),
            mock.patch.object(ezkl_formal_control, "_proc_is_live", return_value=True),
            mock.patch.object(ezkl_formal_control, "_proc_cmdline", return_value=command),
            mock.patch.object(ezkl_formal_control, "_proc_cwd", return_value=str(ezkl_formal_control.ROOT.resolve())),
        ):
            self.assertTrue(
                ezkl_formal_control._alive_exact(
                    identity,
                    expected_script=ezkl_formal_control._SCRIPT,
                    session_id=session_id,
                )
            )
            reused_pid = {**identity, "start_ticks": ticks + 1}
            self.assertFalse(
                ezkl_formal_control._alive_exact(
                    reused_pid,
                    expected_script=ezkl_formal_control._SCRIPT,
                    session_id=session_id,
                )
            )

    def test_formal_owner_accepts_only_bound_matrix_execv_transition(self) -> None:
        session_id = "d" * 32
        output_root = ezkl_formal_control.ROOT / "runs" / "formal"
        matrix = ezkl_formal_control.ROOT / "results" / "formal_matrix.json"
        binary = ezkl_formal_control.ROOT / "target" / "gpu" / "release" / "ezkl"
        wrapper = ezkl_formal_control._child_command(
            output_root=output_root,
            matrix=matrix,
            binary=binary,
            session_id=session_id,
            resume=False,
            capacity_only=False,
        )
        metadata = {
            "baseline": "ezkl",
            "session_id": session_id,
            "output_root": output_root.relative_to(ezkl_formal_control.ROOT).as_posix(),
            "matrix": matrix.relative_to(ezkl_formal_control.ROOT).as_posix(),
            "binary": binary.relative_to(ezkl_formal_control.ROOT).as_posix(),
            "mode": "five_dataset_formal",
            "child_command": wrapper,
        }
        variants = ezkl_formal_control._formal_owner_command_variants(metadata, session_id)
        self.assertIsNotNone(variants)
        assert variants is not None
        identity = {"pid": 12345, "start_ticks": 67890, "cmdline": variants[0]}
        with (
            mock.patch.object(ezkl_formal_control, "process_start_ticks", return_value=67890),
            mock.patch.object(ezkl_formal_control, "_proc_is_live", return_value=True),
            mock.patch.object(ezkl_formal_control, "_proc_cmdline", return_value=variants[1]),
            mock.patch.object(ezkl_formal_control, "_proc_cwd", return_value=str(ezkl_formal_control.ROOT.resolve())),
        ):
            self.assertTrue(ezkl_formal_control._formal_owner_matches(identity, metadata, session_id))
            changed_matrix = dict(metadata, matrix="results/other_matrix.json")
            self.assertFalse(ezkl_formal_control._formal_owner_matches(identity, changed_matrix, session_id))


if __name__ == "__main__":
    unittest.main()
