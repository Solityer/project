#!/usr/bin/env python3
"""Focused checks for scoped preflight and formal-run continuation."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common  # noqa: E402
import formal_preflight  # noqa: E402
import gat_zkgpt_experiment  # noqa: E402
import run_formal  # noqa: E402

SESSION_ID = "a" * 32
SUBSET = ("ppi", "ogbn_arxiv")


class FormalDatasetScopeTests(unittest.TestCase):
    def test_preflight_keeps_trying_after_one_dataset_resource_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-scoped-preflight-") as temporary:
            root = Path(temporary)
            session = {"dataset_scope": list(SUBSET)}
            attempted_capacity: list[str] = []

            def verify_resources(_root: Path, datasets: tuple[str, ...]) -> list[str]:
                dataset = datasets[0]
                return ["PPI checkpoint is unavailable"] if dataset == "ppi" else []

            def run_capacity(_root: Path, dataset: str, backend: str, _logs: Path) -> dict:
                attempted_capacity.append(dataset)
                return {
                    "schema": formal_preflight.CAPACITY_SCHEMA,
                    "status": "READY",
                    "dataset": dataset,
                    "backend": backend,
                    "reason": "",
                }

            patches = (
                mock.patch.object(formal_preflight, "baseline_root", return_value=root),
                mock.patch.object(formal_preflight, "require_formal_session", return_value=(SESSION_ID, session)),
                mock.patch.object(formal_preflight, "check_source", return_value=[]),
                mock.patch.object(formal_preflight, "formal_backend_capability", return_value={
                    "supported": True, "status": "FORMAL_CPU_MASKED_FULL", "reason": "",
                }),
                mock.patch.object(formal_preflight, "build_binary", return_value=root / "build/prover"),
                mock.patch.object(formal_preflight, "verifier_binary", return_value=root / "build/verifier"),
                mock.patch.object(formal_preflight, "check_capabilities", return_value={"masked_full": True}),
                mock.patch.object(formal_preflight, "verify_local_resources", side_effect=verify_resources),
                mock.patch.object(formal_preflight, "run_capacity", side_effect=run_capacity),
                mock.patch.object(formal_preflight, "atomic_write_json",
                                   side_effect=lambda path, payload: common.atomic_write_json(path, payload, root)),
            )
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = formal_preflight.main([])

            payload = json.loads(
                common.formal_preflight_path(root, SESSION_ID).read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "PARTIAL")
            self.assertFalse(payload["formal_experiment_eligible"])
            self.assertEqual(payload["capacity"]["ppi"]["status"], "BLOCKED")
            self.assertEqual(payload["capacity"]["ogbn_arxiv"]["status"], "READY")
            self.assertEqual(attempted_capacity, ["ogbn_arxiv"])

    def test_subset_continues_after_capacity_failure_and_never_claims_full_five(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-scoped-run-") as temporary:
            root = Path(temporary)
            attempted: list[str] = []

            def prepare(_root: Path, requested: tuple[str, ...], session_id: str):
                session_id, _session = common.create_formal_session(root, session_id, requested)
                session_dir = common.formal_results_directory(root, session_id)
                capacities = {
                    dataset: ({"status": "READY", "proof_path": common.CAPACITY_PROOF_PATH,
                               "full_relation_bound": True}
                              if dataset == "ogbn_arxiv" else
                              {"status": "BLOCKED", "reason": "dataset capacity is blocked"})
                    for dataset in common.DATASETS
                }
                preflight = {
                    "schema": formal_preflight.PREFLIGHT_SCHEMA,
                    "status": "PARTIAL",
                    "session_id": session_id,
                    "dataset_scope": list(requested),
                    "shared_preconditions_ready": True,
                    "blockers": [],
                    "formal_experiment_eligible": False,
                    "backends": {"cpu": {"ready": True}},
                    "capacity": capacities,
                }
                profile = {
                    "schema": run_formal.LOCAL_CAPACITY_PROFILE_SCHEMA,
                    "session_id": session_id,
                    "fs_device_id": str(root.stat().st_dev),
                    "datasets": {
                        dataset: ({"status": "READY", "retained_bytes_upper_bound": 1024}
                                  if capacities[dataset]["status"] == "READY" else
                                  {"status": "BLOCKED", "reason": "dataset capacity is blocked"})
                        for dataset in common.DATASETS
                    },
                }
                return session_dir, preflight, profile, common.formal_preflight_path(root, session_id)

            def capacity_gate(_root: Path, _session_id: str, _session_dir: Path,
                              _profile: dict, dataset: str) -> dict:
                attempted.append(dataset)
                if dataset == "ppi":
                    raise common.ChainError("capacity blocked for ppi")
                return {}

            def run_cell(_root: Path, dataset: str, backend: str, preflight: dict, session_id: str) -> dict:
                gat_zkgpt_experiment.validate_preflight_binding(
                    root, preflight, backend, dataset, session_id,
                )
                return {"dataset": dataset, "backend": backend, "status": "VERIFY_OK",
                        "proof_generated": True, "independent_verifier": True}

            run_directory = lambda _root, dataset, session_id: (
                f"baseline-local/runs/formal/{session_id}/{dataset}",
                root / "baseline-local" / "runs" / "formal" / session_id / dataset,
            )
            secrets_patcher = mock.patch.object(run_formal, "secrets")
            patches = (
                secrets_patcher,
                mock.patch.object(run_formal, "install_process_signal_cleanup"),
                mock.patch.object(run_formal, "_prepare_phase", side_effect=prepare),
                mock.patch.object(run_formal, "_capacity_gate", side_effect=capacity_gate),
                mock.patch.object(run_formal, "run_cell", side_effect=run_cell),
                mock.patch.object(run_formal, "run_directory", side_effect=run_directory),
                mock.patch.object(run_formal, "_tree_regular_bytes", return_value=1),
            )
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    item = stack.enter_context(patcher)
                    if patcher is secrets_patcher:
                        item.token_hex.return_value = SESSION_ID
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = run_formal._run_official(root, SUBSET)

            result_path = common.formal_results_directory(root, SESSION_ID) / "subset_validation.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 2)
            self.assertEqual(attempted, list(SUBSET))
            self.assertFalse(payload["full_five_complete"])
            self.assertEqual(payload["status"], "SUBSET_FAILED")
            self.assertEqual([cell["status"] for cell in payload["cells"]], ["FAILED", "VERIFY_OK"])

    def test_full_five_aggregate_requires_all_five_verified_cells(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-full-run-") as temporary:
            root = Path(temporary)
            attempted: list[str] = []

            def prepare(_root: Path, requested: tuple[str, ...], session_id: str):
                session_id, _session = common.create_formal_session(root, session_id, requested)
                session_dir = common.formal_results_directory(root, session_id)
                capacities = {
                    dataset: {"status": "READY", "proof_path": common.CAPACITY_PROOF_PATH,
                              "full_relation_bound": True}
                    for dataset in common.DATASETS
                }
                preflight = {
                    "schema": formal_preflight.PREFLIGHT_SCHEMA,
                    "status": "READY",
                    "session_id": session_id,
                    "dataset_scope": list(requested),
                    "shared_preconditions_ready": True,
                    "blockers": [],
                    "formal_experiment_eligible": True,
                    "backends": {"cpu": {"ready": True}},
                    "capacity": capacities,
                }
                profile = {
                    "schema": run_formal.LOCAL_CAPACITY_PROFILE_SCHEMA,
                    "session_id": session_id,
                    "fs_device_id": str(root.stat().st_dev),
                    "datasets": {
                        dataset: {"status": "READY", "retained_bytes_upper_bound": 1024}
                        for dataset in common.DATASETS
                    },
                }
                return session_dir, preflight, profile, common.formal_preflight_path(root, session_id)

            def capacity_gate(_root: Path, _session_id: str, _session_dir: Path,
                              _profile: dict, dataset: str) -> dict:
                attempted.append(dataset)
                return {}

            def run_cell(_root: Path, dataset: str, backend: str, preflight: dict, session_id: str) -> dict:
                gat_zkgpt_experiment.validate_preflight_binding(
                    root, preflight, backend, dataset, session_id,
                )
                return {"dataset": dataset, "backend": backend, "status": "VERIFY_OK",
                        "proof_generated": True, "independent_verifier": True}

            run_directory = lambda _root, dataset, session_id: (
                f"baseline-local/runs/formal/{session_id}/{dataset}",
                root / "baseline-local" / "runs" / "formal" / session_id / dataset,
            )
            secrets_patcher = mock.patch.object(run_formal, "secrets")
            patches = (
                secrets_patcher,
                mock.patch.object(run_formal, "install_process_signal_cleanup"),
                mock.patch.object(run_formal, "_prepare_phase", side_effect=prepare),
                mock.patch.object(run_formal, "_capacity_gate", side_effect=capacity_gate),
                mock.patch.object(run_formal, "run_cell", side_effect=run_cell),
                mock.patch.object(run_formal, "run_directory", side_effect=run_directory),
                mock.patch.object(run_formal, "_tree_regular_bytes", return_value=1),
            )
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    item = stack.enter_context(patcher)
                    if patcher is secrets_patcher:
                        item.token_hex.return_value = SESSION_ID
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exit_code = run_formal._run_official(root, common.DATASETS)

            result_path = common.formal_final_path(root, SESSION_ID)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(attempted, list(common.DATASETS))
            self.assertEqual(payload["status"], "VALIDATED")
            self.assertEqual(payload["cells_validated"], 5)
            self.assertTrue(payload["full_five_complete"])
            self.assertEqual(payload["verification_accounting"]["independent_crypto_verifier_invocations_total"], 5)

    def test_cell_cleanup_removes_a_stale_verifier_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-stale-report-") as temporary:
            root = Path(temporary)
            run_path = root / "baseline-local" / "runs" / "formal" / SESSION_ID / "ppi"
            run_path.mkdir(parents=True)
            report_path = run_path / "verifier_report.json"
            report_path.write_text('{"status":"VERIFY_OK"}\n', encoding="utf-8")

            common.cleanup_run_outputs(root, run_path)

            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
