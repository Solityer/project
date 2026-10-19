#!/usr/bin/env python3
"""Mocked contract checks for scoped zkGPT formal execution."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common
import formal_preflight
import prepare_resources
import run_formal
import stop_formal


SCOPE = ("ppi", "ogbn_arxiv")
SID = "a" * 32
DIGEST = "b" * 64


def _write_sidecar(directory: Path, names: list[str], sidecar_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in names:
        path = directory / name
        path.write_bytes((directory.name + ":" + name + "\n").encode("utf-8"))
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n")
    (directory / sidecar_name).write_text("".join(lines), encoding="utf-8")


def _write_tiny_resources(root: Path, datasets: tuple[str, ...] = common.DATASETS) -> None:
    config_dir = root / "baseline-local" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "quantization_scale_chain.cfg").write_text("scale=1\n", encoding="utf-8")
    quant = config_dir / "quantization_scale_chain.cfg"
    (config_dir / "quantization_scale_chain.cfg.sha256").write_text(
        f"{hashlib.sha256(quant.read_bytes()).hexdigest()}  {quant.name}\n", encoding="utf-8")
    for dataset in datasets:
        config = config_dir / f"{dataset}_secure.cfg"
        config.write_text(
            "\n".join((
                f"dataset={dataset}", "setup_mode=hyrax_native_local", "setup_resource_scope=baseline-local",
                "privacy_policy=model_and_trace_hidden", "input_privacy=public_graph_public_features",
                "output_mode=public_logits", "proof_task=node_classification_full_gat", "",
            )), encoding="utf-8")
        (config_dir / f"{config.name}.sha256").write_text(
            f"{hashlib.sha256(config.read_bytes()).hexdigest()}  {config.name}\n", encoding="utf-8")
        cache_dir = root / "baseline-local" / "data" / "cache" / dataset
        cache_names = ["meta.cfg", "edges.txt", "features.txt"]
        if dataset == "ppi":
            cache_names.extend(["labels_multi.txt", "labels_multi.npy", "node_ptr.txt",
                                "train_mask.npy", "val_mask.npy", "test_mask.npy"])
        else:
            cache_names.append("labels.txt")
            if dataset != "ogbn_arxiv":
                cache_names.append("test.index")
        _write_sidecar(cache_dir, cache_names, "integrity.sha256")
        checkpoint_dir = root / "baseline-local" / "checkpoints" / f"{dataset}_gat"
        _write_sidecar(checkpoint_dir, ["manifest.json"], "integrity.sha256")


class FormalDatasetScopeSmoke(unittest.TestCase):
    def test_scope_normalization_and_prepare_resource_reads(self) -> None:
        self.assertEqual(common.canonical_dataset_scope(("ogbn_arxiv", "ppi")), SCOPE)
        self.assertEqual(common.canonical_dataset_scope(None), common.DATASETS)
        for invalid in ((), ("ppi", "ppi"), ("unknown",)):
            with self.assertRaises(common.ChainError):
                common.canonical_dataset_scope(invalid)
        legacy_session = {"session_id": SID, "datasets": list(common.DATASETS)}
        self.assertEqual(
            common.formal_resource_manifest_path(Path("/tmp/root"), SID, legacy_session),
            Path("/tmp/root/results/resource_manifest.json"),
        )
        with self.assertRaises(common.ChainError):
            common.formal_resource_manifest_path(Path("/tmp/root"), SID, {**legacy_session, "dataset_scope": None})

        with tempfile.TemporaryDirectory(prefix="zkgpt-scoped-resources-") as temporary:
            root = Path(temporary)
            _write_tiny_resources(root, SCOPE)
            with mock.patch.object(common, "runtime_source_sha256", return_value=DIGEST), \
                    mock.patch.object(common, "source_tree_sha256", return_value=DIGEST):
                _, session, session_sha256 = common.create_formal_session(root, SID, SCOPE)
            manifest_path = root / "results" / "resource_manifest.json"
            scoped_path = root / "results" / "formal" / SID / "resource_manifest.json"
            inspect = prepare_resources.inspect_dataset
            with (
                mock.patch.object(prepare_resources, "baseline_root", return_value=root),
                mock.patch.object(common, "baseline_root", return_value=root),
                mock.patch.object(common, "runtime_source_sha256", return_value=DIGEST),
                mock.patch.object(common, "source_tree_sha256", return_value=DIGEST),
                mock.patch.object(prepare_resources, "inspect_dataset", wraps=inspect) as inspected,
                mock.patch.dict(os.environ, {"ZKGPT_FORMAL_SESSION_ID": SID}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(prepare_resources.main(["--check", "--datasets", *SCOPE]), 0)
            self.assertEqual([call.args[1] for call in inspected.call_args_list], list(SCOPE))
            self.assertFalse(manifest_path.exists())
            payload, actual_path, manifest_sha256 = common.require_formal_resource_manifest(
                root, SID, session, session_sha256
            )
            self.assertEqual(actual_path, scoped_path)
            self.assertEqual(manifest_sha256, common.sha256_file(scoped_path))
            self.assertEqual(payload["dataset_scope"], list(SCOPE))
            self.assertEqual(payload["session_id"], SID)
            self.assertEqual(payload["session_manifest_sha256"], session_sha256)
            self.assertEqual(set(payload["datasets"]), set(SCOPE))
            original_scoped_manifest = scoped_path.read_bytes()
            wrong_binding = json.loads(original_scoped_manifest)
            wrong_binding["session_manifest_sha256"] = "c" * 64
            scoped_path.write_text(json.dumps(wrong_binding), encoding="utf-8")
            with self.assertRaises(common.ChainError):
                common.require_formal_resource_manifest(root, SID, session, session_sha256)
            scoped_path.write_bytes(original_scoped_manifest)

            observed_entries: dict[str, dict] = {}
            observed_quant: dict[str, dict] = {}
            self.assertEqual(common.verify_local_resources(
                root, SCOPE, resource_entries=observed_entries, quantization_identity=observed_quant
            ), [])
            common.validate_scoped_resource_manifest_observation(payload, observed_entries, observed_quant)

            ppi_edges = root / "baseline-local" / "data" / "cache" / "ppi" / "edges.txt"
            original_edges = ppi_edges.read_bytes()
            ppi_edges.write_bytes(original_edges + b"tampered")
            self.assertTrue(common.verify_local_resources(root, SCOPE))
            ppi_edges.write_bytes(original_edges)

            quant_path = root / "baseline-local" / "configs" / "quantization_scale_chain.cfg"
            original_quant = quant_path.read_bytes()
            quant_path.write_bytes(original_quant + b"tampered\n")
            self.assertTrue(common.verify_local_resources(root, SCOPE))
            quant_path.write_bytes(original_quant)

            bad_entries = {dataset: dict(entry) for dataset, entry in payload["datasets"].items()}
            bad_entries["ppi"]["config_sha256"] = "c" * 64
            with self.assertRaises(common.ChainError):
                common.validate_scoped_resource_manifest_observation(
                    {**payload, "datasets": bad_entries}, observed_entries, observed_quant
                )
            with self.assertRaises(common.ChainError):
                common.validate_scoped_resource_manifest_observation(
                    {**payload, "quantization_scale_chain_sha256": "c" * 64}, observed_entries, observed_quant
                )

        with tempfile.TemporaryDirectory(prefix="zkgpt-global-resources-") as temporary:
            root = Path(temporary)
            _write_tiny_resources(root)
            inspect = prepare_resources.inspect_dataset
            with (
                mock.patch.object(prepare_resources, "baseline_root", return_value=root),
                mock.patch.object(common, "baseline_root", return_value=root),
                mock.patch.object(prepare_resources, "inspect_dataset", wraps=inspect) as inspected,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(prepare_resources.main(["--check"]), 0)
            self.assertEqual([call.args[1] for call in inspected.call_args_list], list(common.DATASETS))
            global_payload = common.require_manifest(root, "resource_manifest.json", common.RESOURCE_MANIFEST_SCHEMA)
            self.assertEqual(set(global_payload["datasets"]), set(common.DATASETS))
            self.assertFalse((root / "results" / "formal" / SID / "resource_manifest.json").exists())

    def test_preflight_only_checks_selected_resources_and_capacity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-scope-preflight-") as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            session_dir = root / "results" / "formal" / SID
            session_dir.mkdir(parents=True)
            binary_dir = root / "build"
            binary_dir.mkdir()
            prover = binary_dir / "prover"
            verifier = binary_dir / "verifier"
            prover.write_text("stub", encoding="utf-8")
            verifier.write_text("stub", encoding="utf-8")
            session = {"dataset_scope": list(SCOPE), "native_source_sha256": DIGEST}
            ready: list[str] = []
            capacity = {
                "schema": formal_preflight.CAPACITY_SCHEMA,
                "status": "READY",
                "dataset": "",
                "backend": "cpu",
                "proof_path": common.CAPACITY_PROOF_PATH,
                "full_relation_bound": True,
                "reason": "",
            }

            def run_capacity(_root: Path, dataset: str, _backend: str, _logs: Path) -> dict:
                ready.append(dataset)
                return {**capacity, "dataset": dataset}

            patches = [
                mock.patch.object(formal_preflight, "baseline_root", return_value=root),
                mock.patch.object(formal_preflight, "require_formal_session", return_value=(SID, session, DIGEST)),
                mock.patch.object(formal_preflight, "check_source", return_value=[]),
                mock.patch.object(formal_preflight, "verify_local_resources", return_value=[]),
                mock.patch.object(formal_preflight, "require_formal_resource_manifest", return_value=(
                    {"datasets": {}}, root / "results" / "formal" / SID / "resource_manifest.json", DIGEST)),
                mock.patch.object(formal_preflight, "validate_scoped_resource_manifest_observation"),
                mock.patch.object(formal_preflight, "sha256_file", return_value=DIGEST),
                mock.patch.object(formal_preflight, "file_fingerprint", return_value={"sha256": DIGEST}),
                mock.patch.object(formal_preflight, "formal_backend_capability", return_value={"status": "READY", "supported": True, "reason": ""}),
                mock.patch.object(formal_preflight, "build_binary", return_value=prover),
                mock.patch.object(formal_preflight, "verifier_binary", return_value=verifier),
                mock.patch.object(formal_preflight, "verify_build_provenance", return_value=DIGEST),
                mock.patch.object(formal_preflight, "binary_fingerprint", return_value={"sha256": DIGEST, "size_bytes": 4, "mtime_ns": 1}),
                mock.patch.object(formal_preflight, "check_capabilities", return_value={}),
                mock.patch.object(formal_preflight, "run_capacity", side_effect=run_capacity),
                mock.patch.object(common, "baseline_root", return_value=root),
                contextlib.redirect_stdout(io.StringIO()),
            ]
            with contextlib.ExitStack() as stack:
                entered = [stack.enter_context(item) for item in patches]
                self.assertEqual(formal_preflight.main([]), 0)
                resources = entered[3]
            resources.assert_called_once()
            self.assertEqual(resources.call_args.args, (root, SCOPE))
            self.assertEqual(set(resources.call_args.kwargs), {"resource_entries", "quantization_identity"})
            self.assertEqual(ready, list(SCOPE))
            payload = json.loads((session_dir / "formal_preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_scope"], list(SCOPE))
            self.assertEqual(payload["status"], "PARTIAL")
            self.assertEqual(payload["capacity"]["cora"]["reason"], "OUT_OF_SCOPE:dataset_not_requested")
            self.assertEqual(payload["capacity"]["citeseer"]["reason"], "OUT_OF_SCOPE:dataset_not_requested")
            self.assertEqual(payload["capacity"]["pubmed"]["reason"], "OUT_OF_SCOPE:dataset_not_requested")

    def test_full_five_preflight_does_not_collect_resource_observations(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-full-preflight-") as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            session_dir = root / "results" / "formal" / SID
            session_dir.mkdir(parents=True)
            prover = root / "build" / "prover"
            verifier = root / "build" / "verifier"
            prover.parent.mkdir()
            prover.write_text("stub", encoding="utf-8")
            verifier.write_text("stub", encoding="utf-8")
            session = {"native_source_sha256": DIGEST}  # legacy session: missing scope means all five
            capacity = {
                "schema": formal_preflight.CAPACITY_SCHEMA, "status": "READY", "dataset": "",
                "backend": "cpu", "proof_path": common.CAPACITY_PROOF_PATH,
                "full_relation_bound": True, "reason": "",
            }
            resources = mock.Mock(return_value=[])

            patches = [
                mock.patch.object(formal_preflight, "baseline_root", return_value=root),
                mock.patch.object(formal_preflight, "require_formal_session", return_value=(SID, session, DIGEST)),
                mock.patch.object(formal_preflight, "check_source", return_value=[]),
                mock.patch.object(formal_preflight, "verify_local_resources", resources),
                mock.patch.object(formal_preflight, "require_formal_resource_manifest", return_value=(
                    {"datasets": {}}, root / "results" / "resource_manifest.json", DIGEST)),
                mock.patch.object(formal_preflight, "sha256_file", return_value=DIGEST),
                mock.patch.object(formal_preflight, "file_fingerprint", return_value={"sha256": DIGEST}),
                mock.patch.object(formal_preflight, "formal_backend_capability", return_value={
                    "status": "READY", "supported": True, "reason": ""}),
                mock.patch.object(formal_preflight, "build_binary", return_value=prover),
                mock.patch.object(formal_preflight, "verifier_binary", return_value=verifier),
                mock.patch.object(formal_preflight, "verify_build_provenance", return_value=DIGEST),
                mock.patch.object(formal_preflight, "binary_fingerprint", return_value={
                    "sha256": DIGEST, "size_bytes": 4, "mtime_ns": 1}),
                mock.patch.object(formal_preflight, "check_capabilities", return_value={}),
                mock.patch.object(formal_preflight, "run_capacity", side_effect=lambda _root, dataset, _backend, _logs: {
                    **capacity, "dataset": dataset}),
                mock.patch.object(common, "baseline_root", return_value=root),
                contextlib.redirect_stdout(io.StringIO()),
            ]
            with contextlib.ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                self.assertEqual(formal_preflight.main([]), 0)
            resources.assert_called_once_with(root, common.DATASETS)

    def test_runner_attempts_ogbn_after_ppi_failure_and_writes_subset_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-scope-runner-") as temporary:
            root = Path(temporary)
            (root / "baseline-local").mkdir()
            profile = {"datasets": {dataset: {
                "status": "READY", "cell_retained_artifact_bytes_upper_bound": 1_000_000,
            } for dataset in SCOPE}}
            context = {
                "local_capacity_profile": profile,
                "local_capacity_profile_sha256": DIGEST,
                "resource_manifest_path": f"results/formal/{SID}/resource_manifest.json",
            }
            session = {"dataset_scope": list(SCOPE), "runtime_source_sha256": DIGEST, "native_source_sha256": DIGEST}
            preflight = {"dataset_scope": list(SCOPE)}
            attempts: list[str] = []
            manifest_sha = "c" * 64

            def phase(_root, _lock, sid, session_dir, phase_name, dataset, *_args, **_kwargs):
                if phase_name == "prepare":
                    prepared_dir = Path(session_dir)
                    prepared_dir.mkdir(parents=True, exist_ok=False)
                    (prepared_dir / "session_status.json").write_text(json.dumps({
                        "schema": common.FORMAL_SESSION_SCHEMA,
                        "session_id": sid,
                        "status": "RUNNING",
                        "active_stage": "prepared",
                        "completed_stages": [],
                        "stage_status": {},
                        "dataset_scope": list(SCOPE),
                    }), encoding="utf-8")
                    return {"session_dir": str(session_dir), "dataset_scope": list(SCOPE),
                            "local_capacity_profile_sha256": DIGEST}
                attempts.append(dataset)
                if dataset == "ppi":
                    raise run_formal.ChainError("simulated PPI failure")
                receipts = Path(session_dir) / "receipts"
                receipts.mkdir(parents=True, exist_ok=True)
                run_dir = root / "baseline-local" / "runs" / dataset / sid
                run_dir.mkdir(parents=True, exist_ok=True)
                paths = {
                    "result_path": f"baseline-local/runs/{dataset}/{sid}/result.json",
                    "execution_path": f"baseline-local/runs/{dataset}/{sid}/execution.json",
                    "proof_path": f"baseline-local/runs/{dataset}/{sid}/proof.artifact",
                    "verification_report_path": f"baseline-local/runs/{dataset}/{sid}/verifier_report.json",
                }
                for name in ("result.json", "execution.json", "proof.artifact", "verifier_report.json"):
                    (run_dir / name).write_text("stub\n", encoding="utf-8")
                receipt = {
                    "schema": "zkgpt-gat-cell-receipt-v1", "status": "VERIFY_OK", "session_id": sid,
                    "session_manifest_sha256": manifest_sha, "dataset": dataset,
                    "independent_verification": True,
                    **paths,
                    "result_sha256": DIGEST, "execution_sha256": DIGEST,
                    "proof_sha256": DIGEST, "verification_report_sha256": DIGEST,
                }
                receipt_path = receipts / f"{dataset}.json"
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                return {
                    "receipt_path": str(receipt_path.relative_to(root)), "receipt_sha256": DIGEST,
                    "proof_sha256": DIGEST, "verification_report_path": paths["verification_report_path"],
                    "verification_report_sha256": DIGEST,
                }

            audited = {
                "proof_size_bytes": 5, "paper_metrics": {"accuracy": 1.0}, "timing_ms": {"prove": 1.0},
                "backend": "cpu", "result_path": f"baseline-local/runs/ogbn_arxiv/{SID}/result.json",
                "artifact_path": f"baseline-local/runs/ogbn_arxiv/{SID}/proof.artifact",
                "artifact_sha256": DIGEST,
                "verifier_report_path": f"baseline-local/runs/ogbn_arxiv/{SID}/verifier_report.json",
            }
            patches = [
                mock.patch.object(run_formal, "formal_child_environment"),
                mock.patch.object(run_formal, "install_process_signal_cleanup"),
                mock.patch.object(run_formal, "_recover_stale_owner", return_value=""),
                mock.patch.object(run_formal, "_proc_start_ticks", return_value=123),
                mock.patch.object(run_formal.secrets, "token_hex", side_effect=lambda n: SID if n == 16 else "d" * 64),
                mock.patch.object(run_formal, "_run_owner_phase", side_effect=phase),
                mock.patch.object(run_formal, "_load_prepared", return_value=(session, preflight, DIGEST, context,
                                                                              root / "results/formal" / SID / "formal_preflight.json",
                                                                              manifest_sha)),
                mock.patch.object(run_formal, "_capacity_gate"),
                mock.patch.object(run_formal, "sha256_file", return_value=DIGEST),
                mock.patch.object(run_formal, "run_directory", side_effect=lambda _root, ds, sid: (
                    f"baseline-local/runs/{ds}/{sid}", root / "baseline-local" / "runs" / ds / sid)),
                mock.patch.object(run_formal, "validate_formal_cell", return_value=audited),
                mock.patch.object(run_formal, "_reuse_owner_approved_cell", return_value={
                    "verification_source": "cell_stage_independent_report"}),
            ]
            with mock.patch.dict(os.environ, {}, clear=False), contextlib.ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                with self.assertRaises(run_formal.ChainError):
                    run_formal._run_official(root, SCOPE)
            self.assertEqual(attempts, list(SCOPE))
            report = json.loads((root / "results" / "formal" / SID / "subset_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "SUBSET_FAILED")
            self.assertEqual(report["dataset_scope"], list(SCOPE))
            self.assertFalse(report["full_five_complete"])
            self.assertEqual([cell["status"] for cell in report["cells"]], ["FAILED", "VERIFY_OK"])
            owner = json.loads((root / "baseline-local" / "formal-owner.json").read_text(encoding="utf-8"))
            self.assertEqual(owner["status"], "SUBSET_FAILED")

    def test_stop_requires_exact_scoped_argv(self) -> None:
        root = Path("/tmp/zkgpt-stop-scope-fixture")
        launch_id = "e" * 32
        launch_dir = root / "baseline-local" / "formal-detached" / launch_id
        status_path = launch_dir / "status.json"
        supervisor_script = root / "scripts" / "formal_detached_supervisor.py"
        runner = root / "scripts" / "run_formal.py"
        scope_suffix = ["--datasets", *SCOPE]
        state = {
            "schema": stop_formal.STATUS_SCHEMA, "launch_id": launch_id,
            "root_path": str(root), "status_path": str(status_path),
            "console_path": str(launch_dir / "console.log"), "runner_path": str(runner),
            "status": "STARTING", "requested_datasets": list(SCOPE), "datasets_selector_explicit": True,
            "supervisor_pid": 101, "supervisor_start_ticks": 1, "supervisor_process_group": 101,
            "supervisor_cwd": str(root), "supervisor_executable": sys.executable,
            "supervisor_argv": [sys.executable, str(supervisor_script), "--root", str(root),
                                "--launch-id", launch_id, *scope_suffix],
            "child_pid": None,
        }
        self.assertEqual(stop_formal.validate_status(state, root, launch_id, status_path), "STARTING")
        bad = {**state, "supervisor_argv": state["supervisor_argv"][:-1]}
        with self.assertRaises(stop_formal.StopError):
            stop_formal.validate_status(bad, root, launch_id, status_path)


if __name__ == "__main__":
    unittest.main()
