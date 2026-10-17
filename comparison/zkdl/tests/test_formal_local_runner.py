#!/usr/bin/env python3
"""Targeted regressions for the baseline-local formal lifecycle and admission."""
from __future__ import annotations

import ctypes
import fcntl
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SCRIPTS))
import formal_runner
import formal_provenance
import formal_phase
import result_artifacts
import session_lifecycle


class FormalLocalRunnerTests(unittest.TestCase):
    def test_hyrax_gpu_offload_metadata_accepts_only_cpu_prove_and_cpu_verify(self) -> None:
        record = {"gpu_offload_stage": "hyrax_ipa_fold", "gpu_forward_kernel_executed": "0",
                  "gpu_proof_kernel_executed": "1", "gpu_kernel_executed": "1",
                  "gpu_proof_kernel_count": "12", "gpu_proof_peak_allocated_bytes": "4096",
                  "gpu_device_name": "NVIDIA GB10", "gpu_device_ordinal": "0", "gpu_device_uuid": "GPU-fixture",
                  "verifier_backend": "cpu", "verify_gpu_kernel_executed": "0"}
        envelope = dict(record)
        verification = {"verifier_backend": "cpu", "verify_gpu_kernel_executed": "0"}
        self.assertEqual(result_artifacts._gpu_offload_evidence(record, envelope, verification, "cpu",
                                                                 "hyrax_ipa_fold")["gpu_device_uuid"], "GPU-fixture")
        for mutated, mutated_verification, backend in (
                ({**envelope, "gpu_offload_stage": "none"}, verification, "cpu"),
                ({**envelope, "gpu_forward_kernel_executed": "1"}, verification, "cpu"),
                ({**envelope, "gpu_proof_kernel_executed": "0"}, verification, "cpu"),
                (envelope, {"verifier_backend": "gpu", "verify_gpu_kernel_executed": "1"}, "cpu"),
                (envelope, verification, "gpu")):
            with self.assertRaises(result_artifacts.ArtifactError):
                result_artifacts._gpu_offload_evidence(record, mutated, mutated_verification, backend,
                                                        "hyrax_ipa_fold")

    def test_staged_build_precedes_session_and_prepare_only_checks_published_build(self) -> None:
        with tempfile.TemporaryDirectory(dir=TESTS) as temporary:
            project = Path(temporary)
            runs = project / "runs"
            configs = project / "configs"
            third_party = project / "third_party"
            (project / "scripts").mkdir()
            runs.mkdir()
            configs.mkdir()
            third_party.mkdir()
            (project / "Makefile").write_text("fixture makefile\n")
            (configs / "quantization_scale_chain.cfg").write_text("fixture quantization\n")
            (third_party / "BLST_PROVENANCE.txt").write_text("fixture provenance\n")

            session_id = "a" * 32
            session = runs / f"formal_{session_id}"
            stale_session = runs / ("formal_" + "b" * 32)
            stale_session.mkdir()
            (stale_session / "fixture").write_text("only a temporary fixture\n")

            old_roots = (formal_runner.ROOT, formal_runner.formal_phase.ROOT, formal_provenance.ROOT)
            old_identity = formal_runner.ACTIVE_SESSION_IDENTITY
            environment_names = (*formal_runner.FORMAL_SESSION_IDENTITY_ENV,
                                 formal_runner.DETACHED_LAUNCH_ENV)
            old_environment = {name: os.environ.get(name) for name in environment_names}
            identity = None
            build_commands: list[list[str]] = []
            snapshots: list[str] = []
            admissions: list[str] = []
            prepare_outcome: dict[str, object] = {}

            def fake_subprocess_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                if command and command[0] == "make":
                    build_commands.append(list(command))
                    self.assertFalse(session.exists(), "formal session must not exist while make is active")
                    for path in runs.glob("formal_*"):
                        if path.is_dir() and not path.is_symlink():
                            shutil.rmtree(path)
                    (project / "gat_adapter").mkdir(exist_ok=True)
                    (project / "gat_adapter/hyrax_gpu_fold.cu").write_text("fixture GPU source\n")
                    (project / "gat_runner").write_bytes(b"fixture runner binary")
                    (project / "gat_resource_setup").write_bytes(b"fixture resource binary")
                    return subprocess.CompletedProcess(command, 0)
                self.assertGreaterEqual(len(command), 3)
                self.assertEqual(Path(command[1]).name, "formal_provenance.py")
                build_commands.append(list(command))
                with mock.patch.dict(os.environ, dict(kwargs["env"]), clear=True):
                    with mock.patch.object(sys, "argv", command[1:]):
                        self.assertEqual(formal_provenance.main(), 0)
                return subprocess.CompletedProcess(command, 0)

            try:
                formal_runner.ROOT = project
                formal_runner.formal_phase.ROOT = project
                formal_provenance.ROOT = project
                for name in formal_runner.FORMAL_SESSION_IDENTITY_ENV:
                    os.environ.pop(name, None)
                with mock.patch.object(formal_runner.subprocess, "run", side_effect=fake_subprocess_run):
                    published = formal_runner._build_and_publish_session(runs, session_id)
                self.assertEqual(published, session)
                provenance_commands = [command for command in build_commands
                                       if len(command) > 1 and Path(command[1]).name == "formal_provenance.py"]
                self.assertEqual(len(provenance_commands), 1)
                self.assertIn("--output", provenance_commands[0])
                self.assertNotIn("--check", provenance_commands[0])
                self.assertFalse(stale_session.exists(), "the simulated make cleanup should only see prior temporary roots")
                self.assertEqual(sum(command[:1] == ["make"] for command in build_commands), 1)
                manifest_path = session / "build_manifest.json"
                build_manifest = formal_provenance.validate_manifest(project, manifest_path)
                self.assertEqual(
                    {item["path"] for item in build_manifest["binaries"]},
                    {f"runs/{session.name}/build/gat_runner", f"runs/{session.name}/build/gat_resource_setup"},
                )
                self.assertTrue((session / "logs/build.log").is_file())

                identity = session_lifecycle.capture_session_root(session)
                formal_runner.ACTIVE_SESSION_IDENTITY = identity
                os.environ.update(identity.child_environment())
                state = {"schema": "zkdl_formal_runner_state_v1", "baseline": "zkdl",
                         "session_id": session_id, "status": "RUNNING", "stage": "prepare"}
                formal_runner._write_json(session / "state.json", state)
                launch_dir = runs / "launch_fixture"
                launch_dir.mkdir()
                os.environ[formal_runner.DETACHED_LAUNCH_ENV] = str(launch_dir)
                formal_runner._write_detached_session_link(session, session_id)
                link = json.loads((launch_dir / "formal_session.json").read_text())
                self.assertEqual(link["runner_pid"], os.getpid())
                self.assertEqual(link["session_dir"], str(session))

                def fake_phase_run(command: list[str], *, env: dict[str, str] | None = None,
                                   capture_output: bool = False, **_kwargs: object) -> str:
                    if Path(command[1]).name == "formal_preflight.py":
                        report = {"schema": "zkDL_formal_input_preflight_v1",
                                  "datasets": {dataset: {"status": "PASS"}
                                               for dataset in formal_phase.DATASETS}}
                        Path(env["ZKDL_PREFLIGHT_REPORT"]).write_text(json.dumps(report))
                        return ""
                    if Path(command[1]).name == "formal_session.py":
                        dataset = command[command.index("--dataset") + 1]
                        snapshots.append(dataset)
                        (session / dataset).mkdir()
                        return ""
                    if Path(command[1]).name == "formal_provenance.py":
                        self.assertIn("--check", command)
                        self.assertNotIn("--output", command)
                        with mock.patch.dict(os.environ, dict(env or {}), clear=True):
                            with mock.patch.object(sys, "argv", command[1:]):
                                self.assertEqual(formal_provenance.main(), 0)
                        return ""
                    if "admit" in command:
                        config = Path(command[command.index("--config") + 1])
                        dataset = next(item for item in formal_phase.DATASETS
                                       if config.stem.endswith("_" + item))
                        admissions.append(dataset)
                        if dataset == "citeseer":
                            raise formal_phase.PhaseError("fixture ordinary admission failure")
                        return json.dumps({"schema": "zkdl_native_relation_admission_v2",
                                           "status": "PASS", "dataset": dataset})
                    return ""

                def fake_admission_profile(_session: Path, _session_id: str,
                                           _report: dict[str, object], native: dict[str, dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
                    dataset = next(iter(native))
                    return ({"datasets": {dataset: {}},
                             "finalization_manifest_capacity": {"datasets": {dataset: {}}},
                             "finalization_disk_bytes_upper_bound": 1,
                             "finalization_host_peak_bytes_upper_bound": 1}, {dataset: {}})

                def save_prepare_outcome(_session: Path, _session_id: str, outcomes: dict[str, object],
                                         *_args: object, **_kwargs: object) -> None:
                    prepare_outcome.update(outcomes)

                with mock.patch.object(formal_phase, "_run", side_effect=fake_phase_run), \
                     mock.patch.object(formal_phase, "_admission_profile", side_effect=fake_admission_profile), \
                     mock.patch.object(formal_phase, "_write_prepare_outcome", side_effect=save_prepare_outcome):
                    formal_phase.prepare(session, session_id)

                self.assertEqual(snapshots, list(formal_phase.DATASETS))
                self.assertEqual(admissions, list(formal_phase.DATASETS))
                self.assertEqual(prepare_outcome["citeseer"]["status"], "FAILED")
                self.assertTrue(all(prepare_outcome[dataset]["status"] == "PASS"
                                    for dataset in formal_phase.DATASETS if dataset != "citeseer"))
                self.assertEqual(sum(command[:1] == ["make"] for command in build_commands), 1,
                                 "prepare must not compile after snapshots")
            finally:
                formal_runner.ROOT, formal_runner.formal_phase.ROOT, formal_provenance.ROOT = old_roots
                formal_runner.ACTIVE_SESSION_IDENTITY = old_identity
                for name, value in old_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                if identity is not None and identity.root_fd is not None:
                    os.close(identity.root_fd)

    def test_session_root_lifecycle_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TESTS) as temporary:
            project = Path(temporary)
            session_id = uuid.uuid4().hex
            runs = project / "runs"
            session = runs / f"formal_{session_id}"
            for directory in (runs, project / "configs", project / "results"):
                directory.mkdir(parents=True, exist_ok=True)
            session.mkdir()
            identity = session_lifecycle.capture_session_root(session)
            old_identity = formal_runner.ACTIVE_SESSION_IDENTITY
            old_phase_root = formal_runner.formal_phase.ROOT
            old_provenance_root = formal_provenance.ROOT
            environment_names = (
                session_lifecycle.ROOT_ENV,
                session_lifecycle.DEVICE_ENV,
                session_lifecycle.INODE_ENV,
            )
            old_environment = {name: os.environ.get(name) for name in environment_names}
            os.environ.update(identity.child_environment())
            formal_runner.ACTIVE_SESSION_IDENTITY = identity
            formal_runner.formal_phase.ROOT = project
            formal_provenance.ROOT = project
            try:
                session_lifecycle.require_session_root(identity, session)
                for relative in (
                    "phase", "logs", "build", "cora", "ppi",
                    "resources/cache/cora", "resources/cache/ppi",
                    "resources/checkpoints/cora_gat", "resources/checkpoints/ppi_gat",
                    "resources/canonical/tables",
                ):
                    (session / relative).mkdir(parents=True, exist_ok=True)
                (session / "build/gat_runner").write_text("frozen binary\n")
                (session / "logs/cora.log").write_text("formal log\n")
                (session / "resources/canonical/tables/authenticated.json").write_text("authenticated\n")
                (session / "ppi/manifest.json").write_text("authenticated snapshot\n")
                (session / "cora/partial.bin").write_text("failed snapshot\n")
                (project / "configs" / f"{session.name}_cora.cfg").write_text("failed snapshot config\n")
                result = project / "results" / session.name
                result.mkdir()
                (result / "collected.json").write_text("existing result\n")

                formal_runner._write_json(session / "state.json", {"session_id": session_id})
                state = {"session_id": session_id, "dataset_status": {"cora": {}}}
                formal_runner._record_dataset_failure(session, state, "cora", "fixture", "ordinary failure")
                self.assertTrue((session / "phase/cora_failure.json").is_file())

                formal_runner.formal_phase._remove_failed_snapshot(session, "cora")
                self.assertFalse((session / "cora").exists())
                self.assertFalse((session / "resources/cache/cora").exists())
                self.assertFalse((session / "resources/checkpoints/cora_gat").exists())
                self.assertFalse((project / "configs" / f"{session.name}_cora.cfg").exists())
                self.assertTrue((session / "build/gat_runner").is_file())
                self.assertTrue((session / "logs/cora.log").is_file())
                self.assertFalse((session / "results").exists())
                self.assertTrue((result / "collected.json").is_file())
                self.assertTrue((session / "resources/canonical/tables/authenticated.json").is_file())
                self.assertTrue((session / "ppi/manifest.json").is_file())

                tombstone = runs / f"{session.name}.original"
                session.rename(tombstone)
                session.mkdir()
                with self.assertRaisesRegex(session_lifecycle.SessionRootError, "must not mask"):
                    formal_runner._record_dataset_failure(session, state, "cora", "fixture", "must not mask")
                self.assertTrue(session.is_dir())
                self.assertFalse((session / "phase").exists())

                with mock.patch.object(formal_provenance.subprocess, "run") as build:
                    with mock.patch.object(sys, "argv", ["formal_provenance.py", "--output",
                                                           str(session / "build_manifest.json")]):
                        with self.assertRaises(session_lifecycle.SessionRootError):
                            formal_provenance.main()
                    build.assert_not_called()

                shutil.rmtree(session)
                with self.assertRaises(session_lifecycle.SessionRootError):
                    session_lifecycle.require_session_root(identity, session)
                with self.assertRaises(session_lifecycle.SessionRootError):
                    formal_runner._write_json(session / "state.json", {"session_id": session_id})
                self.assertFalse(session.exists())
            finally:
                formal_runner.ACTIVE_SESSION_IDENTITY = old_identity
                formal_runner.formal_phase.ROOT = old_phase_root
                formal_provenance.ROOT = old_provenance_root
                for name, value in old_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                if identity.root_fd is not None:
                    os.close(identity.root_fd)

    def test_finalization_capacity_uses_full_u64_segment_integer_size(self) -> None:
        fields = ("segment_index", "logical_offset", "actual_length", "padded_length", "basis_offset", "basis_count")
        expected_segment_bytes = (
            sys.getsizeof(dict.fromkeys(fields, 0)) +
            sum(sys.getsizeof(field) + sys.getsizeof((1 << 64) - 1) for field in fields) +
            2 * ctypes.sizeof(ctypes.c_void_p)
        )
        profile = formal_runner.formal_phase._srs_manifest_finalization_capacity(1, 1)
        self.assertEqual(profile["python_object_bytes_upper_bound"], 16 * 1024 + expected_segment_bytes)

    def test_cell_stage_markers_report_elapsed_pass_and_failure(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = formal_runner.formal_phase._cell_stage("setup", "cora", lambda: "done")
        self.assertEqual(result, "done")
        markers = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([marker["status"] for marker in markers], ["START", "PASS"])
        self.assertEqual([marker["zkdl_cell_stage"] for marker in markers], ["setup", "setup"])
        self.assertEqual([marker["dataset"] for marker in markers], ["cora", "cora"])
        self.assertGreaterEqual(markers[1]["elapsed_seconds"], 0.0)

        output = io.StringIO()
        def fail() -> None:
            raise formal_runner.formal_phase.PhaseError("fixture fail")

        with redirect_stdout(output), self.assertRaises(formal_runner.formal_phase.PhaseError):
            formal_runner.formal_phase._cell_stage("prove", "cora", fail)
        markers = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([marker["status"] for marker in markers], ["START", "FAIL"])
        self.assertEqual(markers[-1]["error"], "fixture fail")
        self.assertGreaterEqual(markers[-1]["elapsed_seconds"], 0.0)

    def _profile(self) -> dict[str, object]:
        rows: dict[str, dict[str, object]] = {}
        finalization_rows: dict[str, dict[str, int]] = {}
        for index, dataset in enumerate(formal_runner.DATASETS, start=1):
            retained = index * 10
            finalization_rows[dataset] = formal_runner.formal_phase._srs_manifest_finalization_capacity(index, index)
            gpu_workspace = 144 * (1 << 18) + 608
            rows[dataset] = {
                "additional_disk_peak_bytes": retained + index * 3,
                "cell_retained_artifact_bytes_upper_bound": retained,
                "host_peak_bytes": index * 100,
                "device_peak_bytes": gpu_workspace,
                "gpu_authenticated_basis_size": 1 << 18,
                "gpu_max_opening_padded_generator_size": 1,
                "gpu_workspace_required_bytes": gpu_workspace,
                "gpu_cuda_free_bytes_at_prepare": gpu_workspace + 1,
                "gpu_cuda_total_bytes_at_prepare": gpu_workspace + 2,
                "native_route_source_domain": index,
                "native_route_selector_factor_elements": index + 1,
                "phase_coverage": ["setup", "witness", "commitment", "prove", "verify"],
            }
        return {
            "schema": "zkdl_formal_capacity_profile_v4",
            "baseline": "zkdl",
            "backend": "cpu",
            "gpu_offload_stage": "hyrax_ipa_fold",
            "memory_accounting_kind": "shared_host_cuda",
            "device": {"name": "fixture GPU", "ordinal": 0, "uuid": "GPU-fixture"},
            "session_id": "0" * 32,
            "fs_device_id": 1,
            "preexisting_baseline_artifact_bytes": 0,
            "finalization_disk_bytes_upper_bound": 7,
            "finalization_host_peak_bytes_upper_bound": max(row["host_peak_bytes_upper_bound"] for row in finalization_rows.values()),
            "finalization_manifest_capacity": {"schema": "zkdl_srs_manifest_finalization_heap_v1", "datasets": finalization_rows},
            "bound_methods": {
                "additional_disk_peak": "source_derived_upper_bound",
                "host_peak": "source_derived_upper_bound",
                "gpu_workspace": "authenticated_basis_144B_plus_max_opening_608P",
                "shared_host_cuda": "shared_pool_min_host_cuda_ge_host_plus_gpu_workspace",
                "finalization_host_peak": "srs_manifest_schema_python_heap_upper_bound",
            },
            "datasets": rows,
        }

    def test_remaining_capacity_accounts_for_future_retained_prefix(self) -> None:
        profile = self._profile()
        all_remaining = formal_runner._remaining_capacity_requirements(profile, ())
        gpu_workspace = 144 * (1 << 18) + 608
        self.assertEqual(all_remaining["host_bytes"], 500 + gpu_workspace)
        self.assertEqual(all_remaining["device_bytes"], gpu_workspace)
        self.assertEqual(all_remaining["finalization_host_bytes"], profile["finalization_host_peak_bytes_upper_bound"])
        self.assertEqual(all_remaining["disk_bytes"], 172)
        after_two = formal_runner._remaining_capacity_requirements(profile, ("cora", "citeseer"))
        self.assertEqual(after_two["host_bytes"], 500 + gpu_workspace)
        self.assertEqual(after_two["finalization_host_bytes"], profile["finalization_host_peak_bytes_upper_bound"])
        self.assertEqual(after_two["disk_bytes"], 142)
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(profile, ("citeseer",))
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["device_peak_bytes"] = 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["additional_disk_peak_bytes"] = 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["finalization_manifest_capacity"]["datasets"]["ppi"]["segment_count"] += 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["schema"] = "zkdl_formal_capacity_profile_v1"
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        del malformed["datasets"]["ppi"]["native_route_selector_factor_elements"]
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["native_route_selector_factor_elements"] = 0
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["native_route_selector_factor_elements"] = 3
        malformed["datasets"]["ppi"]["native_route_source_domain"] = 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        del malformed["datasets"]["ppi"]["native_route_source_domain"]
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())
        malformed = json.loads(json.dumps(profile))
        malformed["datasets"]["ppi"]["native_route_source_domain"] = (1 << 30) + 1
        with self.assertRaises(formal_runner.RunnerError):
            formal_runner._remaining_capacity_requirements(malformed, ())

    @staticmethod
    def _pid_is_live(pid: int) -> bool:
        try:
            text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except OSError:
            return False
        close = text.rfind(")")
        if close < 0:
            return False
        fields = text[close + 2:].split()
        return bool(fields) and fields[0] != "Z"

    def test_owner_sigkill_terminates_phase_descendants_and_releases_lock(self) -> None:
        runs = ROOT / "runs"
        runs.mkdir(exist_ok=True)
        session_id = uuid.uuid4().hex
        session = runs / f"formal_{session_id}"
        session.mkdir(mode=0o700)
        ready_path = session / "owner_ready.json"
        pids_path = session / "descendant_pids.json"
        lock_path = runs / ".formal.lock"
        helper = TESTS / "formal_local_owner_death_helper.py"
        owner = subprocess.Popen(
            [sys.executable, str(helper), "owner", str(ROOT), str(session), session_id,
             str(ready_path), str(pids_path), "formal_runner.py"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not pids_path.is_file():
                if owner.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertTrue(pids_path.is_file(), "the phase-owned process group did not start")
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            pids = json.loads(pids_path.read_text(encoding="utf-8"))
            os.kill(owner.pid, signal.SIGKILL)
            owner.wait(timeout=5)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                tracked = [int(ready["phase_pid"]), int(pids["child"]), int(pids["grandchild"])]
                if not any(self._pid_is_live(pid) for pid in tracked):
                    break
                time.sleep(0.05)
            self.assertFalse(self._pid_is_live(int(ready["phase_pid"])), "phase survived its lifecycle owner")
            self.assertFalse(self._pid_is_live(int(pids["child"])), "phase child survived owner death")
            self.assertFalse(self._pid_is_live(int(pids["grandchild"])), "phase grandchild survived owner death")
            state = json.loads((session / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state.get("status"), "INTERRUPTED")
            self.assertEqual(state.get("interrupted_stage"), "cell:cora")
            lock_fd = os.open(lock_path, os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        finally:
            if owner.poll() is None:
                os.kill(owner.pid, signal.SIGKILL)
                owner.wait(timeout=5)
            shutil.rmtree(session, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
