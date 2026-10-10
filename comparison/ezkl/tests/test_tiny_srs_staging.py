from __future__ import annotations

import hashlib
import inspect
import json
import os
import signal
import sys
import subprocess
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import formal_common  # noqa: E402
import run_tiny_cpu_e2e  # noqa: E402


class TinySrsStagingTest(unittest.TestCase):
    def test_source_snapshot_copies_small_code_files_without_sharing_inodes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-tiny-source-copy-") as temporary:
            root = Path(temporary)
            source_root = root / "official"
            fixture_root = root / "fixture"
            (source_root / "src").mkdir(parents=True)
            (source_root / "scripts").mkdir()
            source_file = source_root / "src/lib.rs"
            script_file = source_root / "scripts/runner.py"
            source_file.write_text("pub fn value() -> u8 { 7 }\n", encoding="utf-8")
            script_file.write_text("VALUE = 7\n", encoding="utf-8")
            (source_root / "Cargo.toml").write_text("[package]\nname='fixture'\n", encoding="utf-8")

            run_tiny_cpu_e2e.stage_source_snapshot(source_root, fixture_root)

            for source, staged in (
                (source_file, fixture_root / "src/lib.rs"),
                (script_file, fixture_root / "scripts/runner.py"),
            ):
                with self.subTest(path=source.relative_to(source_root)):
                    self.assertEqual(staged.read_bytes(), source.read_bytes())
                    self.assertNotEqual(
                        (staged.stat().st_dev, staged.stat().st_ino),
                        (source.stat().st_dev, source.stat().st_ino),
                    )
                    source.write_text("changed after staging\n", encoding="utf-8")
                    self.assertNotEqual(staged.read_bytes(), source.read_bytes())

    def test_tiny_fixture_uses_canonical_preprocessing_identity_for_each_dataset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-tiny-input-contract-") as temporary:
            root = Path(temporary)
            run_tiny_cpu_e2e.create_tiny_resources(root)
            for dataset in run_tiny_cpu_e2e.DATASETS:
                expected = "ogbn_arxiv_raw_float32" if dataset == "ogbn_arxiv" else "row_normalize"
                meta = run_tiny_cpu_e2e.parse_cfg(
                    root / "resources/canonical/cache" / dataset / "meta.cfg"
                )
                secure = run_tiny_cpu_e2e.parse_cfg(
                    root / "resources/canonical/configs" / f"{dataset}_secure.cfg"
                )
                checkpoint = json.loads(
                    (root / "resources/canonical/checkpoints" / f"{dataset}_gat/manifest.json").read_text()
                )
                with self.subTest(dataset=dataset):
                    self.assertEqual(meta["input_preprocess_id"], expected)
                    self.assertEqual(secure["input_preprocess_id"], expected)
                    self.assertEqual(checkpoint["input_preprocess_id"], expected)

    def test_same_filesystem_fixture_stages_readonly_hardlink_and_hashes_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-tiny-srs-link-") as temporary:
            root = Path(temporary)
            source_root = root / "official"
            fixture_root = root / "fixture"
            source_path = source_root / "resources/native_srs/shared.srs"
            source_path.parent.mkdir(parents=True)
            source_bytes = b"bounded synthetic srs fixture bytes"
            source_path.write_bytes(source_bytes)
            manifest_source = source_root / "resources/native_srs/srs_manifest.json"
            manifest_source.write_text('{"schema_version":1}\n', encoding="utf-8")
            digest = hashlib.sha256(source_bytes).hexdigest()
            entry = {
                "path": "resources/native_srs/shared.srs",
                "logrows": 4,
                "sha256": digest,
                "trusted": True,
                "reduced": False,
                "generated": False,
                "size": len(source_bytes),
                "srs_id": "synthetic-test-only",
            }
            manifest = {"max_logrows": 4, "entries": {"cora": entry}}
            with mock.patch.object(formal_common, "native_srs_contract", return_value=("https://example.invalid/srs", 4, {4: digest})):
                with mock.patch.object(formal_common, "validate_native_srs_manifest", return_value=manifest):
                    lock_path = source_root / "runs/.ezkl-formal.lock"
                    lock_path.parent.mkdir(parents=True)
                    lock_path.write_bytes(b"existing formal lock payload\n")
                    original_lock_payload = lock_path.read_bytes()
                    lock_fd = run_tiny_cpu_e2e.acquire_tiny_formal_lock(source_root)
                    try:
                        with self.assertRaisesRegex(run_tiny_cpu_e2e.TinyE2EError, "another formal ezkl session"):
                            run_tiny_cpu_e2e.acquire_tiny_formal_lock(source_root)
                        destination, actual_digest, _srs_id, max_logrows, staged_size = run_tiny_cpu_e2e.stage_trusted_full_srs(
                            source_root, fixture_root
                        )

                        source_stat = source_path.stat()
                        destination_stat = destination.stat()
                        self.assertEqual(destination.read_bytes(), source_bytes)
                        self.assertEqual(actual_digest, digest)
                        self.assertEqual(max_logrows, 4)
                        self.assertEqual(staged_size, len(source_bytes))
                        self.assertEqual((destination_stat.st_dev, destination_stat.st_ino), (source_stat.st_dev, source_stat.st_ino))
                        self.assertEqual(
                            (fixture_root / "resources/native_srs/srs_manifest.json").read_bytes(),
                            manifest_source.read_bytes(),
                        )
                        self.assertEqual(lock_path.read_bytes(), original_lock_payload)
                        self.assertTrue(run_tiny_cpu_e2e.unlink_staged_srs_hardlink(source_path, destination))
                        self.assertFalse(destination.exists())
                    finally:
                        run_tiny_cpu_e2e.release_tiny_formal_lock(lock_fd)

    def test_srs_copy_mode_does_not_share_the_official_inode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-tiny-srs-copy-") as temporary:
            root = Path(temporary)
            source_root = root / "official"
            fixture_root = root / "fixture"
            source_path = source_root / "resources/native_srs/shared.srs"
            source_path.parent.mkdir(parents=True)
            source_bytes = b"bounded synthetic copy fixture"
            source_path.write_bytes(source_bytes)
            manifest_source = source_root / "resources/native_srs/srs_manifest.json"
            manifest_source.write_text('{"schema_version":1}\n', encoding="utf-8")
            digest = hashlib.sha256(source_bytes).hexdigest()
            entry = {
                "path": "resources/native_srs/shared.srs",
                "logrows": 4,
                "sha256": digest,
                "trusted": True,
                "reduced": False,
                "generated": False,
                "size": len(source_bytes),
                "srs_id": "synthetic-test-only",
            }
            manifest = {"max_logrows": 4, "entries": {"cora": entry}}
            with mock.patch.object(formal_common, "native_srs_contract", return_value=("https://example.invalid/srs", 4, {4: digest})):
                with mock.patch.object(formal_common, "validate_native_srs_manifest", return_value=manifest):
                    destination, actual_digest, _srs_id, _max_logrows, staged_size = run_tiny_cpu_e2e.stage_trusted_full_srs(
                        source_root, fixture_root, allow_hardlink=False
                    )
            self.assertEqual(destination.read_bytes(), source_bytes)
            self.assertEqual(actual_digest, digest)
            self.assertEqual(staged_size, len(source_bytes))
            self.assertNotEqual(
                (destination.stat().st_dev, destination.stat().st_ino),
                (source_path.stat().st_dev, source_path.stat().st_ino),
            )
            self.assertFalse(run_tiny_cpu_e2e.unlink_staged_srs_hardlink(source_path, destination))

    def test_tiny_setup_and_prove_gates_add_one_manifest_sized_output_srs_copy(self) -> None:
        import run_hybrid_formal

        with tempfile.TemporaryDirectory(prefix="ezkl-tiny-capacity-srs-") as temporary:
            root = Path(temporary)
            source_srs = root / "trusted.srs"
            source_srs.write_bytes(b"1234567")
            query = {
                "device_uuid": "GPU-test",
                "total_memory_bytes": 100,
                "free_memory_bytes": 80,
                "integrated_device": False,
                "device_workspace_peak_bytes": 5,
                "native_type_sizes": {"scalar_bytes": 32, "affine_bytes": 64},
            }
            receipt = {"query": query, "result_sha256": "a" * 64}
            requirements = {
                "host_known_bytes": 10,
                "setup_key_artifact_upper_bytes": 11,
                "setup_mapped_peak_bytes": 13,
                "proof_spill_known_bytes": 17,
            }
            admission = {
                "native_memory_model": {
                    "cuda_total_memory_bytes": 100,
                    "size_of_fr_bytes": 32,
                    "lookup_table_cardinality_upper_bounds": [1],
                },
                "sha256": "b" * 64,
                "logrows": 4,
                "domain_size": 16,
                "constraint_system": {
                    "minimum_rows": 8,
                    "num_unique_lookup_table_expression_groups": 1,
                },
            }
            sample = {
                "device_uuid": "GPU-test",
                "free_memory_bytes": 80,
                "total_memory_bytes": 100,
                "integrated_device": False,
            }
            common_patches = (
                mock.patch("run_hybrid_formal._build_native_cub_workspace_query", return_value=receipt),
                mock.patch("run_hybrid_formal._native_capacity_requirements", return_value=requirements),
                mock.patch("run_hybrid_formal._sample_native_cuda_memory", return_value=sample),
                mock.patch("run_hybrid_formal._effective_host_memory_available_bytes", return_value=(1000, {})),
                mock.patch("run_hybrid_formal._host_os_diagnostic_reserve_bytes", return_value=(0, {})),
                mock.patch("run_hybrid_formal._filesystem_available_bytes", return_value=(1000, {})),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5]:
                capacity = run_tiny_cpu_e2e.run_tiny_setup_capacity_gate(
                    root,
                    root / "runs/tiny",
                    root / "session",
                    "a" * 32,
                    admission,
                    {},
                    {"uuid": "GPU-test"},
                    {},
                    {},
                    source_srs,
                    7,
                )
                self.assertEqual(capacity["setup_check"]["disk_required_known_bytes"], 31)
                self.assertEqual(capacity["setup_check"]["disk_budget_components"]["pending_output_srs_copy_count"], 1)
                self.assertEqual(capacity["setup_check"]["disk_budget_components"]["pending_output_srs_copy_bytes"], 7)

                with mock.patch("run_hybrid_formal._native_prepare_budget", return_value={"taskcap_bytes": 1}), \
                     mock.patch("run_hybrid_formal._validated_native_prepare_exports", return_value=("1", "0", "1")):
                    _exports, report = run_tiny_cpu_e2e.run_tiny_prove_capacity_gate(
                        root,
                        root / "runs/tiny",
                        root / "session",
                        "a" * 32,
                        admission,
                        capacity,
                        {"uuid": "GPU-test"},
                        {},
                        {},
                        source_srs,
                        7,
                    )
                self.assertEqual(report["prove_check"]["disk_required_known_bytes"], 24)
                self.assertEqual(report["prove_check"]["disk_budget_components"]["pending_output_srs_copy_count"], 1)
                self.assertEqual(report["prove_check"]["disk_budget_components"]["pending_output_srs_copy_bytes"], 7)

    def test_admitted_k_above_tiny_limit_is_rejected_before_capacity_and_setup(self) -> None:
        self.assertEqual(run_tiny_cpu_e2e.validate_tiny_admission_logrows({"logrows": 17}, 17), 17)
        for admission in ({"logrows": 18}, {"logrows": True}, {}):
            with self.subTest(admission=admission), self.assertRaises(run_tiny_cpu_e2e.TinyE2EError):
                run_tiny_cpu_e2e.validate_tiny_admission_logrows(admission, 12)

        source = inspect.getsource(run_tiny_cpu_e2e.main)
        ordered_stages = (
            "admission = admissions[TINY_DATASET]",
            "validate_tiny_admission_logrows(admission, args.max_srs_logrows)",
            "capacity = run_tiny_setup_capacity_gate(",
            "admission_negative_cases = negative_admission_consumer(",
            '"initial-prepare", commands["initial_prepare"]',
        )
        positions = [source.index(stage) for stage in ordered_stages]
        self.assertEqual(positions, sorted(positions))

    def test_tiny_setup_logrows_cap_rejects_large_circuit_before_staging(self) -> None:
        with self.assertRaisesRegex(run_tiny_cpu_e2e.TinyE2EError, "must stay small"):
            run_tiny_cpu_e2e.main(
                Namespace(
                    plan=False,
                    stage_timeout=45.0,
                    overall_timeout=300.0,
                    max_srs_logrows=18,
                    backend="gpu",
                    cleanup="never",
                )
            )

    def test_sigterm_reaps_stage_group_including_grandchild_ignoring_sigterm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-tiny-sigterm-") as temporary:
            root = Path(temporary)
            pid_path = root / "grandchild.pid"
            stage_pid_path = root / "stage.pid"
            stage_term_path = root / "stage-term.received"
            log_dir = root / "logs"
            scripts = ROOT / "scripts"
            harness = r"""
import os, signal, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import run_tiny_cpu_e2e as tiny
root = Path(sys.argv[2])
grandchild_code = r'''
import os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path = __import__('pathlib').Path
Path(sys.argv[1]).write_text(f'{os.getpid()} {os.getpgrp()}')
time.sleep(60)
'''
child_code = r'''
import os, signal, subprocess, sys, time
from pathlib import Path
def record_first_term(_signum, _frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(sys.argv[5]).write_text('received')
signal.signal(signal.SIGTERM, record_first_term)
Path(sys.argv[4]).write_text(f'{os.getpid()} {os.getpgrp()}')
subprocess.Popen([sys.argv[1], '-c', sys.argv[2], sys.argv[3]])
time.sleep(60)
'''
signal.signal(signal.SIGTERM, tiny._raise_on_sigterm)
tiny.run_stage(
    "sigterm-child-group",
    [sys.executable, "-c", child_code, sys.executable, grandchild_code, sys.argv[3], sys.argv[5], sys.argv[6]],
    root=root,
    log_dir=Path(sys.argv[4]),
    env=os.environ.copy(),
    timeout=60.0,
    overall_deadline=time.monotonic() + 60.0,
)
"""
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    harness,
                    str(scripts),
                    str(root),
                    str(pid_path),
                    str(log_dir),
                    str(stage_pid_path),
                    str(stage_term_path),
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            ready_deadline = time.monotonic() + 5.0
            while (
                (not pid_path.exists() or not stage_pid_path.exists())
                and process.poll() is None
                and time.monotonic() < ready_deadline
            ):
                time.sleep(0.02)
            self.assertTrue(pid_path.is_file(), "stage grandchild did not start before the bounded deadline")
            self.assertTrue(stage_pid_path.is_file(), "isolated stage did not record its process-group identity")
            grandchild_pid, grandchild_pgid = (int(value) for value in pid_path.read_text().split())
            stage_pid, stage_pgid = (int(value) for value in stage_pid_path.read_text().split())
            self.assertEqual(stage_pgid, grandchild_pgid)
            self.assertEqual(stage_pid, stage_pgid)

            def process_running_in_group(pid: int, pgid: int) -> bool:
                stat_path = Path(f"/proc/{pid}/stat")
                if not stat_path.is_file():
                    return False
                fields = stat_path.read_text().rsplit(")", 1)[-1].split()
                return len(fields) >= 3 and fields[0] != "Z" and fields[2].isdigit() and int(fields[2]) == pgid

            first_term_at = time.monotonic()
            os.kill(process.pid, signal.SIGTERM)
            cleanup_deadline = time.monotonic() + 2.0
            while not stage_term_path.exists() and process.poll() is None and time.monotonic() < cleanup_deadline:
                time.sleep(0.005)
            self.assertTrue(stage_term_path.is_file(), "stage process group did not receive the first SIGTERM")
            self.assertTrue(process_running_in_group(grandchild_pid, stage_pgid), "grandchild left before the cleanup grace window")
            self.assertLess(time.monotonic() - first_term_at, 0.4, "first SIGTERM cleanup grace window elapsed before the repeat")

            # Re-deliver SIGTERM only after the wrapper has entered its process-group
            # cleanup wait. The wrapper must ignore this second signal until cleanup ends.
            os.kill(process.pid, signal.SIGTERM)
            try:
                self.assertEqual(process.wait(timeout=5.0), 128 + signal.SIGTERM)
            except subprocess.TimeoutExpired:
                os.killpg(stage_pgid, signal.SIGKILL)
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=2.0)
                self.fail("SIGTERM did not unwind the active stage within the bounded deadline")

            reap_deadline = time.monotonic() + 3.0
            while process_running_in_group(grandchild_pid, stage_pgid) and time.monotonic() < reap_deadline:
                time.sleep(0.02)
            self.assertFalse(
                process_running_in_group(grandchild_pid, stage_pgid),
                "SIGTERM cleanup left the SIGTERM-ignoring grandchild alive",
            )
            self.assertFalse(
                process_running_in_group(stage_pid, stage_pgid),
                "SIGTERM cleanup left the stage process alive",
            )
            if process.stderr is not None:
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
