#!/usr/bin/env python3
"""Bounded synthetic checks for the zkGPT formal runner repair.

This fixture never loads real resources or invokes a prover/capacity executable.
Its supervisor test copies the production supervisor/stop scripts into a private
temporary root and points them at a Python sleep stub.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import collect_results
import common
import formal_preflight
import gat_zkgpt_experiment
import run_formal
import validate_final_results


HEX = "a" * 64
SESSION = "0123456789abcdef0123456789abcdef"
MANIFEST = "b" * 64
RUNTIME = "c" * 64
NATIVE = "d" * 64
RESOURCE = "e" * 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def must_raise(call, description: str) -> str:
    try:
        call()
    except common.ChainError as exc:
        return str(exc)
    raise AssertionError(f"expected fail-closed rejection: {description}")


def test_signed_capacity_and_native_logs(work: Path) -> None:
    layer = {key: 0 for key in formal_preflight.GKR_LAYER_U64_FIELDS}
    layer.update({
        "layer_type": 1,
        "bit_length": -1,
        "max_bl_u": -1,
        "max_bl_v": -1,
        "need_phase2": True,
        "source_domains_exact": True,
    })
    payload = {"gkr_layer_count_upper_bound": 1, "gkr_layer_work_profile": [layer]}
    formal_preflight.validate_gkr_layer_work_profile(payload, "cora")
    parsed = json.loads(
        '{"max_bl_u":-1,"unsigned_count":0}',
        parse_int=formal_preflight.parse_capacity_integer,
        parse_float=formal_preflight.reject_floating_json_number,
        parse_constant=formal_preflight.reject_nonfinite_json_constant,
    )
    require(parsed["max_bl_u"] == -1 and parsed["unsigned_count"] == 0,
            "strict capacity parser did not preserve signed sentinel")
    try:
        formal_preflight.capacity_u64(-1, "estimated_peak_host_memory_bytes", "cora")
    except common.ChainError:
        pass
    else:
        raise AssertionError("negative resource byte count was accepted")
    bad_layer = dict(layer, size=-1)
    must_raise(lambda: formal_preflight.validate_gkr_layer_work_profile(
        {"gkr_layer_count_upper_bound": 1, "gkr_layer_work_profile": [bad_layer]}, "cora"),
        "negative nested uint64")

    root = work / "capacity-logs"
    (root / "configs").mkdir(parents=True)
    (root / "build").mkdir()
    fake_config = root / "configs" / "cora.json"
    fake_config.write_text("{}\n", encoding="utf-8")
    fake_binary = root / "build" / "prover"
    fake_binary.write_text("synthetic-only\n", encoding="utf-8")
    config = {"input_preprocess_id": "synthetic-preprocess-v1", "batching_rule": "fixed-1"}

    def run_mocked_capacity(output: str, returncode: int, stderr: str, log_subdir: str):
        completed = subprocess.CompletedProcess([str(fake_binary)], returncode, output, stderr)
        with mock.patch.object(formal_preflight, "config_with_auth", return_value=(fake_config, config)), \
             mock.patch.object(formal_preflight, "build_binary", return_value=fake_binary), \
             mock.patch.object(formal_preflight, "formal_child_environment", return_value={}), \
             mock.patch.object(formal_preflight.subprocess, "run", return_value=completed):
            return formal_preflight.run_capacity(root, "cora", "cpu", root / log_subdir)

    parse_logs = root / "parse-failure"
    try:
        run_mocked_capacity("not-json native stdout", 0, "native stderr diagnostic\n", "parse-failure")
    except common.ChainError as exc:
        require("stdout=" in str(exc) and "stderr=" in str(exc),
                "capacity parse failure did not report both native logs")
    else:
        raise AssertionError("malformed capacity JSON unexpectedly passed")
    require((parse_logs / "cora-cpu.stdout.log").read_text(encoding="utf-8") == "not-json native stdout",
            "capacity parse failure did not preserve stdout")
    require((parse_logs / "cora-cpu.stderr.log").read_text(encoding="utf-8") == "native stderr diagnostic\n",
            "capacity parse failure did not preserve stderr")

    blocked = {
        "schema": formal_preflight.CAPACITY_SCHEMA,
        "profile_kind": "formal_admission",
        "dataset": "cora",
        "backend": "cpu",
        "status": "BLOCKED",
        "memory_accounting_kind": "host-only",
        "device_peak_bytes": 0,
        "ready": False,
        "full_relation_bound": False,
        "proof_path": "",
        "reason": "synthetic native admission refusal",
        "num_nodes": 1,
        "num_edges": 1,
        "graph_count": 1,
        "canonical_graph_identity": {
            "digest": HEX,
            "nodes": 1,
            "edges": 1,
            "graphs": 1,
            "node_ptr": [0, 1],
            "edge_ptr": [0, 1],
            "input_preprocess_id": config["input_preprocess_id"],
            "batching_rule": config["batching_rule"],
        },
    }
    blocked_logs = root / "native-blocked"
    result = run_mocked_capacity(json.dumps(blocked), 17, "native admission rejected\n", "native-blocked")
    require(result["status"] == "BLOCKED", "nonzero native blocked admission was not retained")
    require((blocked_logs / "cora-cpu.stdout.log").read_text(encoding="utf-8") == json.dumps(blocked),
            "nonzero capacity run did not preserve stdout")
    require((blocked_logs / "cora-cpu.stderr.log").read_text(encoding="utf-8") == "native admission rejected\n",
            "nonzero capacity run did not preserve stderr")


def report(status: str, *, source_ready: bool = True, resources_ready: bool = True,
           shared_ready: bool = True) -> dict:
    capacity = {}
    for dataset in common.DATASETS:
        if status == "PARTIAL" and dataset == "cora":
            capacity[dataset] = {
                "dataset": dataset, "backend": "cpu", "status": "BLOCKED",
                "reason": "synthetic cell-local capacity blocker",
            }
        else:
            capacity[dataset] = {
                "dataset": dataset, "backend": "cpu", "status": "READY",
                "reason": "", "proof_path": common.CAPACITY_PROOF_PATH,
                "full_relation_bound": True,
            }
    return {
        "schema": common.PREFLIGHT_SCHEMA,
        "status": status,
        "shared_preconditions_ready": shared_ready,
        "source_ready": source_ready,
        "resources_ready": resources_ready,
        "blockers": [],
        "formal_experiment_eligible": status == "READY",
        "static_checks_passed": status == "READY",
        "resource_manifest_sha256": RESOURCE,
        "session_id": SESSION,
        "session_manifest_sha256": MANIFEST,
        "runtime_source_sha256": RUNTIME,
        "native_source_sha256": NATIVE,
        "capacity": capacity,
        "backends": {"cpu": {
            "ready": True,
            "source_tree_sha256": NATIVE,
            "prover_fingerprint": {},
            "verifier_fingerprint": {},
        }},
    }


@contextmanager
def patched_validator(module, root: Path, prover: Path, verifier: Path):
    del root
    replacements = {
        "verify_build_provenance": mock.DEFAULT,
        "build_binary": mock.DEFAULT,
        "verifier_binary": mock.DEFAULT,
        "sha256_file": mock.DEFAULT,
    }
    if hasattr(module, "runtime_source_sha256"):
        replacements["runtime_source_sha256"] = mock.DEFAULT
    with mock.patch.multiple(module, **replacements) as entered:
        if "runtime_source_sha256" in entered:
            entered["runtime_source_sha256"].return_value = RUNTIME
        entered["verify_build_provenance"].return_value = NATIVE
        entered["build_binary"].return_value = prover
        entered["verifier_binary"].return_value = verifier
        entered["sha256_file"].return_value = HEX
        yield


def test_preflight_gates(work: Path) -> None:
    root = work / "preflight-root"
    (root / "build").mkdir(parents=True)
    prover = root / "build" / "prover"
    verifier = root / "build" / "verifier"
    prover.write_text("fake prover\n", encoding="utf-8")
    verifier.write_text("fake verifier\n", encoding="utf-8")
    fingerprints = {
        "sha256": HEX,
        "size_bytes": prover.stat().st_size,
        "mtime_ns": prover.stat().st_mtime_ns,
    }
    partial = report("PARTIAL")
    partial["backends"]["cpu"]["prover_fingerprint"] = dict(fingerprints)
    partial["backends"]["cpu"]["verifier_fingerprint"] = {
        "sha256": HEX, "size_bytes": verifier.stat().st_size,
        "mtime_ns": verifier.stat().st_mtime_ns,
    }
    ready = report("READY")
    ready["backends"]["cpu"] = json.loads(json.dumps(partial["backends"]["cpu"]))

    with patched_validator(collect_results, root, prover, verifier):
        collect_results._validate_preflight(root, "cpu", partial, RESOURCE, SESSION, MANIFEST, NATIVE,
                                            allow_partial=True)
        must_raise(lambda: collect_results._validate_preflight(
            root, "cpu", partial, RESOURCE, SESSION, MANIFEST, NATIVE),
            "default collector strict status gate")
        must_raise(lambda: collect_results._validate_preflight(
            root, "cpu", dict(partial, source_ready=False), RESOURCE, SESSION, MANIFEST, NATIVE,
            allow_partial=True), "PARTIAL source_ready=false")
        must_raise(lambda: collect_results._validate_preflight(
            root, "cpu", dict(partial, resources_ready=False), RESOURCE, SESSION, MANIFEST, NATIVE,
            allow_partial=True), "PARTIAL resources_ready=false")
        must_raise(lambda: collect_results._validate_preflight(
            root, "cpu", dict(partial, shared_preconditions_ready=False), RESOURCE, SESSION, MANIFEST, NATIVE,
            allow_partial=True), "PARTIAL shared gate false")

    session = {"runtime_source_sha256": RUNTIME, "native_source_sha256": NATIVE}
    with patched_validator(gat_zkgpt_experiment, root, prover, verifier):
        gat_zkgpt_experiment.validate_preflight_binding(
            root, partial, RESOURCE, "cpu", "pubmed", SESSION, session, MANIFEST)
        must_raise(lambda: gat_zkgpt_experiment.validate_preflight_binding(
            root, partial, RESOURCE, "cpu", "cora", SESSION, session, MANIFEST),
            "cell-local BLOCKED capacity")
        must_raise(lambda: gat_zkgpt_experiment.validate_preflight_binding(
            root, dict(partial, source_ready=False), RESOURCE, "cpu", "pubmed", SESSION, session, MANIFEST),
            "PARTIAL cell source_ready=false")
        must_raise(lambda: gat_zkgpt_experiment.validate_preflight_binding(
            root, dict(partial, resources_ready=False), RESOURCE, "cpu", "pubmed", SESSION, session, MANIFEST),
            "PARTIAL cell resources_ready=false")

    with patched_validator(validate_final_results, root, prover, verifier):
        must_raise(lambda: validate_final_results._validate_preflight(
            root, partial, RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE),
            "final validator PARTIAL strict gate")
        validate_final_results._validate_preflight(
            root, ready, RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE)
        must_raise(lambda: validate_final_results._validate_preflight(
            root, dict(ready, source_ready=False), RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE),
            "READY source_ready=false")
        must_raise(lambda: validate_final_results._validate_preflight(
            root, dict(ready, resources_ready=False), RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE),
            "READY resources_ready=false")


def test_final_preflight_gate(work: Path) -> None:
    root = work / "final-preflight-root"
    (root / "build").mkdir(parents=True)
    prover = root / "build" / "prover"
    verifier = root / "build" / "verifier"
    prover.write_text("fake prover\n", encoding="utf-8")
    verifier.write_text("fake verifier\n", encoding="utf-8")
    ready = report("READY")
    ready["backends"]["cpu"]["prover_fingerprint"] = {
        "sha256": HEX, "size_bytes": prover.stat().st_size,
        "mtime_ns": prover.stat().st_mtime_ns,
    }
    ready["backends"]["cpu"]["verifier_fingerprint"] = {
        "sha256": HEX, "size_bytes": verifier.stat().st_size,
        "mtime_ns": verifier.stat().st_mtime_ns,
    }
    with patched_validator(validate_final_results, root, prover, verifier):
        validate_final_results._validate_preflight(
            root, ready, RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE)
        must_raise(lambda: validate_final_results._validate_preflight(
            root, report("PARTIAL"), RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE),
            "final validator PARTIAL strict gate")
        must_raise(lambda: validate_final_results._validate_preflight(
            root, dict(ready, source_ready=False), RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE),
            "READY source_ready=false")
        must_raise(lambda: validate_final_results._validate_preflight(
            root, dict(ready, resources_ready=False), RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE),
            "READY resources_ready=false")
        must_raise(lambda: validate_final_results._validate_preflight(
            root, dict(ready, shared_preconditions_ready=False), RESOURCE, SESSION, MANIFEST, RUNTIME, NATIVE),
            "READY shared_preconditions_ready=false")


def test_serial_failures_and_preservation(work: Path) -> None:
    root = work / "owner-root"
    (root / "baseline-local" / "runs" / "formal" / "prior" / "cora").mkdir(parents=True)
    preserved = root / "baseline-local" / "runs" / "formal" / "prior" / "cora" / "result.json"
    preserved.write_text('{"status":"VERIFY_OK","synthetic_prior_artifact":true}\n', encoding="utf-8")
    prior_bytes = preserved.read_bytes()
    session_id = "11111111111111111111111111111111"
    profile_hash = "f" * 64
    datasets = {}
    for dataset in common.DATASETS:
        if dataset == "cora":
            datasets[dataset] = {"status": "BLOCKED", "reason": "synthetic preflight capacity block"}
        else:
            datasets[dataset] = {
                "status": "READY", "cell_retained_artifact_bytes_upper_bound": 100_000,
            }
    profile = {"datasets": datasets}
    calls: list[str] = []
    finalize_calls: list[str] = []

    def fake_phase(fake_root, _lock_stream, sid, session_dir, phase, dataset,
                   _owner_token, _owner_start, _logs):
        require(fake_root == root and sid == session_id, "synthetic owner root/session mismatch")
        if phase == "prepare":
            session_dir.mkdir(parents=True)
            (session_dir / "receipts").mkdir()
            common.atomic_write_json(session_dir / "session_status.json", {
                "session_id": sid, "status": "RUNNING", "active_stage": "prepare",
                "completed_stages": [], "stage_status": {}, "dataset_failures": {},
            }, root)
            run_formal._record_session_stage(root, sid, "prepare")
            owner = run_formal._current_owner_record(root)
            run_formal._update_owner_record(root, completed_stages=owner.get("completed_stages", []) + ["prepare"])
            return {"session_dir": str(session_dir), "local_capacity_profile_sha256": profile_hash}
        if phase == "finalize":
            finalize_calls.append("finalize")
            raise AssertionError("failed synthetic matrix must not finalize")
        require(phase == "cell" and dataset is not None, "unexpected synthetic owner phase")
        calls.append(dataset)
        receipt_path = session_dir / "receipts" / f"{dataset}.json"
        receipt = {
            "status": "VERIFY_OK", "session_id": sid, "dataset": dataset,
            "independent_verification": True, "proof_sha256": hashlib_sha(dataset),
            "verification_report_path": f"synthetic/{dataset}/verifier_report.json",
            "verification_report_sha256": hashlib_sha("report-" + dataset),
        }
        common.atomic_write_json(receipt_path, receipt, root)
        _run_relative, run_path = common.run_directory(root, dataset, sid)
        (run_path / "synthetic.receipt.marker").write_text("bounded fixture\n", encoding="utf-8")
        run_formal._record_session_stage(root, sid, f"cell:{dataset}")
        owner = run_formal._current_owner_record(root)
        completed = owner.get("completed_stages", []) + [f"cell:{dataset}"]
        run_formal._update_owner_record(root, completed_stages=completed)
        rel = str(receipt_path.relative_to(root))
        return {
            "receipt_path": rel,
            "receipt_sha256": common.sha256_file(receipt_path),
            "proof_sha256": receipt["proof_sha256"],
            "verification_report_path": receipt["verification_report_path"],
            "verification_report_sha256": receipt["verification_report_sha256"],
        }

    try:
        with mock.patch.object(run_formal, "formal_child_environment", return_value={}), \
             mock.patch.object(run_formal, "install_process_signal_cleanup", return_value=None), \
             mock.patch.object(run_formal, "_recover_stale_owner", return_value=""), \
             mock.patch.object(run_formal, "_proc_start_ticks", return_value=12345), \
             mock.patch.object(run_formal.secrets, "token_hex", side_effect=[session_id, "2" * 64]), \
             mock.patch.object(run_formal, "_load_prepared", return_value=(
                 {}, {}, RESOURCE, {"local_capacity_profile": profile,
                                   "local_capacity_profile_sha256": profile_hash},
                 root / "fake-preflight.json", MANIFEST)), \
             mock.patch.object(run_formal, "_run_owner_phase", side_effect=fake_phase), \
             mock.patch.object(run_formal, "_capacity_gate", side_effect=lambda _r, _s, _d, _p, _pending, stage:
                               (_ for _ in ()).throw(common.ChainError("synthetic live capacity gate failure"))
                               if stage == "before-cell:citeseer" else {}):
            try:
                run_formal._run_official(root)
            except common.ChainError as exc:
                require("attempted all 5 datasets" in str(exc),
                        "owner did not report attempted full serial matrix")
            else:
                raise AssertionError("synthetic failures unexpectedly completed official owner")

        require(calls == ["pubmed", "ppi", "ogbn_arxiv"],
                f"owner did not continue after blocked/failing cells: {calls}")
        require(not finalize_calls, "owner finalized an incomplete synthetic matrix")
        status_path = root / "results" / "formal" / session_id / "session_status.json"
        status = common.load_json(status_path, "synthetic formal session status")
        require(status["status"] == "FAILED", "owner did not persist terminal failure")
        require(status["stage_status"].get("cell:cora") == "FAILED" and
                status["stage_status"].get("cell:citeseer") == "FAILED",
                "owner did not persist per-cell failures")
        require(set(status["dataset_failures"]) == {"cora", "citeseer"},
                "owner did not persist both dataset failure reasons")
        for dataset in ("pubmed", "ppi", "ogbn_arxiv"):
            receipt = common.load_json(root / "results" / "formal" / session_id / "receipts" / f"{dataset}.json",
                                       "synthetic cell receipt")
            require(receipt["status"] == "VERIFY_OK" and receipt["independent_verification"] is True,
                    f"owner did not retain synthetic successful receipt for {dataset}")
        require(preserved.read_bytes() == prior_bytes,
                "owner altered an existing prior valid result while recording current failures")
    finally:
        os.environ.pop("ZKGPT_FORMAL_SESSION_ID", None)


def hashlib_sha(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def wait_status(path: Path, predicate, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)
            continue
        if predicate(last):
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for synthetic supervisor state: {last}")


def test_safe_stop(temp_parent: Path) -> None:
    root = temp_parent / "safe-stop-root"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts" / "formal_detached_supervisor.py", scripts)
    shutil.copy2(REPO_ROOT / "scripts" / "stop_formal.py", scripts)
    (scripts / "run_formal.py").write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(120)\n", encoding="utf-8")
    launch_id = uuid.uuid4().hex
    launch_dir = root / "baseline-local" / "formal-detached" / launch_id
    launch_dir.mkdir(parents=True)
    console_path = launch_dir / "console.log"
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                              cwd=str(temp_parent), start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    supervisor = None
    child_pid = None
    try:
        with console_path.open("wb") as console:
            supervisor = subprocess.Popen(
                [sys.executable, str(scripts / "formal_detached_supervisor.py"),
                 "--root", str(root), "--launch-id", launch_id],
                cwd=str(root), start_new_session=True, stdout=console, stderr=subprocess.STDOUT)
        status_path = launch_dir / "status.json"
        state = wait_status(status_path, lambda item: item.get("status") == "RUNNING", 8.0)
        child_pid = state["child_pid"]
        require(supervisor.poll() is None and victim.poll() is None,
                "synthetic supervisor or unrelated victim unexpectedly exited")

        tampered = dict(state)
        tampered["supervisor_start_ticks"] = state["supervisor_start_ticks"] + 1
        status_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
        rejected = subprocess.run(
            [sys.executable, str(scripts / "stop_formal.py"), "--launch-id", launch_id],
            cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5, check=False)
        require(rejected.returncode == 2 and "ZKGPT_FORMAL_STOP_REFUSED" in rejected.stderr,
                f"tampered supervisor start ticks were not refused: {rejected.stdout} {rejected.stderr}")
        require(supervisor.poll() is None and victim.poll() is None,
                "metadata mismatch sent a signal to the synthetic supervisor or unrelated victim")

        status_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        stopped = subprocess.run(
            [sys.executable, str(scripts / "stop_formal.py"), "--launch-id", launch_id],
            cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=12, check=False)
        require(stopped.returncode == 0 and "ZKGPT_FORMAL_STOP_COMPLETE" in stopped.stdout,
                f"valid exact-identity stop failed: {stopped.stdout} {stopped.stderr}")
        supervisor.wait(timeout=3)
        terminal = json.loads(status_path.read_text(encoding="utf-8"))
        require(terminal.get("status") == "INTERRUPTED" and terminal.get("received_signal") == "SIGTERM",
                f"synthetic stop did not persist INTERRUPTED/SIGTERM: {terminal}")
        require(victim.poll() is None, "safe stop terminated the unrelated synthetic victim")
        require("ZKGPT_FORMAL_STOP_SIGNALLED" in stopped.stdout,
                "safe stop did not report signaling the exact supervisor")
    finally:
        if supervisor is not None and supervisor.poll() is None:
            supervisor.terminate()
            try:
                supervisor.wait(timeout=3)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.wait(timeout=3)
        # Only clean the child recorded by this private synthetic launch, and
        # only while its exact PID and recorded start ticks still match.
        if child_pid is not None and Path(f"/proc/{child_pid}/stat").exists():
            try:
                raw = Path(f"/proc/{child_pid}/stat").read_text(encoding="ascii")
                fields = raw[raw.rfind(")") + 2:].split()
                live_ticks = int(fields[19])
                current = json.loads(status_path.read_text(encoding="utf-8"))
                if live_ticks == current.get("child_start_ticks") and current.get("child_pid") == child_pid:
                    os.killpg(child_pid, signal.SIGTERM)
            except (OSError, ValueError, IndexError, json.JSONDecodeError, ProcessLookupError):
                pass
        if victim.poll() is None:
            victim.terminate()
            try:
                victim.wait(timeout=3)
            except subprocess.TimeoutExpired:
                victim.kill()
                victim.wait(timeout=3)


def main() -> int:
    resume_from_preflight = sys.argv[1:] == ["--resume-preflight"]
    if sys.argv[1:] and not resume_from_preflight:
        raise SystemExit("usage: runner_contract_fixture.py [--resume-preflight]")
    work = Path(__file__).resolve().parents[1]
    work.mkdir(parents=True, exist_ok=True)
    (work / "functional-fixture.log").parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthetic-", dir=str(work)) as temporary:
        temp_parent = Path(temporary)
        if not resume_from_preflight:
            test_signed_capacity_and_native_logs(temp_parent)
        if resume_from_preflight:
            test_final_preflight_gate(temp_parent)
        else:
            test_preflight_gates(temp_parent)
        test_serial_failures_and_preservation(temp_parent)
        test_safe_stop(temp_parent)
    summary = {
        "status": "PASS",
        "scope": "synthetic runner contracts only; no dataset/resource/prover execution",
        "mode": "resume-from-preflight-gates" if resume_from_preflight else "full-synthetic-fixture",
        "checks": [
            "signed GKR sentinel and negative uint64 rejection",
            "capacity stdout/stderr retained for malformed output and native BLOCKED exit",
            "PARTIAL reader gate, strict collector default, per-cell READY-only, final READY-only",
            "source/resources/shared gates fail closed",
            "serial blocked/failing cells continue, preserve errors/receipts/prior result, skip finalization",
            "safe stop rejects PID start-tick mismatch and signals only exact synthetic supervisor",
        ],
    }
    log = work / "functional-fixture.log"
    log.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(f"FUNCTIONAL_FIXTURE_LOG={log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
