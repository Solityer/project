#!/usr/bin/env python3
"""Single bounded script/session/timing/process contract check; never runs a proof."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from collect_results import _validate_preflight  # noqa: E402
from common import (  # noqa: E402
    DATASETS,
    FORMAL_SESSION_SCHEMA,
    PAPER_TIMING_SEMANTICS,
    PREFLIGHT_SCHEMA,
    ChainError,
    derive_run_nonce,
    formal_session_path,
    require_formal_session,
    require_timing,
    run_captured,
    run_logged,
)
from test_scripts_schema import ProductionContractTests  # noqa: E402


def _run_selected_contract_tests() -> None:
    names = (
        "test_formal_child_environment_rejects_loader_overrides",
        "test_formal_run_lock_fd_can_be_passed_to_owned_child",
        "test_native_runtime_provenance_binds_same_local_mcl",
        "test_runtime_provenance_ignores_path_shadow_ldd",
        "test_build_provenance_rejects_missing_or_changed_mcl_identity",
        "test_result_schema_rejects_missing_privacy_field",
        "test_result_schema_rejects_unsupported_schema",
        "test_final_validator_rejects_noncanonical_cell_without_io",
        "test_cleanup_rejects_volatile_symlink",
    )
    suite = unittest.TestSuite(ProductionContractTests(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("selected production contract tests failed")


def _timing_fixture() -> Dict[str, Any]:
    timing = {
        "load": 0.0,
        "registration": 0.0,
        "witness": 0.0,
        "preparation": 0.2,
        "preparation_net": 0.0,
        "aux_commitment": 0.1,
        "lasso_query_commitment": 0.2,
        "lasso_family_generator_derive": 0.0,
        "lasso_family_commitments": 0.0,
        "lasso_family_selected_row_mapping": 0.0,
        "initial_commitment_blinding_prepare": 0.0,
        "commitment": 0.3,
        "prove_total": 1.0,
        "self_verify": 0.1,
        "prove": 0.9,
        "verify": 0.2,
        "serialize": 0.3,
        "artifact_wire_self_check": 0.05,
        "artifact_write": 0.1,
        "artifact_readback": 0.2,
    }
    return {"paper_timing_semantics": PAPER_TIMING_SEMANTICS, "timing_ms": timing}


def _check_timing_contract() -> None:
    valid = _timing_fixture()
    require_timing(valid, "round-trip tolerance fixture")

    negative = _timing_fixture()
    negative["timing_ms"]["witness"] = -0.001
    try:
        require_timing(negative, "negative timing fixture")
    except ChainError:
        pass
    else:
        raise AssertionError("negative timing was accepted")

    over_budget = _timing_fixture()
    over_budget["timing_ms"]["prove_total"] = 0.1
    try:
        require_timing(over_budget, "over-budget timing fixture")
    except ChainError:
        pass
    else:
        raise AssertionError("negative derived Prove Time was accepted")

    stale = _timing_fixture()
    stale["paper_timing_semantics"] = "zkgpt-gat-paper-timing-v1"
    try:
        require_timing(stale, "stale timing fixture")
    except ChainError:
        pass
    else:
        raise AssertionError("stale timing semantics were accepted")


def _check_native_source_and_session_bindings() -> None:
    session_id = "1" * 32
    session_sha = "2" * 64
    runtime_sha = "3" * 64
    resource_sha = "4" * 64
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "formal_experiment_eligible": True,
        "resource_manifest_sha256": resource_sha,
        "session_id": session_id,
        "session_manifest_sha256": session_sha,
        "runtime_source_sha256": runtime_sha,
        "native_source_sha256": "5" * 64,
    }
    try:
        _validate_preflight(
            ROOT, "cpu", preflight, resource_sha, session_id, session_sha,
            "6" * 64,
        )
    except ChainError as exc:
        if "native source" not in str(exc):
            raise
    else:
        raise AssertionError("preflight with a mixed native-source identity was accepted")

    previous = os.environ.get("ZKGPT_FORMAL_SESSION_ID")
    try:
        for malformed in (
            {
                "schema": FORMAL_SESSION_SCHEMA,
                "session_id": session_id,
                "datasets": list(DATASETS[:-1]),
                "run_nonces": {dataset: derive_run_nonce(session_id, dataset) for dataset in DATASETS[:-1]},
            },
            {
                "schema": FORMAL_SESSION_SCHEMA,
                "session_id": "7" * 32,
                "datasets": list(DATASETS),
                "run_nonces": {dataset: derive_run_nonce(session_id, dataset) for dataset in DATASETS},
            },
        ):
            with tempfile.TemporaryDirectory(prefix="session-contract-", dir=str(ROOT / "results")) as temp:
                test_root = Path(temp)
                manifest_path = formal_session_path(test_root, session_id)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
                os.environ["ZKGPT_FORMAL_SESSION_ID"] = session_id
                try:
                    require_formal_session(test_root)
                except ChainError:
                    pass
                else:
                    raise AssertionError("missing or mixed session identity was accepted")
    finally:
        if previous is None:
            os.environ.pop("ZKGPT_FORMAL_SESSION_ID", None)
        else:
            os.environ["ZKGPT_FORMAL_SESSION_ID"] = previous


_SPAWN_DESCENDANT = r"""
import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")
print("leader-exited", flush=True)
"""

_INHERITED_OWNER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from common import ChainError, run_captured, run_logged
root = Path(sys.argv[2])
mode = sys.argv[3]
pidfile = sys.argv[4]
logs = Path(sys.argv[5])
command = [sys.executable, "-c", sys.argv[6], pidfile]
try:
    if mode == "logged":
        run_logged(command, root, logs / "out.log", logs / "err.log", "inherited-check",
                   extra_env={"ZKGPT_OWNED_PROCESS_GROUP": "1"})
    else:
        run_captured(command, root, extra_env={"ZKGPT_OWNED_PROCESS_GROUP": "1"})
except ChainError:
    print("INHERITED_DRAIN_REJECTED", flush=True)
    raise SystemExit(0)
raise SystemExit(4)
"""


def _pid_is_live(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            state = proc_stat.read_text(encoding="ascii").rsplit(")", 1)[1].split()[0]
            return state != "Z"
        except (OSError, IndexError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _require_pidfile_stopped(pidfile: Path) -> None:
    if not pidfile.is_file():
        raise AssertionError(f"descendant pid was not recorded: {pidfile}")
    pid = int(pidfile.read_text(encoding="ascii"))
    deadline = time.monotonic() + 4.0
    while _pid_is_live(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_is_live(pid):
        raise AssertionError(f"owned descendant {pid} survived process-group cleanup")


def _stop_test_owned_group(pgid: int, leader: subprocess.Popen[Any]) -> None:
    """Clean only the process group this harness itself created with setsid."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if leader.poll() is None:
        leader.wait(timeout=3.0)


def _check_process_helper(mode: str, owned: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="process-contract-", dir=str(ROOT / "results")) as temp:
        directory = Path(temp)
        pidfile = directory / "descendant.pid"
        if owned:
            command = [sys.executable, "-c", _SPAWN_DESCENDANT, str(pidfile)]
            try:
                if mode == "logged":
                    run_logged(
                        command, ROOT, directory / "out.log", directory / "err.log", "owned-check",
                        extra_env={"ZKGPT_OWNED_PROCESS_GROUP": "0"},
                    )
                else:
                    run_captured(command, ROOT, extra_env={"ZKGPT_OWNED_PROCESS_GROUP": "0"})
            except ChainError as exc:
                if "descendants" not in str(exc) and "output pipes" not in str(exc):
                    raise
            else:
                raise AssertionError(f"{mode} accepted an owned descendant holding its pipes")
            _require_pidfile_stopped(pidfile)
            return

        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "scripts")
        env["ZKGPT_OWNED_PROCESS_GROUP"] = "1"
        owner = subprocess.Popen(
            [sys.executable, "-c", _INHERITED_OWNER, str(ROOT / "scripts"), str(ROOT),
             mode, str(pidfile), str(directory), _SPAWN_DESCENDANT],
            cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
        try:
            stdout, stderr = owner.communicate(timeout=18.0)
            if owner.returncode != 0 or "INHERITED_DRAIN_REJECTED" not in stdout:
                raise AssertionError(
                    f"{mode} inherited drain contract failed: exit={owner.returncode}; "
                    f"stdout={stdout[-500:]}; stderr={stderr[-500:]}"
                )
        finally:
            _stop_test_owned_group(owner.pid, owner)
        _require_pidfile_stopped(pidfile)


def _run_process_checks() -> None:
    for mode in ("logged", "captured"):
        for owned in (True, False):
            owner = "owned" if owned else "inherited"
            print(f"process-contract: {mode}/{owner}", flush=True)
            _check_process_helper(mode, owned)


def main() -> int:
    print("closure: selected production-contract tests", flush=True)
    _run_selected_contract_tests()
    print("closure: timing semantics and bounded decomposition", flush=True)
    _check_timing_contract()
    print("closure: session and preflight native-source binding", flush=True)
    _check_native_source_and_session_bindings()
    print("closure: owned/inherited process-group cleanup", flush=True)
    _run_process_checks()
    print("FORMAL_CLOSURE_CHECK_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
