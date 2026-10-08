#!/usr/bin/env python3
"""Bounded orchestration mock; deliberately never reports a formal proof success."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from contextlib import contextmanager


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VARIANT = "wo_shared_neighborhood"


MOCK_CMAKE = r'''#!/usr/bin/env python3
import importlib.util
import json
import os
import pathlib
import shutil
import sys

args = sys.argv[1:]
call_log = pathlib.Path(os.environ["MOCK_CMAKE_CALL_LOG"])
with call_log.open("a", encoding="utf-8") as stream:
    stream.write(" ".join(args) + "\n")
if "--build" not in args:
    root = pathlib.Path(args[args.index("-S") + 1])
    build = pathlib.Path(args[args.index("-B") + 1])
    cuda = next(item.split("=", 1)[1] for item in args if item.startswith("-DGATZK_ENABLE_CUDA_BACKEND="))
    build.mkdir(parents=True, exist_ok=True)
    (build / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\nGATZK_ENABLE_CUDA_BACKEND:BOOL=" + cuda + "\n", encoding="utf-8"
    )
    provenance_path = root / "tools" / "build_provenance.py"
    spec = importlib.util.spec_from_file_location("mock_build_provenance", provenance_path)
    provenance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provenance)
    manifest_path = build / "generated" / "gatzk" / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(provenance.source_manifest_document(root), sort_keys=True), encoding="utf-8")
else:
    build = pathlib.Path(args[args.index("--build") + 1])
    mock_tool = pathlib.Path(os.environ["MOCK_GATZK_TOOL"])
    for target in ("gatzk_run", "gatzk_registry_register", "gatzk_registry_inspect"):
        destination = build / target
        shutil.copyfile(mock_tool, destination)
        destination.chmod(0o755)
'''


MOCK_GIT = r'''#!/usr/bin/env python3
import sys
if sys.argv[1:] == ["rev-parse", "HEAD"]:
    print("mock-source-head")
    raise SystemExit(0)
raise SystemExit(1)
'''


MOCK_GATZK_TOOL = r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import sys

def config(path):
    result = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result

name = pathlib.Path(sys.argv[0]).name
if name == "gatzk_registry_register":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    if cfg.get("protocol_variant") not in {"wo_shared_neighborhood", "wo_feature_folding"}:
        raise SystemExit("mock registration received a wrong variant")
    if "verifier_registry_digest" in cfg:
        raise SystemExit("mock candidate config must not inherit a stale digest")
    project_root = pathlib.Path.cwd()
    srs_manifest = json.loads((project_root / cfg["srs_path"] / "srs_manifest.json").read_text(encoding="utf-8"))
    identity = "|".join((cfg["protocol_variant"], cfg["dataset"], cfg["model_arch_id"], cfg["srs_id"]))
    registry_id = hashlib.sha256(identity.encode()).hexdigest()
    artifact = {
        "dataset": cfg["dataset"],
        "model_arch_id": cfg["model_arch_id"],
        "model_param_id": cfg["model_param_id"],
        "protocol_variant": cfg["protocol_variant"],
        "protocol_schema_id": srs_manifest["protocol_schema_id"],
        "instance_schema_hash": hashlib.sha256(identity.encode()).hexdigest(),
        "registry_id": registry_id,
        "quant_cfg_id": cfg["quant_cfg_id"],
        "security_budget_ledger_digest": hashlib.sha256(b"security-ledger").hexdigest(),
        "srs_id": cfg["srs_id"],
        "srs_manifest_hash": srs_manifest["manifest_hash"],
        "srs_degree_bound": srs_manifest["degree_bound"],
        "static_table_id": cfg["static_table_id"],
    }
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    print(f"candidate_registry_path={path} registry_id={registry_id} candidate_registry_digest={digest}")
elif name == "gatzk_registry_inspect":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--trusted-digest", required=True)
    args = parser.parse_args()
    raw = pathlib.Path(args.registry).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.trusted_digest:
        raise SystemExit("mock inspector digest mismatch")
    artifact = json.loads(raw)
    artifact.update({"artifact_sha256": digest, "registry_digest": digest, "security_budget_passed": True})
    inspect_mode = os.environ.get("MOCK_INSPECT_MODE", "")
    if inspect_mode == "wrong-variant":
        artifact["protocol_variant"] = "zkgat"
    elif inspect_mode == "wrong-digest":
        artifact["artifact_sha256"] = "0" * 64
    elif inspect_mode == "wrong-protocol":
        artifact["protocol_schema_id"] = hashlib.sha256(b"wrong-static-protocol").hexdigest()
    elif inspect_mode == "wrong-srs":
        artifact["srs_id"] = "wrong_srs"
    print(json.dumps(artifact, sort_keys=True))
else:
    mode = os.environ.get("MOCK_RUN_MODE", "failure")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol-variant", required=True)
    parser.add_argument("--compute-backend", required=True)
    parser.add_argument("--benchmark-mode", required=True)
    parser.add_argument("--export-tag", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    backend = "gpu" if args.compute_backend == "cuda_hotspots" else "cpu"
    if mode == "verify-line-only":
        print(f"dataset={cfg['dataset']} config={args.config} backend={backend} protocol_variant={args.protocol_variant} status=VERIFY_OK commitment_time_ms=0.000 prove_time_ms=0.000 verify_time_ms=0.000 proof_size_bytes=1")
        raise SystemExit(0)
    if mode == "wrong-run-variant":
        run_dir = pathlib.Path(cfg["export_dir"]) / args.export_tag / "single"
        run_dir.mkdir(parents=True, exist_ok=True)
        proof = b"fixture-only, not a proof"
        (run_dir / "proof.bin").write_bytes(proof)
        manifest = {"proof_hash": hashlib.sha256(proof).hexdigest(), "protocol_name": "zkgat"}
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        print(f"dataset={cfg['dataset']} config={args.config} backend={backend} protocol_variant={args.protocol_variant} status=VERIFY_OK commitment_time_ms=0.000 prove_time_ms=0.000 verify_time_ms=0.000 proof_size_bytes=1")
        raise SystemExit(0)
    print("MOCK_PROOF_STOP: no formal proof is generated by this test fixture", file=sys.stderr)
    raise SystemExit(17)
'''


class AblationMatrixEntrypointMockTest(unittest.TestCase):
    @contextmanager
    def _run_mock_matrix(self, run_mode: str = "failure", inspect_mode: str = ""):
        with tempfile.TemporaryDirectory(prefix="gatzk-ablation-entrypoint-mock-") as temp:
            temp_root = pathlib.Path(temp)
            project = temp_root / "project"
            project.mkdir()
            scripts = project / "scripts"
            scripts.mkdir()
            shutil.copy2(PROJECT_ROOT / "scripts" / "run_ablation_matrix.sh", scripts / "run_ablation_matrix.sh")
            shutil.copy2(PROJECT_ROOT / "scripts" / "ablation_matrix_support.py", scripts / "ablation_matrix_support.py")
            tools = project / "tools"
            tools.mkdir()
            shutil.copy2(PROJECT_ROOT / "tools" / "build_provenance.py", tools / "build_provenance.py")
            configs = project / "configs"
            configs.mkdir()
            for dataset in ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"):
                base = PROJECT_ROOT / "configs" / f"{dataset}_secure.cfg"
                shutil.copy2(base, configs / base.name)
                values = {}
                for line in base.read_text(encoding="utf-8").splitlines():
                    if "=" in line and not line.lstrip().startswith("#"):
                        key, value = line.split("=", 1)
                        values[key.strip()] = value.strip()
                source_srs = PROJECT_ROOT / values["srs_path"] / "srs_manifest.json"
                target_srs = project / values["srs_path"] / "srs_manifest.json"
                target_srs.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_srs, target_srs)

            mock_bin = temp_root / "mock-bin"
            mock_bin.mkdir()
            fake_cmake = mock_bin / "cmake"
            fake_git = mock_bin / "git"
            mock_tool = temp_root / "mock_gatzk_tool.py"
            fake_cmake.write_text(MOCK_CMAKE, encoding="utf-8")
            fake_git.write_text(MOCK_GIT, encoding="utf-8")
            mock_tool.write_text(MOCK_GATZK_TOOL, encoding="utf-8")
            for executable in (fake_cmake, fake_git, mock_tool):
                executable.chmod(0o755)
            call_log = temp_root / "cmake_calls.log"
            env = os.environ.copy()
            env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
            env["MOCK_CMAKE_CALL_LOG"] = str(call_log)
            env["MOCK_GATZK_TOOL"] = str(mock_tool)
            env["MOCK_RUN_MODE"] = run_mode
            env["MOCK_INSPECT_MODE"] = inspect_mode

            completed = subprocess.run(
                ["bash", str(scripts / "run_ablation_matrix.sh"), "--protocol-variant", VARIANT, "--backend", "cpu"],
                cwd=project,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=90,
            )
            sessions = list((project / "runs" / "ablation_sessions" / VARIANT).iterdir())
            self.assertEqual(len(sessions), 1)
            yield project, completed, sessions[0], call_log

    def test_variant_registry_pinning_and_formal_invocation_fail_closed(self) -> None:
        with self._run_mock_matrix() as (project, completed, session, call_log):
            self.assertEqual(completed.returncode, 17, completed.stdout)
            self.assertIn("MOCK_PROOF_STOP", completed.stdout)
            self.assertNotIn("VERIFY_OK", completed.stdout)
            self.assertEqual(len(list((session / "preflight").glob("*.registry.json"))), 5)
            self.assertFalse(list((session / "evidence").glob("*.json")))

            index = json.loads((session / "ablation_session_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["protocol_variant"], VARIANT)
            self.assertTrue(index["preflight_complete"])
            self.assertEqual(index["status"], "formal_runs_in_progress")
            for dataset, info in index["datasets"].items():
                self.assertEqual(info["protocol_variant"], VARIANT)
                self.assertRegex(info["registry_sha256"], r"^[0-9a-f]{64}$")
                registry = pathlib.Path(info["registry_path"])
                self.assertEqual(info["registry_sha256"], hashlib.sha256(registry.read_bytes()).hexdigest())
                self.assertEqual(info["runs"]["cpu"]["status"], "not_verified")
                base = project / "configs" / f"{dataset}_secure.cfg"
                generated = pathlib.Path(info["generated_config"])
                self.assertNotEqual(info["base_config_sha256"], info["generated_config_sha256"])
                self.assertEqual(info["base_config_sha256"], hashlib.sha256(base.read_bytes()).hexdigest())
                values = {}
                for line in generated.read_text(encoding="utf-8").splitlines():
                    if "=" in line and not line.lstrip().startswith("#"):
                        key, value = line.split("=", 1)
                        values[key.strip()] = value.strip()
                self.assertEqual(values["protocol_variant"], VARIANT)
                self.assertEqual(values["verifier_registry_digest"], info["registry_sha256"])
            run_log = next((session / "logs").glob("cpu_*.log"))
            self.assertIn(f"--protocol-variant {VARIANT}", run_log.read_text(encoding="utf-8"))
            rows = (session / "status.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn("\t17\t", rows[1])
            self.assertIn("--parallel", call_log.read_text(encoding="utf-8"))

    def test_verify_ok_log_without_proof_or_report_is_not_complete(self) -> None:
        with self._run_mock_matrix(run_mode="verify-line-only") as (_project, completed, session, _call_log):
            self.assertEqual(completed.returncode, 1, completed.stdout)
            self.assertIn("status=VERIFY_OK", completed.stdout)
            index = json.loads((session / "ablation_session_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["status"], "formal_runs_in_progress")
            self.assertEqual(index["datasets"]["cora"]["runs"]["cpu"]["status"], "not_verified")
            self.assertFalse(list((session / "evidence").glob("*.json")))
            run_log = next((session / "logs").glob("cpu_*.log"))
            self.assertIn("run manifest must be a regular file", run_log.read_text(encoding="utf-8"))
            status_row = (session / "status.tsv").read_text(encoding="utf-8").splitlines()[1]
            self.assertIn("\t0\t0\t1\t", status_row)

    def test_mismatched_variant_in_run_manifest_is_not_complete(self) -> None:
        with self._run_mock_matrix(run_mode="wrong-run-variant") as (_project, completed, session, _call_log):
            self.assertEqual(completed.returncode, 1, completed.stdout)
            self.assertIn("status=VERIFY_OK", completed.stdout)
            index = json.loads((session / "ablation_session_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["status"], "formal_runs_in_progress")
            self.assertFalse(list((session / "evidence").glob("*.json")))
            run_log = next((session / "logs").glob("cpu_*.log"))
            self.assertIn("run manifest protocol_name does not match", run_log.read_text(encoding="utf-8"))

    def test_registry_identity_mismatches_stop_before_any_formal_cell(self) -> None:
        for mode in ("wrong-variant", "wrong-digest", "wrong-protocol", "wrong-srs"):
            with self.subTest(mode=mode):
                with self._run_mock_matrix(inspect_mode=mode) as (_project, completed, session, _call_log):
                    self.assertNotEqual(completed.returncode, 0, completed.stdout)
                    self.assertFalse(list((session / "logs").glob("cpu_*.log")))
                    self.assertEqual(len(list((session / "preflight").glob("*.registry.json"))), 1)
                    self.assertFalse(list((session / "evidence").glob("*.json")))

    def test_session_index_rejects_evidence_from_another_session(self) -> None:
        spec = importlib.util.spec_from_file_location("ablation_support_for_test", PROJECT_ROOT / "scripts" / "ablation_matrix_support.py")
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        with tempfile.TemporaryDirectory(prefix="gatzk-ablation-index-negative-") as temp:
            project = pathlib.Path(temp) / "project"
            session = project / "runs" / "ablation_sessions" / VARIANT / "session-a"
            session.mkdir(parents=True)
            for dataset in support.DATASETS:
                base = PROJECT_ROOT / "configs" / f"{dataset}_secure.cfg"
                support.write_variant_config(base, session / "configs" / f"{dataset}.cfg", VARIANT,
                                             f"runs/ablation_sessions/{VARIANT}/session-a/results/{dataset}",
                                             f"runs/ablation_sessions/{VARIANT}/session-a/registries/{dataset}.registry", None)
            config_path = session / "configs" / "cora.cfg"
            config, _ = support.parse_config(config_path)
            digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
            evidence_path = session / "evidence" / "cpu_cora.json"
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text(json.dumps({
                "protocol_variant": VARIANT,
                "dataset": "cora",
                "backend": "cpu",
                "registry_digest": "",
                "config_sha256": digest,
                "status": "VERIFY_OK",
                "run_dir": str(project / "runs" / "ablation_sessions" / VARIANT / "session-b" / "results" / "cora"),
            }), encoding="utf-8")
            args = type("Args", (), {"project_root": str(project), "session_root": str(session), "variant": VARIANT, "backends": "cpu"})()
            with self.assertRaisesRegex(ValueError, "outside this session results directory"):
                support.write_session_index(args)


if __name__ == "__main__":
    unittest.main()
