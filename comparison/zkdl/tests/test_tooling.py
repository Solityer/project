#!/usr/bin/env python3
"""Short baseline-local positive/negative tests for schemas and tooling."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"
FIXTURES = ROOT / "tests" / "fixtures"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from final_validate import validate_collection
from result_artifacts import ArtifactError, DATASETS, require_formal_five


class ToolingTests(unittest.TestCase):
    def run_cmd(self, *args: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_schema_fixtures(self) -> None:
        pairs = {
            "public_statement.schema.json": "public_statement.json",
            "proof_artifact.schema.json": "proof_artifact.json",
            "result.schema.json": "verified_run/run.record",
            "model_registry.schema.json": "model_registry.json",
        }

        for schema_name, fixture_name in pairs.items():
            schema = json.loads((SCHEMAS / schema_name).read_text(encoding="utf-8"))
            fixture_path = FIXTURES / fixture_name
            if fixture_name.endswith("run.record"):
                instance = {}
                for line in fixture_path.read_text(encoding="utf-8").splitlines():
                    key, value = line.split("=", 1)
                    if value in {"0", "1"} and key in {"verifier_reads_witness", "gpu_kernel_executed", "silent_fallback", "output_bound"}:
                        instance[key] = value == "1"
                    elif key.endswith("_bytes"):
                        instance[key] = int(value)
                    elif key.endswith("_seconds"):
                        instance[key] = float(value)
                    else:
                        instance[key] = value
            else:
                instance = json.loads(fixture_path.read_text(encoding="utf-8"))
            jsonschema.validate(instance, schema)

    def test_collector_and_final_validator(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory) / "collected.json"

            self.run_cmd("python3", str(SCRIPTS / "collect_results.py"), "--runs", "tests/fixtures/verified_run", "--output", str(output), expected=1)
        self.run_cmd("python3", str(SCRIPTS / "collect_results.py"), "--runs", "tests/fixtures/blocked_run", expected=1)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            collection = str(Path(directory) / "collected.json")
            self.run_cmd("python3", str(SCRIPTS / "final_validate.py"), "--runs", "tests/fixtures/verified_run",
                         "--collection", collection, expected=1)
            self.run_cmd("python3", str(SCRIPTS / "final_validate.py"), "--runs", "tests/fixtures/blocked_run",
                         "--collection", collection, expected=1)

    def test_formal_five_gate(self) -> None:
        rows = [{
            "dataset": dataset,
            "independent_verification": "1",
            "silent_fallback": "0",
            "commitment_time_semantics": "native_tensor_commitments",
            "commitment_time_seconds": "0.1",
            "prove_time_seconds": "0.2",
            "verify_time_seconds": "0.3",
            "proof_size_bytes": "128",
        } for dataset in DATASETS]
        require_formal_five(rows)
        with self.assertRaises(ArtifactError):
            require_formal_five(rows[:-1])
        with self.assertRaises(ArtifactError):
            require_formal_five(rows[:-1] + [dict(rows[0])])
        for field, value in (("proof_size_bytes", "0"), ("commitment_time_seconds", "nan")):
            malformed = [dict(row) for row in rows]
            malformed[0][field] = value
            with self.assertRaises(ArtifactError):
                require_formal_five(malformed)

    def test_collection_binding(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            temporary_root = Path(directory)
            runs = temporary_root / "runs" / "formal_contract_session"
            runs.mkdir(parents=True)
            collection = temporary_root / "results" / runs.name / "collected.json"
            collection.parent.mkdir(parents=True)
            rows = [{
                "dataset": dataset,
                "independent_verification": "1",
                "silent_fallback": "0",
                "commitment_time_semantics": "native_tensor_commitments",
                "commitment_time_seconds": "0.1",
                "prove_time_seconds": "0.2",
                "verify_time_seconds": "0.3",
                "proof_size_bytes": "128",
            } for dataset in DATASETS]
            summary = {
                "schema": "zkDL_gat_result_collection_v3",
                "mode": "formal_single_path",
                "session_id": runs.name,
                "runs_path": "runs/" + runs.name,
                "collection_path": "results/" + runs.name + "/collected.json",
                "rows": rows,
            }

            def validate(candidate: dict[str, object]) -> None:
                collection.write_text(json.dumps(candidate), encoding="utf-8")
                validate_collection(temporary_root, runs, collection, rows, fixture_mode=False)

            validate(summary)
            for invalid_rows in (rows[:-1], rows[:-1] + [dict(rows[0])]):
                malformed = dict(summary, rows=invalid_rows)
                with self.assertRaises(ArtifactError):
                    validate(malformed)
            changed_metric = [dict(row) for row in rows]
            changed_metric[0]["prove_time_seconds"] = "0.25"
            with self.assertRaises(ArtifactError):
                validate(dict(summary, rows=changed_metric))
            with self.assertRaises(ArtifactError):
                validate(dict(summary, session_id="stale_session"))

    def test_manifest_synthetic_and_negative_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            fixture_root = Path(directory)
            (fixture_root / "configs").mkdir()
            (fixture_root / "data" / "cache" / "tiny").mkdir(parents=True)
            (fixture_root / "models" / "checkpoints" / "tiny").mkdir(parents=True)
            (fixture_root / "configs" / "quantization_scale_chain.cfg").write_text("quant_cfg_id=q\n", encoding="utf-8")
            config = fixture_root / "configs" / "tiny.cfg"
            config.write_text("\n".join((
                "dataset=tiny", "cache_root=data/cache", "checkpoint_bundle=models/checkpoints/tiny",
                "export_dir=runs/tiny", "quant_cfg_id=q", "data_root=data",
            )) + "\n", encoding="utf-8")
            (fixture_root / "data" / "cache" / "tiny" / "meta.cfg").write_text("fixture\n", encoding="utf-8")
            (fixture_root / "models" / "checkpoints" / "tiny" / "manifest.json").write_text("{}\n", encoding="utf-8")
            self.run_cmd("python3", str(SCRIPTS / "asset_manifest.py"), "--config", str(config))
            manifest = json.loads((fixture_root / "runs" / "tiny" / "asset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "zkDL_gat_asset_manifest_v1")
            os.symlink(fixture_root / "outside", fixture_root / "data" / "cache" / "tiny" / "escape")
            self.run_cmd("python3", str(SCRIPTS / "asset_manifest.py"), "--config", str(config), expected=1)
            config.write_text(config.read_text(encoding="utf-8").replace("cache_root=data/cache", "cache_root=/tmp"), encoding="utf-8")
            self.run_cmd("python3", str(SCRIPTS / "asset_manifest.py"), "--config", str(config), expected=1)


if __name__ == "__main__":
    unittest.main()
