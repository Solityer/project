#!/usr/bin/env python3
"""Short baseline-local positive/negative tests for schemas and tooling."""

from __future__ import annotations

import json
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
from prepare_form_resources import (
    analytic_sorted_lookup_plan,
    ledger_bounds_plan,
    make_tables,
    relation_family_domains,
    table_digest,
    table_shape_plan,
)
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

    def test_exp_map_covers_exact_leaky_relu_output_span(self) -> None:
        for range_bits, expected_span, expected_padded in ((12, 4915, 8192), (14, 19661, 32768)):
            values = {
                "range_bits": str(range_bits),
                "quant_cfg_id": f"test_quant_{range_bits}",
                "static_table_id": f"test_table_{range_bits}",
                "K_out": "1",
            }
            shape = table_shape_plan(values, "0" * 64)
            shape_plan = analytic_sorted_lookup_plan(values, shape, nodes=3, profile=[(1, 2)], edges=4)
            shape_exp_family = next(item for item in shape_plan["table_families"] if item["table_ref"] == "static_exp")
            self.assertEqual(shape_exp_family["logical_rows"], expected_span + 1)
            self.assertEqual(shape_exp_family["input_max"], expected_span)

            table = make_tables(values, "0" * 64)
            span = max(table["lrelu"]) - min(table["lrelu"])
            self.assertEqual(span, expected_span)
            self.assertEqual(table["exp_max"], span)
            self.assertEqual(len(table["exp"]), span + 1)
            self.assertEqual(table["exp_padded_domain"], expected_padded)

            planner_values = {
                **values,
                "dataset": "synthetic",
                "hidden_profile": "[1x2]",
                "d_in_profile": "[4]",
                "num_classes": "7",
                "batch_graphs": "1",
            }
            _, _, ledger_families, _ = ledger_bounds_plan(planner_values, degree=4, model_max=3)
            family_domains = relation_family_domains(
                planner_values,
                nodes=3,
                edges=4,
                table=table,
                model_domains=[1],
                graph_count=1,
                ledger_families=ledger_families,
                quantization={
                    "s_out_pre": "16",
                    "s_y_prime": "64",
                    "s_h_prime_hidden_0": "64",
                },
            )
            delta_width = expected_span.bit_length()
            max_range_views = [
                name for name in family_domains
                if name.startswith("hidden/layer0/head0/max_range/view/range_bit/")
            ]
            self.assertEqual(len(max_range_views), delta_width)
            self.assertIn("hidden/layer0/head0/trace/range_bits", family_domains)

            plan = analytic_sorted_lookup_plan(values, table, nodes=3, profile=[(1, 2)], edges=4)
            exp_family = next(item for item in plan["table_families"] if item["table_ref"] == "static_exp")
            self.assertEqual(exp_family["logical_rows"], span + 1)
            self.assertEqual(exp_family["input_min"], 0)
            self.assertEqual(exp_family["input_max"], span)
            self.assertEqual(exp_family["padded_domain"], expected_padded)

            stale = dict(table)
            old_max = 1 << range_bits
            stale["exp_max"] = old_max
            stale["exp"] = table["exp"][:old_max + 1]
            self.assertNotEqual(table_digest(stale), table["digest"])

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


if __name__ == "__main__":
    unittest.main()
