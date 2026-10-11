from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formal_common import ContractError  # noqa: E402
import run_hybrid_formal as formal_runner  # noqa: E402
from run_hybrid_formal import _validate_native_shplonk_shape  # noqa: E402


class NativeAdmissionSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = Path(__file__).parent / "fixtures/native_admission_cora_zero_shuffle.json"
        self.admission = json.loads(fixture.read_text(encoding="utf-8"))
        self.cs = self.admission["constraint_system"]
        self.memory = self.admission["native_memory_model"]
        self.domain_size = self.admission["domain_size"]

    def test_zero_shuffle_count_is_valid_only_when_current_cs_also_has_no_shuffles(self) -> None:
        self.assertEqual(self.admission["logrows"], 17)
        self.assertEqual(self.cs["num_shuffle_arguments"], 0)
        self.assertEqual(self.memory["shplonk_shuffle_argument_count"], 0)
        _validate_native_shplonk_shape("cora", self.memory, self.cs, self.domain_size)

        for invalid in (True, -1, 1):
            with self.subTest(shuffle_count=invalid):
                memory = copy.deepcopy(self.memory)
                memory["shplonk_shuffle_argument_count"] = invalid
                with self.assertRaises(ContractError):
                    _validate_native_shplonk_shape("cora", memory, self.cs, self.domain_size)

    def test_other_shplonk_shape_counts_remain_positive(self) -> None:
        for field in (
            "shplonk_query_polynomial_count",
            "shplonk_rotation_set_count",
            "shplonk_max_rotation_count",
            "shplonk_unique_opening_point_count",
            "shplonk_max_commitments_per_rotation_set",
            "shplonk_permutation_product_set_count",
            "shplonk_advice_query_polynomial_count",
            "shplonk_fixed_query_polynomial_count",
            "shplonk_lookup_argument_count",
            "shplonk_proof_instance_count",
            "shplonk_quotient_piece_count",
        ):
            with self.subTest(field=field):
                memory = copy.deepcopy(self.memory)
                memory[field] = 0
                with self.assertRaises(ContractError):
                    _validate_native_shplonk_shape("cora", memory, self.cs, self.domain_size)

        memory = copy.deepcopy(self.memory)
        memory["shplonk_rotation_set_commitment_counts"][0] = 0
        with self.assertRaises(ContractError):
            _validate_native_shplonk_shape("cora", memory, self.cs, self.domain_size)

    def test_permutation_keygen_hashmap_budget_scales_and_rejects_old_schema(self) -> None:
        small = copy.deepcopy(self.admission)
        small_requirements = formal_runner._native_capacity_requirements(
            small, {"scalar_bytes": 32, "affine_bytes": 64}
        )

        larger = copy.deepcopy(small)
        cs = larger["constraint_system"]
        cs["permutation_copy_call_count"] *= 2
        cs["permutation_successor_entry_count_upper"] = min(
            cs["num_permutation_columns"] * larger["domain_size"],
            2 * cs["permutation_copy_call_count"],
        )
        larger_requirements = formal_runner._native_capacity_requirements(
            larger, {"scalar_bytes": 32, "affine_bytes": 64}
        )

        self.assertEqual(
            small_requirements["permutation_hashmap_heap_peak_bytes"],
            3 * self.memory["size_of_permutation_hashmap_header_bytes"]
            + small_requirements["permutation_hashmap_backing_bytes"]
            + small_requirements["permutation_hashmap_resize_old_bytes_upper"],
        )
        self.assertEqual(
            small_requirements["permutation_mapping_entries_upper"],
            small["constraint_system"]["permutation_successor_entry_count_upper"],
        )
        self.assertEqual(
            small_requirements["permutation_aux_entries_upper"],
            small_requirements["permutation_mapping_entries_upper"],
        )
        self.assertEqual(
            small_requirements["permutation_sizes_entries_upper"],
            small["constraint_system"]["permutation_copy_call_count"],
        )
        self.assertGreater(
            larger_requirements["permutation_hashmap_heap_peak_bytes"],
            small_requirements["permutation_hashmap_heap_peak_bytes"],
        )
        self.assertGreater(
            larger_requirements["permutation_pk_host_phase_bytes"],
            small_requirements["permutation_pk_host_phase_bytes"],
        )
        self.assertGreater(
            larger_requirements["permutation_vk_msm_host_known_bytes"],
            small_requirements["permutation_vk_msm_host_known_bytes"],
        )
        self.assertEqual(
            small_requirements["setup_keygen_host_peak_bytes"],
            small_requirements["srs_loaded_host_bytes"]
            + self.memory["graph_witness_scalar_element_count"]
            * self.memory["size_of_graph_witness_scalar_bytes"]
            + max(
                small_requirements["selector_keygen_host_phase_bytes"],
                small_requirements["permutation_pk_host_phase_bytes"],
                small_requirements["fixed_polynomial_read_heap_scratch_bytes"],
                small_requirements["icicle_msm_host_staging_bytes"],
                small_requirements["permutation_vk_msm_host_known_bytes"],
            ),
        )

        old_schema = copy.deepcopy(self.admission)
        old_schema["schema_version"] = 4
        old_schema["native_memory_model"]["schema"] = "ezkl-halo2-memory-model-v4"
        old_schema["native_memory_model"].pop("size_of_permutation_hashmap_header_bytes")
        with self.assertRaises(ContractError):
            self._run_full_native_admission_parser(old_schema)

    def _run_full_native_admission_parser(self, admission: dict[str, object]) -> dict[str, dict[str, object]]:
        """Exercise the production receipt parser without invoking the native binary."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "resources/resource_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
            manifest_sha256 = formal_runner.sha256(manifest)
            admission["resource_manifest_sha256"] = manifest_sha256
            admission["relation_identity"]["resource_manifest_sha256"] = manifest_sha256

            settings = root / "runs/tiny/.sessions/session/admission/cora/settings.json"
            compiled = settings.with_name("compiled.circuit")
            settings.parent.mkdir(parents=True)
            settings.write_bytes(b"tiny settings fixture\n")
            compiled.write_bytes(b"tiny compiled circuit fixture\n")
            admission["settings_sha256"] = formal_runner.sha256(settings)
            admission["compiled_circuit_sha256"] = formal_runner.sha256(compiled)

            admission_path = settings.with_name("admission.json")
            admission_path.write_text(json.dumps(admission), encoding="utf-8")
            output_root = root / "runs/tiny"
            admission_root = output_root / ".sessions/session/admission"
            receipt = {
                "cora": {
                    "prepared_dir": admission_root.joinpath("cora").relative_to(root).as_posix(),
                    "sha256": formal_runner.sha256(admission_path),
                }
            }
            uuid = admission["native_memory_model"]["cuda_uuid"]
            return formal_runner.native_admit_all_datasets(
                root / "unused-ezkl-binary",
                root / "unused-srs",
                output_root,
                "session",
                {"uuid": uuid},
                datasets=("cora",),
                baseline_root=root,
                reuse_receipt=receipt,
            )

    def test_native_admission_parser_rejects_bad_advice_phases_and_minimum_rows(self) -> None:
        self._run_full_native_admission_parser(copy.deepcopy(self.admission))

        malformed_phase_cases = (
            ("length", lambda phases: phases.pop()),
            ("negative", lambda phases: phases.__setitem__(0, -1)),
            ("boolean", lambda phases: phases.__setitem__(0, True)),
        )
        for label, mutate in malformed_phase_cases:
            with self.subTest(phase_case=label):
                candidate = copy.deepcopy(self.admission)
                mutate(candidate["native_memory_model"]["advice_column_phase_ids"])
                with self.assertRaisesRegex(ContractError, "advice phase layout is malformed"):
                    self._run_full_native_admission_parser(candidate)

        impossible_minimum = copy.deepcopy(self.admission)
        impossible_minimum["constraint_system"]["minimum_rows"] = self.domain_size + 1
        with self.assertRaisesRegex(ContractError, "below the current Halo2 minimum row requirement"):
            self._run_full_native_admission_parser(impossible_minimum)


if __name__ == "__main__":
    unittest.main()
