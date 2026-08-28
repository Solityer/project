"""Lightweight result/resource contract tests; no proof or dataset execution."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from resource_policy import (  # noqa: E402
    CELL_ARTIFACTS,
    CUDA_BACKEND_LIBRARY_PATHS,
    CUDA_BACKEND_MANIFEST,
    CUDA_BACKEND_ROOT,
    ICICLE_CUDA_ABI,
    ICICLE_SOURCE_COMMIT,
    atomic_write_json,
    safe_mkdir,
    manifest_payload,
    validate_cuda_backend_material,
    validate_ceremony_material,
    validate_results_tree,
    validate_cell_inventory,
)
from collect_results import selected_cells  # noqa: E402
from validate_final_results import validate_matrix_records  # noqa: E402
from formal_source_manifest import (  # noqa: E402
    FORMAL_SOURCE_ROOTS,
    GENERATED_SOURCE_DIR_NAMES,
    GENERATED_SOURCE_SUFFIXES,
    SOURCE_FILTER_CONTRACT,
    formal_source_files,
    formal_source_id,
    formal_vendor_files,
)
from preflight import main as preflight_main  # noqa: E402
from prepare_resources import (  # noqa: E402
    _validate_resource_layout_paths,
    main as prepare_resources_main,
)
from build_identity_contract import _cuda_backend_provenance  # noqa: E402
from run_gat_ezkl_baseline import (  # noqa: E402
    PRIVATE_DIRECTORY_MODE,
    PRIVATE_FILE_MODE,
    _create_private_stage_file,
    _harden_private_stage_file,
    _redact_private_stage_paths,
    _remove_private_stage_artifacts,
    build_independent_verify_command,
)


DATASETS = (
    "cora_full",
    "citeseer_full",
    "pubmed_full",
    "ppi_full_formal",
    "ogbn_arxiv_full",
)


def matrix_fixture() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset in DATASETS:
        for device in ("cpu", "cuda"):
            record: dict[str, object] = {
                "fixture_kind": "tests_only_mock_contract_v1",
                "dataset": dataset,
                "device": device,
                "verified_flag": True,
                "verifier_backend": "cpu_independent",
                "gpu_backend_execution": (
                    "cuda_icicle_prove_only" if device == "cuda" else "not_applicable"
                ),
                "cuda_backend_provenance": {"status": "fixture_not_production"},
                "gpu_prove_telemetry": "not_applicable",
                "attestation": {
                    "build_identity_bound": True,
                    "source_bound": True,
                    "source_srs_k": 28,
                    "config_bound": True,
                    "resource_bound": True,
                    "proof_size_bytes": 1,
                    "proof_sha256": "sha256:fixture",
                },
            }
            if device == "cuda":
                record["gpu_prove_telemetry"] = {
                    "schema": "ezkl_cuda_kernel_telemetry_v1",
                    "scope": "formal_prove",
                    "kernel_count": 1,
                    "dropped_records": 0,
                    "active_device_id": 0,
                    "scope_nonce": "fixture",
                    "scope_start_ns": 1,
                    "scope_end_ns": 2,
                    "kernel_names": [{"name": "fixture", "category": "fixture", "count": 1, "device_id": 0}],
                }
            records.append(record)
    return records


def make_vendor_fixture(root: Path, entries: tuple[str, ...] = ("a.txt", "nested/b.txt")) -> None:
    vendor = root / "vendor"
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = vendor / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry.encode("utf-8"))
    (configs / "formal_vendor_files.txt").write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")


def make_cuda_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    backend_root = root / CUDA_BACKEND_ROOT
    library_root = backend_root / "lib" / "backend"
    library_root.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    for relative in CUDA_BACKEND_LIBRARY_PATHS:
        path = root / relative
        payload = (Path(relative).name + " fixture").encode("ascii")
        path.write_bytes(payload)
        inventory.append({
            "path": relative,
            "size": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        })
    manifest = {
        "schema": "ezkl_formal_icicle_cuda_backend_v1",
        "backend": "cuda",
        "icicle_commit": ICICLE_SOURCE_COMMIT,
        "abi": ICICLE_CUDA_ABI,
        "library_inventory": inventory,
    }
    manifest_path = root / CUDA_BACKEND_MANIFEST
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return library_root, manifest


def make_source_fixture(root: Path) -> None:
    """Create the minimum production source roots for identity tests."""

    for relative, payload in {
        "Cargo.lock": "# fixture\n",
        "Cargo.toml": "[package]\nname = \"fixture\"\nversion = \"0.0.0\"\n",
        "build.rs": "fn main() {}\n",
        "rust-toolchain": "1.93.0-nightly\n",
        "src/main.rs": "fn main() {}\n",
        "cpp/gat_ezkl_adapter/CMakeLists.txt": "cmake_minimum_required(VERSION 3.20)\n",
        "cpp/gat_ezkl_adapter/src/main.cpp": "int main() { return 0; }\n",
        "cpp/gat_ezkl_adapter/include/main.hpp": "#pragma once\n",
        "configs/formal_vendor_files.txt": "fixture.txt\n",
        "configs/route.json": "{}\n",
        "scripts/entry.py": "print('fixture')\n",
        "scripts/entry.sh": "#!/bin/sh\n",
        "vendor/fixture.txt": "fixture\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def invoke_main(main, argv: list[str]) -> None:
    previous = sys.argv
    sys.argv = ["fixture"] + argv
    try:
        main()
    finally:
        sys.argv = previous


class ResultContractTests(unittest.TestCase):
    def test_formal_source_filter_contract_is_shared_and_ignores_bytecode_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_source_fixture(root)
            baseline_id = formal_source_id(root)
            (root / "scripts" / "__pycache__").mkdir()
            (root / "scripts" / "__pycache__" / "entry.cpython-312.pyc").write_bytes(b"cache")
            (root / "scripts" / "entry.pyc").write_bytes(b"cache")
            (root / "src" / "ignored.tmp").write_bytes(b"temporary")
            self.assertEqual(formal_source_id(root), baseline_id)
            self.assertTrue(GENERATED_SOURCE_DIR_NAMES.issuperset({"__pycache__", "target", "build"}))
            self.assertTrue({".pyc", ".pyo"}.issubset(GENERATED_SOURCE_SUFFIXES))
            build_rs = (Path(__file__).parents[1] / "build.rs").read_text(encoding="utf-8")
            self.assertIn(
                f'SOURCE_FILTER_CONTRACT: &str = "{SOURCE_FILTER_CONTRACT}"',
                build_rs,
            )
            for name in GENERATED_SOURCE_DIR_NAMES:
                self.assertIn(f'"{name}"', build_rs)
            for suffix in GENERATED_SOURCE_SUFFIXES:
                self.assertIn(f'"{suffix}"', build_rs)
            self.assertEqual(tuple(FORMAL_SOURCE_ROOTS)[-1], "vendor")

    def test_formal_source_filter_rejects_unknown_production_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_source_fixture(root)
            unknown = root / "scripts" / "README.md"
            unknown.write_text("not production source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown formal production source entry"):
                formal_source_files(root)

    def test_preflight_rejects_parent_and_destination_alias_without_external_write(self) -> None:
        baseline = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=baseline / "tests") as inside, tempfile.TemporaryDirectory() as outside:
            inside_root = Path(inside)
            outside_root = Path(outside)
            sentinel = outside_root / "sentinel.txt"
            sentinel.write_text("unchanged", encoding="utf-8")

            parent_alias = inside_root / "parent-alias"
            parent_alias.symlink_to(outside_root, target_is_directory=True)
            with self.assertRaises(SystemExit):
                invoke_main(preflight_main, ["--output", str(parent_alias / "preflight.json"), "--datasets", "cora_full"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

            destination_alias = inside_root / "destination-alias.json"
            destination_alias.symlink_to(sentinel)
            with self.assertRaises(SystemExit):
                invoke_main(preflight_main, ["--output", str(destination_alias), "--datasets", "cora_full"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_prepare_resources_rejects_root_and_resource_parent_aliases_before_write(self) -> None:
        baseline = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=baseline / "tests") as inside, tempfile.TemporaryDirectory() as outside:
            inside_root = Path(inside)
            outside_root = Path(outside)
            sentinel = outside_root / "sentinel.txt"
            sentinel.write_text("unchanged", encoding="utf-8")

            root_alias = inside_root / "root-alias"
            root_alias.symlink_to(baseline, target_is_directory=True)
            with self.assertRaises(SystemExit):
                invoke_main(prepare_resources_main, ["--root", str(root_alias), "--datasets", "cora_full"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

            fixture = inside_root / "fixture-root"
            fixture.mkdir()
            (fixture / "resources").mkdir()
            resource_alias = fixture / "resources" / "ceremony"
            resource_alias.symlink_to(outside_root, target_is_directory=True)
            with self.assertRaises(ValueError):
                _validate_resource_layout_paths(fixture, ("cora_full",))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

            fixture2 = inside_root / "fixture-root-2"
            fixture2.mkdir()
            outside_resources = fixture2 / "resources-target"
            outside_resources.mkdir()
            (fixture2 / "resources").symlink_to(outside_resources, target_is_directory=True)
            with self.assertRaises(ValueError):
                _validate_resource_layout_paths(fixture2, ("cora_full",))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_safe_mkdir_rejects_alias_before_creating_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            external = Path(outside)
            (root / "alias").symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_mkdir(root, root / "alias" / "new", "fixture directory")
            self.assertFalse((external / "new").exists())

    def test_formal_vendor_inventory_is_exact_and_rejects_missing_extra_escape_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_vendor_fixture(root)
            self.assertEqual(
                [path.relative_to(root / "vendor").as_posix() for path in formal_vendor_files(root)],
                ["a.txt", "nested/b.txt"],
            )
            (root / "vendor" / "extra.txt").write_bytes(b"extra")
            with self.assertRaises(ValueError):
                formal_vendor_files(root)
            (root / "vendor" / "extra.txt").unlink()
            (root / "configs" / "formal_vendor_files.txt").write_text("a.txt\nmissing.txt\n", encoding="utf-8")
            with self.assertRaises((FileNotFoundError, ValueError)):
                formal_vendor_files(root)
            (root / "configs" / "formal_vendor_files.txt").write_text("../outside\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                formal_vendor_files(root)
            make_vendor_fixture(root)
            link = root / "vendor" / "nested" / "link.txt"
            link.symlink_to(root / "vendor" / "a.txt")
            (root / "configs" / "formal_vendor_files.txt").write_text(
                "a.txt\nnested/b.txt\nnested/link.txt\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                formal_vendor_files(root)
            link.unlink()
            os.link(root / "vendor" / "a.txt", root / "vendor" / "nested" / "hardlink.txt")
            (root / "configs" / "formal_vendor_files.txt").write_text(
                "a.txt\nnested/b.txt\nnested/hardlink.txt\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                formal_vendor_files(root)

    def test_taskflow_and_backend_loader_are_offline_and_fail_closed(self) -> None:
        cmake_files = (
            Path(__file__).parents[1] / "vendor/icicle/icicle/CMakeLists.txt",
            Path(__file__).parents[1] / "vendor/icicle/icicle/cmake/backend_include.cmake",
            Path(__file__).parents[1] / "vendor/icicle/icicle/backend/cpu/CMakeLists.txt",
        )
        for path in cmake_files:
            self.assertNotIn("FetchContent", path.read_text(encoding="utf-8"))
        runtime = (Path(__file__).parents[1] / "vendor/icicle/rust/icicle-runtime/src/runtime.rs").read_text(encoding="utf-8")
        cpp_runtime = (Path(__file__).parents[1] / "vendor/icicle/icicle/src/runtime.cpp").read_text(encoding="utf-8")
        cpp_loader = (Path(__file__).parents[1] / "vendor/icicle/icicle/src/runtime_internal_loader.cpp").read_text(encoding="utf-8")
        runtime_header = (Path(__file__).parents[1] / "vendor/icicle/icicle/include/icicle/runtime.h").read_text(encoding="utf-8")
        runtime_internal_header = (Path(__file__).parents[1] / "vendor/icicle/icicle/src/runtime_internal.h").read_text(encoding="utf-8")
        for source in (runtime, cpp_runtime, cpp_loader):
            self.assertNotIn("load_backend_from_env_or_default", source)
            self.assertNotIn("/opt/icicle", source)
            self.assertNotIn("LD_LIBRARY_PATH", source)
            self.assertNotIn("is_recursive", source)
            self.assertNotIn("readdir", source)
            self.assertNotIn("pub fn load_backend(", source)
            self.assertNotIn("pub fn load_backend_non_recursive(", source)
        self.assertIn("icicle_load_backend_file_internal", runtime)
        self.assertNotIn("icicle_load_backend_file_internal", cpp_runtime)
        self.assertIn("icicle_load_backend_file_internal", cpp_loader)
        self.assertIn("icicle_load_backend_file_internal", runtime_internal_header)
        self.assertNotIn("icicle_load_backend_file", runtime_header)
        self.assertNotIn("icicle_load_backend(", runtime_header)
        for marker in (
            "deny_unknown_fields",
            "symlink_metadata",
            "exact_directory_entries",
            "regular_file_snapshot",
            "ParentSnapshot",
            "parents_before",
            "parents_after",
            "verify_manifest_snapshot",
            "parents_before != parents_after",
            "icicle_load_backend_file_internal",
        ):
            self.assertIn(marker, runtime)
        self.assertIn("dlopen(file_path", cpp_loader)
        self.assertIn("is_single_link_regular_file", cpp_loader)
        self.assertIn("ICICLE_RUNTIME_INTERNAL_EXPORT", runtime_internal_header)
        self.assertIn('visibility("hidden")', runtime_internal_header)

    def test_fft_boundary_is_compile_time_policy(self) -> None:
        source = (Path(__file__).parents[1] / "vendor/halo2/halo2_proofs/src/icicle.rs").read_text(encoding="utf-8")
        self.assertIn("const ICICLE_SMALL_K_FFT: usize = 2;", source)
        self.assertNotIn("env::var(\"ICICLE_SMALL_K_FFT\")", source)
        self.assertNotIn("use std::{env", source)

    def test_cuda_bundle_requires_exact_four_libraries_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_root, _ = make_cuda_fixture(root)
            material = validate_cuda_backend_material(root)
            self.assertEqual(len(material["library_inventory"]), 4)
            self.assertTrue(material["backend_digest"].startswith("sha256:"))
            first = library_root / Path(CUDA_BACKEND_LIBRARY_PATHS[0]).name
            first.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                validate_cuda_backend_material(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_root, _ = make_cuda_fixture(root)
            (library_root / Path(CUDA_BACKEND_LIBRARY_PATHS[-1]).name).unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                validate_cuda_backend_material(root)
            make_cuda_fixture(root)
            (library_root / "extra.so").write_bytes(b"extra")
            with self.assertRaises(ValueError):
                validate_cuda_backend_material(root)

    def test_cuda_bundle_rejects_manifest_identity_inventory_and_link_tamper(self) -> None:
        def rejected(mutator) -> None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                make_cuda_fixture(root)
                manifest_path = root / CUDA_BACKEND_MANIFEST
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutator(payload)
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_cuda_backend_material(root)

        rejected(lambda payload: payload.update({"unexpected": True}))
        rejected(lambda payload: payload.update({"icicle_commit": "wrong-commit"}))
        rejected(lambda payload: payload.update({"abi": "wrong-abi"}))
        rejected(lambda payload: payload["library_inventory"].__setitem__(0, {"path": "wrong", "size": 1, "sha256": "sha256:" + "0" * 64}))
        rejected(lambda payload: payload["library_inventory"].__setitem__(0, {"path": payload["library_inventory"][0]["path"], "size": 1, "sha256": "sha256:" + "0" * 64}))
        rejected(lambda payload: payload["library_inventory"].append(payload["library_inventory"][0]))
        rejected(lambda payload: payload["library_inventory"][0].update({"sha256": "SHA256:" + "0" * 64}))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_cuda_fixture(root)
            backend_root = root / CUDA_BACKEND_ROOT
            real_backend = root / "backend-outside-layout"
            backend_root.rename(real_backend)
            backend_root.symlink_to(real_backend, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_cuda_backend_material(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_root, _ = make_cuda_fixture(root)
            source = library_root / Path(CUDA_BACKEND_LIBRARY_PATHS[0]).name
            hardlink = root / "hardlink-outside-layout.so"
            os.link(source, hardlink)
            with self.assertRaises(ValueError):
                validate_cuda_backend_material(root)

    def test_cpu_identity_does_not_require_cuda_bundle(self) -> None:
        self.assertEqual(_cuda_backend_provenance(Path("/unused"), "cpu"), {"status": "not_applicable"})

    def test_collector_selection_marks_partial_matrix_without_relaxing_cell_identity(self) -> None:
        selected = selected_cells(("cora_full", "pubmed_full"), ("cpu",))
        self.assertEqual(selected, {("cora_full", "cpu"), ("pubmed_full", "cpu")})
        self.assertNotEqual(selected, selected_cells())

    def test_final_validator_fixture_covers_complete_matrix_contract(self) -> None:
        self.assertEqual(validate_matrix_records(matrix_fixture(), fixture=True), [])

    def test_final_validator_fixture_rejects_missing_unknown_duplicate_and_tamper(self) -> None:
        records = matrix_fixture()
        errors = validate_matrix_records(records[:-1], fixture=True)
        self.assertTrue(any("missing required matrix cell" in item for item in errors))
        unknown = matrix_fixture() + [{"fixture_kind": "tests_only_mock_contract_v1", "dataset": "unknown", "device": "cpu"}]
        self.assertTrue(any("unknown matrix cell" in item for item in validate_matrix_records(unknown, fixture=True)))
        duplicate = matrix_fixture() + [matrix_fixture()[0]]
        self.assertTrue(any("duplicate matrix cell" in item for item in validate_matrix_records(duplicate, fixture=True)))
        tampered = matrix_fixture()
        tampered[0]["attestation"]["source_srs_k"] = 27  # type: ignore[index]
        self.assertTrue(any("source SRS k mismatch" in item for item in validate_matrix_records(tampered, fixture=True)))

    def test_final_validator_fixture_rejects_verified_flag_and_cpu_gpu_tamper(self) -> None:
        records = matrix_fixture()
        records[0]["verified_flag"] = False
        records[0]["gpu_prove_telemetry"] = {"unexpected": True}
        self.assertTrue(any("not verified" in item for item in validate_matrix_records(records, fixture=True)))
        self.assertTrue(any("CPU fixture carries GPU telemetry" in item for item in validate_matrix_records(records, fixture=True)))
        records = matrix_fixture()
        records[-1]["gpu_prove_telemetry"]["kernel_count"] = 0  # type: ignore[index]
        self.assertTrue(any("telemetry/verifier contract mismatch" in item for item in validate_matrix_records(records, fixture=True)))

    def test_final_validator_fixture_rejects_build_source_config_resource_size_digest_tamper(self) -> None:
        fields = (
            ("build_identity_bound", "build identity"),
            ("source_bound", "source binding"),
            ("config_bound", "config binding"),
            ("resource_bound", "resource binding"),
        )
        for field, marker in fields:
            records = matrix_fixture()
            records[0]["attestation"][field] = False  # type: ignore[index]
            self.assertTrue(any(marker in item for item in validate_matrix_records(records, fixture=True)))
        for field, expected_marker in (("proof_size_bytes", "proof size"), ("proof_sha256", "proof digest")):
            records = matrix_fixture()
            records[0]["attestation"][field] = 0 if field == "proof_size_bytes" else "tampered"  # type: ignore[index]
            self.assertTrue(any(expected_marker in item for item in validate_matrix_records(records, fixture=True)))

    def test_exact_inventory_accepts_fixture_without_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            cell.mkdir()
            for name in CELL_ARTIFACTS:
                (cell / name).write_bytes(b"fixture")
            validate_cell_inventory(cell)

    def test_inventory_rejects_extra_directory_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            cell.mkdir()
            for name in CELL_ARTIFACTS:
                (cell / name).write_bytes(b"fixture")
            (cell / "unexpected").mkdir()
            with self.assertRaises(ValueError):
                validate_cell_inventory(cell)
            (cell / "unexpected").rmdir()
            (cell / "unexpected").symlink_to(cell / "run_manifest.json")
            with self.assertRaises(ValueError):
                validate_cell_inventory(cell)

    def test_inventory_rejects_private_model_and_witness_artifacts(self) -> None:
        """A committed cell must never contain private prover inputs."""

        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            cell.mkdir()
            for name in CELL_ARTIFACTS:
                (cell / name).write_bytes(b"fixture")
            for private_name in ("model_parameters.i64.txt", "witness.json"):
                (cell / private_name).write_bytes(b"private fixture")
                with self.assertRaises(ValueError):
                    validate_cell_inventory(cell)
                (cell / private_name).unlink()

    def test_private_stage_directory_and_file_modes_are_hardened(self) -> None:
        """Temporary witness/model material is private and single-link."""

        baseline = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=baseline / "tests") as temporary:
            stage = Path(temporary) / "stage"
            stage.mkdir(mode=PRIVATE_DIRECTORY_MODE)
            cell = stage / "cell"
            cell.mkdir(mode=PRIVATE_DIRECTORY_MODE)
            self.assertEqual(stage.stat().st_mode & 0o777, PRIVATE_DIRECTORY_MODE)
            self.assertEqual(cell.stat().st_mode & 0o777, PRIVATE_DIRECTORY_MODE)

            model = _create_private_stage_file(
                cell / "model_parameters.i64.txt", stage, "model parameters"
            )
            witness = _create_private_stage_file(
                cell / "witness.json", stage, "formal witness"
            )
            self.assertEqual(model.stat().st_mode & 0o777, PRIVATE_FILE_MODE)
            self.assertEqual(witness.stat().st_mode & 0o777, PRIVATE_FILE_MODE)

            # Even if a native command changes the mode, the hardening step
            # restores the private-file contract before cleanup.
            model.chmod(0o644)
            _harden_private_stage_file(model, stage, "model parameters")
            self.assertEqual(model.stat().st_mode & 0o777, PRIVATE_FILE_MODE)

    def test_verify_argv_has_only_public_inputs_and_no_private_paths(self) -> None:
        verifier = Path("/baseline/target/release/ezkl_formal")
        private_root = Path("/baseline/results/runs_tmp/private-session")
        command = build_independent_verify_command(
            verifier,
            compiled=Path("/baseline/results/runs/cpu/cora_full/network.compiled"),
            proof=Path("/baseline/results/runs/cpu/cora_full/proof.json"),
            statement=Path("/baseline/results/runs/cpu/cora_full/public_statement.json"),
            settings=Path("/baseline/results/runs/cpu/cora_full/settings.json"),
            vk=Path("/baseline/results/runs/cpu/cora_full/vk.key"),
            output=Path("/baseline/results/runs/cpu/cora_full/verify.json"),
        )
        rendered = "\x00".join(command).lower()
        self.assertNotIn("witness", rendered)
        self.assertNotIn("model", rendered)
        self.assertNotIn("checkpoint", rendered)
        self.assertNotIn(str(private_root).lower(), rendered)
        self.assertEqual(
            set(command) & {"--witness", "--model", "--checkpoint"}, set()
        )

    def test_manifest_redacts_private_stage_paths_recursively(self) -> None:
        stage = "/baseline/results/runs_tmp/private-session"
        model = f"{stage}/cell/model_parameters.i64.txt"
        witness = f"{stage}/cell/witness.json"
        payload = {
            "commands": [["prove", "--witness", witness]],
            "logs": {"model": model},
            "nested": [model, witness],
        }
        redacted = _redact_private_stage_paths(payload, {model, witness})
        encoded = json.dumps(redacted, sort_keys=True)
        self.assertNotIn(stage, encoded)
        self.assertNotIn(model, encoded)
        self.assertNotIn(witness, encoded)
        self.assertIn("<private-stage-artifact>", encoded)

    def test_private_artifacts_are_removed_before_public_cell_publish(self) -> None:
        baseline = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=baseline / "tests") as temporary:
            stage = Path(temporary) / "stage"
            stage.mkdir(mode=PRIVATE_DIRECTORY_MODE)
            cell = stage / "cell"
            cell.mkdir(mode=PRIVATE_DIRECTORY_MODE)
            for name in CELL_ARTIFACTS:
                (cell / name).write_bytes(b"fixture")
            model = _create_private_stage_file(
                cell / "model_parameters.i64.txt", stage, "model parameters"
            )
            witness = _create_private_stage_file(
                cell / "witness.json", stage, "formal witness"
            )
            self.assertTrue(model.exists())
            self.assertTrue(witness.exists())
            _remove_private_stage_artifacts(cell, stage)
            self.assertFalse(model.exists())
            self.assertFalse(witness.exists())
            validate_cell_inventory(cell)

    def test_results_tree_rejects_unknown_cells_and_extra_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "results" / "runs" / "cpu" / "cora_full"
            runs.mkdir(parents=True)
            for name in CELL_ARTIFACTS:
                (runs / name).write_bytes(b"fixture")
            self.assertEqual(validate_results_tree(root), {("cora_full", "cpu")})
            (root / "results" / "runs" / "cuda").mkdir()
            (root / "results" / "runs" / "cuda" / "unexpected").mkdir()
            with self.assertRaises(ValueError):
                validate_results_tree(root)

    def test_results_tree_rejects_links_and_special_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = root / "results" / "runs" / "cpu" / "cora_full"
            cell.mkdir(parents=True)
            for name in CELL_ARTIFACTS:
                (cell / name).write_bytes(b"fixture")
            (root / "results" / "runs" / "cpu" / "citeseer_full").symlink_to(cell)
            with self.assertRaises(ValueError):
                validate_results_tree(root)

    def test_atomic_json_rejects_symlink_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("original\n", encoding="utf-8")
            link = root / "result.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                atomic_write_json(link, {"status": "bad"})
            self.assertEqual(target.read_text(encoding="utf-8"), "original\n")

    def test_ceremony_contract_requires_canonical_digest_and_k28(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ceremony = root / "resources" / "ceremony"
            ceremony.mkdir(parents=True)
            srs = ceremony / "source.srs"
            srs.write_bytes(b"fixture-srs")
            import hashlib

            digest = "sha256:" + hashlib.sha256(srs.read_bytes()).hexdigest()
            descriptor = ceremony / "ceremony.descriptor.json"
            descriptor.write_text(json.dumps({
                "descriptor_kind": "ezkl_trusted_ceremony_srs_v1",
                "ceremony_id": "fixture-ceremony",
                "source_srs_sha256": digest,
                "source_srs_max_k": 28,
            }), encoding="utf-8")
            material = validate_ceremony_material(root)
            self.assertEqual(material["source_srs_sha256"], digest)
            descriptor.write_text(descriptor.read_text(encoding="utf-8").replace('"source_srs_max_k": 28', '"source_srs_max_k": 27'), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_ceremony_material(root)

    def test_manifest_is_baseline_local_and_complete(self) -> None:
        baseline = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=baseline / "tests") as temporary:
            root = Path(temporary) / "resource-root"
            root.mkdir()
            # The hardened lexical path checker intentionally requires every
            # existing parent component before it permits a missing leaf.
            # Build only the baseline-local fixture roots; no production
            # resources or result cells are created by this test.
            for relative in (
                "configs",
                "data/cache",
                "artifacts/checkpoints",
                "resources/public_y",
            ):
                (root / relative).mkdir(parents=True, exist_ok=True)
            payload = manifest_payload(root)
        self.assertEqual(payload["schema"], "ezkl_formal_resource_manifest_v1")
        self.assertEqual(len(payload["datasets"]), 5)
        self.assertEqual(payload["output_policy"]["cell_root"], "results/runs/<backend>/<dataset>")


if __name__ == "__main__":
    unittest.main()
