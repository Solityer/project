from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import zkgpt_gat.registry as registry_module
from zkgpt_gat.registry import canonical_registry_public_path, canonical_registry_secret_path, ensure_model_registry


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_registration_script():
    script = ROOT / "scripts" / "register_model.py"
    spec = importlib.util.spec_from_file_location("zkgpt_registration_script_fixture", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load registration script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegistrationPreflightTests(unittest.TestCase):
    @staticmethod
    def _config(*, outside: pathlib.Path | None = None):
        root = ROOT
        outside = outside or root
        return SimpleNamespace(
            entries={
                "model_arch_id": "tests-registration-arch-v1",
                "model_param_id": "tests-registration-model-v1",
                "static_table_id": "tables:lrelu+elu+exp+range",
                "static_tables_id": "tables:lrelu+elu+exp+range",
                "quant_cfg_id": "exact_integer_gat_v1",
                "resource_manifest": "resources/resource_manifest.json",
            },
            rounding_mode="nearest_ties_away",
            hidden_profile=[SimpleNamespace(head_count=1, head_dim=2)],
            k_out=1,
            num_classes=3,
            data_root=outside / "data",
            cache_root=root / "data" / "cache",
            cache_dir=root / "data" / "cache" / "cora",
            checkpoint_bundle=root / "artifacts" / "checkpoints" / "cora_gat",
            static_table_path=root / "resources" / "static_tables" / "exact_integer_gat_v1.json",
            public_y_path=root / "resources" / "public_y" / "cora.json",
        )

    def test_missing_capability_fails_before_resource_or_registry_access(self) -> None:
        module = _load_registration_script()
        order: list[str] = []

        def blocked_gate() -> None:
            order.append("capability")
            raise RuntimeError("capability gate blocked")

        def unexpected_resource_check(*_args, **_kwargs):
            order.append("resources")
            raise AssertionError("resource check ran after blocked capability")

        with mock.patch.object(module, "require_formal_experiment", blocked_gate), mock.patch.object(
            module, "check_resource_layout", unexpected_resource_check
        ):
            with self.assertRaisesRegex(RuntimeError, "capability gate blocked"):
                module.registration_preflight("cora")
        self.assertEqual(order, ["capability"])

    def test_missing_staged_resource_fails_before_checkpoint_or_registry_access(self) -> None:
        module = _load_registration_script()
        calls: list[str] = []

        def staged_check(*_args, **_kwargs):
            calls.append("resource")
            raise FileNotFoundError("fixture resources are not staged")

        def unexpected_config(*_args, **_kwargs):
            calls.append("config")
            raise AssertionError("checkpoint/config access followed a resource failure")

        with mock.patch.object(module, "require_formal_experiment", lambda: calls.append("capability")), mock.patch.object(
            module, "check_resource_layout", staged_check
        ), mock.patch.object(module, "generated_config", unexpected_config):
            with self.assertRaisesRegex(FileNotFoundError, "not staged"):
                module.registration_preflight("cora")
        self.assertEqual(calls, ["capability", "resource"])

    def test_checkpoint_model_identity_mismatch_fails_closed(self) -> None:
        module = _load_registration_script()
        config = self._config()
        descriptor = {
            "static_tables_id": "tables:lrelu+elu+exp+range",
            "quant_cfg_id": "exact_integer_gat_v1",
        }
        manifest = {
            "model_arch_id": config.entries["model_arch_id"],
            "model_param_id": "wrong-model-id",
            "static_table_id": config.entries["static_table_id"],
            "rounding_mode": config.rounding_mode,
            "hidden_layers": [{"layer_index": 0, "head_count": 1, "head_dim": 2}],
            "K_out": 1,
            "C": 3,
        }
        with self.assertRaisesRegex(ValueError, "model_param_id"):
            module._validate_registration_identity(config, manifest, descriptor)

    def test_registration_path_escape_is_rejected(self) -> None:
        module = _load_registration_script()
        config = self._config(outside=ROOT.parent / "outside-registration")
        with self.assertRaisesRegex(ValueError, "data_root escapes baseline root"):
            module._validate_registration_paths(ROOT / "configs" / "generated" / "cpu" / "cora_full.cfg", config)

    def test_registry_write_failure_leaves_no_public_or_private_pair(self) -> None:
        metadata = {
            "model_arch_id": "tests-registration-arch-v1",
            "model_param_id": "tests-registration-model-v1",
            "quant_cfg_id": "exact_integer_gat_v1",
            "static_table_id": "tables:lrelu+elu+exp+range",
            "rounding_mode": "nearest_ties_away",
            "private_input_layout": "X_pub|W|Aux|L_qry",
            "segment_order": ["X_pub", "W", "Aux", "L_qry"],
            "padding_rule_id": "power_of_two_zero_extension_v1",
            "integer_bound_ledger_digest": "a" * 64,
            "Layout_in_digest": "b" * 64,
            "Layout_in": {"N_in": 2},
            "integer_bound_ledger": {"registered_model_bounds": {"hidden.l1.h0": {"B_W": 1}}},
            "commitment_backend_id": "zkGPT-hyrax-mle-evaluation-commitment-v2",
        }
        with tempfile.TemporaryDirectory(prefix="zkgpt-registration-no-partial-") as temporary:
            root = pathlib.Path(temporary)
            original_write = registry_module.write_json

            def fail_secret(path, payload):
                if pathlib.Path(path).name == "registry.secret.json":
                    raise OSError("fixture secret write failure")
                return original_write(path, payload)

            with mock.patch.object(registry_module, "write_json", fail_secret):
                with self.assertRaisesRegex(OSError, "secret write failure"):
                    ensure_model_registry(root, metadata, [("com_W", [1, 2])])
            self.assertFalse(canonical_registry_public_path(root, metadata["model_param_id"]).exists())
            self.assertFalse(canonical_registry_secret_path(root, metadata["model_param_id"]).exists())
            public_parent = root / "results" / "model_registry"
            private_parent = root / "artifacts" / "private_setup" / "model_registry"
            self.assertFalse(list(public_parent.glob(".model-registry-*")) if public_parent.exists() else [])
            self.assertFalse(list(private_parent.glob(".model-registry-*")) if private_parent.exists() else [])


if __name__ == "__main__":
    unittest.main()
