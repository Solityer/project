from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from zkgpt_gat.config import generated_config, write_config
from zkgpt_gat.util import ensure_dir, write_json, write_text_atomic


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_script(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SecureWriteTests(unittest.TestCase):
    def test_atomic_writer_rejects_symlink_parent_and_preserves_external(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-safe-write-") as name:
            root = pathlib.Path(name)
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                sentinel = outside / "sentinel.json"
                sentinel.write_text("keep\n", encoding="utf-8")
                (root / "results").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(OSError, "symlink"):
                    write_json(root / "results" / "result.json", {"status": "bad"}, root=root)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
                self.assertFalse((outside / "result.json").exists())
            finally:
                shutil.rmtree(outside)

    def test_atomic_writer_rejects_file_symlink_hardlink_fifo_and_publishes_atomically(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-safe-write-") as name:
            root = pathlib.Path(name)
            ensure_dir(root / "results", root=root)
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                outside_file = outside / "outside.json"
                outside_file.write_text("outside\n", encoding="utf-8")

                symlink = root / "results" / "symlink.json"
                symlink.symlink_to(outside_file)
                with self.assertRaisesRegex(OSError, "symlink"):
                    write_json(symlink, {"status": "bad"}, root=root)
                self.assertEqual(outside_file.read_text(encoding="utf-8"), "outside\n")

                hardlink = root / "results" / "hardlink.json"
                os.link(outside_file, hardlink)
                with self.assertRaisesRegex(OSError, "hardlink"):
                    write_text_atomic(hardlink, "bad\n", root=root)
                self.assertEqual(outside_file.read_text(encoding="utf-8"), "outside\n")

                if hasattr(os, "mkfifo"):
                    fifo = root / "results" / "fifo.json"
                    os.mkfifo(fifo)
                    with self.assertRaisesRegex(OSError, "regular file"):
                        write_json(fifo, {"status": "bad"}, root=root)

                output = root / "results" / "atomic.json"
                write_json(output, {"status": "ok"}, root=root)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"status": "ok"})
                self.assertEqual(output.stat().st_nlink, 1)
            finally:
                shutil.rmtree(outside)

    def test_prepare_resources_rejects_preseeded_aliases_without_external_write(self) -> None:
        script = ROOT / "scripts" / "prepare_resources.py"
        module = _load_script("zkgpt_prepare_resources_security", script)
        descriptor = ROOT / "resources" / "static_tables" / "exact_integer_gat.json"
        for kind in ("directory", "manifest"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="zkgpt-prepare-safe-") as name:
                root = pathlib.Path(name)
                static_dir = root / "resources" / "static_tables"
                static_dir.mkdir(parents=True)
                shutil.copyfile(descriptor, static_dir / descriptor.name)
                outside = root.parent / f"{root.name}-outside"
                outside.mkdir()
                try:
                    sentinel = outside / "sentinel"
                    sentinel.write_text("keep\n", encoding="utf-8")
                    if kind == "directory":
                        shutil.rmtree(root / "resources")
                        (root / "resources").symlink_to(outside, target_is_directory=True)
                    else:
                        manifest = root / "resources" / "resource_manifest.json"
                        manifest.symlink_to(sentinel)
                    with mock.patch.object(module, "ROOT", root), mock.patch.object(
                        sys, "argv", [str(script), "--root", str(root)]
                    ):
                        with self.assertRaises((OSError, SystemExit)):
                            module.main()
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
                    self.assertFalse((outside / "resource_manifest.json").exists())
                finally:
                    shutil.rmtree(outside)

    def test_generated_config_rejects_alias_parent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-generated-config-safe-") as name:
            root = pathlib.Path(name)
            formal = root / "configs" / "formal"
            formal.mkdir(parents=True)
            shutil.copyfile(ROOT / "configs" / "formal" / "cora_full.cfg", formal / "cora_full.cfg")
            static_dir = root / "resources" / "static_tables"
            static_dir.mkdir(parents=True)
            shutil.copyfile(
                ROOT / "resources" / "static_tables" / "exact_integer_gat.json",
                static_dir / "exact_integer_gat.json",
            )
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                sentinel = outside / "sentinel"
                sentinel.write_text("keep\n", encoding="utf-8")
                (root / "configs" / "generated").mkdir()
                (root / "configs" / "generated" / "cpu").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(OSError, "symlink"):
                    generated_config("cora", "cpu", root, root)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            finally:
                shutil.rmtree(outside)

    def test_write_config_rejects_hardlink_and_writes_owned_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zkgpt-config-safe-") as name:
            root = pathlib.Path(name)
            ensure_dir(root / "configs" / "generated" / "cpu", root=root)
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                source = outside / "config.cfg"
                source.write_text("outside\n", encoding="utf-8")
                hardlink = root / "configs" / "generated" / "cpu" / "hardlink.cfg"
                os.link(source, hardlink)
                with self.assertRaisesRegex(OSError, "hardlink"):
                    write_config(hardlink, {"dataset": "cora"}, root=root)
                self.assertEqual(source.read_text(encoding="utf-8"), "outside\n")

                output = root / "configs" / "generated" / "cpu" / "cora.cfg"
                write_config(output, {"dataset": "cora"}, root=root)
                self.assertIn("dataset = cora\n", output.read_text(encoding="utf-8"))
                self.assertEqual(output.stat().st_nlink, 1)
            finally:
                shutil.rmtree(outside)

    def test_formal_preflight_rejects_output_alias_before_gate(self) -> None:
        script = ROOT / "scripts" / "formal_preflight.py"
        module = _load_script("zkgpt_formal_preflight_security", script)
        with tempfile.TemporaryDirectory(prefix="zkgpt-preflight-safe-") as name:
            root = pathlib.Path(name)
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                sentinel = outside / "sentinel"
                sentinel.write_text("keep\n", encoding="utf-8")
                output = root / "results.json"
                output.symlink_to(sentinel)
                with mock.patch.object(module, "ROOT", root), mock.patch.object(
                    sys, "argv", [str(script), "--output", str(output), "--source-only"]
                ), mock.patch.object(
                    module, "formal_gate_status", side_effect=AssertionError("gate must not run")
                ):
                    with self.assertRaises(SystemExit):
                        module.main()
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            finally:
                shutil.rmtree(outside)


if __name__ == "__main__":
    unittest.main()
