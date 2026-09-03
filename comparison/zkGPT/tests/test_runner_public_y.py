from __future__ import annotations

import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from zkgpt_gat.runner import _resolve_public_y


class RunnerPublicYTests(unittest.TestCase):
    @staticmethod
    def _config(path: str) -> SimpleNamespace:
        return SimpleNamespace(entries={"public_y_path": path})

    def _root(self) -> tuple[tempfile.TemporaryDirectory[str], pathlib.Path, pathlib.Path]:
        holder = tempfile.TemporaryDirectory(prefix="zkgpt-runner-public-y-")
        root = pathlib.Path(holder.name)
        canonical_dir = root / "resources" / "public_y" / "canonical"
        canonical_dir.mkdir(parents=True)
        return holder, root, canonical_dir / "cora.json"

    def test_canonical_public_y_is_accepted(self) -> None:
        holder, root, path = self._root()
        with holder:
            path.write_text("{}\n", encoding="utf-8")
            resolved = _resolve_public_y(
                self._config("resources/public_y/canonical/cora.json"), root, "cora"
            )
            self.assertEqual(resolved, path)

    def test_root_level_public_y_is_rejected(self) -> None:
        holder, root, path = self._root()
        with holder:
            path.write_text("{}\n", encoding="utf-8")
            legacy = root / "resources" / "public_y" / "cora.json"
            legacy.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical resource path"):
                _resolve_public_y(self._config("resources/public_y/cora.json"), root, "cora")

    def test_public_y_symlink_is_rejected(self) -> None:
        holder, root, path = self._root()
        with holder:
            target = root / "outside.json"
            target.write_text("{}\n", encoding="utf-8")
            path.symlink_to(target)
            with self.assertRaisesRegex((OSError, ValueError), "symlink"):
                _resolve_public_y(
                    self._config("resources/public_y/canonical/cora.json"), root, "cora"
                )

    def test_public_y_nonregular_file_is_rejected(self) -> None:
        holder, root, path = self._root()
        with holder:
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "owned regular file"):
                _resolve_public_y(
                    self._config("resources/public_y/canonical/cora.json"), root, "cora"
                )


if __name__ == "__main__":
    unittest.main()
