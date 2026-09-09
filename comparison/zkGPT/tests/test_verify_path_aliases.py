from __future__ import annotations

import os
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from verify_proof import _canonical_public_y  # noqa: E402


class VerifyPathAliasTests(unittest.TestCase):
    def test_symlinked_config_is_rejected_before_public_y_resolution(self) -> None:
        target = ROOT / "configs/formal/cora_full.cfg"
        alias = ROOT / "configs/formal/.test-config-alias"
        if os.path.lexists(alias):
            self.fail(f"test alias already exists: {alias}")
        os.symlink(target, alias)
        try:
            with self.assertRaises((OSError, ValueError)):
                _canonical_public_y(ROOT, alias, ROOT / "resources/public_y/canonical/cora.json")
        finally:
            alias.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
