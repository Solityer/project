"""Session-record regression checks for the EZKL internal phase API."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_hybrid_formal import (  # noqa: E402
    _INTERNAL_SESSION_FIELDS,
    _load_internal_session,
    update_session_record,
)


class InternalSessionSchemaTest(unittest.TestCase):
    def test_running_stage_update_preserves_nullable_error_field(self) -> None:
        session_id = "0123456789abcdef0123456789abcdef"
        document = {key: None for key in _INTERNAL_SESSION_FIELDS}
        document.update({
            "schema_version": 1,
            "baseline": "ezkl",
            "root_session_id": session_id,
            "session_id": session_id,
            "status": "PREPARING",
            "stage": "session_created",
            "error": None,
        })
        with tempfile.TemporaryDirectory(prefix="ezkl-session-schema-") as temp:
            path = Path(temp) / "session.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            update_session_record(path, status="RUNNING", stage="native_build_identity")

            loaded = _load_internal_session(path, session_id)
            self.assertEqual(loaded["status"], "RUNNING")
            self.assertEqual(loaded["stage"], "native_build_identity")
            self.assertIsNone(loaded["error"])


if __name__ == "__main__":
    unittest.main()
