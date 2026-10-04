"""Bounded fixtures for exact-table archive validation edge cases."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_resources import TABLE_FORMAT_VERSION, TABLE_MAGIC, validate_table_archive  # noqa: E402


def _hash_string(hasher: object, value: str) -> None:
    encoded = value.encode()
    hasher.update(struct.pack("<Q", len(encoded)))  # type: ignore[attr-defined]
    hasher.update(encoded)  # type: ignore[attr-defined]


def _hash_i128(hasher: object, value: int) -> None:
    hasher.update(value.to_bytes(16, "little", signed=True))  # type: ignore[attr-defined]


def _canonical_length(table: dict[str, object]) -> int:
    return (
        len(b"ezkl-gat-exact-table-v1")
        + sum(8 + len(str(table[key]).encode()) for key in ("table_id", "function_id", "domain_id", "generator_id"))
        + 4 + 4 + 16 + 16 + int(table["row_count"]) * 32  # type: ignore[arg-type]
    )


def _archive() -> tuple[bytes, list[tuple[int, int]]]:
    specs = (
        ("lrelu", -2, 1),
        ("exp_neg", 0, 0),
        ("elu", (1 << 63) - 1, (1 << 63) - 1),
        ("max_relu", -(1 << 63), -(1 << 63)),
    )
    tables: list[dict[str, object]] = []
    rows_by_table: list[list[tuple[int, int]]] = []
    for function_id, input_min, input_max in specs:
        rows = [(value, value * 3) for value in range(input_min, input_max + 1)]
        table: dict[str, object] = {
            "table_id": f"tiny-{function_id}",
            "input_scale": 8,
            "output_scale": 8,
            "function_id": function_id,
            "domain_id": "tiny-signed-i64",
            "generator_id": "fixture-v1",
            "input_min": input_min,
            "input_max": input_max,
            "row_count": len(rows),
        }
        table_hasher = hashlib.sha256(b"ezkl-gat-exact-table-v1")
        for key in ("table_id",):
            _hash_string(table_hasher, str(table[key]))
        table_hasher.update(struct.pack("<i", int(table["input_scale"])))
        table_hasher.update(struct.pack("<i", int(table["output_scale"])))
        for key in ("function_id", "domain_id", "generator_id"):
            _hash_string(table_hasher, str(table[key]))
        _hash_i128(table_hasher, input_min)
        _hash_i128(table_hasher, input_max)
        for input_value, output_value in rows:
            _hash_i128(table_hasher, input_value)
            _hash_i128(table_hasher, output_value)
        table["digest"] = table_hasher.hexdigest()
        table["canonical_len"] = _canonical_length(table)
        tables.append(table)
        rows_by_table.append(rows)

    static_hasher = hashlib.sha256(b"ezkl-gat-static-tables-v1")
    _hash_string(static_hasher, "tiny-static-tables")
    _hash_string(static_hasher, "tiny-backend-aux")
    for table, rows in zip(tables, rows_by_table):
        static_hasher.update(struct.pack("<Q", int(table["canonical_len"])))
        static_hasher.update(b"ezkl-gat-exact-table-v1")
        _hash_string(static_hasher, str(table["table_id"]))
        static_hasher.update(struct.pack("<i", int(table["input_scale"])))
        static_hasher.update(struct.pack("<i", int(table["output_scale"])))
        for key in ("function_id", "domain_id", "generator_id"):
            _hash_string(static_hasher, str(table[key]))
        _hash_i128(static_hasher, int(table["input_min"]))
        _hash_i128(static_hasher, int(table["input_max"]))
        for input_value, output_value in rows:
            _hash_i128(static_hasher, input_value)
            _hash_i128(static_hasher, output_value)
        _hash_string(static_hasher, str(table["digest"]))

    output = bytearray(TABLE_MAGIC + struct.pack("<II", TABLE_FORMAT_VERSION, 4))
    registry = {
        "format": "ezkl-gat-static-tables-bin",
        "version": TABLE_FORMAT_VERSION,
        "static_tables_id": "tiny-static-tables",
        "backend_aux_tables_id": "tiny-backend-aux",
        "static_tables_digest": static_hasher.hexdigest(),
        "table_order": ["lrelu", "exp", "elu", "max_relu"],
    }
    registry_bytes = json.dumps(registry, separators=(",", ":"), sort_keys=True).encode()
    output.extend(struct.pack("<Q", len(registry_bytes)))
    output.extend(registry_bytes)
    row_spans: list[tuple[int, int]] = []
    for table, rows in zip(tables, rows_by_table):
        metadata = json.dumps(table, separators=(",", ":"), sort_keys=True).encode()
        output.extend(struct.pack("<Q", len(metadata)))
        output.extend(metadata)
        output.extend(struct.pack("<Q", len(rows)))
        row_start = len(output)
        for input_value, output_value in rows:
            output.extend(input_value.to_bytes(16, "little", signed=True))
            output.extend(output_value.to_bytes(16, "little", signed=True))
        row_spans.append((row_start, len(rows)))
    return bytes(output), row_spans


class ValidateTableArchiveTest(unittest.TestCase):
    def check_archive(self, payload: bytes, expected_error: str | None = None) -> None:
        with tempfile.TemporaryDirectory(prefix="ezkl-table-fixture-") as temp_dir:
            path = Path(temp_dir) / "gat_static_tables.bin"
            path.write_bytes(payload)
            if expected_error is None:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(validate_table_archive(path), hashlib.sha256(payload).hexdigest())
            else:
                stderr = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        validate_table_archive(path)
                self.assertIn(expected_error, stderr.getvalue())

    def test_signed_extension_and_int64_endpoints(self) -> None:
        payload, _ = _archive()
        self.check_archive(payload)

    def test_rejects_noncontiguous_key(self) -> None:
        payload, spans = _archive()
        damaged = bytearray(payload)
        row_start, _ = spans[0]
        damaged[row_start] ^= 1
        self.check_archive(bytes(damaged), "table row input is not contiguous")

    def test_rejects_raw_table_digest_mismatch(self) -> None:
        payload, spans = _archive()
        damaged = bytearray(payload)
        row_start, _ = spans[0]
        damaged[row_start + 16] ^= 1
        self.check_archive(bytes(damaged), "table row digest mismatch")

    def test_rejects_truncation_and_trailing_bytes(self) -> None:
        payload, _ = _archive()
        self.check_archive(payload[:-1], "truncated table rows")
        self.check_archive(payload + b"\x00", "trailing bytes")


if __name__ == "__main__":
    unittest.main()
