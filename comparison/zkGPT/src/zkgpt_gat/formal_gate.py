from __future__ import annotations

import json
import pathlib
from typing import Any, Dict


def baseline_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def capabilities() -> Dict[str, Any]:
    path = baseline_root() / "configs" / "formal_capabilities.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "zkGPT_gat_formal_capabilities_v1":
        raise RuntimeError("invalid zkGPT formal capability manifest")
    return value


def require_formal_experiment() -> None:
    value = capabilities()
    if (
        value.get("formal_experiment_eligible") is not True
        or value.get("implemented_reference_code") != "formal_zkgpt_gat_native_v1"
        or value.get("blockers")
    ):
        ids = ", ".join(item.get("id", "unknown") for item in value.get("blockers", []))
        raise RuntimeError(f"zkGPT formal capability manifest is inconsistent: {ids or 'manifest fields'}")
