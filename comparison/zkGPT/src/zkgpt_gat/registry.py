"""Long-lived zkGPT model registration for the native Hyrax path.

Model parameters are committed once per model/layout reuse key.  Inference
proofs load only the public commitment descriptors and the local setup secret
blindings; every opening still samples fresh masking and IPA randomness.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, Sequence, Tuple

from .native_hyrax import NativeHyrax
from .field import log2_power_of_two
from .transcript import Transcript
from .util import digest_file, digest_json, ensure_dir, read_json, write_json


def registration_binding(metadata: Dict[str, Any], arrays: Sequence[Tuple[str, Sequence[int]]]) -> Dict[str, Any]:
    return {
        "protocol": "zkGPT-native-model-registration-v1",
        "model_arch_id": metadata.get("model_arch_id"),
        "model_param_id": metadata.get("model_param_id"),
        "quant_cfg_id": metadata.get("quant_cfg_id"),
        "static_table_id": metadata.get("static_table_id"),
        "layout": metadata.get("private_input_layout"),
        "segment_order": metadata.get("segment_order"),
        "padding_rule_id": metadata.get("padding_rule_id"),
        "integer_bound_ledger_digest": metadata.get("integer_bound_ledger_digest"),
        "Layout_in_digest": metadata.get("Layout_in_digest"),
        "N_in": metadata.get("Layout_in", {}).get("N_in"),
        "registered_model_bounds": metadata.get("integer_bound_ledger", {}).get("registered_model_bounds", {}),
        "arrays": [
            {"name": name, "length": len(values)}
            for name, values in arrays
        ],
    }


def _registry_dir(zk_root: pathlib.Path, model_param_id: str, layout_digest: str) -> pathlib.Path:
    return zk_root / "results" / "model_registry" / model_param_id / layout_digest


def ensure_model_registry(
    zk_root: pathlib.Path,
    metadata: Dict[str, Any],
    arrays: Sequence[Tuple[str, Sequence[int]]],
) -> Dict[str, Any]:
    """Create or validate a model registry and return public+secret handles."""
    binding = registration_binding(metadata, arrays)
    binding_digest = digest_json(binding)
    layout_digest = str(metadata.get("Layout_in_digest", ""))
    if len(layout_digest) != 64:
        raise ValueError("model registration requires a canonical Layout_in digest")
    directory = _registry_dir(zk_root, str(metadata["model_param_id"]), layout_digest)
    public_path = directory / "registry.json"
    secret_path = directory / "registry.secret.json"
    if public_path.exists() and secret_path.exists():
        public = read_json(public_path)
        secret = read_json(secret_path)
        if public.get("registration_binding_digest") != binding_digest or secret.get("registration_binding_digest") != binding_digest:
            raise RuntimeError("existing model registry does not match model/layout/bound-ledger identity")
        entries = secret.get("entries", {})
        if sorted(entries) != sorted(label for label, _ in arrays):
            raise RuntimeError("existing model registry chunk set does not match canonical W layout")
        validation_dir = directory / "validation_work"
        try:
            hyrax = NativeHyrax(zk_root)
            for label, values in arrays:
                hyrax.open_registered(
                    label,
                    values,
                    [0] * log2_power_of_two(len(values)),
                    entries[label],
                    validation_dir,
                    transcript=Transcript("zkGPT/model-registry/validation-v1"),
                    public_context={"registry_validation": binding_digest},
                )
        finally:
            import shutil

            shutil.rmtree(validation_dir, ignore_errors=True)
        return {"public": public, "secret": secret, "public_path": public_path, "secret_path": secret_path}

    ensure_dir(directory)
    work_dir = directory / "setup_work"
    hyrax = NativeHyrax(zk_root)
    public_entries: Dict[str, Any] = {}
    secret_entries: Dict[str, Any] = {}
    try:
        for label, values in arrays:
            registration = hyrax.commit_vector(label, values, work_dir)
            public_entries[label] = {key: value for key, value in registration.items() if key != "blind"}
            secret_entries[label] = registration
    finally:
        # Setup material is not part of the proof artifacts.  The exact
        # registry secret remains in its dedicated local file only.
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)
    public = {
        "protocol": "zkGPT-native-model-registration-v1",
        "model_param_id": metadata["model_param_id"],
        "model_arch_id": metadata["model_arch_id"],
        "registration_binding": binding,
        "registration_binding_digest": binding_digest,
        "commitment_backend": metadata["commitment_backend_id"],
        "entries": public_entries,
    }
    public["registry_digest"] = digest_json(public)
    secret = {
        "protocol": public["protocol"],
        "model_param_id": public["model_param_id"],
        "registration_binding_digest": binding_digest,
        "entries": secret_entries,
    }
    write_json(public_path, public)
    write_json(secret_path, secret)
    return {"public": public, "secret": secret, "public_path": public_path, "secret_path": secret_path}
