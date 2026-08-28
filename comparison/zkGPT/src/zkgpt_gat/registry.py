"""Long-lived zkGPT model registration for the native Hyrax path.

Model parameters are committed once per model/layout reuse key.  Inference
proofs load only the public commitment descriptors and the local setup secret
blindings; every opening still samples fresh masking and IPA randomness.
"""
from __future__ import annotations

import os
import pathlib
import re
import secrets
import shutil
import stat
import tempfile
from typing import Any, Dict, Sequence, Tuple

from .native_hyrax import NativeHyrax
from .field import log2_power_of_two
from .transcript import Transcript
from .util import digest_json, ensure_dir, lexical_path, read_json, require_dir, write_json


MODEL_REGISTRY_ROOT = pathlib.Path("results") / "model_registry"
PRIVATE_MODEL_SETUP_ROOT = pathlib.Path("artifacts") / "private_setup" / "model_registry"
_REGISTRY_PROTOCOL = "zkGPT-native-model-registration-v1"
_TRANSACTION_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_REGISTRY_KEYS = {
    "protocol", "model_param_id", "model_arch_id", "registration_binding",
    "registration_binding_digest", "registration_transaction_id", "commitment_backend",
    "entries", "registry_digest",
}
_PUBLIC_BINDING_KEYS = {
    "protocol", "model_arch_id", "model_param_id", "quant_cfg_id",
    "rounding_mode", "static_table_id", "layout", "segment_order",
    "padding_rule_id", "integer_bound_ledger_digest", "Layout_in_digest",
    "N_in", "registered_model_bounds", "arrays",
}
_PUBLIC_ENTRY_KEYS = {
    "backend", "parameter_id", "label", "vector_length", "commitment", "padded_length",
}
_SECRET_REGISTRY_KEYS = {
    "protocol", "model_param_id", "registration_binding_digest",
    "registration_transaction_id", "entries",
}
_SECRET_ENTRY_KEYS = {"backend", "parameter_id", "label", "vector_length", "commitment", "blind"}
_FORBIDDEN_PUBLIC_KEY_PARTS = {
    "blind", "value", "tensor", "private_path", "secret_path", "checkpoint_path",
    "witness_path", "model_path", "raw_witness", "private_witness",
}


def canonical_registry_dir(zk_root: pathlib.Path, model_param_id: str) -> pathlib.Path:
    """Return the one baseline-local registry directory for a model identity.

    The path is derived only from the trusted formal config identity.  It must
    never be selected from a proof artifact's metadata, which is untrusted.
    """
    identity = str(model_param_id)
    if not identity or identity in {".", ".."} or pathlib.PurePath(identity).name != identity:
        raise ValueError("model_param_id must be a single canonical path component")
    root = require_dir(zk_root, label="model registry baseline root")
    return _strict_registry_path(
        root / MODEL_REGISTRY_ROOT / identity,
        root,
        "canonical public model registry",
        kind="dir",
    )


def canonical_registry_public_path(zk_root: pathlib.Path, model_param_id: str) -> pathlib.Path:
    directory = canonical_registry_dir(zk_root, model_param_id)
    root = require_dir(zk_root, label="model registry baseline root")
    return _strict_registry_path(
        directory / "registry.json",
        root,
        "canonical public model registry file",
        kind="file",
    )


def canonical_registry_secret_dir(zk_root: pathlib.Path, model_param_id: str) -> pathlib.Path:
    """Return the baseline-local prover-only setup directory.

    The secret registry is intentionally outside ``results/``.  The public
    verifier trust root is therefore incapable of discovering it by walking
    the public model-registry directory, and no secret path is serialized into
    proof metadata.
    """
    identity = str(model_param_id)
    if not identity or identity in {".", ".."} or pathlib.PurePath(identity).name != identity:
        raise ValueError("model_param_id must be a single canonical path component")
    root = require_dir(zk_root, label="private model setup baseline root")
    return _strict_registry_path(
        root / PRIVATE_MODEL_SETUP_ROOT / identity,
        root,
        "canonical private model setup",
        kind="dir",
    )


def canonical_registry_secret_path(zk_root: pathlib.Path, model_param_id: str) -> pathlib.Path:
    directory = canonical_registry_secret_dir(zk_root, model_param_id)
    root = require_dir(zk_root, label="private model setup baseline root")
    return _strict_registry_path(
        directory / "registry.secret.json",
        root,
        "canonical private model setup file",
        kind="file",
    )


def _mkdir_chain(path: pathlib.Path, root: pathlib.Path) -> None:
    """Create a contained directory chain while rejecting aliases/special files."""
    try:
        ensure_dir(path, root=root, label="registry directory")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"registry directory must be a real baseline-local directory: {path}") from exc


def _check_directory_state(path: pathlib.Path, *, allow_missing: bool) -> bool:
    """Authenticate one registry directory and return whether it exists."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return False
        raise RuntimeError(f"registry directory is missing: {path}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"registry directory must be a real directory: {path}")
    return True


def _check_file_state(path: pathlib.Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"registry file must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"registry file must be an owned regular file: {path}")
    return True


def require_registry_transaction_id(value: Any) -> str:
    """Validate the opaque transaction id shared by public/private setup."""
    if not isinstance(value, str) or _TRANSACTION_ID_RE.fullmatch(value) is None:
        raise RuntimeError("model registry transaction id must be lowercase SHA-256-sized hex")
    if value == "0" * 64:
        raise RuntimeError("model registry transaction id must not be all zero")
    return value


def _assert_directory_contents(path: pathlib.Path, allowed: set[str]) -> None:
    """Reject stale setup/debug entries that make registry state ambiguous."""
    try:
        children = list(path.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot inspect model registry directory: {path}") from exc
    for child in children:
        if child.name not in allowed:
            raise RuntimeError(f"unsupported or partial registry entry: {child}")


def _assert_public_payload(public: Dict[str, Any], *, expected_labels: Sequence[str]) -> None:
    def scan(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                if any(part in key_lower for part in _FORBIDDEN_PUBLIC_KEY_PARTS):
                    raise RuntimeError("public model registry contains private tensor material")
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    scan(public)
    if set(public) != _PUBLIC_REGISTRY_KEYS:
        raise RuntimeError("public model registry schema contains private or unsupported fields")
    if public.get("protocol") != _REGISTRY_PROTOCOL:
        raise RuntimeError("public model registry protocol mismatch")
    require_registry_transaction_id(public.get("registration_transaction_id"))
    binding = public.get("registration_binding")
    if not isinstance(binding, dict) or set(binding) != _PUBLIC_BINDING_KEYS:
        raise RuntimeError("public model registry binding contains private or unsupported fields")
    entries = public.get("entries")
    if not isinstance(entries, dict) or sorted(entries) != sorted(expected_labels):
        raise RuntimeError("public model registry chunk set does not match canonical W layout")
    for label in expected_labels:
        entry = entries.get(label)
        if not isinstance(entry, dict) or set(entry) != _PUBLIC_ENTRY_KEYS:
            raise RuntimeError("public model registry entry contains private or unsupported fields")


def _assert_secret_payload(secret: Dict[str, Any], *, expected_labels: Sequence[str]) -> None:
    if set(secret) != _SECRET_REGISTRY_KEYS:
        raise RuntimeError("private model registry schema is unsupported")
    if secret.get("protocol") != _REGISTRY_PROTOCOL:
        raise RuntimeError("private model registry protocol mismatch")
    require_registry_transaction_id(secret.get("registration_transaction_id"))
    entries = secret.get("entries")
    if not isinstance(entries, dict) or sorted(entries) != sorted(expected_labels):
        raise RuntimeError("private model registry chunk set does not match canonical W layout")
    for label in expected_labels:
        entry = entries.get(label)
        if not isinstance(entry, dict) or set(entry) != _SECRET_ENTRY_KEYS:
            raise RuntimeError("private model registry entry schema is unsupported")


def _publish_registry_pair(
    public_dir: pathlib.Path,
    secret_dir: pathlib.Path,
    public: Dict[str, Any],
    secret: Dict[str, Any],
    root: pathlib.Path,
) -> None:
    """Publish public and secret registries together or leave no final pair."""
    _mkdir_chain(public_dir.parent, root)
    _mkdir_chain(secret_dir.parent, root)
    if os.path.lexists(public_dir) or os.path.lexists(secret_dir):
        raise RuntimeError("model registry destination already exists; refusing partial overwrite")
    public_stage = pathlib.Path(tempfile.mkdtemp(prefix=".model-registry-public-", dir=str(public_dir.parent)))
    secret_stage = pathlib.Path(tempfile.mkdtemp(prefix=".model-registry-private-", dir=str(secret_dir.parent)))
    published_public = False
    published_secret = False
    published_pair = False
    try:
        write_json(public_stage / "registry.json", public)
        write_json(secret_stage / "registry.secret.json", secret)
        os.chmod(secret_stage, 0o700)
        os.chmod(secret_stage / "registry.secret.json", 0o600)
        # The destination check is repeated immediately before publication;
        # a pre-existing destination is never replaced by registration.
        if os.path.lexists(public_dir) or os.path.lexists(secret_dir):
            raise RuntimeError("model registry destination appeared during setup")
        os.rename(public_stage, public_dir)
        published_public = True
        os.rename(secret_stage, secret_dir)
        published_secret = True
        published_pair = True
    finally:
        if not published_pair:
            # Both destinations were absent at entry and are owned by this
            # publication attempt.  Remove whichever side was published if a
            # later rename failed; callers must never observe a half-pair.
            if published_public:
                shutil.rmtree(public_dir, ignore_errors=True)
            if published_secret:
                shutil.rmtree(secret_dir, ignore_errors=True)
        shutil.rmtree(public_stage, ignore_errors=True)
        shutil.rmtree(secret_stage, ignore_errors=True)


def _strict_registry_path(
    path: pathlib.Path,
    root: pathlib.Path,
    label: str,
    *,
    kind: str = "path",
) -> pathlib.Path:
    """Resolve a registry path after authenticating every component.

    ``Path.resolve`` must never be the first operation here: it would turn an
    in-baseline alias into its target and allow registration to publish into a
    directory selected by a pre-seeded symlink.  Directory and file callers
    additionally authenticate the final inode type and one-link ownership.
    """
    if kind not in {"path", "dir", "file"}:
        raise ValueError(f"unsupported registry path kind: {kind}")
    root = require_dir(root, label=f"{label} baseline root")
    raw_value = pathlib.Path(path).expanduser()
    raw = lexical_path(raw_value if raw_value.is_absolute() else root / raw_value)
    try:
        relative = raw.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes baseline root: {raw}") from exc
    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            # A missing component is safe to report as missing later; it must
            # not be created by a verifier/runner validation operation.
            continue
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"{label} contains a symlink component: {current}")
        is_leaf = index == len(relative.parts) - 1
        if not is_leaf and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{label} parent must be a real directory: {current}")
        if is_leaf and kind == "dir" and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{label} must be a real directory: {current}")
        if is_leaf and kind == "file":
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError(f"{label} must be an owned regular file: {current}")
    resolved = raw.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} resolves outside baseline root: {resolved}") from exc
    return resolved


def validate_registry_pair(
    zk_root: pathlib.Path,
    model_param_id: str,
    *,
    expected_model_arch_id: str | None = None,
    expected_commitment_backend: str | None = None,
) -> Dict[str, Any]:
    """Validate the canonical public/private registry pair before execution.

    This is intentionally structural: the prover's later ``ensure_model_registry``
    call still authenticates the supplied model vector against the persisted
    blind.  The runner can nevertheless reject missing, aliased, hard-linked,
    legacy, or transaction-mismatched setup files before loading a dataset.
    """
    root = require_dir(zk_root, label="model registry baseline root")
    # Reuse the same single-component identity gate as registration and the
    # independent verifier; do not allow ``..`` or slash-bearing selectors.
    identity = str(model_param_id)
    # First inspect the spelling selected by the caller, so an in-root
    # symlink alias cannot be hidden by ``Path.resolve()`` in the canonical
    # helpers.  The canonical helpers are then used for the actual paths,
    # keeping runner, registration, and verifier on one schema/path API.
    public_dir_raw = root / MODEL_REGISTRY_ROOT / identity
    secret_dir_raw = root / PRIVATE_MODEL_SETUP_ROOT / identity
    _strict_registry_path(public_dir_raw, root, "public model registry", kind="dir")
    _strict_registry_path(secret_dir_raw, root, "private model setup", kind="dir")
    _strict_registry_path(
        public_dir_raw / "registry.json", root, "public model registry file alias", kind="file"
    )
    _strict_registry_path(
        secret_dir_raw / "registry.secret.json", root, "private model registry file alias", kind="file"
    )
    public_dir = _strict_registry_path(
        canonical_registry_dir(root, identity), root, "canonical public model registry", kind="dir"
    )
    secret_dir = _strict_registry_path(
        canonical_registry_secret_dir(root, identity), root, "canonical private model setup", kind="dir"
    )
    public_path = _strict_registry_path(
        canonical_registry_public_path(root, identity), root, "public model registry file", kind="file"
    )
    secret_path = _strict_registry_path(
        canonical_registry_secret_path(root, identity), root, "private model registry file", kind="file"
    )
    legacy_path = _strict_registry_path(
        public_dir_raw / "registry.secret.json", root, "legacy model registry secret", kind="file"
    )

    if os.path.lexists(public_dir_raw):
        _check_directory_state(public_dir, allow_missing=False)
    if os.path.lexists(secret_dir_raw):
        _check_directory_state(secret_dir, allow_missing=False)
    if os.path.lexists(legacy_path):
        raise RuntimeError("legacy public-directory model secret is unsupported")
    if not os.path.lexists(public_path):
        raise FileNotFoundError(f"canonical public model registry is missing: {public_path}")
    if not os.path.lexists(secret_path):
        raise FileNotFoundError(f"canonical private model setup is missing: {secret_path}")
    _check_file_state(public_path)
    _check_file_state(secret_path)
    _assert_directory_contents(public_dir, {"registry.json"})
    _assert_directory_contents(secret_dir, {"registry.secret.json"})
    public = read_json(public_path)
    secret = read_json(secret_path)
    _assert_public_payload(public, expected_labels=["com_W"])
    _assert_secret_payload(secret, expected_labels=["com_W"])
    if public.get("registration_transaction_id") != secret.get("registration_transaction_id"):
        raise RuntimeError("public/private model registry transaction id mismatch")
    if public.get("model_param_id") != str(model_param_id) or secret.get("model_param_id") != str(model_param_id):
        raise RuntimeError("model registry model_param_id mismatch")
    if expected_model_arch_id is not None and public.get("model_arch_id") != expected_model_arch_id:
        raise RuntimeError("public model registry model_arch_id mismatch")
    if expected_commitment_backend is not None and public.get("commitment_backend") != expected_commitment_backend:
        raise RuntimeError("public model registry commitment backend mismatch")
    if public.get("registration_binding_digest") != secret.get("registration_binding_digest"):
        raise RuntimeError("public/private model registry binding digest mismatch")
    return {
        "public": public,
        "secret": secret,
        "public_path": public_path,
        "secret_path": secret_path,
        "transaction_id": public["registration_transaction_id"],
    }


def registration_binding(metadata: Dict[str, Any], arrays: Sequence[Tuple[str, Sequence[int]]]) -> Dict[str, Any]:
    return {
        "protocol": "zkGPT-native-model-registration-v1",
        "model_arch_id": metadata.get("model_arch_id"),
        "model_param_id": metadata.get("model_param_id"),
        "quant_cfg_id": metadata.get("quant_cfg_id"),
        "rounding_mode": metadata.get("rounding_mode"),
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
    # Layout is part of the signed registry binding, but is not a trust-root
    # selector.  Keeping one deterministic path per config/model identity
    # prevents an artifact from nominating an alternate registry directory.
    del layout_digest
    return canonical_registry_dir(zk_root, model_param_id)


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
    root = require_dir(zk_root, label="model registry baseline root")
    identity = str(metadata.get("model_param_id", ""))
    # Authenticate the caller-selected lexical roots before any canonical
    # helper resolves them.  This rejects both escaping aliases and aliases
    # that resolve back inside the baseline, including deep model-registry
    # components pre-seeded by an attacker.
    public_dir_raw = root / MODEL_REGISTRY_ROOT / identity
    secret_dir_raw = root / PRIVATE_MODEL_SETUP_ROOT / identity
    _strict_registry_path(public_dir_raw, root, "public model registry", kind="dir")
    _strict_registry_path(secret_dir_raw, root, "private model setup", kind="dir")
    directory = _registry_dir(root, str(metadata["model_param_id"]), layout_digest)
    public_path = _strict_registry_path(
        directory / "registry.json", root, "public model registry file", kind="file"
    )
    secret_dir = _strict_registry_path(
        canonical_registry_secret_dir(root, str(metadata["model_param_id"])),
        root,
        "private model setup",
        kind="dir",
    )
    secret_path = _strict_registry_path(
        secret_dir / "registry.secret.json", root, "private model registry file", kind="file"
    )
    legacy_secret_path = _strict_registry_path(
        directory / "registry.secret.json", root, "legacy model registry secret", kind="file"
    )
    public_present = _check_file_state(public_path)
    secret_present = _check_file_state(secret_path)
    public_dir_present = _check_directory_state(directory, allow_missing=True)
    secret_dir_present = _check_directory_state(secret_dir, allow_missing=True)
    if os.path.lexists(legacy_secret_path):
        raise RuntimeError(
            "legacy public-directory model secret is unsupported; remove it manually "
            "before registration"
        )
    if public_present != secret_present:
        raise RuntimeError("model registry is partial; refusing to create or overwrite either side")
    if public_present and (not public_dir_present or not secret_dir_present):
        raise RuntimeError("model registry file is not contained by its canonical directory")
    if public_present and secret_present:
        _assert_directory_contents(directory, {"registry.json"})
        _assert_directory_contents(secret_dir, {"registry.secret.json"})
        public = read_json(public_path)
        secret = read_json(secret_path)
        expected_labels = [label for label, _ in arrays]
        _assert_public_payload(public, expected_labels=expected_labels)
        _assert_secret_payload(secret, expected_labels=expected_labels)
        if (
            public.get("model_param_id") != metadata.get("model_param_id")
            or public.get("model_arch_id") != metadata.get("model_arch_id")
            or public.get("commitment_backend") != metadata.get("commitment_backend_id")
        ):
            raise RuntimeError("existing model registry uses an unsupported public schema")
        public_payload = dict(public)
        public_registry_digest = public_payload.pop("registry_digest", None)
        if not public_registry_digest or public_registry_digest != digest_json(public_payload):
            raise RuntimeError("existing model registry public digest is invalid")
        public_entries = public["entries"]
        for label, values in arrays:
            entry = public_entries.get(label)
            try:
                vector_length = int(entry.get("vector_length", -1))
                padded_length = int(entry["padded_length"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("existing model registry public lengths are invalid") from exc
            if (
                entry.get("backend") != metadata.get("commitment_backend_id")
                or entry.get("label") != label
                or vector_length != len(values)
                or padded_length != vector_length
            ):
                raise RuntimeError("existing model registry public entry does not match canonical W")
        if secret.get("model_param_id") != metadata.get("model_param_id"):
            raise RuntimeError("existing model registry secret identity mismatch")
        if public.get("registration_transaction_id") != secret.get("registration_transaction_id"):
            raise RuntimeError("public/private model registry transaction id mismatch")
        if public.get("registration_binding_digest") != binding_digest or secret.get("registration_binding_digest") != binding_digest:
            raise RuntimeError("existing model registry does not match model/layout/bound-ledger identity")
        entries = secret["entries"]
        validation_dir = secret_dir / ".validation_work"
        try:
            hyrax = NativeHyrax(root)
            for label, values in arrays:
                hyrax.register_vector(label, values, entries[label], validation_dir)
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
            shutil.rmtree(validation_dir, ignore_errors=True)
        return {"public": public, "secret": secret, "public_path": public_path, "secret_path": secret_path}

    if public_dir_present or secret_dir_present:
        # Empty directories left by an interrupted external cleanup are not a
        # valid setup target.  Refuse to guess whether they contain hidden
        # material; the owner can remove empty directories explicitly.
        if public_dir_present:
            raise RuntimeError("public model registry directory exists without a complete registry")
        raise RuntimeError("private model setup directory exists without a complete registry")
    work_dir = secret_dir.parent / f".setup-work-{os.getpid()}"
    if os.path.lexists(work_dir):
        raise RuntimeError("private model setup work directory already exists")
    _mkdir_chain(work_dir.parent, root)
    ensure_dir(work_dir, root=root, label="private model setup work directory")
    hyrax = NativeHyrax(root)
    public_entries: Dict[str, Any] = {}
    secret_entries: Dict[str, Any] = {}
    try:
        for label, values in arrays:
            registration = hyrax.commit_vector(label, values, work_dir)
            public_entries[label] = {key: value for key, value in registration.items() if key != "blind"}
            # Keep the public setup descriptor explicit about canonical
            # padding.  The independent verifier authenticates both fields;
            # for a registered model vector they are equal by construction.
            public_entries[label]["padded_length"] = int(public_entries[label]["vector_length"])
            secret_entries[label] = registration
    finally:
        # Setup material is not part of the proof artifacts.  The exact
        # registry secret remains in its dedicated local file only.
        shutil.rmtree(work_dir, ignore_errors=True)
    public = {
        "protocol": _REGISTRY_PROTOCOL,
        "model_param_id": metadata["model_param_id"],
        "model_arch_id": metadata["model_arch_id"],
        "registration_transaction_id": secrets.token_hex(32),
        "registration_binding": binding,
        "registration_binding_digest": binding_digest,
        "commitment_backend": metadata["commitment_backend_id"],
        "entries": public_entries,
    }
    public["registry_digest"] = digest_json(public)
    secret = {
        "protocol": public["protocol"],
        "model_param_id": public["model_param_id"],
        "registration_transaction_id": public["registration_transaction_id"],
        "registration_binding_digest": binding_digest,
        "entries": secret_entries,
    }
    _assert_public_payload(public, expected_labels=[label for label, _ in arrays])
    _assert_secret_payload(secret, expected_labels=[label for label, _ in arrays])
    _publish_registry_pair(directory, secret_dir, public, secret, root)
    return {"public": public, "secret": secret, "public_path": public_path, "secret_path": secret_path}
