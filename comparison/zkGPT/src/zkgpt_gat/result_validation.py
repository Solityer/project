"""Shared structural validation for zkGPT formal artifacts.

The collector and matrix validator use the same fail-closed checks.  No test
fixture is accepted by default; the optional test-only path is explicit and is
never enabled by production scripts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import stat
from typing import Any, Dict, Iterable, Mapping

from .config import load_app_config
from .constants import (
    DATASET_ORDER,
    FORMAL_GAT_CLASS_COUNTS,
    FORMAL_GAT_D_IN_PROFILE,
    FORMAL_GAT_HIDDEN_PROFILE,
    FORMAL_GAT_K_OUT,
    FORMAL_GAT_LAYER_COUNT,
    FORMAL_GAT_MODEL_ARCH_IDS,
    PROOF_BACKEND_ID,
    PROTOCOL_ID,
)
from .model import CANONICAL_CHECKPOINT_SCHEMA, _validate_canonical_manifest
from .quantization import CANONICAL_QUANT_CFG_ID, canonical_input_preprocess_id
from .gpu_execution import validate_gpu_execution_evidence
from .registry import canonical_registry_public_path
from .privacy import verify_privacy_profile
from .resources import (
    STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH,
    STATIC_TABLE_DESCRIPTOR_SCHEMA,
    canonical_dataset_name,
    read_resource_manifest,
    validate_static_table_descriptor,
)
from .util import canonical_json, digest_json


RUN_MANIFEST_SCHEMA = "zkGPT_gat_run_manifest_v4"
INDEPENDENT_REPORT_SCHEMA = "zkGPT_independent_verification_v4"
RESULT_MATRIX_SCHEMA = "zkGPT_gat_formal_result_matrix_v4"
REQUIRED_ARTIFACTS = (
    "metadata.json",
    "commitments.json",
    "transcript.json",
    "proof_payload.json",
    "lookup_proof.json",
    "fixed_lookup_tables.json",
    "matrix_sumcheck_proof.json",
    "gkr_proof.json",
    "combine_proof.json",
    "hyrax_mle_openings.json",
    "privacy_profile.json",
    "verification.json",
    "independent_verification.json",
    "witness_manifest.json",
    "gpu_backend.json",
    "benchmark_summary.json",
    "data_id.json",
    "model_id.json",
    "config_id.json",
)
FORBIDDEN_DUPLICATE_ARTIFACTS = frozenset(("proof_object.json", "verifier_output.json"))
GPU_ARTIFACTS = ("gpu_proof_hot_path.json",)
DATASET_SET = frozenset(DATASET_ORDER)
BACKEND_SET = frozenset(("cpu", "gpu"))
TEST_ONLY_SCHEMAS = frozenset((
    "zkGPT_test_fixture_v1",
    "zkGPT_synthetic_e2e_cache_v1",
))
RUN_MANIFEST_NAME = "run_manifest.json"


class ResultValidationError(ValueError):
    """An artifact/manifest failed a formal result contract check."""


def _lexical_path(raw: str | pathlib.Path, root: pathlib.Path | None = None) -> pathlib.Path:
    """Return an absolute, normalized path without following symlinks."""
    value = pathlib.Path(raw).expanduser()
    if not value.is_absolute():
        value = (root if root is not None else pathlib.Path.cwd()) / value
    return pathlib.Path(os.path.abspath(os.path.normpath(os.fspath(value))))


def _baseline_lexical(root: pathlib.Path) -> pathlib.Path:
    baseline = _lexical_path(root)
    if baseline.is_symlink():
        raise ResultValidationError(f"baseline root must not be a symlink: {baseline}")
    return baseline


def _check_no_symlink_components(
    path: pathlib.Path,
    root: pathlib.Path,
    *,
    allow_missing: bool,
) -> None:
    """Reject symlinks in every component between ``root`` and ``path``.

    This check intentionally uses lexical paths and ``lstat``-style
    ``Path.is_symlink`` checks before any call to ``resolve``.  Resolving first
    would erase the evidence needed to reject an escaping alias.
    """
    root = _lexical_path(root)
    path = _lexical_path(path)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ResultValidationError(f"path escapes baseline root: {path}") from exc
    current = root
    if current.is_symlink():
        raise ResultValidationError(f"path component is a symlink: {current}")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ResultValidationError(f"path component is a symlink: {current}")
        if not allow_missing and not current.exists():
            raise ResultValidationError(f"path component is missing: {current}")


def _require_regular_artifact_file(
    artifact_dir: pathlib.Path,
    name: str | pathlib.Path,
) -> pathlib.Path:
    """Return one canonical artifact file, rejecting aliases and hardlinks."""
    artifact = _lexical_path(artifact_dir)
    if artifact.is_symlink() or not artifact.is_dir():
        raise ResultValidationError(f"artifact directory must be a real directory: {artifact}")
    _check_no_symlink_components(artifact, artifact, allow_missing=False)
    relative_name = pathlib.Path(name)
    if (
        relative_name.is_absolute()
        or len(relative_name.parts) != 1
        or relative_name.name in {"", ".", ".."}
    ):
        raise ResultValidationError(f"artifact name must be a single file name: {name}")
    path = _lexical_path(artifact / relative_name.name)
    if path.parent != artifact:
        raise ResultValidationError(f"artifact parent is not canonical: {path}")
    if path.is_symlink():
        raise ResultValidationError(f"artifact file must not be a symlink: {path}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ResultValidationError(f"artifact file is missing: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ResultValidationError(f"artifact file is not a regular file: {path}")
    if info.st_nlink != 1:
        raise ResultValidationError(f"artifact file hardlink count must be one: {path}")
    try:
        resolved_parent = path.resolve(strict=True).parent
        artifact_resolved = artifact.resolve(strict=True)
    except OSError as exc:
        raise ResultValidationError(f"cannot resolve artifact file: {path}") from exc
    if resolved_parent != artifact_resolved:
        raise ResultValidationError(f"artifact file resolves outside its directory: {path}")
    return path


def required_artifact_file(artifact_dir: pathlib.Path, name: str | pathlib.Path) -> pathlib.Path:
    """Public shared helper for all required production artifact reads."""
    return _require_regular_artifact_file(artifact_dir, name)


def _canonical_artifact_shape(root: pathlib.Path, candidate: pathlib.Path) -> tuple[pathlib.Path, str, str]:
    """Check the lexical canonical layout before reading a run manifest."""
    baseline = _baseline_lexical(root)
    candidate = _lexical_path(candidate, baseline)
    results = _lexical_path(baseline / "results")
    runs = results / "runs"
    try:
        relative = candidate.relative_to(runs)
    except ValueError as exc:
        raise ResultValidationError(
            "artifact directory must be under baseline/results/runs"
        ) from exc
    if len(relative.parts) != 2:
        raise ResultValidationError(
            "artifact directory must be exactly results/runs/<backend>/<dataset>"
        )
    backend, dataset = relative.parts
    if backend not in BACKEND_SET:
        raise ResultValidationError(f"unsupported result backend: {backend}")
    try:
        canonical_dataset = canonical_dataset_name(dataset)
    except ValueError as exc:
        raise ResultValidationError(str(exc)) from exc
    if dataset != canonical_dataset:
        raise ResultValidationError(f"dataset path is not canonical: {dataset}")
    expected = runs / backend / canonical_dataset
    if candidate != expected:
        raise ResultValidationError(f"artifact directory is not canonical: {candidate}")
    return candidate, backend, canonical_dataset


def require_canonical_artifact_dir(
    root: pathlib.Path,
    candidate: pathlib.Path,
    *,
    allow_test_fixture: bool = False,
) -> pathlib.Path:
    """Validate a production artifact directory and all of its path components.

    ``allow_test_fixture`` is an explicit internal escape hatch for isolated
    unit fixtures.  Production scripts never pass it; the canonical layout and
    non-symlink checks still apply when it is enabled.
    """
    del allow_test_fixture  # The flag is intentionally explicit at call sites.
    return validate_canonical_cell_dir_from_path(root, candidate)


def validate_canonical_cell_dir_from_path(
    baseline: pathlib.Path,
    artifact_dir: pathlib.Path,
) -> pathlib.Path:
    """Authenticate a production cell directory before reading any artifact.

    The accepted shape is exactly ``results/runs/<backend>/<dataset>`` under
    the baseline root.  The check is lexical first and then rejects symlinks
    in every component, including the cell directory itself.  This helper is
    intentionally independent of ``run_manifest.json`` so an independent
    verifier can produce that report before the final manifest exists.
    """
    baseline = _baseline_lexical(baseline)
    artifact, _backend, _dataset = _canonical_artifact_shape(baseline, artifact_dir)
    _check_no_symlink_components(artifact, baseline, allow_missing=False)
    if artifact.is_symlink() or not artifact.is_dir():
        raise ResultValidationError(f"artifact directory must be a real directory: {artifact}")
    try:
        resolved = artifact.resolve(strict=True)
    except OSError as exc:
        raise ResultValidationError(f"cannot resolve artifact directory: {artifact}") from exc
    if resolved != artifact:
        raise ResultValidationError(f"artifact directory resolves through an alias: {artifact}")
    return artifact


def required_artifact_names(backend: str, *, include_manifest: bool = False) -> tuple[str, ...]:
    names = list(REQUIRED_ARTIFACTS) + (list(GPU_ARTIFACTS) if backend == "gpu" else [])
    if include_manifest:
        names.append(RUN_MANIFEST_NAME)
    return tuple(names)


def validate_artifact_inventory(
    artifact_dir: pathlib.Path,
    backend: str,
    *,
    require_independent: bool = True,
    require_manifest: bool = False,
) -> Dict[str, pathlib.Path]:
    """Authenticate the complete *top-level* artifact inventory.

    Proof production has one deliberate intermediate state: the independent
    verifier report is absent while the fresh verifier is reading the proof.
    ``require_independent=False`` accepts that state, but still rejects every
    unknown entry, nested directory, alias, hardlink, or special file.  Once
    the fresh report exists, callers use the default and the only additional
    accepted entry is the final run manifest when ``require_manifest=True``.

    Keeping this check centralized is important: the producer, independent
    verifier, collector, and matrix validator must agree on the exact wire
    directory rather than each silently ignoring different leftovers.
    """
    if backend not in BACKEND_SET:
        raise ResultValidationError(f"unsupported result backend: {backend}")
    artifact = _lexical_path(artifact_dir)
    if artifact.is_symlink() or not artifact.is_dir():
        raise ResultValidationError(f"artifact directory must be a real directory: {artifact}")
    # The caller may use this helper before the final manifest exists, so do
    # not require a canonical results path here.  Still authenticate every
    # component below the supplied directory before enumerating entries.
    _check_no_symlink_components(artifact, artifact, allow_missing=False)
    expected = set(required_artifact_names(backend, include_manifest=require_manifest))
    if not require_independent:
        expected.discard("independent_verification.json")
    allowed = set(expected)
    if not require_independent:
        allowed.add("independent_verification.json")
    if not require_manifest:
        # A fresh replay of an already completed cell sees the existing final
        # manifest; producer-side pre-verification simply omits it.  It is
        # optional at this phase, never an unrecognized extra.
        allowed.add(RUN_MANIFEST_NAME)

    entries: Dict[str, pathlib.Path] = {}
    try:
        children = list(artifact.iterdir())
    except OSError as exc:
        raise ResultValidationError(f"cannot enumerate artifact directory: {artifact}") from exc
    for child in children:
        name = child.name
        if name not in allowed:
            raise ResultValidationError(f"non-canonical artifact entry is forbidden: {child}")
        try:
            info = child.lstat()
        except OSError as exc:
            raise ResultValidationError(f"cannot inspect artifact entry: {child}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ResultValidationError(f"artifact entry must not be a symlink: {child}")
        if stat.S_ISDIR(info.st_mode):
            raise ResultValidationError(f"nested artifact directory is forbidden: {child}")
        if not stat.S_ISREG(info.st_mode):
            raise ResultValidationError(f"artifact entry must be a regular file: {child}")
        if info.st_nlink != 1:
            raise ResultValidationError(f"artifact entry hardlink count must be one: {child}")
        entries[name] = child

    missing = sorted(expected - set(entries))
    if missing:
        raise ResultValidationError(f"artifact inventory is incomplete: {missing}")
    # Reuse the stronger per-file check for every expected entry.  This also
    # verifies that no path resolution aliases the authenticated directory.
    return {
        name: required_artifact_file(artifact, name)
        for name in sorted(expected)
    }


def validate_required_artifacts(
    artifact_dir: pathlib.Path,
    backend: str,
    *,
    include_manifest: bool = False,
) -> Dict[str, pathlib.Path]:
    """Validate every required artifact as a canonical one-link regular file."""
    return validate_artifact_inventory(
        artifact_dir,
        backend,
        require_independent=True,
        require_manifest=include_manifest,
    )


def load_canonical_manifest(
    root: pathlib.Path,
    manifest_path: pathlib.Path,
    *,
    allow_test_fixture: bool = False,
) -> tuple[pathlib.Path, Dict[str, Any]]:
    """Validate path containment before reading the canonical run manifest.

    The lexical path is checked first (including the manifest's regular-file
    type and link count).  Only then is the manifest read to authenticate the
    dataset/backend identity and compare it with the canonical directory.
    """
    baseline = _baseline_lexical(root)
    supplied = _lexical_path(manifest_path, baseline)
    if supplied.name != RUN_MANIFEST_NAME:
        raise ResultValidationError("manifest path must name run_manifest.json")
    artifact, path_backend, path_dataset = _canonical_artifact_shape(baseline, supplied.parent)
    _check_no_symlink_components(artifact, baseline, allow_missing=False)
    if artifact.is_symlink() or not artifact.is_dir():
        raise ResultValidationError(f"artifact directory must be a real directory: {artifact}")
    manifest_file = _require_regular_artifact_file(artifact, RUN_MANIFEST_NAME)
    manifest = read_object(manifest_file, "run manifest")
    dataset = str(manifest.get("dataset", ""))
    backend = str(manifest.get("backend", ""))
    try:
        canonical_dataset = canonical_dataset_name(dataset)
    except ValueError as exc:
        raise ResultValidationError(str(exc)) from exc
    if backend not in BACKEND_SET or canonical_dataset not in DATASET_SET:
        raise ResultValidationError("run manifest dataset/backend is invalid")
    if backend != path_backend or canonical_dataset != path_dataset:
        raise ResultValidationError("run manifest dataset/backend does not match canonical path")
    declared = manifest.get("artifact_path", "")
    declared_path = _lexical_path(declared, baseline)
    if declared_path != artifact:
        raise ResultValidationError("run manifest artifact_path is not canonical")
    if not allow_test_fixture and manifest.get("schema") in TEST_ONLY_SCHEMAS:
        raise ResultValidationError("test-only run manifest is not accepted by production validation")
    if not allow_test_fixture:
        validate_artifact_inventory(
            artifact,
            backend,
            require_independent=True,
            require_manifest=True,
        )
    return artifact, manifest


def canonical_results_dir(root: pathlib.Path) -> pathlib.Path:
    """Return the one production result root for this baseline.

    Formal tools may not select a sibling ``runs``/``results`` alias.  The
    directory is resolved through :func:`resolve_inside` so a symlink that
    escapes the baseline is rejected as well.
    """
    baseline = _baseline_lexical(root)
    lexical = baseline / "results"
    _check_no_symlink_components(lexical, baseline, allow_missing=True)
    if lexical.exists() and (lexical.is_symlink() or not lexical.is_dir()):
        raise ResultValidationError("canonical results path must be a real directory")
    return lexical


def resolve_results_dir(
    root: pathlib.Path,
    raw: str | pathlib.Path,
    *,
    allow_test_fixture: bool = False,
) -> pathlib.Path:
    """Resolve the one production result root.

    Production CLIs accept only ``<baseline>/results``.  Test callers may
    explicitly opt into an isolated fixture root, but that escape hatch is
    never used by the formal scripts.
    """
    baseline = _baseline_lexical(root)
    candidate = _lexical_path(raw, baseline)
    if not allow_test_fixture and candidate != canonical_results_dir(baseline):
        raise ResultValidationError(
            "results_dir must equal the canonical baseline results directory"
        )
    _check_no_symlink_components(candidate, baseline, allow_missing=True)
    if candidate.exists() and (candidate.is_symlink() or not candidate.is_dir()):
        raise ResultValidationError("results_dir must be a real directory")
    return candidate


def canonical_artifact_dir(
    root: pathlib.Path,
    backend: str,
    dataset: str,
) -> pathlib.Path:
    """Return the only valid per-cell artifact directory.

    The canonical layout is deliberately centralized so the producer,
    collector, replay validator, and matrix validator cannot drift into
    separate ``results/<backend>`` or other compatibility layouts.
    """
    if backend not in BACKEND_SET:
        raise ResultValidationError(f"unsupported result backend: {backend}")
    try:
        canonical_dataset = canonical_dataset_name(dataset)
    except ValueError as exc:
        raise ResultValidationError(str(exc)) from exc
    return canonical_results_dir(root) / "runs" / backend / canonical_dataset


def canonical_manifest_path(
    root: pathlib.Path,
    backend: str,
    dataset: str,
) -> pathlib.Path:
    """Return the canonical per-cell run manifest path."""
    return canonical_artifact_dir(root, backend, dataset) / "run_manifest.json"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside(root: pathlib.Path, raw: str | pathlib.Path, label: str) -> pathlib.Path:
    root = pathlib.Path(root).expanduser().resolve()
    value = pathlib.Path(raw).expanduser()
    candidate = value.resolve() if value.is_absolute() else (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ResultValidationError(f"{label} escapes baseline root: {candidate}") from exc
    return candidate


def relative_path(root: pathlib.Path, path: pathlib.Path) -> str:
    return str(path.expanduser().resolve().relative_to(pathlib.Path(root).expanduser().resolve()))


def read_object(path: pathlib.Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultValidationError(f"{label} must be a JSON object: {path}")
    return value


def _read_owned_resource_object(path: pathlib.Path, label: str) -> Dict[str, Any]:
    """Read one baseline-local JSON resource without following aliases."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise ResultValidationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ResultValidationError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ResultValidationError(f"{label} must be an owned regular file: {path}")
    return read_object(path, label)


def _validate_formal_model_binding(
    root: pathlib.Path,
    config_path: pathlib.Path,
    metadata: Mapping[str, Any],
    model_id: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Bind a result cell to the canonical L3/multi-head checkpoint schema.

    The independent verifier already reconstructs its context from the
    baseline-local checkpoint manifest.  The collector and final validator
    also call this check so a stale result row cannot survive after a model
    profile migration merely because the proof files have the right names.
    Tensor values are never opened here.
    """
    try:
        config = load_app_config(config_path, root)
        dataset = canonical_dataset_name(config.dataset)
    except (OSError, TypeError, ValueError) as exc:
        raise ResultValidationError(f"formal config/model binding is invalid: {exc}") from exc
    expected_hidden = list(FORMAL_GAT_HIDDEN_PROFILE)
    actual_hidden = [(item.head_count, item.head_dim) for item in config.hidden_profile]
    expected_d_in = list(FORMAL_GAT_D_IN_PROFILE[dataset])
    if config.layer_count != FORMAL_GAT_LAYER_COUNT:
        raise ResultValidationError("formal result config is not L3")
    if actual_hidden != expected_hidden:
        raise ResultValidationError("formal result config hidden_profile is not canonical")
    if config.d_in_profile != expected_d_in:
        raise ResultValidationError("formal result config d_in_profile is not canonical")
    if config.k_out != FORMAL_GAT_K_OUT or config.num_classes != FORMAL_GAT_CLASS_COUNTS[dataset]:
        raise ResultValidationError("formal result config output shape is not canonical")
    expected_arch = FORMAL_GAT_MODEL_ARCH_IDS[dataset]
    if config.entries.get("model_arch_id") != expected_arch:
        raise ResultValidationError("formal result config model_arch_id is not canonical")
    if canonical_dataset_name(str(manifest.get("dataset", ""))) != dataset:
        raise ResultValidationError("run manifest dataset does not match config identity")

    checkpoint_manifest_path = config.checkpoint_bundle / "manifest.json"
    checkpoint_manifest = _read_owned_resource_object(
        checkpoint_manifest_path, "checkpoint manifest"
    )
    try:
        _validate_canonical_manifest(checkpoint_manifest)
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"checkpoint manifest is not canonical: {exc}") from exc
    if checkpoint_manifest.get("model_arch_id") != expected_arch:
        raise ResultValidationError("checkpoint manifest model_arch_id mismatch")
    if checkpoint_manifest.get("model_param_id") != config.entries.get("model_param_id"):
        raise ResultValidationError("checkpoint manifest model_param_id mismatch")
    if checkpoint_manifest.get("quant_cfg_id") != CANONICAL_QUANT_CFG_ID or checkpoint_manifest.get("input_preprocess_id") != config.input_preprocess_id:
        raise ResultValidationError("checkpoint manifest numerical/preprocessing identity mismatch")
    if int(checkpoint_manifest.get("L", -1)) != FORMAL_GAT_LAYER_COUNT:
        raise ResultValidationError("checkpoint manifest L is not canonical")
    if int(checkpoint_manifest.get("K_out", -1)) != FORMAL_GAT_K_OUT:
        raise ResultValidationError("checkpoint manifest K_out is not canonical")
    if int(checkpoint_manifest.get("C", -1)) != FORMAL_GAT_CLASS_COUNTS[dataset]:
        raise ResultValidationError("checkpoint manifest class count mismatch")
    hidden = checkpoint_manifest.get("hidden_layers")
    if not isinstance(hidden, list):
        raise ResultValidationError("checkpoint manifest hidden_layers is malformed")
    try:
        checkpoint_hidden = [
            (int(item["head_count"]), int(item["head_dim"])) for item in hidden
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultValidationError("checkpoint manifest hidden_layers is malformed") from exc
    if checkpoint_hidden != expected_hidden:
        raise ResultValidationError("checkpoint manifest hidden_profile mismatch")
    checkpoint_d_in = checkpoint_manifest.get("d_in_profile")
    try:
        checkpoint_d_in = [int(value) for value in checkpoint_d_in]
    except (TypeError, ValueError) as exc:
        raise ResultValidationError("checkpoint manifest d_in_profile is malformed") from exc
    if checkpoint_d_in != expected_d_in:
        raise ResultValidationError("checkpoint manifest d_in_profile mismatch")
    output_heads = checkpoint_manifest.get("output_head_specs")
    if not isinstance(output_heads, list) or len(output_heads) != FORMAL_GAT_K_OUT:
        raise ResultValidationError("checkpoint manifest output head count mismatch")
    try:
        output_indices = [int(item["head_index"]) for item in output_heads]
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultValidationError("checkpoint manifest output_head_specs is malformed") from exc
    if output_indices != list(range(FORMAL_GAT_K_OUT)):
        raise ResultValidationError("checkpoint manifest output head indices are not contiguous")

    if model_id.get("checkpoint_bundle") != str(config.checkpoint_bundle):
        raise ResultValidationError("model identity checkpoint bundle is not canonical")
    model_manifest_id = model_id.get("manifest")
    if not isinstance(model_manifest_id, Mapping):
        raise ResultValidationError("model identity manifest record is missing")
    if model_manifest_id.get("path") != str(checkpoint_manifest_path):
        raise ResultValidationError("model identity manifest path is not canonical")
    if model_manifest_id.get("size_bytes") != checkpoint_manifest_path.stat().st_size:
        raise ResultValidationError("model identity manifest size mismatch")
    if model_manifest_id.get("sha256") != sha256_file(checkpoint_manifest_path):
        raise ResultValidationError("model identity manifest digest mismatch")

    dim_cfg = metadata.get("dim_cfg")
    if not isinstance(dim_cfg, Mapping):
        raise ResultValidationError("metadata dim_cfg is missing")
    metadata_hidden = dim_cfg.get("hidden_profile")
    if not isinstance(metadata_hidden, list):
        raise ResultValidationError("metadata hidden_profile is missing")
    try:
        normalized_metadata_hidden = [
            (int(item["layer_index"]), int(item["head_count"]), int(item["head_dim"]))
            for item in metadata_hidden
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultValidationError("metadata hidden_profile is malformed") from exc
    expected_metadata_hidden = [
        (index, head_count, head_dim)
        for index, (head_count, head_dim) in enumerate(expected_hidden)
    ]
    if normalized_metadata_hidden != expected_metadata_hidden:
        raise ResultValidationError("metadata hidden_profile mismatch")
    if dim_cfg.get("L") != FORMAL_GAT_LAYER_COUNT or dim_cfg.get("K_out") != FORMAL_GAT_K_OUT or dim_cfg.get("C") != FORMAL_GAT_CLASS_COUNTS[dataset]:
        raise ResultValidationError("metadata output dimensions mismatch")
    if dim_cfg.get("d_in_profile") != expected_d_in:
        raise ResultValidationError("metadata d_in_profile mismatch")
    expected_internal_d_in = expected_d_in + [expected_hidden[-1][0] * expected_hidden[-1][1]]
    if dim_cfg.get("d_in") != expected_internal_d_in:
        raise ResultValidationError("metadata internal d_in dimensions mismatch")
    for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id"):
        if metadata.get(key) != config.entries.get(key):
            raise ResultValidationError(f"metadata {key} does not match formal config")
    if manifest.get("L") != FORMAL_GAT_LAYER_COUNT:
        raise ResultValidationError("run manifest L is not canonical")
    if manifest.get("hidden_profile") != "8x8,8x8":
        raise ResultValidationError("run manifest hidden_profile is not canonical")
    if manifest.get("d_in_profile") != ",".join(str(value) for value in expected_d_in):
        raise ResultValidationError("run manifest d_in_profile is not canonical")
    if manifest.get("K_out") != FORMAL_GAT_K_OUT or manifest.get("C") != FORMAL_GAT_CLASS_COUNTS[dataset]:
        raise ResultValidationError("run manifest output dimensions are not canonical")


def artifact_digest_map(artifact_dir: pathlib.Path, backend: str) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for name in required_artifact_names(backend):
        path = required_artifact_file(artifact_dir, name)
        output[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return output


def manifest_run_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("run_digest", None)
    return digest_json(payload)


def _require_bool(mapping: Mapping[str, Any], key: str, label: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ResultValidationError(f"{label}.{key} must be boolean")
    return value


def _verified_fields(report: Mapping[str, Any]) -> Dict[str, Any]:
    if report.get("schema") != INDEPENDENT_REPORT_SCHEMA:
        raise ResultValidationError("independent verifier report schema mismatch")
    if report.get("status") != "VERIFY_OK":
        raise ResultValidationError("independent verifier did not return VERIFY_OK")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ResultValidationError("independent verifier report contains a failed or incomplete check")
    fields = report.get("verified_fields")
    if not isinstance(fields, dict):
        raise ResultValidationError("independent verifier report has no verified_fields")
    required = (
        "status", "protocol_id", "proof_backend_id", "dataset", "backend",
        "full_dataset", "proof_scope", "full_gat_forward_proved",
        "final_logits_bound", "privacy_extension_profile_id",
        "privacy_profile_status", "verifier_reads_witness",
        "witness_included_in_proof", "public_output_digest",
        "proof_contribution", "gpu_hot_path", "proof_schema_scan",
        "privacy_masking_verified", "verifier_input_boundary",
        "model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id",
        "L", "hidden_profile", "d_in_profile", "K_out", "C",
    )
    missing = [key for key in required if key not in fields]
    if missing:
        raise ResultValidationError(f"independent verifier verified_fields missing: {missing}")
    if fields.get("status") != report.get("status"):
        raise ResultValidationError("verified status is not report-derived")
    if fields.get("protocol_id") != PROTOCOL_ID or fields.get("proof_backend_id") != PROOF_BACKEND_ID:
        raise ResultValidationError("independent verifier protocol identity mismatch")
    if fields.get("backend") not in BACKEND_SET or fields.get("dataset") not in DATASET_SET:
        raise ResultValidationError("independent verifier dataset/backend identity invalid")
    for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id"):
        if not isinstance(fields.get(key), str) or not fields[key]:
            raise ResultValidationError(f"independent verifier {key} is missing")
    for key in ("L", "K_out", "C"):
        value = fields.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ResultValidationError(f"independent verifier {key} is invalid")
    if not isinstance(fields.get("hidden_profile"), list) or not fields["hidden_profile"]:
        raise ResultValidationError("independent verifier hidden_profile is missing")
    if not isinstance(fields.get("d_in_profile"), list) or not fields["d_in_profile"]:
        raise ResultValidationError("independent verifier d_in_profile is missing")
    dataset = str(fields.get("dataset", ""))
    if fields.get("quant_cfg_id") != CANONICAL_QUANT_CFG_ID or fields.get("input_preprocess_id") != canonical_input_preprocess_id(dataset):
        raise ResultValidationError("independent verifier numerical/preprocessing identity mismatch")
    for key in ("full_dataset", "full_gat_forward_proved", "final_logits_bound", "verifier_reads_witness", "witness_included_in_proof", "gpu_hot_path"):
        _require_bool(fields, key, "verified_fields")
    for key in ("proof_schema_scan", "privacy_masking_verified"):
        _require_bool(fields, key, "verified_fields")
    boundary = fields.get("verifier_input_boundary")
    if not isinstance(boundary, dict) or boundary.get("private_input_files_read") != []:
        raise ResultValidationError("independent verifier input boundary is not witness-free")
    contribution = fields.get("proof_contribution")
    if not isinstance(contribution, dict) or not contribution or not all(value is True for value in contribution.values()):
        raise ResultValidationError("independent verifier proof contribution is incomplete")
    if not isinstance(fields.get("proof_scope"), str) or not fields["proof_scope"]:
        raise ResultValidationError("independent verifier proof scope is missing")
    if not isinstance(fields.get("privacy_extension_profile_id"), str) or not fields["privacy_extension_profile_id"]:
        raise ResultValidationError("independent verifier privacy profile is missing")
    if not isinstance(fields.get("privacy_profile_status"), str) or not fields["privacy_profile_status"]:
        raise ResultValidationError("independent verifier privacy status is missing")
    return dict(fields)


def validate_artifact(
    root: pathlib.Path,
    manifest_path: pathlib.Path,
    expected_dataset: str | None = None,
    expected_backend: str | None = None,
    *,
    allow_test_fixture: bool = False,
) -> Dict[str, Any]:
    """Validate one production artifact directory and return a result row."""
    root = _baseline_lexical(root)
    if allow_test_fixture:
        manifest_path = resolve_inside(root, manifest_path, "run manifest")
        manifest = read_object(manifest_path, "run manifest")
    else:
        artifact_dir, manifest = load_canonical_manifest(root, manifest_path)
        manifest_path = artifact_dir / RUN_MANIFEST_NAME
    schema = manifest.get("schema")
    fixture = schema == "zkGPT_test_fixture_v1"
    if fixture:
        if not allow_test_fixture:
            raise ResultValidationError("test-only fixture schema is not accepted by production validation")
        if manifest.get("status") != "TEST_FIXTURE_VALID":
            raise ResultValidationError("test fixture status is invalid")
        return dict(manifest)
    if schema in TEST_ONLY_SCHEMAS or (isinstance(schema, str) and schema.startswith("zkGPT_test_")):
        raise ResultValidationError(f"test-only schema is not accepted by production validation: {schema}")
    if manifest.get("status") == "FAILED_DIAGNOSTIC" or "FAILED_DIAGNOSTIC" in str(schema):
        raise ResultValidationError("FAILED_DIAGNOSTIC artifacts are never accepted by production validation")
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ResultValidationError("run manifest schema mismatch")
    dataset = str(manifest.get("dataset", ""))
    backend = str(manifest.get("backend", ""))
    if dataset not in DATASET_SET or backend not in BACKEND_SET:
        raise ResultValidationError("run manifest dataset/backend is invalid")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ResultValidationError(f"run manifest dataset mismatch: {dataset} != {expected_dataset}")
    if expected_backend is not None and backend != expected_backend:
        raise ResultValidationError(f"run manifest backend mismatch: {backend} != {expected_backend}")
    if allow_test_fixture:
        artifact_dir = resolve_inside(root, manifest.get("artifact_path", ""), "artifact_path")
    else:
        artifact_dir = require_canonical_artifact_dir(root, manifest_path.parent)
    if artifact_dir != manifest_path.parent:
        raise ResultValidationError("run manifest artifact_path does not match its directory")
    if not allow_test_fixture:
        validate_required_artifacts(artifact_dir, backend, include_manifest=True)
    if not allow_test_fixture:
        for duplicate in FORBIDDEN_DUPLICATE_ARTIFACTS:
            if (artifact_dir / duplicate).exists():
                raise ResultValidationError(f"duplicate non-canonical artifact is forbidden: {duplicate}")
    # Load and type-check metadata before any independent-report binding can
    # inspect it.  Missing or malformed metadata must fail closed rather than
    # reaching an unbound-local path in the cross-artifact checks below.
    metadata = read_object(required_artifact_file(artifact_dir, "metadata.json"), "metadata")
    independent_path = required_artifact_file(artifact_dir, "independent_verification.json")
    report = read_object(independent_path, "independent verifier report")
    fields = _verified_fields(report)
    for key in ("dataset", "backend"):
        if fields.get(key) != manifest.get(key):
            raise ResultValidationError(f"manifest {key} is not derived from independent report")
    for key in (
        "status", "protocol_id", "proof_backend_id", "full_dataset",
        "proof_scope", "full_gat_forward_proved", "final_logits_bound",
        "privacy_extension_profile_id", "privacy_profile_status",
        "verifier_reads_witness", "witness_included_in_proof",
        "public_output_digest", "proof_contribution", "gpu_hot_path",
        "proof_schema_scan", "privacy_masking_verified", "verifier_input_boundary",
    ):
        if manifest.get(key) != fields.get(key):
            raise ResultValidationError(f"manifest {key} does not match independent verifier report")
    dim_cfg = metadata.get("dim_cfg")
    if not isinstance(dim_cfg, Mapping):
        raise ResultValidationError("metadata dim_cfg is missing")
    for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id"):
        if fields.get(key) != metadata.get(key):
            raise ResultValidationError(f"independent verifier {key} does not match metadata")
    for key in ("L", "hidden_profile", "d_in_profile", "K_out", "C"):
        if fields.get(key) != dim_cfg.get(key):
            raise ResultValidationError(f"independent verifier {key} does not match metadata dimensions")
    if manifest.get("status") != "VERIFY_OK" or fields.get("status") != "VERIFY_OK":
        raise ResultValidationError("production result status is not verifier-derived VERIFY_OK")
    if manifest.get("full_dataset") is not True:
        raise ResultValidationError("formal result is not marked full_dataset")
    if manifest.get("verifier_reads_witness") is not False or manifest.get("witness_included_in_proof") is not False:
        raise ResultValidationError("privacy/witness boundary is invalid")
    if manifest.get("proof_schema_scan") is not True or manifest.get("privacy_masking_verified") is not True:
        raise ResultValidationError("independent proof schema/privacy scan did not pass")
    expected_backend_record = {
        "backend": backend,
        "status": "VERIFY_OK",
        "gpu_hot_path": backend == "gpu",
    }
    if manifest.get("gpu_backend") != expected_backend_record:
        raise ResultValidationError("manifest backend/evidence record is not derived from independent verification")
    if manifest.get("independent_verification_status") != report.get("status"):
        raise ResultValidationError("independent_verification_status mismatch")
    if manifest.get("independent_verification_sha256") != sha256_file(independent_path):
        raise ResultValidationError("independent verifier report digest mismatch")
    if manifest.get("independent_report_digest") != digest_json(report):
        raise ResultValidationError("independent verifier report binding digest mismatch")

    required_names = list(required_artifact_names(backend))
    recorded = manifest.get("artifact_digests")
    if not isinstance(recorded, dict) or set(recorded) != set(required_names):
        raise ResultValidationError("artifact digest inventory is incomplete or has extra files")
    for name in required_names:
        path = required_artifact_file(artifact_dir, name)
        actual = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if recorded.get(name) != actual:
            raise ResultValidationError(f"artifact digest mismatch: {name}")

    proof_payload = read_object(required_artifact_file(artifact_dir, "proof_payload.json"), "proof payload")
    fixed_lookup_tables = read_object(required_artifact_file(artifact_dir, "fixed_lookup_tables.json"), "fixed lookup tables")
    lasso = read_object(required_artifact_file(artifact_dir, "lookup_proof.json"), "lookup proof")
    transcript = read_object(required_artifact_file(artifact_dir, "transcript.json"), "transcript")
    verification = read_object(required_artifact_file(artifact_dir, "verification.json"), "verification report")
    witness = read_object(required_artifact_file(artifact_dir, "witness_manifest.json"), "witness manifest")
    benchmark = read_object(required_artifact_file(artifact_dir, "benchmark_summary.json"), "benchmark summary")
    if canonical_dataset_name(str(metadata.get("dataset", ""))) != dataset or metadata.get("backend") != backend:
        raise ResultValidationError("metadata dataset/backend mismatch")
    if metadata.get("protocol_id") != PROTOCOL_ID or metadata.get("proof_backend_id") != PROOF_BACKEND_ID:
        raise ResultValidationError("metadata protocol identity mismatch")
    if verification.get("status") != "VERIFY_OK":
        raise ResultValidationError("producer verifier report is not successful")
    if witness.get("witness_generated") is not True or "raw witness" not in str(witness.get("storage_policy", "")):
        raise ResultValidationError("witness manifest does not describe hidden non-raw witness storage")
    if proof_payload.get("Y_digest") != fields.get("public_output_digest"):
        raise ResultValidationError("proof output digest is not bound to verifier report")
    if proof_payload.get("M_pub_digest") != digest_json(metadata):
        raise ResultValidationError("proof public metadata digest mismatch")
    for key in ("Pi_Lasso", "Pi_MatMul", "Pi_GKR", "Pi_combine", "Open_in", "PrivacyProfile", "privacy_profile_digest", "component_artifacts", "transcript_digest"):
        if key not in proof_payload:
            raise ResultValidationError(f"proof contribution is missing: {key}")
    privacy_check = verify_privacy_profile(
        metadata,
        read_object(required_artifact_file(artifact_dir, "commitments.json"), "commitments"),
        proof_payload,
        pre_profile_transcript_digest=str(transcript.get("pre_profile_transcript_digest", "")),
    )
    if privacy_check.get("status") != "VERIFY_OK":
        raise ResultValidationError(f"privacy profile evidence validation failed: {privacy_check.get('reason', 'unknown')}")
    if transcript.get("privacy_profile_digest") != proof_payload.get("privacy_profile_digest"):
        raise ResultValidationError("transcript privacy profile digest mismatch")
    refs = proof_payload.get("component_artifacts")
    if not isinstance(refs, dict) or not refs:
        raise ResultValidationError("proof component artifact references are missing")
    for label, ref in refs.items():
        if not isinstance(ref, dict):
            raise ResultValidationError(f"proof component reference is malformed: {label}")
        ref_path = resolve_inside(root, ref.get("path", ""), f"proof component {label}")
        try:
            ref_path = required_artifact_file(artifact_dir, ref_path.name)
        except ResultValidationError as exc:
            raise ResultValidationError(f"proof component {label} is not a canonical artifact file") from exc
        if ref_path.parent != artifact_dir:
            raise ResultValidationError(f"proof component {label} escapes artifact directory")
        if ref.get("sha256") != sha256_file(ref_path) or int(ref.get("size_bytes", -1)) != ref_path.stat().st_size:
            raise ResultValidationError(f"proof component digest mismatch: {label}")

    config_path = resolve_inside(root, manifest.get("config", ""), "config")
    config_id_path = resolve_inside(root, read_object(required_artifact_file(artifact_dir, "config_id.json"), "config identity").get("path", ""), "config identity path")
    if config_id_path != config_path or not config_path.is_file():
        raise ResultValidationError("config identity is not baseline-local or does not match manifest")
    data_id = read_object(required_artifact_file(artifact_dir, "data_id.json"), "data identity")
    model_id = read_object(required_artifact_file(artifact_dir, "model_id.json"), "model identity")
    config_id = read_object(required_artifact_file(artifact_dir, "config_id.json"), "config identity")
    for identity, label in ((data_id, "data"), (model_id, "model")):
        identity_copy = dict(identity)
        identifier = identity_copy.pop("id", None)
        if not isinstance(identifier, str) or identifier != digest_json(identity_copy):
            raise ResultValidationError(f"{label} identity digest mismatch")
    if config_id.get("id") != config_id.get("sha256") or config_id.get("sha256") != sha256_file(config_path):
        raise ResultValidationError("config identity digest mismatch")
    _validate_formal_model_binding(root, config_path, metadata, model_id, manifest)
    expected_registry = canonical_registry_public_path(
        root, _config_value(config_path, "model_param_id", "")
    )
    metadata_registry_path = metadata.get("model_registry", {}).get("public_path", "")
    if metadata_registry_path != relative_path(root, expected_registry):
        raise ResultValidationError("metadata model registry path is not canonical for config identity")
    for key in ("cache_dir", "checkpoint_bundle"):
        raw = data_id.get(key) if key == "cache_dir" else model_id.get(key)
        path = resolve_inside(root, raw or "", f"{label} resource path")
        if not path.exists():
            raise ResultValidationError(f"identity resource path is missing: {path}")
    resource_manifest = resolve_inside(root, _config_value(config_path, "resource_manifest", "resources/resource_manifest.json"), "resource_manifest")
    static_table = resolve_inside(root, _config_value(config_path, "static_table_path", "resources/static_tables/exact_integer_gat_v1.json"), "static_table_path")
    if not resource_manifest.is_file() or not static_table.is_file():
        raise ResultValidationError("baseline-local resource manifest/static table is missing")
    try:
        resource_payload = read_resource_manifest(root)
        static_payload = validate_static_table_descriptor(root)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(f"baseline resource/static-table identity invalid: {exc}") from exc
    canonical_static_path = resolve_inside(root, STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH, "static_table_path")
    if resource_manifest != resolve_inside(root, "resources/resource_manifest.json", "resource_manifest") or static_table != canonical_static_path:
        raise ResultValidationError("config resource paths are not canonical baseline-local paths")
    if resource_payload.get("datasets", {}).get(dataset, {}).get("status") == "rejected":
        raise ResultValidationError("resource manifest rejects this dataset")
    if static_payload.get("static_tables_id") != _config_value(config_path, "static_tables_id", _config_value(config_path, "static_table_id", "")):
        raise ResultValidationError("config/static-table descriptor identity mismatch")
    descriptor_path = _config_value(config_path, "static_table_descriptor_path", STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH)
    if resolve_inside(root, descriptor_path, "static_table_descriptor_path") != canonical_static_path:
        raise ResultValidationError("config static-table descriptor path is not canonical")
    if _config_value(config_path, "static_table_descriptor_digest", "") != static_payload.get("descriptor_digest"):
        raise ResultValidationError("config static-table descriptor digest mismatch")
    if metadata.get("static_table_descriptor_schema") != STATIC_TABLE_DESCRIPTOR_SCHEMA or metadata.get("static_table_content_digest") != static_payload.get("content_digest") or metadata.get("static_table_descriptor_digest") != static_payload.get("descriptor_digest") or metadata.get("static_table_descriptor_path") != STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH:
        raise ResultValidationError("producer metadata is not bound to canonical static-table descriptor")
    _validate_static_table_closure(root, metadata, proof_payload, fixed_lookup_tables, lasso)

    timing_keys = ("commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes")
    for key in timing_keys:
        value = benchmark.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ResultValidationError(f"benchmark field is invalid: {key}")
    proof_size = len(canonical_json(proof_payload).encode("utf-8"))
    if int(benchmark.get("proof_size_bytes", -1)) != proof_size or int(manifest.get("proof_size_bytes", -1)) != proof_size:
        raise ResultValidationError("proof size accounting mismatch")
    if manifest.get("timings_ms") != {key: benchmark[key] for key in ("commitment_time_ms", "prove_time_ms", "verify_time_ms")}:
        raise ResultValidationError("timing summary mismatch")
    if manifest.get("proof_digest") != sha256_file(required_artifact_file(artifact_dir, "proof_payload.json")):
        raise ResultValidationError("proof digest mismatch")

    gpu_errors: list[str] = []
    if backend == "gpu":
        gpu_errors = _validate_gpu_artifact_closure(artifact_dir, metadata, transcript, proof_payload)
        if gpu_errors or fields.get("gpu_hot_path") is not True:
            raise ResultValidationError("GPU execution evidence is incomplete: " + ";".join(gpu_errors))
    elif fields.get("gpu_hot_path") is not False:
        raise ResultValidationError("CPU report has an unexpected GPU proof contribution")

    if manifest.get("run_digest") != manifest_run_digest(manifest):
        raise ResultValidationError("run manifest digest mismatch")
    return {
        "dataset": dataset,
        "backend": backend,
        "status": "VERIFY_OK",
        "proof_generated": True,
        "verified": True,
        "formal_experiment_eligible": True,
        "artifact_path": relative_path(root, artifact_dir),
        "run_digest": str(manifest.get("run_digest")),
        "proof_digest": str(manifest.get("proof_digest")),
        "proof_size_bytes": int(manifest["proof_size_bytes"]),
        "timings_ms": dict(manifest["timings_ms"]),
        "gpu_contract_errors": ";".join(gpu_errors),
        "manifest": manifest,
    }


def _config_value(path: pathlib.Path, key: str, default: str) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip()
    return default


def _validate_static_table_closure(
    root: pathlib.Path,
    metadata: Mapping[str, Any],
    proof_payload: Mapping[str, Any],
    fixed_lookup_tables: Mapping[str, Any],
    lasso: Mapping[str, Any],
) -> None:
    """Require the serialized fixed-table file to match every proof claim."""
    try:
        descriptor = validate_static_table_descriptor(root)
    except (OSError, ValueError, TypeError) as exc:
        raise ResultValidationError(f"canonical static-table descriptor invalid: {exc}") from exc
    if not isinstance(fixed_lookup_tables, dict):
        raise ResultValidationError("fixed lookup table artifact is not an object")
    if proof_payload.get("FixedLookupTables") != fixed_lookup_tables:
        raise ResultValidationError("proof payload fixed lookup tables mismatch")
    if metadata.get("static_table_registry") != fixed_lookup_tables:
        raise ResultValidationError("metadata fixed lookup tables mismatch")
    if metadata.get("static_table_registry_digest") != digest_json(fixed_lookup_tables):
        raise ResultValidationError("fixed lookup table registry digest mismatch")
    if lasso.get("fixed_table_registry") != fixed_lookup_tables:
        raise ResultValidationError("lookup proof fixed-table registry mismatch")
    if lasso.get("fixed_table_registry_digest") != digest_json(fixed_lookup_tables):
        raise ResultValidationError("lookup proof fixed-table registry digest mismatch")
    if metadata.get("static_table_descriptor_schema") != descriptor.get("schema") or metadata.get("static_table_content_digest") != descriptor.get("content_digest") or metadata.get("static_table_descriptor_digest") != descriptor.get("descriptor_digest") or metadata.get("static_table_descriptor_path") != STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH:
        raise ResultValidationError("metadata is not bound to canonical static-table descriptor")
    expected = {
        "static_table_id": descriptor.get("static_tables_id"),
        "quant_cfg_id": descriptor.get("quant_cfg_id"),
        "static_table_descriptor_schema": descriptor.get("schema"),
        "static_table_content_digest": descriptor.get("content_digest"),
        "static_table_descriptor_digest": descriptor.get("descriptor_digest"),
    }
    if set(fixed_lookup_tables) != {"T_LReLU", "T_ELU", "T_exp", "T_range"}:
        raise ResultValidationError("fixed lookup table family set is not canonical")
    for table_id, table in fixed_lookup_tables.items():
        if not isinstance(table, Mapping) or table.get("table_id") != table_id:
            raise ResultValidationError(f"fixed lookup table {table_id} is malformed")
        if any(table.get(key) != value for key, value in expected.items()):
            raise ResultValidationError(f"fixed lookup table {table_id} descriptor mismatch")


def _validate_gpu_artifact_closure(
    artifact_dir: pathlib.Path,
    metadata: Mapping[str, Any],
    transcript: Mapping[str, Any],
    proof_payload: Mapping[str, Any],
) -> list[str]:
    """Return GPU evidence errors, including all three-record equality."""
    hot_path = read_object(required_artifact_file(artifact_dir, "gpu_proof_hot_path.json"), "GPU proof hot path")
    backend = read_object(required_artifact_file(artifact_dir, "gpu_backend.json"), "GPU backend evidence")
    contribution = proof_payload.get("GPU_proof_contribution")
    errors: list[str] = []
    if not isinstance(contribution, dict):
        errors.append("proof payload GPU contribution is missing")
    elif contribution != hot_path or contribution != backend:
        errors.append("GPU proof contribution/hot-path/backend records differ")
    valid, validation_errors = validate_gpu_execution_evidence(
        hot_path,
        session_nonce=str(metadata.get("session_nonce", "")),
        final_transcript_digest=str(transcript.get("final_digest", "")),
    )
    if not valid:
        errors.extend(validation_errors)
    return errors


def validate_matrix_rows(rows: Iterable[Mapping[str, Any]], *, allow_test_fixture: bool = False) -> None:
    values = list(rows)
    if any(
        row.get("status") == "FAILED_DIAGNOSTIC"
        or "FAILED_DIAGNOSTIC" in str(row.get("schema", ""))
        for row in values
    ):
        raise ResultValidationError("FAILED_DIAGNOSTIC rows are never accepted")
    if allow_test_fixture:
        if len(values) != 10:
            raise ResultValidationError("test fixture matrix must contain ten rows")
        pairs = {(row.get("backend"), row.get("dataset")) for row in values}
        expected = {(backend, dataset) for backend in ("cpu", "gpu") for dataset in DATASET_ORDER}
        if pairs != expected:
            raise ResultValidationError("test fixture matrix coverage mismatch")
        return
    if len(values) != 10:
        raise ResultValidationError("formal matrix must contain exactly ten rows")
    pairs = {(row.get("backend"), row.get("dataset")) for row in values}
    expected = {(backend, dataset) for backend in ("cpu", "gpu") for dataset in DATASET_ORDER}
    if pairs != expected:
        raise ResultValidationError("formal matrix dataset/backend coverage mismatch")
    for row in values:
        expected_artifact = str(
            pathlib.Path("results")
            / "runs"
            / str(row.get("backend", ""))
            / str(row.get("dataset", ""))
        )
        if row.get("artifact_path") != expected_artifact:
            raise ResultValidationError("formal matrix row artifact path is not canonical")
    if any(row.get("status") != "VERIFY_OK" or row.get("verified") is not True for row in values):
        raise ResultValidationError("formal matrix contains an unverified row")
