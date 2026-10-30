#!/usr/bin/env python3
"""Authenticated EZKL resource evidence for the five-dataset formal runner."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from formal_common import (
    DATASETS, ROOT, SRS_MANIFEST, AuthenticatedFormalContext, ContractError, abort,
    canonical_json_digest, load_json, local_path, require_authenticated_formal_context,
    resource_file_identities, resource_manifest_hash, sha256, source_tree_hash,
    selected_datasets_for_session,
)
from compatible_resume import (
    compare_admission_semantics, validate_compatible_resume_request,
    validate_original_session_snapshot, validate_retained_setup_chain,
)

_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_SCHEMA = "ezkl_capacity_source_identity_v2"
_REPORT_SCHEMA = "ezkl.resource_capacity_precheck.v2"
_COMPAT_PROVENANCE_SCHEMA = "ezkl.compatible-capacity-provenance.v1"
_COMPAT_DATASETS = ("ppi", "ogbn_arxiv")
_FORMAT_ID = "halo2_native_pk_v4_vk_fixed_values_permutation_no_fixed_polys"
_FORMULA_ID = "native_pk_v4_fixed_payload_lower_bound_unavailable_from_admission_v1"
_FIELD_BYTES = 32
_FORMAT_SOURCES = (
    "vendor/halo2/halo2_proofs/src/plonk.rs",
    "vendor/halo2/halo2_proofs/src/plonk/keygen.rs",
    "vendor/halo2/halo2_proofs/src/helpers.rs",
)
_SETUP_FILES = (
    "setup_timing.json", "admission.json", "settings.json", "compiled.circuit", "pk.key", "vk.key",
    "native_keygen_budget.json", "native_keygen_receipt.json", "gat_static_tables.json",
    "gat_integer_ledger.json", "circuit_ref.json", "producer_context.json",
)
_COMPAT_PROVENANCE_KEYS = {
    "schema", "session_id", "compatible_resume_request", "old_session_snapshot",
    "old_resume_identity_sha256", "old_setup_identity_sha256", "resume_identity",
    "resume_identity_sha256", "capacity_source_identity", "capacity_source_identity_sha256",
    "native_admissions", "admission_root", "retained_setup",
}
_NATIVE_ADMISSION_RECEIPT_KEYS = {
    "path", "sha256", "prepared_dir", "settings_sha256", "scalar_domain_vector_bytes",
    "total_assignments", "logrows", "domain_size", "constraint_system",
    "native_memory_model", "relation_identity",
}
_TERMINAL = {"FAILED", "INTERRUPTED", "COMPLETE", "COMPLETED", "VALIDATED", "FINAL_VALIDATION_OK", "SUBSET_COMPLETE", "CAPACITY_PREFLIGHT_COMPLETE"}
_ESTIMATE_FIELDS = (
    "setup_key_artifact_upper_bytes", "setup_mapped_peak_bytes", "proof_spill_known_bytes",
    "setup_keygen_host_peak_bytes", "proof_host_peak_bytes", "host_known_bytes",
    "gat_transient_witness_bytes_upper", "native_pk_read_backing_required_bytes",
)


def _fail_symlink(path: Path) -> None:
    try:
        path.relative_to(ROOT)
    except ValueError:
        abort(f"resource-capacity path escapes the EZKL baseline: {path}")
    for item in (path, *path.parents):
        if item == ROOT.parent:
            break
        if item.exists() and item.is_symlink():
            abort(f"resource-capacity path contains a symlink: {item}")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > sys.maxsize:
        abort(f"resource-capacity {label} is invalid or unrepresentable")
    return value


def capacity_source_identity() -> dict[str, Any]:
    """Snapshot geometry, native PK serialization, secure configs, and resources."""
    geometry: list[list[str]] = []
    for directory in (ROOT / "src", ROOT / "gat_adapter"):
        if directory.is_symlink() or not directory.is_dir():
            abort(f"resource-capacity geometry source directory is missing: {directory}")
        files = sorted(
            (path for path in directory.rglob("*.rs") if path.is_file() and not path.is_symlink()),
            key=lambda path: path.relative_to(ROOT).as_posix(),
        )
        if not files:
            abort(f"resource-capacity geometry source directory is empty: {directory}")
        geometry.extend([[path.relative_to(ROOT).as_posix(), sha256(path)] for path in files])
    native_format: list[list[str]] = []
    for relative in _FORMAT_SOURCES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            abort(f"resource-capacity native PK source is missing: {relative}")
        native_format.append([relative, sha256(path)])
    configs: list[list[str]] = []
    for dataset in DATASETS:
        relative = f"resources/canonical/configs/{dataset}_secure.cfg"
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            abort(f"resource-capacity secure config is missing: {relative}")
        configs.append([dataset, sha256(path)])
    srs_identity = _native_srs_identity()
    return {
        "schema": _IDENTITY_SCHEMA,
        "geometry_sources_sha256": geometry,
        "native_pk_format_sources_sha256": native_format,
        "native_pk_format_id": _FORMAT_ID,
        "fixed_payload_formula_id": _FORMULA_ID,
        "serialized_field_bytes": _FIELD_BYTES,
        "secure_config_sha256s": configs,
        "resource_manifest_sha256": resource_manifest_hash(),
        "resource_file_identities": [[name, list(identity)] for name, identity in resource_file_identities()],
        "native_srs_manifest_sha256": srs_identity["manifest_sha256"],
        "native_srs_identity": {key: value for key, value in srs_identity.items() if key != "manifest_sha256"},
        "source_tree_sha256": source_tree_hash(),
    }


def _identity_matches(saved: Any, current: dict[str, Any]) -> bool:
    return isinstance(saved, dict) and saved.get("schema") == _IDENTITY_SCHEMA and saved == current


def _setup_files_match(setup_files: Any) -> bool:
    return isinstance(setup_files, dict) and set(setup_files) == set(_SETUP_FILES)


def _build_receipt_matches_identity(build: Any, resume: Any) -> bool:
    if not isinstance(build, dict) or not isinstance(resume, dict):
        return False
    return (
        build.get("baseline") == "ezkl" and build.get("backend") == "gpu"
        and all(build.get(key) == resume.get(key) for key in (
            "binary_path", "binary_sha256", "source_tree_sha256", "resource_manifest_sha256",
            "build_environment_sha256",
        ))
    )


def _require_fresh_session_binding(
    document: dict[str, Any], authenticated_context: AuthenticatedFormalContext | None,
) -> dict[str, Any]:
    session_id = document.get("session_id")
    resume = document.get("resume_identity")
    if not isinstance(resume, dict) or resume.get("schema") != "ezkl-formal-resume-identity-v1":
        abort("fresh capacity gate lacks the current session resume identity")
    if not isinstance(authenticated_context, AuthenticatedFormalContext):
        abort("fresh capacity gate requires the runner's authenticated formal context")
    context = authenticated_context
    expected_configs = resume.get("config_sha256s")
    context_configs = [[dataset, digest] for dataset, digest in context.config_sha256s]
    if (
        context.session_id != session_id
        or context.source_tree_sha256 != resume.get("source_tree_sha256")
        or context.resource_manifest_sha256 != resume.get("resource_manifest_sha256")
        or context.srs_manifest_sha256 != resume.get("srs_manifest_sha256")
        or context.srs_path != resume.get("srs_path")
        or context.srs_sha256 != resume.get("srs_sha256")
        or context_configs != expected_configs
        or any(document.get(key) != resume.get(key) for key in (
            "source_tree_sha256", "resource_manifest_sha256", "srs_manifest_sha256",
            "binary_sha256", "build_receipt_sha256", "build_environment_sha256",
        ))
    ):
        abort("fresh session, resume identity, and authenticated formal context disagree")
    return resume


def _authenticate_fresh_provenance(
    document: dict[str, Any], authenticated_context: AuthenticatedFormalContext | None,
) -> dict[str, Any]:
    resume = _require_fresh_session_binding(document, authenticated_context)
    current_identity = capacity_source_identity()
    if not _identity_matches(document.get("capacity_source_identity"), current_identity):
        abort("fresh capacity source identity is absent or changed since the session snapshot")
    for key, identity_key in (
        ("source_tree_sha256", "source_tree_sha256"),
        ("resource_manifest_sha256", "resource_manifest_sha256"),
        ("srs_manifest_sha256", "native_srs_manifest_sha256"),
    ):
        if resume.get(key) != current_identity.get(identity_key):
            abort(f"fresh resume identity differs from current source/resource identity: {key}")
    current_configs = current_identity["secure_config_sha256s"]
    if resume.get("config_sha256s") != current_configs:
        abort("fresh resume identity differs from the current five secure configs")
    context = authenticated_context
    if not isinstance(context, AuthenticatedFormalContext):
        abort("fresh provenance has no trusted authenticated formal context")
    srs_identity = current_identity["native_srs_identity"]
    if (
        srs_identity.get("path") != context.srs_path
        or srs_identity.get("sha256") != context.srs_sha256
        or srs_identity.get("file_identity") != list(context.srs_file_identity)
    ):
        abort("fresh SRS manifest/file metadata differs from the authenticated formal context")
    first_config = ROOT / "resources/canonical/configs" / f"{DATASETS[0]}_secure.cfg"
    require_authenticated_formal_context(
        context, session_id=document["session_id"], dataset=DATASETS[0], config=first_config,
    )

    build_path = ROOT / "artifacts/build/gpu/build_receipt.json"
    binary_path = local_path(resume.get("binary_path"), "fresh authenticated native binary")
    if (
        build_path.is_symlink() or not build_path.is_file()
        or sha256(build_path) != resume.get("build_receipt_sha256")
        or binary_path.is_symlink() or not binary_path.is_file()
        or sha256(binary_path) != resume.get("binary_sha256")
    ):
        abort("fresh current build receipt or native binary differs from its session identity")
    build = load_json(build_path, "fresh current GPU build receipt")
    if not _build_receipt_matches_identity(build, resume):
        abort("fresh current GPU build receipt fields differ from the session resume identity")
    return {
        "capacity_source_identity": current_identity,
        "resume_identity_sha256": canonical_json_digest(resume),
        "build_receipt_sha256": resume["build_receipt_sha256"],
        "binary_path": resume["binary_path"],
        "binary_sha256": resume["binary_sha256"],
        "source_tree_sha256": resume["source_tree_sha256"],
        "resource_manifest_sha256": resume["resource_manifest_sha256"],
        "config_sha256s": current_configs,
        "authenticated_formal_context": {
            "session_id": context.session_id,
            "srs_manifest_sha256": context.srs_manifest_sha256,
            "srs_path": context.srs_path,
            "srs_expected_sha256": context.srs_sha256,
            "srs_sha256_evidence": "digest verified by the runner's native SRS authentication before context creation; current manifest and file metadata rechecked without rehashing the SRS",
            "srs_file_identity": list(context.srs_file_identity),
        },
    }


def _native_srs_identity() -> dict[str, Any]:
    manifest = load_json(SRS_MANIFEST, "native SRS manifest")
    entries = manifest.get("entries")
    if manifest.get("baseline") != "ezkl" or not isinstance(entries, dict) or set(entries) != set(DATASETS):
        abort("resource-capacity native SRS manifest has an invalid baseline or dataset set")
    first = entries[DATASETS[0]]
    if not isinstance(first, dict):
        abort("resource-capacity native SRS manifest entry is malformed")
    identity = (first.get("path"), first.get("sha256"), first.get("size"), first.get("srs_id"))
    if (
        not isinstance(identity[0], str) or not isinstance(identity[1], str) or _SHA256.fullmatch(identity[1]) is None
        or not isinstance(identity[2], int) or isinstance(identity[2], bool) or identity[2] <= 0
        or not isinstance(identity[3], str) or not identity[3]
        or any(
            not isinstance(entries[dataset], dict)
            or (entries[dataset].get("path"), entries[dataset].get("sha256"), entries[dataset].get("size"), entries[dataset].get("srs_id")) != identity
            for dataset in DATASETS
        )
    ):
        abort("resource-capacity SRS entries do not share one authenticated identity")
    path = local_path(identity[0], "resource-capacity official SRS")
    _fail_symlink(path)
    if not path.is_file() or path.stat().st_size != identity[2]:
        abort("resource-capacity official SRS file is missing or differs from manifest size")
    stat = path.stat()
    return {
        "manifest_sha256": sha256(SRS_MANIFEST),
        "path": identity[0], "sha256": identity[1], "size_bytes": identity[2], "srs_id": identity[3],
        "sha256_evidence": "manifest-declared expected digest; not rehashed by resource-capacity precheck",
        "file_identity": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
    }


def _fixed_payload_bounds(admission: dict[str, Any]) -> dict[str, Any]:
    dataset = admission.get("dataset")
    cs, memory = admission.get("constraint_system"), admission.get("native_memory_model")
    if dataset not in DATASETS or not isinstance(cs, dict) or not isinstance(memory, dict):
        abort("resource-capacity Admission lacks its canonical dataset/geometry")
    fixed = _integer(cs.get("num_fixed_columns"), f"{dataset} fixed columns", minimum=1)
    selectors = _integer(cs.get("num_selectors"), f"{dataset} virtual selectors")
    selector_columns = _integer(cs.get("selector_fixed_column_count_upper"), f"{dataset} compressed selector columns")
    degree = _integer(cs.get("degree"), f"{dataset} degree", minimum=1)
    rows = _integer(admission.get("domain_size"), f"{dataset} domain", minimum=1)
    field_bytes = _integer(memory.get("size_of_fr_bytes"), f"{dataset} field width", minimum=1)
    if field_bytes != _FIELD_BYTES:
        abort(f"{dataset} Admission field width differs from the authenticated native format")
    if selector_columns > selectors:
        abort(f"{dataset} witnessless selector census exceeds the configured virtual-selector count")
    old_column_field = _integer(cs.get("fixed_column_count_post_compression"), f"{dataset} fixed-column count upper bound", minimum=1)
    source_upper_columns = fixed + selector_columns
    if old_column_field != source_upper_columns:
        abort(f"{dataset} fixed-column count disagrees with the witnessless selector census")
    return {
        "dataset": dataset,
        "fixed_column_count_post_compression": old_column_field,
        "fixed_payload_lower_bound_bytes": 0,
        "fixed_payload_lower_bound": {
            "formula_id": _FORMULA_ID,
            "format_id": _FORMAT_ID,
            "status": "unknown_from_admission_missing_per_column_encoding_and_actual_permutation_count",
            "admission_fixed_column_count_upper": source_upper_columns,
            "uncompressed_fixed_columns_upper": fixed,
            "virtual_selectors": selectors,
            "compressed_selector_columns_upper": selector_columns,
            "degree": degree,
            "domain_rows": rows,
            "serialized_field_bytes": field_bytes,
            "fixed_values_serialized_copies": 1,
            "fixed_polys_serialized": False,
            "formula": "0 bytes counted because Admission does not identify Dense versus Compact columns or actual permutation entries",
            "scope": "structural v4 PK lower bound only; fixed coefficient polynomials are reconstructed when reading the key",
        },
        "source_model_fixed_payload_upper_bound_bytes": None,
        "source_model_fixed_payload_upper_bound": {
            "status": "not_estimated_by_this_structural_lower_bound",
            "evidence_kind": "use source_model_estimates.setup_key_artifact_upper_bytes for the mandatory Setup capacity gate",
        },
    }


def _model_estimates(admission: dict[str, Any]) -> dict[str, Any]:
    # Use the runner's implementation so the report does not fork its budget formulas.
    try:
        from run_hybrid_formal import _native_capacity_requirements
    except (ImportError, AttributeError) as exc:
        abort(f"cannot load the current EZKL source-model capacity calculation: {exc}")
    memory = admission["native_memory_model"]
    requirements = _native_capacity_requirements(
        admission,
        {"scalar_bytes": memory["size_of_fr_bytes"], "affine_bytes": memory["size_of_g1_affine_bytes"]},
    )
    return {
        "evidence_kind": "source_derived_static_capacity_estimates_not_measured_peak",
        "coverage": "known source-model envelope; native census and live phase samples remain separate",
        "estimates_bytes": {key: requirements[key] for key in _ESTIMATE_FIELDS},
    }


def _sample_hardware(path: Path) -> dict[str, Any]:
    try:
        memory_rows = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        memory = {
            key: int(rest.split()[0]) * 1024
            for line in memory_rows
            if (key_rest := line.partition(":"))[1]
            for key, rest in [(key_rest[0], key_rest[2])]
            if len(rest.split()) >= 2 and rest.split()[1] == "kB"
        }
        if "MemTotal" not in memory or "MemAvailable" not in memory:
            abort("current host sample lacks MemTotal/MemAvailable")
        sample_path = path
        while not sample_path.exists():
            if sample_path == sample_path.parent:
                abort(f"no filesystem ancestor exists for output root: {path}")
            sample_path = sample_path.parent
        _fail_symlink(sample_path)
        if not sample_path.is_dir():
            sample_path = sample_path.parent
        usage = os.statvfs(sample_path)
        disk_available = int(usage.f_bavail) * int(usage.f_frsize)
        device_id = int(os.stat(sample_path).st_dev)
    except (OSError, ValueError) as exc:
        abort(f"cannot sample current host/disk capacity: {exc}")
    if disk_available < 0 or disk_available > sys.maxsize:
        abort("current filesystem returned an invalid available-byte count")
    now = datetime.now(timezone.utc).isoformat()
    return {
        "sampled_at_utc": now,
        "disk": {"path": str(sample_path), "device_id": device_id, "available_bytes": disk_available, "evidence_kind": "current_statvfs_sample"},
        "host_memory": {"total_bytes": memory["MemTotal"], "available_bytes": memory["MemAvailable"], "evidence_kind": "current_proc_meminfo_sample"},
    }


def _session_context(session_path: Path | str) -> tuple[Path, Path, Path, dict[str, Any]]:
    raw = Path(session_path)
    session_file = raw if raw.is_absolute() else ROOT / raw
    _fail_symlink(session_file)
    if not session_file.is_file():
        abort(f"formal session file is missing: {session_file}")
    session_file = session_file.resolve()
    session_id = session_file.stem
    if _SESSION_ID.fullmatch(session_id) is None or session_file.parent.name != ".sessions":
        abort("formal session path is not .sessions/<session_id>.json")
    document = load_json(session_file, "formal session record")
    if (
        document.get("schema_version") != 1 or document.get("baseline") != "ezkl"
        or document.get("session_id") != session_id
        or document.get("status") not in (_TERMINAL | {"PREFLIGHT_STARTED", "RUNNING"})
        or not isinstance(document.get("output_root"), str)
    ):
        abort("formal session has an unsupported or mismatched anchor")
    selected_datasets_for_session(document)
    output_root = local_path(document["output_root"], "formal output root")
    session_dir = output_root / ".sessions" / session_id
    if session_file != (output_root / ".sessions" / f"{session_id}.json").resolve():
        abort("formal session file does not match its output-root/session anchor")
    _fail_symlink(output_root)
    _fail_symlink(session_dir)
    return session_file, output_root, session_dir, document


def _admission_paths(session_dir: Path, dataset: str, admission_root: Path | None = None) -> tuple[Path, Path]:
    folder = (admission_root or (session_dir / "admission")) / dataset
    return folder / "admission.json", folder / "settings.json"


def _authenticated_admission(
    document: dict[str, Any], session_dir: Path, dataset: str, admission_path: Path, settings_path: Path,
    *, admission_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_admission, expected_settings = _admission_paths(session_dir, dataset, admission_root)
    _fail_symlink(expected_admission)
    _fail_symlink(expected_settings)
    if admission_path.resolve() != expected_admission.resolve() or settings_path.resolve() != expected_settings.resolve():
        abort(f"{dataset} Admission files are outside their session anchor")
    receipt_map = document.get("native_admissions")
    receipt = receipt_map.get(dataset) if isinstance(receipt_map, dict) else None
    if not isinstance(receipt, dict):
        abort(f"{dataset} session Admission receipt is missing")
    admission = load_json(admission_path, f"{dataset} native Admission")
    load_json(settings_path, f"{dataset} native settings")
    if (
        receipt.get("path") != expected_admission.relative_to(ROOT).as_posix()
        or receipt.get("prepared_dir") != expected_admission.parent.relative_to(ROOT).as_posix()
        or receipt.get("sha256") != sha256(admission_path)
        or receipt.get("settings_sha256") != sha256(settings_path)
        or admission.get("dataset") != dataset
        or admission.get("settings_sha256") != receipt.get("settings_sha256")
        or any(receipt.get(key) != admission.get(key) for key in ("constraint_system", "native_memory_model", "domain_size", "logrows", "relation_identity"))
    ):
        abort(f"{dataset} Admission/settings do not match the authenticated session receipt")
    relation = admission.get("relation_identity")
    resource_hash = document.get("resource_manifest_sha256")
    if (
        not isinstance(relation, dict) or relation.get("dataset") != dataset
        or relation.get("resource_manifest_sha256") != resource_hash
        or admission.get("resource_manifest_sha256") != resource_hash
    ):
        abort(f"{dataset} Admission relation/resource identity disagrees with the session")
    return admission, receipt


def _retained_setup(
    document: dict[str, Any], output_root: Path, dataset: str, admission_digest: str,
) -> tuple[bool, int | None]:
    anchors = document.get("setup_reuse_receipts", {})
    if not isinstance(anchors, dict):
        abort("formal setup-reuse receipt anchors are malformed")
    anchor = anchors.get(dataset)
    if anchor is None:
        return False, None
    output = output_root / dataset
    receipt_path = output / "setup_receipt.json"
    if not isinstance(anchor, str) or _SHA256.fullmatch(anchor) is None or output.is_symlink() or receipt_path.is_symlink():
        abort(f"{dataset} setup receipt anchor/artifact is malformed or symlinked")
    if not receipt_path.is_file() or sha256(receipt_path) != anchor:
        abort(f"{dataset} setup receipt is missing or differs from its session anchor")
    receipt = load_json(receipt_path, f"{dataset} setup receipt")
    resume_identity, setup_files = document.get("resume_identity"), receipt.get("setup_files")
    if (
        not isinstance(resume_identity, dict) or receipt.get("schema") != "ezkl.setup-reuse-receipt.v1"
        or receipt.get("session_id") != document.get("session_id") or receipt.get("dataset") != dataset
        or receipt.get("admission_sha256") != admission_digest
        or receipt.get("resume_identity_sha256") != canonical_json_digest(resume_identity)
        or not _setup_files_match(setup_files)
    ):
        abort(f"{dataset} setup receipt is not bound to current session and Admission")
    for name in _SETUP_FILES:
        artifact = output / name
        if artifact.is_symlink() or not artifact.is_file() or sha256(artifact) != setup_files.get(name):
            abort(f"{dataset} setup artifact failed its session-anchored digest: {name}")
    pk_size = (output / "pk.key").stat().st_size
    return True, pk_size


def _missing_row(dataset: str) -> dict[str, Any]:
    return {"dataset": dataset, "admission_status": "missing", "fixed_payload_lower_bound_bytes": None,
            "lower_bound_gate_status": "unknown_missing_admission", "setup_already_retained": False,
            "observed_pk_file_bytes": None, "source_model_estimates": None}


def _fresh_row(
    document: dict[str, Any], root: Path, session_dir: Path, dataset: str, *,
    admission_root: Path | None = None, retained_setup_override: bool | None = None,
) -> dict[str, Any]:
    admission_path, settings_path = _admission_paths(session_dir, dataset, admission_root)
    if not admission_path.exists() or not settings_path.exists():
        return _missing_row(dataset)
    admission, receipt = _authenticated_admission(
        document, session_dir, dataset, admission_path, settings_path, admission_root=admission_root,
    )
    bounds = _fixed_payload_bounds(admission)
    if retained_setup_override is None:
        retained, observed = _retained_setup(document, root, dataset, receipt["sha256"])
    else:
        retained = retained_setup_override
        pk_path = root / dataset / "pk.key"
        observed = pk_path.stat().st_size if retained and pk_path.is_file() and not pk_path.is_symlink() else None
        if retained and observed is None:
            abort(f"{dataset} compatible setup retention override has no regular PK")
    memory = admission["native_memory_model"]
    return {
        **bounds,
        "admission_status": "authenticated_current_session_admission",
        "admission_sha256": receipt["sha256"],
        "settings_sha256": receipt["settings_sha256"],
        "setup_already_retained": retained,
        "observed_pk_file_bytes": observed,
        "lower_bound_gate_status": "already_retained_authenticated_setup" if retained else "unavailable_from_admission_encoding_metadata",
        "source_model_estimates": _model_estimates(admission),
        "admission_device_sample": {
            "cuda_free_bytes_at_admission": memory.get("cuda_free_bytes_at_admission"),
            "cuda_total_memory_bytes": memory.get("cuda_total_memory_bytes"),
            "cuda_uuid": memory.get("cuda_uuid"),
            "evidence_kind": "sampled by native Admission, not by this precheck",
        },
    }


def audit_fresh_admission(
    session_path: Path | str, dataset: str, admission_path: Path | str | None,
    settings_path: Path | str | None, *, output_root: Path | str,
    authenticated_context: AuthenticatedFormalContext | None = None,
    compatible_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate one current-session Admission and return its dataset row."""
    session_file, root, session_dir, document = _session_context(session_path)
    selected_datasets = selected_datasets_for_session(document)
    supplied_root = Path(output_root)
    supplied_root = supplied_root if supplied_root.is_absolute() else local_path(supplied_root.as_posix(), "formal output root")
    if supplied_root.resolve() != root.resolve() or dataset not in selected_datasets:
        abort("fresh capacity report has an invalid output-root or dataset anchor")
    admission_root = None
    retained_setup_override = None
    if compatible_provenance is not None:
        document, admission_root, retained_setup_override = _compatible_provenance_document(
            document, session_dir, authenticated_context, compatible_provenance,
        )
    _require_fresh_session_binding(document, authenticated_context)
    if admission_path is None or settings_path is None:
        return _missing_row(dataset)
    admission_path = Path(admission_path)
    settings_path = Path(settings_path)
    admission_path = admission_path if admission_path.is_absolute() else local_path(admission_path.as_posix(), f"{dataset} Admission")
    settings_path = settings_path if settings_path.is_absolute() else local_path(settings_path.as_posix(), f"{dataset} settings")
    expected_admission, expected_settings = _admission_paths(session_dir, dataset, admission_root)
    if admission_path.resolve() != expected_admission.resolve() or settings_path.resolve() != expected_settings.resolve():
        abort(f"{dataset} Admission paths do not match the current session")
    return _fresh_row(
        document, root, session_dir, dataset, admission_root=admission_root,
        retained_setup_override=retained_setup_override.get(dataset) if isinstance(retained_setup_override, dict) else None,
    )


def _compatible_provenance_document(
    document: dict[str, Any], session_dir: Path, authenticated_context: AuthenticatedFormalContext | None,
    value: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, bool]]:
    """Validate compatible-resume evidence before making an in-memory current view."""
    if type(value) is not dict or set(value) != _COMPAT_PROVENANCE_KEYS:
        abort("compatible capacity provenance has unknown, missing, or malformed fields")
    if value.get("schema") != _COMPAT_PROVENANCE_SCHEMA:
        abort("compatible capacity provenance has an unsupported schema")
    if not isinstance(authenticated_context, AuthenticatedFormalContext):
        abort("compatible capacity audit requires the runner's authenticated current context")

    context = authenticated_context
    session_id = document.get("session_id")
    if (
        not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None
        or value.get("session_id") != session_id or context.session_id != session_id
        or selected_datasets_for_session(document) != _COMPAT_DATASETS
    ):
        abort("compatible capacity provenance is not bound to the exact current session and dataset selection")

    # The request validator is the sole source of the old session object. The
    # provenance's copy must be the same immutable request already bound there.
    request, old_session = validate_compatible_resume_request(document, expected_session_id=session_id)
    supplied_request = value.get("compatible_resume_request")
    if (
        type(supplied_request) is not dict
        or canonical_json_digest(supplied_request) != canonical_json_digest(request)
    ):
        abort("compatible capacity provenance request differs from the authenticated session request")

    snapshot_ref = value.get("old_session_snapshot")
    if type(snapshot_ref) is not dict or set(snapshot_ref) != {"path", "sha256"}:
        abort("compatible capacity original-session snapshot reference is malformed")
    snapshot_path_value = snapshot_ref.get("path")
    snapshot_sha256 = snapshot_ref.get("sha256")
    if (
        not isinstance(snapshot_path_value, str)
        or snapshot_path_value != request.get("old_session_snapshot_path")
        or not isinstance(snapshot_sha256, str) or _SHA256.fullmatch(snapshot_sha256) is None
        or snapshot_sha256 != request.get("old_session_snapshot_sha256")
    ):
        abort("compatible capacity original-session snapshot differs from the immutable request")
    snapshot_path = local_path(snapshot_path_value, "compatible original-session snapshot")
    _fail_symlink(snapshot_path)
    snapshotted_old_session, actual_snapshot_sha256 = validate_original_session_snapshot(
        snapshot_path, session_id=session_id, expected_sha256=snapshot_sha256,
    )
    if snapshotted_old_session != old_session or actual_snapshot_sha256 != snapshot_sha256:
        abort("compatible capacity original session differs from the validated immutable snapshot")

    old_resume = old_session.get("resume_identity")
    session_old_resume = document.get("resume_identity")
    old_resume_sha256 = value.get("old_resume_identity_sha256")
    if (
        not isinstance(old_resume, dict)
        or not isinstance(session_old_resume, dict)
        or canonical_json_digest(session_old_resume) != canonical_json_digest(old_resume)
        or not isinstance(old_resume_sha256, str) or _SHA256.fullmatch(old_resume_sha256) is None
        or canonical_json_digest(old_resume) != old_resume_sha256
    ):
        abort("compatible capacity provenance does not bind the original resume identity")
    old_setup_identity, actual_old_setup_sha256, old_setup_refs = validate_retained_setup_chain(
        old_session, session_id=session_id,
    )
    old_setup_sha256 = value.get("old_setup_identity_sha256")
    expected_setup_ref_keys = {
        "origin", "setup_identity_sha256", "setup_receipt", "native_keygen_budget",
        "native_keygen_receipt", "setup_files", "bound_setup_admission",
    }
    if (
        not isinstance(old_setup_sha256, str) or _SHA256.fullmatch(old_setup_sha256) is None
        or old_setup_sha256 != actual_old_setup_sha256
        or canonical_json_digest(old_setup_identity) != actual_old_setup_sha256
        or type(old_setup_refs) is not dict or set(old_setup_refs) != expected_setup_ref_keys
        or old_setup_refs.get("origin") != "retained_old"
        or old_setup_refs.get("setup_identity_sha256") != actual_old_setup_sha256
    ):
        abort("compatible capacity provenance does not bind the authenticated retained Setup chain")

    old_setup_map = old_session.get("setup_reuse_receipts")
    if type(old_setup_map) is not dict or set(old_setup_map) != {"ppi"}:
        abort("compatible capacity original Setup selection is not exactly PPI")
    retained = value.get("retained_setup")
    expected_retained = {dataset: dataset in old_setup_map for dataset in _COMPAT_DATASETS}
    if (
        type(retained) is not dict or set(retained) != set(_COMPAT_DATASETS)
        or any(type(flag) is not bool for flag in retained.values())
        or retained != expected_retained
    ):
        abort("compatible capacity retained-Setup map differs from the original Setup selection")

    resume = value.get("resume_identity")
    session_resume = document.get("compatible_prover_identity")
    session_resume_sha256 = document.get("compatible_prover_identity_sha256")
    if (
        type(resume) is not dict or resume.get("schema") != "ezkl-formal-resume-identity-v1"
        or type(session_resume) is not dict
        or canonical_json_digest(resume) != canonical_json_digest(session_resume)
        or resume.get("selected_datasets") != list(_COMPAT_DATASETS)
        or value.get("resume_identity_sha256") != canonical_json_digest(resume)
        or not isinstance(session_resume_sha256, str) or _SHA256.fullmatch(session_resume_sha256) is None
        or session_resume_sha256 != canonical_json_digest(resume)
    ):
        abort("compatible capacity current resume identity/digest differs from the current session")

    capacity_identity = value.get("capacity_source_identity")
    capacity_identity_sha256 = value.get("capacity_source_identity_sha256")
    session_capacity_identity = document.get("compatible_capacity_source_identity")
    session_capacity_sha256 = document.get("compatible_capacity_source_identity_sha256")
    actual_capacity_identity = capacity_source_identity()
    if (
        not isinstance(capacity_identity, dict)
        or not isinstance(capacity_identity_sha256, str) or _SHA256.fullmatch(capacity_identity_sha256) is None
        or capacity_identity_sha256 != canonical_json_digest(capacity_identity)
        or not isinstance(session_capacity_identity, dict)
        or canonical_json_digest(session_capacity_identity) != canonical_json_digest(capacity_identity)
        or not isinstance(session_capacity_sha256, str) or _SHA256.fullmatch(session_capacity_sha256) is None
        or session_capacity_sha256 != canonical_json_digest(session_capacity_identity)
        or not _identity_matches(capacity_identity, actual_capacity_identity)
    ):
        abort("compatible capacity source identity/digest is stale or differs from current source")

    admission_root_value = value.get("admission_root")
    if not isinstance(admission_root_value, str):
        abort("compatible Admission root must be a baseline-local relative path")
    admission_root = local_path(admission_root_value, "compatible Admission root")
    expected_root = session_dir / "compatibility" / "admission"
    expected_root_value = expected_root.relative_to(ROOT).as_posix()
    _fail_symlink(admission_root)
    if (
        admission_root_value != expected_root_value
        or admission_root.resolve(strict=False) != expected_root.resolve(strict=False)
        or admission_root.is_symlink()
    ):
        abort("compatible Admission root is not the exact isolated session sidecar")

    admissions = value.get("native_admissions")
    old_admissions = old_session.get("native_admissions")
    session_admissions = document.get("compatible_native_admissions")
    if (
        type(admissions) is not dict or set(admissions) != set(_COMPAT_DATASETS)
        or type(session_admissions) is not dict or set(session_admissions) != set(_COMPAT_DATASETS)
        or type(old_admissions) is not dict or set(old_admissions) != set(_COMPAT_DATASETS)
    ):
        abort("compatible capacity Admission maps must contain exactly PPI and OGBN-Arxiv")
    for dataset in _COMPAT_DATASETS:
        current_receipt = admissions[dataset]
        if type(current_receipt) is not dict or set(current_receipt) != _NATIVE_ADMISSION_RECEIPT_KEYS:
            abort(f"{dataset} compatible current Admission receipt has unknown, missing, or malformed fields")
    if canonical_json_digest(session_admissions) != canonical_json_digest(admissions):
        abort("compatible capacity current Admission map differs from the exact session receipt anchor")

    current = dict(document)
    current.update({
        "resume_identity": resume,
        "capacity_source_identity": capacity_identity,
        "native_admissions": admissions,
        "source_tree_sha256": resume.get("source_tree_sha256"),
        "resource_manifest_sha256": resume.get("resource_manifest_sha256"),
        "srs_manifest_sha256": resume.get("srs_manifest_sha256"),
        "binary_sha256": resume.get("binary_sha256"),
        "build_receipt_sha256": resume.get("build_receipt_sha256"),
        "build_environment_sha256": resume.get("build_environment_sha256"),
    })
    old_output_root_value = old_session.get("output_root")
    if not isinstance(old_output_root_value, str) or old_output_root_value != document.get("output_root"):
        abort("compatible original and current session output roots differ")
    old_output_root = local_path(old_output_root_value, "compatible original output root")
    old_session_dir = old_output_root / ".sessions" / session_id

    for dataset in _COMPAT_DATASETS:
        current_receipt = admissions[dataset]
        old_receipt = old_admissions[dataset]
        if (
            type(current_receipt) is not dict or set(current_receipt) != _NATIVE_ADMISSION_RECEIPT_KEYS
            or type(old_receipt) is not dict or set(old_receipt) != _NATIVE_ADMISSION_RECEIPT_KEYS
        ):
            abort(f"{dataset} compatible Admission receipt has unknown, missing, or malformed fields")

        current_admission_path, current_settings_path = _admission_paths(session_dir, dataset, admission_root)
        current_admission, _ = _authenticated_admission(
            current, session_dir, dataset, current_admission_path, current_settings_path,
            admission_root=admission_root,
        )
        old_admission_path, old_settings_path = _admission_paths(old_session_dir, dataset)
        old_admission, _ = _authenticated_admission(
            old_session, old_session_dir, dataset, old_admission_path, old_settings_path,
        )
        for receipt, admission, label in (
            (current_receipt, current_admission, "current"),
            (old_receipt, old_admission, "original"),
        ):
            for name in ("scalar_domain_vector_bytes", "total_assignments", "logrows", "domain_size"):
                minimum = 1 if name == "domain_size" else 0
                receipt_value = _integer(receipt.get(name), f"{dataset} {label} Admission receipt {name}", minimum=minimum)
                admission_value = _integer(admission.get(name), f"{dataset} {label} Admission {name}", minimum=minimum)
                if receipt_value != admission_value:
                    abort(f"{dataset} {label} Admission receipt disagrees with its bound {name}")
        compare_admission_semantics(old_admission, current_admission, dataset=dataset)

    old_ppi_admission = old_admissions["ppi"]
    if old_setup_refs.get("bound_setup_admission") != {
        "path": old_ppi_admission.get("path"),
        "sha256": old_ppi_admission.get("sha256"),
        "settings_sha256": old_ppi_admission.get("settings_sha256"),
    }:
        abort("compatible retained Setup is not bound to the original PPI Admission")

    # Revalidate the full current session/build binding before accepting any
    # caller-supplied sidecar identity for capacity accounting.
    _authenticate_fresh_provenance(current, context)
    return current, admission_root, retained


def _current_session_for_root(output_root: Path | str) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = Path(output_root)
    root = root if root.is_absolute() else local_path(root.as_posix(), "current output root")
    sessions = root / ".sessions"
    if sessions.is_symlink() or not sessions.is_dir():
        abort("current output root has no anchored .sessions directory")
    active: list[Path] = []
    for path in sessions.glob("[0-9a-f]" * 32 + ".json"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            document = load_json(path, "formal session record")
        except ContractError:
            continue
        if document.get("status") in {"PREFLIGHT_STARTED", "RUNNING"}:
            active.append(path)
    if len(active) != 1:
        abort("current output root must contain exactly one active session record")
    return _session_context(active[0])


def _write_report(path: Path, document: dict[str, Any], *, session_dir: Path, output_root: Path) -> None:
    compatible_session = False
    session_id = document.get("session_id")
    if isinstance(session_id, str) and _SESSION_ID.fullmatch(session_id) is not None:
        session_file = session_dir.parent / f"{session_id}.json"
        if session_file.is_file() and not session_file.is_symlink():
            _, anchored_root, anchored_session_dir, session_document = _session_context(session_file)
            if (
                anchored_root.resolve() != output_root.resolve()
                or anchored_session_dir.resolve() != session_dir.resolve()
                or session_document.get("session_id") != session_id
            ):
                abort("capacity report session differs from its authenticated output-root anchor")
            compatible_request = session_document.get("compatible_resume_request")
            if compatible_request is not None:
                request, _ = validate_compatible_resume_request(
                    session_document, expected_session_id=session_id,
                )
                supplied_request = document.get("compatible_resume_request")
                if supplied_request is not None and canonical_json_digest(supplied_request) != canonical_json_digest(request):
                    abort("compatible capacity report request differs from its authenticated session")
                current_provenance = document.get("current_provenance")
                current_resume_identity = session_document.get("compatible_prover_identity")
                current_capacity_identity = session_document.get("compatible_capacity_source_identity")
                if (
                    type(current_provenance) is not dict
                    or type(current_resume_identity) is not dict
                    or type(current_capacity_identity) is not dict
                    or current_provenance.get("resume_identity_sha256") != canonical_json_digest(current_resume_identity)
                    or session_document.get("compatible_prover_identity_sha256") != canonical_json_digest(current_resume_identity)
                    or session_document.get("compatible_capacity_source_identity_sha256") != canonical_json_digest(current_capacity_identity)
                    or type(current_provenance.get("capacity_source_identity")) is not dict
                    or canonical_json_digest(current_provenance["capacity_source_identity"]) != canonical_json_digest(current_capacity_identity)
                    or type(document.get("source_identity")) is not dict
                    or canonical_json_digest(document["source_identity"]) != canonical_json_digest(current_capacity_identity)
                ):
                    abort("compatible capacity report does not bind the current session Prover and capacity identities")
                expected_path = session_dir / "compatibility" / "capacity_precheck_current.json"
                if path.is_symlink() or path.resolve(strict=False) != expected_path.resolve(strict=False):
                    abort("compatible current capacity report must use its exact session sidecar path")
                compatible_session = True
    if not compatible_session and (
        document.get("compatible_resume_request") is not None
        or path.is_symlink()
        or path.parent.resolve() != (session_dir / "capacity").resolve()
    ):
        abort("capacity report must be written inside the current session capacity directory")
    try:
        path.resolve(strict=False).relative_to(output_root.resolve())
    except ValueError:
        abort("capacity report escapes the authenticated formal output root")
    if path.parent.is_symlink():
        abort("capacity report directory must not be symlinked")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        abort(f"cannot persist resource-capacity report: {exc}")


def _historical_session(session_file: Path, output_root: Path) -> dict[str, Any]:
    _fail_symlink(session_file)
    session_file = session_file.resolve(strict=True)
    root = output_root.resolve(strict=True)
    session_id = session_file.stem
    if _SESSION_ID.fullmatch(session_id) is None or session_file != (root / ".sessions" / f"{session_id}.json").resolve(strict=True):
        abort("historical session file does not match its output-root/session anchor")
    document = load_json(session_file, "historical EZKL session")
    if (
        document.get("schema_version") != 1 or document.get("baseline") != "ezkl"
        or document.get("session_id") != session_id or document.get("status") not in _TERMINAL
        or document.get("output_root") != root.relative_to(ROOT).as_posix()
    ):
        abort("historical session baseline/status/output-root anchor is invalid")
    resume = document.get("resume_identity")
    if not isinstance(resume, dict) or resume.get("schema") != "ezkl-formal-resume-identity-v1":
        abort("historical session lacks its immutable build/resource identity")
    for key in ("source_tree_sha256", "resource_manifest_sha256", "srs_manifest_sha256", "srs_sha256", "binary_sha256", "build_receipt_sha256"):
        if not isinstance(resume.get(key), str) or _SHA256.fullmatch(resume[key]) is None:
            abort(f"historical session identity is malformed: {key}")
    if any(document.get(key) != resume.get(key) for key in ("source_tree_sha256", "resource_manifest_sha256", "srs_manifest_sha256", "binary_sha256", "build_receipt_sha256")):
        abort("historical session and immutable build/resource identities disagree")
    build_path = ROOT / "artifacts/build/gpu/build_receipt.json"
    binary_path = local_path(resume.get("binary_path"), "historical binary")
    if (
        build_path.is_symlink() or not build_path.is_file() or sha256(build_path) != resume["build_receipt_sha256"]
        or binary_path.is_symlink() or not binary_path.is_file() or sha256(binary_path) != resume["binary_sha256"]
    ):
        abort("historical build receipt/binary no longer matches the session")
    build = load_json(build_path, "historical GPU build receipt")
    if not _build_receipt_matches_identity(build, resume):
        abort("historical build receipt fields do not match the session identity")
    if resource_manifest_hash() != resume["resource_manifest_sha256"]:
        abort("historical resource manifest differs from the currently authenticated manifest")
    srs_identity = _native_srs_identity()
    if srs_identity["manifest_sha256"] != resume["srs_manifest_sha256"]:
        abort("historical native SRS manifest differs from the currently authenticated manifest")
    if (
        srs_identity["path"] != resume.get("srs_path") or srs_identity["sha256"] != resume.get("srs_sha256")
        or srs_identity["file_identity"][2] != srs_identity["size_bytes"]
    ):
        abort("historical session SRS identity differs from the shared five-dataset manifest")
    resource_snapshot = document.get("resource_file_identities")
    if resource_snapshot is not None:
        current_resources = [[name, list(identity)] for name, identity in resource_file_identities()]
        if resource_snapshot != current_resources:
            abort("historical manifest-listed resource identity snapshot changed")
    configs = [[dataset, sha256(ROOT / f"resources/canonical/configs/{dataset}_secure.cfg")] for dataset in DATASETS]
    if resume.get("config_sha256s") != configs:
        abort("historical secure-config identities differ from the current five configs")
    selected_datasets = selected_datasets_for_session(document)
    receipts = document.get("native_admissions")
    if not isinstance(receipts, dict) or not receipts or any(dataset not in selected_datasets for dataset in receipts):
        abort("historical session has no valid authenticated Admission anchors")
    session_dir = root / ".sessions" / session_id
    geometry: dict[str, dict[str, Any]] = {}
    for dataset in receipts:
        admission_path, settings_path = _admission_paths(session_dir, dataset)
        admission, receipt = _authenticated_admission(document, session_dir, dataset, admission_path, settings_path)
        if receipt != receipts[dataset]:
            abort(f"{dataset} Admission anchor differs from historical session")
        geometry[dataset] = _fixed_payload_bounds(admission)
    return {"document": document, "session_dir": session_dir, "selected_datasets": selected_datasets, "geometry": geometry}


def latest_authenticated_historical_session(output_root: Path | str) -> Path | None:
    """Return the newest fully anchored terminal run; ignore untrusted candidates."""
    current = Path(output_root)
    current = current if current.is_absolute() else local_path(current.as_posix(), "current output root")
    formal_root = ROOT / "runs/formal"
    if formal_root.is_symlink() or not formal_root.is_dir():
        return None
    roots = [formal_root, *(p for p in formal_root.iterdir() if p.is_dir() and not p.is_symlink())]
    candidates: list[tuple[str, Path]] = []
    for root in roots:
        if root.resolve() == current.resolve():
            continue
        sessions = root / ".sessions"
        if sessions.is_symlink() or not sessions.is_dir():
            continue
        for session_file in sessions.glob("[0-9a-f]" * 32 + ".json"):
            if session_file.is_symlink() or not session_file.is_file():
                continue
            try:
                doc = _historical_session(session_file, root)["document"]
                started = doc.get("started_at_utc")
                if isinstance(started, str):
                    datetime.fromisoformat(started.replace("Z", "+00:00"))
                    candidates.append((started, session_file))
            except (ContractError, OSError, TypeError, ValueError):
                continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def audit_historical_session(
    session_path: Path | str, output_path: Path | str, *, output_root: Path | str,
) -> dict[str, Any]:
    """Write historical evidence; only exact source-snapshot matches can gate."""
    _, current_root, current_dir, _ = _current_session_for_root(output_root)
    historical_file = Path(session_path)
    historical_file = historical_file if historical_file.is_absolute() else local_path(historical_file.as_posix(), "historical session")
    historical_root = historical_file.parent.parent
    authenticated = _historical_session(historical_file, historical_root)
    historical = authenticated["document"]
    selected_datasets = authenticated["selected_datasets"]
    source_match = _identity_matches(historical.get("capacity_source_identity"), capacity_source_identity())
    hardware = _sample_hardware(current_root)
    rows = []
    profile = historical.get("local_capacity_profile")
    prior_estimates = profile.get("datasets", {}) if isinstance(profile, dict) else {}
    checks = historical.get("capacity_checks", [])
    for dataset in selected_datasets:
        row = dict(authenticated["geometry"].get(dataset, _missing_row(dataset)))
        if dataset not in authenticated["geometry"]:
            row["historical_evidence"] = "no authenticated Admission in this session"
        prior = prior_estimates.get(dataset) if isinstance(prior_estimates, dict) else None
        row["historical_source_model_estimates"] = {
            "evidence_kind": "historical_capacity_profile_estimate_not_measured_complete_peak" if isinstance(prior, dict) else "not_recorded",
            "coverage": profile.get("coverage") if isinstance(profile, dict) else None,
            "values_bytes": prior,
        }
        row["historical_capacity_check_samples"] = [x for x in checks if isinstance(x, dict) and x.get("dataset") == dataset] if isinstance(checks, list) else []
        rows.append(row)
    failed = [
        row["dataset"] for row in rows
        if isinstance(row.get("fixed_payload_lower_bound_bytes"), int)
        and row["fixed_payload_lower_bound_bytes"] > hardware["disk"]["available_bytes"]
    ]
    report = {
        "schema": _REPORT_SCHEMA, "report_kind": "historical_negative_evidence_only",
        "baseline": "ezkl", "historical_session_id": historical["session_id"],
        "selected_datasets": list(selected_datasets),
        "historical_status": historical["status"], "historical_started_at_utc": historical.get("started_at_utc"),
        "historical_source_match": source_match,
        "historical_source_match_reason": "exact_geometry_config_resource_native_format_hash_match" if source_match else "missing_or_mismatched_geometry_config_resource_native_format_snapshot",
        "historical_admission_coverage": {
            "authenticated_datasets": sorted(authenticated["geometry"]),
            "missing_datasets": [dataset for dataset in selected_datasets if dataset not in authenticated["geometry"]],
        },
        "source_matched_disk_floor_failure": bool(source_match and failed),
        "source_matched_disk_floor_failure_datasets": failed if source_match else [],
        "readiness": "NOT_ESTABLISHED", "actual_current_hardware_sample": hardware, "datasets": rows,
        "scope": "historical evidence only; missing/mismatched source snapshot cannot establish current readiness or gate a current run",
    }
    out = Path(output_path)
    out = out if out.is_absolute() else ROOT / out
    _write_report(out, report, session_dir=current_dir, output_root=current_root)
    return report


def write_and_require_fresh_report(
    session_path: Path | str, records: list[dict[str, Any]], output_path: Path | str, *, output_root: Path | str,
    authenticated_context: AuthenticatedFormalContext | None = None,
    compatible_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist selected authenticated Admission rows and report the v4 wire floor."""
    _, root, session_dir, document = _session_context(session_path)
    selected_datasets = selected_datasets_for_session(document)
    supplied_root = Path(output_root)
    supplied_root = supplied_root if supplied_root.is_absolute() else local_path(supplied_root.as_posix(), "formal output root")
    if supplied_root.resolve() != root.resolve() or not isinstance(records, list) or len(records) != len(selected_datasets):
        abort("fresh capacity report has an invalid output root or incomplete selected-dataset record list")
    by_dataset = {row.get("dataset"): row for row in records if isinstance(row, dict)}
    if tuple(by_dataset) != selected_datasets or len(by_dataset) != len(records):
        abort("fresh capacity rows do not match the immutable canonical dataset selection")
    # Re-read every selected anchored file instead of trusting caller-built rows.
    admission_root = None
    retained_setup = None
    if compatible_provenance is not None:
        document, admission_root, retained_setup = _compatible_provenance_document(
            document, session_dir, authenticated_context, compatible_provenance,
        )
    rows = [
        _fresh_row(
            document, root, session_dir, name, admission_root=admission_root,
            retained_setup_override=retained_setup.get(name) if isinstance(retained_setup, dict) else None,
        )
        for name in selected_datasets
    ]
    provenance = _authenticate_fresh_provenance(document, authenticated_context)
    hardware = _sample_hardware(root)
    missing = [row["dataset"] for row in rows if row.get("admission_status") != "authenticated_current_session_admission"]
    retained = {row["dataset"] for row in rows if row.get("setup_already_retained") is True}
    pending = [row for row in rows if row.get("admission_status") == "authenticated_current_session_admission" and row["dataset"] not in retained]
    pending_floor = sum(_integer(row.get("fixed_payload_lower_bound_bytes"), f"{row['dataset']} known floor") for row in pending)
    failing = [row["dataset"] for row in pending if row["fixed_payload_lower_bound_bytes"] > hardware["disk"]["available_bytes"]]
    aggregate_fail = pending_floor > hardware["disk"]["available_bytes"]
    unknown = [row["dataset"] for row in pending if row.get("lower_bound_gate_status") == "unavailable_from_admission_encoding_metadata"]
    status = "incomplete_admission" if missing else "current_v4_pk_payload_lower_bound_unavailable" if unknown else "current_disk_floor_fail" if aggregate_fail else "current_disk_floor_pass"
    report = {
        "schema": _REPORT_SCHEMA, "report_kind": "current_fresh_admission_pre_setup",
        "baseline": "ezkl", "session_id": document["session_id"], "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_datasets": list(selected_datasets),
        "gate_status": status, "readiness": "NOT_ESTABLISHED",
        "source_identity": provenance["capacity_source_identity"],
        "current_provenance": provenance,
        "actual_current_hardware_sample": hardware,
        "disk_floor_gate": {
            "evidence_status": "structural_lower_bound_only; unavailable means this report does not authorize Setup",
            "disk_available_bytes": hardware["disk"]["available_bytes"],
            "pending_known_fixed_payload_lower_bound_bytes": pending_floor,
            "pending_datasets": [row["dataset"] for row in pending], "individually_over_disk": failing,
            "payload_size_unknown_datasets": unknown,
            "aggregate_floor_failure": aggregate_fail, "retained_setup_datasets_excluded": sorted(retained),
            "comparison": "known structural lower bound only; unknown PK payload bytes are not estimated or treated as a capacity pass",
        },
        "datasets": rows,
        "source_model_estimate_scope": "per-dataset source_model_estimates are source-derived, not actual allocation samples or complete measured phase peaks",
        "scope": "Admission geometry is authenticated; source-model Setup upper and actual-census Prove gates remain mandatory and this report never establishes run readiness",
    }
    out = Path(output_path)
    out = out if out.is_absolute() else ROOT / out
    _write_report(out, report, session_dir=session_dir, output_root=root)
    if status not in {"current_disk_floor_pass", "current_v4_pk_payload_lower_bound_unavailable"}:
        abort(f"EZKL capacity precheck {status}: missing={missing}; floor_bytes={pending_floor}; disk_available_bytes={hardware['disk']['available_bytes']}; individually_over_disk={failing}")
    return report
