#!/usr/bin/env python3
"""Support the source-matched, session-scoped GAT-ZKML ablation runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
from typing import Any


VARIANTS = {"wo_feature_folding", "wo_shared_neighborhood"}
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
HEX_256 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_config(path: pathlib.Path) -> tuple[dict[str, str], list[str]]:
    require(path.is_file() and not path.is_symlink(), f"config must be a regular file: {path}")
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        require("=" in line, f"malformed config line {line_number} in {path}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        require(key not in values, f"duplicate config key {key!r} in {path}")
        values[key] = value
    return values, lines


def safe_project_relative(value: str, field: str) -> str:
    path = pathlib.PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts, f"{field} must be a safe project-relative path")
    return path.as_posix()


def write_variant_config(
    base: pathlib.Path,
    output: pathlib.Path,
    variant: str,
    export_dir: str,
    registry_path: str,
    registry_digest: str | None,
) -> None:
    require(variant in VARIANTS, f"unsupported ablation variant: {variant}")
    export_dir = safe_project_relative(export_dir, "export_dir")
    registry_path = safe_project_relative(registry_path, "verifier_registry_path")
    if registry_digest is not None:
        require(HEX_256.fullmatch(registry_digest) is not None, "verifier_registry_digest must be lowercase SHA-256")

    values, lines = parse_config(base)
    require(values.get("dataset") in DATASETS, f"unsupported base dataset in {base}")
    updates = {
        "protocol_variant": variant,
        "export_dir": export_dir,
        "verifier_registry_path": registry_path,
    }
    if registry_digest is not None:
        updates["verifier_registry_digest"] = registry_digest

    output_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key == "verifier_registry_digest" and registry_digest is None:
            continue
        if key in updates:
            indent = line[: len(line) - len(line.lstrip())]
            output_lines.append(f"{indent}{key} = {updates[key]}\n")
            seen.add(key)
        else:
            output_lines.append(line)

    for key in ("protocol_variant", "export_dir", "verifier_registry_path", "verifier_registry_digest"):
        if key in updates and key not in seen:
            output_lines.append(f"{key} = {updates[key]}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("".join(output_lines), encoding="utf-8")
    os.replace(temporary, output)

    rendered, _ = parse_config(output)
    require(rendered.get("protocol_variant") == variant, "generated config lost protocol_variant")
    require(rendered.get("export_dir") == export_dir, "generated config lost session export_dir")
    require(rendered.get("verifier_registry_path") == registry_path, "generated config lost registry path")
    require(rendered.get("verifier_registry_digest") == registry_digest, "generated config has an unexpected registry digest")
    for key, value in values.items():
        if key not in {"protocol_variant", "export_dir", "verifier_registry_path", "verifier_registry_digest"}:
            require(rendered.get(key) == value, f"generated config unexpectedly changed base field {key}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"JSON artifact has duplicate field {key!r}")
        result[key] = value
    return result


def load_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{context} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read {context} as strict JSON: {exc}") from exc
    require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_registry(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    config, _ = parse_config(pathlib.Path(args.config))
    inspection = load_json(pathlib.Path(args.inspection), "registry inspection")
    digest = args.registry_digest
    require(HEX_256.fullmatch(digest) is not None, "registry digest must be lowercase SHA-256")
    registry_path = pathlib.Path(args.registry).resolve()
    configured_registry = pathlib.Path(config.get("verifier_registry_path", ""))
    if not configured_registry.is_absolute():
        configured_registry = project_root / configured_registry
    require(configured_registry.resolve() == registry_path, "variant config points at a different verifier registry path")
    require(config.get("verifier_registry_digest") == digest, "variant config does not pin the inspected registry digest")
    require(sha256_file(registry_path) == digest, "registry artifact SHA-256 differs from pinned digest")
    require(inspection.get("artifact_sha256") == digest and inspection.get("registry_digest") == digest,
            "registry inspection does not authenticate the pinned registry digest")
    require(inspection.get("protocol_variant") == args.variant, "registry inspection protocol_variant mismatch")
    require(inspection.get("security_budget_passed") is True, "registry SecurityBudgetLedger preflight failed")
    srs_root = pathlib.Path(config.get("srs_path", ""))
    if not srs_root.is_absolute():
        srs_root = project_root / srs_root
    srs_manifest_path = srs_root / "srs_manifest.json"
    srs_manifest = load_json(srs_manifest_path, "configured SRS manifest")
    require(srs_manifest.get("srs_id") == config.get("srs_id") == inspection.get("srs_id"),
            "registry inspection SRS id differs from the canonical config and SRS manifest")
    require(srs_manifest.get("manifest_hash") == inspection.get("srs_manifest_hash"),
            "registry inspection SRS manifest hash differs from the configured SRS")
    require(srs_manifest.get("protocol_schema_id") == inspection.get("protocol_schema_id"),
            "registry inspection changed the static protocol schema bound by the configured SRS")
    if "srs_degree_bound" in inspection:
        require(str(srs_manifest.get("degree_bound")) == str(inspection.get("srs_degree_bound")),
                "registry inspection SRS degree bound differs from the configured SRS")
    for config_key, registry_key in (
        ("dataset", "dataset"),
        ("model_arch_id", "model_arch_id"),
        ("model_param_id", "model_param_id"),
        ("quant_cfg_id", "quant_cfg_id"),
        ("static_table_id", "static_table_id"),
        ("srs_id", "srs_id"),
    ):
        if registry_key in inspection:
            require(inspection.get(registry_key) == config.get(config_key), f"registry inspection {registry_key} does not match config")
        else:
            raise ValueError(f"registry inspection is missing {registry_key}")
    require(HEX_256.fullmatch(str(inspection.get("protocol_schema_id", ""))) is not None,
            "registry inspection protocol_schema_id is malformed")
    require(HEX_256.fullmatch(str(inspection.get("instance_schema_hash", ""))) is not None,
            "registry inspection instance_schema_hash is malformed")


def _load_build_provenance_module(project_root: pathlib.Path):
    path = project_root / "tools" / "build_provenance.py"
    spec = importlib.util.spec_from_file_location("gatzk_ablation_build_provenance", path)
    require(spec is not None and spec.loader is not None, f"cannot load build provenance helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _truth(value: Any) -> bool:
    return value is True or value == "true"


def validate_cell(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    run_dir = pathlib.Path(args.run_dir).resolve()
    config_path = pathlib.Path(args.config).resolve()
    expected_binary = pathlib.Path(args.binary).resolve()
    require(args.variant in VARIANTS, f"unsupported ablation variant: {args.variant}")
    require(args.dataset in DATASETS, f"unsupported dataset: {args.dataset}")
    require(args.backend in {"cpu", "gpu"}, f"unsupported backend: {args.backend}")
    manifest = load_json(run_dir / "run_manifest.json", "run manifest")
    proof_path = run_dir / "proof.bin"
    report_path = run_dir / "verifier_report.json"
    receipt_path = run_dir / "build_receipt.json"
    source_manifest_path = run_dir / "source_manifest.json"
    require(proof_path.is_file() and not proof_path.is_symlink() and proof_path.stat().st_size > 0,
            "formal run has no nonempty regular proof.bin")
    proof_hash = sha256_file(proof_path)
    require(manifest.get("proof_hash") == proof_hash, "run manifest proof_hash mismatch")
    require(manifest.get("protocol_name") == args.variant, "run manifest protocol_name does not match the selected variant")
    require(manifest.get("dataset") == args.dataset, "run manifest dataset mismatch")
    require(manifest.get("backend") == args.backend, "run manifest backend mismatch")
    require(manifest.get("status") == "VERIFY_OK" and _truth(manifest.get("verifier_accepted")) and
            _truth(manifest.get("final_formal_run_accepted")), "run manifest is not a successful formal verification")
    require(manifest.get("benchmark_mode") == "single", "run manifest is not a single formal run")
    require(_truth(manifest.get("is_full_dataset")) and _truth(manifest.get("full_dataset")),
            "run manifest is not a full-dataset run")
    require(pathlib.Path(str(manifest.get("config", ""))).resolve() == config_path, "run manifest config path mismatch")
    require(sha256_file(config_path) == manifest.get("config_sha256"), "run manifest config SHA-256 mismatch")
    config, _ = parse_config(config_path)
    require(config.get("dataset") == args.dataset and config.get("protocol_variant") == args.variant,
            "pinned formal config does not match dataset and selected variant")
    require(config.get("verifier_registry_digest") == args.registry_digest, "formal config registry digest mismatch")
    require(manifest.get("verifier_registry_digest") == args.registry_digest and
            manifest.get("trusted_registry_digest") == args.registry_digest, "run manifest registry binding mismatch")
    require(manifest.get("verifier_report_path") == "verifier_report.json", "unexpected verifier report path")
    require(manifest.get("build_receipt_path") == "build_receipt.json", "unexpected build receipt path")
    report = load_json(report_path, "verifier report")
    require(report.get("format") == "zkgat_verifier_report", "verifier report format mismatch")
    require(report.get("trusted_registry_digest") == args.registry_digest and
            report.get("authenticated_registry_digest") == args.registry_digest, "verifier report registry binding mismatch")
    require(report.get("proof_bin_sha256") == proof_hash, "verifier report proof hash mismatch")
    require(_truth(report.get("verify_result")) and _truth(report.get("security_budget_passed")) and
            _truth(report.get("reference_consistency_passed")), "verifier report correctness gates did not pass")
    require(report.get("parsed_output_digest") == report.get("proof_public_output_hash") == manifest.get("proof_public_output_hash"),
            "parsed public output digest is not bound across proof, verifier report, and manifest")

    require(manifest.get("build_receipt_hash") == sha256_file(receipt_path), "build receipt hash mismatch")
    receipt = load_json(receipt_path, "build receipt")
    require(manifest.get("source_manifest_hash") == sha256_file(source_manifest_path), "source manifest hash mismatch")
    source_manifest = load_json(source_manifest_path, "source manifest")
    provenance = _load_build_provenance_module(project_root)
    current_source_hash = provenance.source_manifest_document(project_root)["source_tree_hash"]
    require(manifest.get("source_tree_hash") == current_source_hash == source_manifest.get("source_tree_hash") ==
            receipt.get("source_tree_hash"), "formal run is not matched to the current source tree")
    require(pathlib.Path(str(manifest.get("binary_path", ""))).resolve() == expected_binary,
            "run manifest binary path differs from the selected backend executable")
    require(pathlib.Path(str(receipt.get("binary_path", ""))).resolve() == expected_binary,
            "build receipt binary path differs from the selected backend executable")
    require(manifest.get("binary_hash") == receipt.get("binary_hash") == sha256_file(expected_binary),
            "formal run binary hash does not match the selected executable")
    if args.receipt_path:
        receipt_path = pathlib.Path(args.receipt_path)
        require(not receipt_path.exists() and not receipt_path.is_symlink(),
                f"formal evidence receipt already exists: {receipt_path}")
        evidence = {
            "backend": args.backend,
            "config_sha256": manifest["config_sha256"],
            "dataset": args.dataset,
            "proof_hash": proof_hash,
            "protocol_variant": args.variant,
            "registry_digest": args.registry_digest,
            "run_dir": str(run_dir),
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "status": "VERIFY_OK",
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_name(receipt_path.name + ".tmp")
        temporary.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, receipt_path)
    print(f"FORMAL_EVIDENCE_OK dataset={args.dataset} backend={args.backend} protocol_variant={args.variant} "
          f"registry_digest={args.registry_digest} run_dir={run_dir}")


def write_session_index(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    session_root = pathlib.Path(args.session_root).resolve()
    require(args.variant in VARIANTS, f"unsupported ablation variant: {args.variant}")
    backends = tuple(args.backends.split(","))
    require(backends and all(backend in {"cpu", "gpu"} for backend in backends), "invalid session backend list")
    require(len(set(backends)) == len(backends), "duplicate backend in session backend list")

    datasets: dict[str, Any] = {}
    registrations_complete = True
    for dataset in DATASETS:
        base_path = project_root / "configs" / f"{dataset}_secure.cfg"
        config_path = session_root / "configs" / f"{dataset}.cfg"
        registry_path = session_root / "registries" / f"{dataset}.registry"
        inspection_path = session_root / "preflight" / f"{dataset}.registry.json"
        base_values: dict[str, str] = {}
        if base_path.is_file() and not base_path.is_symlink():
            base_values, _ = parse_config(base_path)
        else:
            registrations_complete = False
        config_values: dict[str, str] = {}
        if config_path.is_file() and not config_path.is_symlink():
            config_values, _ = parse_config(config_path)
        else:
            registrations_complete = False
        inspection: dict[str, Any] = {}
        srs_manifest: dict[str, Any] = {}
        if config_values:
            srs_root = pathlib.Path(config_values.get("srs_path", ""))
            if not srs_root.is_absolute():
                srs_root = project_root / srs_root
            srs_manifest_path = srs_root / "srs_manifest.json"
            if srs_manifest_path.is_file() and not srs_manifest_path.is_symlink():
                srs_manifest = load_json(srs_manifest_path, f"{dataset} configured SRS manifest")
        if registry_path.is_file() and not registry_path.is_symlink() and inspection_path.is_file() and not inspection_path.is_symlink():
            inspection = load_json(inspection_path, f"{dataset} registry inspection")
            digest = sha256_file(registry_path)
            require(inspection.get("artifact_sha256") == digest and inspection.get("registry_digest") == digest,
                    f"{dataset} session registry digest does not match inspection")
            require(inspection.get("protocol_variant") == args.variant, f"{dataset} registry has the wrong protocol variant")
            require(config_values.get("protocol_variant") == args.variant and config_values.get("verifier_registry_digest") == digest,
                    f"{dataset} pinned config does not match the inspected registry")
            require(inspection.get("dataset") == dataset, f"{dataset} registry dataset mismatch")
        else:
            registrations_complete = False
            digest = ""

        generated_config_sha256 = sha256_file(config_path) if config_path.is_file() and not config_path.is_symlink() else ""
        runs: dict[str, Any] = {}
        for backend in backends:
            evidence_path = session_root / "evidence" / f"{backend}_{dataset}.json"
            if evidence_path.is_file() and not evidence_path.is_symlink():
                evidence = load_json(evidence_path, f"{backend}/{dataset} formal evidence receipt")
                require(evidence.get("protocol_variant") == args.variant and evidence.get("dataset") == dataset and
                        evidence.get("backend") == backend and evidence.get("registry_digest") == digest,
                        f"{backend}/{dataset} evidence receipt does not match this session")
                require(evidence.get("status") == "VERIFY_OK" and
                        evidence.get("config_sha256") == generated_config_sha256,
                        f"{backend}/{dataset} evidence receipt does not match the pinned config")
                run_dir = pathlib.Path(str(evidence.get("run_dir", ""))).resolve()
                results_root = (session_root / "results").resolve()
                require(run_dir.is_relative_to(results_root), f"{backend}/{dataset} evidence points outside this session results directory")
                run_manifest_path = run_dir / "run_manifest.json"
                proof_path = run_dir / "proof.bin"
                require(sha256_file(run_manifest_path) == evidence.get("run_manifest_sha256"),
                        f"{backend}/{dataset} evidence run manifest hash mismatch")
                require(sha256_file(proof_path) == evidence.get("proof_hash"), f"{backend}/{dataset} evidence proof hash mismatch")
                run_manifest = load_json(run_manifest_path, f"{backend}/{dataset} evidence run manifest")
                require(run_manifest.get("protocol_name") == args.variant and run_manifest.get("dataset") == dataset and
                        run_manifest.get("backend") == backend and run_manifest.get("status") == "VERIFY_OK" and
                        run_manifest.get("verifier_registry_digest") == digest and run_manifest.get("trusted_registry_digest") == digest,
                        f"{backend}/{dataset} evidence run manifest identity mismatch")
                runs[backend] = evidence
            else:
                runs[backend] = {"status": "not_verified"}
        datasets[dataset] = {
            "base_config": str(base_path),
            "base_config_sha256": sha256_file(base_path) if base_path.is_file() and not base_path.is_symlink() else "",
            "generated_config": str(config_path),
            "generated_config_sha256": generated_config_sha256,
            "registry_path": str(registry_path),
            "registry_sha256": digest,
            "protocol_variant": inspection.get("protocol_variant", ""),
            "registry_id": inspection.get("registry_id", ""),
            "protocol_schema_id": inspection.get("protocol_schema_id", ""),
            "instance_schema_hash": inspection.get("instance_schema_hash", ""),
            "srs_id": inspection.get("srs_id", ""),
            "srs_manifest_hash": inspection.get("srs_manifest_hash", ""),
            "srs_degree_bound": inspection.get("srs_degree_bound", srs_manifest.get("degree_bound", "")),
            "security_budget_ledger_digest": inspection.get("security_budget_ledger_digest", ""),
            "runs": runs,
        }

    all_runs_verified = registrations_complete and all(
        runs.get(backend, {}).get("status") == "VERIFY_OK"
        for dataset_entry in datasets.values()
        for backend, runs in dataset_entry["runs"].items()
    )
    status = "complete" if all_runs_verified else ("formal_runs_in_progress" if registrations_complete else "preflight_in_progress")
    index = {
        "format": "zkgat_ablation_session_index_v1",
        "session_id": session_root.name,
        "protocol_variant": args.variant,
        "backends": list(backends),
        "status": status,
        "preflight_complete": registrations_complete,
        "datasets": datasets,
    }
    output = session_root / "ablation_session_index.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"SESSION_INDEX status={status} path={output}")


def _read_integer(path: pathlib.Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if value == "max":
        return None
    try:
        result = int(value)
    except ValueError:
        return None
    return result if result > 0 else None


def effective_available_memory() -> int:
    available: list[int] = []
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            require(len(fields) >= 2 and fields[1].isdigit(), "cannot parse MemAvailable from /proc/meminfo")
            available.append(int(fields[1]) * 1024)
            break
    require(available, "MemAvailable is absent from /proc/meminfo")

    memberships: list[tuple[str, str]] = []
    try:
        for line in pathlib.Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3:
                memberships.append((fields[1], fields[2].lstrip("/")))
    except (OSError, UnicodeError):
        memberships = []
    candidates: set[pathlib.Path] = {pathlib.Path("/sys/fs/cgroup")}
    for controllers, relative in memberships:
        if not controllers:
            base = pathlib.Path("/sys/fs/cgroup") / relative
            while base != pathlib.Path("/"):
                candidates.add(base)
                base = base.parent
        elif "memory" in controllers.split(","):
            for root in (pathlib.Path("/sys/fs/cgroup/memory"), pathlib.Path("/sys/fs/cgroup")):
                base = root / relative
                while base != pathlib.Path("/"):
                    candidates.add(base)
                    base = base.parent
    for directory in candidates:
        limit = _read_integer(directory / "memory.max")
        current = _read_integer(directory / "memory.current") or 0
        if limit is None:
            limit = _read_integer(directory / "memory.limit_in_bytes")
            current = _read_integer(directory / "memory.usage_in_bytes") or current
        if limit is not None:
            available.append(max(0, limit - current))
    return min(available)


def print_build_jobs(_args: argparse.Namespace) -> None:
    available = effective_available_memory()
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    gib = 1024**3
    reserve = max(2 * gib, min(8 * gib, available // 4))
    per_compile_job = 6 * gib
    memory_limited_jobs = max(1, (available - reserve) // per_compile_job)
    print(max(1, min(cpus, memory_limited_jobs, 4)))


def check_build(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    build_dir = pathlib.Path(args.build_dir).resolve()
    cache_path = build_dir / "CMakeCache.txt"
    require(cache_path.is_file() and not cache_path.is_symlink(), f"missing regular CMake cache: {cache_path}")
    cache: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and "=" in line and not line.startswith("//"):
            key, value = line.split("=", 1)[0].split(":", 1)[0], line.split("=", 1)[1]
            cache[key] = value
    expected_cuda = "ON" if args.cuda else "OFF"
    require(cache.get("CMAKE_BUILD_TYPE") == "Release", f"{build_dir} is not a Release build")
    require(cache.get("GATZK_ENABLE_CUDA_BACKEND") == expected_cuda,
            f"{build_dir} CUDA backend does not match requested build type")
    source_manifest_path = build_dir / "generated" / "gatzk" / "source_manifest.json"
    generated = load_json(source_manifest_path, "generated source manifest")
    provenance = _load_build_provenance_module(project_root)
    current = provenance.source_manifest_document(project_root)
    require(generated.get("source_tree_hash") == current.get("source_tree_hash"),
            f"{build_dir} was not configured against the current source tree")
    targets = ("gatzk_run", "gatzk_registry_register", "gatzk_registry_inspect")
    for target in targets:
        executable = build_dir / target
        require(executable.is_file() and not executable.is_symlink() and os.access(executable, os.X_OK),
                f"missing source-matched executable {executable}")
    print(f"SOURCE_MATCHED_BUILD_OK cuda={expected_cuda.lower()} source_tree_hash={current['source_tree_hash']} build_dir={build_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("write-config")
    config.add_argument("--base", required=True, type=pathlib.Path)
    config.add_argument("--output", required=True, type=pathlib.Path)
    config.add_argument("--variant", required=True)
    config.add_argument("--export-dir", required=True)
    config.add_argument("--registry-path", required=True)
    config.add_argument("--registry-digest")

    registry = commands.add_parser("check-registry")
    registry.add_argument("--config", required=True)
    registry.add_argument("--project-root", required=True)
    registry.add_argument("--variant", required=True)
    registry.add_argument("--registry", required=True)
    registry.add_argument("--registry-digest", required=True)
    registry.add_argument("--inspection", required=True)

    cell = commands.add_parser("validate-cell")
    cell.add_argument("--project-root", required=True)
    cell.add_argument("--run-dir", required=True)
    cell.add_argument("--config", required=True)
    cell.add_argument("--variant", required=True)
    cell.add_argument("--dataset", required=True)
    cell.add_argument("--backend", required=True)
    cell.add_argument("--registry-digest", required=True)
    cell.add_argument("--binary", required=True)
    cell.add_argument("--receipt-path")

    index = commands.add_parser("write-index")
    index.add_argument("--project-root", required=True)
    index.add_argument("--session-root", required=True)
    index.add_argument("--variant", required=True)
    index.add_argument("--backends", required=True, help="comma-separated cpu,gpu backends")

    build = commands.add_parser("check-build")
    build.add_argument("--project-root", required=True)
    build.add_argument("--build-dir", required=True)
    build.add_argument("--cuda", action="store_true")

    commands.add_parser("build-jobs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "write-config":
            write_variant_config(args.base, args.output, args.variant, args.export_dir, args.registry_path, args.registry_digest)
        elif args.command == "check-registry":
            check_registry(args)
        elif args.command == "validate-cell":
            validate_cell(args)
        elif args.command == "write-index":
            write_session_index(args)
        elif args.command == "check-build":
            check_build(args)
        elif args.command == "build-jobs":
            print_build_jobs(args)
        else:
            raise ValueError(f"unknown helper command: {args.command}")
        return 0
    except Exception as error:
        print(f"ablation-entrypoint: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
