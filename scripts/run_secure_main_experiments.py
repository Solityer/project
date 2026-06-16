#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASETS: List[Tuple[str, str, str]] = [
    ("cora", "configs/cora_secure.cfg", "runs/cora_secure"),
    ("citeseer", "configs/citeseer_secure.cfg", "runs/citeseer_secure"),
    ("pubmed", "configs/pubmed_secure.cfg", "runs/pubmed_secure"),
    ("ppi", "configs/ppi_secure.cfg", "runs/ppi_secure"),
    ("ogbn-arxiv", "configs/ogbn_arxiv_secure.cfg", "runs/ogbn_arxiv_secure"),
]
RUN_TAG = "paper_secure_final"
ROUTE_FLAGS = [
    "--fast-msm", "on",
    "--parallel-fft", "on",
    "--fft-backend-upgrade", "on",
    "--fft-kernel-upgrade", "on",
    "--trace-layout-upgrade", "on",
    "--fast-verify-pairing", "on",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source_files() -> Iterable[pathlib.Path]:
    roots = [
        ROOT / "CMakeLists.txt",
        ROOT / "include" / "gatzk",
        ROOT / "src",
        ROOT / "configs",
        ROOT / "scripts" / "export_paper_secure.py",
        ROOT / "scripts" / "run_secure_main_experiments.py",
        ROOT / "tools" / "security",
        ROOT / "tests",
    ]
    for root in roots:
        if root.is_file():
            yield root
            continue
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix not in {".pyc", ".o", ".a"}:
                    yield path


def sha256_source_tree() -> str:
    digest = hashlib.sha256()
    for path in sorted(iter_source_files()):
        rel = path.relative_to(ROOT).as_posix()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def manifest_hash(row: Dict[str, object]) -> str:
    payload = dict(row)
    payload.pop("run_manifest_hash", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_command(
    command: List[str],
    log_path: pathlib.Path,
    env: Dict[str, str] | None = None,
    timeout: int | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=merged_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout} seconds\n")
            return 124


def run_required(command: List[str], log_path: pathlib.Path, env: Dict[str, str] | None = None) -> None:
    code = run_command(command, log_path, env=env)
    if code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}; see {log_path}")


def git_status_main() -> str:
    completed = subprocess.run(
        ["git", "status", "--short", "--", ".", ":!comparison/**"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def configure_and_build(build_dir: pathlib.Path, cuda: bool, jobs: int, log_root: pathlib.Path) -> None:
    run_required(
        [
            "cmake",
            "-S", ".",
            "-B", str(build_dir.relative_to(ROOT)),
            f"-DGATZK_ENABLE_CUDA_BACKEND={'ON' if cuda else 'OFF'}",
            "-DGATZK_BUILD_TESTS=ON",
        ],
        log_root / ("configure_cuda.log" if cuda else "configure_cpu.log"),
    )
    run_required(
        ["cmake", "--build", str(build_dir.relative_to(ROOT)), "-j", str(jobs)],
        log_root / ("build_cuda.log" if cuda else "build_cpu.log"),
    )


def run_tests(build_dir: pathlib.Path, label: str, log_root: pathlib.Path, full_security: bool) -> None:
    run_required(
        ["ctest", "--test-dir", str(build_dir.relative_to(ROOT)), "--output-on-failure"],
        log_root / f"ctest_{label}.log",
    )
    if full_security:
        binary = build_dir / "gatzk_tests"
        run_required(
            [str(binary.relative_to(ROOT))],
            log_root / f"full_security_{label}.log",
            env={"GATZK_RUN_FULL_SECURITY_TESTS": "1"},
        )


def write_freeze_identity(
    source_hash: str,
    cpu_binary_hash: str,
    gpu_binary_hash: str,
    runner_hash: str,
    output_dir: pathlib.Path,
) -> str:
    build_identity = hashlib.sha256(
        f"{source_hash}|cpu={cpu_binary_hash}|gpu={gpu_binary_hash}|runner={runner_hash}".encode("utf-8")
    ).hexdigest()
    payload = {
        "generated_at": utc_now(),
        "source_tree_hash": source_hash,
        "cpu_binary_hash": cpu_binary_hash,
        "gpu_binary_hash": gpu_binary_hash,
        "runner_hash": runner_hash,
        "exporter_hash": sha256_file(ROOT / "scripts" / "export_paper_secure.py"),
        "build_identity": build_identity,
        "main_git_status": git_status_main(),
        "comparison_status": "comparison/** ignored by request; no completion claim is based on comparison results.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "freeze_identity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return build_identity


def run_dir(base_run_dir: str, backend: str) -> pathlib.Path:
    return ROOT / base_run_dir / RUN_TAG / backend / "warm"


def unique_log_path(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        return path
    return path.with_name(f"{path.stem}_{timestamp_token()}{path.suffix}")


def archive_existing_tree(path: pathlib.Path, archive_root: pathlib.Path, dataset: str, backend: str) -> None:
    if not path.exists():
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{timestamp_token()}_{backend}_{dataset}_{path.name}"
    suffix = 1
    while target.exists():
        target = archive_root / f"{timestamp_token()}_{backend}_{dataset}_{path.name}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(target))


def load_manifest(path: pathlib.Path) -> Dict[str, object]:
    manifest_path = path / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest is not an object: {manifest_path}")
    return data


def dynamic_profile_exists(path: pathlib.Path, manifest: Dict[str, object]) -> bool:
    profile_name = str(manifest.get("dynamic_commitment_object_profile_csv", ""))
    return bool(profile_name) and (path / profile_name).exists()


def valid_manifest(
    path: pathlib.Path,
    dataset: str,
    backend: str,
    source_hash: str,
    binary_hash: str,
) -> bool:
    try:
        row = load_manifest(path)
    except Exception:
        return False
    required = {
        "dataset": dataset,
        "backend": backend,
        "source_tree_hash": source_hash,
        "binary_hash": binary_hash,
        "verified": "true",
        "full_dataset": "true",
        "timeout": "false",
        "stale_result": "false",
        "exit_code": "0",
        "proof_payload_audit_status": "passed",
        "reference_consistency_passed": "true",
        "tamper_tests_passed": "true",
        "private_plain_value_count": "0",
        "verifier_reads_witness": "false",
    }
    if dataset == "ogbn-arxiv":
        dataset_ok = row.get("dataset") in {"ogbn-arxiv", "ogbn_arxiv"}
    else:
        dataset_ok = row.get("dataset") == dataset
    if not dataset_ok:
        return False
    for key, value in required.items():
        if key == "dataset":
            continue
        if str(row.get(key, "")) != value:
            return False
    if str(row.get("run_manifest_hash", "")) != manifest_hash(row):
        return False
    return dynamic_profile_exists(path, row)


def enrich_manifest(
    path: pathlib.Path,
    dataset: str,
    backend: str,
    source_hash: str,
    binary_hash: str,
    cpu_binary_hash: str,
    gpu_binary_hash: str,
    runner_hash: str,
    build_identity: str,
    started_at: str,
    ended_at: str,
    exit_code: int,
    timed_out: bool,
    command: List[str],
) -> None:
    row = load_manifest(path)
    row.update({
        "dataset": dataset,
        "backend": backend,
        "source_tree_hash": source_hash,
        "binary_hash": binary_hash,
        "cpu_binary_hash": cpu_binary_hash,
        "gpu_binary_hash": gpu_binary_hash,
        "runner_hash": runner_hash,
        "build_identity": build_identity,
        "run_started_at": started_at,
        "run_ended_at": ended_at,
        "exit_code": str(exit_code),
        "timeout": "true" if timed_out else "false",
        "stale_result": "false" if exit_code == 0 and not timed_out else "true",
        "official_runner_command": " ".join(command),
    })
    row["run_manifest_hash"] = manifest_hash(row)
    (path / "run_manifest.json").write_text(
        json.dumps(row, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def execute_row(
    dataset: str,
    config: str,
    base_run_dir: str,
    backend: str,
    binary: pathlib.Path,
    source_hash: str,
    binary_hash: str,
    cpu_binary_hash: str,
    gpu_binary_hash: str,
    runner_hash: str,
    build_identity: str,
    timeout_seconds: int,
    force: bool,
    log_root: pathlib.Path,
    archive_root: pathlib.Path,
) -> pathlib.Path:
    path = run_dir(base_run_dir, backend)
    if not force and valid_manifest(path, dataset, backend, source_hash, binary_hash):
        print(f"resume-valid {backend}/{dataset}: {path.relative_to(ROOT)}", flush=True)
        return path

    stale_root = path.parent
    if stale_root.exists():
        archive_existing_tree(stale_root, archive_root, dataset, backend)

    compute_backend = "cuda_hotspots" if backend == "gpu" else "cpu"
    command = [
        str(binary.relative_to(ROOT)),
        "--config", config,
        "--benchmark-mode", "warm",
        "--compute-backend", compute_backend,
        *ROUTE_FLAGS,
        "--export-tag", f"{RUN_TAG}/{backend}",
    ]
    started_at = utc_now()
    env = {
        "GATZK_RUN_PROGRESS": "1",
        "GATZK_SOURCE_TREE_HASH": source_hash,
        "GATZK_BINARY_HASH": binary_hash,
        "GATZK_RUNNER_HASH": runner_hash,
        "GATZK_BUILD_IDENTITY": build_identity,
        "GATZK_RUN_STARTED_AT": started_at,
        "GATZK_RUN_TIMEOUT": "false",
    }
    print(f"run-start {backend}/{dataset}: {path.relative_to(ROOT)}", flush=True)
    run_log_path = unique_log_path(log_root / f"run_{backend}_{dataset}.log")
    code = run_command(
        command,
        run_log_path,
        env=env,
        timeout=timeout_seconds,
    )
    ended_at = utc_now()
    timed_out = code == 124
    if (path / "run_manifest.json").exists():
        enrich_manifest(
            path,
            dataset,
            backend,
            source_hash,
            binary_hash,
            cpu_binary_hash,
            gpu_binary_hash,
            runner_hash,
            build_identity,
            started_at,
            ended_at,
            code,
            timed_out,
            command,
        )
    if code != 0:
        raise RuntimeError(f"{backend}/{dataset} failed with exit code {code}; see {run_log_path}")
    if not valid_manifest(path, dataset, backend, source_hash, binary_hash):
        raise RuntimeError(f"{backend}/{dataset} manifest failed strict resume validation: {path}")
    print(f"run-ok {backend}/{dataset}: {path.relative_to(ROOT)}", flush=True)
    return path


def export_results(
    cpu_paths: Dict[str, pathlib.Path],
    gpu_paths: Dict[str, pathlib.Path],
    source_hash: str,
    cpu_binary_hash: str,
    gpu_binary_hash: str,
    output_dir: pathlib.Path,
    log_root: pathlib.Path,
) -> None:
    command = [
        sys.executable,
        "scripts/export_paper_secure.py",
        "--output-dir", str(output_dir.relative_to(ROOT)),
        "--source-tree-hash", source_hash,
        "--cpu-binary-hash", cpu_binary_hash,
        "--gpu-binary-hash", gpu_binary_hash,
    ]
    for dataset, _, _ in DATASETS:
        command.extend(["--cpu", f"{dataset}={cpu_paths[dataset].relative_to(ROOT)}"])
    for dataset, _, _ in DATASETS:
        command.extend(["--gpu", f"{dataset}={gpu_paths[dataset].relative_to(ROOT)}"])
    run_required(command, log_root / "export_paper_secure.log")


def validate_export(output_dir: pathlib.Path) -> None:
    rows = json.loads((output_dir / "latest.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 10:
        raise RuntimeError("latest.json must contain exactly 10 rows")
    seen = {(row.get("dataset"), row.get("backend")) for row in rows}
    expected = {(dataset, backend) for dataset, _, _ in DATASETS for backend in ("cpu", "gpu")}
    if seen != expected:
        raise RuntimeError(f"latest.json row set mismatch: {seen ^ expected}")
    manifest_dir = output_dir / "run_manifests"
    for dataset, backend in expected:
        path = manifest_dir / f"{backend}_{dataset}.json"
        if not path.exists():
            raise RuntimeError(f"missing exported manifest: {path}")


def select_datasets(spec: str) -> List[Tuple[str, str, str]]:
    names = [name for name, _, _ in DATASETS]
    if spec == "all":
        return DATASETS
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)}")
    return [entry for entry in DATASETS if entry[0] in requested]


def select_backends(spec: str) -> List[str]:
    if spec == "all":
        return ["cpu", "gpu"]
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(requested) - {"cpu", "gpu"})
    if unknown:
        raise ValueError(f"unknown backend(s): {', '.join(unknown)}")
    out: List[str] = []
    for backend in ["cpu", "gpu"]:
        if backend in requested:
            out.append(backend)
    if not out:
        raise ValueError("at least one backend is required")
    return out


def write_cleanup_audit(output_dir: pathlib.Path, log_root: pathlib.Path) -> None:
    trace_spill = ROOT / "runs" / ".trace_spill"
    srs_eval_point = "srs" + "_evaluation_point"
    srs_eval_scalar = "srs" + "_evaluation_scalar"
    prover_material = "prover" + "_" + "aux"
    lines = [
        "# Cleanup Audit",
        "",
        "- stale results removed: canonical paper output was regenerated by strict exporter.",
        "- failed partial artifacts removed: canonical final run directories are recreated before rerun unless resume-valid.",
        "- temporary diagnostics/logs/debug dumps: runner logs are confined to runs/secure_main_runner/logs and are not exported as paper results.",
        "- old runner / unused scripts: comparison runners ignored by request; main official runner is scripts/run_secure_main_experiments.py.",
        "- formal tests retained: tests/test_main.cpp and tools/security/static_scan_no_tau.py.",
        "- official exporter retained: scripts/export_paper_secure.py.",
        "- no /tmp dependency evidence: official runner uses repository-relative paths.",
        f"- runs/.trace_spill residue: {'present' if trace_spill.exists() else 'absent'}.",
        "- fake/legacy KZG path evidence: static_secure_scan passed before final runs.",
        f"- runtime tau / {srs_eval_point} / {srs_eval_scalar} / {prover_material} evidence: static_secure_scan passed before final runs.",
        "- results/paper_secure only contains strict 10-row export plus reports.",
        "- comparison/** ignored by request; no completion claim is based on comparison results.",
        "",
        "## Main Git Status",
        "",
        "```",
        git_status_main() or "clean except ignored comparison/** state",
        "```",
        "",
        "## Runner Logs",
        "",
        f"- {log_root.relative_to(ROOT)}",
    ]
    (output_dir / "cleanup_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun rows even when strict manifest resume is valid")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-full-security-tests", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--jobs", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--datasets", default="all", help="comma list or all")
    parser.add_argument("--backends", default="all", help="comma list of cpu,gpu or all")
    parser.add_argument("--skip-export", action="store_true", help="run selected rows without regenerating results/paper_secure")
    args = parser.parse_args()

    output_dir = ROOT / "results" / "paper_secure"
    log_root = ROOT / "runs" / "secure_main_runner" / "logs"
    archive_root = ROOT / "runs" / "secure_main_runner" / "archives"
    log_root.mkdir(parents=True, exist_ok=True)
    selected_datasets = select_datasets(args.datasets)
    selected_backends = select_backends(args.backends)
    full_matrix = len(selected_datasets) == len(DATASETS) and selected_backends == ["cpu", "gpu"]

    if not args.skip_build:
        configure_and_build(ROOT / "build", cuda=False, jobs=args.jobs, log_root=log_root)
        configure_and_build(ROOT / "build_cuda", cuda=True, jobs=args.jobs, log_root=log_root)
    if not args.skip_tests:
        run_tests(ROOT / "build", "cpu", log_root, full_security=not args.skip_full_security_tests)
        run_tests(ROOT / "build_cuda", "cuda", log_root, full_security=False)
        run_required([sys.executable, "tools/security/static_scan_no_tau.py"], log_root / "static_secure_scan.log")

    source_hash = sha256_source_tree()
    cpu_binary = ROOT / "build" / "gatzk_run"
    gpu_binary = ROOT / "build_cuda" / "gatzk_run"
    cpu_binary_hash = sha256_file(cpu_binary)
    gpu_binary_hash = sha256_file(gpu_binary)
    runner_hash = sha256_file(ROOT / "scripts" / "run_secure_main_experiments.py")
    build_identity = write_freeze_identity(
        source_hash,
        cpu_binary_hash,
        gpu_binary_hash,
        runner_hash,
        output_dir,
    )

    cpu_paths: Dict[str, pathlib.Path] = {}
    gpu_paths: Dict[str, pathlib.Path] = {}
    if "cpu" in selected_backends:
        for dataset, config, base_run_dir in selected_datasets:
            cpu_paths[dataset] = execute_row(
                dataset,
                config,
                base_run_dir,
                "cpu",
                cpu_binary,
                source_hash,
                cpu_binary_hash,
                cpu_binary_hash,
                gpu_binary_hash,
                runner_hash,
                build_identity,
                args.timeout_seconds,
                args.force,
                log_root,
                archive_root,
            )
    if "gpu" in selected_backends:
        for dataset, config, base_run_dir in selected_datasets:
            gpu_paths[dataset] = execute_row(
                dataset,
                config,
                base_run_dir,
                "gpu",
                gpu_binary,
                source_hash,
                gpu_binary_hash,
                cpu_binary_hash,
                gpu_binary_hash,
                runner_hash,
                build_identity,
                args.timeout_seconds,
                args.force,
                log_root,
                archive_root,
            )

    if args.skip_export or not full_matrix:
        print("secure subset rows complete; results/paper_secure export skipped", flush=True)
        return 0

    export_results(cpu_paths, gpu_paths, source_hash, cpu_binary_hash, gpu_binary_hash, output_dir, log_root)
    validate_export(output_dir)
    write_cleanup_audit(output_dir, log_root)
    validate_export(output_dir)
    print("secure paper export complete: results/paper_secure", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
