#!/usr/bin/env python3
"""Thin runner for the GAT-ZKML ezkl baseline.

The C++ exporter owns dataset loading, graph normalization, checkpoint checks,
hashing, GAT semantics metadata, and strict-route export decisions. This runner
only invokes the exporter, invokes ezkl if a strict ONNX export is available,
and writes orchestration status files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/home/pzh/project")
EXPORTER = EZKL_ROOT / "build_gat_adapter" / "gat_ezkl_export"
DATASETS = {
    "cora_full": "runs/cora_full",
    "citeseer_full": "runs/citeseer_full",
    "pubmed_full": "runs/pubmed_full",
    "ppi_full_formal": "runs/ppi_full_formal",
    "ogbn_arxiv_full": "runs/ogbn_arxiv_full",
}
CPU_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"
GPU_ROUTE = "ezkl_gat_native_compressed_csr_attention_full_gpu"
REQUIRED_GPU_BACKEND_LIBS = [
    "libicicle_backend_cuda_device.so",
    "libicicle_backend_cuda_curve_bn254.so",
    "libicicle_backend_cuda_field_bn254.so",
    "libicicle_backend_cuda_hash.so",
]
ROUTES = {
    CPU_ROUTE: {
        "backend": "cpu",
        "run_subdir": "",
        "statement_route_id": CPU_ROUTE,
        "gpu_backend": "",
    },
    GPU_ROUTE: {
        "backend": "gpu",
        "run_subdir": "gpu",
        "statement_route_id": CPU_ROUTE,
        "gpu_backend": "icicle_cuda",
    },
}
NATIVE_DEBUG_SCOPES = [
    "level0_mul",
    "inverse_only",
    "segment_sum_inverse",
    "segment_max_only",
    "lrelu_lookup_only",
    "exp_lookup_only",
    "elu_lookup_only",
    "compressed_projection_only",
    "compressed_aggregation_only",
    "single_head_first_layer",
    "first_hidden_layer_all_heads",
    "hidden_only",
    "output_only",
    "output_projection_only",
    "output_attention_only",
    "output_aggregation_only",
    "output_bias_add_only",
    "output_average_only",
    "final_logits_public_binding_only",
    "output_full_derivation",
    "full_model",
]


def ensure_inside(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"path escapes ezkl root: {path}")


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, float, str, str]:
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return completed.returncode, elapsed_ms, completed.stdout, completed.stderr


def parse_time_verbose(path: Path) -> dict:
    if not path.exists():
        return {}
    data: dict[str, object] = {"raw_path": str(path)}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Maximum resident set size" in line:
            try:
                peak_rss_kbytes = int(line.rsplit(":", 1)[1].strip())
                data["peak_rss_kbytes"] = peak_rss_kbytes
                data["peak_rss_bytes"] = peak_rss_kbytes * 1024
            except ValueError:
                pass
        elif line.startswith("Command terminated by signal"):
            try:
                data["signal"] = int(line.rsplit(" ", 1)[1].strip())
            except ValueError:
                data["signal"] = line.rsplit(" ", 1)[1].strip()
        elif "Command being timed" in line:
            data["time_command"] = line.rsplit(":", 1)[1].strip().strip('"')
        elif "Elapsed (wall clock) time" in line:
            data["elapsed_wall_clock"] = line.rsplit(":", 1)[1].strip()
        elif "Exit status" in line:
            data["exit_status"] = line.rsplit(":", 1)[1].strip()
    return data


def run_command_profiled(
    command: list[str],
    cwd: Path,
    profile_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, float, str, str, dict]:
    time_bin = shutil.which("time")
    if not time_bin:
        code, elapsed_ms, stdout, stderr = run_command(command, cwd, env=env)
        return code, elapsed_ms, stdout, stderr, {}
    time_txt = profile_path.with_suffix(".time.txt")
    wrapped = [time_bin, "-v", "-o", str(time_txt)] + command
    code, elapsed_ms, stdout, stderr = run_command(wrapped, cwd, env=env)
    profile = parse_time_verbose(time_txt)
    return code, elapsed_ms, stdout, stderr, profile


def native_command_env(
    run_dir: Path | None = None,
    timing_key: str | None = None,
    backend: str = "cpu",
    icicle_backend_dir: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("RAYON_NUM_THREADS", "4")
    env.setdefault("MALLOC_ARENA_MAX", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("RUST_BACKTRACE", "0")
    run_parts = set(run_dir.parts) if run_dir is not None else set()
    if run_dir is not None and "ogbn_arxiv_full" in run_parts and timing_key == "prove_time_ms":
        env["RAYON_NUM_THREADS"] = "1"
        env["MALLOC_ARENA_MAX"] = "1"
        env.setdefault("MALLOC_CONF", "background_thread:false,narenas:1,dirty_decay_ms:0,muzzy_decay_ms:0")
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["RUST_BACKTRACE"] = "0"
        env.setdefault("RUST_LOG", "warn")
        env.setdefault("EZKL_NATIVE_MOCK_PRECHECK", "0")
    if backend == "gpu":
        env["EZKL_REQUIRE_GPU_BACKEND"] = "1"
        if icicle_backend_dir:
            env["ICICLE_BACKEND_INSTALL_DIR"] = icicle_backend_dir
        icicle_lib_dir = EZKL_ROOT / "target" / "release" / "deps" / "icicle" / "lib"
        if icicle_lib_dir.exists():
            prior = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = f"{icicle_lib_dir}:{prior}" if prior else str(icicle_lib_dir)
    return env


def host_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    return machine


def classify_library_arch(file_output: str) -> str:
    lowered = file_output.lower()
    if "x86-64" in lowered or "x86_64" in lowered:
        return "x86_64"
    if "aarch64" in lowered or "arm aarch64" in lowered or "arm64" in lowered:
        return "aarch64"
    return "unknown"


def default_icicle_backend_dir() -> str | None:
    env_dir = os.environ.get("ICICLE_BACKEND_INSTALL_DIR")
    if env_dir:
        return env_dir
    checkout_root = EZKL_ROOT / ".cargo_local" / "git" / "checkouts"
    candidates = list(checkout_root.glob("halo2-*/**/icicle/backend/cuda"))
    candidates.sort(key=lambda path: (0 if "1dd2090" in str(path) else 1, str(path)))
    for candidate in candidates:
        if all((candidate / name).exists() for name in REQUIRED_GPU_BACKEND_LIBS):
            return str(candidate)
    return None


def gpu_env_profile(env: dict[str, str]) -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "EZKL_REQUIRE_GPU_BACKEND",
        "ICICLE_BACKEND_INSTALL_DIR",
        "LD_LIBRARY_PATH",
    ]
    return {key: env[key] for key in keys if key in env}


def gpu_device_info() -> dict[str, object]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(query, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return {
            "available": False,
            "command": " ".join(query),
            "error": "nvidia_smi_not_found",
        }
    info: dict[str, object] = {
        "available": completed.returncode == 0,
        "command": " ".join(query),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0:
        devices = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                devices.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "driver_version": parts[2],
                    }
                )
        info["devices"] = devices
    return info


def gpu_device_summary(info: dict[str, object]) -> tuple[str, int]:
    devices = info.get("devices")
    if not isinstance(devices, list):
        return "", 0
    names = []
    for device in devices:
        if isinstance(device, dict) and device.get("name"):
            names.append(str(device["name"]))
    return (names[0] if names else "", len(devices))


def gpu_backend_binary_info(icicle_backend_dir: str | None) -> dict[str, object]:
    if not icicle_backend_dir:
        return {
            "available": False,
            "error": "icicle_backend_dir_not_set",
            "required_libraries": REQUIRED_GPU_BACKEND_LIBS,
            "missing_libraries": REQUIRED_GPU_BACKEND_LIBS,
            "arch_match": False,
        }
    backend_dir = Path(icicle_backend_dir)
    expected_arch = host_arch()
    rows = []
    missing = []
    for name in REQUIRED_GPU_BACKEND_LIBS:
        lib = backend_dir / name
        if not lib.exists():
            missing.append(name)
            rows.append({"name": name, "path": str(lib), "exists": False})
            continue
        completed = subprocess.run(["file", str(lib)], text=True, capture_output=True, check=False)
        file_output = completed.stdout.strip() or completed.stderr.strip()
        arch = classify_library_arch(file_output)
        rows.append(
            {
                "name": name,
                "path": str(lib),
                "exists": True,
                "returncode": completed.returncode,
                "file_output": file_output,
                "arch": arch,
                "arch_matches_host": arch == expected_arch,
            }
        )
    primary = rows[0] if rows else {}
    present_arches = sorted({str(row.get("arch")) for row in rows if row.get("exists")})
    return {
        "available": backend_dir.exists(),
        "path": str(backend_dir),
        "host_arch": expected_arch,
        "required_libraries": REQUIRED_GPU_BACKEND_LIBS,
        "missing_libraries": missing,
        "libraries": rows,
        "library_arches": present_arches,
        "primary_library_path": primary.get("path", ""),
        "primary_library_arch": primary.get("arch", ""),
        "arch_match": backend_dir.exists()
        and not missing
        and bool(rows)
        and all(row.get("arch_matches_host") for row in rows),
    }


def gpu_backend_ready(backend_info: dict[str, object], device_info: dict[str, object]) -> bool:
    return bool(
        backend_info.get("available")
        and backend_info.get("arch_match")
        and device_info.get("available")
        and gpu_device_summary(device_info)[1] > 0
    )


def gpu_manifest_fields(
    route_cfg: dict[str, str],
    command_env: dict[str, str],
    device_info: dict[str, object],
    backend_info: dict[str, object],
    active: bool,
) -> dict[str, object]:
    device_name, device_count = gpu_device_summary(device_info)
    return {
        "gpu_backend_required": True,
        "gpu_backend_active": active,
        "gpu_backend_name": route_cfg["gpu_backend"],
        "gpu_backend_library_path": backend_info.get("primary_library_path", ""),
        "gpu_backend_library_arch": backend_info.get("primary_library_arch", ""),
        "gpu_device_name": device_name,
        "gpu_device_count": device_count,
        "cpu_fallback_used": False,
    }


def classify_native_failure(timing_key: str, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "inverse_policy_hash_mismatch" in text:
        return "inverse_policy_hash_mismatch"
    if "quantized_inverse_lookup_mismatch" in text:
        return "quantized_inverse_lookup_mismatch"
    if "alpha_rescale_mismatch" in text:
        return "alpha_rescale_mismatch"
    if "native_full_witness_preflight_failed" in text:
        return "native_full_witness_preflight_failed"
    if "native_full_gat_forward_constraints_incomplete" in text:
        return "native_full_gat_forward_constraints_incomplete"
    if "final_logits_derivation_incomplete" in text:
        return "final_logits_derivation_incomplete"
    if "output_projection_constraint_failed" in text:
        return "output_projection_constraint_failed"
    if "output_attention_constraint_failed" in text:
        return "output_attention_constraint_failed"
    if "output_aggregation_constraint_failed" in text:
        return "output_aggregation_constraint_failed"
    if "output_bias_add_constraint_failed" in text:
        return "output_bias_add_constraint_failed"
    if "output_average_constraint_failed" in text:
        return "output_average_constraint_failed"
    if "final_logits_public_binding_mismatch" in text:
        return "final_logits_public_binding_mismatch"
    if "public_instance_encoding_mismatch" in text:
        return "public_instance_encoding_mismatch"
    if "native_constraint_synthesis_failed" in text:
        return "native_constraint_synthesis_failed"
    if "native_keygen" in text:
        return "native_keygen_failed"
    if "native_prover" in text:
        return "native_prover_failed"
    if "native_verify" in text:
        return "native_verify_failed"
    if (
        "gpu_backend_required" in text
        or "gpu_backend_warmup_failed" in text
        or "libicicle" in text
        or "device api not found" in text
        or "invalid_device" in text
    ):
        return "gpu_backend_unavailable"
    if "ezkl_compressed_csr_resource_boundary" in text:
        return "ezkl_compressed_csr_resource_boundary"
    if "ezkl_native_csr_compile_resource_boundary" in text or "required_logrows" in text:
        return "ezkl_native_csr_compile_resource_boundary"
    if "lookup" in text and "quant_cfg" in text:
        return "ezkl_native_lookup_tables_missing_from_quant_cfg"
    if "inverse" in text:
        return "ezkl_native_segment_inverse_implementation_failed"
    if timing_key == "verify_time_ms":
        return "ezkl_native_verify_failed"
    return "ezkl_native_csr_route_failed"


def read_native_failure(run_dir: Path) -> dict:
    path = run_dir / "native_constraint_failure.json"
    if path.exists():
        try:
            return read_json(path)
        except json.JSONDecodeError:
            return {}
    return {}


def optional_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def payload_sizes(run_dir: Path) -> list[dict[str, object]]:
    payloads = [
        ("pk.key", run_dir / "pk.key"),
        ("pk.srs", run_dir / "pk.srs"),
        ("node_features.f64.bin", run_dir / "node_features.f64.bin"),
        ("csr_segment_layout.bin", run_dir / "csr_segment_layout.bin"),
        ("formal_tables.bin", run_dir / "formal_tables.bin"),
        ("checkpoint_tensors.txt", run_dir / "checkpoint_tensors.txt"),
        ("witness.json", run_dir / "witness.json"),
        ("model.compiled", run_dir / "model.compiled"),
    ]
    rows = []
    for name, path in payloads:
        size = optional_file_size(path)
        if size is not None:
            rows.append({"name": name, "path": str(path), "size_bytes": size})
    rows.sort(key=lambda item: int(item["size_bytes"]), reverse=True)
    return rows


def env_profile(env: dict[str, str]) -> dict[str, str]:
    keys = [
        "RAYON_NUM_THREADS",
        "MALLOC_ARENA_MAX",
        "MALLOC_CONF",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "EZKL_NATIVE_MOCK_PRECHECK",
        "EZKL_REQUIRE_GPU_BACKEND",
        "ICICLE_BACKEND_INSTALL_DIR",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    ]
    return {key: env[key] for key in keys if key in env}


def column_profile(compiled: dict, internal: dict) -> dict[str, int | None]:
    return {
        "advice_columns": internal.get("advice_columns", compiled.get("advice_columns", 4)),
        "fixed_columns": internal.get("fixed_columns", compiled.get("fixed_columns", 12)),
        "instance_columns": internal.get(
            "instance_columns",
            compiled.get("instance_columns", compiled.get("public_instance_column_count")),
        ),
        "lookup_table_columns": internal.get(
            "lookup_table_columns", compiled.get("lookup_table_columns", 3)
        ),
    }


def write_prove_memory_profile(
    run_dir: Path,
    command: list[str],
    elapsed_ms: float,
    returncode: int,
    stdout: str,
    stderr: str,
    time_profile: dict,
    env: dict[str, str],
) -> dict:
    compiled = {}
    if (run_dir / "model.compiled").exists():
        try:
            compiled = read_json(run_dir / "model.compiled")
        except json.JSONDecodeError:
            compiled = {}
    internal = {}
    if (run_dir / "prove_memory_profile_internal.json").exists():
        try:
            internal = read_json(run_dir / "prove_memory_profile_internal.json")
        except json.JSONDecodeError:
            internal = {}
    estimate = compiled.get("estimate", {}) if isinstance(compiled.get("estimate"), dict) else {}
    columns = column_profile(compiled, internal)
    signal = time_profile.get("signal")
    if signal is None and returncode < 0:
        signal = -returncode
    if signal is None and returncode in {137, 247}:
        signal = 9
    profile = {
        "phase": internal.get("phase", "graph-gat-prove"),
        "peak_rss_bytes": time_profile.get("peak_rss_bytes"),
        "circuit_rows": internal.get("circuit_rows"),
        "assigned_row_upper_bound": internal.get(
            "assigned_row_upper_bound", estimate.get("total_rows")
        ),
        "logrows": internal.get("logrows", estimate.get("required_logrows")),
        "advice_columns": columns["advice_columns"],
        "fixed_columns": columns["fixed_columns"],
        "instance_columns": columns["instance_columns"],
        "lookup_table_columns": columns["lookup_table_columns"],
        "public_instance_length": internal.get(
            "public_instance_length",
            compiled.get("public_instance_total_length", compiled.get("public_instance_length")),
        ),
        "proof_backend": internal.get("proof_backend", "halo2_kzg_shplonk_blake2b"),
        "largest_payloads_loaded": internal.get("largest_payloads_loaded", payload_sizes(run_dir)),
        "witness_payload_size": internal.get("witness_payload_size", optional_file_size(run_dir / "witness.json")),
        "compiled_circuit_size": internal.get("compiled_circuit_size", optional_file_size(run_dir / "model.compiled")),
        "pk_size": internal.get("pk_size", optional_file_size(run_dir / "pk.key")),
        "proof_serialization_policy": internal.get(
            "proof_serialization_policy", "snark_json_written_after_create_proof_completes"
        ),
        "thread_env_settings": internal.get("thread_env_settings", env_profile(env)),
        "command": " ".join(command),
        "signal": signal,
        "returncode": returncode,
        "elapsed_ms": elapsed_ms,
        "elapsed_wall_clock": time_profile.get("elapsed_wall_clock"),
        "stderr_tail": tail(stderr),
        "stdout_tail": tail(stdout),
        "proof_generated": (run_dir / "proof.json").exists() and returncode == 0,
        "payload_deduplication": internal.get(
            "payload_deduplication",
            "single output derivation reused in graph-gat-prove when available",
        ),
        "witness_payload_policy": internal.get(
            "witness_payload_policy",
            "compact metadata JSON; binary tensor payloads stay in run artifact files",
        ),
        "setup_only_data_policy": internal.get(
            "setup_only_data_policy",
            "shape-only setup artifacts are not retained by prove",
        ),
        "mock_precheck_policy": internal.get("mock_precheck_policy"),
        "attempted_optimizations": [
            "prove-time payload deduplication for output derivation/public instance/binding/circuit",
            "compact witness metadata; binary witness conversion not used because witness.json is small",
            "drop setup-only data before prove; setup uses shape-only plan",
            "chunked public instance columns without logits duplication in runner",
            "ogbn low-memory environment and single-thread prove",
            "large full-dataset MockProver diagnostic precheck skipped for ogbn while proof and verify remain strict",
        ],
    }
    write_json(run_dir / "prove_memory_profile.json", profile)
    return profile


def update_manifest(run_dir: Path, **updates: object) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(updates)
    write_json(manifest_path, manifest)
    return manifest


def write_blocked_artifacts(run_dir: Path, manifest: dict) -> None:
    proof = {
        "status": manifest.get("status", "blocked"),
        "route_id": manifest.get("route_id"),
        "proof_generated": False,
        "block_reason": manifest.get("block_reason"),
        "limitation": manifest.get("limitation"),
    }
    verify = {
        "status": manifest.get("status", "blocked"),
        "verified": False,
        "verify_invoked": False,
        "block_reason": manifest.get("block_reason"),
    }
    write_json(run_dir / "proof.json", proof)
    write_json(run_dir / "verify.json", verify)


def find_ezkl_command() -> list[str]:
    binary = EZKL_ROOT / "target" / "release" / "ezkl"
    if binary.exists():
        return [str(binary)]
    return ["cargo", "run", "--release", "--bin", "ezkl", "--"]


def intended_native_graph_command(native_debug_scope: str | None = None) -> str:
    ezkl = find_ezkl_command()
    scope_flags = ["--native-debug-scope", native_debug_scope] if native_debug_scope else []
    commands = [
        ezkl
        + [
            "graph-gat-gen-settings",
            "--spec-path",
            "graph_attention_spec.json",
            "--csr-layout-path",
            "csr_segment_layout.json",
            "--settings-path",
            "settings.json",
        ]
        + scope_flags,
        ezkl
        + [
            "graph-gat-compile",
            "--spec-path",
            "graph_attention_spec.json",
            "--csr-layout-path",
            "csr_segment_layout.json",
            "--settings-path",
            "settings.json",
            "--compiled-circuit",
            "model.compiled",
        ]
        + scope_flags,
        ezkl
        + [
            "graph-gat-setup",
            "--compiled-circuit",
            "model.compiled",
            "--vk-path",
            "vk.key",
            "--pk-path",
            "pk.key",
        ],
        ezkl
        + [
            "graph-gat-gen-witness",
            "--compiled-circuit",
            "model.compiled",
            "--output",
            "witness.json",
        ],
        ezkl
        + [
            "graph-gat-prove",
            "--compiled-circuit",
            "model.compiled",
            "--witness",
            "witness.json",
            "--pk-path",
            "pk.key",
            "--proof-path",
            "proof.json",
        ],
        ezkl
        + [
            "graph-gat-verify",
            "--compiled-circuit",
            "model.compiled",
            "--proof-path",
            "proof.json",
            "--settings-path",
            "settings.json",
            "--vk-path",
            "vk.key",
            "--output",
            "verify.json",
        ],
    ]
    return " && ".join(" ".join(command) for command in commands)


def run_native_graph_pipeline(
    run_dir: Path,
    manifest: dict,
    native_debug_scope: str | None = None,
    backend: str = "cpu",
    icicle_backend_dir: str | None = None,
) -> dict:
    ezkl = find_ezkl_command()
    timings: dict[str, float | None] = {
        "gen_settings_ms": None,
        "compile_ms": None,
        "setup_ms": None,
        "gen_witness_ms": None,
        "prove_time_ms": None,
        "verify_time_ms": None,
    }
    scope_flags = []
    if native_debug_scope:
        scope_flags = ["--native-debug-scope", native_debug_scope]
    commands = [
        (
            "gen_settings_ms",
            ezkl
            + [
                "graph-gat-gen-settings",
                "--spec-path",
                "graph_attention_spec.json",
                "--csr-layout-path",
                "csr_segment_layout.json",
                "--settings-path",
                "settings.json",
            ]
            + scope_flags,
        ),
        (
            "compile_ms",
            ezkl
            + [
                "graph-gat-compile",
                "--spec-path",
                "graph_attention_spec.json",
                "--csr-layout-path",
                "csr_segment_layout.json",
                "--settings-path",
                "settings.json",
                "--compiled-circuit",
                "model.compiled",
            ]
            + scope_flags,
        ),
        (
            "setup_ms",
            ezkl
            + [
                "graph-gat-setup",
                "--compiled-circuit",
                "model.compiled",
                "--vk-path",
                "vk.key",
                "--pk-path",
                "pk.key",
            ],
        ),
        (
            "gen_witness_ms",
            ezkl
            + [
                "graph-gat-gen-witness",
                "--compiled-circuit",
                "model.compiled",
                "--output",
                "witness.json",
            ],
        ),
        (
            "prove_time_ms",
            ezkl
            + [
                "graph-gat-prove",
                "--compiled-circuit",
                "model.compiled",
                "--witness",
                "witness.json",
                "--pk-path",
                "pk.key",
                "--proof-path",
                "proof.json",
            ],
        ),
        (
            "verify_time_ms",
            ezkl
            + [
                "graph-gat-verify",
                "--compiled-circuit",
                "model.compiled",
                "--proof-path",
                "proof.json",
                "--settings-path",
                "settings.json",
                "--vk-path",
                "vk.key",
                "--output",
                "verify.json",
            ],
        ),
    ]
    stdout_all = []
    stderr_all = []
    for timing_key, command in commands:
        if timing_key == "setup_ms":
            env = native_command_env(run_dir, timing_key, backend, icicle_backend_dir)
            code, elapsed_ms, stdout, stderr, profile = run_command_profiled(
                command,
                run_dir,
                run_dir / "keygen_memory_profile.json",
                env=env,
            )
            compiled = {}
            if (run_dir / "model.compiled").exists():
                try:
                    compiled = read_json(run_dir / "model.compiled")
                except json.JSONDecodeError:
                    compiled = {}
            profile.update(
                {
                    "command": " ".join(command),
                    "public_instance_length": compiled.get("public_instance_total_length", compiled.get("public_instance_length")),
                    "instance_column_count": compiled.get("public_instance_column_count"),
                    "chunk_size": compiled.get("public_instance_chunk_size"),
                    "public_instance_layout": compiled.get("public_instance_layout"),
                    "compiled_size_bytes": (run_dir / "model.compiled").stat().st_size
                    if (run_dir / "model.compiled").exists()
                    else None,
                    "loaded_payloads": ["model.compiled", "shape metadata only"],
                    "keygen_phase": "graph-gat-setup",
                    "setup_payload_mode": "shape_only",
                    "elapsed_ms": elapsed_ms,
                    "returncode": code,
                }
            )
            write_json(run_dir / "keygen_memory_profile.json", profile)
        elif timing_key == "prove_time_ms":
            env = native_command_env(run_dir, timing_key, backend, icicle_backend_dir)
            code, elapsed_ms, stdout, stderr, prove_profile = run_command_profiled(
                command,
                run_dir,
                run_dir / "prove_memory_profile.json",
                env=env,
            )
            prove_profile = write_prove_memory_profile(
                run_dir,
                command,
                elapsed_ms,
                code,
                stdout,
                stderr,
                prove_profile,
                env,
            )
        else:
            code, elapsed_ms, stdout, stderr = run_command(
                command,
                run_dir,
                env=native_command_env(run_dir, timing_key, backend, icicle_backend_dir),
            )
        stdout_all.append(f"$ {' '.join(command)}\n{stdout}")
        stderr_all.append(f"$ {' '.join(command)}\n{stderr}")
        timings[timing_key] = elapsed_ms
        if code != 0:
            failure = classify_native_failure(timing_key, stdout, stderr)
            native_failure = read_native_failure(run_dir)
            if (
                timing_key == "setup_ms"
                and not native_failure
                and (code < 0 or (not stdout.strip() and not stderr.strip()))
            ):
                failure = "native_keygen_resource_failure"
            if (
                timing_key == "prove_time_ms"
                and not native_failure
                and (code < 0 or (not stdout.strip() and not stderr.strip()))
            ):
                failure = "native_prove_resource_failure"
            manifest.update(timings)
            manifest.update(
                {
                    "status": "blocked"
                    if (
                        "resource_boundary" in failure
                        or "resource_failure" in failure
                        or "lookup_tables" in failure
                        or "backend_unavailable" in failure
                        or "backend_incomplete" in failure
                        or failure == "inverse_policy_hash_mismatch"
                        or failure == "quantized_inverse_lookup_mismatch"
                        or failure == "alpha_rescale_mismatch"
                        or failure == "native_full_witness_preflight_failed"
                        or failure == "native_full_gat_forward_constraints_incomplete"
                        or failure == "final_logits_derivation_incomplete"
                    )
                    else "failed",
                    "block_reason": failure,
                    "failure_step": timing_key,
                    "failed_command": " ".join(command),
                    "command": " && ".join(" ".join(cmd) for _, cmd in commands),
                    "backend": backend,
                    "gpu_backend": manifest.get("gpu_backend", ""),
                    "gpu_env": manifest.get("gpu_env", {}),
                    "gpu_device_info": manifest.get("gpu_device_info", {}),
                    "native_debug_scope": native_debug_scope or manifest.get("native_debug_scope") or "full_model",
                    "strict_public_input_equivalent": False,
                    "full_gat_forward_in_ezkl_proof": False,
                    "verify_passed": False,
                    "proof_verification_status": "not_verified",
                    "proof_size_bytes": None,
                    "commitment_time_ms": timings.get("setup_ms"),
                    "stdout_tail": tail("\n".join(stdout_all)),
                    "stderr_tail": tail("\n".join(stderr_all)),
                    "failure_gate": native_failure.get("gate_name"),
                    "failure_region": native_failure.get("region_name"),
                    "failure_row": native_failure.get("row_offset"),
                    "failure_relation": native_failure.get("expected_relation"),
                    "failure_values": native_failure.get("actual_values"),
                }
            )
            if (run_dir / "model.compiled").exists():
                try:
                    manifest["native_compile_plan"] = read_json(run_dir / "model.compiled")
                except json.JSONDecodeError:
                    pass
            if (run_dir / "compressed_row_estimate.json").exists():
                try:
                    manifest["compressed_row_estimate"] = read_json(run_dir / "compressed_row_estimate.json")
                except json.JSONDecodeError:
                    pass
            if (run_dir / "formal_table_hashes.json").exists():
                try:
                    manifest["formal_table_hashes"] = read_json(run_dir / "formal_table_hashes.json")
                except json.JSONDecodeError:
                    pass
            if (run_dir / "keygen_memory_profile.json").exists():
                try:
                    manifest["keygen_memory_profile"] = read_json(run_dir / "keygen_memory_profile.json")
                except json.JSONDecodeError:
                    pass
            if (run_dir / "prove_memory_profile.json").exists():
                try:
                    manifest["prove_memory_profile"] = read_json(run_dir / "prove_memory_profile.json")
                    if failure == "native_prove_resource_failure":
                        manifest["block_reason"] = "native_prove_memory_phase_resource_failure"
                        manifest["memory_failure_phase"] = manifest["prove_memory_profile"].get("phase")
                        manifest["memory_failure_peak_rss_bytes"] = manifest["prove_memory_profile"].get("peak_rss_bytes")
                except json.JSONDecodeError:
                    pass
            write_json(run_dir / "run_manifest.json", manifest)
            write_blocked_artifacts(run_dir, manifest)
            return manifest
    proof_path = run_dir / "proof.json"
    proof_size = proof_path.stat().st_size if proof_path.exists() else None
    verify = read_json(run_dir / "verify.json") if (run_dir / "verify.json").exists() else {}
    final_binding = read_json(run_dir / "final_logits_binding.json") if (run_dir / "final_logits_binding.json").exists() else {}
    strict_public = bool(verify.get("strict_public_input_equivalent", False))
    full_forward = bool(verify.get("full_gat_forward_in_ezkl_proof", False))
    manifest.update(timings)
    if (run_dir / "keygen_memory_profile.json").exists():
        try:
            manifest["keygen_memory_profile"] = read_json(run_dir / "keygen_memory_profile.json")
        except json.JSONDecodeError:
            pass
    if (run_dir / "prove_memory_profile.json").exists():
        try:
            manifest["prove_memory_profile"] = read_json(run_dir / "prove_memory_profile.json")
        except json.JSONDecodeError:
            pass
    manifest.update(
        {
            "status": "verified",
            "block_reason": "",
            "proof_size_bytes": proof_size,
            "commitment_time_ms": timings.get("setup_ms"),
            "check_mode": "safe",
            "strict_public_input_equivalent": strict_public,
            "full_gat_forward_in_ezkl_proof": full_forward,
            "final_logits_derivation_complete": bool(
                final_binding.get("final_logits_derivation_complete", False)
            ),
            "no_precomputed_logits_bypass": bool(
                final_binding.get("no_precomputed_logits_bypass", False)
            ),
            "verify_passed": bool(verify.get("verify_passed", True)),
            "proof_verification_status": "verified",
            "native_debug_scope": native_debug_scope or verify.get("native_debug_scope") or manifest.get("native_debug_scope") or "full_model",
            "backend": backend,
            "gpu_backend_required": backend == "gpu",
            "gpu_backend_active": backend == "gpu",
            "cpu_fallback_used": False if backend == "gpu" else None,
            "gpu_backend": manifest.get("gpu_backend", ""),
            "gpu_env": manifest.get("gpu_env", {}),
            "gpu_device_info": manifest.get("gpu_device_info", {}),
            "command": " && ".join(" ".join(cmd) for _, cmd in commands),
            "stdout_tail": tail("\n".join(stdout_all)),
            "stderr_tail": tail("\n".join(stderr_all)),
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "verify.json", {"status": "verified", "verified": True})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--mode", default="full", choices=["full"])
    parser.add_argument("--route", default=CPU_ROUTE, choices=sorted(ROUTES))
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--icicle-backend-dir")
    parser.add_argument("--native-debug-scope", choices=NATIVE_DEBUG_SCOPES)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    route_cfg = ROUTES[args.route]
    if args.backend != route_cfg["backend"]:
        raise RuntimeError(f"route {args.route} requires backend={route_cfg['backend']}")

    if not EXPORTER.exists():
        raise RuntimeError(f"C++ exporter is not built: {EXPORTER}")

    run_dir = EZKL_ROOT / DATASETS[args.dataset]
    if route_cfg["run_subdir"]:
        run_dir = run_dir / route_cfg["run_subdir"]
    if args.native_debug_scope:
        run_dir = run_dir / f"debug_{args.native_debug_scope}"
    ensure_inside(run_dir, EZKL_ROOT)
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(EXPORTER),
        "--dataset",
        args.dataset,
        "--project-root",
        str(PROJECT_ROOT),
        "--ezkl-root",
        str(EZKL_ROOT),
        "--output-dir",
        str(run_dir),
    ]
    if args.route:
        command.extend(["--route", args.route])
    code, exporter_ms, stdout, stderr = run_command(command, EZKL_ROOT)
    (run_dir / "exporter.stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "exporter.stderr.txt").write_text(stderr, encoding="utf-8")
    if code != 0:
        manifest = {
            "dataset": args.dataset,
            "status": "failed",
            "block_reason": "exporter_failure",
            "exporter_ms": exporter_ms,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
        }
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(run_dir / "proof.json", {"status": "failed", "proof_generated": False})
        write_json(run_dir / "verify.json", {"status": "failed", "verified": False})
        print(f"dataset={args.dataset} status=failed block_reason=exporter_failure")
        return 1

    icicle_backend_dir = args.icicle_backend_dir or default_icicle_backend_dir()
    command_env = native_command_env(run_dir, None, args.backend, icicle_backend_dir)
    device_info: dict[str, object] = {}
    backend_info: dict[str, object] = {}
    gpu_fields: dict[str, object] = {}
    if args.backend == "gpu":
        device_info = gpu_device_info()
        backend_info = gpu_backend_binary_info(icicle_backend_dir)
        gpu_fields = gpu_manifest_fields(route_cfg, command_env, device_info, backend_info, False)
    manifest = update_manifest(
        run_dir,
        exporter_ms=exporter_ms,
        backend=args.backend,
        statement_route_id=route_cfg["statement_route_id"],
        route_statement_equivalence="same_dataset_checkpoint_preprocessing_quantization_and_final_logits_semantics",
        gpu_backend=route_cfg["gpu_backend"],
        gpu_env=gpu_env_profile(command_env) if args.backend == "gpu" else {},
        gpu_device_info=device_info if args.backend == "gpu" else {},
        gpu_backend_binary_info=backend_info if args.backend == "gpu" else {},
        **gpu_fields,
        native_debug_scope=args.native_debug_scope or "full_model",
        scope=args.native_debug_scope or "full_model",
        stdout_tail=tail(stdout),
        stderr_tail=tail(stderr),
    )
    if args.backend == "gpu" and not gpu_backend_ready(backend_info, device_info):
        arch_detail = backend_info.get("primary_library_arch") or ",".join(
            str(item) for item in backend_info.get("library_arches", [])
        )
        manifest.update(
            {
                "status": "blocked",
                "block_reason": "gpu_backend_unavailable",
                "failure_step": "gpu_backend_preflight",
                "limitation": (
                    "icicle_cuda_backend_unusable_on_host_arch"
                    if not backend_info.get("arch_match")
                    else "icicle_cuda_backend_or_device_unavailable"
                ),
                "backend": "gpu",
                "gpu_backend_required": True,
                "gpu_backend_active": False,
                "cpu_fallback_used": False,
                "gpu_backend_library_arch": arch_detail,
                "strict_public_input_equivalent": False,
                "full_gat_forward_in_ezkl_proof": False,
                "verify_passed": False,
                "proof_verification_status": "not_verified",
                "proof_size_bytes": None,
                "commitment_time_ms": None,
                "prove_time_ms": None,
                "verify_time_ms": None,
                "command": intended_native_graph_command(args.native_debug_scope),
            }
        )
        write_json(run_dir / "run_manifest.json", manifest)
        write_blocked_artifacts(run_dir, manifest)
        print(
            f"dataset={args.dataset} status=blocked "
            f"block_reason={manifest.get('block_reason')} route_id={manifest.get('route_id')}"
        )
        return 0
    if manifest.get("status") == "blocked":
        write_blocked_artifacts(run_dir, manifest)
        print(
            f"dataset={args.dataset} status=blocked "
            f"block_reason={manifest.get('block_reason')} route_id={manifest.get('route_id')}"
        )
        return 0

    if args.run:
        manifest = run_native_graph_pipeline(
            run_dir,
            manifest,
            args.native_debug_scope,
            args.backend,
            icicle_backend_dir,
        )
        print(f"dataset={args.dataset} status={manifest.get('status')} route_id={manifest.get('route_id')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
