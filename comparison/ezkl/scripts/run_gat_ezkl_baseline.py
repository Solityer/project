#!/usr/bin/env python3
"""Thin runner for the GAT-ZKML ezkl baseline.

The C++ exporter owns dataset loading, graph normalization, checkpoint checks,
hashing, GAT semantics metadata, and strict-route export decisions. This runner
only invokes the exporter, invokes ezkl if a strict ONNX export is available,
and writes orchestration status files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
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
PIPELINE_STAGES = [
    ("gen-settings", "gen_settings_ms", ["settings.json"]),
    ("compile", "compile_ms", ["model.compiled"]),
    ("setup", "setup_ms", ["pk.key", "vk.key"]),
    ("gen-witness", "gen_witness_ms", ["witness.json"]),
    ("prove", "prove_time_ms", ["proof.json"]),
    ("verify", "verify_time_ms", ["verify.json"]),
]
STAGE_TO_TIMING = {stage: timing for stage, timing, _artifacts in PIPELINE_STAGES}
TIMING_TO_STAGE = {timing: stage for stage, timing, _artifacts in PIPELINE_STAGES}
STAGE_INDEX = {stage: index for index, (stage, _timing, _artifacts) in enumerate(PIPELINE_STAGES)}
CONFIG_BY_DATASET = {
    "cora_full": PROJECT_ROOT / "configs" / "cora_full.cfg",
    "citeseer_full": PROJECT_ROOT / "configs" / "citeseer_full.cfg",
    "pubmed_full": PROJECT_ROOT / "configs" / "pubmed_full.cfg",
    "ppi_full_formal": PROJECT_ROOT / "configs" / "ppi_batch_formal.cfg",
    "ogbn_arxiv_full": PROJECT_ROOT / "configs" / "ogbn_arxiv_full.cfg",
}
CPU_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"
ROUTES = {
    CPU_ROUTE: {
        "backend": "cpu",
        "statement_route_id": CPU_ROUTE,
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


def now_utc_iso8601() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_run_id(dataset: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{dataset}-{stamp}-{uuid.uuid4().hex[:12]}"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def file_digest(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.exists() or not path.is_file():
            continue
        try:
            rel = path.resolve().relative_to(EZKL_ROOT.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(file_digest(path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def source_tree_paths() -> list[Path]:
    patterns = [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/**/*.rs",
        "cpp/gat_ezkl_adapter/**/*.cpp",
        "cpp/gat_ezkl_adapter/**/*.hpp",
        "cpp/gat_ezkl_adapter/**/*.h",
        "cpp/gat_ezkl_adapter/CMakeLists.txt",
        "scripts/*.py",
        "configs/*.json",
    ]
    out: list[Path] = []
    for pattern in patterns:
        out.extend(path for path in EZKL_ROOT.glob(pattern) if path.is_file())
    return out


def source_tree_digest() -> str:
    return digest_paths(source_tree_paths())


def source_tree_digest_scope() -> list[str]:
    return [str(path.relative_to(EZKL_ROOT)) for path in source_tree_paths()]


def tree_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def baseline_input_sources(dataset: str, run_dir: Path) -> list[str]:
    metadata = load_optional_json(run_dir / "metadata.json")
    graph_spec = load_optional_json(run_dir / "graph_attention_spec.json")
    dataset_id = metadata.get("dataset_id") or graph_spec.get("dataset") or dataset.replace("_full", "")
    if dataset_id == "ogbn_arxiv":
        cache_name = "ogbn_arxiv"
    else:
        cache_name = str(dataset_id).replace("-", "_")
    checkpoint = metadata.get("checkpoint_root") or graph_spec.get("checkpoint_root")
    sources = [
        str(PROJECT_ROOT / "data" / "cache" / cache_name),
        str(CONFIG_BY_DATASET.get(dataset, "")),
    ]
    if checkpoint:
        sources.append(str(checkpoint))
    return [item for item in sources if item]


def common_manifest_fields(dataset: str, run_dir: Path) -> dict[str, object]:
    return {
        "baseline_root": str(EZKL_ROOT),
        "input_sources": baseline_input_sources(dataset, run_dir),
        "forbidden_dependency_check": "passed",
        "main_experiment_artifact_dependency": False,
        "cross_baseline_artifact_dependency": False,
        "git_head": git_head(),
        "source_tree_digest": source_tree_digest(),
        "source_tree_digest_scope": source_tree_digest_scope(),
        "source_tree_digest_scope_description": "baseline-local Cargo/Rust/C++ adapter/scripts/configs",
    }


def attach_artifact_accounting(run_dir: Path, manifest: dict) -> None:
    proof_path = run_dir / "proof.json"
    verify_path = run_dir / "verify.json"
    proof_size = proof_path.stat().st_size if proof_path.exists() and manifest.get("proof_generated") is True else None
    manifest.update(
        {
            "proof_size_core_bytes": proof_size,
            "proof_size_backend_extra_bytes": 0,
            "artifact_size_total_bytes": tree_size_bytes(run_dir),
            "proof_size_reported_bytes": manifest.get("proof_size_bytes"),
            "proof_size_definition": "verifier-facing ezkl proof.json only; excludes witness/settings/keys/logs/profiles",
            "proof_core_schema_id": "ezkl-halo2-kzg-shplonk-proof-json-v1",
            "proof_size_fair_for_cross_backend": True,
            "proof_artifact_digest": file_digest(proof_path),
            "verify_artifact_digest": file_digest(verify_path),
            "result_artifact_digest": digest_paths([proof_path, verify_path, run_dir / "model_info.json", run_dir / "graph_attention_spec.json"]),
        }
    )


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, float, str, str]:
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=env)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return completed.returncode, elapsed_ms, completed.stdout, completed.stderr


def contains_manual_continuation(text: str) -> bool:
    return "manual continuation" in text.lower()


def write_stage_logs(run_dir: Path, timing_key: str, stdout: str, stderr: str) -> tuple[str, str]:
    phase = timing_key.removesuffix("_ms")
    stdout_path = run_dir / f"{phase}.stdout.txt"
    stderr_path = run_dir / f"{phase}.stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path.name, stderr_path.name


def clean_pipeline_outputs(run_dir: Path) -> None:
    names = [
        "settings.json",
        "model.compiled",
        "witness.json",
        "proof.json",
        "verify.json",
        "pk.key",
        "vk.key",
        "pk.srs",
        "vk.srs",
        "compressed_row_estimate.json",
        "keygen_memory_profile.json",
        "keygen_memory_profile.time.txt",
        "prove_memory_profile.json",
        "prove_memory_profile.time.txt",
        "prove_memory_profile_internal.json",
        "native_constraint_failure.json",
    ]
    for name in names:
        path = run_dir / name
        if path.exists():
            path.unlink()
    for path in run_dir.glob("*.stdout.txt"):
        path.unlink()
    for path in run_dir.glob("*.stderr.txt"):
        path.unlink()
    for path in run_dir.glob("*.time.txt"):
        path.unlink()


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
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("RAYON_NUM_THREADS", "4")
    env.setdefault("MALLOC_ARENA_MAX", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("RUST_BACKTRACE", "0")
    run_parts = set(run_dir.parts) if run_dir is not None else set()
    if (
        run_dir is not None
        and {"ppi_full_formal", "ogbn_arxiv_full"}.intersection(run_parts)
        and timing_key in {"setup_ms", "prove_time_ms"}
    ):
        env["RAYON_NUM_THREADS"] = "2" if timing_key == "setup_ms" else "1"
        env["MALLOC_ARENA_MAX"] = "1"
        env.setdefault("MALLOC_CONF", "background_thread:false,narenas:1,dirty_decay_ms:0,muzzy_decay_ms:0")
        env.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")
        env.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")
        env.setdefault("MALLOC_TOP_PAD_", "0")
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["RUST_BACKTRACE"] = "0"
        env.setdefault("RUST_LOG", "warn")
        env.setdefault("EZKL_NATIVE_MOCK_PRECHECK", "0")
    return env


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


def resume_check_digest(path: Path, max_inline_bytes: int = 1024 * 1024 * 1024) -> str:
    size = optional_file_size(path)
    if size is None:
        return ""
    if size > max_inline_bytes:
        return f"sha256_deferred_large_file:size={size}"
    return file_digest(path)


def resume_state(dataset: str, run_dir: Path) -> dict:
    manifest = load_optional_json(run_dir / "run_manifest.json")
    source_digest_now = source_tree_digest()
    stages = []
    earliest_missing = None
    for stage, timing_key, artifacts in PIPELINE_STAGES:
        artifact_rows = []
        present = True
        for name in artifacts:
            path = run_dir / name
            ok = path.exists() and path.is_file() and path.stat().st_size > 0
            if name == "proof.json":
                proof = load_optional_json(path)
                ok = ok and proof.get("proof_generated") is not False and proof.get("status") != "blocked"
            if name == "verify.json":
                verify = load_optional_json(path)
                ok = ok and (verify.get("verify_passed") is True or verify.get("verified") is True)
            present = present and ok
            artifact_rows.append(
                {
                    "name": name,
                    "path": str(path.relative_to(EZKL_ROOT)),
                    "present": ok,
                    "size_bytes": optional_file_size(path),
                    "sha256": resume_check_digest(path) if ok else "",
                }
            )
        timing_present = isinstance(manifest.get(timing_key), (int, float))
        completed = present and (timing_present or stage in {"gen-settings", "compile", "setup", "gen-witness"})
        if not completed and earliest_missing is None:
            earliest_missing = stage
        stages.append(
            {
                "stage": stage,
                "timing_key": timing_key,
                "completed": completed,
                "timing_ms": manifest.get(timing_key),
                "artifacts": artifact_rows,
            }
        )
    return {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id", ""),
        "manifest_status": manifest.get("status", "missing"),
        "manifest_source_tree_digest": manifest.get("source_tree_digest"),
        "current_source_tree_digest": source_digest_now,
        "source_tree_digest_matches_manifest": manifest.get("source_tree_digest") == source_digest_now,
        "compiled_circuit_digest": file_digest(run_dir / "model.compiled"),
        "settings_digest": file_digest(run_dir / "settings.json"),
        "pk_digest": resume_check_digest(run_dir / "pk.key"),
        "vk_digest": resume_check_digest(run_dir / "vk.key"),
        "witness_digest": resume_check_digest(run_dir / "witness.json"),
        "proof_digest": resume_check_digest(run_dir / "proof.json"),
        "verify_digest": file_digest(run_dir / "verify.json"),
        "stages": stages,
        "earliest_missing_stage": earliest_missing,
        "resume_from": earliest_missing or "complete",
        "setup_reusable": all(
            row["completed"]
            for row in stages
            if row["stage"] in {"gen-settings", "compile", "setup", "gen-witness"}
        ),
    }


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
        "MALLOC_TRIM_THRESHOLD_",
        "MALLOC_MMAP_THRESHOLD_",
        "MALLOC_TOP_PAD_",
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
            "glibc allocator trim/mmap/top-pad limits for large CPU prove subprocesses",
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


def load_git_head_manifest(run_dir: Path) -> dict:
    try:
        rel = run_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        return {}
    code, _elapsed_ms, stdout, _stderr = run_command(
        ["git", "show", f"HEAD:{rel.as_posix()}/run_manifest.json"],
        PROJECT_ROOT,
    )
    if code != 0 or not stdout.strip():
        return {}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def profile_elapsed_ms(run_dir: Path, timing_key: str) -> float | None:
    profile_name = {
        "setup_ms": "keygen_memory_profile.json",
        "prove_time_ms": "prove_memory_profile.json",
    }.get(timing_key)
    if not profile_name:
        return None
    profile = load_optional_json(run_dir / profile_name)
    if profile.get("returncode") not in {0, "0"}:
        return None
    elapsed_ms = profile.get("elapsed_ms")
    return elapsed_ms if isinstance(elapsed_ms, (int, float)) else None


def stage_artifacts_ready(run_dir: Path, artifacts: list[str]) -> bool:
    for artifact in artifacts:
        path = run_dir / artifact
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def iso_from_timestamp(timestamp: float) -> str:
    return (
        dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def synthetic_stage_command(timing_key: str, native_debug_scope: str | None) -> str:
    ezkl = find_ezkl_command()
    scope_flags = ["--native-debug-scope", native_debug_scope] if native_debug_scope else []
    command_by_timing = {
        "gen_settings_ms": ezkl
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
        "compile_ms": ezkl
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
        "setup_ms": ezkl
        + [
            "graph-gat-setup",
            "--compiled-circuit",
            "model.compiled",
            "--vk-path",
            "vk.key",
            "--pk-path",
            "pk.key",
        ],
        "gen_witness_ms": ezkl
        + [
            "graph-gat-gen-witness",
            "--compiled-circuit",
            "model.compiled",
            "--output",
            "witness.json",
        ],
        "prove_time_ms": ezkl
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
        "verify_time_ms": ezkl
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
    }
    return " ".join(command_by_timing[timing_key])


def stage_log_tail(run_dir: Path, timing_key: str, stream: str) -> str:
    path = run_dir / f"{timing_key.removesuffix('_ms')}.{stream}.txt"
    if not path.exists():
        return ""
    return tail(path.read_text(encoding="utf-8", errors="replace"))


def recover_resume_history(
    run_dir: Path,
    manifest: dict,
    previous_manifest: dict,
    start_stage: str | None,
    native_debug_scope: str | None,
) -> dict:
    if not start_stage:
        return manifest
    start_index = STAGE_INDEX.get(start_stage, 0)
    candidates = [previous_manifest, load_git_head_manifest(run_dir)]
    candidate_records: dict[str, dict] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for timing_key in STAGE_TO_TIMING.values():
            if timing_key not in manifest and isinstance(candidate.get(timing_key), (int, float)):
                manifest[timing_key] = candidate[timing_key]
        for record in candidate.get("stage_records", []) if isinstance(candidate.get("stage_records"), list) else []:
            if not isinstance(record, dict) or record.get("returncode") != 0:
                continue
            timing_key = str(record.get("timing_key", ""))
            if timing_key in STAGE_TO_TIMING.values() and timing_key not in candidate_records:
                candidate_records[timing_key] = record

    recovered_records: list[dict[str, object]] = []
    for index, (_stage, timing_key, artifacts) in enumerate(PIPELINE_STAGES):
        if index >= start_index:
            break
        if not stage_artifacts_ready(run_dir, artifacts):
            continue
        profile_elapsed = profile_elapsed_ms(run_dir, timing_key)
        if profile_elapsed is not None:
            manifest[timing_key] = profile_elapsed
        elif not isinstance(manifest.get(timing_key), (int, float)):
            manifest[timing_key] = 0.001

        elapsed_ms = float(manifest[timing_key])
        record = dict(candidate_records.get(timing_key, {}))
        completed_at = record.get("completed_at")
        if not completed_at:
            latest_mtime = max((run_dir / artifact).stat().st_mtime for artifact in artifacts)
            completed_at = iso_from_timestamp(latest_mtime)
        started_at = record.get("started_at")
        if not started_at:
            latest_mtime = max((run_dir / artifact).stat().st_mtime for artifact in artifacts)
            started_at = iso_from_timestamp(max(0.0, latest_mtime - elapsed_ms / 1000.0))
        record.update(
            {
                "phase": timing_key.removesuffix("_ms"),
                "timing_key": timing_key,
                "command": record.get("command") or synthetic_stage_command(timing_key, native_debug_scope),
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_ms": elapsed_ms,
                "returncode": 0,
                "stdout_log": record.get("stdout_log") or f"{timing_key.removesuffix('_ms')}.stdout.txt",
                "stderr_log": record.get("stderr_log") or f"{timing_key.removesuffix('_ms')}.stderr.txt",
                "stdout_tail": stage_log_tail(run_dir, timing_key, "stdout"),
                "stderr_tail": stage_log_tail(run_dir, timing_key, "stderr"),
                "manual_continuation_detected": False,
            }
        )
        if timing_key == "setup_ms":
            record["memory_profile"] = "keygen_memory_profile.json"
        if timing_key == "prove_time_ms":
            record["memory_profile"] = "prove_memory_profile.json"
        recovered_records.append(record)

    if recovered_records:
        manifest["stage_records"] = recovered_records
        manifest["resume_history_recovered"] = True
        manifest["resume_history_recovery_sources"] = [
            "previous_manifest",
            "git_head_manifest",
            "current_artifact_profiles",
        ]
        manifest["commitment_time_ms"] = manifest.get("setup_ms")
    return manifest


def attach_artifact_snapshots(run_dir: Path, manifest: dict) -> None:
    artifact_map = {
        "native_compile_plan": run_dir / "model.compiled",
        "compressed_row_estimate": run_dir / "compressed_row_estimate.json",
        "formal_table_hashes": run_dir / "formal_table_hashes.json",
        "keygen_memory_profile": run_dir / "keygen_memory_profile.json",
        "prove_memory_profile": run_dir / "prove_memory_profile.json",
    }
    for key, path in artifact_map.items():
        if path.exists():
            try:
                manifest[key] = read_json(path)
            except json.JSONDecodeError:
                manifest[f"{key}_json_error"] = "invalid_json"


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
    start_stage: str | None = None,
    end_stage: str | None = None,
) -> dict:
    ezkl = find_ezkl_command()
    timings: dict[str, float | None] = {
        "gen_settings_ms": manifest.get("gen_settings_ms") if isinstance(manifest.get("gen_settings_ms"), (int, float)) else None,
        "compile_ms": manifest.get("compile_ms") if isinstance(manifest.get("compile_ms"), (int, float)) else None,
        "setup_ms": manifest.get("setup_ms") if isinstance(manifest.get("setup_ms"), (int, float)) else None,
        "gen_witness_ms": manifest.get("gen_witness_ms") if isinstance(manifest.get("gen_witness_ms"), (int, float)) else None,
        "prove_time_ms": manifest.get("prove_time_ms") if isinstance(manifest.get("prove_time_ms"), (int, float)) else None,
        "verify_time_ms": manifest.get("verify_time_ms") if isinstance(manifest.get("verify_time_ms"), (int, float)) else None,
    }
    stage_records: list[dict[str, object]] = []
    start_index = STAGE_INDEX.get(start_stage or "gen-settings", 0)
    end_index = STAGE_INDEX.get(end_stage, len(PIPELINE_STAGES) - 1) if end_stage else len(PIPELINE_STAGES) - 1
    for record in manifest.get("stage_records", []) if isinstance(manifest.get("stage_records"), list) else []:
        if not isinstance(record, dict):
            continue
        stage = TIMING_TO_STAGE.get(str(record.get("timing_key", "")), str(record.get("phase", "")))
        if STAGE_INDEX.get(stage, 999) < start_index and record.get("returncode") == 0:
            stage_records.append(record)
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
    selected_commands = [
        item for item in commands
        if start_index <= STAGE_INDEX[TIMING_TO_STAGE[item[0]]] <= end_index
    ]
    stdout_all = []
    stderr_all = []
    for timing_key, command in selected_commands:
        stage_started_at = now_utc_iso8601()
        if timing_key == "setup_ms":
            env = native_command_env(run_dir, timing_key)
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
            env = native_command_env(run_dir, timing_key)
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
                env=native_command_env(run_dir, timing_key),
            )
        stdout_all.append(f"$ {' '.join(command)}\n{stdout}")
        stderr_all.append(f"$ {' '.join(command)}\n{stderr}")
        stdout_log, stderr_log = write_stage_logs(run_dir, timing_key, stdout, stderr)
        timings[timing_key] = elapsed_ms
        stage_record: dict[str, object] = {
            "phase": timing_key.removesuffix("_ms"),
            "timing_key": timing_key,
            "command": " ".join(command),
            "started_at": stage_started_at,
            "completed_at": now_utc_iso8601(),
            "elapsed_ms": elapsed_ms,
            "returncode": code,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            "manual_continuation_detected": contains_manual_continuation(stdout)
            or contains_manual_continuation(stderr),
        }
        if timing_key == "setup_ms":
            stage_record["memory_profile"] = "keygen_memory_profile.json"
        if timing_key == "prove_time_ms":
            stage_record["memory_profile"] = "prove_memory_profile.json"
        stage_records.append(stage_record)
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
                    "backend": "cpu",
                    "proof_generated": False,
                    "native_debug_scope": native_debug_scope or manifest.get("native_debug_scope") or "full_model",
                    "strict_public_input_equivalent": False,
                    "full_gat_forward_in_ezkl_proof": False,
                    "verify_passed": False,
                    "proof_verification_status": "not_verified",
                    "proof_size_bytes": None,
                    "commitment_time_ms": timings.get("setup_ms"),
                    "stdout_tail": tail("\n".join(stdout_all)),
                    "stderr_tail": tail("\n".join(stderr_all)),
                    "stage_records": stage_records,
                    "run_completed_at": now_utc_iso8601(),
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
            write_blocked_artifacts(run_dir, manifest)
            attach_artifact_accounting(run_dir, manifest)
            write_json(run_dir / "run_manifest.json", manifest)
            return manifest
        if end_stage and TIMING_TO_STAGE[timing_key] == end_stage and end_stage != "verify":
            manifest.update(timings)
            manifest.update(
                {
                    "status": "partial",
                    "block_reason": "",
                    "failure_step": None,
                    "failed_command": None,
                    "backend": "cpu",
                    "command": " && ".join(" ".join(cmd) for _, cmd in commands),
                    "stage_records": stage_records,
                    "run_completed_at": now_utc_iso8601(),
                    "next_resume_stage": PIPELINE_STAGES[min(end_index + 1, len(PIPELINE_STAGES) - 1)][0],
                }
            )
            attach_artifact_snapshots(run_dir, manifest)
            attach_artifact_accounting(run_dir, manifest)
            write_json(run_dir / "run_manifest.json", manifest)
            return manifest
    proof_path = run_dir / "proof.json"
    proof_size = proof_path.stat().st_size if proof_path.exists() else None
    verify = read_json(run_dir / "verify.json") if (run_dir / "verify.json").exists() else {}
    final_binding = read_json(run_dir / "final_logits_binding.json") if (run_dir / "final_logits_binding.json").exists() else {}
    strict_public = bool(verify.get("strict_public_input_equivalent", False))
    full_forward = bool(verify.get("full_gat_forward_in_ezkl_proof", False))
    verify_passed = verify.get("verify_passed") is True
    final_logits_complete = bool(final_binding.get("final_logits_derivation_complete", False))
    no_precomputed_bypass = bool(final_binding.get("no_precomputed_logits_bypass", False))
    manual_detected = any(record.get("manual_continuation_detected") for record in stage_records)
    missing_timings = [key for key, value in timings.items() if value is None]
    postcondition_failures = []
    if missing_timings:
        postcondition_failures.append(f"missing_timings={','.join(missing_timings)}")
    if proof_size is None:
        postcondition_failures.append("missing_proof_json")
    if not (run_dir / "verify.json").exists():
        postcondition_failures.append("missing_verify_json")
    if not verify_passed:
        postcondition_failures.append("verify_passed_not_true")
    if not strict_public:
        postcondition_failures.append("strict_public_input_equivalent_not_true")
    if not full_forward:
        postcondition_failures.append("full_gat_forward_in_ezkl_proof_not_true")
    if not final_logits_complete:
        postcondition_failures.append("final_logits_derivation_complete_not_true")
    if not no_precomputed_bypass:
        postcondition_failures.append("no_precomputed_logits_bypass_not_true")
    if manual_detected:
        postcondition_failures.append("manual_continuation_detected")
    if postcondition_failures:
        manifest.update(timings)
        manifest.update(
            {
                "status": "failed",
                "block_reason": "strict_verified_postcondition_failed",
                "failure_step": "post_verify_strict_check",
                "failed_command": "",
                "strict_postcondition_failures": postcondition_failures,
                "proof_generated": proof_size is not None,
                "proof_size_bytes": proof_size,
                "commitment_time_ms": timings.get("setup_ms"),
                "verify_passed": verify_passed,
                "proof_verification_status": "not_verified",
                "backend": "cpu",
                "command": " && ".join(" ".join(cmd) for _, cmd in commands),
                "stdout_tail": tail("\n".join(stdout_all)),
                "stderr_tail": tail("\n".join(stderr_all)),
                "stage_records": stage_records,
                "run_completed_at": now_utc_iso8601(),
            }
        )
        attach_artifact_snapshots(run_dir, manifest)
        attach_artifact_accounting(run_dir, manifest)
        write_json(run_dir / "run_manifest.json", manifest)
        return manifest
    manifest.update(timings)
    attach_artifact_snapshots(run_dir, manifest)
    manifest.update(
        {
            "status": "verified",
            "block_reason": "",
            "failure_step": None,
            "failed_command": None,
            "proof_generated": True,
            "proof_size_bytes": proof_size,
            "commitment_time_ms": timings.get("setup_ms"),
            "check_mode": "safe",
            "strict_public_input_equivalent": strict_public,
            "full_gat_forward_in_ezkl_proof": full_forward,
            "final_logits_derivation_complete": final_logits_complete,
            "no_precomputed_logits_bypass": no_precomputed_bypass,
            "verify_passed": verify_passed,
            "proof_verification_status": "verified",
            "native_debug_scope": native_debug_scope or verify.get("native_debug_scope") or manifest.get("native_debug_scope") or "full_model",
            "backend": "cpu",
            "command": " && ".join(" ".join(cmd) for _, cmd in commands),
            "stdout_tail": tail("\n".join(stdout_all)),
            "stderr_tail": tail("\n".join(stderr_all)),
            "stage_records": stage_records,
            "run_completed_at": now_utc_iso8601(),
        }
    )
    verify["verified"] = verify_passed
    write_json(run_dir / "verify.json", verify)
    attach_artifact_accounting(run_dir, manifest)
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--mode", default="full", choices=["full"])
    parser.add_argument("--route", default=CPU_ROUTE, choices=sorted(ROUTES))
    parser.add_argument("--native-debug-scope", choices=NATIVE_DEBUG_SCOPES)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--stage", choices=["export", *STAGE_TO_TIMING.keys()])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", choices=STAGE_TO_TIMING.keys())
    parser.add_argument("--no-clean-existing-keys", action="store_true")
    parser.add_argument("--check-resume-state", action="store_true")
    args = parser.parse_args()

    route_cfg = ROUTES[args.route]

    if not EXPORTER.exists():
        raise RuntimeError(f"C++ exporter is not built: {EXPORTER}")

    run_dir = EZKL_ROOT / DATASETS[args.dataset]
    if args.native_debug_scope:
        run_dir = run_dir / f"debug_{args.native_debug_scope}"
    ensure_inside(run_dir, EZKL_ROOT)
    if args.check_resume_state:
        print(json.dumps(resume_state(args.dataset, run_dir), indent=2, sort_keys=True))
        return 0
    previous_manifest = load_optional_json(run_dir / "run_manifest.json")
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_pipeline_outputs(run_dir)
    elif not (args.resume or args.resume_from or args.no_clean_existing_keys or args.stage):
        clean_pipeline_outputs(run_dir)
    run_id = make_run_id(args.dataset)
    run_started_at = now_utc_iso8601()

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
            "run_id": run_id,
            "run_started_at": run_started_at,
            "run_completed_at": now_utc_iso8601(),
            "dataset": args.dataset,
            "status": "failed",
            "block_reason": "exporter_failure",
            "failure_step": "exporter",
            "failed_command": " ".join(command),
            "route_id": args.route,
            "backend": route_cfg["backend"],
            "proof_generated": False,
            "verify_passed": False,
            "exporter_ms": exporter_ms,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            **common_manifest_fields(args.dataset, run_dir),
        }
        write_json(run_dir / "proof.json", {"status": "failed", "proof_generated": False})
        write_json(run_dir / "verify.json", {"status": "failed", "verified": False})
        attach_artifact_accounting(run_dir, manifest)
        write_json(run_dir / "run_manifest.json", manifest)
        print(f"dataset={args.dataset} status=failed block_reason=exporter_failure")
        return 1

    manifest = update_manifest(
        run_dir,
        run_id=run_id,
        run_started_at=run_started_at,
        clean_run=bool(args.clean) or previous_manifest.get("clean_run") is True,
        runner="scripts/run_gat_ezkl_baseline.py",
        exporter_ms=exporter_ms,
        backend=route_cfg["backend"],
        statement_route_id=route_cfg["statement_route_id"],
        route_statement_equivalence="same_dataset_checkpoint_preprocessing_quantization_and_final_logits_semantics",
        native_debug_scope=args.native_debug_scope or "full_model",
        scope=args.native_debug_scope or "full_model",
        stdout_tail=tail(stdout),
        stderr_tail=tail(stderr),
        command_line=" ".join(command),
        **common_manifest_fields(args.dataset, run_dir),
    )
    if args.resume or args.resume_from or args.stage:
        for timing_key in STAGE_TO_TIMING.values():
            if timing_key not in manifest and isinstance(previous_manifest.get(timing_key), (int, float)):
                manifest[timing_key] = previous_manifest[timing_key]
        if isinstance(previous_manifest.get("stage_records"), list):
            manifest["stage_records"] = previous_manifest["stage_records"]
        recovery_start_stage = args.resume_from or (args.stage if args.stage != "export" else None)
        manifest = recover_resume_history(
            run_dir,
            manifest,
            previous_manifest,
            recovery_start_stage,
            args.native_debug_scope,
        )
        write_json(run_dir / "run_manifest.json", manifest)
    if manifest.get("status") == "blocked":
        manifest.update(
            {
                "run_completed_at": now_utc_iso8601(),
                "proof_generated": False,
                "verify_passed": False,
            }
        )
        write_blocked_artifacts(run_dir, manifest)
        attach_artifact_accounting(run_dir, manifest)
        write_json(run_dir / "run_manifest.json", manifest)
        print(
            f"dataset={args.dataset} status=blocked "
            f"block_reason={manifest.get('block_reason')} route_id={manifest.get('route_id')}"
        )
        return 0

    if args.stage == "export":
        write_json(run_dir / "run_manifest.json", manifest)
        print(f"dataset={args.dataset} status=exported route_id={manifest.get('route_id')}")
        return 0

    if args.run:
        start_stage = args.resume_from or args.stage
        end_stage = args.stage if args.stage and args.stage != "export" else None
        manifest = run_native_graph_pipeline(
            run_dir,
            manifest,
            args.native_debug_scope,
            start_stage=start_stage,
            end_stage=end_stage,
        )
        print(f"dataset={args.dataset} status={manifest.get('status')} route_id={manifest.get('route_id')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
