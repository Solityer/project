#!/usr/bin/env python3
"""Temp-project orchestration regression for the official GPU ablation entrypoint.

All GAT-ZK binaries in this test are PATH-injected fixtures. No formal proof is
generated, and no repository artifacts are written outside the temporary project.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import select
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
VARIANTS = ("wo_shared_neighborhood", "wo_feature_folding")
ONE_VARIANT = VARIANTS[0]
METRIC_FIELDS = ("commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes")
TERMINAL_STATUSES = {"complete", "finished_with_failures", "failed"}


MOCK_CMAKE = r'''#!/usr/bin/env python3
import importlib.util
import json
import os
import pathlib
import shutil
import sys

args = sys.argv[1:]
call_log = pathlib.Path(os.environ["MOCK_CMAKE_CALL_LOG"])
with call_log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\n")

def definition(name, default=""):
    prefix = "-D" + name + "="
    for item in args:
        if item.startswith(prefix):
            return item[len(prefix):]
    return default

if "--build" not in args:
    root = pathlib.Path(args[args.index("-S") + 1]).resolve()
    build = pathlib.Path(args[args.index("-B") + 1]).resolve()
    cuda = definition("GATZK_ENABLE_CUDA_BACKEND")
    if cuda != "ON":
        raise SystemExit("mock requires the GPU-only production build")
    if definition("GATZK_BUILD_CORE_EXECUTION_REGRESSION") != "OFF":
        raise SystemExit("test target must stay disabled in the production build")
    if definition("GATZK_BUILD_PROTOCOL_VARIANT_REGRESSION") != "OFF":
        raise SystemExit("protocol regression target must stay disabled in the production build")
    build.mkdir(parents=True, exist_ok=True)
    cache = [
        "CMAKE_BUILD_TYPE:STRING=" + definition("CMAKE_BUILD_TYPE"),
        "GATZK_ENABLE_CUDA_BACKEND:BOOL=ON",
        "GATZK_BUILD_CORE_EXECUTION_REGRESSION:BOOL=OFF",
        "GATZK_BUILD_PROTOCOL_VARIANT_REGRESSION:BOOL=OFF",
    ]
    (build / "CMakeCache.txt").write_text("\n".join(cache) + "\n", encoding="utf-8")

    path = root / "tools" / "build_provenance.py"
    spec = importlib.util.spec_from_file_location("mock_build_provenance", path)
    provenance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provenance)
    manifest = provenance.source_manifest_document(root)
    generated = build / "generated" / "gatzk"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "source_manifest.json").write_bytes(provenance.canonical_json_bytes(manifest))
    metadata = {
        "source_tree_hash": manifest["source_tree_hash"],
        "source_inventory_id": provenance.SOURCE_INVENTORY_ID,
        "build_identity_algorithm": provenance.BUILD_IDENTITY_ALGORITHM,
        "compiler_id": "MOCK-GNU",
        "compiler_version": "13.3.0-mock",
        "build_type": "Release",
        "compile_flags": "CXX=MOCK;BUILD=Release;CUDA=ON",
    }
    (build / "mock_build_metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
else:
    build = pathlib.Path(args[args.index("--build") + 1]).resolve()
    if os.environ.get("MOCK_BUILD_FAIL") == "1":
        print("MOCK_CMAKE_BUILD_FAIL", file=sys.stderr)
        raise SystemExit(47)
    targets = args[args.index("--target") + 1:]
    expected = {"gatzk_run", "gatzk_registry_register", "gatzk_registry_inspect"}
    if not expected.issubset(set(targets)):
        raise SystemExit("mock build omitted a required production target")
    source = pathlib.Path(os.environ["MOCK_GATZK_TOOL"])
    for target in sorted(expected):
        destination = build / target
        shutil.copyfile(source, destination)
        destination.chmod(0o755)
'''


MOCK_GIT = r'''#!/usr/bin/env python3
import sys
if sys.argv[1:] == ["rev-parse", "HEAD"]:
    print("mock-source-head")
    raise SystemExit(0)
if sys.argv[1:] == ["rev-parse", "--short", "HEAD"]:
    print("mock-source")
    raise SystemExit(0)
if "ls-files" in sys.argv:
    raise SystemExit(1)
raise SystemExit(1)
'''


MOCK_NVIDIA_SMI = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys
args = sys.argv[1:]
path = os.environ.get("MOCK_NVIDIA_CALLS")
if path:
    with pathlib.Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(args) + "\n")
if "-L" in args:
    print("GPU 0: NVIDIA GB10 (UUID: GPU-MOCK-0000)")
else:
    print("GPU-UUID, GPU-Name, Memory-Total, Memory-Free")
    print("GPU-MOCK-0000, NVIDIA GB10, N/A, N/A")
'''


MOCK_TEE = r'''#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys

args = sys.argv[1:]
real_tee = os.environ["MOCK_REAL_TEE"]
copied = subprocess.Popen([real_tee, *args], stdin=subprocess.PIPE)
assert copied.stdin is not None
forwarded = bytearray()
while True:
    chunk = os.read(sys.stdin.fileno(), 4096)
    if not chunk:
        break
    forwarded.extend(chunk)
    copied.stdin.write(chunk)
    copied.stdin.flush()
copied.stdin.close()
copied_rc = copied.wait()
target = args[-1] if args else ""
failure_match = os.environ.get("MOCK_TEE_FAIL_MATCH", "")
failure_content = os.environ.get("MOCK_TEE_FAIL_CONTENT", "")
if failure_match and failure_match in target and (not failure_content or failure_content.encode() in forwarded):
    raise SystemExit(int(os.environ.get("MOCK_TEE_FAIL_RC", "31")))
raise SystemExit(copied_rc)
'''


MOCK_PYTHON3 = r'''#!/usr/bin/env bash
set -u
original=("$@")
if [[ ${1:-} == "${MOCK_SUPPORT_SCRIPT:-}" && -n ${MOCK_SUPPORT_SCRIPT:-} ]]; then
    support=$1 command=${2:-}
    shift 2
    variant="" dataset="" stage="" worker_status="" log_path=""
    while (($#)); do
        case "$1" in
            --variant) variant=${2:-}; shift 2 ;;
            --dataset) dataset=${2:-}; shift 2 ;;
            --stage) stage=${2:-}; shift 2 ;;
            --worker-status) worker_status=${2:-}; shift 2 ;;
            --log) log_path=${2:-}; shift 2 ;;
            *) shift ;;
        esac
    done
    target="$variant/$dataset"
    digest_target_slug=${MOCK_DIGEST_FAIL_TARGET//\//_}
    if [[ $command == extract-registry-digest && -n ${MOCK_DIGEST_FAIL_TARGET:-} &&
          $log_path == *"$digest_target_slug.log" ]]; then
        printf 'ablation-entrypoint: mock digest extraction error rc=%s target=%s\n' \
            "${MOCK_DIGEST_FAIL_RC:-43}" "${MOCK_DIGEST_FAIL_TARGET}" >&2
        exit "${MOCK_DIGEST_FAIL_RC:-43}"
    fi
    if [[ $command == matrix-update && $target == "${MOCK_MATRIX_UPDATE_FAIL_TARGET:-}" &&
          $stage == "${MOCK_MATRIX_UPDATE_FAIL_STAGE:-}" && -n ${MOCK_MATRIX_UPDATE_FAIL_TARGET:-} ]]; then
        marker=${MOCK_MATRIX_UPDATE_FAIL_MARKER:-}
        if [[ -z $marker || ! -e $marker ]]; then
            [[ -z $marker ]] || : > "$marker"
            printf 'ablation-entrypoint: mock matrix-update failure rc=%s target=%s stage=%s\n' \
                "${MOCK_MATRIX_UPDATE_FAIL_RC:-44}" "$target" "$stage" >&2
            exit "${MOCK_MATRIX_UPDATE_FAIL_RC:-44}"
        fi
    fi
    if [[ $command == matrix-update && $worker_status == complete &&
          ${MOCK_MATRIX_TERMINAL_FAIL_ONCE:-0} == 1 ]]; then
        marker=${MOCK_MATRIX_TERMINAL_FAIL_MARKER:-}
        if [[ -z $marker || ! -e $marker ]]; then
            [[ -z $marker ]] || : > "$marker"
            printf 'ablation-entrypoint: mock terminal matrix-update failure rc=%s\n' \
                "${MOCK_MATRIX_TERMINAL_FAIL_RC:-45}" >&2
            exit "${MOCK_MATRIX_TERMINAL_FAIL_RC:-45}"
        fi
    fi
fi
exec "${MOCK_REAL_PYTHON3:?}" "${original[@]}"
'''


MOCK_MKDIR = r'''#!/usr/bin/env bash
set -u
joined=" $* "
match=${MOCK_MKDIR_FAIL_MATCH:-}
if [[ -n $match && $joined == *"$match"* ]]; then
    count_file=${MOCK_MKDIR_COUNT_FILE:?}
    count=0
    if [[ -f $count_file ]]; then read -r count < "$count_file" || count=0; fi
    count=$((count + 1))
    printf '%s\n' "$count" > "$count_file"
    if [[ $count == "${MOCK_MKDIR_FAIL_OCCURRENCE:-1}" ]]; then
        rc=${MOCK_MKDIR_FAIL_RC:-41}
        printf 'ablation-entrypoint: mock mkdir failure rc=%s target=%s\n' "$rc" "$joined" >&2
        exit "$rc"
    fi
fi
exec "${MOCK_REAL_MKDIR:?}" "$@"
'''


MOCK_GATZK_TOOL = r'''#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")

def read_config(path):
    result = {}
    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result

def append_record(name, record):
    path = os.environ.get(name)
    if path:
        with pathlib.Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

def target_key(config):
    return config.get("protocol_variant", "") + "/" + config.get("dataset", "")

def matches_env(name, target):
    return os.environ.get(name, "") == target

def load_provenance(root):
    path = root / "tools" / "build_provenance.py"
    spec = importlib.util.spec_from_file_location("mock_build_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return raw

def run_gatzk():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol-variant", required=True)
    parser.add_argument("--compute-backend", required=True)
    parser.add_argument("--benchmark-mode", required=True)
    parser.add_argument("--export-tag", required=True)
    args, _ = parser.parse_known_args()
    config_path = pathlib.Path(args.config).resolve()
    config = read_config(config_path)
    variant, dataset = args.protocol_variant, config.get("dataset", "")
    target = variant + "/" + dataset
    append_record("MOCK_RUN_CALLS", {"variant": variant, "dataset": dataset, "target": target})
    if args.compute_backend != "cuda_hotspots" or args.benchmark_mode != "single":
        print("MOCK_INVALID_FORMAL_ARGUMENTS", file=sys.stderr)
        return 33

    if matches_env("MOCK_RUN_GATE_TARGET", target):
        pathlib.Path(os.environ["MOCK_RUN_STARTED"]).write_text(target + "\n", encoding="utf-8")
        release = pathlib.Path(os.environ["MOCK_RUN_RELEASE"])
        deadline = time.monotonic() + 90.0
        print("MOCK_RUN_PAUSED_STDOUT " + target, flush=True)
        print("MOCK_RUN_PAUSED_STDERR " + target, file=sys.stderr, flush=True)
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not release.exists():
            print("MOCK_RUN_GATE_TIMEOUT " + target, file=sys.stderr, flush=True)
            return 34

    if matches_env("MOCK_RUN_NO_EVIDENCE_TARGET", target):
        # A VERIFY_OK-looking line alone must never transition a matrix cell
        # to success. This fixture intentionally creates no proof or report.
        print(
            "dataset=" + dataset + " config=" + str(config_path) + " backend=gpu protocol_variant=" +
            variant + " status=VERIFY_OK commitment_time_ms=1.25 prove_time_ms=2.50 "
            "verify_time_ms=3.75 proof_size_bytes=64",
            flush=True,
        )
        return 0

    delay = float(os.environ.get("MOCK_RUN_DELAY_SECONDS", "0"))
    if delay > 0:
        time.sleep(delay)
    if matches_env("MOCK_RUN_FAIL_TARGET", target):
        print("MOCK_CHILD_FAILURE " + target, file=sys.stderr, flush=True)
        return 17

    project = pathlib.Path.cwd().resolve()
    export_dir = pathlib.Path(config["export_dir"])
    if not export_dir.is_absolute():
        export_dir = project / export_dir
    run_dir = export_dir / args.export_tag / "single"
    run_dir.mkdir(parents=True, exist_ok=True)
    binary_path = pathlib.Path(sys.argv[0]).resolve()
    binary_hash = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    build_dir = binary_path.parent
    source_manifest_path = build_dir / "generated" / "gatzk" / "source_manifest.json"
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    source_hash = source_manifest["source_tree_hash"]
    published_source_manifest = run_dir / "source_manifest.json"
    published_source_manifest.write_bytes(source_manifest_bytes)
    source_manifest_hash = hashlib.sha256(source_manifest_bytes).hexdigest()
    metadata = json.loads((build_dir / "mock_build_metadata.json").read_text(encoding="utf-8"))
    provenance = load_provenance(project)
    build_identity = provenance.build_identity(
        source_hash, metadata["compiler_id"], metadata["compiler_version"],
        metadata["build_type"], metadata["compile_flags"], binary_hash
    )

    registry_value = pathlib.Path(config["verifier_registry_path"])
    if not registry_value.is_absolute():
        registry_value = project / registry_value
    registry_bytes = registry_value.read_bytes()
    registry = json.loads(registry_bytes)
    registry_digest = hashlib.sha256(registry_bytes).hexdigest()
    proof = b"mock-only regression fixture; no formal proof is generated\n"
    proof_hash = hashlib.sha256(proof).hexdigest()
    output_hash = hashlib.sha256(("mock-public-output|" + target).encode()).hexdigest()
    proof_path = run_dir / "proof.bin"
    proof_path.write_bytes(proof)
    size = len(proof)

    report = {
        "format": "zkgat_verifier_report",
        "trusted_registry_digest": registry_digest,
        "authenticated_registry_digest": registry_digest,
        "proof_bin_sha256": proof_hash,
        "parsed_output_digest": output_hash,
        "proof_public_output_hash": output_hash,
        "parsed_output_count": "1",
        "verify_result": "true",
        "reference_consistency_passed": "true",
        "tamper_tests_passed": "false",
        "tamper_test_case_map": "",
        "tamper_tests_ms": "0",
        "security_budget_passed": "true",
        "security_budget_ledger_digest": registry["security_budget_ledger_digest"],
        "security_budget_exact": registry["security_budget_exact"],
        "public_input_digest": registry["public_input_digest"],
        "topology_digest": registry["topology_digest"],
        "source_tree_hash": source_hash,
        "binary_hash": binary_hash,
        "build_identity": build_identity,
    }

    build_receipt = {
        "format": "zkgat_build_receipt",
        "source_manifest_path": "source_manifest.json",
        "source_manifest_hash": source_manifest_hash,
        "source_inventory_id": metadata["source_inventory_id"],
        "source_tree_hash": source_hash,
        "compiler_id": metadata["compiler_id"],
        "compiler_version": metadata["compiler_version"],
        "build_type": metadata["build_type"],
        "compile_flags": metadata["compile_flags"],
        "binary_path": str(binary_path),
        "binary_hash": binary_hash,
        "build_identity_algorithm": metadata["build_identity_algorithm"],
        "build_identity": build_identity,
    }
    receipt_path = run_dir / "build_receipt.json"
    receipt_bytes = write_json(receipt_path, build_receipt)
    report_path = run_dir / "verifier_report.json"
    report_bytes = write_json(report_path, report)

    manifest = {
        "format": "zkgat_run_manifest_v1",
        "protocol_name": variant,
        "dataset": dataset,
        "dataset_id": dataset,
        "backend": "gpu",
        "backend_name": "mcl/cuda_hotspots",
        "compute_backend_name": "cuda_hotspots",
        "crypto_backend_name": "mcl",
        "algebra_backend_name": "cuda_hotspots",
        "gpu_backend_requested": "true",
        "gpu_runtime_present": "true",
        "gpu_runtime_loaded": "true",
        "gpu_kernel_invocation_count": "1",
        "gpu_device": "NVIDIA GB10 MOCK",
        "gpu_acceleration_status": "mock_runtime_measured",
        "status": "VERIFY_OK",
        "verifier_accepted": "true",
        "final_formal_run_accepted": "true",
        "verified": "true",
        "reference_consistency_passed": "true",
        "security_budget_passed": "true",
        "verifier_reads_witness": "false",
        "private_plain_value_count": "0",
        "proof_payload_audit_status": "passed",
        "security_budget_ledger_digest": registry["security_budget_ledger_digest"],
        "security_budget_exact": registry["security_budget_exact"],
        "quant_config_hash": hashlib.sha256(("quant-config|" + target).encode()).hexdigest(),
        "public_metadata_hash": hashlib.sha256(("public-metadata|" + target).encode()).hexdigest(),
        "static_table_hash": hashlib.sha256(("static-table|" + target).encode()).hexdigest(),
        "secure_trace_logits_hash": output_hash,
        "reference_quant_logits_hash": output_hash,
        "proof_public_output_hash": output_hash,
        "proof_hash": proof_hash,
        "proof_size_bytes": str(size),
        "benchmark_mode": "single",
        "timing_schema": "online_trace_commit_prove_protocol_verify_v2",
        "commitment_time_ms": "1.25",
        "prove_time_ms": "2.50",
        "verify_time_ms": "3.75",
        "prover_total_ms": "3.75",
        "prover_wall_including_audit_ms": "4.75",
        "prover_reference_audit_ms": "1.00",
        "total_wall_clock_ms": "5.00",
        "is_full_dataset": "true",
        "full_dataset": "true",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "verifier_registry_id": registry["registry_id"],
        "verifier_registry_digest": registry_digest,
        "trusted_registry_digest": registry_digest,
        "protocol_schema_id": registry["protocol_schema_id"],
        "instance_schema_hash": registry["instance_schema_hash"],
        "srs_id": registry["srs_id"],
        "srs_manifest_hash": registry["srs_manifest_hash"],
        "public_input_digest": registry["public_input_digest"],
        "topology_digest": registry["topology_digest"],
        "source_manifest_path": "source_manifest.json",
        "source_manifest_hash": source_manifest_hash,
        "source_tree_hash": source_hash,
        "build_receipt_path": "build_receipt.json",
        "build_receipt_hash": hashlib.sha256(receipt_bytes).hexdigest(),
        "binary_path": str(binary_path),
        "binary_hash": binary_hash,
        "build_identity": build_identity,
        "verifier_report_path": "verifier_report.json",
        "verifier_report_hash": hashlib.sha256(report_bytes).hexdigest(),
        "reference_fp32_logits_hash": output_hash,
        "input_preprocess_id": config.get("input_preprocess_id", "mock-preprocess"),
    }

    mutation_target = os.environ.get("MOCK_RUN_MUTATE_TARGET", "")
    mutation = os.environ.get("MOCK_RUN_MUTATION", "")
    if mutation_target == target:
        if mutation.startswith("manifest:"):
            field, value = mutation.split(":", 2)[1:]
            manifest[field] = value
        elif mutation.startswith("report:"):
            field, value = mutation.split(":", 2)[1:]
            report[field] = value
        elif mutation == "remove-proof-size":
            manifest.pop("proof_size_bytes", None)
        elif mutation == "proof-size-mismatch":
            manifest["proof_size_bytes"] = str(size + 1)
        elif mutation == "report-hash-mismatch":
            manifest["verifier_report_hash"] = "0" * 64
        elif mutation.startswith("nan:"):
            field = mutation.split(":", 1)[1]
            manifest[field] = "NaN"
        elif mutation.startswith("negative:"):
            field = mutation.split(":", 1)[1]
            manifest[field] = "-1"
        elif mutation == "negative-consistent-commitment":
            manifest.update({
                "commitment_time_ms": "-1",
                "prove_time_ms": "4",
                "prover_total_ms": "3",
                "prover_wall_including_audit_ms": "4",
                "prover_reference_audit_ms": "1",
            })
        elif mutation == "zero-gpu-kernels":
            manifest["gpu_kernel_invocation_count"] = "0"
        elif mutation == "gpu-runtime-absent":
            manifest["gpu_runtime_present"] = "false"
            manifest["gpu_runtime_loaded"] = "false"
        elif mutation == "empty-gpu-device":
            manifest["gpu_device"] = ""
        else:
            print("unknown mock mutation: " + mutation, file=sys.stderr)
            return 35

    # Re-emit the report after any report mutation so its digest is authentic
    # unless the test explicitly asks for a mismatched report digest.
    report_bytes = write_json(report_path, report)
    if mutation_target == target and mutation.startswith("report:"):
        manifest["verifier_report_hash"] = hashlib.sha256(report_bytes).hexdigest()
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    backend = "gpu" if args.compute_backend == "cuda_hotspots" else "cpu"
    print(
        "dataset=" + dataset + " config=" + str(config_path) + " backend=" + backend +
        " protocol_variant=" + variant + " status=VERIFY_OK commitment_time_ms=" +
        manifest.get("commitment_time_ms", "") + " prove_time_ms=" + manifest.get("prove_time_ms", "") +
        " verify_time_ms=" + manifest.get("verify_time_ms", "") + " proof_size_bytes=" +
        manifest.get("proof_size_bytes", "")
    , flush=True)
    return 0

name = pathlib.Path(sys.argv[0]).name
if name == "gatzk_registry_register":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args, _ = parser.parse_known_args()
    cfg = read_config(args.config)
    target = target_key(cfg)
    append_record("MOCK_REGISTER_CALLS", {"target": target})
    if matches_env("MOCK_REGISTER_FAIL_TARGET", target):
        print("MOCK_REGISTER_FAILURE " + target, file=sys.stderr)
        raise SystemExit(27)
    if cfg.get("protocol_variant") not in {"wo_shared_neighborhood", "wo_feature_folding"}:
        raise SystemExit("mock registration received a wrong protocol variant")
    if "verifier_registry_digest" in cfg:
        raise SystemExit("mock candidate config inherited a stale registry digest")
    project = pathlib.Path.cwd()
    srs_manifest = json.loads((project / cfg["srs_path"] / "srs_manifest.json").read_text(encoding="utf-8"))
    identity = "|".join((cfg["protocol_variant"], cfg["dataset"], cfg["model_arch_id"], cfg["srs_id"]))
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    artifact = {
        "dataset": cfg["dataset"],
        "protocol_variant": cfg["protocol_variant"],
        "model_arch_id": cfg["model_arch_id"],
        "model_param_id": cfg["model_param_id"],
        "quant_cfg_id": cfg["quant_cfg_id"],
        "static_table_id": cfg["static_table_id"],
        "protocol_schema_id": srs_manifest["protocol_schema_id"],
        "instance_schema_hash": digest("instance|" + identity),
        "registry_id": digest("registry|" + identity),
        "security_budget_ledger_digest": digest("security-ledger|" + identity),
        "security_budget_exact": "64/64",
        "public_input_digest": digest("public-input|" + identity),
        "topology_digest": digest("topology|" + identity),
        "srs_id": cfg["srs_id"],
        "srs_manifest_hash": srs_manifest["manifest_hash"],
        "srs_degree_bound": srs_manifest["degree_bound"],
        "security_budget_passed": True,
    }
    path = pathlib.Path(args.output)
    if not path.is_absolute():
        path = project / path
    raw = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    registry_digest = hashlib.sha256(raw).hexdigest()
    print("candidate_registry_path=" + str(path) + " registry_id=" + artifact["registry_id"] +
          " candidate_registry_digest=" + registry_digest)
    raise SystemExit(0)

if name == "gatzk_registry_inspect":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--trusted-digest", required=True)
    args, _ = parser.parse_known_args()
    path = pathlib.Path(args.registry)
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    artifact = json.loads(raw)
    target = artifact["protocol_variant"] + "/" + artifact["dataset"]
    append_record("MOCK_INSPECT_CALLS", {"target": target})
    if matches_env("MOCK_INSPECT_FAIL_TARGET", target):
        print("MOCK_INSPECT_FAILURE " + target, file=sys.stderr)
        raise SystemExit(29)
    artifact["artifact_sha256"] = digest
    artifact["registry_digest"] = digest
    artifact["security_budget_passed"] = True
    if matches_env("MOCK_INSPECT_BAD_TARGET", target):
        artifact["dataset"] = "wrong_dataset"
    print(json.dumps(artifact, sort_keys=True))
    raise SystemExit(0)

raise SystemExit(run_gatzk())
'''


@dataclass
class Workspace:
    root: pathlib.Path
    project: pathlib.Path
    env: dict[str, str]
    call_log: pathlib.Path
    run_calls: pathlib.Path
    register_calls: pathlib.Path
    inspect_calls: pathlib.Path
    nvidia_calls: pathlib.Path

    @property
    def script(self) -> pathlib.Path:
        return self.project / "scripts" / "run_ablation_matrix.sh"

    def command(self, *args: str, timeout: int = 90) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.project,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )

    def support_command(self, *args: str, timeout: int = 10) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(self.project / "scripts" / "ablation_matrix_support.py"), *args],
            cwd=self.project,
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )

    @property
    def latest_file(self) -> pathlib.Path:
        return self.project / "runs" / "ablation_sessions" / "gpu_matrix" / "latest"

    def latest_session_id(self) -> str:
        return self.latest_file.read_text(encoding="utf-8").strip()

    def session(self) -> pathlib.Path:
        return self.project / "runs" / "ablation_sessions" / "gpu_matrix" / self.latest_session_id()

    def summary(self) -> dict:
        return json.loads((self.session() / "matrix_summary.json").read_text(encoding="utf-8"))

    def call_records(self, path: pathlib.Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@contextmanager
def mock_workspace(
    *,
    run_failure: str = "",
    register_failure: str = "",
    inspect_failure: str = "",
    inspect_bad_metadata: str = "",
    no_evidence: str = "",
    run_mutate_target: str = "",
    run_mutation: str = "",
    build_failure: bool = False,
    tee_failure_match: str = "",
    tee_failure_content: str = "",
    gate_target: str = "",
    mkdir_failure_match: str = "",
    mkdir_failure_occurrence: int = 1,
    mkdir_failure_rc: int = 41,
    digest_failure_target: str = "",
    matrix_update_failure_target: str = "",
    matrix_update_failure_stage: str = "",
    matrix_update_failure_rc: int = 44,
    matrix_terminal_fail_once: bool = False,
    matrix_terminal_failure_rc: int = 45,
):
    with tempfile.TemporaryDirectory(prefix="gatzk-gpu-ablation-entrypoint-") as temp:
        temp_root = pathlib.Path(temp)
        project = temp_root / "project"
        project.mkdir()
        (project / "scripts").mkdir()
        (project / "tools").mkdir()
        (project / "configs").mkdir()
        shutil.copy2(PROJECT_ROOT / "scripts" / "run_ablation_matrix.sh", project / "scripts" / "run_ablation_matrix.sh")
        shutil.copy2(PROJECT_ROOT / "scripts" / "ablation_matrix_support.py", project / "scripts" / "ablation_matrix_support.py")
        shutil.copy2(PROJECT_ROOT / "scripts" / "aggregate_secure_results.py", project / "scripts" / "aggregate_secure_results.py")
        shutil.copy2(PROJECT_ROOT / "tools" / "build_provenance.py", project / "tools" / "build_provenance.py")
        for dataset in DATASETS:
            config_path = PROJECT_ROOT / "configs" / (dataset + "_secure.cfg")
            shutil.copy2(config_path, project / "configs" / config_path.name)
            values = {}
            for line in config_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip()

            # The entrypoint's startup inventory checks that required input
            # paths and the small checkpoint manifest exist before launching
            # any per-cell tools. Supply tiny, explicitly mock-only metadata
            # fixtures so those gates pass without copying real datasets or
            # model checkpoints into the temporary project.
            feature_profile = values["d_in_profile"].strip()[1:-1]
            input_features = int(feature_profile.split(",", 1)[0].strip())
            cache = project / values["cache_root"] / dataset
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "edges.txt").write_text("", encoding="utf-8")
            (cache / "features.txt").write_text("", encoding="utf-8")
            (cache / "labels.txt").write_text("", encoding="utf-8")
            (cache / "meta.cfg").write_text(
                "name = " + dataset + "\n"
                + "num_features = " + str(input_features) + "\n"
                + "num_classes = " + values["num_classes"] + "\n"
                + "num_nodes = 4\n",
                encoding="utf-8",
            )
            checkpoint_bundle = project / values["checkpoint_bundle"]
            checkpoint_bundle.mkdir(parents=True, exist_ok=True)
            checkpoint_manifest = {
                "dataset": values["dataset"],
                "model_arch_id": values["model_arch_id"],
                "model_param_id": values["model_param_id"],
                "quant_cfg_id": values["quant_cfg_id"],
                "input_preprocess_id": values["input_preprocess_id"],
                "L": int(values["L"]),
                "K_out": int(values["K_out"]),
            }
            (checkpoint_bundle / "manifest.json").write_text(
                json.dumps(checkpoint_manifest, sort_keys=True), encoding="utf-8"
            )
            for name in ("tensors.txt", "tensors.npz", "best_model.pt"):
                (checkpoint_bundle / name).write_bytes(b"mock-only placeholder; never loaded by fake binaries\n")

            manifest = PROJECT_ROOT / values["srs_path"] / "srs_manifest.json"
            target = project / values["srs_path"] / "srs_manifest.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, target)

        mock_bin = temp_root / "mock-bin"
        mock_bin.mkdir()
        scripts = {
            "cmake": MOCK_CMAKE,
            "git": MOCK_GIT,
            "nvidia-smi": MOCK_NVIDIA_SMI,
            "tee": MOCK_TEE,
            "python3": MOCK_PYTHON3,
            "mkdir": MOCK_MKDIR,
            "mock_gatzk_tool.py": MOCK_GATZK_TOOL,
        }
        for name, content in scripts.items():
            path = mock_bin / name
            path.write_text(content, encoding="utf-8")
            path.chmod(0o755)

        call_log = temp_root / "cmake_calls.jsonl"
        run_calls = temp_root / "run_calls.jsonl"
        register_calls = temp_root / "register_calls.jsonl"
        inspect_calls = temp_root / "inspect_calls.jsonl"
        nvidia_calls = temp_root / "nvidia_calls.jsonl"
        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env.update({
            "MOCK_CMAKE_CALL_LOG": str(call_log),
            "MOCK_GATZK_TOOL": str(mock_bin / "mock_gatzk_tool.py"),
            "MOCK_REAL_TEE": shutil.which("tee") or "/usr/bin/tee",
            "MOCK_REAL_PYTHON3": shutil.which("python3") or "/usr/bin/python3",
            "MOCK_SUPPORT_SCRIPT": str(project / "scripts" / "ablation_matrix_support.py"),
            "MOCK_REAL_MKDIR": shutil.which("mkdir") or "/bin/mkdir",
            "MOCK_MKDIR_FAIL_MATCH": mkdir_failure_match,
            "MOCK_MKDIR_FAIL_OCCURRENCE": str(mkdir_failure_occurrence),
            "MOCK_MKDIR_FAIL_RC": str(mkdir_failure_rc),
            "MOCK_MKDIR_COUNT_FILE": str(temp_root / "mkdir_count"),
            "MOCK_DIGEST_FAIL_TARGET": digest_failure_target,
            "MOCK_DIGEST_FAIL_RC": "43",
            "MOCK_MATRIX_UPDATE_FAIL_TARGET": matrix_update_failure_target,
            "MOCK_MATRIX_UPDATE_FAIL_STAGE": matrix_update_failure_stage,
            "MOCK_MATRIX_UPDATE_FAIL_RC": str(matrix_update_failure_rc),
            "MOCK_MATRIX_UPDATE_FAIL_MARKER": str(temp_root / "matrix_update_fail_once"),
            "MOCK_MATRIX_TERMINAL_FAIL_ONCE": "1" if matrix_terminal_fail_once else "0",
            "MOCK_MATRIX_TERMINAL_FAIL_RC": str(matrix_terminal_failure_rc),
            "MOCK_MATRIX_TERMINAL_FAIL_MARKER": str(temp_root / "matrix_terminal_fail_once"),
            "MOCK_RUN_CALLS": str(run_calls),
            "MOCK_REGISTER_CALLS": str(register_calls),
            "MOCK_INSPECT_CALLS": str(inspect_calls),
            "MOCK_NVIDIA_CALLS": str(nvidia_calls),
            "MOCK_RUN_FAIL_TARGET": run_failure,
            "MOCK_REGISTER_FAIL_TARGET": register_failure,
            "MOCK_INSPECT_FAIL_TARGET": inspect_failure,
            "MOCK_INSPECT_BAD_TARGET": inspect_bad_metadata,
            "MOCK_RUN_MUTATE_TARGET": run_mutate_target,
            "MOCK_RUN_MUTATION": run_mutation,
            "MOCK_RUN_GATE_TARGET": gate_target,
            "MOCK_RUN_NO_EVIDENCE_TARGET": no_evidence,
            "MOCK_TEE_FAIL_MATCH": tee_failure_match,
            "MOCK_TEE_FAIL_CONTENT": tee_failure_content,
            "MOCK_TEE_FAIL_RC": "31",
            "MOCK_BUILD_FAIL": "1" if build_failure else "0",
        })
        if gate_target:
            env["MOCK_RUN_STARTED"] = str(temp_root / "run_started")
            env["MOCK_RUN_RELEASE"] = str(temp_root / "run_release")
        workspace = Workspace(temp_root, project, env, call_log, run_calls, register_calls, inspect_calls, nvidia_calls)
        try:
            yield workspace
        except BaseException:
            if os.environ.get("GATZK_TEST_KEEP_FAILED_WORKSPACE") == "1":
                preserved = pathlib.Path(tempfile.mkdtemp(prefix="gatzk-ablation-failed-workspace-")) / "workspace"
                shutil.copytree(temp_root, preserved)
                print("MOCK_WORKSPACE_PRESERVED=" + str(preserved), flush=True)
            raise


def expected_cells(variants: tuple[str, ...]) -> list[str]:
    return [variant + "/" + dataset for variant in variants for dataset in DATASETS]


def wait_for(predicate, timeout: float = 30.0, message: str = "condition not reached") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.025)
    raise AssertionError(message)


def read_until(process: subprocess.Popen[bytes], token: bytes, timeout: float = 5.0) -> bytes:
    assert process.stdout is not None
    data = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        import select
        ready, _, _ = select.select([process.stdout], [], [], min(0.1, max(0.0, deadline - time.monotonic())))
        if not ready:
            continue
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        data.extend(chunk)
        if token in data:
            break
    return bytes(data)


class GpuAblationEntrypointMockTest(unittest.TestCase):
    def _assert_finished(self, workspace: Workspace, timeout: float = 30.0) -> dict:
        wait_for(
            lambda: (workspace.session() / "matrix_summary.json").is_file()
            and json.loads((workspace.session() / "matrix_summary.json").read_text(encoding="utf-8")).get("status")
            in TERMINAL_STATUSES,
            timeout=timeout,
            message="matrix summary did not reach a terminal status",
        )
        return workspace.summary()

    def test_default_is_ordered_gpu_only_matrix_with_real_shaped_metadata(self) -> None:
        with mock_workspace() as workspace:
            completed = workspace.command()
            self.assertEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual(summary["variants"], list(VARIANTS))
            self.assertEqual(summary["datasets"], list(DATASETS))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (10, 10, 0, 0))
            records = workspace.call_records(workspace.run_calls)
            self.assertEqual([item["target"] for item in records], expected_cells(VARIANTS))
            self.assertEqual(set(summary["cells"]), set(expected_cells(VARIANTS)))
            for key, cell in summary["cells"].items():
                self.assertEqual(cell["command_rc"], 0, key)
                self.assertEqual(cell["logger_rc"], 0, key)
                self.assertEqual(cell["validation_rc"], 0, key)
                self.assertEqual(set(cell["metrics"]), set(METRIC_FIELDS), key)
                self.assertGreater(cell["metrics"]["proof_size_bytes"], 0, key)
            self.assertEqual(workspace.latest_session_id(), summary["session_id"])
            self.assertFalse((workspace.latest_file.parent / "latest.tmp").exists())
            cmake_records = workspace.call_records(workspace.call_log)
            self.assertEqual(len(cmake_records), 2, "one configure and one build for the GPU production targets")
            configure = cmake_records[0]
            self.assertIn("-DGATZK_ENABLE_CUDA_BACKEND=ON", configure)
            self.assertIn("-DGATZK_BUILD_CORE_EXECUTION_REGRESSION=OFF", configure)
            self.assertIn("-DGATZK_BUILD_PROTOCOL_VARIANT_REGRESSION=OFF", configure)
            self.assertEqual(set(cmake_records[1][cmake_records[1].index("--target") + 1:]),
                             {"gatzk_run", "gatzk_registry_register", "gatzk_registry_inspect"})
            nvidia_calls = workspace.nvidia_calls.read_text(encoding="utf-8")
            self.assertTrue(nvidia_calls.strip(), "entrypoint did not run its GPU probe")

    def test_one_variant_is_supported_but_cpu_and_both_are_rejected(self) -> None:
        with mock_workspace() as workspace:
            for backend in ("cpu", "both"):
                bad = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", backend)
                self.assertNotEqual(bad.returncode, 0, bad.stdout)
                self.assertFalse(workspace.latest_file.exists())
            completed = workspace.command("--protocol-variant", VARIANTS[1], "--backend", "gpu")
            self.assertEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual(summary["variants"], [VARIANTS[1]])
            self.assertEqual(summary["attempted"], 5)
            self.assertEqual(summary["succeeded"], 5)
            self.assertEqual([row["target"] for row in workspace.call_records(workspace.run_calls)],
                             expected_cells((VARIANTS[1],)))

    def test_child_exit_17_continues_all_ten_and_failure_view_reports_it(self) -> None:
        failed_target = ONE_VARIANT + "/cora"
        with mock_workspace(run_failure=failed_target) as workspace:
            completed = workspace.command()
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (10, 9, 1, 0))
            self.assertEqual(summary["status"], "finished_with_failures")
            self.assertEqual([row["target"] for row in workspace.call_records(workspace.run_calls)], expected_cells(VARIANTS))
            failed = summary["cells"][failed_target]
            self.assertEqual(failed["command_rc"], 17)
            self.assertEqual(failed["logger_rc"], 0)
            self.assertNotEqual(failed["validation_rc"], 0)
            view = workspace.command("--failures")
            self.assertEqual(view.returncode, 0, view.stdout)
            self.assertIn("variant=" + ONE_VARIANT, view.stdout)
            self.assertIn("dataset=cora", view.stdout)
            self.assertIn("MOCK_CHILD_FAILURE", (workspace.session() / failed["log"]).read_text(encoding="utf-8"))

    def test_cell_logger_exit_is_recorded_separately_from_child_exit(self) -> None:
        child_target = ONE_VARIANT + "/cora"
        logger_target = ONE_VARIANT + "_pubmed.log"
        with mock_workspace(
            run_failure=child_target,
            tee_failure_match=logger_target,
            tee_failure_content=b"status=VERIFY_OK".decode(),
        ) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 3, 2, 0))
            child = summary["cells"][child_target]
            logger = summary["cells"][ONE_VARIANT + "/pubmed"]
            self.assertEqual((child["command_rc"], child["logger_rc"]), (17, 0))
            self.assertEqual((logger["command_rc"], logger["logger_rc"]), (0, 31))
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 5)

    def _assert_registration_path_failure(self, name: str, **options: str) -> dict:
        with mock_workspace(**options) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 4, 1, 0))
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 4)
            failed = [cell for cell in summary["cells"].values() if cell["status"] == "failed"]
            self.assertEqual(len(failed), 1)
            self.assertIn(name, failed[0]["phase"] + " " + failed[0]["reason"])
            failed_log = workspace.session() / failed[0]["log"]
            return {"cell": failed[0], "log": failed_log.read_text(encoding="utf-8"), "output": completed.stdout}

    def test_registration_failure_is_cell_scoped(self) -> None:
        self._assert_registration_path_failure("register", register_failure=ONE_VARIANT + "/cora")

    def test_registry_inspector_stderr_detail_is_cell_scoped(self) -> None:
        result = self._assert_registration_path_failure("inspect", inspect_failure=ONE_VARIANT + "/pubmed")
        self.assertIn("MOCK_INSPECT_FAILURE", result["cell"]["reason"])
        self.assertIn("MOCK_INSPECT_FAILURE", result["log"])

    def test_registry_metadata_binding_failure_is_cell_scoped(self) -> None:
        self._assert_registration_path_failure("metadata", inspect_bad_metadata=ONE_VARIANT + "/ppi")

    def test_cell_log_creation_failure_is_recorded_without_faking_native_rc(self) -> None:
        target = ONE_VARIANT + "/cora"
        with mock_workspace(
            mkdir_failure_match="/logs",
            mkdir_failure_occurrence=2,
            mkdir_failure_rc=41,
        ) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 4, 1, 0))
            failed = summary["cells"][target]
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["command_rc"])
            self.assertIsNone(failed["validation_rc"])
            self.assertNotEqual(failed["logger_rc"], 41)
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 4)
            self.assertIn("log setup", failed["reason"])
            self.assertIn("rc=41", failed["reason"])
            self.assertIn("CELL_LOG_CREATE_FAILED", completed.stdout)
            self.assertIn("mock mkdir failure rc=41", completed.stdout)
            self.assertEqual(failed["log"], ONE_VARIANT + "_cora.failure.log")
            failure_log = workspace.session() / failed["log"]
            self.assertTrue(failure_log.is_file())
            self.assertIn("mock mkdir failure rc=41", failure_log.read_text(encoding="utf-8"))

    def test_inspection_directory_creation_failure_is_a_cell_scoped_inspect_failure(self) -> None:
        target = ONE_VARIANT + "/cora"
        with mock_workspace(
            mkdir_failure_match="/preflight/" + ONE_VARIANT,
            mkdir_failure_rc=42,
        ) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 4, 1, 0))
            failed = summary["cells"][target]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["phase"], "inspect")
            self.assertIsNone(failed["command_rc"])
            self.assertIsNone(failed["validation_rc"])
            self.assertEqual(failed["stage_rcs"]["inspect"]["command_rc"], 42)
            self.assertIn("mock mkdir failure rc=42", failed["reason"])
            self.assertEqual(len(workspace.call_records(workspace.inspect_calls)), 4)
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 4)
            self.assertIn("mock mkdir failure rc=42", completed.stdout)

    def test_digest_extraction_error_is_recorded_and_later_cells_continue(self) -> None:
        target = ONE_VARIANT + "/cora"
        with mock_workspace(digest_failure_target=target) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 4, 1, 0))
            failed = summary["cells"][target]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["phase"], "register")
            self.assertEqual(failed["stage_rcs"]["register"]["command_rc"], 43)
            self.assertIsNone(failed["command_rc"])
            self.assertIsNone(failed["validation_rc"])
            self.assertIn("mock digest extraction error", failed["reason"])
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 4)
            failed_log = workspace.session() / failed["log"]
            self.assertIn("mock digest extraction error", failed_log.read_text(encoding="utf-8"))

    def test_nonterminal_matrix_update_failure_is_sticky_and_skips_that_cell_proof(self) -> None:
        target = ONE_VARIANT + "/cora"
        with mock_workspace(
            matrix_update_failure_target=target,
            matrix_update_failure_stage="startup_inventory",
            matrix_update_failure_rc=44,
        ) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 4, 1, 0))
            failed = summary["cells"][target]
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["command_rc"])
            self.assertIsNone(failed["validation_rc"])
            self.assertIn("helper_rc=44 logger_rc=0", failed["reason"])
            self.assertIn("proof was not started", failed["reason"])
            self.assertIn("mock matrix-update failure rc=44", failed["reason"])
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 4)
            self.assertIn("mock matrix-update failure rc=44", completed.stdout)
            failed_log = workspace.session() / failed["log"]
            self.assertIn("mock matrix-update failure rc=44", failed_log.read_text(encoding="utf-8"))

    def test_terminal_matrix_update_failure_is_recovered_as_session_failure(self) -> None:
        with mock_workspace(matrix_terminal_fail_once=True) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual(summary["status"], "finished_with_failures")
            self.assertEqual(summary["worker_status"], "finished_with_failures")
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 5, 0, 0))
            self.assertIn("final worker terminal-state update failed", summary["session_failure_reason"])
            self.assertIn("helper_rc=45 logger_rc=0", summary["session_failure_reason"])
            self.assertEqual(summary["session_failure_rc"], 45)
            self.assertIn("mock terminal matrix-update failure rc=45", completed.stdout)
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 5)

    def test_terminal_update_logger_failure_downgrades_session_without_faking_cell_failures(self) -> None:
        with mock_workspace(
            tee_failure_match="global.log",
            tee_failure_content="MATRIX_UPDATED status=complete",
        ) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual(summary["status"], "finished_with_failures")
            self.assertEqual(summary["worker_status"], "finished_with_failures")
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 5, 0, 0))
            self.assertEqual(summary["session_failure_rc"], 31)
            self.assertIn("terminal-state update failed", summary["session_failure_reason"])
            self.assertIn("helper_rc=0 logger_rc=31", summary["session_failure_reason"])
            self.assertEqual(len(workspace.call_records(workspace.run_calls)), 5)

    def test_verify_ok_stdout_without_validated_proof_evidence_never_succeeds(self) -> None:
        target = ONE_VARIANT + "/cora"
        with mock_workspace(no_evidence=target) as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["succeeded"], 4)
            cell = summary["cells"][target]
            self.assertEqual(cell["status"], "failed")
            self.assertNotEqual(cell["validation_rc"], 0)
            self.assertFalse(cell.get("evidence_path"))
            self.assertFalse(cell.get("proof_path"))
            self.assertNotIn("SUCCESS variant=" + target.replace("/", " dataset="), completed.stdout)

    def _matrix_init(self, workspace: Workspace, session_root: pathlib.Path) -> subprocess.CompletedProcess[str]:
        # The shell runner atomically reserves a fresh empty session directory
        # before asking the helper to initialize it.
        session_root.mkdir(parents=True, exist_ok=True)
        return workspace.support_command(
            "matrix-init",
            "--session-root", str(session_root),
            "--variants", ONE_VARIANT,
            "--worker-pid", str(os.getpid()),
        )

    def test_initialized_matrix_session_cannot_be_reused_or_overwritten(self) -> None:
        with mock_workspace() as workspace:
            session_root = workspace.root / "fixed-session"
            first = self._matrix_init(workspace, session_root)
            self.assertEqual(first.returncode, 0, first.stdout)
            summary_path = session_root / "matrix_summary.json"
            original = summary_path.read_bytes()
            second = self._matrix_init(workspace, session_root)
            self.assertNotEqual(second.returncode, 0, second.stdout)
            self.assertEqual(summary_path.read_bytes(), original)

    def test_matrix_success_transition_requires_zero_rc_evidence_proof_and_four_metrics(self) -> None:
        metric_fields = ("commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes")
        with mock_workspace() as workspace:
            cases = (
                ("missing all", None, None, None),
                ("missing command rc", 0, None, None),
                ("missing validation rc", 0, 0, None),
                ("all rc zero but no evidence", 0, 0, 0),
            )
            for name, command_rc, validation_rc, logger_rc in cases:
                with self.subTest(name=name):
                    session_root = workspace.root / ("success-guard-" + name.replace(" ", "-"))
                    initialized = self._matrix_init(workspace, session_root)
                    self.assertEqual(initialized.returncode, 0, initialized.stdout)
                    marked_running = workspace.support_command(
                        "matrix-update", "--session-root", str(session_root),
                        "--variant", ONE_VARIANT, "--dataset", "cora", "--cell-status", "running",
                    )
                    self.assertEqual(marked_running.returncode, 0, marked_running.stdout)
                    args = [
                        "matrix-update", "--session-root", str(session_root),
                        "--variant", ONE_VARIANT, "--dataset", "cora", "--cell-status", "success",
                    ]
                    if command_rc is not None:
                        args.extend(("--command-rc", str(command_rc)))
                    if validation_rc is not None:
                        args.extend(("--validation-rc", str(validation_rc)))
                    if logger_rc is not None:
                        args.extend(("--logger-rc", str(logger_rc)))
                    rejected = workspace.support_command(*args)
                    self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
                    matrix = json.loads((session_root / "matrix_summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(matrix["cells"][ONE_VARIANT + "/cora"]["status"], "running")
                    self.assertEqual(matrix["succeeded"], 0)

            for missing in metric_fields:
                with self.subTest(missing_metric=missing):
                    session_root = workspace.root / ("missing-metric-" + missing)
                    initialized = self._matrix_init(workspace, session_root)
                    self.assertEqual(initialized.returncode, 0, initialized.stdout)
                    marked_running = workspace.support_command(
                        "matrix-update", "--session-root", str(session_root),
                        "--variant", ONE_VARIANT, "--dataset", "cora", "--cell-status", "running",
                    )
                    self.assertEqual(marked_running.returncode, 0, marked_running.stdout)
                    proof = session_root / "results" / ONE_VARIANT / "cora" / "positive-shape" / "single" / "proof.bin"
                    proof.parent.mkdir(parents=True, exist_ok=True)
                    proof.write_bytes(b"fixture-proof-bytes")
                    evidence = session_root / "evidence" / (ONE_VARIANT + "_cora.json")
                    evidence.parent.mkdir(parents=True, exist_ok=True)
                    evidence.write_text("{}\n", encoding="utf-8")
                    values = {
                        field: (len(proof.read_bytes()) if field == "proof_size_bytes" else index + 1.0)
                        for index, field in enumerate(metric_fields)
                        if field != missing
                    }
                    metrics_path = session_root / "evidence" / (ONE_VARIANT + "_cora.metrics.json")
                    metrics_path.write_text(json.dumps({
                        "metrics": values,
                        "proof_path": str(proof.resolve()),
                        "run_dir": str(proof.parent.resolve()),
                    }), encoding="utf-8")
                    rejected = workspace.support_command(
                        "matrix-update", "--session-root", str(session_root),
                        "--variant", ONE_VARIANT, "--dataset", "cora", "--cell-status", "success",
                        "--command-rc", "0", "--logger-rc", "0", "--validation-rc", "0",
                        "--metrics-json", str(metrics_path), "--evidence-path", str(evidence),
                        "--proof-path", str(proof),
                    )
                    self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
                    matrix = json.loads((session_root / "matrix_summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(matrix["cells"][ONE_VARIANT + "/cora"]["status"], "running")
                    self.assertEqual(matrix["succeeded"], 0)

    def test_terminal_complete_converts_residual_running_cell_to_failure_and_keeps_pending_unattempted(self) -> None:
        with mock_workspace() as workspace:
            session_root = workspace.root / "terminal-running-session"
            initialized = self._matrix_init(workspace, session_root)
            self.assertEqual(initialized.returncode, 0, initialized.stdout)
            metrics_path = session_root / "metrics.json"
            metrics_path.write_text(json.dumps({"metrics": {"commitment_time_ms": 1.0}}), encoding="utf-8")
            started = workspace.support_command(
                "matrix-update", "--session-root", str(session_root),
                "--variant", ONE_VARIANT, "--dataset", "cora", "--cell-status", "running",
                "--cell-phase", "proof", "--command-rc", "17", "--logger-rc", "31",
                "--metrics-json", str(metrics_path), "--evidence-path", "evidence/unverified.json",
                "--proof-path", "results/unverified/proof.bin",
            )
            self.assertEqual(started.returncode, 0, started.stdout)
            terminal = workspace.support_command(
                "matrix-update", "--session-root", str(session_root),
                "--worker-status", "complete", "--phase", "matrix_complete",
            )
            self.assertEqual(terminal.returncode, 0, terminal.stdout)
            matrix = json.loads((session_root / "matrix_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(matrix["status"], "finished_with_failures")
            self.assertEqual(matrix["worker_status"], "finished_with_failures")
            self.assertEqual((matrix["total"], matrix["attempted"], matrix["succeeded"], matrix["failed"], matrix["pending"]),
                             (5, 1, 0, 1, 4))
            failed = matrix["cells"][ONE_VARIANT + "/cora"]
            self.assertEqual(failed["status"], "failed")
            self.assertIn("runner_state_incomplete", failed["reason"])
            self.assertEqual((failed["command_rc"], failed["logger_rc"], failed["validation_rc"]), (17, 31, None))
            self.assertEqual(failed["metrics"], {})
            self.assertNotIn("evidence_path", failed)
            self.assertNotIn("proof_path", failed)
            self.assertEqual(matrix["pending"], 4)
            for dataset in DATASETS[1:]:
                pending = matrix["cells"][ONE_VARIANT + "/" + dataset]
                self.assertEqual(pending["status"], "pending")
                self.assertIn("before the cell was attempted", pending["reason"])

    def test_worker_failure_with_all_success_cells_remains_session_failure(self) -> None:
        with mock_workspace() as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertEqual(completed.returncode, 0, completed.stdout)
            session_root = workspace.session()
            matrix = json.loads((session_root / "matrix_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(all(
                matrix["cells"][ONE_VARIANT + "/" + dataset].get("verification_status") == "VERIFY_OK"
                for dataset in DATASETS
            ))
            with (session_root / "successes.csv").open(encoding="utf-8", newline="") as stream:
                success_rows = list(csv.DictReader(stream))
            self.assertEqual(len(success_rows), 5)
            self.assertTrue(all(row.get("verification_status") == "VERIFY_OK" for row in success_rows))
            success_jsonl = [
                json.loads(line)
                for line in (session_root / "successes.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(success_jsonl), 5)
            self.assertTrue(all(row.get("verification_status") == "VERIFY_OK" for row in success_jsonl))
            updated = workspace.support_command(
                "matrix-update", "--session-root", str(session_root),
                "--worker-status", "finished_with_failures", "--phase", "matrix_complete",
                "--session-failure-reason", "injected worker finalization failure",
                "--session-failure-rc", "53",
            )
            self.assertEqual(updated.returncode, 0, updated.stdout)
            matrix = json.loads((session_root / "matrix_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(matrix["status"], "finished_with_failures")
            self.assertEqual(matrix["worker_status"], "finished_with_failures")
            self.assertEqual((matrix["attempted"], matrix["succeeded"], matrix["failed"], matrix["pending"]), (5, 5, 0, 0))
            self.assertEqual(matrix["session_failure_reason"], "injected worker finalization failure")
            self.assertEqual(matrix["session_failure_rc"], 53)
            viewed = workspace.support_command("view", "--project-root", str(workspace.project), "--mode", "summary")
            self.assertEqual(viewed.returncode, 0, viewed.stdout)
            self.assertIn("MATRIX_SESSION_FAILURE", viewed.stdout)
            self.assertIn("injected worker finalization failure", viewed.stdout)
            self.assertIn("rc=53", viewed.stdout)
            self.assertIn("verification_status=VERIFY_OK", viewed.stdout)

    def test_failed_transition_clears_success_metrics_but_retains_proof_diagnostic(self) -> None:
        target = ONE_VARIANT + "/cora"
        with mock_workspace() as workspace:
            completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
            self.assertEqual(completed.returncode, 0, completed.stdout)
            session_root = workspace.session()
            matrix = json.loads((session_root / "matrix_summary.json").read_text(encoding="utf-8"))
            previous = matrix["cells"][target]
            self.assertEqual(previous.get("verification_status"), "VERIFY_OK")
            failed = workspace.support_command(
                "matrix-update", "--session-root", str(session_root),
                "--variant", ONE_VARIANT, "--dataset", "cora", "--cell-status", "failed",
                "--reason", "injected post-validation state failure",
                "--metrics-json", str(session_root / "evidence" / (ONE_VARIANT + "_cora.metrics.json")),
                "--evidence-path", previous["evidence_path"], "--proof-path", previous["proof_path"],
            )
            self.assertEqual(failed.returncode, 0, failed.stdout)
            matrix = json.loads((session_root / "matrix_summary.json").read_text(encoding="utf-8"))
            failed_cell = matrix["cells"][target]
            self.assertEqual(failed_cell["status"], "failed")
            self.assertEqual(failed_cell["metrics"], {})
            self.assertNotIn("verification_status", failed_cell)
            self.assertNotIn("evidence_path", failed_cell)
            self.assertEqual(failed_cell.get("proof_path"), previous["proof_path"])
            self.assertEqual((matrix["succeeded"], matrix["failed"]), (4, 1))
            with (session_root / "failures.csv").open(encoding="utf-8", newline="") as stream:
                failures_csv = list(csv.DictReader(stream))
            self.assertEqual(len(failures_csv), 1)
            failed_row = failures_csv[0]
            self.assertEqual(failed_row["dataset"], "cora")
            self.assertEqual(failed_row.get("verification_status", ""), "")
            for field in METRIC_FIELDS:
                self.assertEqual(failed_row.get(field, ""), "")
            failure_jsonl = [
                json.loads(line)
                for line in (session_root / "failures.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(failure_jsonl), 1)
            self.assertNotIn("verification_status", failure_jsonl[0])
            self.assertEqual(failure_jsonl[0]["metrics"], {})
            success_jsonl = [
                json.loads(line)
                for line in (session_root / "successes.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(success_jsonl), 4)
            self.assertTrue(all(row.get("verification_status") == "VERIFY_OK" for row in success_jsonl))

    def test_sidecar_write_failure_does_not_commit_new_matrix_summary(self) -> None:
        with mock_workspace() as workspace:
            session_root = workspace.root / "sidecar-failure-session"
            initialized = self._matrix_init(workspace, session_root)
            self.assertEqual(initialized.returncode, 0, initialized.stdout)
            summary_path = session_root / "matrix_summary.json"
            previous = summary_path.read_bytes()
            (session_root / "successes.csv").unlink()
            (session_root / "successes.csv").mkdir()
            failed = workspace.support_command(
                "matrix-update", "--session-root", str(session_root), "--phase", "sidecar_failure_probe",
            )
            self.assertNotEqual(failed.returncode, 0, failed.stdout)
            self.assertEqual(summary_path.read_bytes(), previous)

    def _write_view_fixture(self, workspace: Workspace, session_id: str = "session-a") -> pathlib.Path:
        session_root = workspace.project / "runs" / "ablation_sessions" / "gpu_matrix" / session_id
        session_root.mkdir(parents=True)
        log_rel = "logs/" + ONE_VARIANT + "_cora.log"
        cells = {}
        for dataset in DATASETS:
            cells[ONE_VARIANT + "/" + dataset] = {
                "variant": ONE_VARIANT,
                "dataset": dataset,
                "status": "running" if dataset == "cora" else "pending",
                "phase": "proof" if dataset == "cora" else "not_started",
                "log": log_rel if dataset == "cora" else "",
                "reason": "",
                "command_rc": None,
                "logger_rc": None,
                "validation_rc": None,
                "metrics": {},
                "stage_rcs": {},
            }
        (session_root / "matrix_summary.json").write_text(
            json.dumps({
                "format": "zkgat_ablation_gpu_matrix_session_v1",
                "session_id": session_id,
                "session_root": str(session_root.resolve()),
                "status": "running",
                "worker_status": "running",
                "variants": [ONE_VARIANT],
                "datasets": list(DATASETS),
                "cells": cells,
                "total": 5,
                "attempted": 1,
                "succeeded": 0,
                "failed": 0,
                "pending": 4,
                "running": 1,
                "unresolved": 0,
                "current_cell": {
                    "active": True,
                    "variant": ONE_VARIANT,
                    "dataset": "cora",
                    "phase": "proof",
                    "log": "runs/ablation_sessions/gpu_matrix/" + session_id + "/" + log_rel,
                },
            }),
            encoding="utf-8",
        )
        (session_root / "logs").mkdir()
        workspace.latest_file.parent.mkdir(parents=True, exist_ok=True)
        workspace.latest_file.write_text(session_id + "\n", encoding="utf-8")
        (session_root / "current_cell.json").write_text(json.dumps({
            "active": True,
            "variant": ONE_VARIANT,
            "dataset": "cora",
            "phase": "proof",
            "log": "runs/ablation_sessions/gpu_matrix/" + session_id + "/" + log_rel,
        }), encoding="utf-8")
        return session_root

    def test_follow_current_rejects_external_log_and_does_not_leak_contents(self) -> None:
        with mock_workspace() as workspace:
            session_root = self._write_view_fixture(workspace)
            outside = workspace.root / "outside-secret.log"
            outside.write_text("OUTSIDE_SENTINEL_SHOULD_NOT_LEAK\n", encoding="utf-8")
            matrix_path = session_root / "matrix_summary.json"
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            cell = matrix["cells"][ONE_VARIANT + "/cora"]
            cell["log"] = str(outside)
            matrix["current_cell"]["log"] = str(outside)
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            (session_root / "current_cell.json").write_text(json.dumps({
                "active": True, "variant": ONE_VARIANT, "dataset": "cora", "phase": "proof", "log": str(outside),
            }), encoding="utf-8")
            escaped = workspace.support_command(
                "view", "--project-root", str(workspace.project), "--mode", "follow-current", timeout=5,
            )
            self.assertNotEqual(escaped.returncode, 0, escaped.stdout)
            self.assertNotIn("OUTSIDE_SENTINEL_SHOULD_NOT_LEAK", escaped.stdout)

            link = session_root / "logs" / "linked-cell.log"
            canonical_link = session_root / "logs" / (ONE_VARIANT + "_cora.log")
            canonical_link.symlink_to(outside)
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            cell = matrix["cells"][ONE_VARIANT + "/cora"]
            cell["log"] = "logs/" + ONE_VARIANT + "_cora.log"
            matrix["current_cell"]["log"] = "runs/ablation_sessions/gpu_matrix/session-a/" + cell["log"]
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            (session_root / "current_cell.json").write_text(json.dumps({
                "active": True, "variant": ONE_VARIANT, "dataset": "cora", "phase": "proof",
                "log": "runs/ablation_sessions/gpu_matrix/session-a/" + cell["log"],
            }), encoding="utf-8")
            symlinked = workspace.support_command(
                "view", "--project-root", str(workspace.project), "--mode", "follow-current", timeout=5,
            )
            self.assertNotEqual(symlinked.returncode, 0, symlinked.stdout)
            self.assertNotIn("OUTSIDE_SENTINEL_SHOULD_NOT_LEAK", symlinked.stdout)

    def test_follow_current_allows_only_exact_cell_fallback_log(self) -> None:
        with mock_workspace() as workspace:
            session_root = self._write_view_fixture(workspace, "fallback-session")
            fallback_name = ONE_VARIANT + "_cora.failure.log"
            fallback_log = session_root / fallback_name
            fallback_log.write_text("FALLBACK_CELL_LOG_SENTINEL\n", encoding="utf-8")
            matrix_path = session_root / "matrix_summary.json"
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            session_relative = "runs/ablation_sessions/gpu_matrix/fallback-session/" + fallback_name
            matrix["cells"][ONE_VARIANT + "/cora"]["log"] = fallback_name
            matrix["current_cell"]["log"] = session_relative
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            (session_root / "current_cell.json").write_text(json.dumps({
                "active": True, "variant": ONE_VARIANT, "dataset": "cora", "phase": "startup_inventory",
                "log": session_relative,
            }), encoding="utf-8")
            process = subprocess.Popen(
                ["python3", str(workspace.project / "scripts" / "ablation_matrix_support.py"),
                 "view", "--project-root", str(workspace.project), "--mode", "follow-current"],
                cwd=workspace.project,
                env=workspace.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            self.assertIsNotNone(process.stdout)
            output_parts = []
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and "FALLBACK_CELL_LOG_SENTINEL" not in "".join(output_parts):
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    output_parts.append(os.read(process.stdout.fileno(), 4096).decode("utf-8", errors="replace"))
            self.assertIn("FALLBACK_CELL_LOG_SENTINEL", "".join(output_parts))
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix.update({"status": "finished_with_failures", "worker_status": "finished_with_failures"})
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            tail, _ = process.communicate(timeout=5)
            output = "".join(output_parts) + tail.decode("utf-8", errors="replace")
            self.assertEqual(process.returncode, 0, output)
            self.assertIn("MATRIX_FOLLOW_CURRENT_COMPLETE", output)

            unapproved = session_root / "other.failure.log"
            unapproved.write_text("UNAPPROVED_SESSION_ROOT_LOG_SENTINEL\n", encoding="utf-8")
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix.update({"status": "running", "worker_status": "running"})
            matrix["cells"][ONE_VARIANT + "/cora"]["log"] = "other.failure.log"
            matrix["current_cell"]["log"] = "runs/ablation_sessions/gpu_matrix/fallback-session/other.failure.log"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            (session_root / "current_cell.json").write_text(json.dumps({
                "active": True, "variant": ONE_VARIANT, "dataset": "cora", "phase": "startup_inventory",
                "log": "runs/ablation_sessions/gpu_matrix/fallback-session/other.failure.log",
            }), encoding="utf-8")
            rejected = workspace.support_command(
                "view", "--project-root", str(workspace.project), "--mode", "follow-current", timeout=5,
            )
            self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
            self.assertNotIn("UNAPPROVED_SESSION_ROOT_LOG_SENTINEL", rejected.stdout)

    def test_latest_session_pointer_must_match_summary_identity(self) -> None:
        with mock_workspace() as workspace:
            session_root = self._write_view_fixture(workspace, "session-latest")
            matrix_path = session_root / "matrix_summary.json"
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix["session_id"] = "another-session"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            viewed = workspace.support_command(
                "view", "--project-root", str(workspace.project), "--mode", "summary",
            )
            self.assertNotEqual(viewed.returncode, 0, viewed.stdout)
            self.assertNotIn("MATRIX_SUMMARY", viewed.stdout)
            self.assertNotIn("SUCCESS", viewed.stdout)

    def test_build_failure_accounts_for_all_cells_and_never_runs_stale_binary(self) -> None:
        with mock_workspace(build_failure=True) as workspace:
            build = workspace.project / "build_cuda"
            build.mkdir()
            stale = build / "gatzk_run"
            stale.write_text("#!/bin/sh\nprintf stale >> \"$MOCK_RUN_CALLS\"\n", encoding="utf-8")
            stale.chmod(0o755)
            completed = workspace.command()
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            summary = workspace.summary()
            self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (10, 0, 10, 0))
            self.assertEqual(summary["status"], "finished_with_failures")
            self.assertEqual(workspace.call_records(workspace.run_calls), [])
            self.assertTrue(all(cell["status"] == "failed" and cell["phase"] == "build" for cell in summary["cells"].values()))

    def test_detached_background_survives_parent_hup_and_exposes_live_views(self) -> None:
        target = ONE_VARIANT + "/cora"
        with mock_workspace(gate_target=target) as workspace:
            launcher_log = workspace.root / "launcher.log"
            parent_returned = workspace.root / "parent-returned"
            outer = (
                "bash \"$MOCK_ENTRYPOINT\" --background --protocol-variant "
                + ONE_VARIANT
                + " --backend gpu; printf returned > \"$MOCK_PARENT_RETURNED\"; sleep 60"
            )
            env = workspace.env.copy()
            env["MOCK_ENTRYPOINT"] = str(workspace.script)
            env["MOCK_PARENT_RETURNED"] = str(parent_returned)
            launcher_log_stream = launcher_log.open("wb")
            launcher = subprocess.Popen(
                ["bash", "-c", outer],
                cwd=workspace.project,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=launcher_log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_for(parent_returned.exists, timeout=15, message="background launcher did not return")
                wait_for(pathlib.Path(env["MOCK_RUN_STARTED"]).exists, timeout=25, message="detached first cell did not start")
                session_id = workspace.latest_session_id()
                session = workspace.session()
                self.assertRegex(session_id, r"^[A-Za-z0-9_.-]+$")
                summary = workspace.summary()
                self.assertEqual(summary["status"], "running")
                self.assertEqual(summary["current_cell"]["variant"], ONE_VARIANT)
                self.assertEqual(summary["current_cell"]["dataset"], "cora")
                cell_logs = sorted((session / "logs").glob("wo_*_*.log"))
                self.assertEqual([path.name for path in cell_logs], [ONE_VARIANT + "_cora.log"])

                summary_view = workspace.command("--summary")
                self.assertEqual(summary_view.returncode, 0, summary_view.stdout)
                self.assertIn(session_id, summary_view.stdout)
                self.assertIn("running", summary_view.stdout)
                failures_view = workspace.command("--failures")
                self.assertEqual(failures_view.returncode, 0, failures_view.stdout)
                self.assertNotIn("MOCK_CHILD_FAILURE", failures_view.stdout)

                for flag in ("--follow", "--follow-current"):
                    follower = subprocess.Popen(
                        ["bash", str(workspace.script), flag],
                        cwd=workspace.project,
                        env=workspace.env,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        output = read_until(follower, b"MOCK_RUN_PAUSED_STDOUT", timeout=5)
                        self.assertIn(b"MOCK_RUN_PAUSED_STDOUT", output, flag)
                        self.assertIn(b"MOCK_RUN_PAUSED_STDERR", output, flag)
                    finally:
                        follower.send_signal(signal.SIGTERM)
                        try:
                            follower.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            follower.kill()
                            follower.wait(timeout=3)
                        if follower.stdout is not None:
                            follower.stdout.close()

                os.killpg(launcher.pid, signal.SIGHUP)
                launcher.wait(timeout=5)
                worker_pid = int(summary["worker_pid"])
                os.kill(worker_pid, 0)
                pathlib.Path(env["MOCK_RUN_RELEASE"]).write_text("continue\n", encoding="utf-8")
                finished = self._assert_finished(workspace, timeout=30)
                self.assertEqual(finished["status"], "complete")
                self.assertEqual(finished["succeeded"], 5)
                self.assertEqual(len(workspace.call_records(workspace.run_calls)), 5)

                second = workspace.command("--background", "--protocol-variant", ONE_VARIANT, "--backend", "gpu")
                self.assertEqual(second.returncode, 0, second.stdout)
                second_id = workspace.latest_session_id()
                self.assertNotEqual(session_id, second_id)
                second_summary = self._assert_finished(workspace, timeout=30)
                self.assertEqual(second_summary["session_id"], second_id)
                self.assertEqual(workspace.latest_file.read_text(encoding="utf-8").strip(), second_id)
                self.assertFalse((workspace.latest_file.parent / "latest.tmp").exists())
            finally:
                pathlib.Path(env["MOCK_RUN_RELEASE"]).write_text("continue\n", encoding="utf-8")
                if launcher.poll() is None:
                    os.killpg(launcher.pid, signal.SIGTERM)
                    launcher.wait(timeout=5)
                launcher_log_stream.close()

    def test_manifest_report_source_metrics_and_proof_size_reject_bad_values(self) -> None:
        target = ONE_VARIANT + "/cora"
        mutations = (
            ("variant mismatch", "manifest:protocol_name:zkgat", "protocol_name"),
            ("report hash mismatch", "report-hash-mismatch", "verifier_report_hash"),
            ("manifest source mismatch", "manifest:source_tree_hash:" + "0" * 64, "source_tree_hash"),
            ("report source mismatch", "report:source_tree_hash:" + "0" * 64, "source tree"),
            ("commitment NaN", "nan:commitment_time_ms", "commitment_time_ms"),
            ("prove negative", "negative:prove_time_ms", "prove_time_ms"),
            ("verify NaN", "nan:verify_time_ms", "verify_time_ms"),
            ("proof size missing", "remove-proof-size", "proof_size_bytes"),
            ("proof size differs", "proof-size-mismatch", "proof_size_bytes"),
            ("zero CUDA invocations", "zero-gpu-kernels", "CUDA kernel invocation"),
            ("missing CUDA runtime", "gpu-runtime-absent", "CUDA runtime evidence"),
            ("missing CUDA device", "empty-gpu-device", "gpu_device"),
        )
        for name, mutation, expected_field in mutations:
            with self.subTest(name=name), mock_workspace(run_mutate_target=target, run_mutation=mutation) as workspace:
                completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
                self.assertNotEqual(completed.returncode, 0, completed.stdout)
                summary = workspace.summary()
                self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"], summary["pending"]), (5, 4, 1, 0))
                failed = summary["cells"][target]
                self.assertEqual(failed["status"], "failed")
                self.assertNotEqual(failed["validation_rc"], 0)
                self.assertIn(expected_field, failed["reason"] + " " + failed["phase"])

    def test_metric_nonfinite_and_negative_checks_cover_all_headline_fields(self) -> None:
        target = ONE_VARIANT + "/cora"
        mutations = []
        for field in ("commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"):
            mutations.append((field + " NaN", "nan:" + field))
            mutations.append((field + " negative", "negative:" + field))
        mutations.append(("consistent but negative commitment", "negative-consistent-commitment"))
        for name, mutation in mutations:
            with self.subTest(name=name), mock_workspace(run_mutate_target=target, run_mutation=mutation) as workspace:
                completed = workspace.command("--protocol-variant", ONE_VARIANT, "--backend", "gpu")
                self.assertNotEqual(completed.returncode, 0, completed.stdout)
                summary = workspace.summary()
                self.assertEqual(summary["failed"], 1)
                self.assertEqual(summary["succeeded"], 4)
                self.assertGreater(summary["cells"][target]["validation_rc"], 0)


if __name__ == "__main__":
    unittest.main()
