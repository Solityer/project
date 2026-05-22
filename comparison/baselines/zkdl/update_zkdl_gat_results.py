#!/usr/bin/env python3
"""从当前 zkDL-GAT 实验 artifact 同源导出正式表、partial 表、coverage 与 README。"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "zkdl_gat_outputs"
RESULTS = OUT / "results"
EVIDENCE = OUT / "evidence"
RUNS = OUT / "runs"

DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]
OFFICIAL_COLUMNS = [
    "dataset",
    "status",
    "blocker",
    "benchmark_mode",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
]
FORMAL_COLUMNS = [
    "dataset",
    "component",
    "status",
    "blocker",
    "benchmark_mode",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
    "total_chunks",
    "verified_chunks",
    "verified_relations",
    "blocked_relations",
    "proof_artifact_paths",
    "evidence_paths",
    "resume_point",
    "note",
]
DEBUG_COLUMNS = [
    "dataset",
    "status",
    "blocker",
    "benchmark_mode",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
    "proof_path",
]

GPU_OFFICIAL_BLOCKER = (
    "gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、"
    "sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation "
    "覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，"
    "但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、"
    "target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。"
)
CPU_OFFICIAL_BLOCKER = (
    "cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 "
    "hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、"
    "all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；"
    "official 四项保持 NA。"
)
PARTIAL_BLOCKED = (
    "target-node max/shift; ExpMap lookup/range table; LeakyReLU/ELU lookup-range table; "
    "all-layer/all-head runner; output attention aggregation; output bias/average binding; "
    "real rescale chain; rounding sign/tie-breaking public contract"
)
CPU_VERIFIED_RELATIONS = (
    "SparseSourceRoutingRelationProof;SparseDestinationRoutingRelationProof;"
    "GroupedSumRelationProof;denominator GroupedSumRelationProof;"
    "inverse HadamardRelationProof;inverse SparseDestinationRoutingRelationProof;"
    "alpha HadamardRelationProof;alpha-denominator HadamardRelationProof"
)
GPU_VERIFIED_RELATIONS = (
    "SparseSourceRoutingRelationProof;SparseDestinationRoutingRelationProof;"
    "GroupedSumRelationProof;denominator GroupedSumRelationProof;"
    "inverse HadamardRelationProof;inverse SparseDestinationRoutingRelationProof;"
    "alpha HadamardRelationProof;multi-column weighted aggregation relation"
)
GPU_SPARSE_VERIFIED_RELATIONS = (
    "SparseSourceRoutingRelationProof;SparseDestinationRoutingRelationProof;GroupedSumRelationProof"
)
GPU_OUTPUT_BINDING_RELATIONS = "output average LinearRelationProof;public logits MLE binding"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_keyvals(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def cpu_run_dir(dataset: str) -> Path:
    return RUNS / dataset / "cpu" / "warm"


def gpu_run_dir(dataset: str) -> Path:
    return RUNS / dataset / "gpu" / "warm"


def cpu_proof(dataset: str) -> Path:
    return cpu_run_dir(dataset) / "formal_artifacts" / "zkdl_cpu_relation_partial.proof.txt"


def cpu_evidence(dataset: str) -> Path:
    return EVIDENCE / f"cpu_relation_partial_{dataset}.json"


def cpu_blocker(dataset: str) -> Path:
    return cpu_run_dir(dataset) / "formal_artifacts" / "zkdl_cpu_formal_chain.blocker.txt"


def cpu_timeout(dataset: str) -> Path:
    return cpu_run_dir(dataset) / "formal_artifacts" / "zkdl_cpu_formal_chain.timeout.txt"


def cpu_log(dataset: str) -> Path:
    return cpu_run_dir(dataset) / "cpu_formal_chain.log"


def official_manifest(device: str, dataset: str) -> Path:
    run_dir = cpu_run_dir(dataset) if device == "cpu" else gpu_run_dir(dataset)
    return run_dir / "run_manifest.json"


def gpu_proof(dataset: str, kind: str) -> Path:
    filename = {
        "sparse": "zkdl_sparse_grouped_relations.proof.txt",
        "softmax": "zkdl_softmax_aggregation_relations.proof.txt",
    }[kind]
    return gpu_run_dir(dataset) / "formal_artifacts" / filename


def gpu_evidence(dataset: str, kind: str) -> Path:
    prefix = "sparse_grouped_relation" if kind == "sparse" else "softmax_aggregation_relation"
    return EVIDENCE / f"{prefix}_{dataset}.json"


def gpu_output_binding_proof(dataset: str) -> Path:
    return gpu_run_dir(dataset) / "formal_artifacts" / "zkdl_output_binding_relations.proof.txt"


def gpu_output_binding_evidence(dataset: str) -> Path:
    return EVIDENCE / f"output_binding_relation_{dataset}.json"


def full_chain_status_file(device: str, dataset: str) -> Path:
    run_dir = cpu_run_dir(dataset) if device == "cpu" else gpu_run_dir(dataset)
    return run_dir / "formal_artifacts" / "zkdl_full_chain_scope.status.txt"


def full_chain_evidence(device: str, dataset: str) -> Path:
    return EVIDENCE / f"full_chain_scope_{device}_{dataset}.json"


def full_chain_scope_state(device: str, dataset: str) -> dict[str, str]:
    status_path = full_chain_status_file(device, dataset)
    evidence_path = full_chain_evidence(device, dataset)
    data = read_json_object(evidence_path)
    missing = data.get("missing_clauses")
    missing_count = len(missing) if isinstance(missing, list) else ""
    return {
        "full_chain_status_path": rel(status_path) if status_path.exists() else "",
        "full_chain_evidence_path": rel(evidence_path) if evidence_path.exists() else "",
        "full_chain_status": str(data.get("status", "")) if data else "",
        "full_chain_missing_clause_count": str(missing_count),
    }


def migrate_legacy_gpu_runs() -> None:
    for dataset in DATASETS:
        old = RUNS / dataset / "warm"
        new = gpu_run_dir(dataset)
        if not old.exists():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        if not new.exists():
            shutil.move(str(old), str(new))
            continue
        for child in old.iterdir():
            target = new / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
        old.rmdir()


def cpu_status(dataset: str) -> dict[str, str]:
    scope = full_chain_scope_state("cpu", dataset)
    item = read_json(cpu_evidence(dataset))
    if (
        item.get("status") == "partial_formal"
        and item.get("relation_scope") == "hidden_layer0_head0_sparse_softmax_partial"
        and item.get("denominator_grouped_sum") == "PASS"
        and item.get("alpha_times_denominator") == "PASS"
        and cpu_proof(dataset).exists()
    ):
        item["status"] = "partial_formal"
        item["proof_path"] = rel(cpu_proof(dataset))
        item["evidence_path"] = rel(cpu_evidence(dataset))
        item["log_path"] = rel(cpu_log(dataset)) if cpu_log(dataset).exists() else ""
        item.update(scope)
        return item
    if cpu_timeout(dataset).exists():
        item = read_keyvals(cpu_timeout(dataset))
        item.update(
            {
                "dataset": dataset,
                "device": "cpu",
                "status": "failed_with_evidence",
                "blocker": "CPU formal chain 在该 full-dataset 尝试中超过单数据集时间预算；这是该数据集的最小 failure evidence，不是 official proof。",
                "proof_path": rel(cpu_timeout(dataset)),
                "evidence_path": rel(cpu_timeout(dataset)),
                "log_path": rel(cpu_log(dataset)) if cpu_log(dataset).exists() else "",
            }
        )
        item.update(scope)
        return item
    if cpu_blocker(dataset).exists():
        item = read_keyvals(cpu_blocker(dataset))
        item.update(
            {
                "dataset": dataset,
                "device": "cpu",
                "status": "failed_with_evidence",
                "blocker": item.get("blocker", "CPU formal chain 在该数据集未能生成 partial proof artifact；保留最小 failure evidence。"),
                "proof_path": rel(cpu_blocker(dataset)),
                "evidence_path": rel(cpu_blocker(dataset)),
                "log_path": rel(cpu_log(dataset)) if cpu_log(dataset).exists() else "",
            }
        )
        item.update(scope)
        return item
    row = {
        "dataset": dataset,
        "device": "cpu",
        "status": "blocked",
        "blocker": "当前缺少由 --formal-cpu-chain 生成的 current-run artifact；该行不能使用旧 probe 结果。",
        "proof_path": "",
        "evidence_path": "",
        "log_path": "",
    }
    row.update(scope)
    return row


def gpu_status(dataset: str) -> dict[str, str]:
    scope = full_chain_scope_state("gpu", dataset)
    output_binding = read_json(gpu_output_binding_evidence(dataset))
    softmax = read_json(gpu_evidence(dataset, "softmax"))
    if (
        softmax.get("status") == "partial_formal"
        and softmax.get("denominator_grouped_sum") == "PASS"
        and softmax.get("alpha_times_denominator") == "PASS"
        and gpu_proof(dataset, "softmax").exists()
    ):
        softmax["device"] = "gpu"
        softmax["proof_path"] = rel(gpu_proof(dataset, "softmax"))
        softmax["evidence_path"] = rel(gpu_evidence(dataset, "softmax"))
        if (
            output_binding.get("status") == "partial_formal"
            and output_binding.get("output_average") == "PASS"
            and output_binding.get("public_logits_binding") == "PASS"
            and gpu_output_binding_proof(dataset).exists()
        ):
            softmax["output_binding"] = "PASS"
            softmax["output_binding_proof_path"] = rel(gpu_output_binding_proof(dataset))
            softmax["output_binding_evidence_path"] = rel(gpu_output_binding_evidence(dataset))
        softmax.update(scope)
        return softmax
    sparse = read_json(gpu_evidence(dataset, "sparse"))
    if sparse.get("status") == "partial_formal" and gpu_proof(dataset, "sparse").exists():
        sparse["device"] = "gpu"
        sparse["proof_path"] = rel(gpu_proof(dataset, "sparse"))
        sparse["evidence_path"] = rel(gpu_evidence(dataset, "sparse"))
        sparse["status"] = "partial_formal"
        if (
            output_binding.get("status") == "partial_formal"
            and output_binding.get("output_average") == "PASS"
            and output_binding.get("public_logits_binding") == "PASS"
            and gpu_output_binding_proof(dataset).exists()
        ):
            sparse["output_binding"] = "PASS"
            sparse["output_binding_proof_path"] = rel(gpu_output_binding_proof(dataset))
            sparse["output_binding_evidence_path"] = rel(gpu_output_binding_evidence(dataset))
        sparse.update(scope)
        return sparse
    row = {
        "dataset": dataset,
        "device": "gpu",
        "status": "blocked",
        "blocker": "当前缺少 GPU formal partial artifact；需运行 --formal-sparse-grouped 与 --formal-softmax-aggregation。",
        "proof_path": "",
        "evidence_path": "",
    }
    row.update(scope)
    return row


def all_statuses() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "cpu": {dataset: cpu_status(dataset) for dataset in DATASETS},
        "gpu": {dataset: gpu_status(dataset) for dataset in DATASETS},
    }


def official_row_from_manifest(device: str, dataset: str) -> dict[str, object] | None:
    manifest = read_json_object(official_manifest(device, dataset))
    if not manifest:
        return None
    if str(manifest.get("benchmark_mode", "")) != "warm":
        return None
    if str(manifest.get("status", "")).lower() not in {"ok", "verify_ok"}:
        return None
    if str(manifest.get("is_full_dataset", "")).lower() != "true":
        return None
    if str(manifest.get("verified", "")).lower() != "true":
        return None
    if str(manifest.get("proof_scope", "")) != "zkdl_gat_full":
        return None
    return {
        "dataset": dataset,
        "status": "ok",
        "blocker": "",
        "benchmark_mode": "warm",
        "commitment_time_ms": manifest.get("commitment_time_ms", "NA"),
        "prove_time_ms": manifest.get("prove_time_ms", "NA"),
        "verify_time_ms": manifest.get("verify_time_ms", "NA"),
        "proof_size_bytes": manifest.get("proof_size_bytes", "NA"),
    }


def official_rows(device: str) -> list[dict[str, object]]:
    blocker = CPU_OFFICIAL_BLOCKER if device == "cpu" else GPU_OFFICIAL_BLOCKER
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        official = official_row_from_manifest(device, dataset)
        if official is not None:
            rows.append(official)
        else:
            rows.append({
            "dataset": dataset,
            "status": "blocked",
            "blocker": blocker,
            "benchmark_mode": "warm",
            "commitment_time_ms": "NA",
            "prove_time_ms": "NA",
            "verify_time_ms": "NA",
            "proof_size_bytes": "NA",
            })
    return rows


def status_label(status: object) -> str:
    if str(status) == "ok":
        return "已完成"
    if str(status) == "blocked":
        return "阻塞"
    return str(status)


def write_latest_and_summary(statuses: dict[str, dict[str, dict[str, str]]]) -> None:
    aggregate: list[dict[str, object]] = []
    for device in DEVICES:
        rows = official_rows(device)
        write_csv(RESULTS / device / "latest.csv", OFFICIAL_COLUMNS, rows)
        write_json(
            RESULTS / device / "latest.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "device": device,
                "benchmark_mode": "warm",
                "policy": "只有 full dataset + complete proof generation + complete verification 才能写 official 四项。",
                "rows": rows,
            },
        )
        lines = [
            "# 最新正式结果",
            "",
            f"device: `{device}`；partial/debug 数字不进入 official latest。",
            "",
            "| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | 阻塞说明 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['dataset']} | {status_label(row['status'])} | {row['commitment_time_ms']} | "
                f"{row['prove_time_ms']} | {row['verify_time_ms']} | {row['proof_size_bytes']} | {row['blocker']} |"
            )
            aggregate.append({"device": device, **row})
        lines.extend(
            [
                "",
                "五个全量数据集的非 official 状态：",
                "",
                "| dataset | formal/debug state | artifact |",
                "| --- | --- | --- |",
            ]
        )
        for dataset in DATASETS:
            state = statuses[device][dataset]
            artifact = state.get("proof_path") or state.get("evidence_path") or "NA"
            lines.append(f"| {dataset} | {state.get('status', 'blocked')} | {artifact} |")
        (RESULTS / device / "summary.md").parent.mkdir(parents=True, exist_ok=True)
        (RESULTS / device / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(RESULTS / "latest.csv", ["device", *OFFICIAL_COLUMNS], aggregate)
    write_json(
        RESULTS / "latest.json",
        {"generated_at": datetime.now(timezone.utc).isoformat(), "benchmark_mode": "warm", "rows": aggregate},
    )
    summary = [
        "# 最新正式结果",
        "",
        "CPU/GPU 分表见 `results/cpu/summary.md` 与 `results/gpu/summary.md`。",
        "五个全量数据集 cora、citeseer、pubmed、ppi、ogbn-arxiv 均保留单独状态，不能用前三个数据集替代。",
    ]
    (RESULTS / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def formal_row_for_cpu(dataset: str, state: dict[str, str]) -> dict[str, object]:
    evidence_paths = ";".join(filter(None, [
        state.get("evidence_path", ""),
        state.get("log_path", ""),
        state.get("full_chain_evidence_path", ""),
        state.get("full_chain_status_path", ""),
    ]))
    if state.get("status") == "partial_formal":
        return {
            "dataset": dataset,
            "component": "cpu_sparse_softmax_hidden_layer0_head0_full_graph",
            "status": "partial_formal",
            "blocker": "cpu_partial_formal_relation_only_not_full_gat",
            "benchmark_mode": "cpu_only_formal_relation_proof_verify",
            "commitment_time_ms": state.get("commitment_time_ms", "NA"),
            "prove_time_ms": state.get("prove_time_ms", "NA"),
            "verify_time_ms": state.get("verify_time_ms", "NA"),
            "proof_size_bytes": state.get("proof_size_bytes", "NA"),
            "total_chunks": "8",
            "verified_chunks": "8",
            "verified_relations": CPU_VERIFIED_RELATIONS,
            "blocked_relations": PARTIAL_BLOCKED,
            "proof_artifact_paths": state.get("proof_path", ""),
            "evidence_paths": evidence_paths,
            "resume_point": rel(cpu_run_dir(dataset) / "RESUME_POINT.md"),
            "note": "CPU-only host Fr/G1/Commitment/opening relation proof/verify 已运行；full-chain scope 已单独记录，但不是完整 GAT official proof。",
        }
    return {
        "dataset": dataset,
        "component": "cpu_formal_chain_full_dataset_attempt",
        "status": state.get("status", "blocked"),
        "blocker": state.get("blocker", CPU_OFFICIAL_BLOCKER),
        "benchmark_mode": "cpu_only_formal_relation_attempt",
        "commitment_time_ms": "NA",
        "prove_time_ms": "NA",
        "verify_time_ms": "NA",
        "proof_size_bytes": "NA",
        "total_chunks": "8",
        "verified_chunks": "0",
        "verified_relations": "",
        "blocked_relations": PARTIAL_BLOCKED,
        "proof_artifact_paths": state.get("proof_path", ""),
        "evidence_paths": evidence_paths,
        "resume_point": rel(cpu_run_dir(dataset) / "RESUME_POINT.md"),
        "note": "该数据集有独立 CPU 状态；未使用其他数据集或 GPU 结果代替。",
    }


def formal_row_for_gpu(dataset: str, state: dict[str, str]) -> dict[str, object]:
    evidence_paths = ";".join(filter(None, [
        state.get("evidence_path", ""),
        state.get("output_binding_evidence_path", ""),
        state.get("output_binding_proof_path", ""),
        state.get("full_chain_evidence_path", ""),
        state.get("full_chain_status_path", ""),
    ]))
    if state.get("status") == "partial_formal":
        has_weighted = state.get("weighted_aggregation") == "PASS"
        total = "13" if has_weighted else "3"
        verified = total
        return {
            "dataset": dataset,
            "component": "gpu_sparse_softmax_aggregation_hidden_layer0_head0_full_graph"
            if has_weighted else "gpu_sparse_grouped_hidden_layer0_head0_full_graph",
            "status": "partial_formal",
            "blocker": "gpu_partial_formal_relation_only_not_full_gat",
            "benchmark_mode": "gpu_formal_relation_proof_verify_host_verifier",
            "commitment_time_ms": state.get("commitment_time_ms", "NA"),
            "prove_time_ms": state.get("prove_time_ms", "NA"),
            "verify_time_ms": state.get("verify_time_ms", "NA"),
            "proof_size_bytes": state.get("proof_size_bytes", "NA"),
            "total_chunks": total,
            "verified_chunks": verified,
            "verified_relations": (
                (GPU_VERIFIED_RELATIONS if has_weighted else GPU_SPARSE_VERIFIED_RELATIONS)
                + ((";" + GPU_OUTPUT_BINDING_RELATIONS) if state.get("output_binding") == "PASS" else "")
            ),
            "blocked_relations": PARTIAL_BLOCKED,
            "proof_artifact_paths": state.get("proof_path", ""),
            "evidence_paths": evidence_paths,
            "resume_point": rel(gpu_run_dir(dataset) / "RESUME_POINT.md"),
            "note": "GPU prover 使用 CUDA Fr/G1 commitment/opening；当前 verifier 在 host 重放。output binding 与 full-chain scope 已单独记录，仍不能写成 GPU official verifier。",
        }
    return {
        "dataset": dataset,
        "component": "gpu_formal_chain_full_dataset_attempt",
        "status": state.get("status", "blocked"),
        "blocker": state.get("blocker", GPU_OFFICIAL_BLOCKER),
        "benchmark_mode": "gpu_formal_relation_attempt",
        "commitment_time_ms": "NA",
        "prove_time_ms": "NA",
        "verify_time_ms": "NA",
        "proof_size_bytes": "NA",
        "total_chunks": "13",
        "verified_chunks": "0",
        "verified_relations": "",
        "blocked_relations": PARTIAL_BLOCKED,
        "proof_artifact_paths": state.get("proof_path", ""),
        "evidence_paths": evidence_paths,
        "resume_point": rel(gpu_run_dir(dataset) / "RESUME_POINT.md"),
        "note": "该数据集有独立 GPU 状态；未使用 CPU 结果代替。",
    }


def write_formal_partial(statuses: dict[str, dict[str, dict[str, str]]]) -> None:
    aggregate: list[dict[str, object]] = []
    for device in DEVICES:
        rows = []
        for dataset in DATASETS:
            row = formal_row_for_cpu(dataset, statuses[device][dataset]) if device == "cpu" else formal_row_for_gpu(dataset, statuses[device][dataset])
            rows.append(row)
            aggregate.append({"device": device, **row})
        write_csv(RESULTS / device / "formal_partial_debug.csv", FORMAL_COLUMNS, rows)
        write_json(RESULTS / device / "formal_partial_debug.json", {"device": device, "rows": rows})
    write_csv(RESULTS / "formal_partial_debug.csv", ["device", *FORMAL_COLUMNS], aggregate)
    write_json(RESULTS / "formal_partial_debug.json", {"rows": aggregate})


def write_debug_level_b() -> None:
    aggregate: list[dict[str, object]] = []
    for device in DEVICES:
        rows = []
        for dataset in DATASETS:
            row = {
                "dataset": dataset,
                "status": "debug_only",
                "blocker": "debug_level_b 不是 formal benchmark，不进入 official latest；CPU/GPU 结果保持独立。",
                "benchmark_mode": f"{device}_debug_level_b_output_binding_only_not_formal_level_a",
                "commitment_time_ms": "NA",
                "prove_time_ms": "NA",
                "verify_time_ms": "NA",
                "proof_size_bytes": "NA",
                "proof_path": "",
            }
            rows.append(row)
            aggregate.append({"device": device, **row})
        write_csv(RESULTS / device / "debug_level_b.csv", DEBUG_COLUMNS, rows)
        write_json(RESULTS / device / "debug_level_b.json", {"device": device, "rows": rows})
    write_csv(RESULTS / "debug_level_b.csv", ["device", *DEBUG_COLUMNS], aggregate)
    write_json(RESULTS / "debug_level_b.json", {"rows": aggregate})


def write_coverage(statuses: dict[str, dict[str, dict[str, str]]]) -> None:
    cpu_partial = sum(1 for dataset in DATASETS if statuses["cpu"][dataset].get("status") == "partial_formal")
    gpu_partial = sum(1 for dataset in DATASETS if statuses["gpu"][dataset].get("status") == "partial_formal")
    cpu_scope = sum(1 for dataset in DATASETS if statuses["cpu"][dataset].get("full_chain_evidence_path"))
    gpu_scope = sum(1 for dataset in DATASETS if statuses["gpu"][dataset].get("full_chain_evidence_path"))
    gpu_softmax = sum(
        1 for dataset in DATASETS
        if statuses["gpu"][dataset].get("denominator_grouped_sum") == "PASS"
        and statuses["gpu"][dataset].get("alpha_times_denominator") == "PASS"
    )
    gpu_weighted = sum(
        1 for dataset in DATASETS
        if statuses["gpu"][dataset].get("weighted_aggregation") == "PASS"
    )
    gpu_output_binding = sum(
        1 for dataset in DATASETS
        if statuses["gpu"][dataset].get("output_binding") == "PASS"
    )
    cpu_failed = sum(1 for dataset in DATASETS if statuses["cpu"][dataset].get("status") == "failed_with_evidence")
    rows = [
        {
            "relation_name": "commitment / evaluation binding",
            "cpu_status": "supported",
            "gpu_status": "supported",
            "total_scope": "selftest + partial relation artifacts",
            "verified_scope": "Gate1 + relation artifacts",
            "blocker": "NA",
            "cora": True,
            "citeseer": True,
            "pubmed": True,
            "ppi": True,
            "ogbn_arxiv": True,
            "evidence_file": "zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log;zkdl_gat_outputs/results/formal_partial_debug.csv",
            "notes": "zkDL-native commitment/opening；未使用主仓库 KZG。",
        },
        {
            "relation_name": "transcript / sumcheck replay",
            "cpu_status": "supported",
            "gpu_status": "supported",
            "total_scope": "selftest + relation verifier",
            "verified_scope": "Gate2 + Gate3",
            "blocker": "NA",
            "cora": True,
            "citeseer": True,
            "pubmed": True,
            "ppi": True,
            "ogbn_arxiv": True,
            "evidence_file": "zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log;zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log",
            "notes": "zkDL transcript 语义独立于主仓库 transcript challenge。",
        },
        {
            "relation_name": "SparseRouting / GroupedSum",
            "cpu_status": f"partial_formal_{cpu_partial}/5_failed_{cpu_failed}/5",
            "gpu_status": f"partial_formal_{gpu_partial}/5",
            "total_scope": "hidden layer0/head0 full graph / full edge set",
            "verified_scope": "relation-level",
            "blocker": "not full GAT",
            "cora": True,
            "citeseer": True,
            "pubmed": True,
            "ppi": True,
            "ogbn_arxiv": True,
            "evidence_file": "zkdl_gat_outputs/results/formal_partial_debug.csv",
            "notes": "每个 device 分目录保存 artifact，CPU/GPU 不互相覆盖。",
        },
        {
            "relation_name": "denominator / inverse / alpha",
            "cpu_status": f"partial_formal_{cpu_partial}/5_failed_{cpu_failed}/5",
            "gpu_status": f"partial_formal_{gpu_softmax}/5",
            "total_scope": "hidden layer0/head0 denominator, inverse, alpha consistency",
            "verified_scope": "relation-level",
            "blocker": "softmax max/shift 与 ExpMap lookup/range 未闭合",
            "cora": statuses["gpu"]["cora"].get("denominator_grouped_sum") == "PASS",
            "citeseer": statuses["gpu"]["citeseer"].get("denominator_grouped_sum") == "PASS",
            "pubmed": statuses["gpu"]["pubmed"].get("denominator_grouped_sum") == "PASS",
            "ppi": statuses["gpu"]["ppi"].get("denominator_grouped_sum") == "PASS",
            "ogbn_arxiv": statuses["gpu"]["ogbn-arxiv"].get("denominator_grouped_sum") == "PASS",
            "evidence_file": "zkdl_gat_outputs/results/formal_partial_debug.csv",
            "notes": "inverse relation 使用 denom * inv = 1；verifier 不读取 witness。",
        },
        {
            "relation_name": "lookup / range / max-comparison primitive",
            "cpu_status": "partial_formal_selftest_only",
            "gpu_status": "partial_formal_selftest_only",
            "total_scope": "primitive selftest",
            "verified_scope": "range bit decomposition; one-hot lookup; representative max constraint",
            "blocker": "尚未接入 full GAT ExpMap/ELU table 与 per-destination max VK_static 字段",
            "cora": False,
            "citeseer": False,
            "pubmed": False,
            "ppi": False,
            "ogbn_arxiv": False,
            "evidence_file": "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log",
            "notes": "已有 tamper rejection；full GAT 接入仍 blocked。",
        },
        {
            "relation_name": "weighted aggregation",
            "cpu_status": "blocked",
            "gpu_status": f"partial_formal_{gpu_weighted}/5",
            "total_scope": "hidden layer0/head0",
            "verified_scope": "GPU multi-column relation where artifact exists",
            "blocker": "CPU weighted aggregation 未完成；output aggregation 未完成",
            "cora": statuses["gpu"]["cora"].get("weighted_aggregation") == "PASS",
            "citeseer": statuses["gpu"]["citeseer"].get("weighted_aggregation") == "PASS",
            "pubmed": statuses["gpu"]["pubmed"].get("weighted_aggregation") == "PASS",
            "ppi": statuses["gpu"]["ppi"].get("weighted_aggregation") == "PASS",
            "ogbn_arxiv": statuses["gpu"]["ogbn-arxiv"].get("weighted_aggregation") == "PASS",
            "evidence_file": "zkdl_gat_outputs/results/gpu/formal_partial_debug.csv",
            "notes": "GPU partial 不等于 full GAT official。",
        },
        {
            "relation_name": "full-chain scope / proof-object guard",
            "cpu_status": f"scope_evidence_{cpu_scope}/5",
            "gpu_status": f"scope_evidence_{gpu_scope}/5",
            "total_scope": "five full datasets × CPU/GPU",
            "verified_scope": "scope recorded only; no official proof",
            "blocker": "full pi_GAT^zkDL proof object 与 clean verifier 尚未闭合",
            "cora": bool(statuses["cpu"]["cora"].get("full_chain_evidence_path")) and bool(statuses["gpu"]["cora"].get("full_chain_evidence_path")),
            "citeseer": bool(statuses["cpu"]["citeseer"].get("full_chain_evidence_path")) and bool(statuses["gpu"]["citeseer"].get("full_chain_evidence_path")),
            "pubmed": bool(statuses["cpu"]["pubmed"].get("full_chain_evidence_path")) and bool(statuses["gpu"]["pubmed"].get("full_chain_evidence_path")),
            "ppi": bool(statuses["cpu"]["ppi"].get("full_chain_evidence_path")) and bool(statuses["gpu"]["ppi"].get("full_chain_evidence_path")),
            "ogbn_arxiv": bool(statuses["cpu"]["ogbn-arxiv"].get("full_chain_evidence_path")) and bool(statuses["gpu"]["ogbn-arxiv"].get("full_chain_evidence_path")),
            "evidence_file": "zkdl_gat_outputs/evidence/full_chain_scope_<device>_<dataset>.json",
            "notes": "该行是正式流程的 full-chain 范围门禁；它防止 partial artifact 被导出为 official。",
        },
        {
            "relation_name": "output average / public logits binding",
            "cpu_status": "blocked",
            "gpu_status": f"partial_formal_{gpu_output_binding}/5",
            "total_scope": "full graph output heads and public logits MLE",
            "verified_scope": "GPU output-head average linear relation where artifact exists",
            "blocker": "CPU output binding 未接入；output attention/bias 前序链、rescale/rounding 与 clean verifier 未闭合",
            "cora": statuses["gpu"]["cora"].get("output_binding") == "PASS",
            "citeseer": statuses["gpu"]["citeseer"].get("output_binding") == "PASS",
            "pubmed": statuses["gpu"]["pubmed"].get("output_binding") == "PASS",
            "ppi": statuses["gpu"]["ppi"].get("output_binding") == "PASS",
            "ogbn_arxiv": statuses["gpu"]["ogbn-arxiv"].get("output_binding") == "PASS",
            "evidence_file": "zkdl_gat_outputs/evidence/output_binding_relation_<dataset>.json",
            "notes": "该 relation 使用 public logits MLE，不把完整 logits tensor 放入 proof object；仍不是 full GAT official。",
        },
        {
            "relation_name": "all-layer/all-head + output aggregation",
            "cpu_status": "blocked",
            "gpu_status": "blocked",
            "total_scope": "full GAT",
            "verified_scope": "NA",
            "blocker": "scope 已建模；runner 尚未把所有 hidden/output heads、bias、average 与 public logits binding 串成一个完整 proof object",
            "cora": False,
            "citeseer": False,
            "pubmed": False,
            "ppi": False,
            "ogbn_arxiv": False,
            "evidence_file": "README.md;zkdl_gat_outputs/results/formal_partial_debug.csv",
            "notes": "不能把 hidden layer0/head0 partial chain 写成 complete。",
        },
        {
            "relation_name": "real rescale / rounding sign/tie-breaking",
            "cpu_status": "blocked",
            "gpu_status": "blocked",
            "total_scope": "full GAT fixed-point chain",
            "verified_scope": "NA",
            "blocker": "缺公开 rho/delta/rem_bits/tie_policy 字段；不能用 host equality 或 debug comparison 代替 proof",
            "cora": False,
            "citeseer": False,
            "pubmed": False,
            "ppi": False,
            "ogbn_arxiv": False,
            "evidence_file": "README.md;zkdl_gat_outputs/results/coverage_matrix.md",
            "notes": "该 blocker 来自协议字段缺口，不是主仓库 proof scheme 可迁移项。",
        },
        {
            "relation_name": "debug_level_b output binding",
            "cpu_status": "debug_only",
            "gpu_status": "debug_only",
            "total_scope": "debug_level_b",
            "verified_scope": "not official",
            "blocker": "debug_only 不进入 official latest",
            "cora": False,
            "citeseer": False,
            "pubmed": False,
            "ppi": False,
            "ogbn_arxiv": False,
            "evidence_file": "zkdl_gat_outputs/results/debug_level_b.csv",
            "notes": "保留作结果口径说明，不作为验真机制。",
        },
    ]
    columns = [
        "relation_name",
        "cpu_status",
        "gpu_status",
        "total_scope",
        "verified_scope",
        "blocker",
        "cora",
        "citeseer",
        "pubmed",
        "ppi",
        "ogbn_arxiv",
        "evidence_file",
        "notes",
    ]
    write_csv(RESULTS / "coverage_matrix.csv", columns, rows)
    write_json(RESULTS / "coverage_matrix.json", {"rows": rows})
    lines = [
        "# zkDL-GAT coverage_matrix",
        "",
        "| relation | cpu_status | gpu_status | total_scope | verified_scope | blocker | cora | citeseer | pubmed | ppi | ogbn-arxiv | evidence | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['relation_name']} | {row['cpu_status']} | {row['gpu_status']} | {row['total_scope']} | "
            f"{row['verified_scope']} | {row['blocker']} | {str(row['cora']).lower()} | "
            f"{str(row['citeseer']).lower()} | {str(row['pubmed']).lower()} | {str(row['ppi']).lower()} | "
            f"{str(row['ogbn_arxiv']).lower()} | {row['evidence_file']} | {row['notes']} |"
        )
    (RESULTS / "coverage_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_resumes(statuses: dict[str, dict[str, dict[str, str]]]) -> None:
    for device in DEVICES:
        for dataset in DATASETS:
            state = statuses[device][dataset]
            run_dir = cpu_run_dir(dataset) if device == "cpu" else gpu_run_dir(dataset)
            run_dir.mkdir(parents=True, exist_ok=True)
            lines = [
                f"# zkDL-GAT {device.upper()} 继续点",
                "",
                f"- 数据集: `{dataset}`",
                f"- 当前状态: `{state.get('status', 'blocked')}`；official 仍为 `blocked + NA`。",
                f"- artifact: `{state.get('proof_path') or state.get('evidence_path') or 'NA'}`",
                f"- full-chain scope: `{state.get('full_chain_evidence_path') or 'NA'}`；missing clauses `{state.get('full_chain_missing_clause_count') or 'NA'}`。",
                "- 五个全量数据集缺一不可；该文件只描述当前 dataset/device，不可替代其他行。",
                "- Level A blocker: full GAT proof object、all-layer/all-head、output aggregation、real rescale/rounding 仍未闭合。",
            ]
            (run_dir / "RESUME_POINT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# zkDL-GAT 全局继续点", ""]
    for device in DEVICES:
        lines.append(f"## {device.upper()}")
        for dataset in DATASETS:
            state = statuses[device][dataset]
            lines.append(f"- `{dataset}`: `{state.get('status', 'blocked')}`")
        lines.append("")
    (OUT / "RESUME_POINT.md").write_text("\n".join(lines), encoding="utf-8")


def dataset_status_table(statuses: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    lines = [
        "| dataset | CPU 状态 | GPU 状态 | CPU artifact | GPU artifact |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset in DATASETS:
        cpu = statuses["cpu"][dataset]
        gpu = statuses["gpu"][dataset]
        lines.append(
            f"| {dataset} | {cpu.get('status', 'blocked')} | {gpu.get('status', 'blocked')} | "
            f"{cpu.get('proof_path') or cpu.get('evidence_path') or 'NA'} | "
            f"{gpu.get('proof_path') or gpu.get('evidence_path') or 'NA'} |"
        )
    return lines


def official_status_table() -> list[str]:
    lines = [
        "| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for device in DEVICES:
        for row in official_rows(device):
            lines.append(
                f"| {device} | {row['dataset']} | {row['status']} | {row['benchmark_mode']} | "
                f"{row['commitment_time_ms']} | {row['prove_time_ms']} | {row['verify_time_ms']} | "
                f"{row['proof_size_bytes']} | {row['blocker']} |"
            )
    return lines


def write_readmes(statuses: dict[str, dict[str, dict[str, str]]]) -> None:
    rows = dataset_status_table(statuses)
    official_rows_text = official_status_table()
    readme = [
        "# zkDL-GAT baseline",
        "",
        "本目录是 zkDL baseline 的 GAT 正式对比实验目录。实验对象固定为五个全量数据集：`cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`，缺一不可；不能用前三个数据集替代完整实验。",
        "",
        "## 实验目的",
        "",
        "在同一批数据/cache/checkpoint/config 口径下，用 zkDL 自身 proof system 适配 GAT，并与主仓库方案做论文对比。这里比较的是证明方案，不比较 CUDA forward kernel timing，也不把 partial formal 写成 official。",
        "",
        "## 协议边界",
        "",
        "`/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 是最高协议边界。本目录只使用 zkDL baseline 自身的 Hyrax-style tensor commitment、opening、transcript、sumcheck 与 relation proof。",
        "",
        "## 与主仓库 proof scheme 的边界",
        "",
        "禁止迁移主仓库 KZG proof schema、quotient polynomial、verifier equation、opening proof rule、transcript challenge 语义和 proof object layout。本轮修改没有引入这些内容。",
        "",
        "## 目录结构",
        "",
        "- zkDL baseline 核心：`bls12-381.cuh`、`fr-tensor.cuh`、`g1-tensor.cuh`、`commitment.cuh`、`proof.cuh`、`formal_relations.cuh`。",
        "- GAT 数据/config/checkpoint/cache 加载：`gat_modules/data_cache.cuh`、`gat_modules/model_bundle.cuh`、`configs/zkdl_gat_full_datasets.cfg`。",
        "- public topology / group context：`gat_modules/formal_sparse_utils.cuh`。",
        "- CPU forward / trace：`gat_modules/cpu_forward.cuh`。",
        "- GPU forward / trace：`gat_modules/gpu_forward.cuh`。",
        "- CPU formal backend：`gat_modules/cpu_formal_backend.cuh`。",
        "- GPU formal backend：`gat_modules/formal_softmax_runners.cuh`。",
        "- CPU formal runner：`gat_modules/cpu_formal_runner.cuh` 调用 CPU-only relation backend。",
        "- GPU formal runner：`gat_modules/gpu_formal_runner.cuh` 调用 GPU trace 与 relation gadget。",
        "- GPU output binding runner：`gat_modules/output_binding_runner.cuh` 在真实 full graph output trace 上生成 output-head average / public logits MLE relation artifact。",
        "- full-chain scope runner：`gat_modules/full_chain_scope.cuh` 为每个 dataset/device 记录完整协议范围、已有 partial artifact 和未闭合协议子句；它不写 official manifest。",
        "- CLI 调度：`gat_modules/run_dispatch.cuh` 只做 dataset/device 选择，`gat_zkdl.cu` 只保留薄 CLI 入口。",
        "- runner 侧 artifact/evidence/proof_debug/RESUME 写入辅助：`gat_modules/artifact_writer.cuh` 集中管理 CPU/GPU partial 运行产物，避免 runner/backend 内联文件格式。",
        "- result/export 管理：`update_zkdl_gat_results.py` 从真实 artifact 同源导出 latest、summary、coverage、README，并刷新最终 per-device RESUME。",
        "- 五数据集调度脚本：`run_zkdl_gat_experiments.py` 负责自测、per-dataset/per-device 执行与 exporter。",
        "- selftest：`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。",
        "",
        "## CPU path",
        "",
        "`--formal-cpu-chain` 使用 CPU-only host Fr/G1/Commitment/opening，不依赖 CUDA proof backend。当前可生成并验证 hidden layer0/head0 full graph/full edge set 的 sparse routing、grouped sum、denominator、inverse、inverse routing、alpha consistency relation proof。`--formal-full-chain-scope --device cpu` 会额外记录 CPU 全链协议范围和缺失子句，但不会写 official manifest。CPU verifier 不读取 witness，proof object 不包含完整 tensor。",
        "",
        "## GPU path",
        "",
        "`--formal-sparse-grouped`、`--formal-softmax-aggregation` 与 `--formal-output-binding` 使用 GPU Fr/G1 commitment/opening 与 CUDA trace，host 侧重放 verifier。`--formal-output-binding` 证明 output-head average 与 public logits MLE 的线性关系；`--formal-full-chain-scope --device gpu` 会额外记录 GPU 全链协议范围和缺失子句，但不会写 official manifest。GPU kernel timing 不写入 official proof/verify timing；host replay verifier 也不会被写成 GPU full verifier。",
        "",
        "## CPU/GPU 结果口径",
        "",
        "CPU/GPU 的 result、artifact、proof_debug、RESUME 均位于 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，互不覆盖、互不冒充。official 只允许 full dataset + complete proof generation + complete verification 写入四项指标。",
        "",
        "## 如何判断结果有效",
        "",
        "runner 只写本次真实运行产生的 artifact；artifact_writer 统一 runner 侧 artifact/evidence/proof_debug/RESUME 格式；full-chain scope runner 记录完整协议门禁但不写 official manifest；exporter 只读 artifact/evidence 并刷新最终结果表与 RESUME；`latest/summary/formal_partial_debug/coverage` 同源导出。旧 probe、旧 schedule、旧 debug、backend 不清楚的结果不作为有效实验结论。",
        "",
        "## 构建和运行",
        "",
        "```bash",
        "make clean && make all",
        "./run_zkdl_gat_experiments.py --dataset all --device all --output-root zkdl_gat_outputs --commitment-block-size 64",
        "```",
        "",
        "也可以按 dataset/device 分阶段执行；每个 dataset/device 的 relation runner 之后会执行 full-chain scope 记录。任一 dataset/device 失败后，runner 会继续调度剩余项，最后统一导出 result 表。",
        "",
        "## result 文件说明",
        "",
        "- `results/<device>/latest.csv|json`：只含 official 四项；未 full verified 时为 `blocked + NA`。",
        "- `results/<device>/summary.md`：official 摘要与五数据集状态。",
        "- `results/<device>/formal_partial_debug.csv|json`：relation-level partial formal 或 dataset-specific failure evidence。",
        "- `results/<device>/debug_level_b.csv|json`：debug_only，不进入 official。",
        "- `results/coverage_matrix.*`：按 CPU/GPU 区分 supported、partial_formal、debug_only、blocked。",
        "",
        "## supported relation",
        "",
        "当前支持 zkDL-native commitment/opening、transcript/sumcheck、HadamardRelationProof、MatmulRelationProof、LinearRelationProof、SparseSourceRoutingRelationProof、SparseDestinationRoutingRelationProof、GroupedSumRelationProof、MultiColumnGroupedSumRelationProof。GPU 路径新增真实 GAT output-head average / public logits MLE binding 的 LinearRelationProof artifact。",
        "",
        "## partial_formal relation",
        "",
        "CPU partial formal 覆盖 hidden layer0/head0 sparse + denominator/inverse/alpha。GPU partial formal 当前五个数据集都有 sparse/grouped artifact；denominator/inverse/alpha、weighted aggregation 与 output binding 覆盖范围以 coverage_matrix 为准。lookup/range/max/comparison 已有 selftest 级 tamper rejection，但尚未接入 full GAT table/context。",
        "",
        "## blocked relation",
        "",
        "blocked 项包括 target-node max/shift、ExpMap lookup/range table、LeakyReLU/ELU lookup-range table、all-layer/all-head proof runner、output attention aggregation、bias/average/public logits binding、real rescale chain、rounding sign/tie-breaking public contract 和 clean verifier。full-chain scope 已记录这些子句，但尚未生成可 official 的 proof object。",
        "",
        "## 当前五数据集状态",
        "",
        *rows,
        "",
        "## official 结果表",
        "",
        *official_rows_text,
        "",
        "## CPU 当前状态",
        "",
        "CPU path 已实现 `--formal-cpu-chain`，并按五个全量数据集逐一运行。`--formal-full-chain-scope --device cpu` 记录每个数据集的 full-chain 范围、hidden/output head 数和 missing clauses。当前五个数据集均有 full graph/full edge set 的 relation-level `partial_formal` artifact；CPU official 仍为 NA，因为 full GAT proof object、all-layer/all-head proof、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding 尚未闭合。",
        "",
        "## GPU 当前状态",
        "",
        "GPU path 当前五个全量数据集都有 relation-level `partial_formal` sparse/grouped artifact；cora、citeseer、ppi、ogbn-arxiv 还有 denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation artifact。`--formal-output-binding` 在真实 output trace 上生成 output-head average / public logits MLE binding artifact；`--formal-full-chain-scope --device gpu` 记录每个数据集的 full-chain 范围、hidden/output head 数和 missing clauses。pubmed 当前只保留 sparse/grouped artifact，不把旧 softmax 失败日志或缺失 artifact 当作有效结论。GPU verifier 当前是 host replay，不写成 GPU full verifier；GPU official 仍为 NA。",
        "",
        "## Level A 状态",
        "",
        "当前不是 Level A。CPU/GPU official latest 仍为 `blocked + NA`，因为还没有完整 GAT proof generation + complete verification。",
        "",
        "## 已迁移的 proof-system-neutral 优化",
        "",
        "已迁移或保留：模块化组织、一次加载 dataset/cache/checkpoint、public topology/context 复用、trace 复用、CPU/GPU 分目录、artifact_writer 集中写入、full-chain scope 门禁、output binding public MLE、relation-level batch、multi-column grouped sum、CPU commitment block-size 调度、CPU commitment 按行并行与零/一标量跳过、compact artifact、proof_size 只统计 proof object、同源导出、五数据集失败不中断的结果口径。",
        "",
        "## 不能迁移的主仓库优化",
        "",
        "不能迁移：KZG batch opening、quotient 约束、verifier equation、opening proof rule、transcript challenge 语义、proof object layout、任何 protocol/schema 专属约束。",
        "",
        "## 下一步最小有效动作",
        "",
        "把 selftest 级 lookup/range/max/comparison 接入 full GAT 的 VK_static/table/context；随后把 hidden/output 所有层头、output aggregation、bias、average、public logits binding 与 real rescale/rounding 串成单一 zkDL proof object。",
    ]
    text = "\n".join(readme) + "\n"
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    (OUT / "README_ZKDL_GAT_ADAPTER.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    migrate_legacy_gpu_runs()
    statuses = all_statuses()
    write_latest_and_summary(statuses)
    write_formal_partial(statuses)
    write_debug_level_b()
    write_coverage(statuses)
    write_resumes(statuses)
    write_readmes(statuses)


if __name__ == "__main__":
    main()
