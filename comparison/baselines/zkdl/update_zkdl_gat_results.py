#!/usr/bin/env python3
"""同步 zkDL-GAT 的 CPU/GPU official、partial evidence 与覆盖矩阵。"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "zkdl_gat_outputs"
RESULTS = OUT / "results"
EVIDENCE_DIR = OUT / "evidence"
RUNS = OUT / "runs"

DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
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
AGGREGATE_COLUMNS = ["device", *OFFICIAL_COLUMNS]

GPU_BLOCKER = (
    "proof_gadget_missing：GPU 路径当前只有 relation-level partial formal evidence。"
    "CUDA forward trace 与 zkDL 原生 Fr/G1 commitment/opening kernel 已参与 partial prover，"
    "但仍没有完整 GAT proof object、完整 proof generation 与完整 verification。"
    "当前缺 target-node softmax group、softmax max/shift、ExpMap lookup/range、"
    "LeakyReLU/ELU lookup-range、所有层/头 weighted aggregation、output aggregation、"
    "real rescale chain 与 rounding sign/tie-breaking 公开契约；official 四项保持 NA。"
)

CPU_BLOCKER = (
    "cpu_full_gat_proof_incomplete：本轮已在 zkDL 目录内实现 CPU-only host Fr/G1/Commitment/opening "
    "与 relation prover/verifier，并在 Cora/Citeseer full dataset 的 hidden layer0/head0 sparse/grouped "
    "relation 上跑出独立 CPU proof/verify。CPU official Level A 仍 blocked：Pubmed CPU full-dataset "
    "attempt 进入 proof/verify 后因 host G1 opening replay 超过交互窗口未完成；CPU denominator/inverse/alpha、"
    "lookup/range/max/comparison、ExpMap、LeakyReLU/ELU、output aggregation、all-layer/all-head、"
    "real rescale/rounding public contract 与完整 proof object 仍未闭合。official 四项保持 NA，"
    "不能复用 GPU partial timing。"
)

BLOCKED_RELATIONS = (
    "target-node softmax group; softmax group max/shift; ExpMap lookup/range; "
    "LeakyReLU/ELU lookup-range; all-layer/all-head weighted aggregation; output aggregation; "
    "real rescale chain; rounding sign/tie-breaking 公开契约"
)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


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


def migrate_legacy_gpu_runs() -> None:
    """把旧的 runs/<dataset>/warm 迁移到 runs/<dataset>/gpu/warm。"""
    for dataset in DATASETS:
        old = RUNS / dataset / "warm"
        new = RUNS / dataset / "gpu" / "warm"
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


def gpu_proof_path(dataset: str, kind: str) -> Path:
    filename = {
        "sparse": "zkdl_sparse_grouped_relations.proof.txt",
        "softmax": "zkdl_softmax_aggregation_relations.proof.txt",
    }[kind]
    return RUNS / dataset / "gpu" / "warm" / "formal_artifacts" / filename


def cpu_probe_path(dataset: str) -> Path:
    return RUNS / dataset / "cpu" / "warm" / "formal_artifacts" / "zkdl_cpu_backend_probe.blocker.txt"


def cpu_partial_proof_path(dataset: str) -> Path:
    return RUNS / dataset / "cpu" / "warm" / "formal_artifacts" / "zkdl_cpu_relation_partial.proof.txt"


def cpu_partial_evidence_path(dataset: str) -> Path:
    return EVIDENCE_DIR / f"cpu_relation_partial_{dataset}.json"


def cpu_timeout_path(dataset: str) -> Path:
    return RUNS / dataset / "cpu" / "warm" / "formal_artifacts" / "zkdl_cpu_backend_attempt.timeout.txt"


def cpu_probe_log_path(dataset: str) -> Path:
    return RUNS / dataset / "cpu" / "warm" / "cpu_backend_probe.log"


def evidence_path(dataset: str, kind: str) -> Path:
    prefix = "sparse_grouped_relation" if kind == "sparse" else "softmax_aggregation_relation"
    return EVIDENCE_DIR / f"{prefix}_{dataset}.json"


def normalize_evidence() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    sparse: dict[str, dict[str, str]] = {}
    softmax: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        sparse_item = read_json(evidence_path(dataset, "sparse"))
        if sparse_item:
            sparse_item["device"] = "gpu"
            path = gpu_proof_path(dataset, "sparse")
            if path.exists():
                sparse_item["proof_path"] = str(path)
            write_json(evidence_path(dataset, "sparse"), sparse_item)
            sparse[dataset] = sparse_item

        softmax_item = read_json(evidence_path(dataset, "softmax"))
        if softmax_item:
            softmax_item["device"] = "gpu"
            path = gpu_proof_path(dataset, "softmax")
            if path.exists():
                softmax_item["proof_path"] = str(path)
            write_json(evidence_path(dataset, "softmax"), softmax_item)
            softmax[dataset] = softmax_item
    return sparse, softmax


def normalize_cpu_evidence() -> dict[str, dict[str, str]]:
    """Record current CPU evidence and replace stale backend-missing artifacts."""
    cpu: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        item = read_json(cpu_partial_evidence_path(dataset))
        proof = cpu_partial_proof_path(dataset)
        if item:
            item["device"] = "cpu"
            if proof.exists():
                item["proof_path"] = str(proof)
            write_json(cpu_partial_evidence_path(dataset), item)
            cpu[dataset] = item

    for dataset in ["ppi", "ogbn-arxiv"]:
        probe = cpu_probe_path(dataset)
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(
            "\n".join(
                [
                    "ZKDLGATCPU",
                    "version=2",
                    f"dataset={dataset}",
                    "status=not_run_after_pubmed_cpu_timeout",
                    "backend=cpu_only_host_fr_g1_commitment_available",
                    "proof_generation=not_run",
                    "verification=not_run",
                    "blocker=CPU backend is no longer missing, but Pubmed full-dataset CPU proof/verify timed out in host G1 opening replay; larger PPI/ogbn CPU runs were not started to avoid blind long runs.",
                    "note=This supersedes the old backend-missing probe artifact and is not an official proof result.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return cpu


def normalize_run_logs() -> None:
    """修正旧日志中的无 device proof_path，避免与当前 GPU artifact 目录冲突。"""
    for dataset in DATASETS:
        run_dir = RUNS / dataset / "gpu" / "warm"
        if not run_dir.exists():
            continue
        old_abs = (RUNS / dataset / "warm").as_posix()
        new_abs = run_dir.as_posix()
        old_rel = f"runs/{dataset}/warm/"
        new_rel = f"runs/{dataset}/gpu/warm/"
        for path in run_dir.rglob("*.log"):
            text = path.read_text(encoding="utf-8")
            updated = text.replace(old_abs, new_abs).replace(old_rel, new_rel)
            if updated != text:
                path.write_text(updated, encoding="utf-8")


def official_rows(device: str) -> list[dict[str, object]]:
    blocker = CPU_BLOCKER if device == "cpu" else GPU_BLOCKER
    return [
        {
            "dataset": dataset,
            "status": "blocked",
            "blocker": blocker,
            "benchmark_mode": "warm",
            "commitment_time_ms": "NA",
            "prove_time_ms": "NA",
            "verify_time_ms": "NA",
            "proof_size_bytes": "NA",
        }
        for dataset in DATASETS
    ]


def write_latest_and_summary() -> None:
    aggregate_rows: list[dict[str, object]] = []
    for device in ["cpu", "gpu"]:
        rows = official_rows(device)
        device_dir = RESULTS / device
        write_csv(device_dir / "latest.csv", OFFICIAL_COLUMNS, rows)
        write_json(
            device_dir / "latest.json",
            {
                "device": device,
                "benchmark_mode": "warm",
                "official_policy": "只有 full dataset + complete formal proof generation + complete formal verification 成功才填写四项指标。",
                "rows": rows,
            },
        )

        lines = [
            f"# zkDL-GAT {device.upper()} 最新正式结果",
            "",
            "当前不是 Level A；下表只展示 official 四项，未把 partial/debug 数字写入正式结果。",
            "",
            "| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in rows:
            lines.append(
                f"| {row['dataset']} | {row['status']} | NA | NA | NA | NA | {row['blocker']} |"
            )
        (device_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        aggregate_rows.extend({"device": device, **row} for row in rows)

    write_csv(RESULTS / "latest.csv", AGGREGATE_COLUMNS, aggregate_rows)
    write_json(
        RESULTS / "latest.json",
        {
            "benchmark_mode": "warm",
            "device_layout": {
                "cpu": "zkdl_gat_outputs/results/cpu/latest.csv",
                "gpu": "zkdl_gat_outputs/results/gpu/latest.csv",
            },
            "rows": aggregate_rows,
        },
    )

    lines = [
        "# zkDL-GAT 最新正式结果",
        "",
        "CPU 与 GPU official 结果分目录导出；顶层表只作为聚合索引。当前二者均未达到 Level A，四项指标均保持 NA。",
        "",
        "## CPU official",
        "",
        "| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in official_rows("cpu"):
        lines.append(f"| {row['dataset']} | blocked | NA | NA | NA | NA | {CPU_BLOCKER} |")
    lines.extend(
        [
            "",
            "## GPU official",
            "",
            "| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in official_rows("gpu"):
        lines.append(f"| {row['dataset']} | blocked | NA | NA | NA | NA | {GPU_BLOCKER} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selftest_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "NA",
            "component": "formal_opening_selftest",
            "status": "success",
            "benchmark_mode": "formal_partial_opening_proof_verify",
            "commitment_time_ms": "NA",
            "prove_time_ms": "NA",
            "verify_time_ms": "NA",
            "proof_size_bytes": "see_log",
            "total_chunks": "NA",
            "verified_chunks": "NA",
            "verified_relations": "commitment/evaluation binding",
            "blocked_relations": "",
            "proof_artifact_paths": "zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log",
            "evidence_paths": "zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log",
            "resume_point": "",
            "blocker": "Gate 1 opening selftest PASS；该 selftest 使用 zkDL 当前 CUDA-backed tensor/commitment backend。",
        },
        {
            "dataset": "NA",
            "component": "formal_sumcheck_selftest",
            "status": "success",
            "benchmark_mode": "formal_partial_inner_product_sumcheck_verify",
            "commitment_time_ms": "NA",
            "prove_time_ms": "NA",
            "verify_time_ms": "NA",
            "proof_size_bytes": "see_log",
            "total_chunks": "NA",
            "verified_chunks": "NA",
            "verified_relations": "inner-product sumcheck transcript replay",
            "blocked_relations": "",
            "proof_artifact_paths": "zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log",
            "evidence_paths": "zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log",
            "resume_point": "",
            "blocker": "Gate 2 inner-product transcript replay verifier PASS。",
        },
        {
            "dataset": "NA",
            "component": "formal_relation_selftest",
            "status": "success",
            "benchmark_mode": "formal_partial_hadamard_matmul_linear_sparse_grouped_verify",
            "commitment_time_ms": "NA",
            "prove_time_ms": "NA",
            "verify_time_ms": "NA",
            "proof_size_bytes": "see_log",
            "total_chunks": "NA",
            "verified_chunks": "NA",
            "verified_relations": (
                "HadamardRelationProof;MatmulRelationProof;LinearRelationProof;"
                "SparseSourceRoutingRelationProof;SparseDestinationRoutingRelationProof;"
                "GroupedSumRelationProof;MultiColumnGroupedSumRelationProof"
            ),
            "blocked_relations": "",
            "proof_artifact_paths": "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log",
            "evidence_paths": "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log",
            "resume_point": "",
            "blocker": "Gate 3 relation selftest PASS，包含 sparse/grouped tamper rejection。",
        },
    ]


def gpu_formal_rows(
    sparse: dict[str, dict[str, str]],
    softmax: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows = selftest_rows()
    for dataset in DATASETS:
        item = sparse.get(dataset)
        if not item:
            continue
        rows.append(
            {
                "dataset": dataset,
                "component": "sparse_grouped_hidden_layer0_head0_full_graph",
                "status": "partial_formal",
                "benchmark_mode": "gpu_partial_formal_sparse_grouped_relation_only_not_full_gat",
                "commitment_time_ms": item["commitment_time_ms"],
                "prove_time_ms": item["prove_time_ms"],
                "verify_time_ms": item["verify_time_ms"],
                "proof_size_bytes": item["proof_size_bytes"],
                "total_chunks": "3",
                "verified_chunks": "3",
                "verified_relations": "SparseSourceRoutingRelationProof;SparseDestinationRoutingRelationProof;GroupedSumRelationProof",
                "blocked_relations": BLOCKED_RELATIONS,
                "proof_artifact_paths": item["proof_path"],
                "evidence_paths": rel(evidence_path(dataset, "sparse")),
                "resume_point": rel(RUNS / dataset / "gpu" / "warm" / "RESUME_POINT.md"),
                "blocker": (
                    "GPU partial evidence 只覆盖 hidden layer0/head0 的 source/destination sparse routing 与一维 grouped sum；"
                    "不是完整 GAT proof，也不是 official GPU proof/verify。"
                ),
            }
        )
    for dataset in DATASETS:
        item = softmax.get(dataset)
        if not item:
            continue
        verified = int(item.get("verified_feature_columns", "0"))
        available = int(item.get("available_feature_columns", "0"))
        feature_scope = "full_feature" if available > 0 and verified == available else "partial_feature"
        rows.append(
            {
                "dataset": dataset,
                "component": f"softmax_aggregation_hidden_layer0_head0_full_graph_{feature_scope}",
                "status": "partial_formal",
                "benchmark_mode": "gpu_partial_formal_softmax_aggregation_relation_only_not_full_gat",
                "commitment_time_ms": item["commitment_time_ms"],
                "prove_time_ms": item["prove_time_ms"],
                "verify_time_ms": item["verify_time_ms"],
                "proof_size_bytes": item["proof_size_bytes"],
                "total_chunks": item["available_feature_columns"],
                "verified_chunks": item["verified_feature_columns"],
                "verified_relations": (
                    "denominator grouped sum;inverse Hadamard;inverse destination routing;"
                    "alpha multiplication;alpha_times_denominator;multi-column sparse routing;"
                    "multi-column alpha Hadamard;multi-column grouped sum weighted aggregation"
                ),
                "blocked_relations": BLOCKED_RELATIONS,
                "proof_artifact_paths": item["proof_path"],
                "evidence_paths": rel(evidence_path(dataset, "softmax")),
                "resume_point": rel(RUNS / dataset / "gpu" / "warm" / "RESUME_POINT.md"),
                "blocker": (
                    f"GPU partial evidence 覆盖 hidden layer0/head0 的 denominator/inverse/alpha 与 "
                    f"{verified}/{available} feature weighted aggregation；ExpMap、max/range、所有层/头、"
                    "output aggregation、ELU、rescale 与 rounding 仍未闭合。"
                ),
            }
        )
    return rows


def cpu_formal_rows(cpu: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        item = cpu.get(dataset)
        probe_log = cpu_probe_log_path(dataset)
        if item:
            evidence_paths = [rel(cpu_partial_evidence_path(dataset))]
            if probe_log.exists():
                evidence_paths.append(rel(probe_log))
            rows.append(
                {
                    "dataset": dataset,
                    "component": "cpu_sparse_grouped_hidden_layer0_head0_full_graph",
                    "status": "partial_formal",
                    "benchmark_mode": "cpu_partial_formal_sparse_grouped_relation_only_not_full_gat",
                    "commitment_time_ms": item["commitment_time_ms"],
                    "prove_time_ms": item["prove_time_ms"],
                    "verify_time_ms": item["verify_time_ms"],
                    "proof_size_bytes": item["proof_size_bytes"],
                    "total_chunks": "3",
                    "verified_chunks": "3",
                    "verified_relations": "SparseSourceRoutingRelationProof;SparseDestinationRoutingRelationProof;GroupedSumRelationProof",
                    "blocked_relations": (
                        "CPU denominator/inverse/alpha extension; "
                        + BLOCKED_RELATIONS
                    ),
                    "proof_artifact_paths": item.get("proof_path", rel(cpu_partial_proof_path(dataset))),
                    "evidence_paths": ";".join(evidence_paths),
                    "resume_point": rel(RUNS / dataset / "cpu" / "warm" / "RESUME_POINT.md"),
                    "blocker": (
                        "CPU-only host Fr/G1/Commitment relation proof/verify succeeded for sparse/grouped "
                        "hidden layer0/head0, but this is not a full GAT proof and does not fill official latest."
                    ),
                }
            )
            continue

        timeout = cpu_timeout_path(dataset)
        if timeout.exists():
            rows.append(
                {
                    "dataset": dataset,
                    "component": "cpu_sparse_grouped_full_dataset_attempt",
                    "status": "blocked_after_attempt_timeout",
                    "benchmark_mode": "cpu_partial_formal_attempt_not_completed",
                    "commitment_time_ms": "NA",
                    "prove_time_ms": "NA",
                    "verify_time_ms": "NA",
                    "proof_size_bytes": "NA",
                    "total_chunks": "3",
                    "verified_chunks": "0",
                    "verified_relations": "",
                    "blocked_relations": "host G1 opening replay did not finish for full-dataset CPU proof/verify attempt",
                    "proof_artifact_paths": rel(timeout),
                    "evidence_paths": rel(timeout),
                    "resume_point": rel(RUNS / dataset / "cpu" / "warm" / "RESUME_POINT.md"),
                    "blocker": CPU_BLOCKER,
                }
            )
            continue

        artifact = cpu_probe_path(dataset)
        evidence_paths = [rel(artifact)] if artifact.exists() else []
        if probe_log.exists():
            evidence_paths.append(rel(probe_log))
        rows.append(
            {
                "dataset": dataset,
                "component": "cpu_sparse_grouped_full_dataset_not_run_after_pubmed_timeout",
                "status": "not_run_after_pubmed_timeout",
                "benchmark_mode": "cpu_full_dataset_not_run_to_avoid_blind_long_run",
                "commitment_time_ms": "NA",
                "prove_time_ms": "NA",
                "verify_time_ms": "NA",
                "proof_size_bytes": "NA",
                "total_chunks": "3",
                "verified_chunks": "0",
                "verified_relations": "",
                "blocked_relations": "Pubmed CPU host opening replay timeout; larger CPU run deferred",
                "proof_artifact_paths": rel(artifact) if artifact.exists() else "",
                "evidence_paths": ";".join(evidence_paths),
                "resume_point": rel(RUNS / dataset / "cpu" / "warm" / "RESUME_POINT.md"),
                "blocker": CPU_BLOCKER,
            }
        )
    return rows


def write_formal_partial(
    sparse: dict[str, dict[str, str]],
    softmax: dict[str, dict[str, str]],
    cpu: dict[str, dict[str, str]],
) -> None:
    columns = [
        "dataset",
        "component",
        "status",
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
        "blocker",
    ]
    by_device = {
        "cpu": cpu_formal_rows(cpu),
        "gpu": gpu_formal_rows(sparse, softmax),
    }
    aggregate: list[dict[str, object]] = []
    for device, rows in by_device.items():
        write_csv(RESULTS / device / "formal_partial_debug.csv", columns, rows)
        write_json(RESULTS / device / "formal_partial_debug.json", {"device": device, "rows": rows})
        aggregate.extend({"device": device, **row} for row in rows)
    write_csv(RESULTS / "formal_partial_debug.csv", ["device", *columns], aggregate)
    write_json(RESULTS / "formal_partial_debug.json", {"rows": aggregate})


def write_debug_level_b() -> None:
    columns = [
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
    aggregate: list[dict[str, object]] = []
    for device in ["cpu", "gpu"]:
        rows = []
        for dataset in DATASETS:
            rows.append(
                {
                    "dataset": dataset,
                    "status": "not_run_after_cleanup",
                    "blocker": (
                        "debug_level_b 不是 formal benchmark，不写入 official latest；"
                        f"{device} 结果保持独立，不复用另一 device 的 timing。"
                    ),
                    "benchmark_mode": f"{device}_diagnostic_level_b_output_binding_only_not_formal_level_a",
                    "commitment_time_ms": "NA",
                    "prove_time_ms": "NA",
                    "verify_time_ms": "NA",
                    "proof_size_bytes": "NA",
                    "proof_path": "",
                }
            )
        write_csv(RESULTS / device / "debug_level_b.csv", columns, rows)
        write_json(RESULTS / device / "debug_level_b.json", {"device": device, "rows": rows})
        aggregate.extend({"device": device, **row} for row in rows)
    write_csv(RESULTS / "debug_level_b.csv", ["device", *columns], aggregate)
    write_json(RESULTS / "debug_level_b.json", {"rows": aggregate})


def coverage_rows(
    sparse: dict[str, dict[str, str]],
    softmax: dict[str, dict[str, str]],
    cpu: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    sparse_evidence = ";".join(rel(evidence_path(dataset, "sparse")) for dataset in DATASETS if dataset in sparse)
    softmax_evidence = ";".join(rel(evidence_path(dataset, "softmax")) for dataset in DATASETS if dataset in softmax)
    cpu_evidence = ";".join(rel(cpu_partial_evidence_path(dataset)) for dataset in DATASETS if dataset in cpu)
    return [
        {
            "relation_name": "commitment/evaluation binding",
            "status": "supported",
            "cpu_status": "cpu_only_relation_backend_cora_citeseer; pubmed_timeout",
            "gpu_status": "supported",
            "total_scope": "selftest",
            "verified_scope": "Gate 1 PASS plus CPU relation openings for Cora/Citeseer",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": "zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log;" + cpu_evidence,
            "blocker_reason": "CPU-only backend 已在 relation-level 路径实现；完整 GAT proof 仍未闭合。",
        },
        {
            "relation_name": "HadamardRelationProof/MatmulRelationProof/LinearRelationProof",
            "status": "supported",
            "cpu_status": "cpu_backend_available_not_full_gat",
            "gpu_status": "supported",
            "total_scope": "selftest",
            "verified_scope": "Gate 3 PASS",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log",
            "blocker_reason": "primitive 支持不等于完整 GAT proof；CPU Matmul/Linear 尚未接入完整 GAT CPU official runner。",
        },
        {
            "relation_name": "SparseSourceRoutingRelationProof",
            "status": "partial_formal",
            "cpu_status": "partial_formal_2/5_pubmed_timeout_ppi_ogbn_not_run",
            "gpu_status": "partial_formal_5/5",
            "total_scope": "five full datasets hidden_layer0/head0",
            "verified_scope": "CPU 2/5 + GPU 5/5 dataset evidence",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": sparse_evidence + ";" + cpu_evidence,
            "blocker_reason": "只覆盖 hidden layer0/head0 source routing；完整所有层/头未闭合。",
        },
        {
            "relation_name": "SparseDestinationRoutingRelationProof",
            "status": "partial_formal",
            "cpu_status": "partial_formal_2/5_pubmed_timeout_ppi_ogbn_not_run",
            "gpu_status": "partial_formal_5/5",
            "total_scope": "five full datasets hidden_layer0/head0",
            "verified_scope": "CPU 2/5 + GPU 5/5 dataset evidence",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": sparse_evidence + ";" + cpu_evidence,
            "blocker_reason": "只覆盖 hidden layer0/head0 destination routing；完整所有层/头未闭合。",
        },
        {
            "relation_name": "GroupedSumRelationProof",
            "status": "partial_formal",
            "cpu_status": "partial_formal_2/5_pubmed_timeout_ppi_ogbn_not_run",
            "gpu_status": "partial_formal_5/5",
            "total_scope": "five full datasets hidden_layer0/head0",
            "verified_scope": "CPU 2/5 + GPU 5/5 dataset evidence",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": sparse_evidence + ";" + cpu_evidence,
            "blocker_reason": "一维 grouped sum 已验证；不等同于完整 softmax。",
        },
        {
            "relation_name": "MultiColumnGroupedSumRelationProof",
            "status": "partial_formal",
            "cpu_status": "not_run_cpu_extension",
            "gpu_status": "partial_formal_5/5_full_feature",
            "total_scope": "selftest + five full datasets hidden_layer0/head0",
            "verified_scope": "5/5 dataset 8/8 feature",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log;" + softmax_evidence,
            "blocker_reason": "multi-column grouped sum 使用 zkDL 自身 column compression + matrix opening；仍未覆盖所有层/头与 output aggregation。",
        },
        {
            "relation_name": "softmax denominator sum / inverse / alpha multiplication",
            "status": "partial_formal",
            "cpu_status": "not_run_cpu_extension_after_sparse_grouped_success",
            "gpu_status": "partial_formal_5/5",
            "total_scope": "five full datasets hidden_layer0/head0",
            "verified_scope": "denominator/inverse/alpha PASS",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": softmax_evidence,
            "blocker_reason": "U 仍未由 ExpMap lookup/range 证明；max/shift 与所有层/头未闭合。",
        },
        {
            "relation_name": "weighted aggregation",
            "status": "partial_formal",
            "cpu_status": "not_run_cpu_extension",
            "gpu_status": "partial_formal_5/5_full_feature",
            "total_scope": "five full datasets hidden_layer0/head0",
            "verified_scope": "5/5 dataset 8/8 feature",
            "formal_prover_supported": "true",
            "formal_verifier_supported": "true",
            "transcript_bound": "true",
            "opening_bound": "true",
            "proof_object_clean": "true",
            "evidence_file": softmax_evidence,
            "blocker_reason": "PPI/ogbn 已为 hidden layer0/head0 8/8 feature；所有层/头与 output aggregation 仍未实现。",
        },
        {
            "relation_name": "output aggregation",
            "status": "blocked",
            "cpu_status": "blocked_after_cpu_backend_attempt",
            "gpu_status": "blocked",
            "total_scope": "NA",
            "verified_scope": "NA",
            "formal_prover_supported": "false",
            "formal_verifier_supported": "false",
            "transcript_bound": "false",
            "opening_bound": "false",
            "proof_object_clean": "false",
            "evidence_file": "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md",
            "blocker_reason": "现有 primitive 可表达 source routing、alpha*message、grouped sum；但输出层还缺 output alpha 来源、bias 前 rescale、偏置和最终平均链条接入。",
        },
        {
            "relation_name": "softmax max/shift",
            "status": "blocked",
            "cpu_status": "blocked_no_zkdl_max_comparison_primitive",
            "gpu_status": "blocked",
            "total_scope": "NA",
            "verified_scope": "NA",
            "formal_prover_supported": "false",
            "formal_verifier_supported": "false",
            "transcript_bound": "false",
            "opening_bound": "false",
            "proof_object_clean": "false",
            "evidence_file": "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md",
            "blocker_reason": "缺 per-destination max 代表边选择、非负差分 range 与 comparison primitive。",
        },
        {
            "relation_name": "ExpMap / LeakyReLU / ELU lookup-range",
            "status": "blocked",
            "cpu_status": "blocked_no_zkdl_lookup_range_primitive",
            "gpu_status": "blocked",
            "total_scope": "NA",
            "verified_scope": "NA",
            "formal_prover_supported": "false",
            "formal_verifier_supported": "false",
            "transcript_bound": "false",
            "opening_bound": "false",
            "proof_object_clean": "false",
            "evidence_file": "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md",
            "blocker_reason": "缺选择矩阵 lookup/table/range verifier gadget 与 VK_static 表承诺绑定。",
        },
        {
            "relation_name": "real rescale chain / rounding sign/tie-breaking",
            "status": "blocked",
            "cpu_status": "blocked_no_public_rounding_contract",
            "gpu_status": "blocked",
            "total_scope": "NA",
            "verified_scope": "NA",
            "formal_prover_supported": "false",
            "formal_verifier_supported": "false",
            "transcript_bound": "false",
            "opening_bound": "false",
            "proof_object_clean": "false",
            "evidence_file": "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md",
            "blocker_reason": "缺公开 rho/delta/rem_bits/tie_policy；不能用 host equality 或 residual debug 代替 proof。",
        },
        {
            "relation_name": "debug_level_b output binding",
            "status": "debug_only",
            "cpu_status": "not_run",
            "gpu_status": "not_run_after_cleanup",
            "total_scope": "diagnostic only",
            "verified_scope": "NA",
            "formal_prover_supported": "false",
            "formal_verifier_supported": "false",
            "transcript_bound": "false",
            "opening_bound": "false",
            "proof_object_clean": "false",
            "evidence_file": "zkdl_gat_outputs/results/debug_level_b.csv",
            "blocker_reason": "debug_only 不进入 official latest。",
        },
    ]


def write_coverage(
    sparse: dict[str, dict[str, str]],
    softmax: dict[str, dict[str, str]],
    cpu: dict[str, dict[str, str]],
) -> None:
    rows = coverage_rows(sparse, softmax, cpu)
    columns = [
        "relation_name",
        "status",
        "cpu_status",
        "gpu_status",
        "total_scope",
        "verified_scope",
        "formal_prover_supported",
        "formal_verifier_supported",
        "transcript_bound",
        "opening_bound",
        "proof_object_clean",
        "evidence_file",
        "blocker_reason",
    ]
    write_csv(RESULTS / "coverage_matrix.csv", columns, rows)
    write_json(
        RESULTS / "coverage_matrix.json",
        {
            "coverage_level": "Level A blocked；CPU/GPU device-separated relation-level coverage only",
            "allowed_status": ["supported", "partial_formal", "debug_only", "blocked", "not_implemented"],
            "rows": rows,
        },
    )
    lines = [
        "# GAT proof 覆盖矩阵",
        "",
        "本矩阵区分 `cpu_status` 与 `gpu_status`。`partial_formal` 和 `debug_only` 不计入 Level A。",
        "",
        "| relation | status | CPU | GPU | total | verified | prover | verifier | transcript | opening | clean | evidence | blocker |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['relation_name']} | {row['status']} | {row['cpu_status']} | {row['gpu_status']} | "
            f"{row['total_scope']} | {row['verified_scope']} | {row['formal_prover_supported']} | "
            f"{row['formal_verifier_supported']} | {row['transcript_bound']} | {row['opening_bound']} | "
            f"{row['proof_object_clean']} | {row['evidence_file']} | {row['blocker_reason']} |"
        )
    (RESULTS / "coverage_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocker_evidence() -> None:
    lines = [
        "# zkDL-GAT blocker 证据",
        "",
        "- 本目录只使用 zkDL baseline 自身的 commitment、opening、transcript、sumcheck 与 relation proof。没有引入主仓库 GAT-ZKML 的 KZG proof schema、quotient polynomial、verifier equation、opening proof 规则或 proof object layout。",
        "- CPU-only backend 已实现到 relation-level：`gat_modules/cpu_forward.cuh` 构造 CPU trace，`gat_modules/cpu_formal_backend.cuh` 提供 host Fr/G1/Commitment/opening 与 CPU transcript/sumcheck prover/verifier，`gat_modules/run_dispatch.cuh` 负责 CPU runner/export。",
        "- CPU Cora/Citeseer full dataset hidden layer0/head0 sparse/grouped relation 已真实 proof generation + verification；artifact 分别位于 `runs/cora/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt` 与 `runs/citeseer/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt`。",
        "- CPU Pubmed full dataset 已进入 CPU proof/verify attempt，但 host G1 opening replay 未在交互窗口内完成；failure evidence 位于 `runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_backend_attempt.timeout.txt`。",
        "- PPI/ogbn-arxiv CPU proof 未继续盲跑：Pubmed 已暴露 CPU host opening replay 性能 blocker，较大数据集等待 batched opening/verifier 后再跑。",
        "- GPU partial path 当前真实状态：CUDA forward trace 参与 witness 生成；zkDL 原生 Fr/G1 commitment/opening kernel 参与 relation-level prover；verification 主要由 host 重放 transcript/opening 检查；仍不是完整 GPU GAT proof/verify。",
        "- 五个全量数据集 hidden layer0/head0 sparse/grouped relation 已通过 proof/verify。",
        "- 五个全量数据集 hidden layer0/head0 denominator/inverse/alpha 与 8/8 feature weighted aggregation 已通过 relation-level proof/verify。",
        "- 上述 evidence 仍只是 supporting evidence：softmax max/shift、ExpMap、所有层/头 weighted aggregation、output aggregation、LeakyReLU/ELU、rescale 或 final logits chain 仍未完整证明。",
        "- `formal_relations.cuh` 当前只有 Hadamard/Matmul/Linear/SparseRouting/GroupedSum/MultiColumnGroupedSum relation；没有 selection-matrix lookup/table commitment primitive、bit-decomposition range primitive、Bmax max/comparison relation 或 denominator nonzero/inverse soundness gadget。",
        "- ExpMap、LeakyReLU 与 ELU 仍 blocked：当前缺少 zkDL-native public table 或 table commitment、selection matrix one-hot/range 约束与 verifier-side table binding。",
        "- output aggregation 已做代码级可表达性检查：现有 Sparse/Grouped/MultiColumnGrouped/Hadamard/Linear 可表达 `alpha * message` 与按 destination grouped sum，但输出层还缺 output-head alpha 的 max/ExpMap/range 来源、输出 bias/rescale、head average 与 public logits binding 的同一 proof object。",
        "- all-layer/all-head runner 仍 blocked：hidden layer0/head0 trace 已接 relation proof；其余 head/layer 与 output attention head 尚缺完整 per-head trace manifest、relation scheduling 与 cross-relation opening aggregation。",
        "- real rescale chain 与 rounding sign/tie-breaking 仍 blocked：当前 manifest/checkpoint 没有逐 relation 公开 `rho/delta/rem_bits/tie_policy` 字段，不能用 host equality、residual range 或 debug comparison 伪造 rounding proof。",
        "- `results/cpu/latest.*` 与 `results/gpu/latest.*` 只写 official 四项；当前 CPU/GPU 均未 full verified，所以保持 NA。",
        "- `debug_level_b.*` 不作为 formal benchmark；旧 debug artifact 已清理。",
    ]
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "rounding_sparse_blocker_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_resumes(
    sparse: dict[str, dict[str, str]],
    softmax: dict[str, dict[str, str]],
    cpu: dict[str, dict[str, str]],
) -> None:
    for dataset in DATASETS:
        cpu_dir = RUNS / dataset / "cpu" / "warm"
        cpu_dir.mkdir(parents=True, exist_ok=True)
        item = cpu.get(dataset)
        cpu_lines = [
            "# zkDL-GAT CPU 继续点",
            "",
            f"- 数据集: `{dataset}`",
            "- official latest: `blocked + NA`。",
        ]
        if item:
            cpu_lines.extend(
                [
                    "- 当前状态: `partial_formal`，不是 Level A。",
                    "- CPU 参与范围: CPU-only host Fr/G1/Commitment/opening + CPU relation prover/verifier。",
                    "- 已验证 relation: `SparseSourceRoutingRelationProof`, `SparseDestinationRoutingRelationProof`, `GroupedSumRelationProof`。",
                    f"- proof artifact: `{item.get('proof_path', rel(cpu_partial_proof_path(dataset)))}`",
                    f"- evidence: `{rel(cpu_partial_evidence_path(dataset))}`",
                    f"- timing: commitment `{item.get('commitment_time_ms', 'NA')}` ms, prove `{item.get('prove_time_ms', 'NA')}` ms, verify `{item.get('verify_time_ms', 'NA')}` ms, proof_size `{item.get('proof_size_bytes', 'NA')}` bytes。",
                    "- blocker: CPU denominator/inverse/alpha、lookup/range/max、output aggregation、all-layer/all-head 与 rescale/rounding 仍未闭合。",
                ]
            )
        elif cpu_timeout_path(dataset).exists():
            cpu_lines.extend(
                [
                    "- 当前状态: `blocked_after_attempt_timeout`。",
                    "- CPU 参与范围: CPU-only host Fr/G1/Commitment/opening + CPU relation prover/verifier 已进入 full-dataset attempt。",
                    f"- timeout evidence: `{rel(cpu_timeout_path(dataset))}`",
                    "- blocker: host G1 opening replay 未在交互窗口内完成；需要 zkDL-native batched opening/verifier 或更快 CPU MSM/opening replay。",
                ]
            )
        else:
            cpu_lines.extend(
                [
                    "- 当前状态: `not_run_after_pubmed_timeout`。",
                    f"- evidence: `{rel(cpu_probe_path(dataset)) if cpu_probe_path(dataset).exists() else 'not_run'}`",
                    "- blocker: Pubmed 已暴露 CPU host opening replay timeout；较大数据集等待 batched opening/verifier 后运行。",
                ]
            )
        cpu_lines.extend(
            [
                f"- log: `{rel(cpu_probe_log_path(dataset)) if cpu_probe_log_path(dataset).exists() else 'not_run'}`",
                "- 下一最小有效动作: 先做 CPU batched opening/verifier，再接 CPU denominator/inverse/alpha 与 full GAT runner。",
            ]
        )
        (cpu_dir / "RESUME_POINT.md").write_text("\n".join(cpu_lines) + "\n", encoding="utf-8")

        gpu_dir = RUNS / dataset / "gpu" / "warm"
        gpu_dir.mkdir(parents=True, exist_ok=True)
        s = sparse.get(dataset, {})
        a = softmax.get(dataset, {})
        lines = [
            "# zkDL-GAT GPU 继续点",
            "",
            f"- 数据集: `{dataset}`",
            "- 当前状态: `partial_formal`，不是 Level A。",
            "- GPU 参与范围: CUDA forward trace + zkDL 原生 CUDA-backed Fr/G1 commitment/opening；verifier 主要为 host transcript/opening 重放。",
            "- official latest: `blocked + NA`。",
        ]
        if s:
            lines.extend(
                [
                    "",
                    "## sparse/grouped relation",
                    "",
                    "- relation: `SparseSourceRoutingRelationProof`, `SparseDestinationRoutingRelationProof`, `GroupedSumRelationProof`",
                    f"- 证明范围: hidden layer0/head0 full graph / full edge set，edges `{s.get('edge_count', 'NA')}`",
                    f"- 结果: `{s.get('source_sparse_routing', 'NA')}/{s.get('destination_sparse_routing', 'NA')}/{s.get('grouped_sum_by_dst', 'NA')}`",
                    f"- proof artifact: `{s.get('proof_path', '')}`",
                    f"- evidence: `{rel(evidence_path(dataset, 'sparse'))}`",
                ]
            )
        if a:
            lines.extend(
                [
                    "",
                    "## softmax/aggregation partial relation",
                    "",
                    "- relation: `denominator grouped sum`, `inverse Hadamard`, `alpha multiplication`, `multi-column weighted aggregation`",
                    f"- 证明范围: hidden layer0/head0 full graph / full edge set，feature columns `{a.get('verified_feature_columns', '0')}/{a.get('available_feature_columns', '0')}`",
                    f"- 结果: denom `{a.get('denominator_grouped_sum', 'NA')}`, inverse `{a.get('inverse', 'NA')}`, alpha `{a.get('alpha_from_inverse', 'NA')}`, weighted `{a.get('weighted_aggregation', 'NA')}`",
                    f"- proof artifact: `{a.get('proof_path', '')}`",
                    f"- evidence: `{rel(evidence_path(dataset, 'softmax'))}`",
                ]
            )
        lines.extend(
            [
                "",
                "## Level A blocker",
                "",
                "- softmax max/shift、ExpMap lookup/range、LeakyReLU/ELU lookup-range、output aggregation、real rescale chain 与 rounding sign/tie-breaking 仍未闭合。",
                "- 下一最小有效动作: 在 zkDL primitive 内实现 lookup/range/max/rescale 后，把 hidden/output 所有层头接入同一个完整 GAT proof object。",
            ]
        )
        (gpu_dir / "RESUME_POINT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        debug_lines = [f"dataset={dataset} device=gpu status=partial_formal official=blocked"]
        if s:
            debug_lines.append(
                "stage=formal_sparse_grouped "
                f"status=PASS nodes={s.get('node_count')} edges={s.get('edge_count')} "
                f"commitment_ms={s.get('commitment_time_ms')} prove_ms={s.get('prove_time_ms')} "
                f"verify_ms={s.get('verify_time_ms')} proof_size_bytes={s.get('proof_size_bytes')}"
            )
        if a:
            debug_lines.append(
                "stage=formal_softmax_aggregation "
                f"status={a.get('weighted_aggregation')} nodes={a.get('node_count')} edges={a.get('edge_count')} "
                f"features={a.get('verified_feature_columns')}/{a.get('available_feature_columns')} "
                f"commitment_ms={a.get('commitment_time_ms')} prove_ms={a.get('prove_time_ms')} "
                f"verify_ms={a.get('verify_time_ms')} proof_size_bytes={a.get('proof_size_bytes')}"
            )
        debug_lines.append("note=partial_formal_only_not_official_gpu_proof")
        (gpu_dir / "proof_debug.log").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")


def write_root_resume() -> None:
    lines = [
        "# zkDL-GAT 当前继续点",
        "",
        "- 当前状态: `not_Level_A`。",
        "- Gate 1/2/3 最新执行结果: `PASS`；日志在 `zkdl_gat_outputs/runs/selftest/*_latest.log`。",
        "- CPU official path: `blocked + NA`；CPU-only relation backend 已实现并在 Cora/Citeseer sparse/grouped full dataset 通过 proof/verify，但未达到完整 GAT proof。",
        "- CPU Pubmed: full-dataset proof/verify attempt 因 host G1 opening replay timeout 未完成；PPI/ogbn CPU 等待 batched opening/verifier 后运行。",
        "- GPU official path: `blocked + NA`，原因是当前只有 relation-level partial formal evidence，不是完整 GAT proof/verify。",
        "- GPU partial evidence: 五个全量数据集 hidden layer0/head0 sparse/grouped、denominator/inverse/alpha 与 8/8 feature weighted aggregation。",
        "- 结果路径: `zkdl_gat_outputs/results/cpu/` 与 `zkdl_gat_outputs/results/gpu/`；顶层 `results/latest.*` 只是聚合索引。",
        "- 下一最小有效动作: 实现 zkDL lookup/range/max primitive 与 batched opening/verifier；随后补 output aggregation、rescale/rounding 公开契约和所有层/头 proof object。",
    ]
    (OUT / "RESUME_POINT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def readme_body(
    sparse: dict[str, dict[str, str]],
    softmax: dict[str, dict[str, str]],
    cpu: dict[str, dict[str, str]],
    detailed: bool,
) -> str:
    sparse_lines = []
    for dataset in DATASETS:
        item = sparse.get(dataset)
        if item:
            sparse_lines.append(
                f"| {dataset} | {item['node_count']} | {item['edge_count']} | PASS | "
                f"{item.get('commitment_block_size', '1')} | {item['commitment_time_ms']} | {item['prove_time_ms']} | {item['verify_time_ms']} | {item['proof_size_bytes']} |"
            )
    softmax_lines = []
    for dataset in DATASETS:
        item = softmax.get(dataset)
        if item:
            softmax_lines.append(
                f"| {dataset} | {item['node_count']} | {item['edge_count']} | {item['verified_feature_columns']}/{item['available_feature_columns']} | "
                f"{item['denominator_grouped_sum']} | {item['inverse']} | {item['alpha_from_inverse']} | {item['alpha_times_denominator']} | {item['weighted_aggregation']} | "
                f"{item.get('commitment_block_size', '1')} | {item['commitment_time_ms']} | {item['prove_time_ms']} | {item['verify_time_ms']} | {item['proof_size_bytes']} |"
            )
    cpu_lines = []
    for dataset in DATASETS:
        item = cpu.get(dataset)
        if item:
            cpu_lines.append(
                f"| {dataset} | {item['node_count']} | {item['edge_count']} | partial_formal | "
                f"{item['commitment_time_ms']} | {item['prove_time_ms']} | {item['verify_time_ms']} | {item['proof_size_bytes']} | "
                "SparseSourceRouting;SparseDestinationRouting;GroupedSum |"
            )
        elif cpu_timeout_path(dataset).exists():
            cpu_lines.append(
                f"| {dataset} | 19717 | 108365 | blocked_after_attempt_timeout | NA | NA | NA | NA | host G1 opening replay timeout |"
            )
        elif dataset in ["ppi", "ogbn-arxiv"]:
            cpu_lines.append(
                f"| {dataset} | NA | NA | not_run_after_pubmed_timeout | NA | NA | NA | NA | deferred after Pubmed CPU timeout |"
            )

    title = "# zkDL-GAT baseline 对比实验目录" if not detailed else "# zkDL-GAT adapter 详细状态"
    lines = [
        title,
        "",
        "本目录是 zkDL baseline 的 GAT 适配实验目录，用于在同一批五个全量数据集、同一套数据/cache/checkpoint/config 输出口径下，对比 zkDL proof system 与主仓库方案。这里对比的是证明方案，不是把主仓库方案混入 zkDL。",
        "",
        "## 协议边界",
        "",
        "- 本适配严格以 `/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 为最高协议边界。",
        "- 本轮修改未改变该文件规定的 GAT 前向语义：隐藏层多头 attention + ELU，输出层仍是 GAT attention head，输出头先聚合再加偏置并做算术平均。",
        "- 本轮修改未混入主仓库 GAT-ZKML 证明方案；没有搬入 KZG proof schema、quotient polynomial、verifier equation、opening proof 规则、transcript challenge 语义或 proof object layout。",
        "- 未达到 Level A 时，`latest.csv/json` 保持 `blocked + NA`。",
        "- 所有 `partial_formal` 只作为 supporting evidence，不能写入 official 四项。",
        "",
        "## 当前目录结构",
        "",
        "- zkDL 原 baseline 核心：`bls12-381.cuh`、`fr-tensor.cuh`、`g1-tensor.cuh`、`commitment.cuh`、`proof.cuh`、`zkfc.cuh`、`zkmatmul.cuh`、`zkprod.cuh`、`zkrelu.cuh`。",
        "- GAT 适配入口：`gat_zkdl.cu`，只保留 CLI 解析和显式调度；默认旧 `--run-all` 已禁用，避免 mixed debug/official 结果。",
        "- shared/common 工具：`gat_modules/common.cuh`，包含 timing、路径、格式化和 CUDA buffer 辅助。",
        "- 数据与配置加载：`gat_modules/data_cache.cuh`，负责 official full dataset cache、特征、边和标签加载。",
        "- checkpoint / manifest / cache 加载：`gat_modules/model_bundle.cuh`，负责 checkpoint bundle、manifest 与 GAT 参数解析。",
        "- public topology / edge / group context：`gat_modules/data_cache.cuh` 与 `gat_modules/formal_sparse_utils.cuh`，负责归一化图、`src/dst`、topology hash、group pointer 与 padded shape。",
        "- GAT forward trace 构造：`gat_modules/gpu_forward.cuh`，保留 CUDA forward kernel 与 trace 构造。",
        "- shared formal 工具：`gat_modules/formal_common.cuh`，负责 field 编码、commitment registry、proof size 与 proof_debug 精简。",
        "- CPU relation / gadget：`gat_modules/cpu_forward.cuh`、`gat_modules/cpu_formal_backend.cuh` 与 `gat_modules/run_dispatch.cuh`，提供 CPU trace、host Fr/G1/Commitment/opening、CPU relation prover/verifier 与 runner/export。",
        "- GPU relation / gadget：`gat_modules/formal_sparse_utils.cuh` 与 `gat_modules/formal_softmax_runners.cuh`，负责 sparse/grouped、denominator/inverse/alpha 与 multi-column weighted aggregation partial formal runner。",
        "- CPU/GPU benchmark timing：CPU probe 与 GPU partial formal 分别写入 `runs/<dataset>/cpu/warm/` 和 `runs/<dataset>/gpu/warm/`，不共享 timing row。",
        "- proof_size / artifact / evidence 管理：`gat_modules/formal_common.cuh`、`gat_modules/formal_linear_trace.cuh` 与 `update_zkdl_gat_results.py`；proof_size 只统计 proof 传输对象。",
        "- latest / summary / coverage / formal_partial_debug / debug_level_b 导出：`update_zkdl_gat_results.py` 是当前正式同源导出入口。",
        "- selftest：`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。",
        "- zkDL relation/gadget primitive：`formal_relations.cuh` 与 `formal_*_selftest.cu`。",
        "- 复现实验配置：`configs/zkdl_gat_full_datasets.cfg`。",
        "- 结果同源导出：`update_zkdl_gat_results.py`。",
        "- CPU official 结果：`zkdl_gat_outputs/results/cpu/`。",
        "- GPU official 结果：`zkdl_gat_outputs/results/gpu/`。",
        "- GPU partial evidence：`zkdl_gat_outputs/evidence/` 与 `zkdl_gat_outputs/runs/<dataset>/gpu/warm/`。",
        "",
        "## CPU/GPU 口径",
        "",
        "- CPU official path 当前存在独立结果目录，但 official 仍为 `blocked + NA`；本轮已实现 CPU-only relation backend，并在 Cora/Citeseer full dataset 上真实完成 sparse/grouped proof generation 与 verification。",
        "- CPU Pubmed full dataset 已尝试并进入 CPU proof/verify，但 host G1 opening replay 超过交互窗口未完成；PPI/ogbn-arxiv 因该性能 blocker 未继续盲跑。",
        "- GPU path 当前不是完整 GPU GAT proof。CUDA forward trace 真实参与 witness 生成，zkDL 原生 CUDA-backed Fr/G1 commitment/opening 真实参与 relation-level prover；verifier 主要是 host transcript/opening 重放。",
        "- CPU 结果不会复制 GPU 结果；GPU kernel timing 不写入 official proof/verify timing；GPU partial formal 只写入 `formal_partial_debug.*`。",
        "- CPU/GPU artifact 与 result 已分目录：`results/cpu/`、`results/gpu/`、`runs/<dataset>/cpu/warm/`、`runs/<dataset>/gpu/warm/`。",
        "",
        "## official latest 规则",
        "",
        "基础字段保持为 `dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes`。只有 full dataset + complete formal proof generation + complete formal verification 成功后才填四项指标；当前 CPU/GPU 均未达到，因此均为 `NA`。",
        "",
        "## 已审计并迁移的 proof-system-neutral 工程优化",
        "",
        "- 已迁移：模块化组织；一次加载 dataset/cache/checkpoint；复用 public topology、`src/dst`、topology hash、padded shape 与 hidden layer0/head0 trace；CPU/GPU result/artifact 分目录；relation-level batch runner；`commitment_block_size=64`；multi-column compressed relation；compact artifact；`latest/summary/coverage/formal_partial_debug/debug_level_b` 同源导出；proof_size 只统计 relation proof 传输对象，不包含 witness/tensor/debug dump。",
        "- 不能迁移：主仓库 KZG proof schema、quotient 约束、verifier equation、opening proof 规则、transcript challenge 语义、proof object layout，因为这些是主仓库 proof-system-specific 内容。",
        "- 理论上可迁移但尚未完成：zkDL 自己的 cross-relation batched opening/verifier、lookup/range primitive、max/comparison primitive、rescale/range primitive、GPU buffer 复用、pinned memory、failure 后自动缩 batch、所有层/头 single-process full runner。",
        "",
        "## 当前 supported relation",
        "",
        "`commitment/evaluation binding`、`transcript binding`、CPU-only host Fr/G1/Commitment relation backend、`HadamardRelationProof`、`MatmulRelationProof`、`LinearRelationProof`、`SparseSourceRoutingRelationProof`、`SparseDestinationRoutingRelationProof`、一维 `GroupedSumRelationProof` 与 `MultiColumnGroupedSumRelationProof`。这些是 primitive 或 relation-level 支持，不等于完整 GAT proof。",
        "",
        "## 当前 partial_formal relation",
        "",
        "CPU：Cora/Citeseer full dataset hidden layer0/head0 sparse routing 与 grouped sum 已通过 CPU-only relation proof/verify；Pubmed CPU full-dataset attempt 因 host opening replay timeout 未完成。GPU：五个全量数据集 hidden layer0/head0 已验证 sparse routing、grouped sum、denominator grouped sum、inverse Hadamard、inverse destination routing、alpha multiplication、alpha_times_denominator、multi-column sparse routing、multi-column alpha Hadamard 与 8/8 feature multi-column grouped sum weighted aggregation。projection/dot、hidden concat、output bias add、output-head average、public logits binding、fixed-to-rounded decomposition 与 residual bounded range 只能作为 supporting evidence。",
        "",
        "## 当前 blocked relation",
        "",
        "`target-node softmax group`、`softmax group max/shift`、`ExpMap lookup/range`、`LeakyReLU/ELU lookup-range`、所有层/头 weighted aggregation、`output aggregation`、`output rescale`、real rescale chain 与 rounding sign/tie-breaking 仍 blocked。当前没有把 sparse GAT 改成 dense attention，没有跳过 output attention head，没有用 host equality/debug comparison/witness comparison 冒充 verifier。",
        "",
        "## sparse/grouped evidence",
        "",
        "| dataset | nodes | edges | status | commitment_block_size | commitment_ms | prove_ms | verify_ms | proof_size_bytes |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        *sparse_lines,
        "",
        "## CPU-only sparse/grouped evidence",
        "",
        "| dataset | nodes | edges | status | commitment_ms | prove_ms | verify_ms | proof_size_bytes | relation scope |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        *cpu_lines,
        "",
        "## denominator / inverse / alpha / weighted aggregation evidence",
        "",
        "| dataset | nodes | edges | weighted columns | denom | inverse | alpha_from_inverse | alpha_times_denom | weighted aggregation | commitment_block_size | commitment_ms | prove_ms | verify_ms | proof_size_bytes |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *softmax_lines,
        "",
        "## 复现命令",
        "",
        "```bash",
        "make all",
        "./formal_opening_selftest",
        "./formal_sumcheck_selftest",
        "./formal_relation_selftest",
        "./gat_zkdl --formal-cpu-backend-probe --dataset cora --output-root zkdl_gat_outputs",
        "./gat_zkdl --formal-cpu-backend-probe --dataset citeseer --output-root zkdl_gat_outputs",
        "# Pubmed CPU full-dataset attempt is expected to expose the current host opening replay bottleneck.",
        "./gat_zkdl --formal-cpu-backend-probe --dataset pubmed --output-root zkdl_gat_outputs",
        "for dataset in cora citeseer pubmed ppi ogbn-arxiv; do",
        "  ./gat_zkdl --formal-sparse-grouped --dataset \"$dataset\" --formal-commitment-block-size 64 --output-root zkdl_gat_outputs",
        "  ./gat_zkdl --formal-softmax-aggregation --dataset \"$dataset\" --formal-commitment-block-size 64 --output-root zkdl_gat_outputs",
        "done",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "`--formal-cpu-backend-probe` 现在会运行 CPU-only relation-level backend；Cora/Citeseer 已产出 CPU proof/verify，Pubmed 记录 timeout failure evidence。它仍不是 CPU official full GAT proof，不能填 official 四项。",
        "",
        "## 当前是否 Level A",
        "",
        "不是 Level A。最具体 blocker 是缺 zkDL-native lookup/range/max/rescale primitive、CPU/GPU batched opening/verifier、all-layer/all-head runner、output aggregation、final logits binding 与完整 proof object；CPU Pubmed 还暴露 host G1 opening replay timeout。",
        "",
        "## 下一步最小有效动作",
        "",
        "先在 zkDL proof primitive 内实现选择矩阵 lookup/table/range 与 max/comparison gadget，并把 ExpMap、LeakyReLU、ELU 接入 verifier；并行实现 zkDL-native batched opening/verifier。随后补 output aggregation、rescale/rounding 公开契约和所有层/头 full runner。",
    ]
    return "\n".join(lines) + "\n"


def write_readmes(
    sparse: dict[str, dict[str, str]],
    softmax: dict[str, dict[str, str]],
    cpu: dict[str, dict[str, str]],
) -> None:
    (ROOT / "README.md").write_text(readme_body(sparse, softmax, cpu, detailed=False), encoding="utf-8")
    (OUT / "README_ZKDL_GAT_ADAPTER.md").write_text(readme_body(sparse, softmax, cpu, detailed=True), encoding="utf-8")


def main() -> None:
    if ROOT != Path("/home/pzh/project/comparison/baselines/zkdl"):
        raise RuntimeError("update_zkdl_gat_results.py 只能在 zkDL baseline 目录内写结果")
    RESULTS.mkdir(parents=True, exist_ok=True)
    migrate_legacy_gpu_runs()
    sparse, softmax = normalize_evidence()
    cpu = normalize_cpu_evidence()
    normalize_run_logs()
    write_blocker_evidence()
    write_latest_and_summary()
    write_debug_level_b()
    write_formal_partial(sparse, softmax, cpu)
    write_coverage(sparse, softmax, cpu)
    write_dataset_resumes(sparse, softmax, cpu)
    write_root_resume()
    write_readmes(sparse, softmax, cpu)


if __name__ == "__main__":
    main()
