#!/usr/bin/env python3
"""从 full proof/verifier manifest 同源导出 zkDL-GAT official 结果。"""

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


def rel(path: Path) -> str:
    """把 zkDL 内部路径写成 README/JSON 可读的相对路径。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json_object(path: Path) -> dict[str, object]:
    """读取 manifest JSON；格式错误时返回空对象并让 official gate 拒绝。"""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: object) -> None:
    """写结果 JSON；只写 zkDL 输出目录内的派生文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
    """按固定列顺序写 latest CSV，保持 CPU/GPU 分表口径一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def run_dir(device: str, dataset: str) -> Path:
    """返回单个 full-chain run 目录；CPU/GPU 目录互不覆盖。"""
    return RUNS / dataset / device / "warm"


def full_proof_manifest(device: str, dataset: str) -> Path:
    """full proof manifest 是 official 四项的唯一 proof 侧来源。"""
    return run_dir(device, dataset) / "full_proof_manifest.json"


def full_verifier_manifest(device: str, dataset: str) -> Path:
    """full verifier manifest 是 verified 与 verifier_reads_witness gate 的唯一来源。"""
    return run_dir(device, dataset) / "full_verifier_manifest.json"


def full_public_metadata(device: str, dataset: str) -> Path:
    """public metadata/VK/table/quant context 由 full-chain runner 写入 formal_artifacts。"""
    return run_dir(device, dataset) / "formal_artifacts" / "zkdl_full_chain_public_metadata.json"


def path_from_manifest(value: object) -> Path:
    """解析 manifest 内的 proof_object_path；相对路径按 zkDL 根目录解释。"""
    text = str(value or "")
    if not text:
        return Path()
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def official_row_from_manifest(device: str, dataset: str) -> dict[str, object] | None:
    """严格 official gate：只接受 full dataset、proof/verifier 双 manifest 均通过的结果。

    该函数不会读取 partial/debug artifact，也不会从 relation-level evidence 回填 official。
    """
    proof_manifest = read_json_object(full_proof_manifest(device, dataset))
    verifier_manifest = read_json_object(full_verifier_manifest(device, dataset))
    if not proof_manifest or not verifier_manifest:
        return None
    if str(proof_manifest.get("benchmark_mode", "")) != "warm":
        return None
    if str(proof_manifest.get("protocol_id", "")) != "GAT-ZKML-zkDL":
        return None
    if str(verifier_manifest.get("protocol_id", "")) != "GAT-ZKML-zkDL":
        return None
    if str(proof_manifest.get("proof_scope", "")) != "zkdl_gat_full":
        return None
    if str(verifier_manifest.get("proof_scope", "")) != "zkdl_gat_full":
        return None
    if str(proof_manifest.get("dataset", "")) != dataset or str(verifier_manifest.get("dataset", "")) != dataset:
        return None
    if str(proof_manifest.get("device", "")) != device or str(verifier_manifest.get("device", "")) != device:
        return None
    if str(proof_manifest.get("is_full_dataset", "")).lower() != "true":
        return None
    if str(proof_manifest.get("relation_proofs_generated", "")).lower() != "true":
        return None
    if str(proof_manifest.get("proof_generated", "")).lower() != "true":
        return None
    if str(proof_manifest.get("verified", "")).lower() != "true":
        return None
    if str(verifier_manifest.get("proof_generated", "")).lower() != "true":
        return None
    if str(verifier_manifest.get("verified", "")).lower() != "true":
        return None
    if str(verifier_manifest.get("verifier_reads_witness", "")).lower() != "false":
        return None
    if str(proof_manifest.get("witness_included_in_proof", "")).lower() != "false":
        return None
    if str(verifier_manifest.get("witness_included_in_proof", "")).lower() != "false":
        return None
    proof_missing = proof_manifest.get("missing_relations", [])
    verifier_missing = verifier_manifest.get("missing_relations", [])
    if not isinstance(proof_missing, list) or proof_missing:
        return None
    if not isinstance(verifier_missing, list) or verifier_missing:
        return None
    proof_object_path = path_from_manifest(proof_manifest.get("proof_object_path"))
    if not proof_object_path.exists():
        return None
    if not full_public_metadata(device, dataset).exists():
        return None
    return {
        "dataset": dataset,
        "status": "ok",
        "blocker": "",
        "benchmark_mode": "warm",
        "commitment_time_ms": proof_manifest.get("commitment_time_ms", "NA"),
        "prove_time_ms": proof_manifest.get("prove_time_ms", "NA"),
        "verify_time_ms": verifier_manifest.get("verify_time_ms", "NA"),
        "proof_size_bytes": proof_manifest.get("proof_size_bytes", "NA"),
    }


def official_rows(device: str) -> list[dict[str, object]]:
    """按固定五数据集顺序生成单 device official 表。"""
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        official = official_row_from_manifest(device, dataset)
        if official is not None:
            rows.append(official)
        else:
            rows.append({
                "dataset": dataset,
                "status": "blocked",
                "blocker": "full proof/verifier manifest 未满足 official gate",
                "benchmark_mode": "warm",
                "commitment_time_ms": "NA",
                "prove_time_ms": "NA",
                "verify_time_ms": "NA",
                "proof_size_bytes": "NA",
            })
    return rows


def status_label(status: object) -> str:
    """README 分表显示用中文状态；CSV/JSON 保持机器可读 status。"""
    return "已完成" if str(status) == "ok" else "阻塞"


def write_latest_and_summary() -> list[dict[str, object]]:
    """导出 aggregate latest、CPU/GPU latest 和三份 summary.md。"""
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
            f"device: `{device}`；本表只由 full proof/verifier manifest 导出。",
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
        "五个全量数据集 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv` 均保留单独状态。",
        "",
        "| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        summary.append(
            f"| {row['device']} | {row['dataset']} | {row['status']} | {row['benchmark_mode']} | "
            f"{row['commitment_time_ms']} | {row['prove_time_ms']} | {row['verify_time_ms']} | {row['proof_size_bytes']} |"
        )
    (RESULTS / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return aggregate


def official_status_table() -> list[str]:
    """生成 README 中的 official 结果表。"""
    lines = [
        "| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for device in DEVICES:
        for row in official_rows(device):
            lines.append(
                f"| {device} | {row['dataset']} | {row['status']} | {row['benchmark_mode']} | "
                f"{row['commitment_time_ms']} | {row['prove_time_ms']} | {row['verify_time_ms']} | "
                f"{row['proof_size_bytes']} |"
            )
    return lines


def write_readme() -> None:
    """按当前 official 状态重写正式目录 README，不保留开发记录或旧 blocker。"""
    readme = [
        "# zkDL-GAT baseline",
        "",
        "本目录是 zkDL baseline 的 GAT 正式对比实验目录。实验对象固定为五个全量数据集：`cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`。CPU 与 GPU 是两条独立 official 路径，结果分别写入 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，再由 exporter 从 full proof/verifier manifest 导出 latest 与 summary。",
        "",
        "## 实验目的",
        "",
        "本目录在同一批主仓库 data/cache/checkpoint/config 口径下，用 zkDL 自身 proof system 适配 GAT 前向证明，并导出可与主仓库结果对齐的 official 四项指标：`commitment_time_ms`、`prove_time_ms`、`verify_time_ms`、`proof_size_bytes`。",
        "",
        "## 协议边界",
        "",
        "`/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 是最高协议边界。本目录只使用 zkDL baseline 自身的 Hyrax-style tensor commitment、opening、transcript、sumcheck 与 relation proof；证明范围固定为 GAT 前向计算证明。",
        "",
        "## 与主仓库 proof scheme 的边界",
        "",
        "本目录不迁移主仓库 KZG proof schema、quotient polynomial、verifier equation、opening proof rule、transcript challenge 语义和 proof object layout。主仓库只提供数据、checkpoint、config 与结果口径参照。",
        "",
        "## 目录结构",
        "",
        "- zkDL 核心：`bls12-381.cuh`、`fr-tensor.cuh`、`g1-tensor.cuh`、`commitment.cuh`、`proof.cuh`、`formal_relations.cuh`。",
        "- GAT 加载与前向：`gat_modules/data_cache.cuh`、`gat_modules/model_bundle.cuh`、`gat_modules/cpu_forward.cuh`、`gat_modules/gpu_forward.cuh`。",
        "- 协议公共上下文：`gat_modules/protocol_context.cuh` 生成 public metadata、VK_static/VK_model、table context 与 quant/rounding contract。",
        "- relation proof：`projection_runner.cuh`、`rescale_rounding_runner.cuh`、`attention_dot_runner.cuh`、`range_comparison_runner.cuh`、`expmap_lookup_runner.cuh`、`nonlinear_lookup_runner.cuh`、`softmax_lookup_runner.cuh`、`output_attention_runner.cuh`、`hidden_concat_runner.cuh`、`full_output_binding_runner.cuh`。",
        "- full-chain runner：`gat_modules/full_chain_runner.cuh` 和 `gat_modules/cpu_full_chain_runner.cuh` 遍历 hidden/output 全部 head 并写 full proof/verifier manifest。",
        "- proof object 与 gate：`gat_modules/full_proof_object.cuh` 写 proof object 摘要、`full_proof_manifest.json`、`full_verifier_manifest.json`；`gat_modules/full_verifier.cuh` 执行 clean verifier gate。",
        "- CLI 与调度：`gat_zkdl.cu` 是薄 CLI，`gat_modules/run_dispatch.cuh` 只分发 `--formal-full-chain`。",
        "- 全量调度：`run_zkdl_gat_experiments.py` 可按 dataset/device 调度 full-chain 运行，并在末尾调用 exporter。",
        "- 结果导出：`update_zkdl_gat_results.py` 只从 full proof/verifier manifest 导出 official latest/summary，并同步重写本 README。",
        "- 构建与自测：`Makefile`、`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。",
        "",
        "## CPU/GPU official 路径",
        "",
        "`--formal-full-chain --device cpu` 使用 CPU commitment/opening/relation verifier；`--formal-full-chain --device gpu` 使用 CUDA trace 与 zkDL-native Fr/G1 commitment/opening。两条路径都写入 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，CPU/GPU 结果互不覆盖。",
        "",
        "## full-chain 产物",
        "",
        "每个 dataset/device 成功运行后必须保留以下文件：",
        "",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/full_proof_manifest.json`：proof 侧 manifest，记录 `protocol_id`、`proof_scope`、`is_full_dataset`、`proof_generated`、`verified`、`missing_relations`、proof object 路径、relation proof 摘要，以及 `commitment_time_ms`、`prove_time_ms`、`proof_size_bytes`。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/full_verifier_manifest.json`：verifier 侧 manifest，记录 clean verifier 是否接受，并显式保留 `verifier_reads_witness=false`、`witness_included_in_proof=false`、`verify_time_ms`。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_proof_object.txt`：zkDL full-chain proof object 摘要和子证明摘要，不包含完整 witness 或完整 tensor。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_chain_public_metadata.json`：public metadata、VK_static/VK_model、table context、topology、quant/rounding contract。",
        "",
        "## official gate",
        "",
        "exporter 的 official gate 在 `update_zkdl_gat_results.py` 中实现，只接受同时满足以下条件的行：",
        "",
        "- proof manifest 与 verifier manifest 均存在且 JSON 格式有效。",
        "- `protocol_id=GAT-ZKML-zkDL`。",
        "- `proof_scope=zkdl_gat_full`。",
        "- `benchmark_mode=warm`。",
        "- manifest 中的 `dataset`、`device` 与当前导出的 dataset/device 一致。",
        "- `is_full_dataset=true`。",
        "- `relation_proofs_generated=true`。",
        "- proof manifest 与 verifier manifest 均满足 `proof_generated=true`、`verified=true`。",
        "- proof manifest 与 verifier manifest 均满足 `missing_relations=[]`。",
        "- `verifier_reads_witness=false`。",
        "- proof manifest 与 verifier manifest 均满足 `witness_included_in_proof=false`。",
        "- proof object 文件与 public metadata 文件真实存在。",
        "",
        "partial、debug、relation-level evidence、旧失败日志和临时诊断产物不能进入 official；exporter 不从这些文件回填 official 四项指标。",
        "",
        "## 结果文件",
        "",
        "- `zkdl_gat_outputs/results/latest.csv|json`：CPU/GPU 合并 official 表。",
        "- `zkdl_gat_outputs/results/summary.md`：CPU/GPU 合并摘要。",
        "- `zkdl_gat_outputs/results/cpu/latest.csv|json` 与 `summary.md`：CPU official 分表。",
        "- `zkdl_gat_outputs/results/gpu/latest.csv|json` 与 `summary.md`：GPU official 分表。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_proof_object.txt`：proof metadata 与子证明摘要，不包含完整 witness/tensor。",
        "- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_chain_public_metadata.json`：public metadata、VK_static/VK_model、table context 与 quant/rounding contract。",
        "",
        "## 完整复现实验流程",
        "",
        "完整复现需要覆盖五个全量数据集的 CPU/GPU 两条路径，共十个 dataset/device 组合。全量实验耗时较长；如果只需要检查当前目录仍可构建、可运行、可导出 official，可先运行单个 Cora CPU/GPU 示例，但这不代表完整五数据集复现。",
        "",
        "推荐的完整流程如下：",
        "",
        "```bash",
        "make all",
        "./formal_opening_selftest",
        "./formal_sumcheck_selftest",
        "./formal_relation_selftest",
        "python3 run_zkdl_gat_experiments.py --dataset all --device all --output-root zkdl_gat_outputs --commitment-block-size 64 --skip-selftests",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "`run_zkdl_gat_experiments.py` 的等价显式命令清单如下。每条命令都会运行一个完整 dataset/device 的 full-chain proof，不会调用 partial/debug runner。",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device cpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset citeseer --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset pubmed --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset ppi --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device cpu --dataset ogbn-arxiv --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset citeseer --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset pubmed --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset ppi --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "./gat_zkdl --formal-full-chain --device gpu --dataset ogbn-arxiv --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "## 单数据集单设备复现",
        "",
        "单个 dataset/device 的模板命令如下：",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device <cpu|gpu> --dataset <cora|citeseer|pubmed|ppi|ogbn-arxiv> --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "单独运行某个 dataset/device 只验证该组合的复现能力。完整实验必须覆盖 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv` 的 CPU/GPU 十个组合。",
        "",
        "## Cora 示例",
        "",
        "GPU 路径：",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device gpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "```",
        "",
        "CPU 路径：",
        "",
        "```bash",
        "./gat_zkdl --formal-full-chain --device cpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
        "```",
        "",
        "参数含义如下：",
        "",
        "- `--formal-full-chain`：运行完整 zkDL-GAT proof chain，而不是 partial relation 或 debug 入口。",
        "- `--device`：选择 `cpu` 或 `gpu` official 路径；CPU/GPU 产物分别写入独立目录。",
        "- `--dataset`：选择全量数据集名称，可用值为 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`。",
        "- `--output-root`：指定输出根目录；CLI 要求该目录留在当前 zkDL baseline 目录内。",
        "- `--formal-commitment-block-size`：指定 commitment/opening 相关分块参数，用于控制 full-chain proof 过程中的资源开销；当前 official 复现使用 `64`。",
        "",
        "## 结果导出",
        "",
        "运行或重跑任意 dataset/device 后，用同源 exporter 重新导出 official 表：",
        "",
        "```bash",
        "python3 update_zkdl_gat_results.py",
        "```",
        "",
        "exporter 会重新生成 `zkdl_gat_outputs/results/latest.csv`、`zkdl_gat_outputs/results/latest.json`、`zkdl_gat_outputs/results/summary.md`，以及 `results/cpu/`、`results/gpu/` 下的分表。它只从 full proof/verifier manifest 读取数据；不读取 partial/debug 产物。",
        "",
        "## manifest 检查",
        "",
        "可以用下面的命令检查 Cora CPU/GPU 是否满足 official gate 的核心字段：",
        "",
        "```bash",
        "python3 - <<'PY'",
        "import json",
        "from pathlib import Path",
        "",
        "root = Path('zkdl_gat_outputs/runs')",
        "for device in ['gpu', 'cpu']:",
        "    run = root / 'cora' / device / 'warm'",
        "    proof = json.loads((run / 'full_proof_manifest.json').read_text())",
        "    verifier = json.loads((run / 'full_verifier_manifest.json').read_text())",
        "    proof_object = Path(proof['proof_object_path'])",
        "    public_metadata = run / 'formal_artifacts' / 'zkdl_full_chain_public_metadata.json'",
        "    print(device, {",
        "        'proof_generated': proof['proof_generated'] and verifier['proof_generated'],",
        "        'verified': proof['verified'] and verifier['verified'],",
        "        'missing_relations': proof['missing_relations'] + verifier['missing_relations'],",
        "        'verifier_reads_witness': verifier['verifier_reads_witness'],",
        "        'witness_included_in_proof': proof['witness_included_in_proof'] or verifier['witness_included_in_proof'],",
        "        'proof_object_exists': proof_object.exists(),",
        "        'public_metadata_exists': public_metadata.exists(),",
        "    })",
        "PY",
        "```",
        "",
        "## 指标口径",
        "",
        "- `commitment_time_ms` 来自 full proof manifest 中所有 relation commitment 时间之和。",
        "- `prove_time_ms` 来自 full proof manifest 中 proof generation 时间之和。",
        "- `verify_time_ms` 来自 full verifier manifest；该值表示 clean verifier 重放 full-chain proof 的时间。",
        "- `proof_size_bytes` 统计 zkDL full proof object 摘要及子证明大小；不包含完整 witness、完整 tensor、debug evidence 或 public metadata。",
        "- CUDA/CPU forward timing 只服务于运行日志与诊断，不写入 official 四项。",
        "- Cora 或其他单数据集重跑后 timing 可能自然变化；official 表只能由真实 full-chain 运行后的 manifest 通过 exporter 生成，不能手写。",
        "",
        "## official 结果表",
        "",
        *official_status_table(),
        "",
        "## selftest / Gate",
        "",
        "Gate 1 为 commitment/opening 自测，Gate 2 为 sumcheck 自测，Gate 3 为 relation proof/verifier 自测；日志位于 `zkdl_gat_outputs/runs/selftest/`。这些 Gate 不替代 full-chain official gate，只证明基础 zkDL 后端可用。",
    ]
    (ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def cleanup_stale_outputs() -> None:
    """删除旧 exporter/partial/debug/scope 产物，避免读者误认为它们是 official。"""
    stale_files = [
        ROOT / "README_ZKDL_GAT_ADAPTER.md",
        OUT / "README_ZKDL_GAT_ADAPTER.md",
        OUT / "RESUME_POINT.md",
    ]
    for name in ["formal_partial_debug", "debug_level_b", "coverage_matrix"]:
        for suffix in [".csv", ".json", ".md"]:
            stale_files.append(RESULTS / f"{name}{suffix}")
            stale_files.append(RESULTS / "cpu" / f"{name}{suffix}")
            stale_files.append(RESULTS / "gpu" / f"{name}{suffix}")
    for path in stale_files:
        if path.exists():
            path.unlink()

    evidence_dir = OUT / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    pycache_dir = ROOT / "__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)

    stale_run_names = {
        "RESUME_POINT.md",
        "proof_debug.log",
        "cpu_formal_chain.log",
        "formal_sparse_grouped_latest.log",
        "formal_softmax_aggregation_latest.log",
    }
    stale_artifact_names = {
        "zkdl_cpu_relation_partial.proof.txt",
        "zkdl_sparse_grouped_relations.proof.txt",
        "zkdl_softmax_aggregation_relations.proof.txt",
        "zkdl_output_binding_relations.proof.txt",
        "zkdl_full_chain_scope.status.txt",
    }
    if RUNS.exists():
        for path in RUNS.rglob("*"):
            if path.is_file() and path.name in stale_run_names:
                path.unlink()
            elif path.is_file() and path.name in stale_artifact_names:
                path.unlink()
            elif path.is_dir() and path.name == "debug_artifacts":
                shutil.rmtree(path)

    for directory in sorted(OUT.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def main() -> None:
    """清理旧派生产物后重新导出 official latest/summary/README。"""
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cleanup_stale_outputs()
    write_latest_and_summary()
    write_readme()


if __name__ == "__main__":
    main()
