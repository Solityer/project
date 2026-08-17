#!/usr/bin/env python3
"""zkDL-GAT 五数据集 official full-chain 调度入口。

该脚本只调度 Gate 自测、`--formal-full-chain` 和同源 exporter；不会生成或导出
partial/debug 结果，也不会在 zkDL 目录外写入文件。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]


def ensure_inside_zkdl(path: Path) -> Path:
    """确认 output-root 留在当前 zkDL 目录内，避免实验产物写到主仓库其他位置。"""
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise SystemExit(f"output-root 必须位于 zkDL 目录内: {resolved}") from error
    return resolved


def selected_items(value: str, allowed: list[str], label: str) -> list[str]:
    """解析 all 或逗号列表，并阻止未知 dataset/device 混入 official 调度。"""
    if value == "all":
        return allowed
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise SystemExit(f"未知 {label}: {','.join(unknown)}")
    return items


def run_command(command: list[str], timeout_seconds: int | None) -> int:
    """执行一个子命令；失败只返回退出码，由上层决定是否继续其他数据集。"""
    print("+ " + " ".join(command), flush=True)
    try:
        completed = subprocess.run(command, cwd=ROOT, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        print("命令超时，调度继续后续 dataset/device。", flush=True)
        return 124
    return completed.returncode


def run_selftests(timeout_seconds: int | None) -> bool:
    """运行 Gate 1/2/3 自测；只检查后端，不写日志或 official artifact。"""
    ok = True
    for binary in ["formal_opening_selftest", "formal_sumcheck_selftest", "formal_relation_selftest"]:
        rc = run_command([str(BIN / binary)], timeout_seconds)
        ok = ok and rc == 0
    return ok


def run_dataset_device(
    dataset: str,
    device: str,
    output_root: Path,
    project_root: Path,
    commitment_block_size: int,
    timeout_seconds: int | None,
) -> bool:
    """运行单个 dataset/device 的完整 GAT proof，不触发非 official partial/debug runner。"""
    base = [
        str(BIN / "gat_zkdl"),
        "--formal-full-chain",
        "--device",
        device,
        "--dataset",
        dataset,
        "--project-root",
        str(project_root),
        "--output-root",
        str(output_root),
        "--formal-commitment-block-size",
        str(commitment_block_size),
    ]
    return run_command(base, timeout_seconds) == 0


def main() -> int:
    """解析调度参数；五数据集 official 结果最终由 exporter 从 full manifest 重新导出。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", help="all 或逗号分隔的数据集")
    parser.add_argument("--device", default="all", help="all、cpu、gpu 或逗号分隔")
    parser.add_argument("--output-root", default=str(ROOT / "zkdl_gat_outputs"))
    parser.add_argument("--project-root", default="/home/pzh/project")
    parser.add_argument("--commitment-block-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--skip-selftests", action="store_true")
    parser.add_argument("--selftests-only", action="store_true")
    args = parser.parse_args()

    output_root = ensure_inside_zkdl(Path(args.output_root))
    project_root = Path(args.project_root).resolve()
    timeout_seconds = args.timeout_seconds if args.timeout_seconds > 0 else None
    datasets = selected_items(args.dataset, DATASETS, "dataset")
    devices = selected_items(args.device, DEVICES, "device")

    all_ok = True
    if not args.skip_selftests:
        all_ok = run_selftests(timeout_seconds) and all_ok
    if args.selftests_only:
        return 0 if all_ok else 1

    for device in devices:
        for dataset in datasets:
            ok = run_dataset_device(
                dataset,
                device,
                output_root,
                project_root,
                args.commitment_block_size,
                timeout_seconds,
            )
            all_ok = all_ok and ok

    exporter_rc = run_command([sys.executable, str(ROOT / "tools" / "update_zkdl_gat_results.py")], timeout_seconds)
    all_ok = all_ok and exporter_rc == 0
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
