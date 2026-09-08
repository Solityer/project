#!/usr/bin/env python3
"""Train the canonical multi-label PPI GAT using the shared GAT family.

PPI keeps one 121-dimensional multi-hot target per node.  The trainer uses
``BCEWithLogitsLoss`` and selects the checkpoint from validation loss only;
the test mask is never used during optimization or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from train_planetoid_gat import (
    PlanetoidGAT,
    _override,
    add_training_cli,
    build_optimizer,
    checkpoint_metadata,
    cfg_float,
    cfg_int,
    parse_cfg,
    require_canonical_model_config,
    set_seed,
    quantized_input_features,
)
from quantization_scale_chain import (
    canonical_input_preprocess,
    canonical_input_preprocess_id,
    canonical_quantization_scale_chain,
)


@dataclass
class PPIData:
    features: torch.Tensor
    labels_multi: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    num_nodes: int
    num_features: int
    num_classes: int
    input_quantization_diagnostic: Dict[str, object]


def _ensure_cache(
    project_root: pathlib.Path,
    data_root: pathlib.Path,
    cache_root: pathlib.Path,
) -> pathlib.Path:
    required = (
        "meta.cfg",
        "features.npy",
        "labels_multi.npy",
        "labels_multi.txt",
        "train_mask.npy",
        "val_mask.npy",
        "test_mask.npy",
        "node_ptr.txt",
        "edges.txt",
    )
    if (cache_root / "labels.txt").exists():
        raise ValueError(
            "PPI cache contains unsupported single-label labels.txt; regenerate the cache"
        )
    if not all((cache_root / name).exists() for name in required):
        script = pathlib.Path(__file__).with_name("prepare_ppi.py")
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--data-root",
                str(data_root / "ppi"),
                "--output-root",
                str(cache_root),
            ],
            cwd=project_root,
            check=True,
        )
        if (cache_root / "labels.txt").exists():
            raise ValueError(
                "PPI cache contains unsupported single-label labels.txt after preparation"
            )
    return cache_root


def _load_size_vector(path: pathlib.Path) -> np.ndarray:
    values = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if len(values) < 2 or values[0] != 0:
        raise ValueError(f"invalid PPI node_ptr: {path}")
    vector = np.asarray(values, dtype=np.int64)
    if np.any(vector[1:] < vector[:-1]):
        raise ValueError("PPI node_ptr must be non-decreasing")
    return vector


def _read_cache_metadata(path: pathlib.Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"PPI cache metadata is missing: {path}")
    metadata: Dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: malformed metadata line")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in metadata:
            raise ValueError(f"{path}:{line_number}: duplicate or empty metadata key")
        metadata[key] = value
    return metadata


def _load_sorted_edges(
    path: pathlib.Path,
    node_ptr: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, dtype=np.int64)
    rows = np.asarray(raw, dtype=np.int64).reshape(-1, 3)
    graph_count = int(node_ptr.size - 1)
    seen: set[tuple[int, int, int]] = set()
    kept: list[tuple[int, int, int, int]] = []
    for graph_id, src, dst in rows.tolist():
        if graph_id < 0 or graph_id >= graph_count:
            raise ValueError(f"PPI edge graph id out of range: {graph_id}")
        begin, end = int(node_ptr[graph_id]), int(node_ptr[graph_id + 1])
        if src < begin or dst < begin or src >= end or dst >= end:
            raise ValueError("PPI edge endpoint is outside its graph boundary")
        key = (int(graph_id), int(src), int(dst))
        if key not in seen:
            seen.add(key)
            kept.append((int(graph_id), int(src), int(dst), len(kept)))

    graph_for_node = np.repeat(
        np.arange(graph_count, dtype=np.int64), np.diff(node_ptr)
    )
    if graph_for_node.size != num_nodes:
        raise ValueError("PPI node_ptr does not cover all feature rows")
    for node, graph_id in enumerate(graph_for_node.tolist()):
        key = (int(graph_id), node, node)
        if key not in seen:
            seen.add(key)
            kept.append((int(graph_id), node, node, len(kept)))

    if not kept:
        raise ValueError("PPI edge list is empty after self-loop insertion")
    edge_array = np.asarray(kept, dtype=np.int64)
    order = np.lexsort((edge_array[:, 3], edge_array[:, 2], edge_array[:, 0]))
    sorted_edges = edge_array[order]
    src = sorted_edges[:, 1]
    dst = sorted_edges[:, 2]
    if np.any(src < 0) or np.any(src >= num_nodes) or np.any(dst < 0) or np.any(dst >= num_nodes):
        raise ValueError("PPI normalized edge endpoint is out of range")
    return src.astype(np.int64, copy=False), dst.astype(np.int64, copy=False)


def load_ppi_dataset(
    project_root: pathlib.Path,
    data_root: pathlib.Path,
    cache_root: pathlib.Path,
    device: torch.device,
    *,
    chain=None,
    range_bits: int | None = None,
) -> PPIData:
    cache = _ensure_cache(project_root, data_root, cache_root)
    metadata = _read_cache_metadata(cache / "meta.cfg")
    expected_preprocess_id = canonical_input_preprocess_id("ppi")
    if metadata.get("name") != "ppi" or metadata.get("input_preprocess_id") != expected_preprocess_id:
        raise ValueError(
            "PPI cache is missing the canonical row_normalize producer identity"
        )
    features = np.asarray(np.load(cache / "features.npy"), dtype=np.float32)
    labels_multi = np.asarray(np.load(cache / "labels_multi.npy"), dtype=np.float32)
    if (
        features.ndim != 2
        or labels_multi.ndim != 2
        or features.shape[0] != labels_multi.shape[0]
    ):
        raise ValueError("PPI features and labels_multi must be aligned matrices")
    if labels_multi.shape[1] != 121:
        raise ValueError(
            f"PPI multi-label target must have 121 columns, got {labels_multi.shape[1]}"
        )
    if not np.all((labels_multi == 0.0) | (labels_multi == 1.0)):
        raise ValueError("PPI multi-label target must contain only 0/1 values")

    masks: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        mask = np.asarray(np.load(cache / f"{split}_mask.npy"), dtype=bool)
        if mask.shape != (features.shape[0],):
            raise ValueError(f"PPI {split} mask shape mismatch")
        masks[split] = mask
    if np.any(masks["train"] & masks["val"]) or np.any(masks["train"] & masks["test"]) or np.any(masks["val"] & masks["test"]):
        raise ValueError("PPI train/val/test masks must be disjoint")
    if not np.all(masks["train"] | masks["val"] | masks["test"]):
        raise ValueError("PPI train/val/test masks must cover every node")

    node_ptr = _load_size_vector(cache / "node_ptr.txt")
    if int(node_ptr[-1]) != features.shape[0]:
        raise ValueError("PPI node_ptr and feature row count disagree")
    edge_src, edge_dst = _load_sorted_edges(
        cache / "edges.txt", node_ptr, int(features.shape[0])
    )
    chain = chain or canonical_quantization_scale_chain()
    normalized_features = canonical_input_preprocess(
        features, "ppi", expected_preprocess_id
    )
    quantized_features, input_diagnostic = quantized_input_features(
        normalized_features,
        chain,
        range_bits=range_bits,
    )
    return PPIData(
        features=torch.tensor(quantized_features, dtype=torch.float32, device=device),
        labels_multi=torch.tensor(labels_multi, dtype=torch.float32, device=device),
        train_mask=torch.tensor(masks["train"], dtype=torch.bool, device=device),
        val_mask=torch.tensor(masks["val"], dtype=torch.bool, device=device),
        test_mask=torch.tensor(masks["test"], dtype=torch.bool, device=device),
        edge_src=torch.tensor(edge_src, dtype=torch.long, device=device),
        edge_dst=torch.tensor(edge_dst, dtype=torch.long, device=device),
        num_nodes=int(features.shape[0]),
        num_features=int(features.shape[1]),
        num_classes=int(labels_multi.shape[1]),
        input_quantization_diagnostic=input_diagnostic,
    )


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        raise ValueError("PPI split mask must contain at least one node")
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def main() -> None:
    parser = argparse.ArgumentParser()
    add_training_cli(parser)
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = parse_cfg(cfg_path)
    if cfg.get("dataset") != "ppi":
        raise ValueError("train_ppi_gat only supports dataset=ppi")
    project_root = cfg_path.parent.parent.resolve()
    data_root = (project_root / cfg.get("data_root", "data")).resolve()
    cache_root = (project_root / cfg.get("cache_root", "data/cache") / "ppi").resolve()
    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (project_root / cfg.get("checkpoint_bundle", "artifacts/checkpoints/ppi_gat")).resolve()
    )
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {checkpoint_path}; pass --overwrite for the canonical retraining"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_profile = require_canonical_model_config(cfg, "ppi")
    k_out = 2
    chain = canonical_quantization_scale_chain(
        cfg["quant_cfg_id"], hidden_profile, k_out
    )

    epochs = int(_override(args.epochs, cfg_int(cfg, "epochs", 500)))
    patience = int(_override(args.patience, cfg_int(cfg, "patience", 100)))
    learning_rate = float(_override(args.learning_rate, cfg_float(cfg, "learning_rate", 0.005)))
    weight_decay = float(_override(args.weight_decay, cfg_float(cfg, "weight_decay", 0.0)))
    dropout = float(_override(args.dropout, cfg_float(cfg, "dropout", 0.0)))
    seed = int(_override(args.seed, cfg_int(cfg, "seed", 11)))
    use_cuda = bool(_override(args.use_cuda, True))
    if (epochs, patience, learning_rate, weight_decay, dropout) != (
        500,
        100,
        0.005,
        0.0,
        0.0,
    ) or seed != 11:
        raise ValueError(
            "PPI formal training requires epochs=500, patience=100, lr=0.005, "
            "weight_decay=0, dropout=0, seed=11"
        )
    if not use_cuda:
        raise RuntimeError("formal training requires CUDA; refusing to create a CPU checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training is required but torch.cuda.is_available() is false")

    device = torch.device("cuda")
    set_seed(seed)
    data = load_ppi_dataset(
        project_root,
        data_root,
        cache_root,
        device,
        chain=chain,
        range_bits=cfg_int(cfg, "range_bits", 14),
    )
    input_quantization = data.input_quantization_diagnostic
    if data.num_features != 50:
        raise ValueError(f"PPI formal feature width must be 50, got {data.num_features}")
    model = PlanetoidGAT(
        data.num_features,
        num_classes=data.num_classes,
        hidden_profile=hidden_profile,
        k_out=k_out,
        quantization=chain,
    ).to(device)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    metadata = checkpoint_metadata(
        dataset="ppi",
        in_dim=data.num_features,
        hidden_profile=hidden_profile,
        k_out=k_out,
        num_classes=data.num_classes,
        seed=seed,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_param_id="ppi_gat_seed11",
        extra={
            "task_type": "inductive_multi_graph_node_classification",
            "report_unit": "graph",
            "batching_rule": "multi_graph_batch",
            "subgraph_rule": "whole_graph",
            "self_loop_rule": "per_node",
            "edge_sort_rule": "edge_gid_then_dst_stable",
            "label_format": "multi_hot",
            "loss": "BCEWithLogitsLoss",
            "selection_metric": "validation_loss",
            "split_format": "node_masks",
        },
    )

    best_state = None
    best_metrics = None
    best_val_loss = math.inf
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(
            data.features,
            data.edge_src,
            data.edge_dst,
            dropout=dropout,
            training=True,
            return_trace=False,
        )
        loss = masked_bce_with_logits(logits, data.labels_multi, data.train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(
                data.features,
                data.edge_src,
                data.edge_dst,
                dropout=0.0,
                training=False,
                return_trace=False,
            )
            validation_loss = float(
                masked_bce_with_logits(eval_logits, data.labels_multi, data.val_mask).item()
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.item()),
                "val_loss": validation_loss,
            }
        )
        print(
            f"[ppi] epoch={epoch:04d} train_loss={loss.item():.6f} "
            f"val_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": validation_loss,
            }
            torch.save(
                {"bundle": metadata, "state_dict": best_state},
                checkpoint_path,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_logits, _ = model(
            data.features,
            data.edge_src,
            data.edge_dst,
            dropout=0.0,
            training=False,
            return_trace=False,
        )
    np.save(
        output_dir / "qat_best_logits.npy",
        best_logits.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset": "ppi",
                "qat_float32_non_formal": True,
                "device": str(device),
                "wall_time_sec": time.perf_counter() - start,
                "best_metrics": best_metrics,
                "input_quantization_diagnostic": input_quantization,
                "history_tail": history[-10:],
                "checkpoint_metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
