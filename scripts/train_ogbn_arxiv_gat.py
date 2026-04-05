#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Dict, List

import numpy as np
import torch

from ogbn_arxiv_utils import build_sorted_edge_index, ensure_cache, load_raw_arrays, row_normalize_dense
from train_planetoid_gat import (
    PlanetoidGAT,
    build_optimizer,
    cfg_bool,
    cfg_float,
    cfg_int,
    parse_cfg,
    parse_hidden_profile_expr,
    set_seed,
)


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> float:
    predictions = logits[indices].argmax(dim=-1)
    return float((predictions == labels[indices]).float().mean().item())


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits[indices], labels[indices])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = parse_cfg(cfg_path)
    dataset = cfg["dataset"]
    if dataset != "ogbn_arxiv":
        raise ValueError("train_ogbn_arxiv_gat only supports dataset=ogbn_arxiv")

    project_root = cfg_path.parent.parent.resolve()
    data_root = (project_root / cfg.get("data_root", "data")).resolve()
    output_dir = (project_root / cfg.get("checkpoint_dir", "artifacts/checkpoints/ogbn_arxiv_gat")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_cache(project_root, data_root)

    seed = cfg_int(cfg, "seed", 11)
    if "hidden_profile" not in cfg:
        raise ValueError("training config must expose hidden_profile")
    hidden_profile = parse_hidden_profile_expr(cfg["hidden_profile"])
    if len(hidden_profile) != 1:
        raise ValueError("train_ogbn_arxiv_gat currently supports one hidden layer only")
    hidden_heads = int(hidden_profile[0]["head_count"])
    hidden_dim = int(hidden_profile[0]["head_dim"])
    k_out = cfg_int(cfg, "K_out", 1)
    if k_out != 1:
        raise ValueError("train_ogbn_arxiv_gat currently supports K_out=1 only")
    epochs = cfg_int(cfg, "epochs", 1)
    patience = cfg_int(cfg, "patience", 1)
    lr = cfg_float(cfg, "learning_rate", 0.005)
    weight_decay = cfg_float(cfg, "weight_decay", 0.0005)
    dropout = cfg_float(cfg, "dropout", 0.6)
    use_cuda = cfg_bool(cfg, "use_cuda", False)
    symmetrize = cfg_bool(cfg, "symmetrize_edges", False)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    set_seed(seed)

    arrays = load_raw_arrays(data_root)
    normalized_features = row_normalize_dense(arrays.features)
    edge_src, edge_dst = build_sorted_edge_index(
        arrays.edge_src,
        arrays.edge_dst,
        normalized_features.shape[0],
        symmetrize=symmetrize,
        add_self_loops=True,
    )
    features = torch.tensor(normalized_features, dtype=torch.float32, device=device)
    labels = torch.tensor(arrays.labels, dtype=torch.long, device=device)
    train_idx = torch.tensor(arrays.split_idx["train"], dtype=torch.long, device=device)
    valid_idx = torch.tensor(arrays.split_idx["valid"], dtype=torch.long, device=device)
    test_idx = torch.tensor(arrays.split_idx["test"], dtype=torch.long, device=device)
    edge_src_tensor = torch.tensor(edge_src, dtype=torch.long, device=device)
    edge_dst_tensor = torch.tensor(edge_dst, dtype=torch.long, device=device)

    model = PlanetoidGAT(features.shape[1], hidden_dim, int(labels.max().item()) + 1, hidden_heads=hidden_heads).to(device)
    optimizer = build_optimizer(model, lr, weight_decay)

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
            features,
            edge_src_tensor,
            edge_dst_tensor,
            dropout=dropout,
            training=True,
            return_trace=False,
        )
        loss = masked_loss(logits, labels, train_idx)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(
                features,
                edge_src_tensor,
                edge_dst_tensor,
                dropout=0.0,
                training=False,
                return_trace=False,
            )
            train_acc = masked_accuracy(eval_logits, labels, train_idx)
            val_loss = float(masked_loss(eval_logits, labels, valid_idx).item())
            val_acc = masked_accuracy(eval_logits, labels, valid_idx)
            test_acc = masked_accuracy(eval_logits, labels, test_idx)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
        )
        print(
            f"[ogbn-arxiv] epoch={epoch:04d} train_loss={loss.item():.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
            torch.save(
                {
                    "bundle": {
                        "dataset": "ogbn-arxiv",
                        "L": 2,
                        "d_in_profile": [int(features.shape[1])],
                        "hidden_profile": hidden_profile,
                        "K_out": k_out,
                        "C": int(labels.max().item()) + 1,
                        "num_features": int(features.shape[1]),
                        "num_classes": int(labels.max().item()) + 1,
                        "seed": seed,
                        "dropout": dropout,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "task_type": "transductive_node_classification",
                        "report_unit": "node",
                        "batching_rule": "whole_graph_single",
                        "subgraph_rule": "whole_graph",
                        "self_loop_rule": "per_node",
                        "edge_sort_rule": "edge_gid_then_dst_stable",
                        "model_arch_id": f"gat_family_L2_hid{hidden_heads}x{hidden_dim}_kout1_c{int(labels.max().item()) + 1}",
                        "model_param_id": f"ogbn_arxiv_seed{seed}",
                        "quant_cfg_id": "fp32_torch_checkpoint",
                        "static_table_id": "tables:lrelu+elu+exp+range",
                        "degree_bound_id": "auto",
                        "output_average_rule": "per_head_bias_then_arithmetic_mean",
                    },
                    "state_dict": best_state,
                },
                output_dir / "best_model.pt",
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
            features,
            edge_src_tensor,
            edge_dst_tensor,
            dropout=0.0,
            training=False,
            return_trace=False,
        )

    np.save(output_dir / "best_logits.npy", best_logits.detach().cpu().numpy().astype(np.float32, copy=False))
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset": "ogbn-arxiv",
                "device": str(device),
                "wall_time_sec": time.perf_counter() - start,
                "best_metrics": best_metrics,
                "history_tail": history[-10:],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
