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
    add_training_cli,
    build_optimizer,
    checkpoint_metadata,
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
    add_training_cli(parser)
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = parse_cfg(cfg_path)
    dataset = cfg["dataset"]
    if dataset != "ogbn_arxiv":
        raise ValueError("train_ogbn_arxiv_gat only supports dataset=ogbn_arxiv")

    project_root = cfg_path.parent.parent.resolve()
    data_root = (project_root / cfg.get("data_root", "data")).resolve()
    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            project_root
            / cfg.get(
                "checkpoint_dir",
                "artifacts/checkpoints/ogbn_arxiv_gat",
            )
        ).resolve()
    )
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {checkpoint_path}; pass a fresh --output-dir "
            "or explicitly use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_cache(project_root, data_root)

    seed = cfg_int(cfg, "seed", 11)
    if "hidden_profile" not in cfg:
        raise ValueError("training config must expose hidden_profile")
    hidden_profile = parse_hidden_profile_expr(cfg["hidden_profile"])
    k_out = cfg_int(cfg, "K_out", 1)
    if k_out <= 0:
        raise ValueError("K_out must be positive")
    seed = args.seed if args.seed is not None else seed
    epochs = args.epochs if args.epochs is not None else cfg_int(cfg, "epochs", 1)
    patience = (
        args.patience
        if args.patience is not None
        else cfg_int(cfg, "patience", 1)
    )
    lr = (
        args.learning_rate
        if args.learning_rate is not None
        else cfg_float(cfg, "learning_rate", 0.005)
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else cfg_float(cfg, "weight_decay", 0.0005)
    )
    dropout = (
        args.dropout
        if args.dropout is not None
        else cfg_float(cfg, "dropout", 0.6)
    )
    use_cuda = (
        args.use_cuda
        if args.use_cuda is not None
        else cfg_bool(cfg, "use_cuda", False)
    )
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but torch.cuda.is_available() is false"
        )
    symmetrize = cfg_bool(cfg, "symmetrize_edges", False)

    device = torch.device("cuda" if use_cuda else "cpu")
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

    num_classes = int(labels.max().item()) + 1
    model = PlanetoidGAT(
        int(features.shape[1]),
        num_classes=num_classes,
        hidden_profile=hidden_profile,
        k_out=k_out,
    ).to(device)
    optimizer = build_optimizer(model, lr, weight_decay)
    metadata = checkpoint_metadata(
        dataset="ogbn-arxiv",
        in_dim=int(features.shape[1]),
        hidden_profile=hidden_profile,
        k_out=k_out,
        num_classes=num_classes,
        seed=seed,
        dropout=dropout,
        learning_rate=lr,
        weight_decay=weight_decay,
        model_param_id=f"ogbn_arxiv_seed{seed}",
        extra={
            "batching_rule": "whole_graph_single",
            "subgraph_rule": "whole_graph",
            "self_loop_rule": "per_node",
            "edge_sort_rule": "edge_gid_then_dst_stable",
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
                    "bundle": metadata,
                    "state_dict": best_state,
                },
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
                "checkpoint_metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
