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

from ogbn_arxiv_data import (
    build_sorted_edge_index,
    ensure_cache,
    load_raw_arrays,
    preprocess_input_features,
)
from train_planetoid_gat import (
    PlanetoidGAT,
    add_training_cli,
    build_optimizer,
    checkpoint_metadata,
    cfg_bool,
    cfg_float,
    cfg_int,
    quantized_input_features,
    parse_cfg,
    require_canonical_model_config,
    set_seed,
    _override,
)
from quantization_scale_chain import (
    OGBN_ARXIV_INTERNAL_NAME,
    canonical_quantization_scale_chain,
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
    if dataset != OGBN_ARXIV_INTERNAL_NAME:
        raise ValueError("train_ogbn_arxiv_gat only supports dataset=ogbn_arxiv")

    project_root = cfg_path.parent.parent.resolve()
    data_root = (project_root / cfg.get("data_root", "data")).resolve()
    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            project_root
            / cfg.get(
                "checkpoint_bundle",
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

    cache_root = ensure_cache(project_root, data_root)

    seed = cfg_int(cfg, "seed", 11)
    hidden_profile = require_canonical_model_config(cfg, dataset)
    k_out = 2
    chain = canonical_quantization_scale_chain(
        cfg["quant_cfg_id"], hidden_profile, k_out
    )
    seed = int(_override(args.seed, seed))
    epochs = int(_override(args.epochs, cfg_int(cfg, "epochs", 500)))
    patience = int(_override(args.patience, cfg_int(cfg, "patience", 100)))
    lr = float(_override(args.learning_rate, cfg_float(cfg, "learning_rate", 0.005)))
    weight_decay = float(
        _override(args.weight_decay, cfg_float(cfg, "weight_decay", 0.0005))
    )
    dropout = float(_override(args.dropout, cfg_float(cfg, "dropout", 0.5)))
    use_cuda = bool(_override(args.use_cuda, True))
    if (epochs, patience, lr, weight_decay, dropout) != (
        500,
        100,
        0.005,
        0.0005,
        0.5,
    ) or seed != 11:
        raise ValueError(
            "ogbn_arxiv formal training requires epochs=500, patience=100, "
            "lr=0.005, weight_decay=0.0005, dropout=0.5, seed=11"
        )
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not use_cuda:
        raise RuntimeError(
            "formal training requires CUDA; refusing to create a CPU checkpoint"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training is required but torch.cuda.is_available() is false"
        )
    symmetrize = cfg_bool(cfg, "symmetrize_edges", False)

    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(seed)

    arrays = load_raw_arrays(data_root)
    cached_features = np.load(cache_root / "features.npy", allow_pickle=False)
    if (
        cached_features.dtype != np.dtype(np.float32)
        or cached_features.shape != arrays.features.shape
        or not np.array_equal(cached_features, arrays.features)
    ):
        raise ValueError(
            "ogbn_arxiv cache features.npy is not the exact released float32 producer output"
        )
    # OGBN-Arxiv is registered as raw_float32: the released float32
    # feature matrix enters the canonical ties-away quantizer directly.  Do
    # not apply the Planetoid/PPI row-normalization contract here.
    input_preprocess_id = str(cfg["input_preprocess_id"])
    raw_features = preprocess_input_features(
        cached_features,
        dataset=OGBN_ARXIV_INTERNAL_NAME,
        input_preprocess_id=input_preprocess_id,
    )
    quantized_features, input_quantization = quantized_input_features(
        raw_features,
        chain,
        range_bits=cfg_int(cfg, "range_bits", 21),
    )
    edge_src, edge_dst = build_sorted_edge_index(
        arrays.edge_src,
        arrays.edge_dst,
        raw_features.shape[0],
        symmetrize=symmetrize,
        add_self_loops=True,
    )
    features = torch.tensor(quantized_features, dtype=torch.float32, device=device)
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
        quantization=chain,
    ).to(device)
    optimizer = build_optimizer(model, lr, weight_decay)
    metadata = checkpoint_metadata(
        dataset=OGBN_ARXIV_INTERNAL_NAME,
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

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(loss.item()),
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_metrics)
        print(
            f"[ogbn_arxiv] epoch={epoch:04d} train_loss={loss.item():.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}",
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
        final_test_acc = masked_accuracy(best_logits, labels, test_idx)
        final_test_loss = float(masked_loss(best_logits, labels, test_idx).item())

    np.save(output_dir / "qat_best_logits.npy", best_logits.detach().cpu().numpy().astype(np.float32, copy=False))
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset": OGBN_ARXIV_INTERNAL_NAME,
                "qat_float32_non_formal": True,
                "device": str(device),
                "wall_time_sec": time.perf_counter() - start,
                "best_metrics": best_metrics,
                "final_test_metrics": {
                    "test_loss": final_test_loss,
                    "test_acc": final_test_acc,
                },
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
