#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from planetoid_utils import (
    build_sorted_edge_index,
    labels_to_int,
    load_planetoid_dataset_with_masks,
    row_normalize_features,
)


def hidden_seq_name(head_index: int) -> str:
    if head_index == 0:
        return "conv1d/kernel"
    return f"conv1d_{head_index * 3}/kernel"


def hidden_dst_name(head_index: int) -> str:
    index = 1 if head_index == 0 else head_index * 3 + 1
    return f"conv1d_{index}"


def hidden_src_name(head_index: int) -> str:
    index = 2 if head_index == 0 else head_index * 3 + 2
    return f"conv1d_{index}"


def parse_cfg(path: pathlib.Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def cfg_int(cfg: Dict[str, str], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def cfg_float(cfg: Dict[str, str], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def cfg_bool(cfg: Dict[str, str], key: str, default: bool) -> bool:
    value = cfg.get(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetTensors:
    features: torch.Tensor
    labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    num_nodes: int
    num_features: int
    num_classes: int


def load_dataset(data_root: pathlib.Path, dataset: str, device: torch.device) -> DatasetTensors:
    sparse_features, labels, adjacency, train_mask, val_mask, test_mask = load_planetoid_dataset_with_masks(
        data_root,
        dataset,
    )
    features = row_normalize_features(sparse_features)
    label_ids = labels_to_int(labels)
    edge_src_np, edge_dst_np = build_sorted_edge_index(adjacency, features.shape[0], add_self_loops=True)
    return DatasetTensors(
        features=torch.tensor(features, dtype=torch.float32, device=device),
        labels=torch.tensor(label_ids, dtype=torch.long, device=device),
        train_mask=torch.tensor(train_mask, dtype=torch.bool, device=device),
        val_mask=torch.tensor(val_mask, dtype=torch.bool, device=device),
        test_mask=torch.tensor(test_mask, dtype=torch.bool, device=device),
        edge_src=torch.tensor(edge_src_np, dtype=torch.long, device=device),
        edge_dst=torch.tensor(edge_dst_np, dtype=torch.long, device=device),
        num_nodes=int(features.shape[0]),
        num_features=int(features.shape[1]),
        num_classes=int(labels.shape[1]),
    )


class AttentionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.seq = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=True)
        self.attn_src = nn.Linear(out_dim, 1, bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
        *,
        dropout: float,
        training: bool,
        apply_activation: bool,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
        working_inputs = F.dropout(inputs, p=dropout, training=training) if dropout > 0.0 else inputs
        h_prime = self.seq(working_inputs)
        e_dst = self.attn_dst(h_prime).squeeze(-1)
        e_src = self.attn_src(h_prime).squeeze(-1)
        scores = F.leaky_relu(e_dst[edge_dst] + e_src[edge_src], negative_slope=0.2)

        max_per_dst = torch.full((num_nodes,), -torch.inf, device=inputs.device, dtype=inputs.dtype)
        max_per_dst.scatter_reduce_(0, edge_dst, scores, reduce="amax", include_self=True)

        shifted = scores - max_per_dst[edge_dst]
        exp_scores = torch.exp(shifted)
        sum_per_dst = torch.zeros((num_nodes,), device=inputs.device, dtype=inputs.dtype)
        sum_per_dst.index_add_(0, edge_dst, exp_scores)
        inv = sum_per_dst.clamp_min(1e-12).reciprocal()
        alpha = exp_scores * inv[edge_dst]
        if dropout > 0.0:
            alpha = F.dropout(alpha, p=dropout, training=training)

        msg_inputs = F.dropout(h_prime, p=dropout, training=training) if dropout > 0.0 else h_prime
        out = torch.zeros((num_nodes, h_prime.shape[1]), device=inputs.device, dtype=inputs.dtype)
        out.index_add_(0, edge_dst, msg_inputs[edge_src] * alpha.unsqueeze(-1))
        activated = F.elu(out) if apply_activation else out

        if not return_trace:
            return activated, None

        trace = {
            "H_prime": h_prime.detach().cpu(),
            "E_src": e_src.detach().cpu(),
            "E_dst": e_dst.detach().cpu(),
            "S": (e_dst[edge_dst] + e_src[edge_src]).detach().cpu(),
            "Z": scores.detach().cpu(),
            "M": max_per_dst.detach().cpu(),
            "Delta": (max_per_dst[edge_dst] - scores).detach().cpu(),
            "U": exp_scores.detach().cpu(),
            "Sum": sum_per_dst.detach().cpu(),
            "inv": inv.detach().cpu(),
            "alpha": alpha.detach().cpu(),
            "H_agg_pre_bias": out.detach().cpu(),
            "H_agg": activated.detach().cpu(),
        }
        return activated, trace


class PlanetoidGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, hidden_heads: int = 8) -> None:
        super().__init__()
        self.hidden_heads = nn.ModuleList([AttentionHead(in_dim, hidden_dim) for _ in range(hidden_heads)])
        self.output_head = AttentionHead(hidden_dim * hidden_heads, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        *,
        dropout: float,
        training: bool,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object] | None]:
        hidden_outputs: List[torch.Tensor] = []
        hidden_traces: List[Dict[str, torch.Tensor]] = []
        for head in self.hidden_heads:
            out, trace = head.forward(
                features,
                edge_src,
                edge_dst,
                features.shape[0],
                dropout=dropout,
                training=training,
                apply_activation=True,
                return_trace=return_trace,
            )
            hidden_outputs.append(out)
            if trace is not None:
                hidden_traces.append(trace)
        hidden_concat = torch.cat(hidden_outputs, dim=-1)
        logits, output_trace = self.output_head.forward(
            hidden_concat,
            edge_src,
            edge_dst,
            features.shape[0],
            dropout=dropout,
            training=training,
            apply_activation=False,
            return_trace=return_trace,
        )
        if not return_trace:
            return logits, None
        return logits, {
            "hidden_concat": hidden_concat.detach().cpu(),
            "hidden_heads": hidden_traces,
            "output": output_trace,
        }


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.Adam(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = torch.argmax(logits[mask], dim=-1)
    return float((preds == labels[mask]).float().mean().item())


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[mask], labels[mask])


def load_checkpoint(path: pathlib.Path, map_location: torch.device) -> Dict[str, object]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_bundle_state(bundle_dir: pathlib.Path) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    archive = np.load(bundle_dir / "tensors.npz")
    tensor_index = manifest["tensor_index"]

    bundle_meta = {
        "hidden_dim": int(archive[tensor_index["BiasAdd/biases"]["archive_key"]].shape[0]),
        "hidden_heads": int(manifest["hidden_head_count"]),
        "num_classes": int(archive[tensor_index["BiasAdd_8/biases"]["archive_key"]].shape[0]),
    }

    state: Dict[str, torch.Tensor] = {}
    for head_index in range(bundle_meta["hidden_heads"]):
        prefix = f"hidden_heads.{head_index}"
        seq = archive[tensor_index[hidden_seq_name(head_index)]["archive_key"]]
        dst_name = hidden_dst_name(head_index)
        src_name = hidden_src_name(head_index)
        state[f"{prefix}.seq.weight"] = torch.from_numpy(seq.reshape(seq.shape[1], seq.shape[2]).T.astype(np.float32, copy=False))
        state[f"{prefix}.attn_dst.weight"] = torch.from_numpy(
            archive[tensor_index[f"{dst_name}/kernel"]["archive_key"]].reshape(1, -1).astype(np.float32, copy=False)
        )
        state[f"{prefix}.attn_dst.bias"] = torch.from_numpy(
            archive[tensor_index[f"{dst_name}/bias"]["archive_key"]].reshape(1).astype(np.float32, copy=False)
        )
        state[f"{prefix}.attn_src.weight"] = torch.from_numpy(
            archive[tensor_index[f"{src_name}/kernel"]["archive_key"]].reshape(1, -1).astype(np.float32, copy=False)
        )
        state[f"{prefix}.attn_src.bias"] = torch.from_numpy(
            archive[tensor_index[f"{src_name}/bias"]["archive_key"]].reshape(1).astype(np.float32, copy=False)
        )

    output_seq = archive[tensor_index["conv1d_24/kernel"]["archive_key"]]
    state["output_head.seq.weight"] = torch.from_numpy(
        output_seq.reshape(output_seq.shape[1], output_seq.shape[2]).T.astype(np.float32, copy=False)
    )
    state["output_head.attn_dst.weight"] = torch.from_numpy(
        archive[tensor_index["conv1d_25/kernel"]["archive_key"]].reshape(1, -1).astype(np.float32, copy=False)
    )
    state["output_head.attn_dst.bias"] = torch.from_numpy(
        archive[tensor_index["conv1d_25/bias"]["archive_key"]].reshape(1).astype(np.float32, copy=False)
    )
    state["output_head.attn_src.weight"] = torch.from_numpy(
        archive[tensor_index["conv1d_26/kernel"]["archive_key"]].reshape(1, -1).astype(np.float32, copy=False)
    )
    state["output_head.attn_src.bias"] = torch.from_numpy(
        archive[tensor_index["conv1d_26/bias"]["archive_key"]].reshape(1).astype(np.float32, copy=False)
    )
    return bundle_meta, state


def run_trace_from_checkpoint(
    checkpoint_path: pathlib.Path,
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    bundle = checkpoint["bundle"]
    tensors = load_dataset(data_root, dataset, device)
    model = PlanetoidGAT(
        tensors.num_features,
        int(bundle["hidden_dim"]),
        tensors.num_classes,
        hidden_heads=int(bundle["hidden_heads"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits, trace = model(
            tensors.features,
            tensors.edge_src,
            tensors.edge_dst,
            dropout=0.0,
            training=False,
            return_trace=True,
        )
    assert trace is not None
    return {
        "logits": logits.detach().cpu().numpy(),
        "hidden_concat": trace["hidden_concat"].numpy(),
        "hidden_heads": [
            {key: value.numpy() for key, value in head.items()}
            for head in trace["hidden_heads"]
        ],
        "output": {key: value.numpy() for key, value in trace["output"].items()},
    }


def run_trace_from_bundle(
    bundle_dir: pathlib.Path,
    data_root: pathlib.Path,
    dataset: str,
    device: torch.device,
) -> Dict[str, object]:
    bundle_meta, state = load_bundle_state(bundle_dir)
    tensors = load_dataset(data_root, dataset, device)
    model = PlanetoidGAT(
        tensors.num_features,
        int(bundle_meta["hidden_dim"]),
        tensors.num_classes,
        hidden_heads=int(bundle_meta["hidden_heads"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits, trace = model(
            tensors.features,
            tensors.edge_src,
            tensors.edge_dst,
            dropout=0.0,
            training=False,
            return_trace=True,
        )
    assert trace is not None
    return {
        "logits": logits.detach().cpu().numpy(),
        "hidden_concat": trace["hidden_concat"].numpy(),
        "hidden_heads": [
            {key: value.numpy() for key, value in head.items()}
            for head in trace["hidden_heads"]
        ],
        "output": {key: value.numpy() for key, value in trace["output"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = parse_cfg(cfg_path)
    dataset = cfg["dataset"]
    project_root = cfg_path.parent.parent
    data_root = (project_root / cfg.get("data_root", "data")).resolve()
    output_dir = (project_root / cfg.get("checkpoint_dir", f"artifacts/checkpoints/{dataset}_gat")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg_int(cfg, "seed", 11)
    hidden_dim = cfg_int(cfg, "hidden_dim", 8)
    hidden_heads = cfg_int(cfg, "hidden_heads", 8)
    epochs = cfg_int(cfg, "epochs", 500)
    patience = cfg_int(cfg, "patience", 100)
    lr = cfg_float(cfg, "learning_rate", 0.005)
    weight_decay = cfg_float(cfg, "weight_decay", 0.0005)
    dropout = cfg_float(cfg, "dropout", 0.6)
    use_cuda = cfg_bool(cfg, "use_cuda", False)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    set_seed(seed)
    tensors = load_dataset(data_root, dataset, device)
    model = PlanetoidGAT(tensors.num_features, hidden_dim, tensors.num_classes, hidden_heads=hidden_heads).to(device)
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
            tensors.features,
            tensors.edge_src,
            tensors.edge_dst,
            dropout=dropout,
            training=True,
            return_trace=False,
        )
        loss = masked_loss(logits, tensors.labels, tensors.train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(
                tensors.features,
                tensors.edge_src,
                tensors.edge_dst,
                dropout=0.0,
                training=False,
                return_trace=False,
            )
            train_acc = masked_accuracy(eval_logits, tensors.labels, tensors.train_mask)
            val_loss = float(masked_loss(eval_logits, tensors.labels, tensors.val_mask).item())
            val_acc = masked_accuracy(eval_logits, tensors.labels, tensors.val_mask)
            test_acc = masked_accuracy(eval_logits, tensors.labels, tensors.test_mask)

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
            f"[{dataset}] epoch={epoch:04d} train_loss={loss.item():.6f} train_acc={train_acc:.4f} "
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
                        "dataset": dataset,
                        "hidden_dim": hidden_dim,
                        "hidden_heads": hidden_heads,
                        "num_features": tensors.num_features,
                        "num_classes": tensors.num_classes,
                        "seed": seed,
                        "dropout": dropout,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "formal_semantics": "single_layer_hidden8_output1_no_bias_output_attention",
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
            tensors.features,
            tensors.edge_src,
            tensors.edge_dst,
            dropout=0.0,
            training=False,
            return_trace=False,
        )

    np.save(output_dir / "best_logits.npy", best_logits.detach().cpu().numpy().astype(np.float32, copy=False))
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
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
