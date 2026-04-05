#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_hidden_profile_expr(raw: str) -> List[Dict[str, int]]:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    entries = [item.strip() for item in text.split(",") if item.strip()]
    profile: List[Dict[str, int]] = []
    for entry in entries:
        if "x" not in entry:
            raise ValueError(f"invalid hidden_profile entry: {entry}")
        head_count_text, head_dim_text = entry.split("x", 1)
        profile.append({"head_count": int(head_count_text), "head_dim": int(head_dim_text)})
    if not profile:
        raise ValueError("hidden_profile must be non-empty")
    return profile


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def row_normalize_features(features: np.ndarray) -> np.ndarray:
    rowsum = features.sum(axis=1, keepdims=True)
    inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum > 0
    inv[nonzero] = 1.0 / rowsum[nonzero]
    return features * inv


@dataclass
class GraphBatch:
    features: torch.Tensor
    labels: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    num_nodes: int
    graph_count: int


@dataclass
class GraphSample:
    features: np.ndarray
    labels: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    split: str


def load_ppi_samples(data_root: pathlib.Path) -> Tuple[List[GraphSample], int, int]:
    graph_payload = json.loads((data_root / "ppi-G.json").read_text(encoding="utf-8"))
    id_map_raw = json.loads((data_root / "ppi-id_map.json").read_text(encoding="utf-8"))
    class_map_raw = json.loads((data_root / "ppi-class_map.json").read_text(encoding="utf-8"))
    features_raw = np.load(data_root / "ppi-feats.npy").astype(np.float32, copy=False)

    id_map = {str(key): int(value) for key, value in id_map_raw.items()}
    num_nodes = len(id_map)
    adjacency: List[set[int]] = [set() for _ in range(num_nodes)]
    for link in graph_payload["links"]:
        src = int(id_map[str(link["source"])])
        dst = int(id_map[str(link["target"])])
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    node_attrs_by_new_id: Dict[int, Dict[str, object]] = {}
    for node in graph_payload["nodes"]:
        original = str(node["id"])
        if original not in id_map:
            continue
        node_attrs_by_new_id[id_map[original]] = node
    if len(node_attrs_by_new_id) != num_nodes:
        raise ValueError("ppi node metadata is incomplete")

    visited = [False] * num_nodes
    components: List[List[int]] = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        component: List[int] = []
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda comp: comp[0])

    num_classes = len(next(iter(class_map_raw.values())))
    samples: List[GraphSample] = []
    for component in components:
        local_index = {node: idx for idx, node in enumerate(component)}
        labels = np.zeros((len(component), num_classes), dtype=np.float32)
        split: str | None = None
        edges: set[Tuple[int, int]] = set()
        for node in component:
            attrs = node_attrs_by_new_id[node]
            node_split = "val" if bool(attrs.get("val", False)) else "test" if bool(attrs.get("test", False)) else "train"
            if split is None:
                split = node_split
            elif split != node_split:
                raise ValueError("PPI component split is inconsistent")
            vector = class_map_raw[str(attrs["id"])]
            labels[local_index[node], :] = np.asarray(vector, dtype=np.float32)
            edges.add((local_index[node], local_index[node]))
            for neighbor in adjacency[node]:
                if neighbor in local_index:
                    edges.add((local_index[node], local_index[neighbor]))
        if split is None:
            raise ValueError("empty PPI component")
        ordered_edges = sorted(edges, key=lambda pair: (pair[1], pair[0]))
        edge_src = np.asarray([pair[0] for pair in ordered_edges], dtype=np.int64)
        edge_dst = np.asarray([pair[1] for pair in ordered_edges], dtype=np.int64)
        features = row_normalize_features(features_raw[np.asarray(component, dtype=np.int64)])
        samples.append(
            GraphSample(
                features=features.astype(np.float32, copy=False),
                labels=labels,
                edge_src=edge_src,
                edge_dst=edge_dst,
                split=split,
            )
        )

    return samples, int(features_raw.shape[1]), int(num_classes)


def pack_graphs(samples: Sequence[GraphSample], device: torch.device) -> GraphBatch:
    feature_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    src_parts: List[torch.Tensor] = []
    dst_parts: List[torch.Tensor] = []
    node_offset = 0
    for sample in samples:
        feature_parts.append(torch.from_numpy(sample.features))
        label_parts.append(torch.from_numpy(sample.labels))
        src_parts.append(torch.from_numpy(sample.edge_src + node_offset))
        dst_parts.append(torch.from_numpy(sample.edge_dst + node_offset))
        node_offset += int(sample.features.shape[0])
    return GraphBatch(
        features=torch.cat(feature_parts, dim=0).to(device=device, dtype=torch.float32),
        labels=torch.cat(label_parts, dim=0).to(device=device, dtype=torch.float32),
        edge_src=torch.cat(src_parts, dim=0).to(device=device, dtype=torch.long),
        edge_dst=torch.cat(dst_parts, dim=0).to(device=device, dtype=torch.long),
        num_nodes=node_offset,
        graph_count=len(samples),
    )


class AttentionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.seq = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=True)
        self.attn_src = nn.Linear(out_dim, 1, bias=True)
        self.output_bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.float32))

    def forward(
        self,
        inputs: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        *,
        dropout: float,
        training: bool,
        apply_activation: bool,
    ) -> torch.Tensor:
        working_inputs = F.dropout(inputs, p=dropout, training=training) if dropout > 0.0 else inputs
        h_prime = self.seq(working_inputs)
        e_dst = self.attn_dst(h_prime).squeeze(-1)
        e_src = self.attn_src(h_prime).squeeze(-1)
        scores = F.leaky_relu(e_dst[edge_dst] + e_src[edge_src], negative_slope=0.2)

        max_per_dst = torch.full((inputs.shape[0],), -torch.inf, device=inputs.device, dtype=inputs.dtype)
        max_per_dst.scatter_reduce_(0, edge_dst, scores, reduce="amax", include_self=True)

        shifted = scores - max_per_dst[edge_dst]
        exp_scores = torch.exp(shifted)
        sum_per_dst = torch.zeros((inputs.shape[0],), device=inputs.device, dtype=inputs.dtype)
        sum_per_dst.index_add_(0, edge_dst, exp_scores)
        alpha = exp_scores * sum_per_dst.clamp_min(1e-12).reciprocal()[edge_dst]
        if dropout > 0.0:
            alpha = F.dropout(alpha, p=dropout, training=training)

        msg_inputs = F.dropout(h_prime, p=dropout, training=training) if dropout > 0.0 else h_prime
        out = torch.zeros((inputs.shape[0], h_prime.shape[1]), device=inputs.device, dtype=inputs.dtype)
        out.index_add_(0, edge_dst, msg_inputs[edge_src] * alpha.unsqueeze(-1))
        out = out + self.output_bias.view(1, -1)
        return F.elu(out) if apply_activation else out


class HiddenLayer(nn.Module):
    def __init__(self, in_dim: int, head_count: int, head_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(in_dim, head_dim) for _ in range(head_count)])
        self.output_dim = head_count * head_dim

    def forward(
        self,
        inputs: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        *,
        dropout: float,
        training: bool,
    ) -> torch.Tensor:
        outputs = [
            head(inputs, edge_src, edge_dst, dropout=dropout, training=training, apply_activation=True)
            for head in self.heads
        ]
        return torch.cat(outputs, dim=-1)


class GATFamily(nn.Module):
    def __init__(self, input_dim: int, hidden_profile: Sequence[Dict[str, int]], k_out: int, num_classes: int) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        for layer in hidden_profile:
            module = HiddenLayer(current_dim, int(layer["head_count"]), int(layer["head_dim"]))
            self.hidden_layers.append(module)
            current_dim = module.output_dim
        self.output_heads = nn.ModuleList([AttentionHead(current_dim, num_classes) for _ in range(k_out)])

    def forward(
        self,
        features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        *,
        dropout: float,
        training: bool,
    ) -> torch.Tensor:
        hidden = features
        for layer in self.hidden_layers:
            hidden = layer(hidden, edge_src, edge_dst, dropout=dropout, training=training)
        outputs = [
            head(hidden, edge_src, edge_dst, dropout=dropout, training=training, apply_activation=False)
            for head in self.output_heads
        ]
        if len(outputs) == 1:
            return outputs[0]
        return torch.stack(outputs, dim=0).mean(dim=0)


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or name.endswith(".output_bias"):
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


def micro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)
    labels_float = labels.to(dtype=torch.float32)
    tp = float((predictions * labels_float).sum().item())
    fp = float((predictions * (1.0 - labels_float)).sum().item())
    fn = float(((1.0 - predictions) * labels_float).sum().item())
    if tp == 0.0:
        return 0.0
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate(
    model: GATFamily,
    graphs: Sequence[GraphSample],
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    if not graphs:
        return 0.0, 0.0
    with torch.no_grad():
        batch = pack_graphs(graphs, device)
        logits = model(batch.features, batch.edge_src, batch.edge_dst, dropout=0.0, training=False)
        loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
        micro_f1 = micro_f1_from_logits(logits, batch.labels)
    return float(loss.item()), float(micro_f1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ppi")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-profile", default="[1x8]")
    parser.add_argument("--k-out", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(args.data_root).resolve()
    hidden_profile = parse_hidden_profile_expr(args.hidden_profile)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    all_graphs, num_features, num_classes = load_ppi_samples(data_root)
    train_graphs = [graph for graph in all_graphs if graph.split == "train"]
    val_graphs = [graph for graph in all_graphs if graph.split == "val"]
    test_graphs = [graph for graph in all_graphs if graph.split == "test"]
    if not train_graphs or not val_graphs or not test_graphs:
        raise RuntimeError("PPI full-dataset split is incomplete")

    model = GATFamily(num_features, hidden_profile, args.k_out, num_classes).to(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)

    best_state = None
    best_metrics = None
    best_val_loss = math.inf
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        random.shuffle(train_graphs)
        for start_index in range(0, len(train_graphs), args.batch_size):
            batch_graphs = train_graphs[start_index:start_index + args.batch_size]
            batch = pack_graphs(batch_graphs, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.features, batch.edge_src, batch.edge_dst, dropout=args.dropout, training=True)
            loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        train_loss, train_f1 = evaluate(model, train_graphs, device)
        val_loss, val_f1 = evaluate(model, val_graphs, device)
        test_loss, test_f1 = evaluate(model, test_graphs, device)
        mean_epoch_loss = epoch_loss / max(steps, 1)
        history.append(
            {
                "epoch": float(epoch),
                "epoch_loss": mean_epoch_loss,
                "train_loss": train_loss,
                "train_micro_f1": train_f1,
                "val_loss": val_loss,
                "val_micro_f1": val_f1,
                "test_loss": test_loss,
                "test_micro_f1": test_f1,
            }
        )
        print(
            f"[ppi] epoch={epoch:04d} batch_loss={mean_epoch_loss:.6f} train_f1={train_f1:.4f} "
            f"val_f1={val_f1:.4f} test_f1={test_f1:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": train_loss,
                "train_micro_f1": train_f1,
                "val_loss": val_loss,
                "val_micro_f1": val_f1,
                "test_loss": test_loss,
                "test_micro_f1": test_f1,
            }
            torch.save(
                {
                    "bundle": {
                        "dataset": "ppi",
                        "L": len(hidden_profile) + 1,
                        "d_in_profile": [num_features],
                        "hidden_profile": hidden_profile,
                        "K_out": args.k_out,
                        "C": num_classes,
                        "num_features": num_features,
                        "num_classes": num_classes,
                        "seed": args.seed,
                        "dropout": args.dropout,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "task_type": "inductive_multi_graph_node_classification",
                        "report_unit": "graph",
                        "batching_rule": "multi_graph_batch",
                        "subgraph_rule": "whole_graph",
                        "self_loop_rule": "per_node",
                        "edge_sort_rule": "edge_gid_then_dst_stable",
                        "model_arch_id": f"gat_family_L{len(hidden_profile)+1}_hid"
                        + "-".join(f"{layer['head_count']}x{layer['head_dim']}" for layer in hidden_profile)
                        + f"_kout{args.k_out}_c{num_classes}",
                        "model_param_id": f"ppi_full_seed{args.seed}",
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
            if bad_epochs >= args.patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a checkpoint")

    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "数据集": "ppi",
                "设备": str(device),
                "总图数": len(all_graphs),
                "训练图数": len(train_graphs),
                "验证图数": len(val_graphs),
                "测试图数": len(test_graphs),
                "总耗时秒": time.perf_counter() - start,
                "最佳指标": best_metrics,
                "最后十轮": history[-10:],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
