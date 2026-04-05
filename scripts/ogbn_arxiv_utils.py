#!/usr/bin/env python3
from __future__ import annotations

import gzip
import pathlib
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class OgbnArxivData:
    features: np.ndarray
    labels: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    split_idx: Dict[str, np.ndarray]


def row_normalize_dense(features: np.ndarray) -> np.ndarray:
    rowsum = features.sum(axis=1, keepdims=True)
    normalized = features.copy()
    non_zero = rowsum.squeeze(-1) != 0.0
    normalized[non_zero] /= rowsum[non_zero]
    return normalized.astype(np.float32, copy=False)


def dataset_root(data_root: pathlib.Path) -> pathlib.Path:
    return data_root / "ogbn-arxiv"


def cache_root(project_root: pathlib.Path) -> pathlib.Path:
    return project_root / "data" / "cache" / "ogbn_arxiv"


def load_raw_arrays(data_root: pathlib.Path) -> OgbnArxivData:
    root = dataset_root(data_root)
    raw_root = root / "raw"
    split_root = root / "split" / "time"

    features = np.loadtxt(
        gzip.open(raw_root / "node-feat.csv.gz", "rt"),
        delimiter=",",
        dtype=np.float32,
    )
    labels = np.loadtxt(
        gzip.open(raw_root / "node-label.csv.gz", "rt"),
        delimiter=",",
        dtype=np.int64,
    ).reshape(-1)
    edges = np.loadtxt(
        gzip.open(raw_root / "edge.csv.gz", "rt"),
        delimiter=",",
        dtype=np.int64,
    )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("ogbn-arxiv edge.csv.gz must have shape [E, 2]")

    split_idx = {
        "train": np.loadtxt(gzip.open(split_root / "train.csv.gz", "rt"), delimiter=",", dtype=np.int64).reshape(-1),
        "valid": np.loadtxt(gzip.open(split_root / "valid.csv.gz", "rt"), delimiter=",", dtype=np.int64).reshape(-1),
        "test": np.loadtxt(gzip.open(split_root / "test.csv.gz", "rt"), delimiter=",", dtype=np.int64).reshape(-1),
    }
    return OgbnArxivData(
        features=features,
        labels=labels,
        edge_src=edges[:, 0].astype(np.int64, copy=False),
        edge_dst=edges[:, 1].astype(np.int64, copy=False),
        split_idx=split_idx,
    )


def build_sorted_edge_index(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    num_nodes: int,
    *,
    symmetrize: bool,
    add_self_loops: bool,
) -> tuple[np.ndarray, np.ndarray]:
    src = edge_src.astype(np.int64, copy=False)
    dst = edge_dst.astype(np.int64, copy=False)
    if symmetrize:
        src = np.concatenate([src, dst])
        dst = np.concatenate([dst, edge_src.astype(np.int64, copy=False)])
    if add_self_loops:
        loops = np.arange(num_nodes, dtype=np.int64)
        src = np.concatenate([src, loops])
        dst = np.concatenate([dst, loops])

    packed = (dst.astype(np.int64) << 32) | src.astype(np.int64)
    unique_packed = np.unique(packed)
    sorted_dst = (unique_packed >> 32).astype(np.int64, copy=False)
    sorted_src = (unique_packed & np.int64(0xFFFFFFFF)).astype(np.int64, copy=False)
    return sorted_src, sorted_dst


def accuracy_for_indices(logits: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> float:
    predictions = logits[indices].argmax(axis=-1)
    return float((predictions == labels[indices]).mean())


def ensure_cache(project_root: pathlib.Path, data_root: pathlib.Path) -> pathlib.Path:
    output_root = cache_root(project_root)
    meta_path = output_root / "meta.cfg"
    if meta_path.exists():
        return output_root

    arrays = load_raw_arrays(data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "features.npy", arrays.features.astype(np.float32, copy=False))
    with (output_root / "labels.txt").open("w", encoding="utf-8") as handle:
        for label in arrays.labels:
            handle.write(f"{int(label)}\n")
    with (output_root / "edges.txt").open("w", encoding="utf-8") as handle:
        for src, dst in zip(arrays.edge_src.tolist(), arrays.edge_dst.tolist()):
            handle.write(f"{src} {dst}\n")
    meta_path.write_text(
        "\n".join(
            [
                "name = ogbn-arxiv",
                f"num_nodes = {arrays.features.shape[0]}",
                f"num_features = {arrays.features.shape[1]}",
                f"num_classes = {int(arrays.labels.max()) + 1}",
                "graph_count = 1",
                "task_type = transductive_node_classification",
                "report_unit = node",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output_root
