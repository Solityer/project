from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import AppConfig


@dataclass
class GraphDataset:
    name: str
    num_nodes: int
    num_features: int
    num_classes: int
    features: np.ndarray
    labels: np.ndarray
    edges_src: np.ndarray
    edges_dst: np.ndarray
    edge_graph: np.ndarray
    node_ptr: np.ndarray
    graph_count: int
    task_type: str
    report_unit: str


@dataclass
class LocalGraph:
    name: str
    num_nodes: int
    num_features: int
    num_classes: int
    features: np.ndarray
    labels: np.ndarray
    absolute_ids: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    edge_graph: np.ndarray
    node_ptr: np.ndarray
    edge_ptr: np.ndarray
    graph_count: int
    n_total: int
    task_type: str
    report_unit: str
    batching_rule: str
    self_loop_rule: str
    edge_sort_rule: str
    chunking_rule: str


def _read_meta(path: pathlib.Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_size_vector(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=np.int64)
    values = [int(item) for item in path.read_text(encoding="utf-8").split()]
    return np.asarray(values, dtype=np.int64)


def _row_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    sums = features.sum(axis=1)
    mask = sums != 0.0
    features[mask] = features[mask] / sums[mask, None]
    return features


def _load_features(root: pathlib.Path, rows: int, cols: int) -> np.ndarray:
    text_path = root / "features.txt"
    if text_path.exists():
        features = np.zeros((rows, cols), dtype=np.float64)
        with text_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().split()
            if len(header) != 2 or int(header[0]) != rows or int(header[1]) != cols:
                raise ValueError(f"feature header mismatch: {text_path}")
            for row_index, line in enumerate(handle):
                if row_index >= rows:
                    raise ValueError(f"too many feature rows in {text_path}")
                for item in line.split():
                    col, value = item.split(":", 1)
                    features[row_index, int(col)] = float(value)
        return _row_normalize(features)
    npy_path = root / "features.npy"
    features = np.load(npy_path)
    if features.shape != (rows, cols):
        raise ValueError(f"feature npy shape mismatch: {npy_path} has {features.shape}, expected {(rows, cols)}")
    return _row_normalize(features.astype(np.float64, copy=False))


def _load_labels(path: pathlib.Path, rows: int) -> np.ndarray:
    if not path.exists():
        return np.zeros(rows, dtype=np.int64)
    labels = np.loadtxt(path, dtype=np.int64)
    labels = np.atleast_1d(labels)
    if labels.shape[0] != rows:
        raise ValueError(f"label count mismatch: {path}")
    return labels


def _load_edges(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, dtype=np.int64)
    raw = np.atleast_2d(raw)
    if raw.shape[1] == 2:
        graph = np.zeros(raw.shape[0], dtype=np.int64)
        src = raw[:, 0]
        dst = raw[:, 1]
    elif raw.shape[1] == 3:
        graph = raw[:, 0]
        src = raw[:, 1]
        dst = raw[:, 2]
    else:
        raise ValueError(f"malformed edge file: {path}")
    return src.astype(np.int64), dst.astype(np.int64), graph.astype(np.int64)


def load_dataset(config: AppConfig) -> GraphDataset:
    root = config.cache_dir
    meta = _read_meta(root / "meta.cfg")
    num_nodes = int(meta["num_nodes"])
    num_features = int(meta["num_features"])
    num_classes = int(meta["num_classes"])
    node_ptr = _load_size_vector(root / "node_ptr.txt")
    if node_ptr.size < 2:
        node_ptr = np.asarray([0, num_nodes], dtype=np.int64)
    graph_count = int(node_ptr.size - 1)
    src, dst, graph = _load_edges(root / "edges.txt")
    return GraphDataset(
        name=meta["name"],
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=num_classes,
        features=_load_features(root, num_nodes, num_features),
        labels=_load_labels(root / "labels.txt", num_nodes),
        edges_src=src,
        edges_dst=dst,
        edge_graph=graph,
        node_ptr=node_ptr,
        graph_count=graph_count,
        task_type=meta.get("task_type", config.entries.get("task_type", "transductive_node_classification")),
        report_unit=meta.get("report_unit", config.entries.get("report_unit", "node")),
    )


def normalize_graph(dataset: GraphDataset, config: AppConfig) -> LocalGraph:
    selected_graphs = np.arange(min(max(config.batch_graphs, 1), dataset.graph_count), dtype=np.int64)
    absolute_ids_parts = []
    local_src_parts = []
    local_dst_parts = []
    local_graph_parts = []
    node_ptr = [0]
    edge_ptr = [0]

    for batch_gid, dataset_gid in enumerate(selected_graphs.tolist()):
        begin = int(dataset.node_ptr[dataset_gid])
        end = int(dataset.node_ptr[dataset_gid + 1])
        graph_nodes = np.arange(begin, end, dtype=np.int64)
        local_offset = node_ptr[-1]
        node_ptr.append(local_offset + graph_nodes.size)
        absolute_ids_parts.append(graph_nodes)

        mask = dataset.edge_graph == dataset_gid
        src_global = dataset.edges_src[mask]
        dst_global = dataset.edges_dst[mask]
        in_graph = (src_global >= begin) & (src_global < end) & (dst_global >= begin) & (dst_global < end)
        src_local = src_global[in_graph] - begin + local_offset
        dst_local = dst_global[in_graph] - begin + local_offset

        if config.symmetrize_edges:
            src_original = src_local
            dst_original = dst_local
            rev_mask = src_local != dst_local
            src_local = np.concatenate([src_original, dst_original[rev_mask]])
            dst_local = np.concatenate([dst_original, src_original[rev_mask]])

        self_nodes = np.arange(local_offset, local_offset + graph_nodes.size, dtype=np.int64)
        src_local = np.concatenate([src_local, self_nodes])
        dst_local = np.concatenate([dst_local, self_nodes])
        graph_local = np.full(src_local.shape[0], batch_gid, dtype=np.int64)

        if config.deduplicate_edges:
            packed = np.stack([graph_local, dst_local, src_local], axis=1)
            _, unique_indices = np.unique(packed, axis=0, return_index=True)
            unique_indices.sort()
            src_local = src_local[unique_indices]
            dst_local = dst_local[unique_indices]
            graph_local = graph_local[unique_indices]

        order = np.lexsort((np.arange(src_local.size), dst_local, graph_local))
        src_local = src_local[order]
        dst_local = dst_local[order]
        graph_local = graph_local[order]

        local_src_parts.append(src_local)
        local_dst_parts.append(dst_local)
        local_graph_parts.append(graph_local)
        edge_ptr.append(edge_ptr[-1] + src_local.size)

    absolute_ids = np.concatenate(absolute_ids_parts) if absolute_ids_parts else np.arange(dataset.num_nodes, dtype=np.int64)
    src = np.concatenate(local_src_parts) if local_src_parts else np.array([], dtype=np.int64)
    dst = np.concatenate(local_dst_parts) if local_dst_parts else np.array([], dtype=np.int64)
    edge_graph = np.concatenate(local_graph_parts) if local_graph_parts else np.array([], dtype=np.int64)
    features = dataset.features[absolute_ids]
    labels = dataset.labels[absolute_ids]
    local = LocalGraph(
        name=f"{dataset.name}_normalized",
        num_nodes=int(absolute_ids.size),
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        features=features,
        labels=labels,
        absolute_ids=absolute_ids,
        src=src.astype(np.int64),
        dst=dst.astype(np.int64),
        edge_graph=edge_graph.astype(np.int64),
        node_ptr=np.asarray(node_ptr, dtype=np.int64),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
        graph_count=len(node_ptr) - 1,
        n_total=dataset.num_nodes,
        task_type=config.entries.get("task_type", dataset.task_type),
        report_unit=config.entries.get("report_unit", dataset.report_unit),
        batching_rule=config.entries.get("batching_rule", "whole_graph_single"),
        self_loop_rule=config.entries.get("self_loop_rule", "per_node"),
        edge_sort_rule=config.entries.get("edge_sort_rule", "edge_gid_then_dst_stable"),
        chunking_rule=config.entries.get("chunking_rule", "none"),
    )
    _validate_local_graph(local)
    return local


def _validate_local_graph(local: LocalGraph) -> None:
    if local.node_ptr[0] != 0 or local.node_ptr[-1] != local.num_nodes:
        raise ValueError("node_ptr does not bound the local node set")
    if local.edge_ptr[0] != 0 or local.edge_ptr[-1] != local.src.size:
        raise ValueError("edge_ptr does not bound the local edge set")
    if np.any(local.src < 0) or np.any(local.src >= local.num_nodes):
        raise ValueError("source index out of range")
    if np.any(local.dst < 0) or np.any(local.dst >= local.num_nodes):
        raise ValueError("destination index out of range")
    incoming = np.bincount(local.dst, minlength=local.num_nodes)
    if np.any(incoming == 0):
        raise ValueError("every node must have at least one incoming edge")
    if np.any(local.edge_graph[1:] < local.edge_graph[:-1]):
        raise ValueError("edges are not sorted by graph")
    same_graph = local.edge_graph[1:] == local.edge_graph[:-1]
    if np.any(local.dst[1:][same_graph] < local.dst[:-1][same_graph]):
        raise ValueError("edges are not sorted by dst inside each graph")
    for edge_index, gid in enumerate(local.edge_graph):
        src = int(local.src[edge_index])
        dst = int(local.dst[edge_index])
        if not (local.node_ptr[gid] <= src < local.node_ptr[gid + 1]):
            raise ValueError("source edge crosses graph boundary")
        if not (local.node_ptr[gid] <= dst < local.node_ptr[gid + 1]):
            raise ValueError("destination edge crosses graph boundary")
