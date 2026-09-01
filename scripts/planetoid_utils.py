#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp

from quantization_scale_chain import ROW_NORMALIZE_INPUT_PREPROCESS_ID


def parse_index_file(path: pathlib.Path) -> List[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_raw_planetoid_dataset(
    root: pathlib.Path,
    name: str,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict[int, List[int]], List[int]]:
    base = root / name
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for suffix in names:
        with (base / f"ind.{name}.{suffix}").open("rb") as handle:
            objects.append(pickle.load(handle, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects
    test_idx_reorder = parse_index_file(base / f"ind.{name}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if name == "citeseer":
        full_range = np.arange(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(full_range), x.shape[1]), dtype=tx.dtype)
        tx_extended[test_idx_range - min(test_idx_reorder), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(full_range), y.shape[1]), dtype=ty.dtype)
        ty_extended[test_idx_range - min(test_idx_reorder), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    adjacency = {int(src): [int(dst) for dst in dsts] for src, dsts in dict(graph).items()}
    return features.tocsr(), labels, adjacency, test_idx_reorder


def load_planetoid_dataset_with_masks(
    root: pathlib.Path,
    name: str,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict[int, List[int]], np.ndarray, np.ndarray, np.ndarray]:
    base = root / name
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for suffix in names:
        with (base / f"ind.{name}.{suffix}").open("rb") as handle:
            objects.append(pickle.load(handle, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects
    test_idx_reorder = parse_index_file(base / f"ind.{name}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if name == "citeseer":
        full_range = np.arange(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(full_range), x.shape[1]), dtype=tx.dtype)
        tx_extended[test_idx_range - min(test_idx_reorder), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(full_range), y.shape[1]), dtype=ty.dtype)
        ty_extended[test_idx_range - min(test_idx_reorder), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    adjacency = {int(src): [int(dst) for dst in dsts] for src, dsts in dict(graph).items()}

    train_mask = np.zeros(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)
    train_mask[: y.shape[0]] = True
    val_mask[y.shape[0] : y.shape[0] + 500] = True
    test_mask[test_idx_range.tolist()] = True
    return features.tocsr(), labels, adjacency, train_mask, val_mask, test_mask


def _read_cache_metadata(path: pathlib.Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical Planetoid cache metadata is missing: {path}")
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: malformed metadata line")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in values:
            raise ValueError(f"{path}:{line_number}: duplicate or empty metadata key")
        values[key] = value
    return values


def load_canonical_planetoid_features(
    cache_root: pathlib.Path,
    dataset: str,
) -> np.ndarray:
    """Read the exact sparse decimal feature producer consumed by C++.

    The formal loader reads ``features.txt`` and parses each value as binary64
    before applying the row-normalization contract.  Training must consume that
    same producer output instead of normalizing the pickle object independently.
    """
    cache = cache_root / dataset
    metadata = _read_cache_metadata(cache / "meta.cfg")
    if metadata.get("name") != dataset:
        raise ValueError(
            f"canonical Planetoid cache name mismatch: {metadata.get('name')!r}"
        )
    if metadata.get("input_preprocess_id") != ROW_NORMALIZE_INPUT_PREPROCESS_ID:
        raise ValueError(
            "canonical Planetoid cache is missing row_normalize producer identity"
        )
    features_path = cache / "features.txt"
    if not features_path.is_file():
        raise FileNotFoundError(
            f"canonical Planetoid cache features.txt is missing: {features_path}"
        )
    lines = features_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"canonical Planetoid feature cache is empty: {features_path}")
    shape = lines[0].split()
    if len(shape) != 2:
        raise ValueError(f"{features_path}: malformed feature shape header")
    try:
        rows, cols = (int(value) for value in shape)
    except ValueError as error:
        raise ValueError(f"{features_path}: feature shape is not integral") from error
    expected_rows = int(metadata.get("num_nodes", rows))
    expected_cols = int(metadata.get("num_features", cols))
    if rows <= 0 or cols <= 0 or (rows, cols) != (expected_rows, expected_cols):
        raise ValueError(f"{features_path}: feature shape conflicts with cache metadata")
    if len(lines) != rows + 1:
        raise ValueError(f"{features_path}: expected exactly {rows} feature rows")
    dense = np.zeros((rows, cols), dtype=np.float64)
    for row_index, line in enumerate(lines[1:]):
        seen_columns = set()
        for token in line.split():
            if ":" not in token:
                raise ValueError(f"{features_path}:{row_index + 2}: malformed sparse entry")
            column_text, value_text = token.split(":", 1)
            try:
                column = int(column_text)
                value = float(value_text)
            except ValueError as error:
                raise ValueError(
                    f"{features_path}:{row_index + 2}: malformed sparse value"
                ) from error
            if not 0 <= column < cols or not np.isfinite(value):
                raise ValueError(
                    f"{features_path}:{row_index + 2}: sparse entry is out of bounds or non-finite"
                )
            if column in seen_columns:
                raise ValueError(
                    f"{features_path}:{row_index + 2}: duplicate sparse column"
                )
            seen_columns.add(column)
            dense[row_index, column] = value
    return dense


def build_dense_adjacency(adjacency: Dict[int, Iterable[int]], num_nodes: int) -> np.ndarray:
    dense = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dsts in adjacency.items():
        for dst in dsts:
            dense[int(src), int(dst)] = 1.0
    return dense


def build_sorted_edge_index(
    adjacency: Dict[int, Iterable[int]],
    num_nodes: int,
    add_self_loops: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    edges: List[Tuple[int, int]] = []
    for src in sorted(adjacency):
        for dst in adjacency[src]:
            edges.append((int(src), int(dst)))
    if add_self_loops:
        seen = set(edges)
        for node in range(num_nodes):
            key = (node, node)
            if key not in seen:
                edges.append(key)
    edges.sort(key=lambda item: (item[1], item[0]))
    src = np.fromiter((edge[0] for edge in edges), dtype=np.int64, count=len(edges))
    dst = np.fromiter((edge[1] for edge in edges), dtype=np.int64, count=len(edges))
    return src, dst


def adj_to_bias(adjacency: Dict[int, Iterable[int]], num_nodes: int, nhood: int = 1) -> np.ndarray:
    dense = build_dense_adjacency(adjacency, num_nodes)
    mt = np.eye(num_nodes, dtype=np.float32)
    for _ in range(nhood):
        mt = mt @ (dense + np.eye(num_nodes, dtype=np.float32))
    mt[mt > 0.0] = 1.0
    return -1e9 * (1.0 - mt)


def labels_to_int(labels: np.ndarray) -> np.ndarray:
    return np.argmax(labels, axis=1).astype(np.int64)


def write_sparse_features(path: pathlib.Path, features: sp.csr_matrix) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{features.shape[0]} {features.shape[1]}\n")
        for row in range(features.shape[0]):
            start = features.indptr[row]
            end = features.indptr[row + 1]
            entries = [
                f"{int(col)}:{float(val):.12g}"
                for col, val in zip(features.indices[start:end], features.data[start:end])
            ]
            handle.write(" ".join(entries))
            handle.write("\n")


def write_labels(path: pathlib.Path, labels: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for label in labels_to_int(labels):
            handle.write(f"{int(label)}\n")


def write_edges(path: pathlib.Path, adjacency: Dict[int, Iterable[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for src in sorted(adjacency):
            for dst in adjacency[src]:
                handle.write(f"{src} {int(dst)}\n")
