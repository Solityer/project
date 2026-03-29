#!/usr/bin/env python3
import argparse
import pathlib
import pickle
from typing import Dict, Iterable, List

import numpy as np
import scipy.sparse as sp


def parse_index_file(path: pathlib.Path) -> List[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_raw_dataset(root: pathlib.Path, name: str):
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

    adjacency: Dict[int, Iterable[int]] = dict(graph)
    return features.tocsr(), labels, adjacency, test_idx_reorder


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
        for row in labels:
            handle.write(f"{int(np.argmax(row))}\n")


def write_edges(path: pathlib.Path, adjacency: Dict[int, Iterable[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for src in sorted(adjacency):
            for dst in adjacency[src]:
                handle.write(f"{src} {int(dst)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)
    cache_root = pathlib.Path(args.cache_root) / args.dataset
    cache_root.mkdir(parents=True, exist_ok=True)

    features, labels, adjacency, test_idx = load_raw_dataset(data_root, args.dataset)

    write_sparse_features(cache_root / "features.txt", features)
    write_labels(cache_root / "labels.txt", labels)
    write_edges(cache_root / "edges.txt", adjacency)

    with (cache_root / "meta.cfg").open("w", encoding="utf-8") as handle:
        handle.write(f"name = {args.dataset}\n")
        handle.write(f"num_nodes = {features.shape[0]}\n")
        handle.write(f"num_features = {features.shape[1]}\n")
        handle.write(f"num_classes = {labels.shape[1]}\n")
        handle.write(f"test_count = {len(test_idx)}\n")

    with (cache_root / "test.index").open("w", encoding="utf-8") as handle:
        for idx in test_idx:
            handle.write(f"{idx}\n")


if __name__ == "__main__":
    main()
