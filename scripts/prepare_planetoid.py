#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

from planetoid_utils import load_raw_planetoid_dataset, write_edges, write_labels, write_sparse_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)
    cache_root = pathlib.Path(args.cache_root) / args.dataset
    cache_root.mkdir(parents=True, exist_ok=True)

    features, labels, adjacency, test_idx = load_raw_planetoid_dataset(data_root, args.dataset)

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
