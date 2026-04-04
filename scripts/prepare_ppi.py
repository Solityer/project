#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from collections import deque

import numpy as np


def write_meta(path: pathlib.Path, entries: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries.items():
            handle.write(f"{key} = {value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    graph = json.loads((data_root / "ppi-G.json").read_text(encoding="utf-8"))
    id_map = json.loads((data_root / "ppi-id_map.json").read_text(encoding="utf-8"))
    class_map = json.loads((data_root / "ppi-class_map.json").read_text(encoding="utf-8"))
    features = np.load(data_root / "ppi-feats.npy")

    node_count = len(id_map)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    edge_pairs: list[tuple[int, int]] = []
    for link in graph["links"]:
        src = int(id_map[str(link["source"])])
        dst = int(id_map[str(link["target"])])
        adjacency[src].append(dst)
        adjacency[dst].append(src)
        edge_pairs.append((src, dst))

    visited = [False] * node_count
    components: list[list[int]] = []
    for start in range(node_count):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda comp: comp[0])

    permutation = [node for component in components for node in component]
    inverse = {old: new for new, old in enumerate(permutation)}
    features_permuted = features[permutation]

    class_vectors = np.zeros((node_count, len(next(iter(class_map.values())))), dtype=np.int64)
    for raw_node, vector in class_map.items():
        class_vectors[int(raw_node)] = np.asarray(vector, dtype=np.int64)
    labels_permuted = np.argmax(class_vectors[permutation], axis=1)

    node_ptr = [0]
    for component in components:
        node_ptr.append(node_ptr[-1] + len(component))

    graph_id_by_node = np.zeros(node_count, dtype=np.int64)
    for graph_id, component in enumerate(components):
        for node in component:
            graph_id_by_node[node] = graph_id

    with (output_root / "edges.txt").open("w", encoding="utf-8") as handle:
        for src, dst in edge_pairs:
            graph_id = int(graph_id_by_node[src])
            if graph_id != int(graph_id_by_node[dst]):
                continue
            handle.write(f"{graph_id} {inverse[src]} {inverse[dst]}\n")

    with (output_root / "node_ptr.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(str(value) for value in node_ptr))
        handle.write("\n")

    with (output_root / "labels.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(str(int(value)) for value in labels_permuted))
        handle.write("\n")

    np.save(output_root / "features.npy", features_permuted.astype(np.float32, copy=False))
    write_meta(
        output_root / "meta.cfg",
        {
            "name": "ppi",
            "num_nodes": features_permuted.shape[0],
            "num_features": features_permuted.shape[1],
            "num_classes": class_vectors.shape[1],
            "graph_count": len(components),
            "task_type": "inductive_multi_graph_node_classification",
            "report_unit": "graph",
        },
    )


if __name__ == "__main__":
    main()
