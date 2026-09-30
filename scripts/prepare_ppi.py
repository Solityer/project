#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from collections import deque

import numpy as np

from quantization_scale_chain import ROW_NORMALIZE_INPUT_PREPROCESS_ID


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
    if features.ndim != 2 or features.shape[0] != node_count or features.shape[1] != 50:
        raise ValueError(f"PPI features must have shape ({node_count}, 50), got {features.shape}")
    if len(class_map) != node_count:
        raise ValueError(f"PPI class map must cover {node_count} nodes, got {len(class_map)}")
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
    inverse = {original_node: new for new, original_node in enumerate(permutation)}
    features_permuted = features[permutation]

    class_vectors = np.zeros((node_count, len(next(iter(class_map.values())))), dtype=np.int64)
    for raw_node, vector in class_map.items():
        class_index = int(id_map[raw_node])
        raw_class_vector = np.asarray(vector)
        if raw_class_vector.shape != (class_vectors.shape[1],):
            raise ValueError(f"PPI class vector shape mismatch for node {raw_node}")
        if not np.all((raw_class_vector == 0) | (raw_class_vector == 1)):
            raise ValueError(f"PPI class vector is not multi-hot for node {raw_node}")
        class_vector = raw_class_vector.astype(np.int64, copy=False)
        class_vectors[class_index] = class_vector

    split_masks = {"train": np.ones(node_count, dtype=bool), "val": np.zeros(node_count, dtype=bool), "test": np.zeros(node_count, dtype=bool)}
    for node in graph["nodes"]:
        raw_node = str(node["id"])
        node_index = int(id_map[raw_node])
        is_val = bool(node.get("val", False))
        is_test = bool(node.get("test", False))
        if is_val and is_test:
            raise ValueError(f"PPI node belongs to both val and test splits: {raw_node}")
        split_masks["val"][node_index] = is_val
        split_masks["test"][node_index] = is_test
        split_masks["train"][node_index] = not (is_val or is_test)
    if len(graph["nodes"]) != node_count:
        raise ValueError(f"PPI graph node list must cover {node_count} nodes, got {len(graph['nodes'])}")

    labels_multi_permuted = class_vectors[permutation].astype(np.float32, copy=False)
    split_masks_permuted = {name: mask[np.asarray(permutation, dtype=np.int64)] for name, mask in split_masks.items()}

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

    np.save(output_root / "features.npy", features_permuted.astype(np.float32, copy=False))
    np.save(output_root / "labels_multi.npy", labels_multi_permuted)
    np.savetxt(
        output_root / "labels_multi.txt",
        labels_multi_permuted.astype(np.int8, copy=False),
        fmt="%d",
        header=f"{node_count} {class_vectors.shape[1]}",
        comments="",
    )
    for split_name, mask in split_masks_permuted.items():
        np.save(output_root / f"{split_name}_mask.npy", mask)
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
            "label_format": "multi_hot",
            "split_format": "node_masks",
            "input_preprocess_id": ROW_NORMALIZE_INPUT_PREPROCESS_ID,
        },
    )


if __name__ == "__main__":
    main()
