#!/usr/bin/env python3
"""Check all five formal input configurations without materializing a dataset."""
from __future__ import annotations
import argparse
import ast
import json
import re
from pathlib import Path
from asset_manifest import local_path, read_config, reject_source_path_values

DATASETS = ('cora', 'citeseer', 'pubmed', 'ppi', 'ogbn_arxiv')

def regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'formal input is missing, empty or symlinked: {path}')
    return path

def check_config(root: Path, dataset: str) -> None:
    path = regular(root / 'configs' / f'{dataset}_secure.cfg')
    values = read_config(path)
    reject_source_path_values(values)
    if values.get('dataset') != dataset or values.get('auto_prepare_dataset') != 'false' or values.get('dump_trace') != 'false':
        raise ValueError(f'formal dataset identity/input/trace policy mismatch: {path}')
    for key, expected in {'L': '3', 'hidden_profile': '[8x8,8x8]', 'K_out': '2', 'privacy_policy': 'model_and_trace_hidden', 'input_privacy': 'public_graph_public_features', 'output_mode': 'public_logits', 'proof_task': 'node_classification_full_gat', 'srs_type': 'native_local_ephemeral_random'}.items():
        if values.get(key) != expected:
            raise ValueError(f'formal config {key} mismatch: {path}')
    for key in ('data_root', 'cache_root', 'export_dir', 'checkpoint_bundle', 'static_table_descriptor', 'sorted_lookup_descriptor', 'integer_bound_ledger', 'srs_path', 'model_registry_path', 'model_blinder_sidecar_path'):
        local_path(root, values[key], key)
    cache = local_path(root, values['cache_root'], 'cache_root') / dataset
    meta = read_config(regular(cache / 'meta.cfg'))
    nodes, features, classes = (int(meta[key]) for key in ('num_nodes', 'num_features', 'num_classes'))
    if min(nodes, features, classes) < 1 or meta.get('name') != dataset or meta.get('input_preprocess_id') != values.get('input_preprocess_id') or int(values['num_classes']) != classes:
        raise ValueError(f'formal cache metadata mismatch: {dataset}')
    profile = [int(item) for item in re.findall(r'\d+', values['d_in_profile'])]
    if profile != [features, 64]:
        raise ValueError(f'formal input dimensions mismatch: {dataset}')
    required_cache = ['edges.txt', 'labels_multi.txt' if dataset == 'ppi' else 'labels.txt']
    if dataset in ('cora', 'citeseer', 'pubmed'):
        required_cache.append('test.index')
    for name in required_cache:
        regular(cache / name)
    if dataset in ('ppi', 'ogbn_arxiv'):
        with regular(cache / 'features.npy').open('rb') as stream:
            if stream.read(6) != b'\x93NUMPY':
                raise ValueError(f'formal NPY magic mismatch: {dataset}')
            version = stream.read(2)
            width = 2 if version[0] == 1 else 4
            header_length = int.from_bytes(stream.read(width), 'little')
            if header_length > 65536:
                raise ValueError(f'formal NPY header is too large: {dataset}')
            header = ast.literal_eval(stream.read(header_length).decode('latin1').strip())
            if header.get('descr') != '<f4' or header.get('fortran_order') is not False or header.get('shape') != (nodes, features):
                raise ValueError(f'formal NPY dtype/layout/shape mismatch: {dataset}')
            if (cache / 'features.npy').stat().st_size != stream.tell() + nodes * features * 4:
                raise ValueError(f'formal NPY payload size mismatch: {dataset}')
    else:
        with regular(cache / 'features.txt').open() as stream:
            if [int(item) for item in stream.readline().split()] != [nodes, features]:
                raise ValueError(f'formal sparse feature shape mismatch: {dataset}')
    graph_count = int(meta.get('graph_count', '1'))
    if dataset == 'ppi':
        ptr = [int(item) for item in regular(cache / 'node_ptr.txt').read_text().split()]
        if len(ptr) != graph_count + 1 or ptr[0] != 0 or ptr[-1] != nodes or any(a >= b for a, b in zip(ptr, ptr[1:])):
            raise ValueError('formal PPI graph boundaries mismatch')
    elif graph_count != 1:
        raise ValueError(f'formal non-PPI graph count mismatch: {dataset}')
    checkpoint = local_path(root, values['checkpoint_bundle'], 'checkpoint_bundle')
    manifest = json.loads(regular(checkpoint / 'manifest.json').read_text())
    for name in ('tensors.txt', 'tensors.npz', 'best_model.pt'):
        regular(checkpoint / name)
    for key, expected in {'dataset': dataset, 'L': 3, 'K_out': 2, 'C': classes, 'd_in_profile': profile, 'checkpoint_prefix': values['checkpoint_bundle'] + '/best_model.pt'}.items():
        if manifest.get(key) != expected:
            raise ValueError(f'formal checkpoint {key} mismatch: {dataset}')
    print(f'formal_input_preflight=PASS dataset={dataset} nodes={nodes} features={features} classes={classes} graphs={graph_count}')

def main() -> int:
    argparse.ArgumentParser().parse_args()
    root = Path(__file__).resolve().parent.parent
    regular(root / 'configs/quantization_scale_chain.cfg')
    for dataset in DATASETS:
        check_config(root, dataset)
    print('formal_input_preflight=PASS datasets=5 backend=cpu')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
