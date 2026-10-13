#!/usr/bin/env python3
"""Check all five formal input configurations without materializing a dataset."""
from __future__ import annotations
import argparse
import ast
import hashlib
import math
import json
import os
import re
import struct
import sys
from pathlib import Path
from asset_manifest import local_path, read_config, reject_source_path_values, sha256

DATASETS = ('cora', 'citeseer', 'pubmed', 'ppi', 'ogbn_arxiv')
CANONICAL_FULL_SHAPES = {
    'cora': (2708, 1433, 7, 1, 10858),
    'citeseer': (3327, 3703, 6, 1, 9464),
    'pubmed': (19717, 500, 3, 1, 88676),
    'ppi': (56944, 50, 121, 295, 818716),
    'ogbn_arxiv': (169343, 128, 40, 1, 1166243),
}
MAX_NATIVE_DOMAIN = 1 << 30

def padded_extent(value: int) -> int:
    return 1 << (value - 1).bit_length()

def decimal_int(token: str, label: str) -> int:
    if not re.fullmatch(r'(?:0|[1-9][0-9]*)', token):
        raise ValueError(f'{label} is not an unsigned decimal integer')
    value = int(token, 10)
    if value > (1 << 64) - 1:
        raise ValueError(f'{label} exceeds u64')
    return value

def regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'formal input is missing, empty or symlinked: {path}')
    return path

def audit_resource_prerequisites(config: Path) -> dict[str, object]:
    from prepare_form_resources import audit_formal_prerequisites
    evidence = audit_formal_prerequisites(config)
    if not isinstance(evidence, dict):
        raise ValueError(f'formal resource prerequisite audit returned invalid evidence: {config}')
    return evidence

def npy_payload(path: Path, expected_dtype: str, expected_shape: tuple[int, ...]) -> tuple[object, int]:
    """Validate the exact NPY header supported by the native reader."""
    with regular(path).open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError(f'formal NPY magic mismatch: {path.name}')
        version = stream.read(2)
        if version not in (b'\x01\x00', b'\x02\x00'):
            raise ValueError(f'unsupported formal NPY format version: {path.name}')
        width = 2 if version == b'\x01\x00' else 4
        raw_length = stream.read(width)
        if len(raw_length) != width:
            raise ValueError(f'truncated formal NPY header length: {path.name}')
        header_length = int.from_bytes(raw_length, 'little')
        if header_length == 0 or header_length > 65536:
            raise ValueError(f'formal NPY header length is invalid: {path.name}')
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f'truncated formal NPY header: {path.name}')
        header_text = raw_header.decode('latin1').strip()
        try:
            parsed = ast.parse(header_text, mode='eval').body
            if not isinstance(parsed, ast.Dict):
                raise ValueError('header is not a dictionary')
            keys = [ast.literal_eval(key) for key in parsed.keys]
            if len(keys) != len(set(keys)):
                raise ValueError('header has duplicate dictionary keys')
            header = ast.literal_eval(header_text)
        except (UnicodeDecodeError, SyntaxError, TypeError, ValueError) as exc:
            raise ValueError(f'formal NPY header is malformed: {path.name}: {exc}') from exc
        if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'} or header.get('descr') != expected_dtype or header.get('fortran_order') is not False or header.get('shape') != expected_shape:
            raise ValueError(f'formal NPY dtype/layout/shape mismatch: {path.name}')
        if expected_dtype == '<f4':
            native_descr = "'descr': '<f4'" in header_text or '"descr": "<f4"' in header_text
            native_order = "'fortran_order': False" in header_text or '"fortran_order": False' in header_text
            shape_position = header_text.find('shape')
            left = header_text.find('(', shape_position)
            right = header_text.find(')', left)
            raw_axes = header_text[left + 1:right].split(',') if left >= 0 and right >= 0 else []
            if len(raw_axes) != 2 or any(not re.fullmatch(r'\s*[0-9]+\s*', axis) for axis in raw_axes):
                raise ValueError(f'formal NPY shape syntax is not accepted by native reader: {path.name}')
            if not native_descr or not native_order:
                raise ValueError(f'formal NPY header syntax is not accepted by native reader: {path.name}')
        payload_offset = stream.tell()
    return header, payload_offset

def quantize_feature(value: float, scale: int, where: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f'non-finite formal feature: {where}')
    scaled = value * scale
    if not math.isfinite(scaled):
        raise ValueError(f'formal feature quantization overflow: {where}')
    quantized = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    if quantized < -(1 << 63) or quantized > (1 << 63) - 1:
        raise ValueError(f'formal feature exceeds signed 64-bit quantization range: {where}')
    return quantized

def audit_npy_features(path: Path, nodes: int, features: int, scale: int, normalize: bool) -> dict[str, int]:
    _, payload_offset = npy_payload(path, '<f4', (nodes, features))
    value_count = nodes * features
    if value_count > ((1 << 63) - 1) // 4 or path.stat().st_size != payload_offset + value_count * 4:
        raise ValueError(f'formal NPY payload size mismatch: {path.name}')
    maximum_abs = 0
    with path.open('rb') as stream:
        stream.seek(payload_offset)
        for row in range(nodes):
            raw = stream.read(features * 4)
            if len(raw) != features * 4:
                raise ValueError(f'truncated formal NPY feature row: {row}')
            values = struct.unpack(f'<{features}f', raw)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'non-finite formal NPY feature value in row {row}')
            denominator = math.fsum(values) if normalize else 1.0
            if not math.isfinite(denominator):
                raise ValueError(f'non-finite formal feature row sum: {row}')
            for col, source in enumerate(values):
                transformed = source / denominator if normalize and denominator != 0 else source
                maximum_abs = max(maximum_abs, abs(quantize_feature(transformed, scale, f'{row}:{col}')))
        if stream.read(1):
            raise ValueError(f'formal NPY feature payload has trailing bytes: {path.name}')
    return {'rows_checked': nodes, 'values_checked': value_count, 'max_abs_quantized': maximum_abs}

def audit_sparse_features(path: Path, nodes: int, features: int, scale: int, normalize: bool) -> dict[str, int]:
    maximum_abs = 0
    value_count = 0
    with regular(path).open(encoding='utf-8') as stream:
        shape = stream.readline().split()
        if len(shape) != 2 or tuple(decimal_int(item, 'sparse feature shape') for item in shape) != (nodes, features):
            raise ValueError('formal sparse feature shape mismatch')
        for row in range(nodes):
            line = stream.readline()
            if not line:
                raise ValueError(f'truncated formal sparse feature file at row {row}')
            entries: list[tuple[int, float]] = []
            columns: set[int] = set()
            for token in line.split():
                if token.count(':') != 1:
                    raise ValueError(f'malformed formal sparse feature token at row {row}')
                raw_col, raw_value = token.split(':', 1)
                try:
                    col = decimal_int(raw_col, 'sparse feature column')
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(f'malformed formal sparse feature at row {row}') from exc
                if not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', raw_value):
                    raise ValueError(f'formal sparse feature value is not a complete decimal float at row {row}')
                if col >= features or not math.isfinite(value):
                    raise ValueError(f'formal sparse feature index/value out of range at row {row}')
                if col in columns:
                    raise ValueError(f'duplicate formal sparse feature column at row {row}')
                columns.add(col)
                entries.append((col, value))
                value_count += 1
            denominator = math.fsum(value for _, value in entries) if normalize else 1.0
            if not math.isfinite(denominator):
                raise ValueError(f'non-finite formal sparse feature row sum: {row}')
            for col, source in entries:
                transformed = source / denominator if normalize and denominator != 0 else source
                maximum_abs = max(maximum_abs, abs(quantize_feature(transformed, scale, f'{row}:{col}')))
        if any(line.split() for line in stream):
            raise ValueError('formal sparse feature file has trailing rows')
    return {'rows_checked': nodes, 'values_checked': value_count, 'max_abs_quantized': maximum_abs}

def audit_labels(cache: Path, dataset: str, nodes: int, classes: int, meta: dict[str, str]) -> dict[str, int]:
    if dataset == 'ppi':
        if meta.get('label_format') != 'multi_hot':
            raise ValueError('formal PPI label format mismatch')
        path = regular(cache / 'labels_multi.txt')
        with path.open(encoding='utf-8') as stream:
            header = stream.readline().split()
            if len(header) != 2 or tuple(decimal_int(item, 'PPI label shape') for item in header) != (nodes, classes):
                raise ValueError('formal PPI multi-hot label shape mismatch')
            positive = 0
            for row in range(nodes):
                tokens = stream.readline().split()
                if len(tokens) != classes or any(token not in ('0', '1') for token in tokens):
                    raise ValueError(f'formal PPI multi-hot labels malformed at row {row}')
                positive += sum(token == '1' for token in tokens)
            if any(line.split() for line in stream):
                raise ValueError('formal PPI labels have trailing rows')
        if positive == 0:
            raise ValueError('formal PPI label matrix is empty')
        return {'rows_checked': nodes, 'classes_checked': classes, 'positive_labels': positive}
    path = regular(cache / 'labels.txt')
    with path.open(encoding='utf-8') as stream:
        for row in range(nodes):
            tokens = stream.readline().split()
            if len(tokens) != 1 or not re.fullmatch(r'[0-9]+', tokens[0]):
                raise ValueError(f'formal class labels malformed at row {row}')
            try:
                label = decimal_int(tokens[0], 'formal class label')
            except ValueError as exc:
                raise ValueError(f'formal class label is not an integer at row {row}') from exc
            if label < 0 or label >= classes:
                raise ValueError(f'formal class label out of range at row {row}')
        if any(line.split() for line in stream):
            raise ValueError('formal class labels have trailing rows')
    return {'rows_checked': nodes, 'classes_checked': classes}

def audit_test_indices(cache: Path, nodes: int, meta: dict[str, str]) -> dict[str, int]:
    expected = decimal_int(meta['test_count'], 'meta.test_count')
    seen: set[int] = set()
    with regular(cache / 'test.index').open(encoding='utf-8') as stream:
        for row, line in enumerate(stream):
            tokens = line.split()
            if len(tokens) != 1 or not re.fullmatch(r'[0-9]+', tokens[0]):
                raise ValueError(f'formal test index malformed at line {row + 1}')
            try:
                index = decimal_int(tokens[0], 'formal test index')
            except ValueError as exc:
                raise ValueError(f'formal test index is not an integer at line {row + 1}') from exc
            if index < 0 or index >= nodes or index in seen:
                raise ValueError(f'formal test index is duplicate or out of range at line {row + 1}')
            seen.add(index)
    if len(seen) != expected:
        raise ValueError(f'formal test index count mismatch: expected={expected} actual={len(seen)}')
    return {'indices_checked': len(seen), 'min_index': min(seen), 'max_index': max(seen)}

def audit_ppi_masks(cache: Path, nodes: int, meta: dict[str, str]) -> dict[str, int]:
    if meta.get('split_format') != 'node_masks':
        raise ValueError('formal PPI split format mismatch')
    result: dict[str, int] = {}
    for split in ('train', 'val', 'test'):
        path = cache / f'{split}_mask.npy'
        _, payload_offset = npy_payload(path, '|b1', (nodes,))
        if path.stat().st_size != payload_offset + nodes:
            raise ValueError(f'formal PPI {split} mask payload size mismatch')
        count = 0
        with path.open('rb') as stream:
            stream.seek(payload_offset)
            while True:
                block = stream.read(1 << 16)
                if not block:
                    break
                if any(value not in (0, 1) for value in block):
                    raise ValueError(f'formal PPI {split} mask contains non-boolean values')
                count += sum(block)
        if count == 0:
            raise ValueError(f'formal PPI {split} mask is empty')
        result[f'{split}_mask_count'] = count
    return result

def check_config(root: Path, dataset: str, config_path: Path | None = None) -> dict[str, object]:
    path = regular(config_path or (root / 'configs' / f'{dataset}_secure.cfg'))
    values = read_config(path)
    reject_source_path_values(values)
    quant_path = regular(root / 'configs' / 'quantization_scale_chain.cfg')
    quant_digest = sha256(quant_path)
    from prepare_form_resources import formal_input_scale_for
    scale = formal_input_scale_for(path)
    if sha256(quant_path) != quant_digest:
        raise ValueError('formal quantization scale chain changed during prerequisite scan')
    if values.get('dataset') != dataset or values.get('auto_prepare_dataset') != 'false' or values.get('dump_trace') != 'false':
        raise ValueError(f'formal dataset identity/input/trace policy mismatch: {path}')
    expected_preprocess = 'ogbn_arxiv_raw_float32' if dataset == 'ogbn_arxiv' else 'row_normalize'
    if values.get('input_preprocess_id') != expected_preprocess:
        raise ValueError(f'formal input preprocessing contract mismatch: {dataset}')
    for key, expected in {'L': '3', 'hidden_profile': '[8x8,8x8]', 'K_out': '2', 'privacy_policy': 'model_and_trace_hidden', 'input_privacy': 'public_graph_public_features', 'output_mode': 'public_logits', 'proof_task': 'node_classification_full_gat', 'srs_type': 'native_local_ephemeral_random'}.items():
        if values.get(key) != expected:
            raise ValueError(f'formal config {key} mismatch: {path}')
    for key in ('data_root', 'cache_root', 'export_dir', 'checkpoint_bundle', 'static_table_descriptor', 'sorted_lookup_descriptor', 'integer_bound_ledger', 'srs_path', 'model_registry_path', 'model_blinder_sidecar_path'):
        local_path(root, values[key], key)
    cache = local_path(root, values['cache_root'], 'cache_root') / dataset
    meta = read_config(regular(cache / 'meta.cfg'))
    nodes, features, classes = (decimal_int(meta[key], f'meta.{key}') for key in ('num_nodes', 'num_features', 'num_classes'))
    if min(nodes, features, classes) < 1 or meta.get('name') != dataset or meta.get('input_preprocess_id') != values.get('input_preprocess_id') or decimal_int(values['num_classes'], 'config.num_classes') != classes:
        raise ValueError(f'formal cache metadata mismatch: {dataset}')
    graph_count = decimal_int(meta.get('graph_count', '1'), 'meta.graph_count')
    expected_nodes, expected_features, expected_classes, expected_graphs, expected_raw_edges = CANONICAL_FULL_SHAPES[dataset]
    if (nodes, features, classes, graph_count) != (expected_nodes, expected_features, expected_classes, expected_graphs):
        raise ValueError(f'formal canonical full-scale shape mismatch: {(nodes, features, classes, graph_count)}')
    expected_range_bits = {'cora': 12, 'citeseer': 12, 'pubmed': 12, 'ppi': 14, 'ogbn_arxiv': 21}[dataset]
    for key, expected in {'seed': '11', 'range_bits': str(expected_range_bits)}.items():
        if values.get(key) != expected:
            raise ValueError(f'formal config {key} mismatch: {path}')
    if dataset == 'ppi' and values.get('batch_graphs') != '295':
        raise ValueError(f'formal config batch_graphs mismatch: {path}')
    padded_nodes, padded_features = padded_extent(nodes), padded_extent(features)
    if padded_nodes > MAX_NATIVE_DOMAIN // padded_features or nodes > sys.maxsize // features:
        raise ValueError(f'formal input feature relation shape exceeds native product/address bounds: {dataset}')
    if values['d_in_profile'] != f'[{features},64]':
        raise ValueError(f'formal input dimensions mismatch: {dataset}')
    profile = [features, 64]
    regular(cache / 'edges.txt')
    if scale < 1:
        raise ValueError('formal input quantization scale must be positive')
    if dataset in ('ppi', 'ogbn_arxiv'):
        feature_evidence = audit_npy_features(cache / 'features.npy', nodes, features, scale,
                                               values['input_preprocess_id'] == 'row_normalize')
    else:
        feature_evidence = audit_sparse_features(cache / 'features.txt', nodes, features, scale,
                                                  values['input_preprocess_id'] == 'row_normalize')
    label_evidence = audit_labels(cache, dataset, nodes, classes, meta)
    if dataset in ('cora', 'citeseer', 'pubmed'):
        if decimal_int(meta.get('test_count', ''), 'meta.test_count') != 1000:
            raise ValueError(f'formal canonical test split size mismatch: {dataset}')
        test_evidence = audit_test_indices(cache, nodes, meta)
    else:
        test_evidence = None
    if dataset == 'ppi':
        mask_evidence = audit_ppi_masks(cache, nodes, meta)
    else:
        mask_evidence = None
    if dataset == 'ppi':
        ptr = [decimal_int(item, 'PPI node_ptr') for item in regular(cache / 'node_ptr.txt').read_text().split()]
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
    evidence = audit_resource_prerequisites(path)
    if evidence.get('quantization_scale_chain_sha256') != quant_digest or sha256(quant_path) != quant_digest:
        raise ValueError('formal quantization scale chain changed during prerequisite scan')
    graph_evidence = evidence.get('graph', {})
    if graph_evidence.get('raw_edge_rows') != expected_raw_edges:
        raise ValueError(f"formal canonical raw edge count mismatch: expected={expected_raw_edges} actual={graph_evidence.get('raw_edge_rows')}")
    def tree_identity(directory: Path, normalize_checkpoint_manifest: bool = False) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for item in sorted(directory.rglob('*')):
            if item.is_symlink():
                raise ValueError(f'formal input tree contains a symlink: {item}')
            if item.is_dir():
                continue
            if not item.is_file():
                raise ValueError(f'formal input tree contains a non-file entry: {item}')
            relative = item.relative_to(directory).as_posix()
            if normalize_checkpoint_manifest and relative == 'manifest.json':
                normalized_manifest = dict(manifest)
                normalized_manifest.pop('checkpoint_prefix', None)
                digest = hashlib.sha256(json.dumps(normalized_manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            else:
                digest = sha256(item)
            result[relative] = {'bytes': item.stat().st_size, 'sha256': digest}
        return result
    input_identity = {
        'cache_files': tree_identity(cache),
        'checkpoint_files': tree_identity(checkpoint, normalize_checkpoint_manifest=True),
        'quantization_scale_chain_sha256': quant_digest,
    }
    cache_copy_bytes = sum(int(item['bytes']) for item in input_identity['cache_files'].values())
    checkpoint_copy_bytes = sum(int(item['bytes']) for item in input_identity['checkpoint_files'].values())
    session_name: str | None = None
    report_raw_path = os.environ.get('ZKDL_PREFLIGHT_REPORT')
    if report_raw_path:
        report_parent = Path(report_raw_path).parent.resolve()
        try:
            session_name = report_parent.relative_to(root / 'runs').as_posix()
        except ValueError as exc:
            raise ValueError('formal preflight report directory is outside baseline-local runs/') from exc
    if values['checkpoint_bundle'].startswith('runs/'):
        target_checkpoint_prefix = values['checkpoint_bundle'] + '/best_model.pt'
    elif session_name:
        target_checkpoint_prefix = f'runs/{session_name}/resources/checkpoints/{dataset}_gat/best_model.pt'
    else:
        target_checkpoint_prefix = values['checkpoint_bundle'] + '/best_model.pt'
    snapshot_manifest = dict(manifest)
    snapshot_manifest['checkpoint_prefix'] = target_checkpoint_prefix
    manifest_copy_bytes = len((json.dumps(snapshot_manifest, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    checkpoint_manifest_bytes = int(input_identity['checkpoint_files']['manifest.json']['bytes'])
    snapshot_copy_bytes = cache_copy_bytes + checkpoint_copy_bytes - checkpoint_manifest_bytes + manifest_copy_bytes
    result = {'dataset': dataset, 'input_identity': input_identity, 'snapshot_copy_bytes': snapshot_copy_bytes,
              'resource_prerequisites': evidence}
    print(f'formal_input_preflight=PASS dataset={dataset} nodes={nodes} features={features} classes={classes} graphs={graph_count} feature_audit={json.dumps(feature_evidence, sort_keys=True, separators=(",", ":"))} label_audit={json.dumps(label_evidence, sort_keys=True, separators=(",", ":"))} test_index_audit={json.dumps(test_evidence, sort_keys=True, separators=(",", ":"))} split_audit={json.dumps(mask_evidence, sort_keys=True, separators=(",", ":"))} input_identity={json.dumps(input_identity, sort_keys=True, separators=(",", ":"))} resource_prerequisites={json.dumps(evidence, sort_keys=True, separators=(",", ":"))}')
    return result

def main() -> int:
    argparse.ArgumentParser().parse_args()
    root = Path(__file__).resolve().parent.parent
    report: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        try:
            item = check_config(root, dataset)
            item['status'] = 'PASS'
            report[dataset] = item
        except SystemExit as exc:
            detail = exc.code if isinstance(exc.code, str) else f'exit={exc.code}'
            report[dataset] = {'dataset': dataset, 'status': 'FAILED', 'failure': str(detail)}
        except Exception as exc:
            report[dataset] = {'dataset': dataset, 'status': 'FAILED',
                               'failure': f'{type(exc).__name__}: {exc}'}
    report_raw_path = os.environ.get('ZKDL_PREFLIGHT_REPORT')
    if report_raw_path:
        report_path = Path(report_raw_path)
        try:
            report_path.resolve().relative_to(root / 'runs')
        except ValueError as exc:
            raise SystemExit('formal_input_preflight=BLOCKED report path must be baseline-local under runs/') from exc
        if report_path.is_symlink() or report_path.exists():
            raise SystemExit('formal_input_preflight=BLOCKED report path must be fresh and non-symlinked')
        if not report_path.parent.is_dir():
            raise SystemExit('formal_input_preflight=BLOCKED report directory is missing')
        expected_devices = {report_path.parent.stat().st_dev}
        for label, target in (('snapshots', root / 'runs'), ('SRS resources', report_path.parent)):
            try:
                target_device = target.stat().st_dev
            except OSError as exc:
                target_device = None
                shared_failure = f'cannot inspect {label} filesystem: {exc}'
                break
            if target_device not in expected_devices:
                shared_failure = f'{label} are on a different filesystem; combined capacity cannot be certified'
                break
        else:
            shared_failure = ''
        statvfs = os.statvfs(report_path.parent)
        available_bytes = statvfs.f_bavail * statvfs.f_frsize
        if shared_failure:
            for dataset, item in report.items():
                if item.get('status') == 'PASS':
                    item['status'] = 'BLOCKED'
                    item['failure'] = f'shared capacity preflight failed: {shared_failure}'
        report_bytes = 0
        document: dict[str, object] = {}
        for _ in range(16):
            planned = {dataset: dict(item) for dataset, item in report.items()}
            snapshot_bytes = 0
            srs_bytes = 0
            for dataset in DATASETS:
                item = planned[dataset]
                if item.get('status') != 'PASS':
                    continue
                snapshot = int(item['snapshot_copy_bytes'])
                srs = int(item['resource_prerequisites']['srs_plan']['estimated_materialized_bytes'])
                required = snapshot_bytes + srs_bytes + snapshot + srs + report_bytes
                if required > available_bytes:
                    item['status'] = 'FAILED'
                    item['failure'] = (f'combined snapshot/SRS capacity blocked before {dataset}: '
                                       f'need={required} available={available_bytes}')
                else:
                    item['capacity_reservation_bytes'] = {'snapshot_copy_bytes': snapshot,
                                                          'srs_materialized_bytes': srs}
                    snapshot_bytes += snapshot
                    srs_bytes += srs
            required_bytes = snapshot_bytes + srs_bytes + report_bytes
            document = {'schema': 'zkDL_formal_input_preflight_v1', 'datasets': planned,
                        'combined_capacity': {'snapshot_copy_bytes': snapshot_bytes,
                                              'srs_materialized_bytes': srs_bytes,
                                              'preflight_report_bytes': report_bytes,
                                              'required_bytes': required_bytes,
                                              'available_bytes': available_bytes}}
            next_report_bytes = len((json.dumps(document, sort_keys=True, indent=2) + '\n').encode('utf-8'))
            report = planned
            if next_report_bytes == report_bytes:
                break
            report_bytes = next_report_bytes
        else:
            raise SystemExit('formal_input_preflight=BLOCKED capacity report size did not stabilize')
        report_path.write_text(json.dumps(document, sort_keys=True, indent=2) + '\n')
    passed = sum(item.get('status') == 'PASS' for item in report.values())
    failed = sum(item.get('status') != 'PASS' for item in report.values())
    print(f'formal_input_preflight={"PASS" if failed == 0 else "PARTIAL"} datasets=5 ready={passed} failed={failed}')
    if not report_raw_path and failed:
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
