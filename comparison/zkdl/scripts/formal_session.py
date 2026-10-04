#!/usr/bin/env python3
"""Snapshot formal public inputs and create a fresh one-session registration context."""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from asset_manifest import local_path, read_config, reject_source_path_values

DATASETS = ('cora', 'citeseer', 'pubmed', 'ppi', 'ogbn_arxiv')

def session_updates(dataset: str, session: str) -> dict[str, str]:
    if dataset not in DATASETS or not re.fullmatch(r'runs/[A-Za-z0-9_]+', session):
        raise ValueError('formal session/dataset identity is invalid')
    resources = session + '/resources'
    return {'data_root': resources + '/data', 'cache_root': resources + '/cache', 'checkpoint_bundle': resources + '/checkpoints/' + dataset + '_gat', 'export_dir': session + '/' + dataset + '_cpu', 'static_table_descriptor': resources + '/canonical/tables/' + dataset + '_gat_static_tables.json', 'sorted_lookup_descriptor': resources + '/canonical/tables/' + dataset + '_gat_sorted_lookup.json', 'integer_bound_ledger': resources + '/canonical/ledgers/' + dataset + '_gat_integer_ledger.json', 'srs_id': Path(session).name + '_' + dataset + '_formal_srs', 'srs_path': resources + '/srs/' + dataset, 'srs_manifest_digest': 'CEREMONY_REQUIRED', 'model_registry_path': resources + '/verifier_registry/' + dataset + '.json', 'verifier_registry_digest': 'CEREMONY_REQUIRED', 'model_blinder_sidecar_path': resources + '/model_blinders/' + dataset + '.json', 'model_blinder_sidecar_digest': 'CEREMONY_REQUIRED', 'auto_prepare_dataset': 'false', 'dump_trace': 'false'}

def snapshot(root: Path, session: Path, dataset: str, source_config: Path | None = None,
             expected_input_identity: dict[str, object] | None = None) -> Path:
    original = source_config or root / 'configs' / (dataset + '_secure.cfg')
    values = read_config(original)
    reject_source_path_values(values)
    if values.get('dataset') != dataset:
        raise ValueError('formal config/dataset mismatch')
    updates = session_updates(dataset, session.relative_to(root).as_posix())
    cache_source = local_path(root, values['cache_root'], 'cache_root') / dataset
    model_source = local_path(root, values['checkpoint_bundle'], 'checkpoint_bundle')
    for source in (cache_source, model_source):
        if source.is_symlink() or not source.is_dir() or any(path.is_symlink() for path in source.rglob('*')):
            raise ValueError(f'formal snapshot input contains a missing directory or symlink: {source}')
    shutil.copytree(cache_source, root / updates['cache_root'] / dataset)
    shutil.copytree(model_source, root / updates['checkpoint_bundle'])
    manifest_path = root / updates['checkpoint_bundle'] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['checkpoint_prefix'] = updates['checkpoint_bundle'] + '/best_model.pt'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    values.update(updates)
    config_path = root / 'configs' / (session.name + '_' + dataset + '.cfg')
    with config_path.open('x') as stream:
        stream.write(''.join(f'{key} = {value}\n' for key, value in values.items()))
    run = root / updates['export_dir']
    run.mkdir(parents=True)
    (run / 'formal_config.path').write_text(config_path.relative_to(root).as_posix() + '\n')
    if expected_input_identity is not None:
        from formal_preflight import check_config
        snapshot_evidence = check_config(root, dataset, config_path)
        actual_identity = json.loads(json.dumps(snapshot_evidence.get('input_identity')))
        expected_identity = json.loads(json.dumps(expected_input_identity))
        for identity in (actual_identity, expected_identity):
            manifest_identity = identity.get('checkpoint_files', {}).get('manifest.json')
            if isinstance(manifest_identity, dict):
                manifest_identity['bytes'] = None
        if actual_identity != expected_identity:
            raise ValueError(f'formal snapshot input identity differs from successful preflight: {dataset}')
    print(f'formal_session_snapshot=PASS dataset={dataset} config={config_path} run={run}', flush=True)
    return config_path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if os.environ.get('ZKDL_ALLOW_FULL_DATASET_RUN') != '1':
        raise SystemExit('formal-session: full input snapshots belong to the user formal entry')
    root = Path(__file__).resolve().parent.parent
    session = args.session.resolve()
    relative = session.relative_to(root).as_posix()
    if args.session.is_symlink() or not session.is_dir() or not re.fullmatch(r'runs/[A-Za-z0-9_]+', relative) or (session / 'resources').exists():
        raise SystemExit('formal-session: use a fresh baseline-local runs/<session> directory')
    if args.config:
        source_config = args.config.resolve()
        if source_config.parent != root / 'configs' or args.config.is_symlink():
            raise SystemExit('formal-session: source config must be baseline-local')
        dataset = read_config(source_config).get('dataset')
        if dataset not in DATASETS:
            raise SystemExit('formal-session: noncanonical dataset')
        config_path = snapshot(root, session, dataset, source_config)
        (session / 'formal_config.path').write_text(config_path.relative_to(root).as_posix() + '\n')
        (session / 'formal_run.path').write_text(relative + '/' + dataset + '_cpu\n')
    else:
        report_path = session / 'dataset_prerequisites.json'
        if report_path.is_symlink() or not report_path.is_file():
            raise SystemExit('formal-session: successful all-five preflight report is missing, non-regular or symlinked')
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f'formal-session: successful all-five preflight report is missing or invalid: {exc}')
        if report.get('schema') != 'zkDL_formal_input_preflight_v1' or not isinstance(report.get('datasets'), dict):
            raise SystemExit('formal-session: all-five preflight report schema is invalid')
        dataset_reports = report['datasets']
        if set(dataset_reports) != set(DATASETS):
            raise SystemExit('formal-session: all-five preflight report dataset set is incomplete')
        for dataset in DATASETS:
            item = dataset_reports.get(dataset)
            if not isinstance(item, dict) or item.get('dataset') != dataset or not isinstance(item.get('input_identity'), dict):
                raise SystemExit(f'formal-session: preflight input identity is malformed: {dataset}')
            snapshot(root, session, dataset, expected_input_identity=item['input_identity'])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
