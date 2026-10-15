#!/usr/bin/env python3
"""Build and authenticate the baseline-local formal binaries and their sources."""
from __future__ import annotations
import argparse
import hashlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from session_lifecycle import (
    SessionRootError,
    require_session_artifact,
    require_session_root,
)

ROOT = Path(__file__).resolve().parent.parent

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()

def source_entries(root: Path) -> list[dict[str, object]]:
    paths = {root / 'Makefile', root / 'configs/quantization_scale_chain.cfg'}
    for pattern in ('*.cu', '*.cuh', '*.hpp', '*.h'):
        paths.update(root.glob(pattern))
    for directory in ('gat_adapter', 'scripts'):
        paths.update(path for path in (root / directory).rglob('*') if path.suffix in ('.cu', '.cuh', '.hpp', '.h', '.py', '.sh'))
    paths.add(root / 'third_party/BLST_PROVENANCE.txt')
    vendor = root / 'third_party/blst'
    if vendor.is_dir():
        paths.update(path for path in vendor.rglob('*') if path.is_file() and (path.suffix in ('.c', '.h', '.S', '.s', '.asm', '.inc', '.pl') or path.name in ('LICENSE', 'NOTICE', 'VERSION', 'build.sh', 'UPSTREAM.json')))
    entries = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'formal source is missing or symlinked: {path}')
        entries.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    return entries

def identity(entries: list[dict[str, object]]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def validate_manifest(root: Path, path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'build manifest is missing or symlinked: {path}')
    value = json.loads(path.read_text())
    if value.get('schema') != 'zkDL_gat_build_manifest_v1' or value.get('backend') != 'cpu' or value.get('build_target') != 'baseline-local-build':
        raise ValueError('formal build manifest schema/backend/target mismatch')
    entries = source_entries(root)
    if value.get('sources') != entries or value.get('source_digest') != identity(entries):
        raise ValueError('formal sources differ from the recorded build')
    binaries = value.get('binaries')
    if not isinstance(binaries, list) or len(binaries) != 2 or {item.get('name') for item in binaries} != {'gat_runner', 'gat_resource_setup'}:
        raise ValueError('formal build manifest binary set mismatch')
    for item in binaries:
        binary = (root / item['path']).resolve()
        binary.relative_to(root)
        if binary.name != item['name'] or binary.parent.name != 'build' or binary.parent.parent.parent != root / 'runs':
            raise ValueError('formal binary is not a session snapshot')
        if binary.is_symlink() or not binary.is_file() or binary.stat().st_size != item.get('bytes') or digest(binary) != item.get('sha256'):
            raise ValueError(f'formal binary differs from the recorded build: {binary}')
    return value

def native_binary(root: Path, name: str) -> Path:
    raw = os.environ.get('ZKDL_BUILD_MANIFEST')
    if not raw:
        raise ValueError('a matched session build manifest is required; use the formal entry')
    path = Path(raw).resolve()
    path.relative_to(root)
    value = validate_manifest(root, path)
    return root / next(item['path'] for item in value['binaries'] if item['name'] == name)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--binary', choices=('gat_runner', 'gat_resource_setup'))
    args = parser.parse_args()
    if bool(args.output) == bool(args.check):
        parser.error('supply exactly one of --output or --check')
    path = Path(os.path.abspath(args.output or args.check))
    session_identity = require_session_artifact(path, ROOT)
    path = path.resolve(strict=False)
    path.relative_to(ROOT)
    if args.check:
        if session_identity is not None:
            require_session_root(session_identity)
        value = validate_manifest(ROOT, path)
        if session_identity is not None:
            require_session_root(session_identity)
        if args.binary:
            print(ROOT / next(item['path'] for item in value['binaries'] if item['name'] == args.binary))
        else:
            print(f'build_provenance=PASS manifest={path}')
        return 0
    with (ROOT / '.formal_build.lock').open('a') as build_lock:
        fcntl.flock(build_lock.fileno(), fcntl.LOCK_EX)
        if session_identity is not None:
            require_session_root(session_identity)
        before = source_entries(ROOT)
        build_env = dict(os.environ)
        for variable in ('MAKEFLAGS', 'MFLAGS', 'GNUMAKEFLAGS'):
            build_env.pop(variable, None)
        subprocess.run(['make', '-B', 'baseline-local-build'], cwd=ROOT, env=build_env, check=True)
        after = source_entries(ROOT)
        if before != after:
            raise SystemExit('formal build sources changed during compilation')
        if session_identity is not None:
            require_session_root(session_identity)
        binaries = []
        snapshot_dir = path.parent / 'build'
        if snapshot_dir.parent.is_symlink() or not snapshot_dir.parent.is_dir():
            raise SystemExit(f'formal session root disappeared before build snapshot: {snapshot_dir.parent}')
        snapshot_dir.mkdir()
        for name in ('gat_runner', 'gat_resource_setup'):
            if session_identity is not None:
                require_session_root(session_identity)
            binary = ROOT / name
            if binary.is_symlink() or not binary.is_file():
                raise SystemExit(f'formal build did not produce a regular binary: {binary}')
            frozen = snapshot_dir / name
            shutil.copy2(binary, frozen)
            if session_identity is not None:
                require_session_root(session_identity)
            binaries.append({'name': name, 'path': frozen.relative_to(ROOT).as_posix(), 'bytes': frozen.stat().st_size, 'sha256': digest(frozen)})
        value = {'schema': 'zkDL_gat_build_manifest_v1', 'backend': 'cpu', 'build_target': 'baseline-local-build', 'source_digest': identity(after), 'sources': after, 'binaries': binaries}
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise SystemExit(f'formal session root disappeared before build manifest write: {path.parent}')
        if session_identity is not None:
            require_session_root(session_identity)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        if session_identity is not None:
            require_session_root(session_identity)
        validate_manifest(ROOT, path)
        print(f'build_provenance=PASS source_digest={value["source_digest"]} manifest={path}')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SessionRootError as exc:
        print(f'formal-provenance: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(2)
