import os
import runpy
import sys
from pathlib import Path

repo = Path('/home/pzh/project/comparison/zkGPT')
check_root = Path('/home/pzh/project/comparison/zkGPT/baseline-local/repair/20261014-zkgpt-minimal-fix-fCAcPB')
sys.path.insert(0, str(repo / 'scripts'))
import common

isolated_verifier = check_root / 'cmake' / 'gat_adapter' / 'zkgpt_gat_verifier'
if not isolated_verifier.is_file() or not os.access(isolated_verifier, os.X_OK):
    raise SystemExit('isolated verifier binary is unavailable')
common.verifier_binary = lambda root, backend: isolated_verifier
os.environ['ZKGPT_PREFLIGHT_SMOKE_BINARY'] = 'baseline-local/repair/20261014-zkgpt-minimal-fix-fCAcPB/cmake/tests/gat_adapter_smoke'
os.environ['ZKGPT_PREFLIGHT_TINY_TIMEOUT_SECONDS'] = '300'
sys.argv = [
    str(repo / 'tests' / 'check_formal_closure.py'),
    '--backend', 'cpu',
    '--retain-dir', 'baseline-local/repair/20261014-zkgpt-minimal-fix-fCAcPB/tiny-evidence',
]
runpy.run_path(str(repo / 'tests' / 'check_formal_closure.py'), run_name='__main__')
