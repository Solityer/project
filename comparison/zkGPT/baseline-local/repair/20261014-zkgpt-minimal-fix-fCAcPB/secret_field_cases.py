import os
import sys
import tempfile
from pathlib import Path

repo = Path('/home/pzh/project/comparison/zkGPT')
sys.path.insert(0, str(repo / 'scripts'))
from common import ChainError, parse_prover_secret_metadata

check_root = Path('/home/pzh/project/comparison/zkGPT/baseline-local/repair/20261014-zkgpt-minimal-fix-fCAcPB')
base = ['zkgpt-gat-com-w-secret-v1', '1', 'a' * 64, 'b' * 64, '3', '4', 'c' * 64]
cases = [
    ('zero', '0', True),
    ('native_short_scalar', 'f' * 63, True),
    ('redundant_leading_zero', '01', False),
    ('native_modulus', '30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001', False),
    ('uppercase_hex', 'A', False),
    ('oversized_hex', '1' * 65, False),
]
with tempfile.TemporaryDirectory(prefix='secret-cases-', dir=check_root) as directory:
    for name, blinder, expected_valid in cases:
        path = Path(directory) / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='ascii') as stream:
            stream.write(' '.join(base + [blinder]) + '\n')
        os.chmod(path, 0o600)
        try:
            parsed = parse_prover_secret_metadata(path)
            accepted = True
            if parsed['registration_digest'] != 'a' * 64:
                raise AssertionError('public identity prefix changed')
        except ChainError:
            accepted = False
        if accepted != expected_valid:
            raise AssertionError('canonical blinder case failed: ' + name)
        print('SECRET_FIELD_CASE name=' + name + ' result=PASS')
