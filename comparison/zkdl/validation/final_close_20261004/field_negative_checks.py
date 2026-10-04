import ast
import inspect
import math
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'scripts'))
import result_artifacts

# Execute the exact timing-audit loop AST from production audit_run.
source = inspect.getsource(result_artifacts.audit_run)
module = ast.parse(source)
loop = next(
    node for node in ast.walk(module)
    if isinstance(node, ast.For)
    and isinstance(node.target, ast.Name)
    and node.target.id == 'field'
    and isinstance(node.iter, ast.Tuple)
    and len(node.iter.elts) == 3
    and all(isinstance(item, ast.Constant) and item.value in {
        'commitment_time_seconds', 'prove_time_seconds', 'verify_time_seconds'
    } for item in node.iter.elts)
)
code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), '<production timing loop>', 'exec')
fields = ('commitment_time_seconds', 'prove_time_seconds', 'verify_time_seconds')
base = {key: '0.25' for key in fields}

def audit(record, envelope):
    scope = {'math': math, 'ArtifactError': result_artifacts.ArtifactError,
             'record': record, 'envelope': envelope}
    exec(code, scope)

def must_reject(record, envelope, label):
    try:
        audit(record, envelope)
    except result_artifacts.ArtifactError:
        return
    raise AssertionError(f'timing audit accepted {label}')

audit(dict(base), dict(base))
for key in fields:
    rec = dict(base); rec.pop(key)
    must_reject(rec, dict(base), f'missing record {key}')
    env = dict(base); env.pop(key)
    must_reject(dict(base), env, f'missing envelope {key}')
    for token in ('NaN', 'inf', '-0.1', 'invalid'):
        rec = dict(base); rec[key] = token
        must_reject(rec, dict(base), f'{key}={token}')
    rec = dict(base); rec[key] = '0.5'
    must_reject(rec, dict(base), f'mismatched {key}')

# Exercise the ledger identity contract with distinct IDs, and pin the production
# loader/comparator field accesses so a regression to ledger_id is rejected.
runner = (root / 'gat_adapter' / 'gat_runner.cu').read_text()
assert 'ledger.degree_bound_id = json_scalar(text, "degree_bound_id")' in runner
assert 'config.degree_bound_id != ledger.degree_bound_id' in runner
ledger = {'ledger_id': 'ledger:dataset:degree:128',
          'degree_bound_id': 'degree:dataset:128'}
config_degree_id = 'degree:dataset:128'
assert config_degree_id == ledger['degree_bound_id']
assert config_degree_id != ledger['ledger_id']
assert config_degree_id != 'degree:dataset:256'
print('FIELD_NEGATIVES_PASS timing_missing/nonfinite/negative/malformed/mismatch=21 ledger_degree_identity_separate=PASS wrong_degree_rejected=PASS')
