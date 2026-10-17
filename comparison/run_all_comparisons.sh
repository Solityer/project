import importlib.util
import fcntl
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('serial_formal', Path(__file__).resolve().parents[1] / 'scripts/serial_formal.py')
serial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serial)


class SerialFormalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'comparison'
        self.root.mkdir()
        for key in serial.BASELINES:
            location = self.root / ('zkGPT' if key == 'zkgpt' else key)
            (location / 'scripts').mkdir(parents=True)
            (location / serial.command_for(key)[1]).write_text('# mock official launcher\n')
        (self.root / 'runs' / 'serial_comparisons').mkdir(parents=True)
        self.sid = 'a' * 32
        self.session_dir = self.root / 'runs' / 'serial_comparisons' / self.sid
        self.session_dir.mkdir()
        self.state = serial.blank_state(self.sid)
        serial.write_json(self.session_dir / 'status.json', self.state)
        self.patch_root = mock.patch.object(serial, 'ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def _write(self, rel, payload):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def test_official_command_order(self):
        self.assertEqual(serial.BASELINES, ('ezkl', 'zkdl', 'zkgpt'))
        self.assertEqual(serial.DATASETS, ('cora', 'citeseer', 'pubmed', 'ppi', 'ogbn_arxiv'))
        self.assertEqual(serial.command_for('ezkl')[1:], ['scripts/ezkl_formal_control.py', 'start'])
        self.assertEqual(serial.command_for('zkdl')[1:], ['scripts/run_five_detached.sh'])
        self.assertEqual(serial.command_for('zkgpt')[1:], ['scripts/run_formal_detached.sh'])

    def test_failed_dataset_does_not_abort_other_baselines(self):
        started = []

        def launched(name, row):
            started.append(name)
            return {'name': name, 'launch_id': 'fake-' + name, 'dir': self.root}

        def snap(job, row):
            cells = {dataset: ('PASS', '') for dataset in serial.DATASETS}
            if job['name'] == 'ezkl':
                cells['citeseer'] = ('FAILED', 'mock proof failure')
                return True, 2, cells
            return True, 0, cells

        with mock.patch.object(serial, 'launch', side_effect=launched), \
             mock.patch.object(serial, 'snapshot', side_effect=snap), \
             mock.patch.object(serial, 'safe_to_transition', return_value=True):
            ret = serial.run_session(self.session_dir, self.state)
        self.assertEqual(ret, 0)
        self.assertEqual(started, ['ezkl', 'zkdl', 'zkgpt'])
        self.assertEqual(self.state['status'], 'COMPLETE_WITH_FAILURES')
        self.assertEqual(self.state['baselines']['ezkl']['datasets']['citeseer']['status'], 'FAILED')
        self.assertEqual(self.state['baselines']['zkgpt']['datasets']['ogbn_arxiv']['status'], 'PASS')

    def test_unsafe_native_process_prevents_next_launch(self):
        started = []

        def launched(name, row):
            started.append(name)
            return {'name': name, 'launch_id': 'mock-' + name, 'dir': self.root}

        with mock.patch.object(serial, 'launch', side_effect=launched), \
             mock.patch.object(serial, 'snapshot', return_value=(True, 2, {'cora': ('FAILED', 'intentional')})), \
             mock.patch.object(serial, 'safe_to_transition', return_value=False), \
             mock.patch.object(serial, 'DRAIN_GRACE_SECONDS', -1), \
             mock.patch.object(serial, 'POLL_SECONDS', 0):
            ret = serial.run_session(self.session_dir, self.state)
        self.assertEqual(ret, 2)
        self.assertEqual(started, ['ezkl'])
        self.assertEqual(self.state['status'], 'HALTED')
        self.assertIn('did not drain', self.state['failure'])

    def test_ezkl_completed_and_failed_cells_from_exact_session(self):
        sid = '1' * 32
        self._write(f'ezkl/runs/formal/.sessions/{sid}/controller.json', {
            'schema': 'ezkl.formal-controller.v1', 'session_id': sid, 'status': 'EXITED', 'exit_code': 2})
        self._write(f'ezkl/runs/formal/.sessions/{sid}.json', {
            'baseline': 'ezkl', 'session_id': sid, 'completed_cells': {'cora': {'proof': 'verified'}},
            'dataset_failures': {'citeseer': {'error': 'prover failed'}}, 'active_dataset': None})
        job = {'name': 'ezkl', 'launch_id': sid, 'dir': self.root / 'ezkl/runs/formal/.sessions' / sid}
        record = {}
        terminal, code, cells = serial.snapshot(job, record)
        self.assertTrue(terminal)
        self.assertEqual(code, 2)
        self.assertEqual(cells['cora'][0], 'PASS')
        self.assertEqual(cells['citeseer'], ('FAILED', 'prover failed'))
        self.assertNotIn('pubmed', cells)

    def test_zkdl_session_link_binding_and_failure(self):
        sid = 'launch_20261017T090000Z_ABCDEFGH'
        native = 'b' * 32
        ld = self.root / 'zkdl' / 'runs' / sid
        self._write(f'zkdl/runs/{sid}/exit.json', {
            'schema': 'zkdl_detached_launch_exit_v1', 'status': 'FAILED', 'exit_code': 2})
        self._write(f'zkdl/runs/{sid}/formal_session.json', {
            'launch_id': sid, 'session_id': native,
            'session_dir': str(self.root / 'zkdl/runs' / ('formal_' + native))})
        self._write(f'zkdl/runs/formal_{native}/state.json', {
            'session_id': native, 'baseline': 'zkdl',
            'dataset_status': {'cora': {'status': 'FAILED', 'failure': 'native panic'},
                               'citeseer': {'status': 'PASS', 'failure': ''}}})
        terminal, code, cells = serial.snapshot({'name': 'zkdl', 'launch_id': sid, 'dir': ld}, {})
        self.assertTrue(terminal)
        self.assertEqual(code, 2)
        self.assertEqual(cells['cora'], ('FAILED', 'native panic'))
        self.assertEqual(cells['citeseer'], ('PASS', ''))

    def test_zkgpt_exact_owner_binding_and_receipt(self):
        sid = 'a' * 32
        native = 'c' * 32
        self._write(f'zkGPT/baseline-local/formal-detached/{sid}/status.json', {
            'schema': 'zkgpt-gat-detached-launch-v2', 'launch_id': sid, 'status': 'COMPLETE',
            'exit_code': 0, 'child_pid': 12345, 'child_start_ticks': 56789})
        self._write('zkGPT/baseline-local/formal-owner.json', {
            'session_id': native, 'owner_pid': 12345, 'owner_start_ticks': 56789})
        self._write(f'zkGPT/results/formal/{native}/session_status.json', {
            'session_id': native, 'schema': 'zkgpt-gat-formal-session-v1',
            'stage_status': {'cell:cora': 'PASS', 'cell:citeseer': 'FAILED'},
            'dataset_failures': {'citeseer': 'mock failed'}})
        self._write(f'zkGPT/results/formal/{native}/receipts/cora.json', {
            'session_id': native, 'dataset': 'cora', 'status': 'VERIFY_OK', 'independent_verification': True})
        terminal, code, cells = serial.snapshot({
            'name': 'zkgpt', 'launch_id': sid,
            'dir': self.root / 'zkGPT/baseline-local/formal-detached' / sid}, {})
        self.assertTrue(terminal)
        self.assertEqual(code, 0)
        self.assertEqual(cells['cora'], ('PASS', ''))
        self.assertEqual(cells['citeseer'], ('FAILED', 'mock failed'))

    def test_native_flock_blocks_transition(self):
        lock = self.root / serial.LOCKS['ezkl']
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('w+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(serial.all_native_locks_idle())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        self.assertTrue(serial.all_native_locks_idle())

    def test_stray_live_group_blocks_transition_without_leader(self):
        # An orphaned native process group must block handoff even when its
        # recorded leader PID has exited or was reaped.
        group = os.getpgrp()
        self.assertFalse(serial.safe_to_transition({'child': {'pid': -1, 'pgid': group}}))

    def test_missing_terminal_receipt_cannot_authorize_next_run(self):
        job = {'name': 'ezkl', 'launch_id': 'd' * 32,
               'dir': self.root / 'ezkl/runs/formal/.sessions' / ('d' * 32)}
        terminal, exit_code, cells = serial.snapshot(job, {})
        self.assertFalse(terminal)
        self.assertIsNone(exit_code)
        self.assertEqual(cells, {})

    def test_wrong_launch_link_rejected(self):
        launch = 'launch_20261017T090000Z_ABCDEFGH'
        native = 'b' * 32
        self._write(f'zkdl/runs/{launch}/formal_session.json', {
            'launch_id': 'another-launch', 'session_id': native,
            'session_dir': str(self.root / 'zkdl/runs' / ('formal_' + native))})
        with self.assertRaises(serial.UnsafeTransition):
            serial.snapshot({'name': 'zkdl', 'launch_id': launch,
                             'dir': self.root / 'zkdl/runs' / launch}, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)

# The tests above intentionally mock the long-running proof engines; they do not
# generate proofs or validate GPU runtime behavior.
