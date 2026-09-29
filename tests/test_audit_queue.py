import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('run_audit_queue', ROOT / 'tools/run_audit_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class AuditQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'runs', prefix='audit-queue-test-')
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to((ROOT / 'runs').resolve())

    def tearDown(self):
        self.temporary.cleanup()

    def write_status(self, state, seeds=None):
        record = {'state': state}
        if seeds is not None:
            record['completed_seeds'] = seeds
        (self.root / 'status.json').write_text(json.dumps(record))

    def test_live_producer_blocks_even_with_complete_status(self):
        self.write_status('complete', [0, 1, 2])
        (self.root / 'controller.lock').write_text(str(os.getpid()))
        self.assertFalse(queue.producer_ready(self.root, [0, 1, 2]))

    def test_stopped_incomplete_paused_failed_or_stop_are_rejected(self):
        for state in ('running', 'failed', 'paused'):
            self.write_status(state)
            with self.assertRaises(InterruptedError):
                queue.producer_ready(self.root, [0, 1, 2])
        self.write_status('complete', [0, 1, 2])
        (self.root / 'STOP').write_text('stop')
        with self.assertRaises(InterruptedError):
            queue.producer_ready(self.root, [0, 1, 2])

    def test_expected_seeds_required_once(self):
        for seeds in ([0, 1], [0, 1, 2, 2], [0, 1, 3]):
            self.write_status('complete', seeds)
            with self.assertRaises(ValueError):
                queue.producer_ready(self.root, [0, 1, 2])
        self.write_status('complete', [2, 0, 1])
        self.assertTrue(queue.producer_ready(self.root, [0, 1, 2]))

    def test_existing_reports_preserved(self):
        first = queue.reserve_report(self.root / 'report')
        (first / 'evidence.json').write_text('existing evidence')
        second = queue.reserve_report(self.root / 'report')
        self.assertNotEqual(first, second)
        self.assertEqual((first / 'evidence.json').read_text(), 'existing evidence')


if __name__ == '__main__':
    unittest.main()
