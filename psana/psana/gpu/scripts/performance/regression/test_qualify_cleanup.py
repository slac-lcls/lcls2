"""Offline rejection checks for disposable harness; no Slurm invocation."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import qualify_cleanup as q
from native_contract import write


class CleanupTests(unittest.TestCase):
    def test_identity_rejects_wrong_allocation_and_mixed_steps(self):
        for jobs, steps in ((['123','456'],['1','1']), (['123','123'],['1','2']),
                            (['123','123'],['batch','batch'])):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                for rank in range(2):
                    write(root / ('ready-%d.json' % rank), dict(rank=rank, job=jobs[rank], step=steps[rank]))
                with self.assertRaises(ValueError):
                    q.identity(root, '123', 2)

    def test_wait_rejects_dead_launcher_before_readiness(self):
        with patch.object(q.subprocess, 'Popen') as process:
            process.poll.return_value = 1
            with self.assertRaisesRegex(ValueError, 'launcher exited'):
                q.wait_for(lambda: False, 1, 'ready', process)

    def test_controller_rejects_non_timeout_exit(self):
        with tempfile.TemporaryDirectory() as name:
            request = Path(name) / 'request.json'
            write(request, {})
            with patch.object(q, 'run_command', return_value=1):
                with self.assertRaisesRegex(RuntimeError, 'did not time out'):
                    q.child('_controller', str(request), str(Path(name) / 'target'))
            self.assertFalse((Path(name) / 'controller-timeout.json').exists())


if __name__ == '__main__':
    unittest.main()
