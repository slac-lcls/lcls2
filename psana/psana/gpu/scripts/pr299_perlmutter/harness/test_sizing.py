import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import perlmutter_screen as screen

class SizingTests(unittest.TestCase):
    def sample(self,**changes):
        result=dict(complete=True,events=10000,runtime='base',preflight=False,mpi_process_wall_seconds=160.04,loop_seconds=148.1,label='full')
        result.update(changes)
        return result
    def test_complete_bound(self):
        self.assertEqual(screen.case_sizing(self.sample())['timeout_seconds'],541)
    def test_incomplete_short_head_or_invalid_time_rejected(self):
        for changes in [dict(complete=False),dict(events=120),dict(runtime='head'),dict(preflight=True),dict(mpi_process_wall_seconds=0),dict(mpi_process_wall_seconds=float('nan'))]:
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):screen.case_sizing(self.sample(**changes))
    def test_discovery_uses_one_gpu_step_and_saves_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(screen.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'{"complete":true,"gpu_uuid":"test"}','')) as run:
                result=screen.discover_gpu(Path(temp))
            argv=run.call_args[0][0]
            for argument in ['--ntasks=1','--gpus-per-node=1','--exact','--mpi=none']:self.assertIn(argument,argv)
            self.assertEqual(result['gpu_uuid'],'test')
            self.assertEqual(json.loads((Path(temp)/'gpu-identity-command.json').read_text())['returncode'],0)
    def test_discovery_timeout_preserves_partial_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(screen.subprocess,'run',side_effect=subprocess.TimeoutExpired(['srun'],45,output=b'partial',stderr=b'error')):
                with self.assertRaises(subprocess.TimeoutExpired):screen.discover_gpu(Path(temp))
            evidence=json.loads((Path(temp)/'gpu-identity-command.json').read_text())
            self.assertTrue(evidence['timed_out']);self.assertEqual(evidence['stdout'],'partial');self.assertEqual(evidence['stderr'],'error')

if __name__=='__main__':unittest.main()
