import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import perlmutter_screen as screen
from perlmutter_support import validate_rows, rank_environment

class PerlmutterTests(unittest.TestCase):
    def rows(self):
        return [dict(rank=i,local_rank=i,role=['smd0','eb','bd','bd'][i],physical_cores=[['0',str(i)]],mpi_library='CRAY MPICH',gpu_uuid='GPU-one' if i>1 else None) for i in range(4)]
    def test_rank_binding_and_device_mismatch(self):
        validate_rows(self.rows(),'gpu')
        for key,value in [('local_rank',0),('role','eb'),('gpu_uuid','GPU-two'),('physical_cores',[['0','0']]),('mpi_library','Open MPI')]:
            rows=self.rows(); rows[-1][key]=value
            with self.subTest(key=key), self.assertRaises(RuntimeError): validate_rows(rows,'gpu')
    def test_cpu_role_isolation(self):
        rows=self.rows(); rows[0]['gpu_uuid']='GPU-one'
        with self.assertRaises(RuntimeError): validate_rows(rows,'gpu')
    def test_rank_must_be_single_node(self):
        with patch.dict('os.environ',dict(SLURM_PROCID='2',SLURM_LOCALID='1')):
            with self.assertRaises(RuntimeError): rank_environment()
    def test_missing_review_prevents_submission_path(self):
        with tempfile.TemporaryDirectory() as d, patch.object(screen,'ROOT',Path(d)):
            with self.assertRaises(FileNotFoundError): screen.review_gate('pilot')
    def test_review_manifest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(screen,'ROOT',Path(d)):
            p=Path(d); (p/'manifest.json').write_text('{}')
            digest=screen.file_hash(p/'manifest.json')
            (p/'review.json').write_text(json.dumps(dict(approved_phase='pilot',reviewed_manifest_sha256='wrong',reviewed_commit=screen.HEAD,verdict='pass',findings=[])))
            (p/'review-approval-pilot.json').write_text(json.dumps(dict(approved_phase='pilot',manifest_sha256=digest,review_json=str(p/'review.json'))))
            with self.assertRaisesRegex(RuntimeError,'different manifest'): screen.review_gate('pilot')
    def test_pilot_requires_completion_and_late_rate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'sample.json').write_text(json.dumps(dict(events=10000,complete=True,mpi_process_wall_seconds=160)))
            (p/'progress-rank-2.jsonl').write_text('\n'.join(json.dumps(dict(events=e,seconds=t)) for e,t in [(2500,25),(5000,50),(7500,100),(10000,150)]))
            value=screen.pilot_estimate(p)
            self.assertEqual(value['late_events_per_second'],50)
            self.assertEqual(value['stability_ratio'],0.5)
            self.assertGreaterEqual(value['timeout_seconds'],660)
            (p/'sample.json').write_text(json.dumps(dict(events=9900,complete=True)))
            with self.assertRaises(RuntimeError): screen.pilot_estimate(p)
    def test_runtime_env_replaces_old_python(self):
        with patch.dict('os.environ',dict(PYTHONPATH='/old/install',SCREEN_RUNTIME_LIBS='/cuda/lib',SIT_PSDM_OFFSITE='1')):
            env,install=screen.environment('base','GPU-one')
            self.assertNotIn('/old/install',env['PYTHONPATH'])
            self.assertNotIn('SIT_PSDM_OFFSITE',env)
            self.assertEqual(env['KVIKIO_COMPAT_MODE'],'ON')


class ReviewFixTests(unittest.TestCase):
    def test_pilot_review_cannot_approve_campaign(self):
        with tempfile.TemporaryDirectory() as d, patch.object(screen,'ROOT',Path(d)):
            p=Path(d); (p/'manifest.json').write_text('{}'); digest=screen.file_hash(p/'manifest.json')
            review=dict(approved_phase='pilot',reviewed_manifest_sha256=digest,reviewed_commit=screen.HEAD,verdict='pass',findings=[],pilot_evidence_sha256='')
            (p/'review.json').write_text(json.dumps(review))
            (p/'review-approval-campaign.json').write_text(json.dumps(dict(approved_phase='campaign',manifest_sha256=digest,review_json=str(p/'review.json'))))
            with self.assertRaisesRegex(RuntimeError,'phase mismatch'): screen.review_gate('campaign')
    def test_campaign_rejects_modified_pilot_evidence(self):
        with tempfile.TemporaryDirectory() as d, patch.object(screen,'ROOT',Path(d)), patch.object(screen,'pilot_evidence_digest',return_value='new'):
            p=Path(d); (p/'manifest.json').write_text('{}'); digest=screen.file_hash(p/'manifest.json')
            (p/'review.json').write_text(json.dumps(dict(approved_phase='campaign',reviewed_manifest_sha256=digest,pilot_evidence_sha256='old')))
            (p/'review-approval-campaign.json').write_text(json.dumps(dict(approved_phase='campaign',manifest_sha256=digest,review_json=str(p/'review.json'),pilot_evidence_sha256='old')))
            with self.assertRaisesRegex(RuntimeError,'pilot evidence'): screen.review_gate('campaign')
    def test_timeout_terminates_real_child_group(self):
        import os, signal, subprocess, sys
        with tempfile.TemporaryDirectory() as d, patch.object(screen.os,'killpg',wraps=os.killpg) as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                screen.run_command([sys.executable,'-c','import time; time.sleep(60)'],dict(os.environ),Path(d)/'child.log',0.1)
            self.assertEqual(kill.call_args.args[1],signal.SIGTERM)
    def test_launch_failure_preserves_status_and_stops_monitors(self):
        import subprocess
        from unittest.mock import Mock
        gpu=Mock()
        with tempfile.TemporaryDirectory() as d, patch.object(screen,'environment',return_value=({},Path('/install'))), patch.object(screen,'Monitor') as monitor, patch.object(screen.subprocess,'Popen',return_value=gpu), patch.object(screen,'run_command',side_effect=subprocess.TimeoutExpired('test',1)):
            # Command evidence is constructed before the launch.
            env={k:'test' for k in ['PYTHONPATH','LD_LIBRARY_PATH','KVIKIO_COMPAT_MODE','KVIKIO_NTHREADS','KVIKIO_TASK_SIZE','SCREEN_GPU_UUID','PS_EB_NODES','PS_SRV_NODES','PS_PARALLEL']}
            with patch.object(screen,'environment',return_value=(env,Path('/install'))):
                with self.assertRaises(subprocess.TimeoutExpired):
                    screen.launch(dict(bds=1),dict(runtime='base',mode='gpu',iterator='events'),'failure',120,1,Path(d),'GPU-one',{})
            status=json.loads((Path(d)/'failure/process-status.json').read_text())
            self.assertFalse(status['complete']); self.assertIsNone(status['mpi_process_wall_seconds'])
            monitor.return_value.stop.assert_called_once(); gpu.terminate.assert_called_once()
    def test_insufficient_walltime_stops_before_launch(self):
        with patch.object(screen,'remaining_allocation_seconds',return_value=600):
            with self.assertRaisesRegex(RuntimeError,'Insufficient allocation time'): screen.require_remaining(600)

if __name__=='__main__': unittest.main()
