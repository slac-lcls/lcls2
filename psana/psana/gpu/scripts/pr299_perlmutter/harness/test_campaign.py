from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import perlmutter_screen as screen

class PilotExecutionTests(unittest.TestCase):
    def execute(self,root,fail_case=None):
        manifest=dict(cases=[c for c in screen.cases() if c['run']==51])
        (root/'results').mkdir();(root/'manifest.json').write_text('{}')
        observed=[]
        def launch(case,variant,label,count,timeout,*args,**kwargs):
            observed.append((case['name'],variant['runtime'],label,count,timeout))
            if count==10000 and case['name']==fail_case:raise RuntimeError('simulated sample failure')
            return dict(complete=True,events=count,runtime=variant['runtime'],preflight=count==120,mpi_process_wall_seconds=100,loop_seconds=90,label=label)
        with ExitStack() as stack:
            for name,value in [('ROOT',root),('check_frozen',lambda:manifest),('review_gate',lambda phase:None),('allocation_identity',lambda phase:{}),('discover_gpu',lambda result:dict(gpu_uuid='GPU-test')),('inventory',lambda:dict(lustre_stats_observable=False,scope='test')),('require_remaining',lambda seconds:None),('launch',launch),('pilot_estimate',lambda sample:dict(timeout_seconds=500,stability_ratio=1))]:
                stack.enter_context(patch.object(screen,name,value))
            stack.enter_context(patch.object(screen.signal,'signal'))
            stack.enter_context(patch.dict('os.environ',dict(SLURM_JOB_ID='123')))
            stack.enter_context(patch.object(screen.subprocess,'check_output',return_value='{"filesystems":[{"fstype":"lustre"}]}'))
            screen.execute('pilot')
        return observed
    def test_six_sustained_cases_and_merged_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls=self.execute(root)
            timed=[call for call in calls if call[3]==10000]
            self.assertEqual(len(calls),18);self.assertEqual(len(timed),6)
            self.assertTrue(all(call[1]=='base' and call[4]==600 for call in timed))
            self.assertEqual(timed[0][2],'sustained-base-gpu-bd1')
            estimate=json.loads((root/'pilot-estimate.json').read_text())
            self.assertEqual(len(estimate['case_estimates']),6)
            self.assertEqual(estimate['case_estimates']['r51-gpu-events-bd1']['timeout_seconds'],500)
            self.assertTrue(json.loads((root/'results/pilot-job-123/status.json').read_text())['complete'])
    def test_mid_sequence_failure_writes_no_estimate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(RuntimeError,'simulated'):self.execute(root,fail_case='r51-cpu-steps-bd1')
            self.assertFalse((root/'pilot-estimate.json').exists())
            self.assertFalse(json.loads((root/'results/pilot-job-123/status.json').read_text())['complete'])

class CampaignGuardTests(unittest.TestCase):
    def fixture(self,root):
        directory=root/'results/pilot-job-123';directory.mkdir(parents=True)
        estimate=dict(sample_directory=str(directory/'sustained-base-gpu-bd1'),case_estimates={})
        plan=dict(cases={},rounds=4,timed_samples=48,sizing_result_directory=str(directory),allocation_minutes=120)
        for i,case in enumerate(c for c in screen.cases() if c['run']==51):
            label='sustained-base-gpu-bd1' if i==0 else 'sample'+str(i)
            sample=directory/label/'sample.json';sample.parent.mkdir()
            sample.write_text(json.dumps(dict(case=case['name'],complete=True,events=10000,runtime='base',preflight=False,mpi_process_wall_seconds=100,loop_seconds=90,label=label)))
            estimate['case_estimates'][case['name']]=dict(measured_mpi_wall_seconds=100,timeout_seconds=360,sample_label=label)
            plan['cases'][case['name']]=dict(timeout_seconds=360,source_sample=str(sample),source_sample_sha256=screen.file_hash(sample))
        (directory/'status.json').write_text('{"phase":"pilot","complete":true}')
        (root/'pilot-estimate.json').write_text(json.dumps(estimate))
        return estimate,plan,directory
    def test_old_single_case_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'six-case'):screen.validate_sizing_evidence({},Path(temp))
    def test_completed_six_case_digest_changes_with_results(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(screen,'ROOT',Path(temp)):
            _,_,directory=self.fixture(Path(temp));before=screen.pilot_evidence_digest()
            (directory/'extra-evidence.json').write_text('{}')
            self.assertNotEqual(before,screen.pilot_evidence_digest())
    def test_case_plan_bounds_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(screen,'ROOT',Path(temp)):
            _,plan,directory=self.fixture(Path(temp));self.assertEqual(screen.campaign_plan(dict(campaign_plan=plan)),plan)
            item=plan['cases']['r51-gpu-events-bd1'];item['timeout_seconds']=359
            with self.assertRaisesRegex(RuntimeError,'below sizing'):screen.campaign_plan(dict(campaign_plan=plan))
            item['timeout_seconds']=360;item['source_sample_sha256']='wrong'
            with self.assertRaisesRegex(RuntimeError,'provenance'):screen.campaign_plan(dict(campaign_plan=plan))
    def test_campaign_allocation_matches_reviewed_minutes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(screen,'ROOT',Path(temp)):
            (Path(temp)/'manifest.json').write_text('{"campaign_plan":{"allocation_minutes":120}}')
            env=dict(SLURM_JOB_NUM_NODES='1',SLURM_NTASKS='4',SLURM_CPUS_PER_TASK='8',SLURM_JOB_ID='1')
            with patch.dict('os.environ',env):
                for minutes in [30,119,120,121]:
                    job=dict(time_limit=minutes,account='lcls_g',qos='gpu_premium')
                    with patch.object(screen.subprocess,'check_output',return_value=json.dumps(dict(jobs=[job]))):
                        if minutes==120:self.assertIn('slurm_job',screen.allocation_identity('campaign'))
                        else:
                            with self.assertRaises(RuntimeError):screen.allocation_identity('campaign')

if __name__=='__main__':unittest.main()
