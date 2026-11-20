"""Offline contract tests. Synthetic adapters only; no psana/GPU/scheduler jobs."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import campaign as c


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = c.read(Path(__file__).with_name('profile.json'))
        self.config['pairs'] = 2
        self.config['cases'] = self.config['cases'][:2]
        self.config['dataset'].update(event_sha256='f'*64, requested_bytes=1000)
        self.config['qualification'] = {key:True for key in self.config['qualification']}
        for i, runtime in enumerate(self.config['runtimes'].values()):
            runtime.update(commit=str(i+1)*40,install='/fixture/install/'+str(i))
        for key in sorted({'dataset_manifest','event_reference','baseline_runtime','candidate_runtime','adapter'} |
                          set(c.QUALIFICATION_RECEIPTS.values())):
            path = self.root / (key+'.json')
            path.write_text('{}')
            self.config['artifacts'][key] = dict(path=str(path),sha256=c.file_hash(path))
        self.config['adapter_command'] = [sys.executable,self.config['artifacts']['adapter']['path'],
                                          '{request}','{result}']
        self.source = self.root/'source.json'
        c.write(self.source,self.config)
        self.prepared = self.root/'prepared'
        c.prepare(self.source,self.prepared)
        self.config,self.plan = c.verify_prepared(self.prepared)

    def native_fixture(self,argv,log,timeout):
        request=c.read(argv[-2])
        result=self.row(request,Path(argv[-1]).parent)
        c.write(argv[-1],result)
        Path(log).write_text('OFFLINE SYNTHETIC ADAPTER; no measured performance\n')
        return 0

    def row(self,request,folder=None):
        folder=folder or self.root/'validation'
        folder.mkdir(parents=True,exist_ok=True)
        evidence={}
        for key in ('runtime_provenance','correctness','cache','resources'):
            path=folder/(key+'.json')
            c.write(path,dict(synthetic=True,kind=key,sample_id=request['sample_id']))
            evidence[key]=dict(path=str(path),sha256=c.file_hash(path))
        row={k:request[k] for k in ('schema','profile_sha256','sample_id','phase','runtime_commit')}
        row.update(status='pass',observed_contract=copy.deepcopy(request['contract']),observed_contract_sha256=request['contract_sha256'],events=self.config['dataset']['events'],
                   event_sha256='f'*64,requested_bytes=1000,correctness_pass=True,
                   setup_seconds=1,first_batch_seconds=1,loop_seconds=10,process_seconds=12,
                   cache=dict(policy='client-cold',verified=True,resident_bytes_before=0,measured_bytes=1000),
                   nic=dict(rx_bytes=None,tx_bytes=None),warmup_completed=True,
                   placement=dict(node='TEST-NODE',cpu_affinity={'0':[0],'1':[1],'2':[2]},
                                  gpu_uuids=['TEST-GPU'] if request['case']['gpus'] else []),
                   evidence=evidence)
        return row

    def execute(self,phase):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'OFFLINE-TEST'}), patch.object(c,'run_command',self.native_fixture):
            return c.execute(self.prepared,phase)

    def test_draft_cannot_execute(self):
        config=c.read(Path(__file__).with_name('profile.json'))
        self.assertIn('native_adapter',c.validate(config))
        with self.assertRaisesRegex(ValueError,'qualification incomplete'): c.validate(config,ready=True)

    def test_frozen_copy_and_original_change(self):
        original=self.root/'adapter.json'
        original.write_text('original changed after freeze')
        c.verify_prepared(self.prepared)
        self.assertNotEqual(c.file_hash(original),self.config['artifacts']['adapter']['sha256'])

    def test_modified_frozen_artifact_rejected(self):
        Path(self.config['artifacts']['adapter']['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'): c.verify_prepared(self.prepared)

    def test_balanced_schedule_aa_same_runtime(self):
        rows=c.schedule(self.config,'aa')
        self.assertTrue(all(r['runtime']=='baseline' for r in rows))
        self.assertEqual([r['side'] for r in rows[:4]],['left','right','right','left'])
        self.assertEqual(len({r['id'] for r in rows}),len(rows))

    def test_phase_gates_and_full_orchestration(self):
        with self.assertRaises((FileNotFoundError,ValueError)):self.execute('ab')
        self.assertEqual(self.execute('correctness')['acceptance'],'correctness-pass')
        self.assertEqual(self.execute('aa')['acceptance'],'equivalent')
        summary=self.execute('ab')
        self.assertEqual(summary['acceptance'],'equivalent')
        self.assertEqual(summary['site'],self.config['site'])
        for cell in summary['cells'].values():
            self.assertEqual(len(cell['network_samples']),2*self.config['pairs'])
            self.assertEqual({r['runtime'] for r in cell['network_samples']},{'baseline','candidate'})
            self.assertTrue(all(r['rx_bytes'] is None and r['rx_per_selected_byte'] is None for r in cell['network_samples']))
        with self.assertRaises(FileExistsError):self.execute('ab')

    def test_failed_process_stops_and_preserves(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'OFFLINE-TEST'}),patch.object(c,'run_command',return_value=9):
            with self.assertRaisesRegex(ValueError,'adapter failed'):c.execute(self.prepared,'correctness')
        state=c.read(self.prepared/'correctness/state.json')
        self.assertEqual(state['status'],'failed')
        with self.assertRaisesRegex(ValueError,'phase not complete'):c.summarize(self.prepared,'correctness')

    def test_missing_pair_rejected(self):
        self.execute('correctness');self.execute('aa')
        sample=c.schedule(self.config,'aa')[0]
        (self.prepared/'aa'/sample['id']/'result.json').unlink()
        with self.assertRaises(FileNotFoundError):c.summarize(self.prepared,'aa')

    def test_request_tamper_rejected(self):
        self.execute('correctness')
        sample=c.schedule(self.config,'correctness')[0]
        path=self.prepared/'correctness'/sample['id']/'request.json'
        request=c.read(path);request['contract']['settings']['batch_size']=999;c.write(path,request)
        with self.assertRaisesRegex(ValueError,'request changed'):c.summarize(self.prepared,'correctness')

    def test_placement_drift_rejected(self):
        self.execute('correctness');self.execute('aa')
        sample=c.schedule(self.config,'aa')[1]
        path=self.prepared/'aa'/sample['id']/'result.json'
        row=c.read(path);row['placement']['cpu_affinity']['2']=[999];c.write(path,row)
        with self.assertRaisesRegex(ValueError,'accepted result changed'):c.summarize(self.prepared,'aa')

    def test_allocation_change_rejected(self):
        self.execute('correctness')
        with patch.dict(os.environ,{'SLURM_JOB_ID':'DIFFERENT'}):
            with self.assertRaisesRegex(ValueError,'allocation changed'):c.execute(self.prepared,'aa')
        self.assertFalse((self.prepared/'aa').exists())

    def test_cross_phase_placement_rejected_and_failed(self):
        self.execute('correctness')
        native=self.native_fixture
        def moved(argv,log,timeout):
            code=native(argv,log,timeout)
            row=c.read(argv[-1]);row['placement']['node']='OTHER-NODE';c.write(argv[-1],row)
            return code
        with patch.dict(os.environ,{'SLURM_JOB_ID':'OFFLINE-TEST'}),patch.object(c,'run_command',moved):
            with self.assertRaisesRegex(ValueError,'placement changed since correctness'):c.execute(self.prepared,'aa')
        self.assertEqual(c.read(self.prepared/'aa/state.json')['status'],'failed')

    def test_wrong_event_identity_contract_cache_and_nan(self):
        request=c.make_request(self.config,self.plan,c.schedule(self.config,'correctness')[0])
        row=self.row(request)
        variants=[]
        for key,value in [('event_sha256','a'*64),('observed_contract_sha256','b'*64),('loop_seconds',float('nan')),
                          ('events',12),('requested_bytes',0),('correctness_pass',False)]:
            bad=copy.deepcopy(row);bad[key]=value;variants.append(bad)
        bad=copy.deepcopy(row);bad['cache']['resident_bytes_before']=1;variants.append(bad)
        bad=copy.deepcopy(row);bad['nic']['rx_bytes']=-1;variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):c.validate_result(self.config,request,bad,self.root/'validation')

    def test_calibration_requires_additional_gates(self):
        self.config['cases'][0]['workload']='cpu-calib'
        self.config['qualification']['nondegenerate_mask']=False
        with self.assertRaisesRegex(ValueError,'nondegenerate_mask'):c.validate(self.config,ready=True)

    def test_statistical_outcomes(self):
        for right,expected in [([10,10],'equivalent'),([12,12],'regression'),([8,8],'improvement'),([8,12],'inconclusive')]:
            self.assertEqual(c.paired([10,10],right,5)['acceptance'],expected)
        with self.assertRaises(ValueError):c.paired([10],[10,10],5)

    def test_timeout_kills_local_process(self):
        # Capture the real launcher PID without requiring the child interpreter
        # and scratch filesystem to finish startup within the timeout.
        original, launched = c.subprocess.Popen, []
        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            launched.append(process)
            return process
        with patch.object(c.subprocess, 'Popen', side_effect=launch):
            with self.assertRaises(c.subprocess.TimeoutExpired):
                c.run_command([sys.executable,'-c','import time; time.sleep(60)'],
                              self.root/'timeout.log',2)
        self.assertEqual(len(launched), 1)
        self.assertIsNotNone(launched[0].poll())
        with self.assertRaises(ProcessLookupError):os.kill(launched[0].pid,0)

    def test_observed_contract_cannot_echo_digest(self):
        request=c.make_request(self.config,self.plan,c.schedule(self.config,'correctness')[0])
        row=self.row(request)
        row['observed_contract']['settings']['batch_size']=999
        with self.assertRaisesRegex(ValueError,'observed settings differ: settings'):
            c.validate_result(self.config,request,row,self.root/'validation')

    def test_evidence_must_be_local_and_distinct(self):
        request=c.make_request(self.config,self.plan,c.schedule(self.config,'correctness')[0])
        row=self.row(request)
        row['evidence']['cache']=row['evidence']['correctness']
        with self.assertRaisesRegex(ValueError,'evidence reused'):
            c.validate_result(self.config,request,row,self.root/'validation')
        row=self.row(request)
        row['evidence']['cache']=self.config['artifacts']['adapter']
        with self.assertRaisesRegex(ValueError,'inside sample folder'):
            c.validate_result(self.config,request,row,self.root/'validation')

    def test_flags_require_receipts_and_absolute_executable(self):
        for key in set(c.QUALIFICATION_RECEIPTS.values()):
            bad=copy.deepcopy(self.config);del bad['artifacts'][key]
            with self.assertRaisesRegex(ValueError,'artifact:'+key):c.validate(bad,ready=True)
        bad=copy.deepcopy(self.config);bad['adapter_command'][0]='python'
        with self.assertRaisesRegex(ValueError,'executable must be absolute'):c.validate(bad,ready=True)

    def test_accepted_timing_change_rejected(self):
        self.execute('correctness');self.execute('aa')
        sample=c.schedule(self.config,'aa')[0]
        path=self.prepared/'aa'/sample['id']/'result.json'
        row=c.read(path);row['loop_seconds']=1;c.write(path,row)
        with self.assertRaisesRegex(ValueError,'accepted result changed'):c.summarize(self.prepared,'aa')

    def test_prepare_never_overwrites(self):
        with self.assertRaises(FileExistsError):c.prepare(self.source,self.prepared)


if __name__=='__main__':
    unittest.main()
