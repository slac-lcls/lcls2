"""Overhead isolation and evidence rejection; no psana or hardware execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import overhead as o
import native_contract as n
import perlmutter_adapter as adapter
from test_native import NativeTests


class OverheadContractTests(unittest.TestCase):
    setUp, tearDown = NativeTests.setUp, NativeTests.tearDown
    add_artifact, rows = NativeTests.add_artifact, NativeTests.rows
    def test_mode_cannot_leak_into_normal_phase(self):
        config = self.config
        item = o.schedule(config['cases'],True)[0]
        request = o.make_request(config,item)
        n.load_contract(request)
        for kind in ('schema','phase','mode'):
            bad=copy.deepcopy(request)
            if kind == 'schema': bad['schema'] = n.SCHEMA
            if kind == 'phase': bad['phase'] = 'aa'
            if kind == 'mode': bad['contract']['instrumentation'] = 'off'
            bad['contract_sha256'] = n.digest(bad['contract'])
            with self.subTest(kind=kind),self.assertRaises(ValueError): n.load_contract(bad)
        with self.assertRaisesRegex(ValueError,'cannot become'):
            adapter.assemble(request,{}, {}, {},0,self.root,self.site)

    def test_minimal_counts_do_not_claim_identity_or_bytes(self):
        case=dict(bds=2,workload='gpu-raw',iterator='events')
        rows=self.rows(2,True)
        for row in rows:
            row['event_count']=len(row['timestamps'])
            for key in ('timestamps','steps','cpu_raw_bytes','io','delivery','first_batch_seconds','callback_sizes'):
                row[key]=None
        expected=dict(events=4,event_sha256=None,requested_bytes=None)
        self.assertEqual(n.check_minimal_rows(rows,self.reference['events'],case),expected)
        for kind in ('count','callback','invented','rank','cores'):
            bad=copy.deepcopy(rows)
            if kind=='count': bad[2]['event_count']-=1
            if kind=='callback': bad[2]['callback_events']-=1
            if kind=='invented': bad[2]['cpu_raw_bytes']=1000
            if kind=='rank': bad[1]['rank']=0
            if kind=='cores': bad[2]['physical_cores']=bad[0]['physical_cores']
            with self.subTest(kind=kind),self.assertRaises(ValueError):n.check_minimal_rows(bad,self.reference['events'],case)


del NativeTests


class OverheadTests(unittest.TestCase):
    def test_balanced_schedule_fixed_size(self):
        cases=n.read(Path(o.__file__).with_name('profile.json'))['cases']
        for pilot,size in ((True,24),(False,72)):
            samples=o.schedule(cases,pilot)
            self.assertEqual(len(samples),size)
            self.assertEqual(len(set(s['id'] for s in samples)),size)
            for start in range(0,size,4):
                self.assertEqual([s['mode'] for s in samples[start:start+4]],['minimal','full','full','minimal'])
        config=dict(schema=n.SCHEMA,site='perlmutter',timeout_seconds=1950,dataset={},settings={},cache_policy='client-cold',warmup={},environment={},artifacts={},runtimes={'baseline':dict(commit='a'*40,install='/x')})
        before=copy.deepcopy(config)
        a=o.make_request(config,samples[0]);b=o.make_request(config,samples[1])
        self.assertEqual(config,before)
        self.assertNotEqual(a['contract_sha256'],b['contract_sha256'])

    def test_missing_extra_and_changed_plan_fail_before_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=n.read(Path(o.__file__).with_name('profile.json'))
            config['dataset']['events']=1000
            plan=dict(schema=o.SCHEMA,profile_sha256=n.digest(config),pilot=True,margin_percent=1.0,samples=o.schedule(config['cases'],True))
            for name,value in (('profile.json',config),('plan.json',plan),('site.json',{})):
                n.write(root/name,value)
            (root/'prefix').mkdir();n.write(root/'prefix/event-reference.json',{})
            (root/'samples').mkdir()
            with self.assertRaisesRegex(ValueError,'incomplete/extra'):o.collect(root)
            for sample in plan['samples']:(root/'samples'/sample['id']).mkdir()
            (root/'samples/extra').mkdir()
            with self.assertRaisesRegex(ValueError,'incomplete/extra'):o.collect(root)
            plan['margin_percent']=5.0;(root/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'plan changed'):o.collect(root)

    def test_nonfinite_and_nonpositive_timings_rejected(self):
        for value in (0,-1,True,float('nan'),float('inf'),'1'):
            self.assertFalse(o.finite_positive(value))
        self.assertTrue(o.finite_positive(.1))


    def test_primary_statistics_are_paired_and_never_zero_pad_missing_pairs(self):
        cases=[dict(id='x')]
        timings={("x",pair,mode):dict(loop_seconds=100 if mode=='minimal' else 101,
                    setup_seconds=1,process_seconds=110) for pair in range(6) for mode in ('minimal','full')}
        result=o.compare(timings,cases,False)['x']['loop_seconds']
        self.assertAlmostEqual(result['mean_percent_change'],1.0)
        self.assertEqual(result['pairs'],6)
        del timings[('x',5,'full')]
        with self.assertRaises(KeyError):o.compare(timings,cases,False)

    def test_pilot_requires_pinned_completion_before_reading_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);n.write(root/'complete.json',dict(mechanical_status='pass',overhead_qualified=False))
            with self.assertRaisesRegex(ValueError,'completion changed'):
                o.check_pilot(root,'0'*64,{}, {})
            with self.assertRaises(KeyError):
                o.check_pilot(root,n.file_hash(root/'complete.json'),{}, {})

    def test_capture_rejects_count_only_as_full_or_profiler_as_clean(self):
        rows=[]
        for rank in range(3):
            rows.append(dict(rank=rank,role=('smd0','eb','bd')[rank],physical_cores=[[0,rank]],affinity=[rank],node='nid1',
                 instrumentation='minimal',qualification_stack_traces=False,event_count=2 if rank==2 else 0,
                 timestamps=None,steps=None,cpu_raw_bytes=None,io=None,delivery=None,first_batch_seconds=None,
                 callback_sizes=None,callback_events=0,gpu=None,observed_settings={'batch_size':20},environment={},
                 setup_seconds=1,local_loop_seconds=2,loop_seconds=2.1,close_seconds=.1,cpu_seconds=.2,
                 provenance_before={'manifest_commit':'a'},provenance_after={'manifest_commit':'a'}))
        request=dict(runtime_commit='a',case=dict(bds=1,gpus=0,workload='cpu-raw',iterator='events'),
                     contract=dict(instrumentation='minimal',warmup=dict(events=2),settings={'batch_size':20},environment={}))
        complete=dict(status='pass',stage='warmup',diagnostic=False,rows=rows,cache=None,
                      work=dict(events=2,event_sha256=None,requested_bytes=None))
        site=dict(node='nid1',cpu_affinity={'1':[[0],[1],[2]]})
        ref=dict(events=[[1,0,500],[2,0,500]])
        self.assertFalse(o.check_capture(request,complete,site,ref,'warmup')['exact_identity_and_bytes_checked'])
        for kind in ('profile','trace','placement','count','runtime'):
            bad=copy.deepcopy(complete)
            if kind=='profile':bad['diagnostic']=True
            if kind=='trace':bad['rows'][2]['qualification_stack_traces']=True
            if kind=='placement':bad['rows'][2]['affinity']=[3]
            if kind=='count':bad['rows'][2]['event_count']=1
            if kind=='runtime':bad['rows'][2]['provenance_after']['manifest_commit']='b'
            with self.subTest(kind=kind),self.assertRaises(ValueError):o.check_capture(request,bad,site,ref,'warmup')
        request['contract']['instrumentation']='full'
        with self.assertRaises((TypeError,ValueError)):o.check_capture(request,complete,site,ref,'warmup')


if __name__=='__main__':unittest.main()
