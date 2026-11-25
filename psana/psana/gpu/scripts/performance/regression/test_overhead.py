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
            with self.assertRaises((KeyError,ValueError)):
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



class FivePercentScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        n.write(self.root/'plan.json',dict(margin_percent=1.0))
        n.write(self.root/'complete.json',{})
        n.write(self.root/'started.json',dict(sources={'overhead.py':'old','native_worker.py':'unchanged'},inputs={'sha256':'input'}))

    def tearDown(self):self.temp.cleanup()

    def test_interval_not_mean_selects_retry_and_original_is_unchanged(self):
        def row(mean,low,high):
            return dict(loop_seconds=dict(mean_percent_change=mean,interval95=[low,high],acceptance='inconclusive'))
        result=dict(pilot=False,overhead_qualified=False,comparisons={
            'cpu':row(.49,.21,.77),'gpu2':row(-.42,-3.76,2.91),
            'gpu1':row(-5.43,-22.13,11.28),'crosses':row(0,-6,6),
            'negative':row(-7,-8,-6),'positive':row(7,6,8)})
        before=copy.deepcopy(result)
        with patch.object(o,'collect',return_value=result):
            actual=o.screen(self.root)
        self.assertEqual(result,before)
        self.assertEqual(actual['margin_percent'],5)
        self.assertFalse(actual['screen_passed'])
        self.assertEqual(actual['retry_cases'],['gpu1','crosses','negative','positive'])
        self.assertEqual(actual['cases']['gpu2']['action'],'pass')
        self.assertEqual(n.read(self.root/'plan.json'),{'margin_percent':1.0})

    def test_pilot_and_repeat_retry_cannot_be_screened(self):
        with patch.object(o,'collect',return_value=dict(pilot=True)):
            with self.assertRaisesRegex(ValueError,'pilot'):o.screen(self.root)
        (self.root/'plan.json').write_text(json.dumps(dict(retry_parent={})))
        with self.assertRaisesRegex(ValueError,'new original'):o.screen(self.root)

    def test_parent_pins_inputs_and_worker_sources(self):
        entry=o.run_smoke.artifact(self.root/'complete.json')
        sources={'overhead.py':'new','native_worker.py':'unchanged'}
        inputs={'sha256':'input'}
        with patch.object(o,'screen',return_value=dict(retry_cases=['gpu'])):
            o.retry_parent(entry,sources,inputs)
            with self.assertRaisesRegex(ValueError,'inputs differ'):o.retry_parent(entry,sources,{})
            with self.assertRaisesRegex(ValueError,'workload sources differ'):
                o.retry_parent(entry,dict(sources,native_worker='added'),inputs)
            with self.assertRaisesRegex(ValueError,'workload sources differ'):
                o.retry_parent(entry,dict(sources,**{'native_worker.py':'changed'}),inputs)
            with self.assertRaisesRegex(ValueError,'pinned artifact changed'):
                o.retry_parent(dict(entry,sha256='0'*64),sources,inputs)
        with patch.object(o,'screen',return_value=dict(retry_cases=[])):
            with self.assertRaisesRegex(ValueError,'no cases'):o.retry_parent(entry,sources,inputs)

    def test_retry_selection_is_fixed_complete_and_balanced(self):
        config=dict(cases=[dict(id='cpu'),dict(id='gpu')])
        plan=dict(pilot=False,margin_percent=5.0,retry_parent={},retry_cases=['gpu'])
        with patch.object(o,'retry_parent',return_value=dict(retry_cases=['gpu'])):
            cases=o.selected_cases(self.root,plan,config)
            self.assertEqual(cases,[dict(id='gpu')])
            self.assertEqual(len(o.schedule(cases,False)),12)
            for change in (dict(retry_cases=[]),dict(retry_cases=['cpu']),dict(retry_cases=['gpu','gpu']),dict(pilot=True),dict(margin_percent=1)):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    o.selected_cases(self.root,dict(plan,**change),config)


class SealedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=n.read(Path(o.__file__).with_name('profile.json'))
        self.reference=dict(events=[[i+1,0,500] for i in range(1000)])
        self.config['dataset'].update(events=1000,event_sha256=n.digest(list(range(1,1001))),requested_bytes=500000)
        self.config['runtimes']={k:dict(commit='a',install='/fixture') for k in ('baseline','candidate')}
        self.site=dict(node='nid1',job_id='123',gpu_uuid='GPU-fixture',cpu_affinity={'1':[[0],[1],[2]],'2':[[0],[1],[2],[3]]})
        (self.root/'prefix').mkdir()
        for name,value in (('site.json',self.site),('prefix/event-reference.json',self.reference),('prefix/dataset-manifest.json',{})):
            n.write(self.root/name,value)
        self.config['artifacts']={k:o.run_smoke.artifact(self.root/name) for k,name in
            (('native_site','site.json'),('event_reference','prefix/event-reference.json'),('dataset_manifest','prefix/dataset-manifest.json'))}
        n.write(self.root/'profile.json',self.config)
        self.plan=dict(schema=o.SCHEMA,pilot=True,margin_percent=1.0,profile_sha256=n.digest(self.config),samples=o.schedule(self.config['cases'],True))
        n.write(self.root/'plan.json',self.plan);(self.root/'samples').mkdir()

    def tearDown(self):self.temp.cleanup()

    def capture(self,request,stage,folder):
        case,mode=request['case'],request['contract']['instrumentation']
        events=self.reference['events'][:200] if stage=='warmup' else self.reference['events']
        rows=[]
        for rank in range(case['bds']+2):
            population=[e[0] for i,e in enumerate(events) if i%case['bds']==rank-2] if rank>=2 else []
            count=len(population);gpu=case['gpus'] and rank>=2
            full=mode=='full'
            row=dict(rank=rank,role='smd0' if rank==0 else 'eb' if rank==1 else 'bd',physical_cores=[[0,rank]],affinity=[rank],node='nid1',
                instrumentation=mode,qualification_stack_traces=False,event_count=count,
                timestamps=population if full else None,steps=[[t,0] for t in population] if full and case['iterator']=='steps' else [] if full else None,
                cpu_raw_bytes=count*500 if full and not case['gpus'] else 0 if full else None,
                io=dict(requested_bytes=count*500,useful_bytes=count*500,total_bytes=count*500,total_requests=count*5,fallback=True,bulk=True) if full and gpu else None,
                delivery=dict(l1_per_delivery_window=[count],drain_l1=0) if full else None,
                first_batch_seconds=.1 if full and rank>=2 else None,callback_sizes=[count] if full else None,
                callback_events=count if gpu else 0,gpu=dict(uuid='GPU-fixture') if gpu else None,
                observed_settings={'batch_size':20},environment=self.config['environment'],
                setup_seconds=1,local_loop_seconds=1,loop_seconds=1,close_seconds=.1,cpu_seconds=.2,
                provenance_before={'manifest_commit':'a'},provenance_after={'manifest_commit':'a'})
            rows.append(row)
        work=(n.check_rows if mode=='full' else n.check_minimal_rows)(rows,events,case)
        complete=dict(status='pass',stage=stage,diagnostic=False,rows=rows,work=work,cache=None)
        if stage=='measured':
            obs=dict(policy='client-cold',verified=True,resident_bytes_before=0,measured_bytes=500*len(events))
            saved=dict(attempt=1,passed=True,seconds=.1,observations=[obs,obs])
            path=folder/'cache-attempt-1.json';n.write(path,saved)
            complete['cache']=dict(observations=[obs,obs],attempts=[dict(attempt=1,passed=True,seconds=.1,resident_bytes=[0,0],artifact=o.run_smoke.artifact(path))])
            def snap(value,clock):
                return dict(monotonic=clock,unavailable={},interfaces={'hsn0':dict(device='/pci/0',samples={d:dict(value=value,timestamp_ns=clock) for d in ('rx','tx')})})
            complete.update(cxi_before=snap(0,1),cxi_after=snap(500*len(events),2))
        return complete

    def seal(self):
        for item in self.plan['samples']:
            folder=self.root/'samples'/item['id'];folder.mkdir()
            request=o.make_request(self.config,item);n.write(folder/'request.json',request)
            records={}
            for stage in ('warmup','measured'):
                (folder/stage).mkdir()
                complete=self.capture(request,stage,folder/stage)
                n.write(folder/stage/'complete.json',complete)
                scheduler=dict(status='pass',returncode=0,no_active_steps=True,seconds=1)
                n.write(folder/stage/'scheduler.json',scheduler)
                records[stage]=o.check_capture(request,complete,self.site,self.reference,stage,folder/stage)
                records[stage]['process_seconds']=1
            index={str(p.relative_to(folder)):n.file_hash(p) for stage in ('warmup','measured') for p in (folder/stage).rglob('*') if p.is_file()}
            n.write(folder/'index.json',index)
            n.write(folder/'result.json',dict(schema=request['schema'],status='pass',sample_id=item['id'],request=o.run_smoke.artifact(folder/'request.json'),index=o.run_smoke.artifact(folder/'index.json'),**records))
        n.write(self.root/'started.json',dict(job='123',sources={},inputs={}))
        n.write(self.root/'final-integrity.json',dict(status='pass',sources={},inputs={},private_files_checked=True))
        n.write(self.root/'final-step-audit.json',dict(status='pass',steps=['123.batch','123.extern']))
        n.write(self.root/'summary.json',o._collect_samples(self.root))
        n.write(self.root/'index.json',{str(p.relative_to(self.root)):n.file_hash(p) for p in self.root.rglob('*') if p.is_file()})
        n.write(self.root/'complete.json',dict(mechanical_status='pass',overhead_qualified=False,index=o.run_smoke.artifact(self.root/'index.json'),summary=o.run_smoke.artifact(self.root/'summary.json')))

    def test_complete_pilot_collects_but_never_qualifies_and_mutation_rejected(self):
        self.seal()
        self.assertFalse(o.collect(self.root)['overhead_qualified'])
        self.assertEqual(o.check_pilot(self.root,n.file_hash(self.root/'complete.json'),{}, {})['path'],str(self.root))
        p=self.root/'samples'/self.plan['samples'][0]['id']/'measured/complete.json'
        p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'capture missing/changed'):o.collect(self.root)

    def full_plan(self, parent=None):
        self.reference=dict(events=[[i+1,0,500] for i in range(10000)])
        self.config['dataset'].update(events=10000,event_sha256=n.digest(list(range(1,10001))),requested_bytes=5000000)
        (self.root/'prefix/event-reference.json').write_text(json.dumps(self.reference))
        self.config['artifacts']['event_reference']=o.run_smoke.artifact(self.root/'prefix/event-reference.json')
        (self.root/'profile.json').write_text(json.dumps(self.config))
        self.plan.update(pilot=False,margin_percent=5.0,profile_sha256=n.digest(self.config))
        cases=self.config['cases']
        if parent:
            self.plan.update(retry_parent=parent,retry_cases=['gpu-raw-events-bd1'])
            cases=[c for c in cases if c['id']=='gpu-raw-events-bd1']
        self.plan['samples']=o.schedule(cases,False)
        (self.root/'plan.json').write_text(json.dumps(self.plan))

    def test_sealed_retry_retains_original_and_never_qualifies_whole_cohort(self):
        self.full_plan()
        capture=self.capture
        def slow_full(request,stage,folder):
            result=capture(request,stage,folder)
            if request['case']['id']=='gpu-raw-events-bd1' and request['contract']['instrumentation']=='full':
                for row in result['rows']:row['loop_seconds']=1.10
            return result
        with patch.object(self,'capture',side_effect=slow_full):self.seal()
        original=o.run_smoke.artifact(self.root/'complete.json')
        self.assertEqual(o.screen(self.root)['retry_cases'],['gpu-raw-events-bd1'])
        child=SealedCaptureTests();child.setUp()
        try:
            child.full_plan(original);child.seal()
            result=o.collect(child.root)
            self.assertTrue(result['retry_cases_passed'])
            self.assertFalse(result['overhead_qualified'])
            self.assertEqual(len(result['samples']),12)
            self.assertEqual(original,o.run_smoke.artifact(self.root/'complete.json'))
            with self.assertRaisesRegex(ValueError,'new original'):o.screen(child.root)
            (child.root/'samples'/'extra').mkdir()
            with self.assertRaisesRegex(ValueError,'incomplete/extra'):o._collect_samples(child.root)
        finally:child.tearDown()

    def test_new_margin_and_historical_margin_both_collect(self):
        self.plan['margin_percent']=o.MARGIN
        (self.root/'plan.json').write_text(json.dumps(self.plan))
        self.seal()
        self.assertFalse(o.collect(self.root)['overhead_qualified'])

    def test_failed_or_unsealed_run_cannot_report_pass(self):
        with self.assertRaises(FileNotFoundError):o.collect(self.root)
        self.seal()
        n.write(self.root/'failed.json',dict(error='final audit failed'))
        with self.assertRaisesRegex(ValueError,'failed run'):o.collect(self.root)
        (self.root/'failed.json').unlink();(self.root/'final-step-audit.json').unlink()
        with self.assertRaisesRegex(ValueError,'capture missing/changed'):o.collect(self.root)

    def test_measured_capture_rejects_bad_cache_traffic_and_attempt_history(self):
        for mode in ('minimal','full'):
            request=o.make_request(self.config,dict(self.plan['samples'][0],mode=mode))
            folder=self.root/mode;folder.mkdir()
            complete=self.capture(request,'measured',folder)
            self.assertEqual(o.check_capture(request,complete,self.site,self.reference,'measured',folder)['events'],1000)
            for kind in ('resident','count','traffic','attempt_order','attempt_pass','artifact','last_observations'):
                bad=copy.deepcopy(complete)
                if kind=='resident':bad['cache']['observations'][0]['resident_bytes_before']=1
                if kind=='count':bad['cache']['observations'].pop()
                if kind=='traffic':bad['cxi_after']['interfaces']['hsn0']['samples']['rx']['value']=1
                if kind=='attempt_order':bad['cache']['attempts'][0]['attempt']=2
                if kind=='attempt_pass':bad['cache']['attempts'][0]['passed']=False
                if kind=='artifact':bad['cache']['attempts'][0]['artifact']['sha256']='0'*64
                if kind=='last_observations':bad['cache']['observations'][0]=dict(bad['cache']['observations'][0],changed=True)
                with self.subTest(mode=mode,kind=kind),self.assertRaises(ValueError):o.check_capture(request,bad,self.site,self.reference,'measured',folder)
            warm=self.capture(request,'warmup',folder);warm['cache']=complete['cache']
            with self.assertRaisesRegex(ValueError,'warmup unexpectedly'):o.check_capture(request,warm,self.site,self.reference,'warmup',folder)

    def test_reference_and_matrix_changes_rejected(self):
        self.seal()
        (self.root/'prefix/event-reference.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact differs'):o._collect_samples(self.root)


if __name__=='__main__':unittest.main()
