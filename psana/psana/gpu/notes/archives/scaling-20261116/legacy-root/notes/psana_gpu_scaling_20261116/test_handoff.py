"""Synthetic full receipts exercise auditing and screen handoff without hardware."""
import copy,contextlib,csv,hashlib,io,json,os,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import audit,campaign,prepare,report,support
from test_scaling import receipt,probes

COMMIT='a'*40

def save(path,data):path.write_text(json.dumps(data,separators=(',',':'))+'\n')

class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='scale-handoff-');cls.root=Path(cls.tmp.name)
        cls.info=receipt(1,case=campaign.schedule('screen')[0])[2]
        cls.patches=contextlib.ExitStack()
        cls.patches.enter_context(patch.object(audit,'load_inputs',return_value=cls.info))
        cls.patches.enter_context(patch.object(audit,'verify_payloads',return_value=None))
        cls.patches.enter_context(patch.object(prepare,'ROOT',cls.root))
        cls.pilot=cls.build('pilot',None)
        save(cls.pilot/'results/audit.json',audit.audit(cls.pilot,COMMIT))
        cls.screen=cls.build('screen',cls.pilot)
    @classmethod
    def tearDownClass(cls):cls.patches.close();cls.tmp.cleanup()
    @classmethod
    def build(cls,mode,pilot):
        attempt=cls.root/('scale-'+mode);out=attempt/'results';out.mkdir(parents=True);bundle=attempt/'bundle';bundle.mkdir()
        plan=dict(mode=mode,commit=COMMIT,requested_seconds=campaign.LIMITS[mode],per_case_timeout=600,pilot=None if pilot is None else dict(attempt=str(pilot),audit_sha256=support.sha(pilot/'results/audit.json')))
        save(attempt/'plan.json',plan)
        end='00:45:00' if mode=='pilot' else '02:00:00'
        slurm='Account=lcls_g QOS=gpu_premium NumNodes=1 NodeList=node StartTime=2026-11-16T00:00:00 EndTime=2026-11-16T'+end
        (attempt/'slurm-job.txt').write_text(slurm)
        until=campaign.deadline(plan,slurm);cases=campaign.schedule(mode)
        save(out/'schedule.json',dict(seed=campaign.SEED,cases=cases))
        original={}
        for tasks in (6,10,14,18):
            # Deliberately different step masks: all share exactly 54 cores.
            cores={6:60,10:60,14:56,18:54}[tasks]
            mask=list(range(cores))+list(range(64,64+cores))
            original[str(tasks)]=probes(tasks,mask)
            for r in original[str(tasks)]:save(out/('affinity-%d-r%d.json'%(tasks,r['rank'])),r)
        maps=support.campaign_affinity(original);save(out/'affinity.json',dict(maps=maps,probes=original))
        uuids=['GPU-%d'%g for g in range(4)];save(out/'gpu-inventory.json',dict(uuids=uuids))
        expected=json.loads((support.HERE/'expected-native.json').read_text())
        for g in range(4):save(out/('gpu-tests-%d.json'%g),dict(exitcode=0,collected=expected['collected'],sources=expected['sources'],reports=[dict(nodeid=n,when='call',outcome='passed') for n in expected['collected']],device=dict(uuid=uuids[g])))
        ranges={'f%d'%i:dict(offsets=list(range(100000)),size=i+100) for i in range(5)};save(out/'ranges.json',ranges);save(out/'witness.json',{})
        save(out/'input-validation.json',dict(state='pass',ranges_sha256=support.sha(out/'ranges.json'),witness_sha256=support.sha(out/'witness.json')))
        provenance=attempt/'build-provenance.json';save(provenance,dict(commit=support.SHA,installed={'psana.py':'modulehash'}))
        save(bundle/'external-pins.json',{str(provenance):support.sha(provenance)})
        pins=dict(commit=COMMIT,files={'external-pins.json':support.sha(bundle/'external-pins.json')});save(bundle/'bundle.json',pins)
        completed=[]
        for case in cases:
            r,_,_=receipt(case['gpus'],case['cpu_policy'],case)
            # Block-specific base rates make accidental cross-block pairing visible.
            rate=(2+case['block'])*case['gpus'];seconds=r['bytes']/2**30/rate
            r.update(job='123',gib_s=rate,seconds=seconds)
            for row in r['rows']:
                row['affinity']=maps['legacy' if case['cpu_policy']=='legacy' else str(case['gpus'])][str(row['rank'])]
                row['elapsed']=seconds;row['runtime']['psana']['sha256']='modulehash'
                if row['rank']>=2:
                    for path in row['io']['range_sha256']:
                        ref=ranges[Path(path).name];h=hashlib.sha256()
                        for t in row['timestamps']:h.update(struct.pack('<QQQ',t,ref['offsets'][t],ref['size']))
                        row['io']['range_sha256'][path]=h.hexdigest()
            save(out/(case['id']+'.json'),r)
            samples=[dict(time=t/2,total=40*2**30,free=30*2**30,used=10*2**30) for t in range(25)]
            memory=dict(errors=[],devices=[dict(uuid=u,samples=samples,errors=[],max_used=10*2**30,min_free=30*2**30) for u in uuids[:case['gpus']]])
            save(out/(case['id']+'.memory.json'),memory)
            completed.append(dict(case=case,result_sha256=support.sha(out/(case['id']+'.json')),memory_sha256=support.sha(out/(case['id']+'.memory.json')),wall_seconds=200.,seconds=seconds,gib_s=rate))
        complete=dict(state='cases_passed_pending_slurm',bundle=pins,mode=mode,plan=plan,deadline_unix=until,finished=until-60,job='123',completed=completed)
        save(out/'complete.json',complete);save(out/'status.json',complete)
        with (attempt/'accounting-utc.txt').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['JobID','State','ExitCode','Account','QOS','NNodes','ElapsedRaw'],delimiter='|');writer.writeheader()
            for suffix in ['','.batch','.extern']+['.%d'%i for i in range(len(cases)+9)]:writer.writerow(dict(JobID='123'+suffix,State='COMPLETED',ExitCode='0:0',Account='lcls_g',QOS='gpu_premium',NNodes='1',ElapsedRaw='100'))
        return attempt
    def test_full_audit_screen_gate_and_paired_report(self):
        plan=json.loads((self.screen/'plan.json').read_text());prepare.validate_plan(plan)
        self.assertEqual(len(audit.audit(self.pilot,COMMIT)['cases']),6)
        self.assertEqual(len(audit.audit(self.screen,COMMIT)['cases']),12)
        with contextlib.redirect_stdout(io.StringIO()):report.report(self.screen,COMMIT)
        rows=json.loads((self.screen/'results/scaling-summary.json').read_text())['rows']
        for g,row in enumerate(rows,1):
            self.assertAlmostEqual(row['median_paired_speedup'],g);self.assertAlmostEqual(row['efficiency'],1.)
    def test_attempt_corruptions_rejected(self):
        out=self.pilot/'results'
        changes=[(self.pilot/'accounting-utc.txt',lambda text:'\n'.join(text.splitlines()[:-1])+'\n'),(self.pilot/'slurm-job.txt',lambda text:text.replace('NodeList=node','NodeList=other')),(out/'affinity-18-r0.json',lambda text:text.replace('"host":"node"','"host":"other"')),(out/'tail-g1.memory.json',lambda text:text.replace('"errors":[]','"errors":["missing"]',1)),(out/'gpu-tests-0.json',lambda text:text.replace('"uuid":"GPU-0"','"uuid":"GPU-1"'))]
        for path,mutate in changes:
            original=path.read_text()
            try:
                path.write_text(mutate(original))
                with self.subTest(path=path.name),self.assertRaises(ValueError):audit.audit(self.pilot,COMMIT)
            finally:path.write_text(original)
        plan=json.loads((self.screen/'plan.json').read_text());plan['pilot']['audit_sha256']='wrong'
        with self.assertRaises(ValueError):prepare.validate_plan(plan)
    def test_mismatched_block_baseline_rejected(self):
        path=self.screen/'results/r2-g1.json';text=path.read_text()
        try:
            path.write_text((self.screen/'results/r1-g1.json').read_text())
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):report.report(self.screen,COMMIT)
        finally:path.write_text(text)
    def test_runtime_affinity_reads_topology_and_rejects_cross_rank_siblings(self):
        topology=self.root/'sys';topology.mkdir(exist_ok=True)
        for c in range(128):
            p=topology/str(c);p.mkdir(exist_ok=True);(p/'physical_package_id').write_text('0');(p/'core_id').write_text(str(c%64))
        mapping=support.select_affinity(probes(6),4);current=[set(range(128))]
        def path_factory(value):return topology/value.split('/cpu')[-1].split('/')[0]
        def set_mask(pid,mask):current[0]=set(mask)
        env=dict(SLURM_PROCID='2',JF_CPU_POLICY='compact',JF_CPU_AFFINITY=json.dumps(mapping))
        with patch.dict(os.environ,env),patch.object(support,'Path',side_effect=path_factory),patch.object(os,'sched_getaffinity',side_effect=lambda pid:current[0]),patch.object(os,'sched_setaffinity',side_effect=set_mask):
            self.assertEqual(support.affinity(),mapping['2'])
            current[0]=set(range(128));bad=copy.deepcopy(mapping);bad['3']=mapping['2'];os.environ['JF_CPU_AFFINITY']=json.dumps(bad)
            with self.assertRaises(ValueError):support.affinity()

if __name__=='__main__':unittest.main(verbosity=2)
