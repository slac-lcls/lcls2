import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit
import bundle
import campaign as c


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.attempt=Path(self.tmp.name)
        self.root=self.attempt/'results';self.root.mkdir()
        self.b=self.attempt/'bundle';self.b.mkdir();(self.b/'reader').write_bytes(b'production fixture')
        self.prov={'files':{'reader':{'sha256':c.sha(self.b/'reader')}}}
        groups=[]
        for count in c.COUNTS:
            rows=[]
            for i in range(5):
                size=c.REQUEST*(i+1)+3
                rows.append({'path':str(self.attempt/f'data-{count}-{i}'),'bytes':size,'identity':{'st_dev':1,'st_ino':100*count+i,'st_size':size,'st_mtime_ns':123}})
            groups.append({'count':count,'verified':rows})
        self.staging={'sets':groups};self.inputs={g['count']:g['verified'] for g in groups}
        self.write(self.b/'staging-complete.json',self.staging)
        self.pin=c.sha(self.b/'staging-complete.json')
        self.write(self.root/'cpu-map.json',[{'package':0,'core':i,'cpu':i,'numa_node':i//10} for i in range(40)])
        (self.attempt/'slurm-job.txt').write_text('NodeList=fixture-node\n')
        (self.attempt/'accounting-utc.txt').write_text('JobID|Account|QOS|State|ExitCode\n'+''.join(f'123{suffix}|lcls_g|gpu_premium|COMPLETED|0:0\n' for suffix in ['', '.batch','.extern','.0']))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self,path,value):
        path.write_text(json.dumps(value))

    def prepare(self,mode):
        cases=c.schedule(mode);self.cases=cases
        self.write(self.root/'schedule.json',{'mode':mode,'seed':c.SEED,'cases':cases})
        finished=[]
        global_cache={'state':'complete','confirmed_zero_scans':2,'files':[dict(path=r['path'],identity=r['identity'],resident_pages=0) for count in c.COUNTS for r in self.inputs[count]]}
        memory={'meminfo_kib':{k:0 for k in ['MemFree','MemAvailable','Cached','Buffers','Dirty','Writeback','Slab']},'lustre_max_cached_mb':{'available':False}}
        for case in cases:
            count,threads=case['count'],case['threads'];expected=self.inputs[count]
            total=sum(e['bytes'] for e in expected);rate=1+count/100+threads/1000+case['block']/10000
            elapsed=total/rate/2**30
            measured={'state':'complete','threads':threads,'request_bytes':c.REQUEST,'warm_probe':mode=='pilot','eviction_attempts':1,'eviction_seconds':0.1,'page_size':4096,'confirmed_zero_scans':2,'bytes':total,'seconds':elapsed,'gib_per_second':total/elapsed/2**30,'files':[],'workers':[]}
            for exp in expected:
                measured['files'].append({'path':exp['path'],'identity':exp['identity'],'pages':math.ceil(exp['bytes']/4096),'cold_resident_pages':0,'warm_resident_pages':1 if mode=='pilot' else 0})
            per=threads//5
            for i in range(threads):
                f,part=divmod(i,per);size=expected[f]['bytes'];chunks=math.ceil(size/c.REQUEST)
                begin=min(size,(chunks*part//per)*c.REQUEST);end=min(size,(chunks*(part+1)//per)*c.REQUEST)
                measured['workers'].append({'index':i,'file':f,'cpu':i,'actual_start_cpu':i,'actual_end_cpu':i,'begin':begin,'end':end,'bytes':end-begin,'calls':math.ceil((end-begin)/c.REQUEST),'start_seconds':0,'end_seconds':elapsed,'cpu_seconds':0,'test_checksum':0})
            raw=self.root/(case['id']+'.stdout.json');self.write(raw,measured)
            ev=self.root/(case['id']+'.eviction.json');self.write(ev,global_cache)
            command=[str(self.b/'reader'),str(threads),str(c.REQUEST),','.join(map(str,range(threads))),'1' if mode=='pilot' else '0',*[e['path'] for e in expected]]
            receipt={'case':case,'reader':measured,'command':command,'raw_sha256':c.sha(raw),'reader_sha256':self.prov['files']['reader']['sha256'],'audit':'pass','started_unix':1,'finished_unix':2+elapsed,'global_eviction':global_cache,'eviction_sha256':c.sha(ev),'memory_before':memory,'memory_after':memory}
            self.write(self.root/(case['id']+'.json'),receipt)
            finished.append({'id':case['id'],'gib_per_second':measured['gib_per_second'],'seconds':elapsed})
        complete={'state':'complete','mode':mode,'job':'123','host':'fixture-node','bundle_provenance':self.prov,'completed':finished,'case_count':len(cases),'cache_scope':'fixture'}
        self.write(self.root/'complete.json',complete);self.write(self.root/'status.json',complete)

    def run_audit(self):
        with patch.object(bundle,'verify',return_value=self.prov),patch.object(c,'validate_inputs',return_value=self.inputs),patch.object(c,'STAGING_SHA256',self.pin):
            return audit.audit(self.attempt)

    def test_pilot_estimate(self):
        self.prepare('pilot');r=self.run_audit()
        self.assertEqual(r['trials'],2)
        self.assertEqual(r['sweep_estimate_seconds'],54*max(x['wall_seconds'] for x in r['rows']))
        self.assertGreater(r['suggested_walltime_seconds'],r['sweep_estimate_seconds'])

    def test_full54_summary(self):
        self.prepare('sweep');r=self.run_audit()
        self.assertEqual(r['trials'],54);self.assertEqual(len(r['cells']),18)
        baseline=[g for g in r['cells'] if g['count']==1]
        self.assertTrue(all(g['paired_geomean_vs_count1']==1 for g in baseline))
        self.assertEqual(len((self.root/'trials.csv').read_text().splitlines()),55)

    def test_missing_receipt(self):
        self.prepare('pilot');(self.root/(self.cases[0]['id']+'.json')).unlink()
        with self.assertRaises(FileNotFoundError):self.run_audit()

    def test_raw_hash_mismatch(self):
        self.prepare('pilot');(self.root/(self.cases[0]['id']+'.stdout.json')).write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'raw hash'):self.run_audit()

    def test_wrong_command(self):
        self.prepare('pilot');p=self.root/(self.cases[0]['id']+'.json');r=json.loads(p.read_text());r['command'][1]='999';self.write(p,r)
        with self.assertRaisesRegex(RuntimeError,'executed command'):self.run_audit()

    def test_schedule_change(self):
        self.prepare('pilot');p=self.root/'schedule.json';r=json.loads(p.read_text());r['cases'].reverse();self.write(p,r)
        with self.assertRaisesRegex(RuntimeError,'schedule changed'):self.run_audit()

    def test_failed_slurm(self):
        self.prepare('pilot');p=self.attempt/'accounting-utc.txt';p.write_text(p.read_text().replace('COMPLETED','FAILED'))
        with self.assertRaisesRegex(RuntimeError,'Slurm not successful'):self.run_audit()

    def test_global_cache_or_memory_missing(self):
        self.prepare('pilot');p=self.root/(self.cases[0]['id']+'.json');r=json.loads(p.read_text());r['memory_before']['meminfo_kib'].pop('Cached');self.write(p,r)
        with self.assertRaisesRegex(RuntimeError,'memory fields'):self.run_audit()


class BundleTests(unittest.TestCase):
    def test_required_keys_and_reviewed_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);root=attempt/'bundle';root.mkdir()
            for name in bundle.KEYS:(root/name).write_text(name)
            pins={'sha256':{name:c.sha(root/name) for name in ['reader','reader.cpp','compiler.txt']},'test_binary_sha256':'f'*64,'production_flags':bundle.FLAGS}
            (root/'build-pins.json').write_text(json.dumps(pins))
            provenance={'commit':'a'*40,'production_flags':bundle.FLAGS,'files':{name:{'sha256':c.sha(root/name)} for name in bundle.KEYS}}
            path=attempt/'bundle-provenance.json';path.write_text(json.dumps(provenance))
            self.assertEqual(bundle.verify(attempt),provenance)
            missing=copy.deepcopy(provenance);missing['files'].pop('reader.cpp');path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(RuntimeError,'file set'):bundle.verify(attempt)
            (root/'reader').write_text('modified binary')
            provenance['files']['reader']['sha256']=c.sha(root/'reader');path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(RuntimeError,'build pin'):bundle.verify(attempt)


if __name__=='__main__':unittest.main(verbosity=2)
