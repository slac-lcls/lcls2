"""Recompute acceptance and rates from saved native benchmark artifacts."""
import json, math, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'jf_raw_cache_20261109'))
import common
from network_watch import counter_delta
need=common.require

def audit(root):
    def read(name):return json.loads((root/name).read_text())
    provenance=read('provenance.json');completion=read('complete.json');cleanup=read('cleanup.json')
    expected_names=['00-raw.json','01-sim10.json','02-sim10.json','03-raw.json']
    need(completion==dict(status='pass',cases=expected_names),'Completion matrix')
    need(cleanup['removed'] is True,'FFB temporary input not cleaned')
    need(Path(cleanup['stage']).name=='jf-cold-sim10-'+provenance['job'],'Cleanup identity')
    need(provenance['compat_mode']=='ON','Fallback not forced')
    need(provenance['effective']==dict(num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'Settings')
    need(len(set(provenance['cpu_affinity']))==4,'CPU affinity')
    need(provenance['gpu_mask'].startswith('GPU-') and provenance['gpu_mask'] in provenance['gpu_inventory'],'GPU identity')
    need(provenance['raw_sha256']==json.loads((common.HERE/'expected-sha256.json').read_text()),'Raw identity')
    need(provenance['raw_timestamp_sha256']==common.REF['10000']['timestamp_sha256'],'Raw timestamps')
    receipt=json.loads((HERE.parent/'mfx_expand10_10k_20261106/files.json').read_text())
    expected_small=[{k:r[k] for k in ('path','bytes','stream','sha256')} for r in receipt['files'] if r['kind']=='bd' and r['stream'] in range(5,10)]
    need(provenance['sim10_identity']['generation_sha256']==receipt['generation_sha256'] and provenance['sim10_identity']['files']==expected_small,'Sim10 identity')
    need(provenance['sim10_paths']==[str(Path(cleanup['stage'])/Path(r['path']).name) for r in expected_small],'Sim10 stage paths')
    need(set(provenance['mounts'])=={'raw','sim10'},'Mount inventory')
    expected_sources={str(p) for p in [HERE/'run.py',HERE/'run.sbatch',common.HERE/'common.py',HERE.parent/'mfx_kvikio_read_20261108/plan_reads.py',HERE.parent/'mfx_kvikio_read_20261108/network_watch.py']}
    need(set(provenance['source_sha256'])==expected_sources,'Code inventory')
    for label,mount in provenance['mounts'].items():
        source=json.loads(mount)['filesystems'][0]['source'];need(source=='drpsrcf/ffb','Not FFB '+source)
    for path,digest in provenance['source_sha256'].items():need(common.sha(path)==digest,'Code changed '+path)
    result=[]
    for name in expected_names:
        r=read(name);label='sim10' if 'sim10' in name else 'raw'
        total=335571760000 if label=='raw' else 33582000000
        files=[str(common.SOURCE/n) for n in common.NAMES] if label=='raw' else provenance['sim10_paths']
        sizes=[common.REF['10000']['stage_bytes'][n] for n in common.NAMES] if label=='raw' else [x['bytes'] for x in provenance['sim10_identity']['files']]
        need(r['status']=='pass' and r['dataset']==label and r['events']==10000 and r['requests']==50000 and r['bytes']==total,'Totals')
        need(r['seconds']>0 and math.isclose(r['gib_s'],total/(1<<30)/r['seconds']) and math.isclose(r['hz'],10000/r['seconds']),'Rates')
        need(r['depth']==35 and len(r['witnesses'])==105,'Ring/witness count')
        need({w['slot'] for w in r['witnesses']}==set(range(35)) and all(w['size']==64 for w in r['witnesses']),'Witness slots')
        for key in ('initial_cache','final_cache'):
            need([x['path'] for x in r[key]]==files and [x['bytes'] for x in r[key]]==sizes,'Cache file identity')
            for row in r[key]:
                need(row['pages']==(row['bytes']+4095)//4096 and 0<=row['resident_pages']<=row['pages'],'Cache pages')
                if key=='initial_cache':need(row['resident_pages']==0,'Cold gate')
        delta=counter_delta(r['nic_before'],r['nic_after'],total)
        need(delta==r['nic_delta'] and all(v['seconds']>=r['seconds'] for v in delta.values()),'NIC delta/bracket')
        need(counter_delta(r['idle']['before'],r['idle']['after'])==r['idle']['delta'],'Idle counters')
        result.append(dict(case=name,seconds=r['seconds'],gib_s=r['gib_s'],hz=r['hz'],payload_gib=total/(1<<30),
            nic_rx_gib={k:v['rx_bytes']/(1<<30) for k,v in delta.items()},idle_rx_gib_s={k:v['rx_gib_s'] for k,v in r['idle']['delta'].items()}))
    return dict(status='pass',host=provenance['host'],job=provenance['job'],cases=result)

if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1])),indent=2))
