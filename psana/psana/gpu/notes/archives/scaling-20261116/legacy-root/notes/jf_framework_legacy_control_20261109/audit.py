"""Validate the two historical-settings controls against the framework matrix."""
import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'jf_raw_cache_20261109'))
import common
need=common.require

def audit(root,primary):
    read=lambda name:json.loads((root/name).read_text())
    reference=json.loads((primary/'00-raw-field.json').read_text());p=read('provenance.json');completion=read('complete.json')
    need(completion['status']=='pass' and [r['name'] for r in completion['cases']]==['legacy-raw-cold-r1','legacy-raw-cold-r2'],'Matrix')
    need(p['source_sha256']==common.sources(),'Harness source changed')
    for path,digest in p['launcher_sha256'].items():need(common.sha(path)==digest,'Launcher changed')
    result=[]
    for rep in (1,2):
        r=read('legacy-raw-cold-r%d.json'%rep);m=r['measurement'];rows=r['rows'];bds=[x for x in rows if x['is_bd']]
        need(r['status']=='pass' and r['settings']['cache']=='cold' and r['settings']['events']==10000 and not r['settings']['preflight'] and r['settings']['bds']==1,'Case configuration')
        need(r['settings']['directory']==str(common.SOURCE),'Input directory')
        need(len(rows)==3 and len(bds)==1 and all(x['host']==reference['host'] for x in rows),'Same-node topology')
        bd=bds[0];need(bd['device']['pci']==reference['rows'][2]['device']['pci'] and bd['device']['mask']==reference['rows'][2]['device']['mask'],'Same GPU')
        need(bd['device']['compat_mode']=='ON' and bd['device']['effective']==dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'Settings')
        need(all(x['affinity']==p['allowed_physical_cores'] for x in rows),'Legacy affinity')
        need(common.ts_hash(bd['timestamps'])==common.REF['10000']['timestamp_sha256'] and len(bd['timestamps'])==len(set(bd['timestamps']))==10000,'Timestamps')
        need(all(not x['timestamps'] for x in rows if not x['is_bd']),'Helper delivery')
        counts=bd['counts'];need(counts['requests']==50000 and counts['bytes']==r['bytes']==335571760000,'I/O totals')
        need(counts['cpu_bd_reads']==counts['pressure_drains']==0 and counts['peak_owned_and_held']<=counts['budget_limit'],'CPU/pressure/budget')
        need(not bd['checks'],'No timing validation D2H')
        for module,record in bd['runtime_files'].items():need(record==reference['runtime_files'][module],'Same runtime')
        need(r['source_sha256']==common.sources(),'Case source changed')
        need(m['conditioning']['kind']=='cold' and m['conditioning']['status']=='pass','Cache condition')
        for cache in (m['conditioning']['cache'],m['cache_after']):
            need([x['path'] for x in cache['files']]==[str(common.SOURCE/n) for n in common.NAMES],'Cache paths')
            for x in cache['files']:need(x['bytes']==common.REF['10000']['stage_bytes'][Path(x['path']).name] and 0<=x['resident_pages']<=x['pages'],'Cache bounds')
        need(all(x['resident_pages']==0 for x in m['conditioning']['cache']['files']),'Cold gate')
        delta=common.counter_delta(m['nic_before'],m['nic_after']);need(delta==m['nic_delta'] and all(v['seconds']>=m['loop_seconds'] for v in delta.values()),'NIC bracket/delta')
        need(common.counter_delta(m['idle']['before'],m['idle']['after'])==m['idle']['delta'],'Idle NIC')
        need(m['loop_seconds']>0 and math.isclose(r['gib_s'],r['bytes']/2**30/m['loop_seconds']) and math.isclose(r['events_per_second'],10000/m['loop_seconds']),'Rates')
        final=m['cache_after'];result.append(dict(repeat=rep,gib_s=r['gib_s'],hz=r['events_per_second'],seconds=m['loop_seconds'],
            final_resident_fraction=final['resident_pages']/final['pages'],nic_rx_gib={k:v['rx_bytes']/2**30 for k,v in delta.items()}))
    return dict(status='pass',host=reference['host'],cases=result)
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
