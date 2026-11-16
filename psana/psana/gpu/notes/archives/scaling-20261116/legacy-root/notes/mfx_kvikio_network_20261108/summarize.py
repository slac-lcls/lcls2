"""Independently validate read accounting, cache gates, and NIC deltas."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mfx_kvikio_read_20261108'))
from audit import check_case, require


def summarize(root):
    rows=[]; artifacts={}; identity=None
    for repeat in range(1,4):
        path=root/('network-r%d.json'%repeat)
        r=json.loads(path.read_text());check_case(r)
        require(r['config']==dict(name='network-r%d'%repeat,pattern='events',chunk_mib=0,
            threads=4,task_mib=4,bounce_mib=16,depth=35,direct=False,cpus=4,network_profile=True),'Unexpected configuration')
        require(len(r['passes'])==3,'Incomplete cold/warm series')
        key={k:r[k] for k in ('host','job','identity','runtime','source_sha256','kvikio_source_sha256')}
        if identity is None: identity=key
        require(key==identity,'Changed provenance')
        artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for number,p in enumerate(r['passes']):
            n=p['network'];b=n['nic_before'];a=n['nic_after'];d=n['nic_delta']
            require(n['kind']==('node-cache-cold' if number==0 else 'warm-repeat'),'Wrong pass label')
            for cache in (n['cache_before'],n['cache_after']):
                require(len(cache['files'])==5 and {f['path'] for f in cache['files']}=={f['path'] for f in r['identity']['files']},'Cache file coverage')
                require(sum(f['resident_pages'] for f in cache['files'])==cache['resident_pages'],'Residency sum')
                require(all(0<=f['resident_pages']<=f['pages'] for f in cache['files']),'Invalid residency')
            if number==0:
                require(n['cache_before']['resident_pages']==0 and n['conditioning']['resident_pages']==0,'Cold cache gate')
            require(b.keys()==a.keys()==d.keys() and bool(b),'Missing NIC endpoints')
            for name in b:
                rx=a[name]['rx_bytes_phy']-b[name]['rx_bytes_phy'];tx=a[name]['tx_bytes_phy']-b[name]['tx_bytes_phy']
                seconds=a[name]['monotonic']-b[name]['monotonic']
                require(a[name]['device']==b[name]['device'] and rx>=0 and tx>=0 and seconds>=p['seconds'],'Counter reset or interval too short')
                require(d[name]['rx_bytes']==rx and d[name]['tx_bytes']==tx and d[name]['seconds']==seconds,'NIC arithmetic')
                require(math.isclose(d[name]['rx_gib_s'],rx/(1<<30)/seconds,rel_tol=1e-12) and math.isclose(d[name]['rx_to_requested_bytes'],rx/p['bytes'],rel_tol=1e-12),'NIC rates')
            rows.append(dict(repeat=repeat,pass_index=number,kind=n['kind'],read_gib_s=p['gib_s'],
                read_seconds=p['seconds'],read_bytes=p['bytes'],nic=d,
                cache_before_fraction=n['cache_before']['resident_pages']/n['cache_before']['pages'],
                cache_after_fraction=n['cache_after']['resident_pages']/n['cache_after']['pages']))
    source_verified={}
    for path,expected in {**identity['source_sha256'],**identity['kvikio_source_sha256']}.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,'Source changed: '+path)
        source_verified[path]=expected
    return dict(status='pass',provenance=identity,artifacts=artifacts,passes=rows,source_verified=source_verified,
                median_read_gib_s={kind:statistics.median(x['read_gib_s'] for x in rows if x['kind']==kind)
                                   for kind in ('node-cache-cold','warm-repeat')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as f: json.dump(summarize(args.root),f,indent=2);f.write('\n')
