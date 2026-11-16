"""Validate every fresh-process result and summarize confirmation medians."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

GIB=1<<30

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_case(r):
    c=r['config'];e=r['effective']
    require(r['status']=='pass','Case failed')
    require(e['compat_mode']=='ON' and e['num_threads']==c['threads'] and
            e['task_size']==c['task_mib']*(1<<20) and e['bounce_buffer_size']==c['bounce_mib']*(1<<20)
            and e['auto_direct_io_read']==c['direct'],'Effective settings mismatch')
    require(len(r['cpu_affinity'])==c['cpus'],'CPU affinity mismatch')
    expected_bytes=r['identity']['event_bytes'] if c['pattern']=='events' else r['identity']['whole_file_bytes']
    expected_requests=50000 if c['pattern']=='events' else sum(math.ceil(f['bytes']/(c['chunk_mib']*(1<<20))) for f in r['identity']['files'])
    require(r['requested_bytes_per_pass']==expected_bytes and r['requests_per_pass']==expected_requests,'Wrong read plan')
    require(1<=len(r['passes'])<=3,'Pass count')
    for p in r['passes']:
        require(p['bytes']==expected_bytes and p['requests']==expected_requests and p['seconds']>0,'Pass accounting')
        require(math.isclose(p['gib_s'],p['bytes']/GIB/p['seconds'],rel_tol=1e-12),'Pass rate')
    seconds=sum(p['seconds'] for p in r['passes']);total=sum(p['bytes'] for p in r['passes'])
    require(r['total_bytes']==total and math.isclose(r['seconds'],seconds,rel_tol=1e-12),'Total accounting')
    require(math.isclose(r['gib_s'],total/GIB/seconds,rel_tol=1e-12),'Total rate')
    require(r['validation']['completed_requests']==expected_requests*len(r['passes']),'Completed requests')
    witnesses=r['validation']['witnesses']
    require(r['validation']['gpu_cpu_witnesses']==len(witnesses)==3*c['depth'],'Incomplete GPU witnesses')
    require({w['slot'] for w in witnesses}==set(range(c['depth'])),'Witness slot coverage')
    require(all(w['size']==64 and 0<=w['file']<5 and
                0<=w['offset']<=r['identity']['files'][w['file']]['bytes']-64 for w in witnesses),'Invalid witness')
    require(r['device_buffer_bytes']<=4*GIB,'GPU buffer budget')

def audit(root):
    summary=json.loads((root/'summary.json').read_text())
    require(summary['status']=='pass' and len(summary['results'])==33,'Campaign incomplete')
    require(len({x['config']['name'] for x in summary['results']})==33,'Duplicate case')
    reports=[];artifacts={}
    for row in summary['results']:
        p=Path(row['path']);require(p.parent.resolve()==root.resolve(),'Foreign report')
        r=json.loads(p.read_text());check_case(r)
        require(row['config']==r['config'] and row['gib_s']==r['gib_s'],'Progress differs from native')
        artifacts[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
        reports.append(r)
    for key in ('host','job','runtime','identity','source_sha256','kvikio_source_sha256'):
        require(all(r[key]==reports[0][key] for r in reports),'Changed '+key)
    source_verified={}
    for p,h in {**reports[0]['source_sha256'],**reports[0]['kvikio_source_sha256']}.items():
        require(sha(Path(p))==h,'Source changed: '+p);source_verified[p]=h
    confirm={}
    for label,declared in summary['confirmation'].items():
        rows=[r for r in reports if r['config']['name'].startswith('confirm-'+label+'-r')]
        require(len(rows)==3,'Missing confirmations')
        settings={k:v for k,v in declared['config'].items() if k!='name'}
        require(all({k:v for k,v in r['config'].items() if k!='name'}==settings for r in rows),'Confirmation settings')
        values=[r['gib_s'] for r in rows]
        med=statistics.median(values)
        require(med==declared['median_gib_s'] and min(values)==declared['min_gib_s'] and max(values)==declared['max_gib_s'],'Confirmation summary')
        confirm[label]=dict(config=settings,rates_gib_s=values,median_gib_s=med,min_gib_s=min(values),max_gib_s=max(values),
                            event_equivalent_hz=med*GIB/(reports[0]['identity']['event_bytes']/10000))
    return dict(status='pass',host=reports[0]['host'],job=reports[0]['job'],runtime=reports[0]['runtime'],
                identity=reports[0]['identity'],case_count=33,total_read_bytes=sum(r['total_bytes'] for r in reports),
                total_measured_seconds=sum(r['seconds'] for r in reports),confirmation=confirm,
                screen=[dict(name=r['config']['name'],config=r['config'],gib_s=r['gib_s'],seconds=r['seconds'],
                             pass_rates=[p['gib_s'] for p in r['passes']],direct_fd_count=r['input_direct_fd_count']) for r in reports],
                source_verified=source_verified,artifacts=artifacts)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as out:json.dump(audit(args.root),out,indent=2);out.write('\n')
