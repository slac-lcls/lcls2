"""Coordinator audit of immutable capture; reads metadata only, never XTC payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--failure',action='store_true',help='preserve failed capture index, checkpoints and stack logs without acceptance')
parser.add_argument('capture',type=Path)
parser.add_argument('source',type=Path)
parser.add_argument('production_commit')
parser.add_argument('output',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.source))
import campaign
from native_contract import check_rows,cxi_delta,digest,file_hash,read,require,write
from run_smoke import derive_prefix,pinned,private_files,verify_prefix

capture=args.capture
if args.failure:
    require((capture/'failed.json').exists() and not (capture/'complete.json').exists(),'expected failed capture only')
    index={str(p.relative_to(capture)):file_hash(p) for p in sorted(capture.rglob('*')) if p.is_file()}
    stages={str(p.relative_to(capture)):read(p) for name in ('setup-ready.json','cache-ready.json','loop-complete.json','failure.json') for p in capture.rglob(name)}
    stacks={str(p.relative_to(capture)):dict(sha256=file_hash(p),bytes=p.stat().st_size,tail=p.read_text()[-12000:]) for p in capture.rglob('*-stacks.log')}
    write(args.output,dict(status='failed',production_commit=args.production_commit,capture=str(capture),failure=read(capture/'failed.json'),index=index,stages=stages,stacks=stacks,scope='failure evidence only; no accepted case or timing qualification'))
    sys.exit(0)
complete,started=read(capture/'complete.json'),read(capture/'started.json')
require(complete['status']=='pass' and len(complete['cases'])==6,'incomplete capture')
require(complete['index']['sha256']==file_hash(capture/'index.json'),'index changed')
index=read(capture/'index.json')
for name,sha in index.items():
    path=capture/name
    require(capture.resolve() in path.resolve().parents and file_hash(path)==sha,'indexed artifact changed: '+name)
for name,sha in started['sources'].items():
    require(file_hash(args.source/name)==sha,'frozen source changed')
    blob=subprocess.check_output(['git','show',args.production_commit+':psana/psana/gpu/scripts/performance/regression/'+name])
    require(hashlib.sha256(blob).hexdigest()==sha,'Git source differs: '+name)
spec=pinned(started['inputs'])
for entry in spec.values():pinned(entry)
verify_prefix(capture/'prefix',spec,10000)
parent,plan,reference=[pinned(spec[k]) for k in ('manifest','plan','reference')]
manifest=read(capture/'prefix/dataset-manifest.json')
derived,expected=derive_prefix(parent,plan,reference,10000)
require(expected==read(capture/'prefix/event-reference.json'),'independent prefix differs')
for output,original in zip(manifest['files'],parent['files']):
    require(output['bigdata']==original['bigdata'],'full prefix differs from accepted input')
private_files(parent)
config=read(capture/'smoke-profile.json')
require(not any(config['qualification'].values()),'qualification flag changed')
require('native_adapter' in campaign.validate(config,ready=False),'readiness blocker missing')
site=read(capture/'site.json')
site_index=pinned(site['capture_index'])
for name,sha in site_index.items():
    require(file_hash(Path(site['capture_index']['path']).parent/name)==sha,'placement artifact changed')
controls=read(capture/'controls/complete.json')
require(controls['status']=='pass','controls not complete')
from accounting_probe import check_control
control_rows=check_control(read(capture/'controls/traffic/complete.json')['rows'])
def intervals(before,after):
    return dict(monotonic_seconds=after['monotonic']-before['monotonic'],
        device_seconds={n:{d:(after['interfaces'][n]['samples'][d]['timestamp_ns']-v['samples'][d]['timestamp_ns'])/1e9 for d in ('rx','tx')} for n,v in before['interfaces'].items()})
for row in control_rows:
    row['sampling_intervals']=intervals(row['cxi_before'],row['cxi_after'])
rows=[]
for case in config['cases']:
    folder=capture/case['id'];request=read(folder/'request.json');result=read(folder/'result.json')
    campaign.validate_result(config,request,result,folder)
    resources=read(folder/'resources.json')
    check_rows(resources['rows'],reference['events'],case)
    cache=read(folder/'cache.json')
    require(len(cache['observations'])==2 and all(r['resident_bytes_before']==0 and
        r['measured_bytes']==config['dataset']['requested_bytes'] for r in cache['observations']),'cache evidence differs')
    traffic=cxi_delta(resources['cxi_before'],resources['cxi_after'])
    require(traffic['rx_bytes']>=0.9*config['dataset']['requested_bytes'],'traffic coverage missing')
    for stage in ('warmup','measured'):
        # Stage folder names are defined by the adapter and indexed in resources.
        require(resources['scheduler'][stage]['status']=='pass' and resources['scheduler'][stage]['no_active_steps'],'step incomplete')
    logs={str(p.relative_to(folder)):p.read_text().count('Builtin communicator 44000000 has pending 1 references.')
          for p in folder.rglob('native.log')}
    rows.append(dict(case=case['id'],events=result['events'],bytes=result['requested_bytes'],
        event_sha256=result['event_sha256'],loop_seconds=result['loop_seconds'],process_seconds=result['process_seconds'],
        bd_events=[len(r['timestamps']) for r in resources['rows'][2:]],
        cache_resident_bytes=[r['resident_bytes_before'] for r in cache['observations']],
        nic=result['nic'],cxi=traffic,rx_per_selected_byte=traffic['rx_bytes']/result['requested_bytes'],
        cxi_before=resources['cxi_before'],cxi_after=resources['cxi_after'],
        sampling_intervals=intervals(resources['cxi_before'],resources['cxi_after']),
        gpu_io=[r['io'] for r in resources['rows'][2:]],scheduler=resources['scheduler'],mpi_warning_counts=logs,
        stages={name:read(folder/'measured'/name) for name in ('setup-ready.json','cache-ready.json','loop-complete.json')},
        qualification_stack_traces=[r['qualification_stack_traces'] for r in resources['rows']],
        stack_logs={str(p.relative_to(folder)):dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in folder.rglob('*-stacks.log')},
        timing_scope='instrumented correctness diagnostics; faulthandler armed; no regression comparison'))
mpi={mode:dict(receipt=read(capture/('controls/mpi-'+mode+'-pass.json')),
    rows=read(capture/('controls/mpi-'+mode+'/complete.json'))['rows']) for mode in ('mpi-only','observe','cancel')}
# Remove repetitive runtime file maps from compact output, retaining raw file hashes/index.
for value in mpi.values():
    for row in value['rows']:
        for key in ('provenance_before','provenance_after'):row.pop(key,None)
write(args.output,dict(status='pass',production_commit=args.production_commit,job=started['job'],node=started['node'],
    capture=str(capture),indexed_files=len(index),placement_files=len(site_index),index=complete['index'],
    sources=started['sources'],input_spec=spec,prefix=read(capture/'prefix/complete.json'),controls=control_rows,mpi=mpi,cases=rows,
    qualification_flags=config['qualification'],in_job_final_audit=read(capture/'final-step-audit.json'),
    scope='10k raw cache/byte accounting and empirical traffic coverage; no timing, component or full MPI lifecycle qualification'))
print('Audited',len(index),'files;',len(rows),'cases; output',args.output)
