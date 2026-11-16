"""Independent acceptance of case receipts and native job accounting."""
import csv,hashlib,json,math,struct,re
from pathlib import Path
from support import require,sha,load_inputs,view,histogram,EVENT_BYTES,SHA,verify_payloads,select_affinity
from campaign import schedule
from memory_watch import validate_memory
from launch_plan import check_allocation,make_plan

def audit_case(r,case,info):
    n=case['events'];stamps=info['timestamps'][:n];rows=r['rows'];bds=rows[2:];g=view(info,case['count']);paths={x['path']:x for x in g['files']}
    require(r['state']=='pass' and r['execution_sha']==SHA,'State/code')
    require(all(r['parameters'][k]==case[k] for k in ['count','streams','bds','events','kind','batch_size','workers_per_bd']),'Caseparameters')
    require(r['events']==n and r['bytes']==n*EVENT_BYTES and len(rows)==case['bds']+2,'Events/bytes/ranks')
    require(r['seconds']>0 and math.isclose(r['seconds'],max(x['elapsed'] for x in rows),rel_tol=1e-12),'Timing')
    require(math.isclose(r['gib_s'],r['bytes']/2**30/r['seconds'],rel_tol=1e-12),'Rate')
    merged=[t for row in bds for t in row['timestamps']];require(sorted(merged)==stamps and len(set(merged))==n,'Globaleventpartition')
    require(all(not x['timestamps'] and x['device'] is None for x in rows[:2]),'Helpers')
    for i,row in enumerate(rows):
        require(row['rank']==i and row['cleanup_complete'] and row['host']==r['host'],'Rankcleanup')
        require(len(row['affinity'])==(3 if i<2 else 7),'Affinity size')
        require(row['runtime'] and all('/source/install_psana/' in v['path'] for v in row['runtime'].values()),'Runtimepaths')
    cores=[c for row in rows for c in row['affinity']];require(len(cores)==len(set(cores))==34,'Overlapping rank masks')
    for row in bds:
        local=row['timestamps']
        if case['kind'] in ('sizing','screen'):require(local,'Full case empty BD')
        peaks=row['budget_high_water']
        require((len(peaks)==1 if local else len(peaks)<=1) and all(p['limit']==6*2**30 and p['observations']>0 and 0<=p['peak_committed']<=p['peak_committed_plus_held']<=p['limit'] for p in peaks),'Budget peaks')
        require(local==sorted(set(local)),'Localorder')
        require([t for c in row['callbacks'] for t in c['timestamps']]==local,'Callbacks')
        require(row['cpu_reads']==0 and row['io']['pressure_drains']==0,'UnexpectedCPU/pressure')
        io=row['io'];require(io['requests']==io['completed_requests']==len(local)*5 and io['requested_bytes']==io['completed_bytes']==len(local)*EVENT_BYTES,'I/Ocounts')
        require(io['read_size_histogram']==(histogram(len(local)) if local else {}) and set(io['range_sha256'])==(set(paths) if local else set()),'Readhistogram')
        require(row['device']['compat_mode']=='ON' and row['device']['effective']==dict(num_threads=4,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'EffectiveKvikIO')
        require((len(row['drain'])==1 if local else len(row['drain'])<=1) and all(d==dict(closed=True,pending=0,active=0,held=0) for d in row['drain']),'Drainreceipt')
        require(row['device']==bds[0]['device'] and row['device']['uuid'] in row['inherited_uuids'],'GPU')
        require((len(row['admission'])==1 if local else len(row['admission'])<=1) and all(a['peers']==case['bds'] and a['budget']==6*2**30 for a in row['admission']),'Admission')
        require(set(x['path'] for x in row['handles']['dgram'])==set(paths),'Dgrambinding')
        require(set(x['primary']['path'] for x in row['handles']['kvikio'])==(set(paths) if local else set()),'KvikIObinding')
        for fd in row['handles']['dgram']+[v for x in row['handles']['kvikio'] for v in x['all_new_handles']]:
            expected=paths[fd['path']];require(fd['dev']==expected['identity']['st_dev'] and fd['ino']==expected['identity']['st_ino'] and fd['bytes']==expected['bytes'],'ActualFDidentity')
        if case['kind']!='binding' and (case['kind'] in ('sizing','screen') and local or len(row['callbacks'])>=case['streams']):require(len({c['stream'] for c in row['callbacks']})==case['streams'],'Streamcount')
        require(row['witness_values']==(len(local)*96 if case['kind']=='tail' else 0),'Witness coverage')
    require(any(len({c['stream'] for c in row['callbacks']})==case['streams'] for row in bds),'Depth not exercised')
    sizes=[c['size'] for row in bds for c in row['callbacks']];require(sorted(sizes)==sorted([200]*(n//200)+([n%200] if n%200 else [])),'Globalcallbackhist')
    gate=r['measurement']['cache'];allpaths={x['path'] for s in info['sets'] for x in s['files']}
    require(len(gate['all30_scans'])==2,'Doublecachegate')
    for scan in gate['all30_scans']:require(len(scan)==30 and {x['path'] for x in scan}==allpaths and all(x['resident']==0 for x in scan),'Globalcachegate')
    require(len(gate['selected'])==5 and {x['path'] for x in gate['selected']}==set(paths) and all(x['resident']==0 for x in gate['selected']),'Selectedcachegate')
    return {'id':case['id'],'seconds':r['seconds'],'gib_s':r['gib_s'],'events':n,'bds':case['bds'],'streams':case['streams'],'count':case['count'],'kind':case['kind']}

def audit(attempt,reviewed_commit):
    output=attempt/'results';complete=json.loads((output/'complete.json').read_text());info=load_inputs();cases=schedule(complete['mode'])
    require(complete['bundle']['commit']==reviewed_commit,'Unexpected reviewed commit')
    require(complete==json.loads((output/'status.json').read_text()) and complete['state']=='cases_passed_pending_slurm','Notcomplete')
    require(json.loads((output/'schedule.json').read_text())=={'mode':complete['mode'],'seed':20261115,'cases':cases},'Schedule')
    require(len(complete['completed'])==len(cases),'Missingcases')
    plan=complete['launch_plan'];check_allocation(plan,(attempt/'slurm-job.txt').read_text(),complete['mode'])
    require(plan==json.loads((attempt/'launch-plan.json').read_text()) and sha(plan['ledger_path'])==plan['ledger_sha256'],'Launch plan provenance')
    records=list(csv.DictReader((attempt/'accounting-utc.txt').open(),delimiter='|'));job=complete['job']
    require({x['JobID'] for x in records}>={job,job+'.batch',job+'.extern'},'Slurmrecords')
    require(all(x['State']=='COMPLETED' and x['ExitCode']=='0:0' for x in records),'Slurmfailure')
    parent=next(x for x in records if x['JobID']==job);require(parent['Account']=='lcls_g' and parent['QOS']=='gpu_premium','QoSaccount')
    ranges=json.loads((output/'ranges.json').read_text());idx={t:i for i,t in enumerate(info['timestamps'])};result=[];affinity=json.loads((output/'affinity.json').read_text());affinity_set=select_affinity(affinity['probes'])
    require(affinity_set==affinity['cpus'] and sorted(r['rank'] for r in affinity['probes'])==list(range(6)),'Affinity probes')
    for case,summary in zip(cases,complete['completed']):
        file=output/(case['id']+'.json');require(summary['case']==case and summary['result_sha256']==sha(file),'Casehash')
        r=json.loads(file.read_text())
        require(all(row['affinity']==affinity_set[str(row['rank'])] for row in r['rows']),'Cross-case affinity')
        result.append(audit_case(r,case,info));require(r['job']==job,'Jobidentity')
        mf=output/(case['id']+'.memory.json');require(sha(mf)==summary['memory_sha256'],'Memory receipt changed')
        mr=json.loads(mf.read_text());require(mr['uuid']==r['rows'][2]['device']['uuid'],'Memory GPU mismatch')
        validate_memory(mr,r['measurement']['started_unix'],r['measurement']['finished_unix'])
        result[-1].update(whole_case_seconds=summary['wall_seconds'],sampled_device_peak_gib=mr['max_used']/2**30,sampled_device_min_free_gib=mr['min_free']/2**30,
            tracked_peak_sum_gib=sum(p['peak_committed_plus_held'] for row in r['rows'][2:] for p in row['budget_high_water'])/2**30)
        require(result[-1]['tracked_peak_sum_gib']<=24,'Aggregate tracked budget')
        require(re.findall(r'(?:^|\s)NodeList=(\S+)',(attempt/'slurm-job.txt').read_text())==[r['host']],'Exact allocated node')
        for row in r['rows'][2:]:
            for path,digest in row['io']['range_sha256'].items():
                ref=ranges[Path(path).name];h=hashlib.sha256()
                for t in row['timestamps']:h.update(struct.pack('<QQQ',t,ref['offsets'][idx[t]],ref['size']))
                require(h.hexdigest()==digest,'Physicalrangehash')
    for name in ['gpu-tests','input-validation']:
        data=json.loads((output/(name+'.json')).read_text());require(data.get('state')=='pass' if name=='input-validation' else data['exitcode']==0,'Preflight')
    tests=json.loads((output/'gpu-tests.json').read_text())
    expected=json.loads((attempt/'bundle/expected-native.json').read_text())
    require(tests['collected']==expected['collected'] and tests['sources']==expected['sources'],'Native test inventory/source hashes')
    validation=json.loads((output/'input-validation.json').read_text())
    for name in ('witness','ranges'):require(sha(output/(name+'.json'))==validation[name+'_sha256'],'Validation artifact hash')
    require(tests['collected'] and not any(x['outcome'] in ('skipped','failed') for x in tests['reports']),'Native test skips/failures')
    require(sum(x['when']=='call' and x['outcome']=='passed' for x in tests['reports'])==len(tests['collected']),'Missing native test calls')
    expected_steps=len(cases)+3
    require(len([x for x in records if x['JobID'].startswith(job+'.') and x['JobID'].split('.')[-1].isdigit()])==expected_steps,'Missing Slurm steps')
    pins=json.loads((attempt/'bundle/external-pins.json').read_text())
    require(all(sha(path)==h for path,h in pins.items()),'External provenance changed')
    for name,h in complete['bundle']['files'].items():require(sha(attempt/'bundle'/name)==h,'Harness provenance changed')
    provenance=json.loads(Path(next(path for path in pins if path.endswith('build-provenance.json'))).read_text())
    require(provenance['commit']==SHA,'Build commit')
    for case in cases:
        r=json.loads((output/(case['id']+'.json')).read_text())
        for row in r['rows']:
            for entry in row['runtime'].values():
                rel=entry['path'].split('/source/install_psana/',1)[1]
                require(provenance['installed'].get(rel)==entry['sha256'],'Runtime source hash')
    verify_payloads(info)
    return {'state':'pass','job':job,'mode':complete['mode'],'cases':result,'scope':'staging only; local page cache cold; no storage ceiling/remote bytes claim'}
if __name__=='__main__':
    import sys
    from support import write
    attempt=Path(sys.argv[1]);r=audit(attempt,sys.argv[2]);write(attempt/'results/audit.json',r);print(json.dumps(r,indent=2))
