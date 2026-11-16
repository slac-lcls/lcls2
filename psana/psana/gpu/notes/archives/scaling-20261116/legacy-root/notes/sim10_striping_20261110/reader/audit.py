"""Re-audit all raw receipts and derive screening-only descriptive summaries."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import campaign as c
import bundle as snapshot


def audit(attempt):
    attempt = attempt.resolve()
    root = attempt / 'results'
    complete = json.loads((root / 'complete.json').read_text())
    provenance = snapshot.verify(attempt)
    c.require(complete['state'] == 'complete' and complete['bundle_provenance'] == provenance, 'incomplete/provenance mismatch')
    c.require(json.loads((root / 'status.json').read_text()) == complete, 'status/complete disagreement')
    with (attempt / 'accounting-utc.txt').open() as f:
        accounting = list(csv.DictReader(f, delimiter='|'))
    job = complete['job']
    c.require({r['JobID'] for r in accounting} >= {job, job+'.batch', job+'.extern', job+'.0'}, 'missing Slurm records')
    c.require(all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0' for r in accounting), 'Slurm not successful')
    parent = next(r for r in accounting if r['JobID'] == job)
    c.require(parent['Account'] == 'lcls_g' and parent['QOS'] == 'gpu_premium', 'wrong account/QoS')
    c.require('NodeList='+complete['host'] in (attempt / 'slurm-job.txt').read_text(), 'native node mismatch')
    c.require(c.sha(attempt / 'bundle/staging-complete.json') == c.STAGING_SHA256, 'wrong input receipt')
    staging = json.loads((attempt / 'bundle/staging-complete.json').read_text())
    inputs = {g['count']: sorted(g['verified'], key=lambda r: r['path']) for g in staging['sets']}
    expected = c.schedule(complete['mode'])
    c.require(json.loads((root / 'schedule.json').read_text()) == {'mode': complete['mode'], 'seed': c.SEED, 'cases': expected}, 'schedule changed')
    c.require(len(complete['completed']) == len(expected) == complete['case_count'], 'missing completed trials')
    topology = json.loads((root / 'cpu-map.json').read_text())[:40]
    c.require(len(topology) == 40 and len({(r['package'], r['core']) for r in topology}) == 40, 'not40 distinct physical cores')
    cpus = [r['cpu'] for r in topology]
    c.require(len(set(cpus)) == 40, 'duplicate logical CPUs')
    rows=[]
    for expected_case, finished in zip(expected, complete['completed']):
        r=json.loads((root / (expected_case['id'] + '.json')).read_text())
        raw=root / (expected_case['id'] + '.stdout.json')
        c.require(r['case'] == expected_case and finished['id'] == expected_case['id'], 'wrong case order')
        c.require(r['audit'] == 'pass' and r['raw_sha256'] == c.sha(raw), 'raw hash mismatch')
        measured=json.loads(raw.read_text())
        c.require(r['reader'] == measured, 'parsed receipt differs from raw')
        c.require(r['reader_sha256'] == provenance['files']['reader']['sha256'], 'binary identity mismatch')
        count,threads=expected_case['count'],expected_case['threads']
        warm=complete['mode']=='pilot'
        wanted=[str(attempt / 'bundle/reader'), str(threads), str(c.REQUEST), ','.join(map(str,cpus[:threads])), '1' if warm else '0', *[x['path'] for x in inputs[count]]]
        c.require(r['command'] == wanted, 'wrong executed command')
        c.audit_reader(measured,expected_case,inputs[count],cpus[:threads],warm)
        eviction = root / (expected_case['id'] + '.eviction.json')
        c.require(c.sha(eviction) == r['eviction_sha256'] and json.loads(eviction.read_text()) == r['global_eviction'], 'global eviction receipt mismatch')
        c.audit_global_cache(r['global_eviction'], [row for count in c.COUNTS for row in inputs[count]])
        c.audit_memory(r['memory_before'])
        c.audit_memory(r['memory_after'])
        c.require(all(w['test_checksum'] == 0 for w in measured['workers']), 'test binary measured')
        c.require(finished['gib_per_second'] == measured['gib_per_second'] and finished['seconds'] == measured['seconds'], 'summary mismatch')
        rows.append(dict(expected_case, seconds=measured['seconds'], gib_per_second=measured['gib_per_second'],
                         eviction_seconds=measured['eviction_seconds'], wall_seconds=r['finished_unix']-r['started_unix'],
                         max_start_skew_seconds=max(w['start_seconds'] for w in measured['workers'])))
    # Fresh read-only metadata/layout checks supplement the runner's before/after checks.
    c.validate_inputs(staging)
    result={'state':'pass','mode':complete['mode'],'job':complete['job'],'trials':len(rows),'rows':rows,
            'scope':complete['cache_scope'],'recommendation':'screening only; independent-placement confirmation still required'}
    if complete['mode']=='pilot':
        result['sweep_estimate_seconds']=54*max(r['wall_seconds'] for r in rows)
        result['suggested_walltime_seconds']=math.ceil((result['sweep_estimate_seconds']*1.5+300)/60)*60
    else:
        cells=[]
        for threads in c.THREADS:
            baseline={r['block']:r['gib_per_second'] for r in rows if r['count']==1 and r['threads']==threads}
            for count in c.COUNTS:
                group=[r for r in rows if r['count']==count and r['threads']==threads]
                rates=[r['gib_per_second'] for r in group]
                ratios=[r['gib_per_second']/baseline[r['block']] for r in group]
                cells.append({'count':count,'threads':threads,'median_gib_s':statistics.median(rates),
                              'min_gib_s':min(rates),'max_gib_s':max(rates),'paired_geomean_vs_count1':statistics.geometric_mean(ratios)})
        result['cells']=cells
    c.save(root / 'audit.json',result)
    with (root / 'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);args=p.parse_args()
    result=audit(args.attempt)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','cells')},indent=2))
