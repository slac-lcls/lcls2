"""Verify a completed campaign and emit evidence for the timing review."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from pairs import validate_pair
from run import POINTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',required=True)
    a=p.parse_args();root=a.root;output=root/('job-'+a.job)
    provenance=json.loads((output/'provenance.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    assert provenance['complete'] and summary['complete'] and not summary['preflight_only']
    assert 'STAGE5C_CAMPAIGN_COMPLETE' in (root/('job-'+a.job+'.log')).read_text()
    for name,expected in json.loads((root/'hashes.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    rows=json.loads((output/'results.json').read_text())
    assert len(rows)==92 and sum(x['diagnostic'] for x in rows)==16
    groups={}
    diagnostics=[]
    for row in rows:
        assert hashlib.sha256((output/(row['tag']+'.log')).read_bytes()).hexdigest()==row['log_sha256']
        key=(row['ngpus'],row['nbds'],row['batch_size'],row['depth'],row['cache'],row['round'],row['diagnostic'])
        values=groups.setdefault(key,{})
        assert row['variant'] not in values
        values[row['variant']]=row
        workers=[r for r in row['ranks'] if r['is_bd']]
        assert len(workers)==row['nbds'] and all(r['timestamps'] for r in workers)
        assert sum(len(r['timestamps']) for r in workers)==row['events']
        if row['diagnostic']:
            for r in workers:
                work=r['public_result'];n=work['events']
                calls=n if row['variant']=='event_loop' else work['framework_callbacks']
                assert work['analysis_calls']==calls
                assert work['actual_kernel_launches']==2*calls and work['copy_groups']==calls
                assert sum(work['callback_sizes'])==n
                assert all(v>0 for v in work['kernel_ms'].values())
            diagnostics.append(dict(point=key[:4],variant=row['variant'],
                calibration_ms_per_event=sum(r['public_result']['kernel_ms']['calibrate'] for r in workers)/row['events'],
                integration_ms_per_event=sum(r['public_result']['kernel_ms']['integrate'] for r in workers)/row['events'],
                events_by_bd=[len(r['timestamps']) for r in workers]))
    comparisons=[]
    for key,values in groups.items():
        change=validate_pair(values['event_loop'],values['batched_task'])
        if not key[-1]:comparisons.append(dict(point=key[:4],cache=key[4],round=key[5],**change))
    assert len(comparisons)==38
    results=[]
    for point,cache in [(p,'warm') for p in POINTS]+[(POINTS[0],'cold')]:
        matches=[x for x in comparisons if x['point']==point and x['cache']==cache]
        assert len(matches)==(4 if cache=='cold' else 6 if point==POINTS[0] else 4)
        selected=[x for x in rows if not x['diagnostic'] and
                  (x['ngpus'],x['nbds'],x['batch_size'],x['depth'])==point and x['cache']==cache]
        variants={v:[x for x in selected if x['variant']==v] for v in ('event_loop','batched_task')}
        results.append(dict(point=point,cache=cache,pairs=len(matches),
            median_paired_percent=statistics.median(x['percent'] for x in matches),
            paired_percent_range=[min(x['percent'] for x in matches),max(x['percent'] for x in matches)],
            variants={v:dict(median_loop_s=statistics.median(x['loop_s'] for x in xs),
                median_rate=10000/statistics.median(x['loop_s'] for x in xs),
                median_setup_s=statistics.median(max(r['setup_s'] for r in x['ranks']) for x in xs),
                median_wall_s=statistics.median(x['wall_s'] for x in xs)) for v,xs in variants.items()}))
    print(json.dumps(dict(verified=True,diagnostics=diagnostics,results=results),indent=2))


if __name__=='__main__':main()
