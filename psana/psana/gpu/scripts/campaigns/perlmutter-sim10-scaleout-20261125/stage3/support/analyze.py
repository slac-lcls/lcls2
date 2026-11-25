"""Descriptive weak scaling; short loops are labelled, never silently enlarged."""
import math,statistics


def summarize(state):
    rows=[];drift=[]
    for stage in ('S','E','EC','ECC'):
        cells=[e for e in state['completed'] if e['case']['kind']=='full' and e['case']['stage']==stage]
        if len(cells)!=3:raise ValueError('Three repeats required')
        times=[e['seconds'] for e in cells]
        if not all(math.isfinite(t) and t>0 for t in times):raise ValueError('Finite times')
        nodes=cells[0]['case']['nodes'];events=100000*nodes
        rates=[events/t for t in times];cv=100*statistics.stdev(times)/statistics.mean(times)
        rows.append(dict(stage=stage,nodes=nodes,events=events,loop_seconds=times,
                         median_rate=statistics.median(rates),time_cv_percent=cv,noisy=cv>5,
                         median_per_node_rate=statistics.median(rates)/nodes))
    for repeat in (1,2,3):
        byname={e['case']['name']:e for e in state['completed']}
        ratio=byname['r%d-S'%repeat]['seconds']/byname['r%d-anchor-S'%repeat]['seconds']
        if abs(ratio-1)>.05:raise ValueError('Drift exceeds 5%')
        drift.append(dict(repeat=repeat,rate_ratio=ratio))
    return dict(status='pass',job=state['job'],hosts=state['hosts'],results=rows,drift=drift,
                method='Whole-loop rate, median of 3; weak efficiency R(N)/(N*R(1)); not fixed-work speedup')


def windows(ticks_by_rank,seconds):
    # Counts are sampled at every 200th delivered event. Binning is a diagnostic
    # approximation, not a replacement for the exact whole-loop denominator.
    count=int((seconds-10)//10)
    if count<4:return dict(status='inconclusive',reason='Fewer than four complete interior 10-second windows')
    values=[]
    for i in range(count):
        lo=5+10*i;hi=lo+10
        values.append(sum(200 for ticks in ticks_by_rank for t,n in ticks if lo<=t<hi)/10)
    early=statistics.mean(values[:2]);late=statistics.mean(values[-2:])
    return dict(status='diagnostic_only',rates=values,late_over_early=late/early if early else None,
                limitation='200-event clock samples; edge bins and cross-rank clock start skew; no plateau guarantee')


if __name__=='__main__':
    import json
    from pathlib import Path
    from design import plans
    from gate import completed,identity
    groups=[completed(Path(p['attempt'])) for p in plans()]
    pins=[identity(p) for p,s,u in groups]
    if not all(p==pins[0] for p in pins):raise ValueError('Mixed input or harness')
    references={r['stage']:r['median_rate'] for r in groups[0][2]['results']}
    result=[]
    for p,state,summary in groups:
        for r in summary['results']:
            result.append(dict(**r,weak_efficiency=r['median_rate']/(r['nodes']*references[r['stage']])))
    print(json.dumps(dict(scope='1/2/4-node fixed work per node',results=result),indent=2))
