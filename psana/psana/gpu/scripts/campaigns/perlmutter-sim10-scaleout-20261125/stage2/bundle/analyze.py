"""Descriptive three-pair regression screen; no statistical equivalence claim."""
import math,statistics

def screen(pairs):
 if len(pairs)!=3:raise ValueError('Exactly three fixed pairs')
 if any(a<=0 or b<=0 or not math.isfinite(a+b) for a,b in pairs):raise ValueError('Finite positive loop times')
 ratios=[b/a for a,b in pairs]
 change=100*(statistics.median(ratios)-1)
 cvs=[100*statistics.stdev(x)/statistics.mean(x) for x in zip(*pairs)]
 return dict(status='regression_signal' if change>10 else 'no_large_slowdown_observed',pairs=3,median_time_change_percent=change,paired_time_ratios=ratios,loop_time_cv_percent=dict(pre_sync=cvs[0],candidate=cvs[1]),noisy=any(x>5 for x in cvs),positive='candidate slower',method='descriptive median paired loop-time ratio; 10% slowdown screen, 5% CV noise flag; not equivalence')

if __name__=='__main__':
 import json
 from pathlib import Path
 from design import plans,ROOT
 from gate import completed,bundle_identity
 groups=[completed(Path(p['attempt'])) for p in plans()]
 identities=[bundle_identity(p) for p,s,u in groups]
 if not all(i==identities[0] for i in identities):raise ValueError('Mixed harness or input identities')
 result=[]
 for bd in (1,4):
  for stage in ('S','E','EC','ECC'):
   pairs=[]
   for plan,state,summary in groups:
    if plan['bds']!=bd:continue
    cells={e['case']['role']:e['seconds'] for e in state['completed'] if e['case']['kind']=='full' and e['case']['stage']==stage}
    pairs.append((cells['P'],cells['C']))
   result.append(dict(bds=bd,stage=stage,**screen(pairs)))
 print(json.dumps(dict(scope='direct pre-sync versus pinned PR331 regression screen',results=result),indent=2))
