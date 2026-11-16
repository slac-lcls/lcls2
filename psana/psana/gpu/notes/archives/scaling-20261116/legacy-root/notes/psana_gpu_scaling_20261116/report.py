"""Compute paired strong-scaling results only after a passing raw-evidence audit."""
import json,statistics,sys
from pathlib import Path
from audit import audit
from support import write,require

def report(attempt,commit):
    result=audit(attempt,commit);require(result['mode']=='screen','Screen required')
    cases=result['cases'];groups={g:[r for r in cases if r['gpus']==g] for g in range(1,5)}
    baseline={r['id'].split('-')[0]:r['gib_s'] for r in groups[1]};rows=[]
    for g,group in groups.items():
        speeds=[r['gib_s'] for r in group];ratios=[r['gib_s']/baseline[r['id'].split('-')[0]] for r in group]
        rows.append(dict(gpus=g,bds=4*g,median_gib_s=statistics.median(speeds),min_gib_s=min(speeds),max_gib_s=max(speeds),median_paired_speedup=statistics.median(ratios),efficiency=statistics.median(ratios)/g,per_gpu_gib_s=statistics.median(speeds)/g))
    write(attempt/'results/scaling-summary.json',dict(rows=rows,scope=result['scope']))
    print('| GPUs | BDs | Median GiB/s | Range | Paired speedup | Efficiency |')
    print('|---:|---:|---:|---:|---:|---:|')
    for r in rows:print('| {gpus} | {bds} | {median_gib_s:.3f} | {min_gib_s:.3f}–{max_gib_s:.3f} | {median_paired_speedup:.2f}× | {efficiency:.1%} |'.format(**r))

if __name__=='__main__':report(Path(sys.argv[1]).resolve(),sys.argv[2])
