"""Balanced public-loop comparisons with frozen reference/candidate runtimes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event-runtime',type=Path,required=True)
    p.add_argument('--batch-runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--profiles',nargs='+',choices=('micro','jungfrau'),default=['micro','jungfrau'])
    p.add_argument('--modes',nargs='+',choices=('compact','compact_prealloc','image','image_fresh_host','image_pinned'),default=['compact','compact_prealloc','image'])
    p.add_argument('--batch-sizes',nargs='+',type=int,choices=(1,3,20),default=[1,3,20])
    p.add_argument('--depths',nargs='+',type=int,choices=(1,2),default=[1,2])
    p.add_argument('--micro-submissions',type=int,default=1000)
    p.add_argument('--jungfrau-submissions',type=int,default=100)
    a=p.parse_args()
    if a.rounds<2 or a.rounds%2:p.error('use an even number of rounds >=2')
    a.output.mkdir(parents=True,exist_ok=False)
    cases=[];start=time.monotonic()
    for round_id in range(1,a.rounds+1):
        profiles=a.profiles if round_id%2 else list(reversed(a.profiles))
        order=('event','batch') if round_id%2 else ('batch','event')
        for profile in profiles:
            for dispatch in order:
                runtime=a.event_runtime if dispatch=='event' else a.batch_runtime
                name=f'r{round_id}-{profile}-{dispatch}'
                output=a.output/(name+'.json')
                command=[sys.executable,str(Path(__file__).with_name('publication_cost.py')),
                    '--output',str(output),'--dispatch',dispatch,'--profile',profile,'--submissions',
                    str(a.micro_submissions if profile=='micro' else a.jungfrau_submissions)]
                command+=['--modes',*a.modes,'--batch-sizes',*map(str,a.batch_sizes),'--depths',*map(str,a.depths)]
                if round_id%2==0:command.append('--reverse')
                print('PUBLICATION_CASE_START',name,flush=True);begin=time.monotonic()
                with (a.output/(name+'.log')).open('w') as log:
                    subprocess.run(command,env=dict(os.environ,PYTHONPATH=str(runtime)),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                d=json.loads(output.read_text())
                assert d['complete'] and len(d['samples'])==len(d['preflights'])==len(a.modes)*len(a.batch_sizes)*len(a.depths)
                assert Path(d['psana']).resolve().is_relative_to(runtime.resolve())
                cases.append(dict(round=round_id,profile=profile,dispatch=dispatch,
                                  seconds=time.monotonic()-begin,output=str(output),data=d))
                (a.output/'progress.json').write_text(json.dumps([{k:v for k,v in c.items() if k!='data'} for c in cases],indent=2)+'\n')
                print('PUBLICATION_CASE_COMPLETE',name,round(time.monotonic()-begin,2),flush=True)
    summary=[]
    for profile in a.profiles:
        for size in a.batch_sizes:
            for depth in a.depths:
                for mode in a.modes:
                    pairs=[]
                    for round_id in range(1,a.rounds+1):
                        pair={}
                        for dispatch in ('event','batch'):
                            c=next(c for c in cases if (c['round'],c['profile'],c['dispatch'])==(round_id,profile,dispatch))
                            pair[dispatch]=next(r for r in c['data']['samples'] if (r['batch_size'],r['depth'],r['mode'])==(size,depth,mode))
                        pairs.append(pair)
                    key='loop_us_per_event'
                    percents=[100*(p['batch'][key]/p['event'][key]-1) for p in pairs]
                    summary.append(dict(profile=profile,batch_size=size,depth=depth,mode=mode,
                        event_median_us=statistics.median(p['event'][key] for p in pairs),
                        batch_median_us=statistics.median(p['batch'][key] for p in pairs),
                        paired_percent=percents,median_paired_percent=statistics.median(percents),
                        paired_delta_us=[p['batch'][key]-p['event'][key] for p in pairs]))
    (a.output/'summary.json').write_text(json.dumps(dict(complete=True,rounds=a.rounds,profiles=a.profiles,modes=a.modes,
        wall_seconds=time.monotonic()-start,event_runtime=str(a.event_runtime),
        batch_runtime=str(a.batch_runtime),summary=summary),indent=2)+'\n')
    print('PUBLICATION_COMPARISON_COMPLETE',round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':
    main()
