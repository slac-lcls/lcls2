"""Summarize delivered-event windows and observed job memory pressure."""
import argparse
import json
from pathlib import Path


def analyze(directory):
    result = dict(source=str(directory),progress={},limitations=['Node/network counters are not process-attributed. Cache state is uncontrolled.'])
    for path in sorted(directory.glob('progress-rank-*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        windows=[]
        # Non-overlapping reported 1000-event windows, retaining actual endpoints.
        selected=[row for row in rows if row['events']%1000==0]
        for left,right in zip(selected,selected[1:]):
            windows.append(dict(first_events=left['events'],last_events=right['events'],seconds=right['seconds']-left['seconds'],events_per_second=(right['events']-left['events'])/(right['seconds']-left['seconds'])))
        result['progress'][path.name]=dict(last=rows[-1] if rows else None,thousand_event_windows=windows)
    first=last=None
    peaks={}
    errors=set()
    periods=[]
    job_memory_stats={}
    with (directory/'storage.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if first is None:first=row
            if last is not None:periods.append(row['monotonic']-last['monotonic'])
            last=row
            errors.update(row['errors'])
            for name,value in row['values'].items():
                if name.endswith('memory.current'):
                    peaks[name]=max(peaks.get(name,0),int(value))
                if '/job_' in name and '/step_' not in name and name.endswith('memory.stat'):
                    parsed={k:int(v) for k,v in (line.split() for line in value.splitlines())}
                    for key in ['anon','file','pgscan','pgsteal','workingset_refault_file','workingset_activate_file']:
                        if key in parsed:
                            job_memory_stats.setdefault(key,dict(first=parsed[key],last=parsed[key],peak=parsed[key]))
                            job_memory_stats[key]['last']=parsed[key]
                            job_memory_stats[key]['peak']=max(job_memory_stats[key]['peak'],parsed[key])
    result.update(storage_samples=len(periods)+1 if first else 0,counter_read_error_paths=sorted(errors),memory_peak_bytes_by_path=peaks,job_memory_stat=job_memory_stats)
    if periods:result['observed_monitor_period_seconds']=dict(min=min(periods),max=max(periods),mean=sum(periods)/len(periods))
    if first and last:
        result['memory_events']={}
        for name,raw in first['values'].items():
            if not name.endswith('memory.events') or name not in last['values']:continue
            begin={k:int(v) for k,v in (line.split() for line in raw.splitlines())}
            end={k:int(v) for k,v in (line.split() for line in last['values'][name].splitlines())}
            result['memory_events'][name]={k:end[k]-v for k,v in begin.items()}
        result['network_rx_bytes_delta']={name:int(last['values'][name])-int(raw) for name,raw in first['values'].items() if name.endswith('/statistics/rx_bytes') and name in last['values']}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps(analyze(args.directory),indent=2))
