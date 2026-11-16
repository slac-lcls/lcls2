"""Measure cross-stream CUDA kernel overlap from a single-process Nsight export.

Full capture, including warmup. Device intervals, not CUDA API durations or
summed CUDA event spans. This does not measure copy/compute overlap.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

TASK_NAMES = {'expand10', 'calibrate', 'accept_events', 'accumulate'}


def intervals(rows):
    """Union time and time with >=2 distinct active streams; half-open intervals."""
    if not rows:
        raise ValueError('No kernel intervals')
    edges = defaultdict(Counter)
    for start, end, stream in rows:
        if end <= start:
            raise ValueError('Invalid kernel interval')
        edges[start][stream] += 1
        edges[end][stream] -= 1
    active = Counter()
    previous = min(edges)
    busy = overlap = maximum = 0
    for timestamp in sorted(edges):
        depth = sum(count > 0 for count in active.values())
        duration = timestamp - previous
        if depth:
            busy += duration
        if depth >= 2:
            overlap += duration
        maximum = max(maximum, depth)
        active.update(edges[timestamp])
        if any(count < 0 for count in active.values()):
            raise ValueError('Unbalanced kernel interval')
        previous = timestamp
    return dict(kernel_union_ns=busy, cross_stream_overlap_ns=overlap,
                overlap_fraction_of_kernel_busy_time=overlap/busy,
                max_concurrent_streams=maximum,
                first_start_ns=min(edges), last_end_ns=max(edges))


def analyze(path, report):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
        strings = dict(db.execute('select id,value from StringIds'))
        rows = list(db.execute('select start,end,streamId,deviceId,contextId,globalPid,demangledName,correlationId '
                               'from CUPTI_ACTIVITY_KIND_KERNEL'))
        if not rows or len({(r[3],r[4],r[5]) for r in rows}) != 1:
            raise ValueError('Expected one GPU/context/process with CUDA kernels')
        kernels = [dict(start=r[0],end=r[1],stream=r[2],name=strings[r[6]],correlation=r[7]) for r in rows]
        task = [r for r in kernels if r['name'].split('(')[0].strip() in TASK_NAMES]
        counts = Counter(r['name'].split('(')[0].strip() for r in task)
        callbacks = len(report['callbacks'])
        if report['status'] != 'pass' or counts != Counter({n:callbacks for n in TASK_NAMES}):
            raise ValueError('Task kernel counts do not match accepted callbacks: '+str(counts))
        streams = sorted({r['stream'] for r in task})
        if len(streams) != report['streams']:
            raise ValueError('Task did not exercise every configured CUDA stream')
        executions = []
        for stream in streams:
            expansions = sorted((r for r in task if r['stream']==stream and r['name']=='expand10'), key=lambda r:r['start'])
            sums = sorted((r for r in task if r['stream']==stream and r['name']=='accumulate'), key=lambda r:r['start'])
            if len(expansions) != len(sums):
                raise ValueError('Unmatched task kernels on execution stream')
            for expansion, cube_sum in zip(expansions, sums):
                launches = list(db.execute('select start,end from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=?',
                                           (expansion['correlation'],)))
                if len(launches) != 1 or expansion['start'] >= cube_sum['end']:
                    raise ValueError('Ambiguous expansion launch or task ordering')
                executions.append(dict(stream=stream, launch_start=launches[0][0], sum_end=cube_sum['end']))
        executions.sort(key=lambda r:r['launch_start'])
        gaps = [b['launch_start']-a['sum_end'] for a,b in zip(executions,executions[1:])]
        handoffs = dict(count=len(gaps), next_launch_before_previous_sum_end=sum(g<0 for g in gaps),
                        min_launch_gap_ns=min(gaps), max_launch_gap_ns=max(gaps),
                        method='Pair expansion and sum FIFO within each stream; order executions by expansion launch API start. Gap = next launch API start minus previous sum device end.')
        groups = defaultdict(lambda: dict(count=0,duration_ns=0))
        for r in kernels:
            g = groups[(r['name'],r['stream'])]
            g['count'] += 1
            g['duration_ns'] += r['end']-r['start']
        api = []
        for name,count,total in db.execute('select nameId,count(*),sum(end-start) '
                                          'from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId'):
            api.append(dict(name=strings[name],count=count,total_ns=total))
    return dict(path=str(path), scope='Full 1k trace including warmup; no steady-only or copy/compute overlap claim',
                device_id=rows[0][3],context_id=rows[0][4],global_pid=rows[0][5],
                callbacks=callbacks,task_streams=streams,task_kernel_counts=dict(counts),
                submission_handoffs=handoffs,
                task_kernels=intervals([(r['start'],r['end'],r['stream']) for r in task]),
                all_kernels=intervals([(r['start'],r['end'],r['stream']) for r in kernels]),
                kernel_groups=[dict(name=k[0],stream=k[1],**v) for k,v in sorted(groups.items())],
                runtime_api=sorted(api,key=lambda r:r['total_ns'],reverse=True),
                interpretation='API totals are inclusive and not critical-path attribution. Nsight perturbs timing; compare throughput using separate clean runs.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite',type=Path)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.sqlite,json.loads(args.report.read_text()))
    with args.output.open('x') as out:
        out.write(json.dumps(result,indent=2)+'\n')
