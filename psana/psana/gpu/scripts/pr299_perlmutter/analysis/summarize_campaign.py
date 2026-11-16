"""Summarize completed, paired run51 samples without converting failures into passes."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(directory):
    status=json.loads((directory/'status.json').read_text())
    if status.get('phase')!='campaign' or not status.get('complete'):
        raise ValueError('Campaign is not complete; no performance verdict')
    pairs=json.loads((directory/'paired-results.json').read_text())
    if len(pairs)!=24:
        raise ValueError('Expected 6 cases * 4 complete pairs')
    groups={}
    for pair in pairs:
        left=json.loads((directory/pair['left']/'sample.json').read_text())
        right=json.loads((directory/pair['right']/'sample.json').read_text())
        if any(not s['complete'] or s['events']!=10000 or s['n_gpu_streams']!=1 or s['preflight'] for s in [left,right]):
            raise ValueError('Incomplete or mismatched workload')
        if left['runtime']!='base' or right['runtime']!='head' or left['timestamp_sha256']!=right['timestamp_sha256']:
            raise ValueError('Paired runtime/timestamps mismatch')
        if left['case']!=right['case'] or left['case']!=pair['case']:
            raise ValueError('Case mismatch')
        groups.setdefault(pair['case'],[]).append((left,right))
    results=[]
    for case,values in sorted(groups.items()):
        if len(values)!=4: raise ValueError('Missing alternating pairs')
        def pair_number(value):
            left,right=value
            number=int(left['label'].rsplit('-pair',1)[1].split('-')[0])
            if not left['label'].endswith('-0') or right['label'] != left['label'][:-1]+'1':
                raise ValueError('Invalid paired labels')
            return number
        values=sorted(values,key=pair_number)
        if [pair_number(value) for value in values] != [0,1,2,3]:raise ValueError('Missing/duplicate alternating round')
        placement_equal=[]
        for left,right in values:
            def placement(sample):
                return [(r['rank'],r['physical_cores'],r['gpu_uuid']) for r in sorted(sample['rows'],key=lambda r:r['rank'])]
            placement_equal.append(placement(left)==placement(right))
        rates=[100*(right['events_per_second']/left['events_per_second']-1) for left,right in values]
        ratios=[right['loop_seconds']/left['loop_seconds'] for left,right in values]
        # pair0/2 run base then head; pair1/3 reverse. Preserve order sensitivity.
        head_second=statistics.median(rates[0::2])
        head_first=statistics.median(rates[1::2])
        io_pairs=[]
        for left,right in values:
            totals=[]
            for sample in (left,right):
                summaries=[row['cumulative_io'] for row in sample['rows'] if row.get('cumulative_io')]
                totals.append(sum(item['total_bytes'] for item in summaries) if summaries else None)
            io_pairs.append(totals)
        results.append(dict(case=case,pairs=4,base_loop_median_s=statistics.median(a['loop_seconds'] for a,b in values),
            head_loop_median_s=statistics.median(b['loop_seconds'] for a,b in values),
            paired_throughput_change_percent=rates,paired_throughput_median_percent=statistics.median(rates),
            paired_throughput_range_percent=[min(rates),max(rates)],
            head_second_median_percent=head_second,head_first_median_percent=head_first,
            order_median_gap_percentage_points=head_second-head_first,
            base_head_completed_io_bytes_by_pair=io_pairs,
            paired_physical_core_and_gpu_placement_equal=placement_equal,
            paired_loop_geomean_change_percent=100*(math.exp(statistics.mean(math.log(x) for x in ratios))-1),
            base_process_wall_median_s=statistics.median(a['mpi_process_wall_seconds'] for a,b in values),
            head_process_wall_median_s=statistics.median(b['mpi_process_wall_seconds'] for a,b in values)))
    if len(results)!=6: raise ValueError('Missing cases')
    return dict(manifest_sha256=status['manifest_sha256'],complete_samples=48,results=results,
        run77='Deferred; local data unavailable. Full eight-case coverage NOT achieved.',
        c1='Known CPU Step-parent lifetime blocker remains unfixed; not merge approval.',
        interpretation='Local base/head ratios only. CPU controls do not read the same workload as GPU. Uncontrolled Lustre/cache state requires separate storage/progress/order analysis before any regression conclusion.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory',type=Path); args=parser.parse_args()
    print(json.dumps(summarize(args.directory),indent=2))
