"""Conservative timing gates backed by the independent real-input kernel check.

These gates check measurement consistency, not whether batching wins. A failure
holds subsequent scaling work for inspection instead of accepting partial data.
"""
import argparse
import json
import statistics
from pathlib import Path


def check(review,kernel):
    assert review['verified'] and kernel['total_valid_pixels']>0 and kernel['nonzero_sums']>0
    bandwidth=2*kernel['device']['memoryClockRate']*1000*kernel['device']['memoryBusWidth']/8
    pixels=1
    for n in kernel['shape']:pixels*=n
    # Raw uint16 input plus float32 image and uint8 validity output. Ignoring
    # constants gives a conservative lower bound on calibration memory traffic.
    minimum_calibration_ms=pixels*7/bandwidth*1000
    base=kernel['median_ms']
    for batch in (1,20):
        assert base[str(batch)]['calibrate']/batch>=minimum_calibration_ms*.8
    evidence=[]
    for row in review['diagnostics']:
        g,b,n,d=row['point']
        if (g,b,n)!=(1,1,20):continue
        batch=1 if row['variant']=='event_loop' else 20
        for name,metric in (('calibrate','calibration_ms_per_event'),('integrate','integration_ms_per_event')):
            isolated=base[str(batch)][name]/batch
            ratio=row[metric]/isolated
            assert .65<=ratio<=1.65, ('pipeline and isolated kernel timers disagree',row,name,ratio)
            evidence.append(dict(point=row['point'],variant=row['variant'],kernel=name,
                                 pipeline_ms_per_event=row[metric],isolated_ms_per_event=isolated,ratio=ratio))
    assert len(evidence)==8
    main=next(x for x in review['results'] if x['point']==[1,1,20,2] and x['cache']=='warm')
    for variant,batch in (('event_loop',1),('batched_task',20)):
        kernel_s=sum(base[str(batch)].values())/batch*10
        loop_s=main['variants'][variant]['median_loop_s']
        # The two kernels use one stream in this single-BD comparison. Allow
        # substantial clock/cache differences, but reject impossible loop times
        # or an unexplained multi-minute slowdown before launching more work.
        assert loop_s>=.6*kernel_s,(variant,'loop shorter than measured kernel work',loop_s,kernel_s)
        assert loop_s<=max(300.,5*kernel_s),(variant,'large unexplained slowdown',loop_s,kernel_s)
    return dict(passed=True,calibration_minimum_ms_per_event=minimum_calibration_ms,
                kernel_timer_comparisons=evidence,
                interpretation='Consistent with the independent hot-buffer kernel check; no direction-of-speedup acceptance threshold.',
                limits='The isolated check repeats the first real event with hot buffers. It does not isolate all causes of the batching benefit or replace full-loop measurements.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--kernel-log',type=Path,required=True)
    a=p.parse_args()
    lines=[s.removeprefix('KERNEL_CHECK_RESULT ') for s in a.kernel_log.read_text().splitlines()
           if s.startswith('KERNEL_CHECK_RESULT ')]
    assert len(lines)==1
    print(json.dumps(check(json.loads(a.review.read_text()),json.loads(lines[0])),indent=2))


if __name__=='__main__':main()
