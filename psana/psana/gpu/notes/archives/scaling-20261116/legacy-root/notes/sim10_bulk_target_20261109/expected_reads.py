"""Independent expected request counts from accepted fixture offsets, batch100.

No production planner imports. Gaps (including intervening transitions) break
adjacency. Each batch is limited to 100 events from each of the five files.
"""
from collections import Counter

def expected_reads(plan):
    output={}
    for events in (200,10000):
        rows=plan[:events*5]
        assert len(rows)==events*5
        targets={}
        for target in (524288,4194304,8388608):
            histogram=Counter()
            for start in range(0,events,100):
                batch=rows[start*5:min(start+100,events)*5]
                for file_id in range(5):
                    group_size=0;end=None
                    for f,offset,size in batch:
                        if f!=file_id:continue
                        if group_size and (end!=offset or group_size+size>target):
                            histogram[str(group_size)]+=1;group_size=0
                        group_size+=size;end=offset+size
                        if group_size>=target:
                            histogram[str(group_size)]+=1;group_size=0
                    if group_size:histogram[str(group_size)]+=1
            assert sum(int(size)*count for size,count in histogram.items())==3358200*events
            targets[str(target)]={'count':sum(histogram.values()),'histogram':dict(histogram)}
        output[str(events)]=targets
    return output
