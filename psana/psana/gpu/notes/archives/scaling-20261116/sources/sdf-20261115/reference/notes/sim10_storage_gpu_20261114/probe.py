"""Bounded CPU pread slab probe: representative sizes, not an XTC event replay."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import os
from pathlib import Path
import sys
import time

from plan import CONTROLS
from support import require, write
from storage import cache, counters, evidence

SIZES = (524724, 629664, 734604, 734604, 734604)
EVENT_EQUIVALENTS = 16000
DEPTH = 16


def requests(file_sizes, count=EVENT_EQUIVALENTS):
    require(len(file_sizes) == 5 and count > 0, 'Probe needs five files')
    require(all(length >= count * size for length, size in zip(file_sizes, SIZES)), 'Probe extent exceeds file')
    return ((index, step * size, size) for step in range(count) for index, size in enumerate(SIZES))


def main(info, output):
    assignment = json.loads(os.environ['SIM10_CPU_PLAN'])
    affinity = [cpu for rank in assignment['ranks'][2:] for cpu in rank][:DEPTH]
    require(len(affinity) == DEPTH and set(affinity) <= os.sched_getaffinity(0), 'Probe physical CPU allocation')
    os.sched_setaffinity(0, affinity)
    files = [Path(path) for path in info['files']]
    read_plan = requests([path.stat().st_size for path in files])
    descriptors = [os.open(path, os.O_RDONLY | os.O_NOFOLLOW) for path in files]
    try:
        idle_before = counters(info['observed_leaves'])
        time.sleep(2)
        idle_after = counters(info['observed_leaves'])
        conditioning = cache(files, info['directory'], info['media'], True)
        initial = cache(files, info['directory'], info['media'])
        require(all(row['resident_pages'] == 0 for row in initial), 'Probe cold gate')

        def read(request):
            index, offset, length = request
            actual = len(os.pread(descriptors[index], length, offset))
            require(actual == length, 'Short probe read')
            return actual

        completed = 0
        total = 0
        with ThreadPoolExecutor(max_workers=DEPTH) as pool:
            record = dict(before=counters(info['observed_leaves']), idle_before=idle_before, idle_after=idle_after)
            start = time.perf_counter()
            pending = deque()
            for request in read_plan:
                if len(pending) == DEPTH:
                    total += pending.popleft().result()
                    completed += 1
                pending.append(pool.submit(read, request))
            while pending:
                total += pending.popleft().result()
                completed += 1
            seconds = time.perf_counter() - start
            record['after'] = counters(info['observed_leaves'])
        require(completed == 5 * EVENT_EQUIVALENTS and total == sum(SIZES) * EVENT_EQUIVALENTS, 'Probe totals')
        write(output, dict(status='pass', media=info['media'], seconds=seconds, bytes=total,
              requests=completed, depth=DEPTH, affinity=affinity, gib_s=total / 2**30 / seconds,
              conditioning=conditioning, initial_cache=initial, counters=record,
              evidence=evidence(record, info['media'], total, seconds),
              sizes=SIZES, event_equivalents=EVENT_EQUIVALENTS,
              limitation='CPU pread of disjoint contiguous slabs, not descriptor-aligned events, KvikIO, H2D, or a storage ceiling; no throughput acceptance threshold'))
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
