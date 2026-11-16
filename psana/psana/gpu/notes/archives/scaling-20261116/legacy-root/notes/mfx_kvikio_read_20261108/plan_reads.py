"""CPU-only, bounded read plans for the accepted sample10 fixture."""
import hashlib
import json
from pathlib import Path
import sys

MIB = 1 << 20
GIB = 1 << 30
NOTES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTES/'mfx_expand10_performance_20261106'))
from fixture_check import check_10k, ROOT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*MIB), b''):
            h.update(block)
    return h.hexdigest()


def interleave(per_file):
    return [row for i in range(max(map(len,per_file))) for rows in per_file for row in rows[i:i+1]]


def slab_plan(sizes, chunk):
    require(chunk > 0 and all(n > 0 for n in sizes), 'Positive sizes required')
    return interleave([[(i,offset,min(chunk,n-offset)) for offset in range(0,n,chunk)]
                       for i,n in enumerate(sizes)])


def load_fixture():
    manifest, identity = check_10k(ROOT, sha)
    streams = [s for s in manifest['streams'] if s['panels']]
    require([s['stream'] for s in streams] == [5,6,7,8,9], 'Unexpected JF streams')
    paths = [Path(s['files']['bd']['path']) for s in streams]
    sizes = [s['files']['bd']['bytes'] for s in streams]
    per_file = []
    for i,s in enumerate(streams):
        require(len(s['records']) == len(manifest['schedule']), 'Incomplete records')
        rows = []
        previous_end = 0
        for record, schedule in zip(s['records'],manifest['schedule']):
            offset,size = record['offsets']['bd'],record['sizes']['bd']
            require(record['source_index'] == schedule['source_index'], 'Source mismatch')
            require(offset == previous_end and size >= 24, 'Gap/invalid datagram')
            previous_end = offset+size
            if schedule['service'] == 12:
                rows.append((i,offset,size))
        require(previous_end == sizes[i] and len(rows) == 10000, 'File coverage mismatch')
        per_file.append(rows)
    events = interleave(per_file)
    require(len(events)==50000 and sum(r[2] for r in events)==33582000000, 'Event byte count differs from pipeline')
    identity.update(files=[dict(path=str(p),bytes=n,stream=s['stream'],sha256=s['files']['bd']['sha256'])
                           for p,n,s in zip(paths,sizes,streams)],
                    event_bytes=sum(r[2] for r in events), whole_file_bytes=sum(sizes))
    return paths,sizes,events,identity


def run_queue(requests, depth, submit, complete):
    """FIFO bounded ring. A buffer is reused only after its prior future completes."""
    from collections import deque
    require(depth > 0, 'Positive queue depth required')
    pending = deque()
    total = count = 0
    last = {}
    try:
        for index,request in enumerate(requests):
            slot = index % depth
            if len(pending) == depth:
                old_slot,old_request,future = pending.popleft()
                require(old_slot == slot, 'Ring ownership mismatch')
                got = complete(future)
                require(got == old_request[2], 'Short read')
                total += got
                count += 1
                last[old_slot] = old_request
            pending.append((slot,request,submit(slot,request)))
        while pending:
            slot,request,future = pending.popleft()
            got = complete(future)
            require(got == request[2], 'Short read')
            total += got
            count += 1
            last[slot] = request
    except BaseException:
        for _,_,future in pending:
            try: complete(future)
            except BaseException: pass
        raise
    return total,count,last
