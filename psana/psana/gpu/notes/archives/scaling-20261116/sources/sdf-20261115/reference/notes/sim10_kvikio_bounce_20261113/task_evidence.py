"""Audit observed POSIX read chunking, separate from inferred native task count."""
import collections
import hashlib
import re
from pathlib import Path

HEADER = re.compile(r'^\s*(\d+)\s+(\d+\.\d+)\s+(.*)$')
READ = re.compile(r'^pread64\(.*?,\s*(\d+),\s*(\d+)\)\s+=\s+(-?\d+)(?:\s|$)')


def parse_trace(path, start, end):
    if not 0 < start < end:
        raise ValueError('Invalid syscall observation window')
    pending = {}
    counts = collections.Counter()
    for line in Path(path).read_text().splitlines():
        match = HEADER.match(line)
        if not match:
            if 'pread64' in line:
                raise ValueError('Unparsed syscall trace line: ' + line)
            continue
        tid, timestamp, body = match.groups()
        timestamp = float(timestamp)
        if body.startswith('pread64(') and body.endswith('<unfinished ...>'):
            if tid in pending:
                raise ValueError('Duplicate unfinished syscall')
            pending[tid] = (timestamp, body[:-len('<unfinished ...>')])
            continue
        if body.startswith('<... pread64 resumed>'):
            if tid not in pending:
                raise ValueError('Unmatched resumed syscall')
            timestamp, prefix = pending.pop(tid)
            body = prefix + body[len('<... pread64 resumed>'):]
        if not body.startswith('pread64('):
            continue
        match = READ.match(body)
        if not match:
            raise ValueError('Unparsed pread64: ' + body)
        size, offset, done = map(int, match.groups())
        if start <= timestamp <= end:
            if size <= 0 or done != size:
                raise ValueError('Short/failed POSIX read during pilot')
            counts[str(size)] += 1
    if pending:
        raise ValueError('Incomplete syscall trace')
    return dict(calls=sum(counts.values()), bytes=sum(int(k)*v for k,v in counts.items()),
                size_histogram=dict(sorted(counts.items())))


def pilot_evidence(root, result, events, logical_histogram):
    rows = []
    for r in result['rows'][2:]:
        window = r['trace_window']
        if not window['tracer_pid'] > 0:
            raise ValueError('Pilot BD was not traced')
        path = Path(str(root) + '.rank%d.strace' % r['rank'])
        observed = parse_trace(path, window['start'], window['end'])
        n = len(r['timestamps'])
        if observed['bytes'] != 3358200*n or observed['calls'] < 5*n:
            raise ValueError('Pilot POSIX payload coverage mismatch')
        rows.append(dict(rank=r['rank'], path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **observed))
    combined = collections.Counter()
    for row in rows:
        combined.update(row['size_histogram'])
    if sum(row['bytes'] for row in rows) != events*3358200:
        raise ValueError('Global pilot POSIX byte mismatch')
    return dict(rows=rows, calls=sum(row['calls'] for row in rows),
        bytes=sum(row['bytes'] for row in rows), size_histogram=dict(sorted(combined.items())),
        logical_read_histogram=logical_histogram,
        interpretation='observed_posix_unsplit' if dict(combined)==logical_histogram else 'observed_posix_chunking',
        limitation='POSIX syscalls are observed; native thread-pool task counts are not directly measured. Pilots are traced; full timing runs are not.')
