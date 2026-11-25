"""Host-memory evidence; modeled buffer demand is not measured pinned usage."""
from pathlib import Path


def demand(bds, workers, bounce_mib):
    per_bd = workers * (bounce_mib << 20)
    return dict(per_bd_estimate_bytes=per_bd, aggregate_estimate_bytes=bds * per_bd,
                basis='One reusable buffer per concurrently active POSIX worker; excludes other host allocations',
                limitation='Source-based estimate, not observed pinned allocation count or a total process-memory bound')


def parse_kib(text, keys):
    values = {}
    for line in text.splitlines():
        key, _, value = line.partition(':')
        if key in keys:
            parts = value.split()
            if len(parts) != 2 or parts[1] != 'kB':
                raise ValueError('Expected procfs kB field: ' + key)
            values[key] = int(parts[0]) * 1024
    if set(values) != set(keys) or any(v < 0 for v in values.values()):
        raise ValueError('Missing or invalid procfs memory fields')
    return values


def snapshot():
    proc = parse_kib(Path('/proc/self/status').read_text(), ('VmRSS', 'VmHWM', 'VmLck', 'VmPin'))
    proc['mem_available_bytes'] = parse_kib(Path('/proc/meminfo').read_text(), ('MemAvailable',))['MemAvailable']
    return proc


def validate_host_memory(record, bds, workers, bounce_mib):
    expected = demand(bds, workers, bounce_mib)
    if record['demand'] != expected:
        raise ValueError('Host buffer demand mismatch')
    for name in ('startup', 'before_loop', 'after_loop'):
        sample = record[name]
        for key in ('VmRSS', 'VmHWM', 'VmLck', 'VmPin', 'mem_available_bytes'):
            if type(sample[key]) is not int or sample[key] < 0:
                raise ValueError('Invalid host memory sample')
        if sample['VmRSS'] <= 0 or sample['VmHWM'] < sample['VmRSS']:
            raise ValueError('Invalid process RSS/high-water sample')
    if record['before_loop']['mem_available_bytes'] < expected['aggregate_estimate_bytes'] + (8 << 30):
        raise ValueError('Insufficient host headroom for bounce buffers')
    # VmHWM is the process lifetime peak, not the event-loop-only peak.
    if record['after_loop']['VmHWM'] < record['before_loop']['VmHWM']:
        raise ValueError('Host high-water mark decreased')
