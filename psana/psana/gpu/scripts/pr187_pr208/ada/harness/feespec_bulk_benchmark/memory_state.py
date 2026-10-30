"""Read-only host and effective cgroup memory evidence, outside timed work."""
import os
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return {'unavailable': str(exc)}


def snapshot():
    paths = ['/proc/meminfo', '/proc/loadavg', '/proc/net/dev',
             '/proc/self/cgroup', '/proc/self/status', '/proc/self/numa_maps',
             '/proc/pressure/memory', '/proc/sys/vm/zone_reclaim_mode']
    result = {p: read(p) for p in paths}
    result['pid'] = os.getpid()
    for p in Path('/sys/devices/system/node').glob('node*/meminfo'):
        result[str(p)] = read(p)
    groups = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        groups.append((set(controllers.split(',')), relative))
    names = ['memory.current', 'memory.max', 'memory.high', 'memory.events',
             'memory.stat', 'memory.numa_stat', 'memory.swap.current',
             'memory.usage_in_bytes', 'memory.limit_in_bytes',
             'memory.soft_limit_in_bytes', 'memory.failcnt',
             'memory.use_hierarchy', 'cpuset.mems', 'cpuset.mems.effective',
             'cpuset.effective_mems']
    # Include ancestor limits: a leaf's unlimited value can hide a job cap.
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fs, _, options = right.split()[:3]
        if fs not in ('cgroup', 'cgroup2'):
            continue
        fields = left.split()
        root, mount = Path(fields[3]), Path(fields[4])
        for controllers, relative in groups:
            if fs == 'cgroup2':
                matches = controllers == {''}
            else:
                matches = bool(controllers & set(options.split(',')) & {'memory', 'cpuset'})
            if not matches:
                continue
            try:
                current = mount / Path(relative).relative_to(root)
            except ValueError:
                continue
            while True:
                for name in names:
                    p = current / name
                    try:
                        exists = p.exists()
                    except OSError as exc:
                        result[str(p)] = {'unavailable': str(exc)}
                        continue
                    if exists:
                        result[str(p)] = read(p)
                if current == mount:
                    break
                current = current.parent
    return result
