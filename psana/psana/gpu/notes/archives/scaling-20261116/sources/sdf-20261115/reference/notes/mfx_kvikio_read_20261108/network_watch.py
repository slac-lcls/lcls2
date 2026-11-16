"""Physical NIC endpoints and file-specific Linux cache conditioning.

Never combine netdev, RDMA, and physical counters. All NICs remain separate.
Only the five owner-controlled, accepted fixture files may be evicted.
"""
import ctypes
import mmap
import os
from pathlib import Path
import subprocess
import time

from plan_reads import ROOT, require


def physical_counters():
    records = {}
    for interface in sorted(Path('/sys/class/net').iterdir()):
        if not (interface/'device').exists():
            continue
        result = subprocess.run(['ethtool', '-S', interface.name], capture_output=True, text=True, timeout=5)
        values = {}
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                key, sep, value = line.strip().partition(':')
                if sep and key in ('rx_bytes_phy', 'tx_bytes_phy'):
                    values[key] = int(value)
        if len(values) == 2:
            records[interface.name] = dict(**values, monotonic=time.perf_counter(),
                device=str((interface/'device').resolve()))
    require(bool(records), 'No physical NIC counters available; refusing unobserved measurement')
    return records


def counter_delta(before, after, payload=0):
    require(set(before) == set(after), 'NIC inventory changed')
    result = {}
    for name, first in before.items():
        last = after[name]
        elapsed = last['monotonic']-first['monotonic']
        require(elapsed > 0 and last['device'] == first['device'], 'Invalid NIC endpoint')
        rx = last['rx_bytes_phy']-first['rx_bytes_phy']
        tx = last['tx_bytes_phy']-first['tx_bytes_phy']
        require(rx >= 0 and tx >= 0, 'Physical NIC counter reset')
        result[name] = dict(seconds=elapsed, rx_bytes=rx, tx_bytes=tx,
                           rx_gib_s=rx/(1<<30)/elapsed,
                           rx_to_requested_bytes=rx/payload if payload else None)
    return result


def residency(paths, evict=False):
    import numpy as np
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    records = []
    for path in paths:
        path = Path(path)
        require(path.resolve().parent == ROOT.resolve(), 'Foreign cache target')
        require(any(path.name.endswith('-s%03d-c000.xtc2'%s) for s in range(5,10)), 'Non-JF cache target')
        fd = os.open(path, os.O_RDONLY)
        try:
            st = os.fstat(fd)
            # Linux may hide mincore residency from callers without ownership.
            require(st.st_uid == os.geteuid(), 'Cache target must be owned by benchmark user')
            pages = (st.st_size+mmap.PAGESIZE-1)//mmap.PAGESIZE
            def count():
                with mmap.mmap(fd, st.st_size, access=mmap.ACCESS_COPY) as mapping:
                    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                    vector = np.empty(pages, dtype=np.uint8)
                    if libc.mincore(address, st.st_size, vector.ctypes.data):
                        raise OSError(ctypes.get_errno(), 'mincore failed')
                    return int(np.count_nonzero(vector & 1))
            before = count()
            after = before
            if evict:
                os.fsync(fd)
                for attempt in range(20):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    after = count()
                    if after == 0:
                        break
                    time.sleep(.1)
                require(after == 0, 'Cache eviction failed: '+str(path))
            records.append(dict(path=str(path), bytes=st.st_size, pages=pages,
                                resident_before=before, resident_pages=after))
        finally:
            os.close(fd)
    return dict(files=records, pages=sum(r['pages'] for r in records),
                resident_pages=sum(r['resident_pages'] for r in records))


class NetworkProfile:
    def __init__(self, paths):
        self.paths = paths
        self.initial_cache = residency(paths)
        first = physical_counters()
        time.sleep(1)
        last = physical_counters()
        self.idle = dict(before=first, after=last, delta=counter_delta(first,last))

    def begin(self, iteration):
        conditioning = residency(self.paths, evict=True) if iteration == 0 else None
        cache = residency(self.paths)
        if iteration == 0:
            require(cache['resident_pages'] == 0, 'Cache refilled before cold pass')
        return dict(kind='node-cache-cold' if iteration == 0 else 'warm-repeat',
                    conditioning=conditioning, cache_before=cache, nic_before=physical_counters())

    def finish(self, record, payload):
        record['nic_after'] = physical_counters()
        record['nic_delta'] = counter_delta(record['nic_before'],record['nic_after'],payload)
        record['cache_after'] = residency(self.paths)
        return record
