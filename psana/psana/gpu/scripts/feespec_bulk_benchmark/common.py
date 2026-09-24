"""Helpers for the isolated feespec FFB benchmark; no production policy changes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(text, prefix):
    return [json.loads(x[len(prefix):]) for x in text.splitlines() if x.startswith(prefix)]


def network():
    result = {}
    for interface in Path('/sys/class/net').iterdir():
        if interface.name == 'lo':
            continue
        command = subprocess.run(['ethtool', '-S', interface.name], capture_output=True, text=True)
        if command.returncode:
            continue
        for line in command.stdout.splitlines():
            key, sep, value = line.strip().partition(':')
            if sep and key in ('rx_bytes_phy', 'tx_bytes_phy'):
                result[interface.name + '/' + key] = int(value)
    if not any(k.endswith('/rx_bytes_phy') for k in result):
        raise RuntimeError('No physical Ethernet byte counters available')
    return result


def cache(path, mode, prepare=True):
    from page_cache_residency import file_residency
    path = Path(path)
    if prepare:
        with path.open('rb', buffering=0) as source:
            if mode == 'cold':
                os.fsync(source.fileno())
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            else:
                while source.read(4 * 1024**2):
                    pass
    state = file_residency(str(path))
    fraction = state['resident_fraction']
    if mode == 'warm' and fraction < .99:
        raise RuntimeError(f'Warm residency {fraction} below 99%')
    if prepare and mode == 'cold' and fraction > .01:
        raise RuntimeError(f'Cold residency {fraction} above 1%')
    return state


def prefix_residency(path, length):
    """Measure only the input prefix exercised by a short run, without reading it."""
    import ctypes
    import mmap
    import numpy as np
    with open(path, 'rb', buffering=0) as source:
        if not 0 < length <= os.fstat(source.fileno()).st_size:
            raise ValueError('invalid measured input extent')
        mapping = mmap.mmap(source.fileno(), length, flags=mmap.MAP_PRIVATE,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
    pages = (length + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vector = (ctypes.c_ubyte * pages)()
    anchor = ctypes.c_char.from_buffer(mapping)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.mincore(ctypes.c_void_p(ctypes.addressof(anchor)),
                         ctypes.c_size_t(length), ctypes.byref(vector))
        if rc:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(path))
        resident = int((np.ctypeslib.as_array(vector) & 1).sum(dtype=np.uint64))
    finally:
        del anchor
        mapping.close()
    return dict(path=str(path), bytes=length, pages=pages, resident_pages=resident,
                resident_fraction=resident/pages)


def cache_inputs(directory, mode, prepare=True, include_jf=False, ranges=None):
    if ranges is not None:
        if mode != 'cold':
            raise ValueError('bounded-prefix cache preparation is cold-only')
        rows = []
        for name, length in sorted(ranges.items()):
            path = Path(directory)/name
            assert Path(name).name == name
            if prepare:
                with path.open('rb', buffering=0) as source:
                    os.fsync(source.fileno())
                    # Evict the entire private file, verify only the tested prefix.
                    os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            row = prefix_residency(path, length)
            if prepare and row['resident_fraction'] > .01:
                raise RuntimeError(f'Cold prefix residency too high: {row}')
            rows.append(row)
        pages = sum(r['pages'] for r in rows)
        resident = sum(r['resident_pages'] for r in rows)
        return dict(files=rows, pages=pages, resident_pages=resident,
                    resident_fraction=resident/pages, measured_prefixes=True)
    if not include_jf:
        return cache(Path(directory)/'mfx101210926-r0387-s000-c000.xtc2', mode, prepare)
    paths = sorted(Path(directory).glob('*.xtc2'))
    assert len(paths) == 6
    # Spread the 336 GB cache over NUMA nodes; do not alter timed workers.
    if prepare and mode == 'warm':
        subprocess.run(['numactl', '--interleave=all', sys.executable,
                        str(Path(__file__).with_name('warm_cache.py')),
                        *map(str, paths)], check=True)
    rows = [cache(p, mode, prepare and mode == 'cold') for p in paths]
    pages = sum(r['pages'] for r in rows)
    resident = sum(r['resident_pages'] for r in rows)
    return dict(files=rows, pages=pages, resident_pages=resident,
                resident_fraction=resident/pages)


def digest(array):
    import numpy as np
    value = np.array(array, copy=True, order='C')
    if np.issubdtype(value.dtype, np.floating):
        value[value == 0] = 0
        value[np.isnan(value)] = np.nan
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                nonzero=int(np.count_nonzero(value)))


def tier(paths):
    output = subprocess.check_output(['weka', 'fs', 'tier', 'location',
        *map(str, paths), '--format', 'json', '--raw-units'], text=True)
    data = json.loads(output)
    # Persist original output for audit, and require purely SSD-resident private files.
    rows = data if isinstance(data, list) else data.get('data', [])
    if not rows:
        raise RuntimeError(f'Unrecognized Weka tier response: {output}')
    for row in rows:
        if int(row['object_storage_bytes']) or int(row['remote_storage_bytes']):
            raise RuntimeError(f'Non-FFB backing: {row}')
        if int(row['file_size']) and int(row['ssd_write_cache_bytes']) + int(row['ssd_read_cache_bytes']) < int(row['file_size']):
            raise RuntimeError(f'Incomplete SSD coverage: {row}')
    return data
