"""Sequentially warm private staged files under numactl --interleave=all."""
import json
import subprocess
import sys

from memory_state import snapshot


def warm_prefix(path, length):
    with open(path, 'rb', buffering=0) as stream:
        remaining = length
        while remaining:
            chunk = stream.read(min(16 * 1024**2, remaining))
            if not chunk:
                raise RuntimeError(f'short warm-prefix read: {path}')
            remaining -= len(chunk)


def warm_missing_prefix(path, length):
    from common import prefix_residency
    state = prefix_residency(path, length, missing_ranges=True)
    read_bytes = 0
    with open(path, 'rb', buffering=0) as stream:
        for offset, size in state['missing_ranges']:
            stream.seek(offset)
            remaining = size
            while remaining:
                chunk = stream.read(min(16 * 1024**2, remaining))
                if not chunk:
                    raise RuntimeError(f'short warm-prefix repair: {path}')
                remaining -= len(chunk)
                read_bytes += len(chunk)
    print('CACHE_WARM_REPAIRED ' + json.dumps(dict(path=str(path),
        prefix_bytes=length, read_bytes=read_bytes,
        missing_spans=len(state['missing_ranges']))), flush=True)
    return read_bytes


if __name__ == '__main__':
    print('CACHE_WARM_POLICY ' + subprocess.check_output(
        ['numactl', '--show'], text=True).strip(), flush=True)
    print('CACHE_WARM_MEMORY_BEFORE ' + json.dumps(snapshot()), flush=True)
    if sys.argv[1:2] == ['--missing-prefixes']:
        for path, length in json.loads(sys.argv[2]):
            warm_missing_prefix(path, length)
    elif sys.argv[1:2] == ['--prefixes']:
        for path, length in json.loads(sys.argv[2]):
            warm_prefix(path, length)
    else:
        for path in sys.argv[1:]:
            with open(path, 'rb', buffering=0) as stream:
                while stream.read(16 * 1024**2):
                    pass
    print('CACHE_WARM_MEMORY_AFTER ' + json.dumps(snapshot()), flush=True)
