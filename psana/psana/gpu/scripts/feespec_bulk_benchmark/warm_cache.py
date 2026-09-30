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


if __name__ == '__main__':
    print('CACHE_WARM_POLICY ' + subprocess.check_output(
        ['numactl', '--show'], text=True).strip(), flush=True)
    print('CACHE_WARM_MEMORY_BEFORE ' + json.dumps(snapshot()), flush=True)
    if sys.argv[1:2] == ['--prefixes']:
        for path, length in json.loads(sys.argv[2]):
            warm_prefix(path, length)
    else:
        for path in sys.argv[1:]:
            with open(path, 'rb', buffering=0) as stream:
                while stream.read(16 * 1024**2):
                    pass
    print('CACHE_WARM_MEMORY_AFTER ' + json.dumps(snapshot()), flush=True)
