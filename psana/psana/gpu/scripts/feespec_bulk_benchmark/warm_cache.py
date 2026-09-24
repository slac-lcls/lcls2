"""Sequentially warm private staged files under numactl --interleave=all."""
import json
import subprocess
import sys

from memory_state import snapshot


if __name__ == '__main__':
    print('CACHE_WARM_POLICY ' + subprocess.check_output(
        ['numactl', '--show'], text=True).strip(), flush=True)
    print('CACHE_WARM_MEMORY_BEFORE ' + json.dumps(snapshot()), flush=True)
    for path in sys.argv[1:]:
        with open(path, 'rb', buffering=0) as stream:
            while stream.read(16 * 1024**2):
                pass
    print('CACHE_WARM_MEMORY_AFTER ' + json.dumps(snapshot()), flush=True)
