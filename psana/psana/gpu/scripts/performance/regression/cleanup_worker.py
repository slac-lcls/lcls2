"""Disposable sleeping ranks for cleanup qualification; never opens experiment data."""
import os
from pathlib import Path
import sys
import time

from native_contract import read, require, write


def main():
    request_path, folder, stage = sys.argv[1:]
    request, folder = read(request_path), Path(folder)
    rank = int(os.environ['SLURM_PROCID'])
    job, step = os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID']
    require(job == request['site']['job_id'], 'wrong allocation')
    require(os.uname().nodename == request['site']['node'], 'wrong node')
    ready = folder / ('ready-%d.json' % rank)
    write(ready.with_suffix('.tmp'), dict(job=job, step=step, rank=rank,
          node=os.uname().nodename, pid=os.getpid(), monotonic=time.monotonic(),
          affinity=sorted(os.sched_getaffinity(0)), stage=stage))
    ready.with_suffix('.tmp').replace(ready)
    # Independent upper bound even if the coordinator disappears.
    deadline = time.monotonic() + 240
    tick = 0
    while time.monotonic() < deadline:
        if stage == 'sentinel':
            # Atomic replacement keeps readers from observing partial JSON.
            temporary = folder / 'heartbeat.tmp'
            temporary.write_text(str(tick))
            temporary.replace(folder / 'heartbeat')
            tick += 1
            if (folder / 'release').exists():
                write(folder / 'released.json', dict(job=job, step=step, tick=tick))
                return
        time.sleep(0.2)
    raise TimeoutError('disposable rank safety deadline expired')


if __name__ == '__main__':
    main()
