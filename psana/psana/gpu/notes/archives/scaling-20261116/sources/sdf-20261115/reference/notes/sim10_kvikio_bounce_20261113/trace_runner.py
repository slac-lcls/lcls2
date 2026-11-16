"""Trace only pilot BD payload pread64 calls; full runs never use this wrapper."""
import json
import os
from pathlib import Path
import shutil
import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    events = int(args[args.index('--events') + 1])
    if events == 100000:
        raise RuntimeError('Tracing full timing runs is forbidden')
    command = [sys.executable, *args]
    if rank >= 2:
        from cpu_plan import bind_rank
        bind_rank(json.loads(os.environ['SIM10_CPU_PLAN']), rank)
        executable = shutil.which('strace')
        if executable is None:
            raise RuntimeError('Pilot syscall evidence requires strace')
        info = json.loads(Path(args[args.index('--inputs') + 1]).read_text())
        output = args[args.index('--output') + 1] + '.rank%d.strace' % rank
        command = [executable, '-f', '-ttt', '-s', '0', '-e', 'trace=pread64', '-o', output]
        for path in info['files']:
            command += ['-P', path]
        command += ['--', sys.executable, *args]
    os.execv(command[0], command)
