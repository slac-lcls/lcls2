"""Run bounded real-data pedestal A/B cases; invoke after sourcing setup_env.sh."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/tmp/jf-callback-ab/matrix')
ROOT.mkdir(parents=True, exist_ok=True)
DRIVER = Path(__file__).with_name('run_stage.py').resolve()
CASES = [('fixed-large', 'fixed', 10000), ('fixed-small', 'fixed', 100),
         ('unfixed-large', 'unfixed', 10000), ('unfixed-small', 'unfixed', 100),
         ('unfixed-step', 'unfixed', 1000)]
DS = repr(dict(exp='mfx100848724', run=49,
               dir='/sdf/data/lcls/ds/mfx/mfx100848724/xtc', monitor=False))
commands = []


def execute(case, version, chunk, stage, extra):
    path = ROOT / case / stage
    path.mkdir(parents=True, exist_ok=True)
    overlay = ('/tmp/smd-callback-chunk-state/python' if version == 'fixed'
               else '/tmp/jf-callback-ab/unfixed-python')
    env = dict(os.environ, PYTHONPATH=overlay + ':' + os.environ['PYTHONPATH'],
               PS_PARALLEL='mpi' if stage == 'stage2' else 'none',
               PS_SMD_N_EVENTS=str(chunk), PS_SRV_NODES='1', PS_EB_NODES='1')
    repo = ROOT / case / ('stage2-repo' if stage == 'stage2' else 'stage1-repo')
    command = [sys.executable, str(DRIVER), str(path), '--', '-k', DS,
               '-d', 'jungfrau', '-o', str(repo), '--segind', '0', *extra]
    if stage == 'stage2':
        command = ['mpirun', '--mca', 'osc', '^ucx', '-n', '5', *command]
    started = time.monotonic()
    with (path / 'stdout.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    item = dict(case=case, version=version, chunk=chunk, stage=stage, command=command,
                overlay=overlay, exit_code=code, timed_out=timed_out,
                seconds=round(time.monotonic()-started, 2))
    commands.append(item)
    (ROOT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(json.dumps({key: item[key] for key in
                      ('case', 'stage', 'exit_code', 'timed_out', 'seconds')}), flush=True)


for case, version, chunk in CASES:
    for step in range(3):
        execute(case, version, chunk, 'stage1-%d' % step,
                ['--nrecs', '10', '--nrecs1', '10', '--stepnum', str(step)])

gates = ROOT / 'fixed-large/stage1-repo/jungfrau/block_results'
assert len(list(gates.glob('*.npy'))) == 6, 'Need correct reference gates first'
for case, version, chunk in CASES:
    target = ROOT / case / 'stage2-repo/jungfrau/block_results'
    shutil.copytree(gates, target)
    execute(case, version, chunk, 'stage2',
            ['--nrecs', '12', '--nrecs1', '0', '--evskip', '2'])
