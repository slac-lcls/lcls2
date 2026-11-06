"""Current-master versus clean-PR real-data comparison; source setup_env first."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/tmp/smd-callback-pr')
INSTALL = ROOT / 'worktree/install/lib/python3.9/site-packages'
MASTER = ROOT / 'master'
OVERLAY = ROOT / 'master-python'
CASES = ROOT / 'jf'
DRIVER = Path(__file__).with_name('run_case.py').resolve()
assert subprocess.check_output(['git', '-C', str(MASTER), 'rev-parse', 'HEAD'],
    text=True).strip() == '9ab5c193b4277e44e3e8ef9563d830c8943fede3'
OVERLAY.mkdir()
for directory, dirs, files in os.walk(INSTALL):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    relative = Path(directory).relative_to(INSTALL)
    target = OVERLAY / relative
    target.mkdir(exist_ok=True, parents=True)
    for name in files:
        source = Path(directory) / name
        if name.endswith('.py') and relative.parts and relative.parts[0] in ('psana', 'psalg'):
            source = MASTER / relative.parts[0] / relative / name
        if source.exists():
            (target / name).symlink_to(source)

# This oracle bypasses SMD callback/chunk selection entirely.
from psana import DataSource
ds = DataSource(files='/sdf/data/lcls/ds/mfx/mfx100848724/xtc/smalldata/'
                'mfx100848724-r0049-s000-c000.smd.xtc2', skip_calib_load='all', monitor=False)
steps = [[int(evt.timestamp) for evt in step.events()] for step in next(ds.runs()).steps()]
assert list(map(len, steps)) == [1000, 1000, 1000]
(ROOT / 'oracle.json').write_text(json.dumps(steps) + '\n')

CASES.mkdir()
commands = []
for case, setting, pythonpath in [('master-default', None, OVERLAY),
                                  ('master-10000', '10000', OVERLAY),
                                  ('fixed-default', None, INSTALL)]:
    path = CASES / case
    path.mkdir()
    env = dict(os.environ, PYTHONPATH=str(pythonpath), PS_PARALLEL='none',
               SIT_PSDM_DATA='/sdf/data/lcls/ds/')
    env.pop('PS_SMD_N_EVENTS', None)
    if setting:
        env['PS_SMD_N_EVENTS'] = setting
    command = [sys.executable, str(DRIVER), '-k', 'exp=mfx100848724,run=49',
               '-d', 'jungfrau', '-o', './work1', '--nrecs', '50', '--nrecs1', '50', '--stepnum', '2']
    started = time.monotonic()
    timed_out = False
    with (path / 'stdout.log').open('w') as log:
        process = subprocess.Popen(command, cwd=path, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    item = dict(case=case, setting=setting, pythonpath=str(pythonpath), command=command,
                cwd=str(path), exit_code=code, timed_out=timed_out,
                seconds=round(time.monotonic() - started, 2))
    commands.append(item)
    (CASES / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(json.dumps(item), flush=True)
