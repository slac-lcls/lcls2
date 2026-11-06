"""Use setup_env.sh; all output directories must be new. Serial, 600s per case."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/tmp/jf-exact-command')
DRIVER = Path(__file__).with_name('run_case.py').resolve()
CLI = ['-k', 'exp=mfx100848724,run=49', '-d', 'jungfrau', '-o', './work1',
       '--nrecs', '50', '--nrecs1', '50', '--stepnum', '2']
checkouts = dict(unfixed=Path('/tmp/jf-callback-ab/unfixed'), fixed=ROOT / 'fixed')
expected = dict(unfixed='71378658a54dcdd941c9137355a985fa4b637595',
                fixed='b59c2fd1aad012de195b14b62401bc966e6755c3')
template = Path('/tmp/smd-callback-chunk-state/python')
prefix = '/sdf/home/m/monarin/lcls2/psana/psana/'
for version, checkout in checkouts.items():
    sha = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    assert sha == expected[version]
    assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain',
                                       '--untracked-files=no'], text=True).strip()
    overlay = ROOT / (version + '-python')
    overlay.mkdir()  # Refuse to reuse a potentially stale overlay.
    for directory, dirs, files in os.walk(str(template)):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        target = overlay / Path(directory).relative_to(template)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            resolved = str((Path(directory) / name).resolve())
            if resolved.startswith(prefix):
                resolved = str(checkout / 'psana/psana' / resolved[len(prefix):])
            if Path(resolved).exists():
                (target / name).symlink_to(resolved)

commands = []
for case, version, setting in [('unfixed-default', 'unfixed', None),
                                ('unfixed-10000', 'unfixed', '10000'),
                                ('fixed-default', 'fixed', None)]:
    path = ROOT / case
    path.mkdir()
    overlay = ROOT / (version + '-python')
    env = dict(os.environ, PYTHONPATH=str(overlay) + ':' + os.environ['PYTHONPATH'],
               PS_PARALLEL='none', SIT_PSDM_DATA='/sdf/data/lcls/ds/')
    env.pop('PS_SMD_N_EVENTS', None)
    if setting is not None:
        env['PS_SMD_N_EVENTS'] = setting
    command = [sys.executable, str(DRIVER), *CLI]
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
    item = dict(case=case, source_sha=expected[version], overlay=str(overlay),
                initial_smd_n_events=setting, command=command, cwd=str(path),
                exit_code=code, timed_out=timed_out,
                seconds=round(time.monotonic() - started, 2))
    commands.append(item)
    (ROOT / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    print(json.dumps(item), flush=True)
