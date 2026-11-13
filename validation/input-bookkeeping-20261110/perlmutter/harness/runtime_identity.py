"""Explicit identities for the frozen baseline and three-file Python overlay."""
import json
import os
from pathlib import Path
from support import require, sha

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'sim10_stream_length_20261109'
FROZEN = Path('/pscratch/sd/m/monarin/psana2-gpu/pr187-pr208-perlmutter-20261026/attempt01/install-B/lib/python3.11/site-packages')
CHANGED = ('gpu_events', 'gpu_input_group', 'gpu_input_window')


def validate(runtime):
    expected = json.loads((BASE/'perlmutter-runtime.json').read_text())['modules']
    for name in ('gpu_input_group', 'gpu_input_window'):
        expected['psana.gpu.'+name] = dict(path=str(FROZEN/'psana/gpu'/(name+'.py')),
            sha256=sha(HERE.parent/'sim10_trace_20261110/frozen'/(name+'.py')))
    if os.environ['SIM10_VARIANT'] == 'new':
        for name in CHANGED:
            expected['psana.gpu.'+name] = dict(path=str(Path(os.environ['SIM10_OVERLAY'])/'psana/gpu'/(name+'.py')),
                sha256=sha(HERE.parents[1]/'psana/psana/gpu'/(name+'.py')))
    normalized = {n:dict(path=str(Path(r['path']).resolve()),sha256=r['sha256']) for n,r in runtime.items()}
    require(normalized == expected, 'Runtime identity differs from explicit old/new manifest')
