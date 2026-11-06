"""Prepare pre-fix Python overlay and direct-file timestamp oracle.

Prerequisite: detached pre-fix checkout at /tmp/jf-callback-ab/unfixed and the
source/native overlay from the callback regression tests. Run in setup_env.sh
environment with that overlay first on PYTHONPATH and PS_PARALLEL=none.
"""

import json
import os
from pathlib import Path

from psana import DataSource

root = Path('/tmp/jf-callback-ab')
src = Path('/tmp/smd-callback-chunk-state/python')
dst = root / 'unfixed-python'
for directory, dirs, files in os.walk(str(src)):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    target = dst / Path(directory).relative_to(src)
    target.mkdir(parents=True, exist_ok=True)
    for name in files:
        resolved = str((Path(directory) / name).resolve())
        prefix = '/sdf/home/m/monarin/lcls2/psana/psana/'
        if resolved.startswith(prefix):
            resolved = resolved.replace(prefix, str(root / 'unfixed/psana/psana') + '/', 1)
        link = target / name
        if Path(resolved).exists():
            if link.is_symlink():
                assert str(link.resolve()) == resolved
            else:
                link.symlink_to(resolved)

xtc = Path('/sdf/data/lcls/ds/mfx/mfx100848724/xtc')
smd = xtc / 'smalldata/mfx100848724-r0049-s000-c000.smd.xtc2'
ds = DataSource(files=str(smd), skip_calib_load='all', monitor=False)
run = next(ds.runs())
steps = [[int(evt.timestamp) for evt in step.events()] for step in run.steps()]
(root / 'oracle.json').write_text(json.dumps(steps))
print(json.dumps(dict(step_counts=list(map(len, steps)), smd_file=str(smd),
                     bigdata_bytes=sum(f.stat().st_size for f in xtc.glob('*-r0049-*.xtc2')))))
