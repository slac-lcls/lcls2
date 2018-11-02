import os
from psana import DataSource

def filter_fn(evt):
    return True

xtc_dir = os.path.join(os.getcwd(),'.tmp')
ds = DataSource('exp=xpptut13:run=1:dir=%s'%(xtc_dir), filter=filter_fn)
def event_fn(event, det):
    for d in event:
        assert d.xppcspad.raw.arrayRaw.shape == (18,)

for run in ds.runs():
    det = run.Detector('xppcspad')
    run.analyze(event_fn=event_fn, det=det)
