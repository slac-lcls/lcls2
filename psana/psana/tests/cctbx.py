# NOTE:
# This example will be merged with user.py
# when detector interface is ready for all detectors

import os
from psana import DataSource

def filter(evt):
        return True

xtc_dir = "/reg/d/psdm/xpp/xpptut15/scratch/mona/cxid9114"
ds = DataSource('exp=xpptut13:run=1:dir=%s'%(xtc_dir), filter=filter)
det = None
if ds.nodetype == "bd":
    det = ds.Detector("DsdCsPad")

for run in ds.runs():
    for evt in run.events():
        if det:
            raw = det.raw(evt)
            print(raw.shape)
