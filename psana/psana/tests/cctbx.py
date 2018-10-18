# NOTE:
# This example only works with psana2-python2.7 environment

import os
from psana import DataSource

def filter(evt):
    return True

#os.environ['PS_CALIB_DIR'] = "/reg/d/psdm/cxi/cxid9114/scratch/mona/l2/psana-nersc/demo18/cxid9114/input"
os.environ['PS_CALIB_DIR'] = "/reg/d/psdm/cxi/cxid9114/scratch/mona/l2/psana-nersc/demo18/cxic0415/input"
os.environ['PS_SMD_NODES'] = '1'
os.environ['PS_SMD_N_EVENTS'] = '100'
xtc_dir = "/reg/d/psdm/xpp/xpptut15/scratch/mona/cxic0415"
ds = DataSource('exp=cxic0415:run=24:dir=%s'%(xtc_dir), filter=filter, max_events=10, det_name="DscCsPad")

for run in ds.runs():
    det = run.Detector(ds.det_name)
    for evt in run.events():
        raw = det.raw(evt)
        ped = det.pedestals(run)
        gain_mask = det.gain_mask(run, gain=6.85)
        calib = det.calib(evt)
        raw_data = det.raw_data(evt)
        print(raw.shape, ped.shape, gain_mask.shape, calib.shape, raw_data.shape)
