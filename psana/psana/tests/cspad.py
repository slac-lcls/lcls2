#------------------------------
# nosetests -sv psana/psana/tests
#
#------------------------------

doPlot = 0

#------------------------------

import psana
if doPlot: import matplotlib.pyplot as plt

# FIXME MONA: remove this when we know where to get calibration files from
import os
os.environ['PS_CALIB_DIR'] = "/reg/common/package/mona/cxid9114"

ds = psana.DataSource('/reg/d/psdm/cxi/cxitut13/scratch/yoon82/crystal_101_evts.xtc')
det = ds.Detector("DscCsPad")
run = 101
for i, evt in enumerate(ds.events()):
    if i == 3: break
    raw = det.raw(evt)
    ped = det.pedestals(run)
    photonEnergy = det.photonEnergy(evt)
    print("Event: ", i, raw.shape, ped.shape if ped is not None else None)
    print(photonEnergy)

if doPlot:
    for i in range(raw.shape[0]):
        plt.imshow(raw[i])
        plt.title(i)
        plt.colorbar()
        plt.show()
