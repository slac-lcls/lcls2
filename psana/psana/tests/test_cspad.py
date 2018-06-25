#------------------------------
# nosetests -sv psana/psana/tests
#
#------------------------------

doPlot = 1

#------------------------------

import psana
if doPlot: import matplotlib.pyplot as plt

ds = psana.DataSource('/reg/common/package/temp/crystal_dark_evts.xtc')
det = ds.Detector("DscCsPad")
for i, evt in enumerate(ds.events()):
    raw = det.raw(evt)
    print(i, raw.shape)

if doPlot:
    for i in range(raw.shape[0]):
        plt.imshow(raw[i])
        plt.title(i)
        plt.colorbar()
        plt.show()
