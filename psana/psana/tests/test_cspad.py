#------------------------------
# nosetests -sv psana/psana/tests
#
#------------------------------

doPlot = 0

#------------------------------

import psana
if doPlot: import matplotlib.pyplot as plt

ds = psana.DataSource('/reg/neh/home/yoon82/Software/lcls2/psana/psana/dgramPort/crystal_xray_evts.xtc')
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
