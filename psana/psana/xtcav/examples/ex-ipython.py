
""" to research dataset and event-loop object in ipython
"""

from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana import DataSource

ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0104-e000010-xtcav.xtc2')
orun = next(ds.runs())
det = orun.Detector('xtcav')

print('test_xtcav_data    expt: %s runnum: %d\n' % (orun.expt, orun.runnum))

for nev,evt in enumerate(orun.events()):
    if nev>10 : break
    print('Event %03d'%nev, end='')
    #print_ndarr(det.raw.array(evt), '  det.raw.array(evt):')
    print_ndarr(det.raw(evt), '  det.raw(evt):')
    #print('XXXXX', evt._dgrams[0].xtcav[0].raw.raw)

#----------
