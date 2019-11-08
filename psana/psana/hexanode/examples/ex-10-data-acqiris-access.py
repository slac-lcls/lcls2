#----------

from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana import DataSource

ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
myrun = next(ds.runs())
det = myrun.Detector('tmo_hexanode')
det_raw = det.raw

myrun = next(ds.runs())
for nevt,evt in enumerate(myrun.events()):
    if nevt>10 : break
    print('Event %d'%nevt)
    print_ndarr(det_raw.times(evt), '  times : ')
    print_ndarr(det_raw.waveforms(evt), '  wforms: ')

#----------
