#----------

from psana.pyalgos.generic.NDArrUtils import print_ndarr

from psana import DataSource
ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
myrun = next(ds.runs())
det_raw = myrun.Detector('tmo_hexanode').raw
evts = myrun.events()
evt = next(evts)

#----------

for n,evt in enumerate(evts):
    if n>10 : break
    print('Event %d'%n)
    print_ndarr(det_raw.times(evt), '  times : ', last=4)
    print_ndarr(det_raw.waveforms(evt), '  wforms: ', last=4)

#----------

for n in range(10):
    evt = next(evts)
    print('Event %d'%n)
    print_ndarr(det_raw.times(evt), '  times : ', last=4)
    print_ndarr(det_raw.waveforms(evt), '  wforms: ', last=4)

#----------

dg0 = evt._dgrams[0]
det0 = evt._dgrams[0].tmo_hexanode[0]
det_raw = evt._dgrams[0].tmo_hexanode[0].raw

#----------
