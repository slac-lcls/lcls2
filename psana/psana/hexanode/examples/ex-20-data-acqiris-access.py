#!/usr/bin/env python
#----------
"""Loop over events of psana dataset (xtc2 file) and 
   print raw acqiris waveforms and associated sample times
"""

from psana import DataSource
from psana.pyalgos.generic.NDArrUtils import print_ndarr

ds   = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
orun = next(ds.runs())
det  = orun.Detector('tmo_hexanode')

myrun = next(ds.runs())
for n,evt in enumerate(orun.events()):
    if n>10 : break
    print('Event %d'%n)
    print_ndarr(det.raw.times(evt),     '  times : ', last=4)
    print_ndarr(det.raw.waveforms(evt), '  wforms: ', last=4)

#----------
