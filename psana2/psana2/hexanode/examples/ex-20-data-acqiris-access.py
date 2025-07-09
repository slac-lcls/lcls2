#!/usr/bin/env python

"""- loop over events of psana dataset (xtc2 file) and
   - print raw acqiris waveforms and associated sample times
"""

from psana import DataSource
from psana2.pyalgos.generic.NDArrUtils import print_ndarr
from psana2.hexanode.examples.ex_test_data import DIR_DATA_TEST

ds   = DataSource(files='%s/%s' % (DIR_DATA_TEST, 'data-amox27716-r0100-acqiris-e000100.xtc2'))
orun = next(ds.runs())
det  = orun.Detector('tmo_quadanode') # 'tmo_hexanode'

for nev,evt in enumerate(orun.events()):
    if nev>10 : break
    print('Event %d'%nev)
    print_ndarr(det.raw.times(evt),     '  times : ', last=4)
    print_ndarr(det.raw.waveforms(evt), '  wforms: ', last=4)

# EOF
