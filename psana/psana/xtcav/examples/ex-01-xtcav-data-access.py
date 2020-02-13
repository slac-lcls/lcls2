#!/usr/bin/env python
""" Access xtcav opal1k camera in xtc2 files
"""

from psana.xtcav.examples.ex_utils import data_file, sys
from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana import DataSource

print('e.g.: [python] %s [test-number]' % sys.argv[0])

#----------

def test_xtcav_data_access() :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'

    ds = DataSource(files=data_file(tname))
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

if __name__ == "__main__":
    test_xtcav_data_access()

#----------
