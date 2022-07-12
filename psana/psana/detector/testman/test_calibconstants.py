#!/usr/bin/env python

from psana.detector.calibconstants import CalibConstants
from psana.detector.NDArrUtils import info_ndarr


import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]

def calib_constants(exp="ueddaq02", detname="epixquad", runnum=0):
    """direct access to calibration constants for run=0
    """
    from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types
    return calib_constants_all_types(detname, exp=exp, run=runnum)


def test_calib_constants():
    d = calib_constants()
    #print('\nd:',d)
    print(50*'-')
    print('d.keys():',d.keys())

    cc = CalibConstants(d)
    print(info_ndarr(cc.pedestals(), 'pedestals'))
    print(info_ndarr(cc.rms(), 'rms'))
    print(info_ndarr(cc.status(), 'status'))
    print(info_ndarr(cc.mask_calib(), 'mask_calib'))
    print(info_ndarr(cc.common_mode(), 'common_mode'))
    print(info_ndarr(cc.gain(), 'gain'))
    print(info_ndarr(cc.gain_factor(), 'gain_factor'))
    print('geotxt_and_meta', cc.geotxt_and_meta())
    ix, iy = cc.pixel_coord_indexes()
    print(info_ndarr(ix, 'ix:'))
    print(info_ndarr(iy, 'iy:'))
    x, y, z = cc.pixel_coords()
    print(info_ndarr(x, 'x:'))
    print(info_ndarr(y, 'y:'))
    print(info_ndarr(z, 'z:'))

    print('shape_as_daq:', cc.shape_as_daq())
    print('number_of_segments_total:', cc.number_of_segments_total())


USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel-e.g.-DEBUG-or-INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - test_calib_constants'\
      + '\n    2 - '\
      + '\n    3 - '\

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('0',): test_calib_constants()
elif TNAME in  ('1',): test_calib_constants()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

# EOF
