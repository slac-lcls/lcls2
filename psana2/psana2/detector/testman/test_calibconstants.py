#!/usr/bin/env python

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]
STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'

import logging
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

from psana2.detector.calibconstants import CalibConstants
from psana2.detector.NDArrUtils import info_ndarr


def dict_calib_constants(exp="ueddaq02", detname="epixquad", runnum=0):
    """direct access to calibration constants for run=0
    """
    from psana2.pscalib.calib.MDBWebUtils import calib_constants_all_types
    d = calib_constants_all_types(detname, exp=exp, run=runnum)
    print('==== dict calibcons keys: %s' % (' '.join([k for k in d.keys()])))
    return d


def test_calib_constants():
    d = dict_calib_constants()
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

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('0',): print(USAGE)
elif TNAME in  ('1',): test_calib_constants()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

# EOF
