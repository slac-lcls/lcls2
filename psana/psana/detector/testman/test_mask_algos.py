#!/usr/bin/env python

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]
STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'

import logging
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

from psana.detector.NDArrUtils import info_ndarr
from psana.detector.calibconstants import CalibConstants
from psana.detector.mask_algos import MaskAlgos, DTYPE_MASK
import psana.detector.UtilsGraphics as ug

def test_umask(mo):
    import numpy as np
    m = mo.mask_default() # (4, 352, 384)
    mask = np.ones_like(m, dtype=DTYPE_MASK)
    mask[1, 100:200, 200:300] = 0
    mask[2, 50:150, 200:300] = 0
    return mask


def test_mask_select(tname, mo):
    mask = None
    if tname == '0':
        import psana.pyalgos.generic.NDArrGenerators as ag
        status = mo.mask_from_status()
        sh = status.shape # (4, 352, 384)
        mask = ag.random_standard(shape=sh, mu=0, sigma=0.25, dtype=float)

    elif tname == '1':
        mask = mo.mask_from_status()  # status_bits=0xffff, gain_range_inds=(0,1,2,3,4), dtype=DTYPE_MASK, **kwa)

    elif tname == '2':
        msts = mo.mask_from_status()
        mask = mo.mask_neighbors(msts, rad=9, ptrn='r')

    elif tname == '3':
        mask = mo.mask_edges(width=0, edge_rows=10, edge_cols=5)#, dtype=DTYPE_MASK, **kwa)

    elif tname == '4':
        mask = mo.mask_center(wcenter=0, center_rows=5, center_cols=3)#, dtype=DTYPE_MASK, **kwa)

    elif tname == '5':
        mask = mo.mask_calib_or_default()#, dtype=DTYPE_MASK)

    elif tname == '6':
        mask = test_umask(mo)

    elif tname == '7':
        mask = mo.mask_comb(\
                            status=True, status_bits=0xffff, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=True,\
                    umask=test_umask(mo),\
                    force_update=False)
    else:
        mask = None

    logger.info(info_ndarr(mask, 'mask'))
    return mask


def dict_calib_constants(exp='uedcom103', detname="epixquad", runnum=7):
    """direct access to calibration constants for run=0
    """
    from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types
    d = calib_constants_all_types(detname, exp=exp, run=runnum)
    logging.info('==== dict calibcons keys: %s' % ' '.join([k for k in d.keys()]))
    return d


def calib_constants(dcc):
    return CalibConstants(dcc)


def mask_algos(dcc):
    return MaskAlgos(dcc)


def test_mask_algos(tname):
    """
    """
    dcc = dict_calib_constants()
    mo = mask_algos(dcc)
    cc = calib_constants(dcc)
    #logging.info(info_ndarr(cc.pedestals(), 'pedestals'))
    #logging.info(info_ndarr(mo.mask_from_status(), 'pedestals'))

    mask = test_mask_select(tname, mo)
    arr = mask + 1
    logger.info(info_ndarr(arr, 'test_mask arr for image'))

    rows, cols = cc.pixel_coord_indexes()
    #cc.cached_pixel_coord_indexes()
    #rows, cols = cc.pix_rc()

    from psana.detector.UtilsAreaDetector import img_from_pixel_arrays
    img = img_from_pixel_arrays(rows, cols, weight=arr, vbase=0)
    logger.info(info_ndarr(img, 'img'))

    flimg = ug.fleximagespec(img, arr=arr, amin=0, amax=2)
    flimg.axtitle(title='test_mask %s' % tname)
    #   ug.gr.show(mode='NO HOLD')
    ug.gr.show()

    #sys.exit('TEST EXIT')


USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel e.g. DEBUG or INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n\n Direct call to MaskAlgos.mask_*'\
      + '\n   0 - test random image'\
      + '\n   1 - mask from status'\
      + '\n   2 - mask from status and neighbors'\
      + '\n   3 - mask edges'\
      + '\n   4 - mask center rows/columns'\
      + '\n   5 - mask calib or default'\
      + '\n   6 - mask users'\
      + '\n   7 - mask combined'\


TNAME = sys.argv[1] if len(sys.argv)>1 else '0'


if TNAME in ('0','1','2','3','4','5','6','7'):  # ,'8','9'):
    test_mask_algos(TNAME)
else:
    logging.info(USAGE)
    sys.exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

sys.exit('END OF TEST %s'%TNAME)

#if __name__ == "__main__":
# EOF
