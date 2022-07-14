#!/usr/bin/env python

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]
STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'

import logging
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

from psana.detector.NDArrUtils import info_ndarr
from psana.detector.mask import Mask
import psana.detector.UtilsGraphics as ug


def test_umask(det):
    import numpy as np
    mask = np.ones(det.raw._shape_as_daq(), dtype=np.int8)
    mask[1, 100:200, 200:300] = 0
    mask[2, 50:150, 200:300] = 0
    return mask


def test_mask_select(tname, det):
    mask = None
    if tname == '9':
        import psana.pyalgos.generic.NDArrGenerators as ag
        sh = det.raw._shape_as_daq()  # (4, 352, 384)
        mask = ag.random_standard(shape=sh, mu=0, sigma=0.25, dtype=float)

    elif tname == '1':
        mask = Mask(det, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None)\
               .mask()

    elif tname == '2':
        mask = Mask(det,\
                    status=True, status_bits=0xffff, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=False, \
                    center=False,\
                    calib=False,\
                    umask=None).mask()

    elif tname == '3':
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=False,\
                    calib=False,\
                    umask=None).mask()

    elif tname == '4':
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=False,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=False,\
                    umask=None,\
                    force_update=False).mask()

    elif tname == '5':
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=False,\
                    center=False,\
                    calib=True,\
                    umask=None).mask()

    elif tname == '6':
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=False,\
                    center=False,\
                    calib=False,\
                    umask=test_umask(det)).mask()

    elif tname == '7':
        mask = Mask(det,\
                    status=True, status_bits=0xffff, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=True,\
                    umask=test_umask(det),\
                    force_update=False).mask()

    elif tname == '11':
        mask = Mask(det).mask_from_status(status_bits=0xffff)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '12':
        o = Mask(det)
        msts = o.mask_from_status(status_bits=0xffff)    # dtype=DTYPE_MASK, **kwa)
        mask = o.mask_neighbors(msts, rad=9, ptrn='r')

    elif tname == '13':
        mask = Mask(det).mask_edges(width=0, edge_rows=10, edge_cols=5)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '14':
        mask = Mask(det).mask_center(wcenter=0, center_rows=5, center_cols=3)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '15':
        mask = Mask(det).mask_calib_or_default()  # dtype=DTYPE_MASK)

    elif tname == '16':
        mask = test_umask(det)

    elif tname == '17':
        mask = Mask(det).mask_comb(\
                    status=True, status_bits=0xffff, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=True,\
                    umask=test_umask(det),\
                    force_update=False)  # dtype=DTYPE_MASK)

    elif tname == '21':
        mask = det.raw._mask_from_status(status_bits=0xffff)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '22':
        msts = det.raw._mask_from_status(status_bits=0xffff)  # dtype=DTYPE_MASK, **kwa)
        mask = det.raw._mask_neighbors(msts, rad=9, ptrn='r')

    elif tname == '23':
        mask = det.raw._mask_edges(width=0, edge_rows=10, edge_cols=5)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '24':
        mask = det.raw._mask_center(wcenter=0, center_rows=5, center_cols=3)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '25':
        mask = det.raw._mask_calib_or_default()  # dtype=DTYPE_MASK)

    elif tname == '26':
        mask = test_umask(det)

    elif tname == '27':
        mask = det.raw._mask_comb(\
                    status=True, status_bits=0xffff, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=True,\
                    umask=test_umask(det),\
                    force_update=False)  # dtype=DTYPE_MASK)

    logger.info(info_ndarr(mask, '\nmask'))
    return mask


def test_mask(tname):
    """
    """
    from psana import DataSource
    ds = DataSource(exp='uedcom103',run=7, dir='/cds/data/psdm/prj/public01/xtc')
    orun = next(ds.runs())
    det = orun.Detector('epixquad')  # epixquad is replaced by epix10ka_000002
    peds, meta = det.calibconst['pedestals']
    logger.info('\nmetadata\n', meta)
    logger.info(info_ndarr(peds, '\npedestals'))

    mask = test_mask_select(tname, det)  # [0,:]

    #sys.exit('TEST EXIT')

    evt = next(orun.events())

    arr = mask + 1

    logger.info(info_ndarr(arr, '\ntest_mask arr for image'))

    img = det.raw.image(evt, nda=arr)
    logger.info(info_ndarr(img, '\nimg'))

    flimg = ug.fleximagespec(img, arr=arr, amin=0, amax=2)
    #   else: flimg.update(img)#, arr=arr)
    flimg.axtitle(title='test_mask %s' % tname)
    #   ug.gr.show(mode='NO HOLD')
    ug.gr.show()


USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel e.g. DEBUG or INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n\n Mask(det, **kwa)'\
      + '\n    0 - print usage'\
      + '\n    1 - mask from status'\
      + '\n    2 - mask from status and neighbors'\
      + '\n    3 - mask edges'\
      + '\n    4 - mask center rows/columns'\
      + '\n    5 - mask calib'\
      + '\n    6 - mask users'\
      + '\n    7 - mask combined'\
      + '\n    9 - random normal distribution'\
      + '\n\n Mask(det).mask_*(**kwa)'\
      + '\n   11 - mask from status'\
      + '\n   12 - mask from status and neighbors'\
      + '\n   13 - mask edges'\
      + '\n   14 - mask center rows/columns'\
      + '\n   15 - mask calib or default'\
      + '\n   16 - mask users'\
      + '\n   17 - mask combined'\
      + '\n\n Direct call to det.raw._mask_*(**kwa)'\
      + '\n   21 - mask from status'\
      + '\n   22 - mask from status and neighbors'\
      + '\n   23 - mask edges'\
      + '\n   24 - mask center rows/columns'\
      + '\n   25 - mask calib or default'\
      + '\n   26 - mask users'\
      + '\n   27 - mask combined'\


TNAME = sys.argv[1] if len(sys.argv)>1 else '0'


if TNAME in ( '1', '2', '3', '4', '5', '6', '7', '9',\
             '11','12','13','14','15','16','17',\
             '21','22','23','24','25','26','27'): test_mask(TNAME)
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

#if __name__ == "__main__":
# EOF
