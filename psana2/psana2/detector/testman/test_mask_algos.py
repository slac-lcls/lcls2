#!/usr/bin/env python

from psana2.detector.UtilsLogging import sys, logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES
from psana2.detector.NDArrUtils import info_ndarr, save_ndarray_in_textfile
from psana2.detector.mask_algos import MaskAlgos, DTYPE_MASK, DTYPE_STATUS


def test_umask(mo):
    import numpy as np
    m = mo.mask_default() # (4, 352, 384)
    mask = np.ones_like(m, dtype=DTYPE_MASK)
    mask[1, 100:200, 200:300] = 0
    mask[2, 50:150, 200:300] = 0
    return mask


def deploy_status_extra(mo, fname = 'epix10ka-status_extra-test.data',\
       cmd = 'cdb add -e uedcom103 -d epix10ka_000002 -c status_extra -r 7 -f %s'):
    """ test is designed for epix10ka quad ONLY!!!"""

    #import numpy as np
    from psana2.detector.Utils import os, np, get_cwd

    m = mo.mask_default() # (4, 352, 384) -> (7, 4, 352, 384)
    logger.info(info_ndarr(m, 'deploy_status_extra - grab mask_default and make test status'))
    nsegs = m.shape[0]
    shape = (7,) + tuple(m.shape)
    a = np.zeros(shape, dtype=DTYPE_STATUS)
    for i in range(nsegs):
        for n in range(1,5): a[0, i, 50*n:50*n+20, :] = n
    _fname = os.path.join(get_cwd(), fname)
    save_ndarray_in_textfile(a, _fname, 0o664, '%d', umask=0o0, group='ps-users')
    logger.info(info_ndarr(a, 'test array of status_extra'))
    print('file saved: %s' % _fname)
    _cmd = cmd % _fname
    print('deploy file: %s' % _cmd)

    if args.deploy:
        os.system(_cmd)
    else:
        print('WARNING: to deploy constants in DB (if they are not already there) add option -D to the command line')

    a1 = a[0]
    logger.info(info_ndarr(a1, 'test array for one of gain ranges:'))
    return a1


def test_mask_select(tname, mo):
    mask = None
    if tname == '9':
import psana2.pyalgos.generic.NDArrGenerators as ag
        status = mo.mask_from_status()
        sh = status.shape  # (4, 352, 384)
        mask = ag.random_standard(shape=sh, mu=0, sigma=0.25, dtype=float)

    elif tname == '1':
        mask = mo.mask_from_status()  # status_bits=0xffff, gain_range_inds=(0,1,2,3,4), dtype=DTYPE_MASK, **kwa)

    elif tname == '2':
        msts = mo.mask_from_status()
        mask = mo.mask_neighbors(msts, rad=9, ptrn='r')

    elif tname == '3':
        mask = mo.mask_edges(width=0, edge_rows=10, edge_cols=5)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '4':
        mask = mo.mask_center(wcenter=0, center_rows=5, center_cols=3)  # dtype=DTYPE_MASK, **kwa)

    elif tname == '5':
        mask = mo.mask_calib_or_default()  # dtype=DTYPE_MASK)

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
    elif tname == '8':
        mask = deploy_status_extra(mo)
    else:
        mask = None
        sys.exit('TEST %s IS NOT IMPLEMENTED' % tname)

    logger.info(info_ndarr(mask, 'mask'))
    return mask


def dict_calib_constants(exp='uedcom103', detname='epix10ka_000002', runnum=7):
    """returns dict of all calibration constants using direct access DB
    """
    from psana2.pscalib.calib.MDBWebUtils import calib_constants_all_types
    d = calib_constants_all_types(detname, exp=exp, run=runnum)
    logging.info('==== exp: %s runnum: %d detname: %s' % (exp, runnum, detname))
    logging.info('dict calibcons keys: %s' % ' '.join([k for k in d.keys()]))
    return d


def calib_constants(dcc):
    from psana2.detector.calibconstants import CalibConstants
    return CalibConstants(dcc)


def mask_algos(dcc):
    return MaskAlgos(dcc)


def test_mask_algos(tname):
    """
    """
    dcc = dict_calib_constants(exp='uedcom103', detname='epix10ka_000002', runnum=7)
    mo = mask_algos(dcc)
    cc = calib_constants(dcc)
    #logging.info(info_ndarr(cc.pedestals(), 'pedestals'))
    #logging.info(info_ndarr(mo.mask_from_status(), 'pedestals'))

    mask = test_mask_select(tname, mo)

    if mask is None:
        logger.info('mask is None')
        return
    arr = mask + 1
    logger.info(info_ndarr(arr, 'test_mask arr for image'))

    rows, cols = cc.pixel_coord_indexes()
    #cc.cached_pixel_coord_indexes()
    #rows, cols = cc.pix_rc()

    from psana2.detector.UtilsAreaDetector import img_from_pixel_arrays
    img = img_from_pixel_arrays(rows, cols, weight=arr, vbase=0)
    logger.info(info_ndarr(img, 'img'))


import psana2.detector.UtilsGraphics as ug

    flimg = ug.fleximagespec(img, arr=arr) #, amin=0, amax=2)
    flimg.axtitle(title='test_mask %s' % tname)
    #   ug.gr.show(mode='NO HOLD')
    ug.gr.show()

    #sys.exit('TEST EXIT')


SCRNAME = sys.argv[0].rsplit('/')[-1]


USAGE = '  python %s <test-name> <loglevel e.g. DEBUG or INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n\n Direct call to MaskAlgos.mask_*'\
      + '\n   0 - print usage'\
      + '\n   1 - mask from status'\
      + '\n   2 - mask from status and neighbors'\
      + '\n   3 - mask edges'\
      + '\n   4 - mask center rows/columns'\
      + '\n   5 - mask calib or default'\
      + '\n   6 - mask users'\
      + '\n   7 - mask combined'\
      + '\n   8 - deploy in the DB constants of type status_extra for test purpose'\
      + '\n   9 - test random image'


def argument_parser():
    import argparse
    d_logmode = 'INFO'
    d_tname = '0'
    h_logmode = 'logging mode, one of the list %s, default=%s' % (STR_LEVEL_NAMES, d_logmode)
    parser = argparse.ArgumentParser(description='test of detector/mask_algos.py', usage=USAGE)
    parser.add_argument('tname', default=d_tname, type=str,   help='test name, default=%s' % d_tname)
    parser.add_argument('-L', '--logmode', default=d_logmode, type=str,   help=h_logmode)
    parser.add_argument('-D', '--deploy', action='store_true', help='deploy test status_extra to DB')
    return parser


global args
parser = argument_parser()
args = parser.parse_args()
#print('args:', args)

tname = args.tname
loglevel = DICT_NAME_TO_LEVEL[args.logmode]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=loglevel)

logging.info('\nUsage:\n' + USAGE)
test_mask_algos(tname)
sys.exit('END OF TEST %s' % tname)

#if __name__ == "__main__":
# EOF
