#!/usr/bin/env python

from psana.detector.UtilsLogging import sys, logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES

from psana.detector.NDArrUtils import info_ndarr, reshape_to_3d, shape_nda_as_3d
from psana.detector.mask import Mask
import psana.detector.UtilsGraphics as ug


def test_umask(det):
    import numpy as np
    shape_raw = det.raw._shape_as_daq()
    mask = np.ones(shape_raw, dtype=np.int8)
    mask.shape = shape_nda_as_3d(mask)
    info_ndarr(mask, 'test_umask:')
    mask[0, 100:200, 200:300] = 0
    mask[1, 50:150, 200:300] = 0
    mask.shape = shape_raw
    return mask


def test_mask_select(tname, det):
    mask = None
    if   tname ==  '0': # print this list of tests
        pass
    #    tname ==
    #    tname ==  Tests of mask using class Mask(det, **kwargs).mask() with **kwargs arguments passed to the object
    if   tname ==  '1': # Mask(det, status=True, ... - mask from status
        mask = Mask(det, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None)\
               .mask()

    elif tname ==  '2': # Mask(det, status=True, ..., neighbors=True - mask from status and neighbors
        mask = Mask(det,\
                    status=True, status_bits=(1<<64)-1, stextra_bits=(1<<64)-1, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=9, ptrn='r',\
                    edges=False, \
                    center=False,\
                    calib=False,\
                    umask=None).mask()

    elif tname ==  '3': # Mask(det, edges=True, ... - mask edges
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=False,\
                    calib=False,\
                    umask=None).mask()

    elif tname ==  '4': # Mask(det, center=True, ... - mask center rows/columns
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=False,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=False,\
                    umask=None).mask()

    elif tname ==  '5': # Mask(det, calib=True, ... - mask calib
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=False,\
                    center=False,\
                    calib=True,\
                    umask=None).mask()

    elif tname ==  '6': # Mask(det, umask=test_umask(det), ... - mask users
        mask = Mask(det,\
                    status=False,\
                    neighbors=False,\
                    edges=False,\
                    center=False,\
                    calib=False,\
                    umask=test_umask(det)).mask()

    elif tname ==  '7': # Mask(det, status=True, edges=True, calib=True, center=True, calib=True,umask=..., ... - mask combined
        mask = Mask(det,\
                    status=True, status_bits=0xffff, stextra_bits=(1<<64)-1, gain_range_inds=(0,1,2),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=True,\
                    umask=test_umask(det)).mask()

    elif tname ==  '9': # random normal distribution
        import psana.pyalgos.generic.NDArrGenerators as ag
        sh = det.raw._shape_as_daq()  # (4, 352, 384)
        mask = ag.random_standard(shape=sh, mu=0, sigma=0.25, dtype=float)
    #    tname ==
    #    tname ==  Tests of mask using class Mask(det).mask_*(**kwargs) - with **kwargs passed through specific methods
    elif tname == '11': # Mask(det).mask_from_status - mask from status
        mask = Mask(det).mask_from_status(status_bits=0xffff, stextra_bits=(1<<64)-1)

    elif tname == '12': # Mask(det).mask_from_status -> mask_neighbors - mask from status and neighbors
        o = Mask(det)
        msts = o.mask_from_status(status_bits=0xffff, stextra_bits=(1<<64)-1)
        mask = o.mask_neighbors(msts, rad=9, ptrn='r')

    elif tname == '13': # Mask(det).mask_edges - mask edges
        mask = Mask(det).mask_edges(width=0, edge_rows=10, edge_cols=5)

    elif tname == '14': # Mask(det).mask_center - mask center rows/columns
        mask = Mask(det).mask_center(wcenter=0, center_rows=5, center_cols=3)

    elif tname == '15': # Mask(det).mask_calib_or_default - mask calib or default
        mask = Mask(det).mask_calib_or_default()

    elif tname == '16': # test_umask - somehow defined user's mask
        mask = test_umask(det)

    elif tname == '17': # Mask(det).mask_comb - mask combined
        mask = Mask(det).mask_comb(\
                    status=True, status_bits=0xffff, stextra_bits=(1<<64)-1, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=True,\
                    umask=test_umask(det),\
                    force_update=False)

    #    tname ==
    #    tname ==  Tests of mask using det.raw._mask_* - directly from hidden methods
    elif tname == '21': # det.raw._mask_from_status - mask from status
        mask = det.raw._mask_from_status(status_bits=0xffff, stextra_bits=(1<<64)-1)

    elif tname == '22': # det.raw._mask_from_status and _mask_neighbors - mask from status and neighbors
        msts = det.raw._mask_from_status(status_bits=0xffff, stextra_bits=(1<<64)-1)
        mask = det.raw._mask_neighbors(msts, rad=9, ptrn='r')

    elif tname == '23': # det.raw._mask_edges - mask edges
        mask = det.raw._mask_edges(width=0, edge_rows=10, edge_cols=5)

    elif tname == '24': # det.raw._mask_center - mask center rows/columns
        mask = det.raw._mask_center(wcenter=0, center_rows=5, center_cols=3)

    elif tname == '25': # det.raw._mask_calib_or_default - mask calib or default
        mask = det.raw._mask_calib_or_default()

    elif tname == '26': # test_umask - mask users
        mask = test_umask(det)

    elif tname == '27': # det.raw._mask_comb - mask combined
        mask = det.raw._mask_comb(\
                    status=True, status_bits=0xffff, stextra_bits=(1<<64)-1, gain_range_inds=(0,1,2,3,4),\
                    neighbors=True, rad=5, ptrn='r',\
                    edges=True, width=0, edge_rows=10, edge_cols=5,\
                    center=True, wcenter=0, center_rows=5, center_cols=3,\
                    calib=True,\
                    umask=test_umask(det),\
                    force_update=False)

    else:
        sys.exit('TEST %s IS NOT IMPLEMENTED' % tname)

    logger.info(info_ndarr(mask, '\nmask'))
    return mask


def test_mask(**kwargs):
    """
    """
    from psana import DataSource
    import psana.detector.utils_psana as up
    tname = kwargs.get('tname', '0')
    str_dskwargs = kwargs.get('dskwargs', 'exp=ued101066,run=181,dir=/sdf/data/lcls/ds/prj/public01/xtc')
    dskwargs = up.datasource_kwargs_from_string(str_dskwargs)
    detname = kwargs.get('detname', 'epixquad')
    print(f'dskwargs: {str(dskwargs)}\ndetname: {detname}')

    #ds = DataSource(exp='ued101066', run=181, dir='/sdf/data/lcls/ds/prj/public01/xtc')
    ds = DataSource(**dskwargs)
    orun = next(ds.runs())
    det = orun.Detector(detname)  # epixquad is replaced by epix10ka_000002
    peds, meta = det.calibconst['pedestals']
    logger.info('\nmetadata\n', meta)
    logger.info(info_ndarr(peds, '\npedestals'))

    mask = test_mask_select(tname, det)

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


SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = '  python %s <test-name> <loglevel e.g. DEBUG or INFO>' % SCRNAME\
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

def USAGE():
    import inspect
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
         + '\n'.join([s for s in inspect.getsource(test_mask_select).split('\n') if "tname ==" in s])\
         + '\n\nHELP:\n  list of parameters: ./%s -h\n  list of tests:      ./%s 0' % (SCRNAME, SCRNAME)

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'


#def argument_parser():
#    import argparse
#    d_logmode = 'INFO'
#    d_tname = '0'
#    h_logmode = 'logging mode, one of the list %s, default=%s' % (STR_LEVEL_NAMES, d_logmode)
#    parser = argparse.ArgumentParser(description='test of detector/mask_algos.py', usage=USAGE)
#    parser.add_argument('tname', default=d_tname, type=str,   help='test name, default=%s' % d_tname)
#    parser.add_argument('-L', '--logmode', default=d_logmode, type=str,   help=h_logmode)
#    parser.add_argument('-D', '--deploy', action='store_true', help='deploy test status_extra to DB')
#    return parser


def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    #d_dskwargs = 'exp=ued101066,run=181,dir=/sdf/data/lcls/ds/prj/public01/xtc'  # None
    #d_detname  = 'epixquad'
    d_dskwargs = 'exp=xpp101570426,run=26'
    d_detname  = 'jungfrau1M' # None
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    h_tname    = 'test name, usually numeric number 0,...,>20, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    parser = ArgumentParser(description=f'{SCRNAME} is a bunch of tests for mask methods',\
                            usage=f'list of implemented tests:\n{USAGE()}')
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
#    parser.add_argument('-D', '--deploy', action='store_true', help='deploy test status_extra to DB')
    return parser


global args
parser = argument_parser()
args = parser.parse_args() # Namespace
kwargs = vars(args)        # dict
#print('args:', args)
#print('kwargs:', kwargs)
#sys.exit('TEST EXIT')

loglevelnum = DICT_NAME_TO_LEVEL[args.loglevel]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=loglevelnum)

print('\nUsage:\n' + USAGE())
test_mask(**kwargs)
exit(f'END OF TEST {args.tname}')

#if __name__ == "__main__":
# EOF
