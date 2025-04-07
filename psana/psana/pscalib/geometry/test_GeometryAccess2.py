#!/usr/bin/env python

import os
import sys
import logging
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]
SCRDIR = os.path.dirname(os.path.realpath(__file__)) # <abs-path-to>/lcls2/psana/psana/pscalib/geometry/

def test_plot_quad():
    """ Test geometry acess methods of the class GeometryAccess object for CSPAD quad.
    """
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays
    from psana.pyalgos.generic.NDArrGenerators import cspad_ndarr # for test purpose only

    basedir = '/sdf/group/lcls/ds/ana/detector/alignment/cspad/calib-cxi-ds1-2014-03-19/'
    fname_data     = basedir + 'cspad-ndarr-ave-cxii0114-r0227.dat'
    fname_geometry = basedir + 'calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data'
    amp_range = (0,500)

    logger.info('%s\nfname_geometry: %s\nfname_data: %s' %(120*'_', fname_geometry, fname_data))

    geometry = GeometryAccess(fname_geometry)
    rows, cols = geometry.get_pixel_coord_indexes('QUAD:V1', 1, pix_scale_size_um=None, xy0_off_pix=None, do_tilt=True)

    # get intensity array
    arr = cspad_ndarr(n2x1=rows.shape[0])
    arr.shape = (8,185,388)
    amp_range = (0,185+388)

    logger.info('shapes rows: %s cols: %s weight: %s' % (str(rows.shape), str(cols.shape), str(arr.shape)))
    img = img_from_pixel_arrays(rows,cols,W=arr)

    import psana.pyalgos.generic.Graphics as gg
    gg.plotImageLarge(img,amp_range=amp_range)
    gg.move(500,10)
    gg.show()


def test_jungfrau16M():
    """
    """
    from time import time
    import psana.pscalib.geometry.GeometryAccess as ga # import GeometryAccess, img_from_pixel_arrays
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d
    import psana.pyalgos.generic.NDArrGenerators as ag

    fname_geo = os.path.join(SCRDIR, 'data/geometry-def-jungfrau16M.data')
    assert os.path.exists(fname_geo)
    logger.info('fngeo: %s' % fname_geo)

    geo = ga.GeometryAccess(fname_geo)
    rows, cols = geo.get_pixel_coord_indexes()
    print(ndu.info_ndarr(rows, 'geo.get_pixel_coord_indexes()'))
    sh3d = ndu.shape_nda_as_3d(rows)
    rows.shape = cols.shape = sh3d
    print(ndu.info_ndarr(rows, 'rows'))
    arr = ag.arr3dincr(sh3d)
    arr1 = ag.arr2dincr() # gg.np.array(arr[0,:])
    for n in range(sh3d[0]):
        arr[n,:] += (10+n)*arr1
    print(ndu.info_ndarr(arr, 'arr'))
    img = ga.img_from_pixel_arrays(rows, cols, W=arr)
    print(ndu.info_ndarr(img, 'img'))
    if True:
        import psana.pyalgos.generic.Graphics as gg
        gg.plotImageLarge(img) #, amp_range=amp_range)
        gg.move(500,10)
        gg.show()
        gg.save_plt(fname='img.png')


def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc'  # None
    d_detname  = 'archon' # None
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    h_tname    = 'test name, usually numeric number, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME, usage=USAGE())
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
    return parser


def USAGE():
    import inspect
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "TNAME in" in s])


def selector():
    parser = argument_parser()
    args = parser.parse_args()
    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    TNAME = args.tname  # sys.argv[1] if len(sys.argv)>1 else '0'

    if   TNAME in  ('0',): test_plot_quad() # cspad quad image
    elif TNAME in  ('1',): test_jungfrau16M() # image of jungfrau16M
    else:
        print(USAGE())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%TNAME)

    exit('END OF TEST %s'%TNAME)


if __name__ == "__main__":
    selector()

# EOF
