#!/usr/bin/env python

"""
      + '\n== TEST epix10ka2m:'\
      + '\n  cp /sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-epix10ka2m-test.txt .'\
      + '\n  %s -f geo-epix10ka2m-test.txt -o geo-epix10ka2m-test-cframe-psana.geom --cframe 0' % scrname\
      + '\n  %s -d epix10ka -f geo-epix10ka2m-test-cframe-psana.geom -o geo-epix10ka2m-test-back.txt' % scrname\
      + '\n  then see the minor difference between geo-epix10ka2m-test.txt and geo-epix10ka2m-test-back.txt'\
      + '\n  or check image with each of geometry files:'\
      + '\n  geometry_image -g geo-epix10ka2m-test-back.txt -a /sdf/group/lcls/ds/ana/detector/data2_test/npy/nda-mfxc00118-r0184-epix10ka2m-silver-behenate-max.txt -R10 -i0'\
      + '\n'\
      + '\n== TEST cspad (xpp for complete consistency between initial and final file):'\
      + '\n  cp /sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-cspad-test.data .'\
      + '\n  %s -f geo-cspad-test.data -o geo-cspad-test-cframe-psana.geom --cframe 0' % scrname\
      + '\n  %s -d cspadv2 -f geo-cspad-test-cframe-psana.geom -o geo-cspad-test-back.data' % scrname\
      + '\n  geometry_image -g geo-cspad-test.data -a /sdf/group/lcls/ds/ana/detector/data2_test/npy/nda-mfx11116-r0624-e005365-MfxEndstation-0-Cspad-0-max.txt -R10 -i0'\
      + '\n'\
      + '\n== TEST jungfrau:'\
      + '\n  cp /sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-jungfrau-8-test.data .'\
      + '\n  %s -f geo-jungfrau-8-test.data -o geo-jungfrau-8-test-cframe-psana.geom --cframe 0' % scrname\
      + '\n  %s -d jungfrau -f geo-jungfrau-8-test-cframe-psana.geom -o geo-jungfrau-8-test-back.data' % scrname\
      + '\n  geometry_image -g geo-jungfrau-8-test.data -a /sdf/group/lcls/ds/ana/detector/data2_test/npy/nda-cxilv9518-r0008-jungfrau-lysozyme-max.npy -R10 -i0'\
"""
#    /sdf/group/lcls/ds/ana/detector/data2_test/ >>> /sdf/group/lcls/ds/ana/detector/data2_test/
import sys
#from Detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES
from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES
#import logging

import argparse

def do_main():

    scrname = sys.argv[0].rsplit('/')[-1]

    fname_jungfrau_8     = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-jungfrau-8-segment.data'
    fname_epix10ka2m_16  = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-epix10ka2m-16-segment.data'
    fname_epix10ka2m_def = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-epix10ka2m-default.data'
    fname_cspad_cxi      = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-cspad-cxi.data'
    fname_pnccd_amo      = '/sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-pnccd-amo.data'

    d_dettype = 'epix10ka'
    d_fname   = fname_epix10ka2m_16
    d_ofname  = 'geo-crystfel.geom'
    d_loglev  ='INFO'
    d_cframe  = 1
    d_dsname  = None
    d_zpvname = None
    d_f_um    = 1000.

    h_cframe = 'coordinate frame 0/1 for psana/LAB, def=%s. Works for PSANA->CRYSTFEL conversion ONLY'% d_cframe\
             + ' where it selects frame for pixel coordinates. Backward conversion CRYSTFEL->PSANA'\
               ' does not change frame. For test perpose it is assumed that forth and back conversion'\
               ' PSANA->CRYSTFEL->PSANA should not change constants in the geometry file (up to precision lose at conversion)'\
               ' if they are defined for the same coordinate frame. For this type of test parameter should be set to 0 to keep psana frame unchanged.'
    h_fname = 'input geometry file name. File name extention *.geom launches converter from CrystFEL to psana, def=%s' % d_fname
    h_dettype = 'USED FOR CRYSTFEL TO PSANA ONLY - detector type, one of epix10ka, jungfrau, cspad (with quads), cspadv2 (no quads), pnccd, def=%s' % d_dettype

    usage = '\n  %s -h' % scrname\
      + '\n  %s -f %s -o geo-crystfel.geom' % (scrname, d_fname)\
      + '\n'\
      + '\n  cp /sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-cspad-xpp.data .'\
      + '\n  %s -f geo-cspad-xpp.data -o geo-cspad-xpp-crystfel.geom                 # conversion from psana to crystfel' % scrname\
      + '\n  %s -d cspadv2 -f geo-cspad-xpp-crystfel.geom -o geo-cspad-xpp-back.data # conversion from crystfel to psana' % scrname\
      + '\n'\
      + '\n  cp /sdf/group/lcls/ds/ana/detector/data2_test/geometry/geo-jungfrau-8-segment.data .'\
      + '\n  %s -f geo-jungfrau-8-segment.data -o geo-jungfrau-8-segment.geom        # conversion from psana to crystfel' % scrname\
      + '\n  %s -f geo-jungfrau-8-segment.data -o geo-jungfrau-8-segment.geom --dsname exp=cxic00318:run=123:smd --zpvname CXI:DS1:MMS:06.RBV' % scrname\
      + '\n  %s -d jungfrau -f geo-jungfrau-8-test-cframe-psana.geom -o geo-jungfrau-8-test-back.data  # conversion from crystfel to psana' % scrname\
      + '\n  (TBD) geometry_image -g geo-jungfrau-8-test.data -a /sdf/group/lcls/ds/ana/detector/data_test/npy/nda-cxilv9518-r0008-jungfrau-lysozyme-max.npy'
      + '\n'

    parser = argparse.ArgumentParser(usage=usage, description='Converts geometry constants from psana to CrystFEL format and backward (see --fname).')
    parser.add_argument('-f', '--fname',   default=d_fname,   type=str, help=h_fname)
    parser.add_argument('-o', '--ofname',  default=d_ofname,  type=str, help='output file name, def=%s' % d_ofname)
    parser.add_argument('-l', '--loglev',  default=d_loglev,  type=str, help='logging level name, one of %s, def=%s' % (STR_LEVEL_NAMES, d_loglev))
    parser.add_argument('--cframe',        default=d_cframe,  type=int, help=h_cframe)
    parser.add_argument('-d', '--dettype', default=d_dettype, type=str, help=h_dettype)
    parser.add_argument('--dsname',        default=d_dsname,  type=str, help='FOR Z CORRECTION FROM DATA - dataset (str) like exp=<experiment>:run=<run-number>:smd:..., def=%s' % d_dsname)
    parser.add_argument('--zpvname',       default=d_zpvname, type=str, help='FOR Z CORRECTION FROM DATA - z-correction variable name ex: CXI:DS1:MMS:06.RBV or alias ex: DscCsPad_z, def=%s' % d_zpvname)
    parser.add_argument('--f_um',          default=d_f_um, type=float, help='FOR Z CORRECTION FROM DATA - factor for conversion PV value to um, def=%f' % d_f_um)

    args = parser.parse_args()
    s = 'Arguments:'
    for k,v in vars(args).items(): s += '  %12s : %s' % (k, str(v))

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(filename)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=DICT_NAME_TO_LEVEL[args.loglev])
    logging.debug('Logger is initialized for level %s' % args.loglev)
    logging.info(s)

    extent = args.fname.rsplit('.',1)[-1]
    logging.info('input file name extension %s' % extent)
    if extent == 'geom':
#        from PSCalib.UtilsConvertCrystFEL import convert_crystfel_to_geometry
        from psana.pscalib.geometry.UtilsConvertCrystFEL import convert_crystfel_to_geometry
        convert_crystfel_to_geometry(args)
    else:
#        from PSCalib.UtilsConvert import convert_geometry_to_crystfel
        from psana.pscalib.geometry.UtilsConvert import convert_geometry_to_crystfel
        convert_geometry_to_crystfel(args)

    sys.exit('END OF %s' % scrname)


if __name__ == "__main__":
    do_main()

# EOF
