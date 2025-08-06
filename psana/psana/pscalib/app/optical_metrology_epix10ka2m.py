#!/usr/bin/env python
#--------------------
""" Created on 2019-01-24 by Mikhail Dubrovin
    2020-03-12 add parameter --rot to account for location of the Q0 in optical measurements.
               expected detector orientation with Q0 in upper-left corner is shown on photo
               Confluence PSDM EPIX10KA2M and EPIX10KAQUAD - Metrology map from ChrisKenney
               Metrology from 2020-02-25 was done with Q0 in the lower-left corner...
    2020-08-12 add parameter --vers to generate constants without quads for uniform detector.
    2020-08-25 add parameters --xip, --yip, --zip, --azip, --qoff, --usez.
    2025-08-06 adopted to lcls2
"""

from psana.pscalib.geometry.UtilsOpticAlignment import OpticalMetrologyEpix10ka2M
import os
import sys
from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, STR_LEVEL_NAMES, init_logger
logger = logging.getLogger(__name__)

usage = '\nCommand to run:'+\
        '\n  %prog'+\
        ' -i <input-file-name> -o <output-file-name> ...'+\
        '\n\n  Example:'+\
        '\n      %prog -i optical-metrology.txt -o results/optmet-2019-01-24'+\
        '\n      %prog -i optical-metrology.txt -r 3 -o results/optmet-2020-02-25'+\
        '\n  Alternative:'+\
        '\n      %prog optical-metrology.txt'


def option_parser():

    from optparse import OptionParser

    d_ifn = './optical_metrology.txt'
    d_ofn = './geometry-epix10ka2m.txt'
    d_log = 'DEBUG'
    d_xc  = 78000
    d_yc  = -4150
    d_xcip= 0
    d_ycip= 0
    d_zcip= 10000 # 10cm
    d_azip= 90
    d_rot = 0
    d_qoff= 0
    d_vers= 1
    d_usez= True
    d_docorr= True

    h_ifn = 'input file name, default = %s' % d_ifn
    h_ofn = 'output file(s) name, default = %s' % d_ofn
    h_log = 'logging level from list (%s), default = %s' % (STR_LEVEL_NAMES, d_log)
    h_xc  = 'x coordinate [um] of camera center offset in optical frame (if nquads<4), default = %d' % d_xc
    h_yc  = 'y coordinate [um] of camera center offset in optical frame (if nquads<4), default = %d' % d_yc
    h_xcip= 'x coordinate [um] of camera center relative IP, default = %d' % d_xcip
    h_ycip= 'y coordinate [um] of camera center relative IP, default = %d' % d_ycip
    h_zcip= 'z coordinate [um] of camera center relative IP, default = %d' % d_zcip
    h_azip= 'angle [deg] of camera Z rotation around IP, default = %.3f' % d_azip
    h_rot = 'detector rotation in optical metrology: 0-Q0 upper-left, 3-Q0 down-left , default = %d' % d_rot
    h_qoff= 'quad index offset relative to DAQ (works in v0 only), default = %d' % d_qoff
    h_vers= 'processor version = 0-quads, 1-monolitic, default = %s' % d_vers
    h_usez= 'use z from optical metrology, otherwise zeros, default = %s' % d_usez
    h_docorr= 'apply detector center mean offset and tilt correction to points, default = %s' % d_docorr

    parser = OptionParser(description='Optical metrology processing for epix10ka2m', usage=usage)
    parser.add_option('-i', '--ifn', default=d_ifn, action='store', type='string', help=h_ifn)
    parser.add_option('-o', '--ofn', default=d_ofn, action='store', type='string', help=h_ofn)
    parser.add_option('-l', '--log', default=d_log, action='store', type='string', help=h_log)
    parser.add_option('-x', '--xc',  default=d_xc,  action='store', type='int',    help=h_xc)
    parser.add_option('-y', '--yc',  default=d_yc,  action='store', type='int',    help=h_yc)
    parser.add_option('-X', '--xcip',default=d_xcip,action='store', type='int',    help=h_xcip)
    parser.add_option('-Y', '--ycip',default=d_ycip,action='store', type='int',    help=h_ycip)
    parser.add_option('-Z', '--zcip',default=d_zcip,action='store', type='int',    help=h_zcip)
    parser.add_option('-A', '--azip',default=d_azip,action='store', type='float',  help=h_azip)
    parser.add_option('-r', '--rot', default=d_rot, action='store', type='int',    help=h_rot)
    parser.add_option('-q', '--qoff',default=d_qoff,action='store', type='int',    help=h_qoff)
    parser.add_option('-v', '--vers',default=d_vers,action='store', type='int',    help=h_vers)
    parser.add_option('-N', '--usez',default=d_usez,action='store_false',          help=h_usez)
    parser.add_option('-C', '--docorr',default=d_docorr,action='store_false',      help=h_docorr)

    return parser


def proc_optical_metrology_epix10ka2m():
    parser = option_parser()
    (popts, pargs) = parser.parse_args()
    print('optional loglevel: %s' % popts.log)

    int_level=DICT_NAME_TO_LEVEL[popts.log]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=int_level)
    #logger.setLevel(int_level)
    #logger.debug('logger.level %d: %s' % (logger.level, logging.getLevelName(logger.level)))

    OpticalMetrologyEpix10ka2M(parser)


if __name__ == '__main__':
    proc_optical_metrology_epix10ka2m()
    sys.exit()

# EOF
