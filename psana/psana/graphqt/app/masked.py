#!/usr/bin/env python

import sys
from psana.detector.UtilsLogging import STR_LEVEL_NAMES  #, logging
#logger = logging.getLogger(__name__)

from psana.detector.Utils import info_dict, info_command_line, info_namespace, info_parser_arguments, str_tstamp

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = 'Usage:'\
      + '\n  %s -i <fname-nda.npy> -g <fname-geometry.npy> -k <DataSource-kwargs> -d <detector> [-L <logging-mode>] [...]' % SCRNAME\
      + '\n      or short form w/o keyword arguments for image and geometry file names'\
      + '\n  %s <fname-nda.npy> <fname-geometry.npy> [other kwards ...], NOTE: -i and -g override positional arguments' % SCRNAME\
      + '\n\nHelp:\n  %s -h' % SCRNAME
#      + '\n\nExamples:'\
#      + '\n  %s -k exp=tmox49720,run=209 -d epix100 -D' % SCRNAME\
#      + '\n  %s -k exp=tmoc00318,run=10,dir=/a/b/c/xtc -d epix100 -D' % SCRNAME\
#      + '\n  %s -k "{\'exp\':\'abcd01234\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\', ' % SCRNAME\
#      + '\'detectors\':[\'epicsinfo\', \'tmo_opal1\', \'ebeam\']}" -d tmo_opal1 -D'\
#      + '\n\nTest:'\
#      + '\n  %s -k "{\'exp\':\'tmoc00118\', \'run\':123}" -d tmoopal -o ./work' % SCRNAME\
#      + '\n  %s -k exp=tmoc00118,run=123 -d tmoopal -o ./work' % SCRNAME\
#      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/tmoc00318-r0010-s000-c000.xtc2 -d epix100 -o ./work' % SCRNAME\
#      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/tmoc00118-r0222-s006-c000.xtc2 -d tmo_atmopal -o ./work' % SCRNAME\
#      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/rixl1013320-r0093-s006-c000.xtc2 -d atmopal -o ./work' % SCRNAME\

def mask_editor():
    parser = argument_parser()
    namesp = parser.parse_args()
    posargs = namesp.posargs
    kwargs = vars(namesp)
    print('parser: %s' % info_parser_arguments(parser))
    if len(posargs)>0 and kwargs['ndafname'] is None: kwargs['ndafname'] = posargs[0]
    if len(posargs)>1 and kwargs['geofname'] is None: kwargs['geofname'] = posargs[1]

    import psana.graphqt.MEDMain as mm
    mm.mask_editor(**kwargs)
    sys.exit('End of %s'%SCRNAME)

def argument_parser():
    from argparse import ArgumentParser

    from psana.detector.dir_root import DIR_DATA_TEST  # , DIR_REPO # _DARK_PROC
    # /cds/group/psdm/detector/data2_test/geometry
    #ndafname = '%s/misc/epix10kaquad-meclv2518-0099-LAB6-max.npy' % DIR_DATA_TEST
    #geofname = '%s/geometry/geo-epix10kaquad-meclv2518.data' % DIR_DATA_TEST

    ndafname = '%s/misc/jungfrau4m-cxic00318-r0124-lysozyme-max.npy' % DIR_DATA_TEST
    geofname = '%s/geometry/geo-jungfrau-8-segment-cxilv9518.data' % DIR_DATA_TEST

    d_posargs   = [ndafname, geofname]
    d_ndafname = None
    d_geofname = None
    d_dskwargs  = None    # 'files=<fname.xtc>,exp=<expname>,run=<runs>,dir=<xtc-dir>, ...'
    d_det       = None    # 'tmoopal'
#    d_dirrepo = DIR_REPO  # '<DIR_ROOT>/detector/calib2'
    d_logmode = 'INFO'
    d_dirmode = 0o2775
    d_filemode= 0o664
    d_group   = 'ps-users'
    d_fraclm  = 0.1     # allowed fraction limit
    d_fraclo  = 0.05    # fraction of statistics [0,1] below low limit
    d_frachi  = 0.95    # fraction of statistics [0,1] below high limit
    d_deploy  = False
    d_tstamp  = None    # 20180910111049 or run number <10000
    d_comment = 'no comment'
    d_version = 'V2023-09-15'
    d_ictab = 3

    h_posargs = 'list of positional arguments: [<fname-nda.npy>] [<fname-geometry.txt>], default = %s' % d_posargs
    h_ndafname= 'image array file name*.nda, default = %s' % d_ndafname
    h_geofname= 'geometry description constants file name *.txt, *.data, default = %s' % d_geofname
    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_det     = 'detector name, default = %s' % d_det
#    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_group   = 'group ownership for all files, default = %s' % d_group
    h_fraclm  = 'fraction of statistics [0,1] below low or above high gate limit to assign pixel bad status, default = %f' % d_fraclm
    h_fraclo  = 'fraction of statistics [0,1] below low  limit of the gate, default = %f' % d_fraclo
    h_frachi  = 'fraction of statistics [0,1] above high limit of the gate, default = %f' % d_frachi
    h_deploy  = 'deploy constants to the calibration DB, default = %s' % d_deploy
    h_tstamp  = 'non-default time stamp in format YYYYmmddHHMMSS or run number(<10000) for constants selection in repo. '\
                'By default run time is used, default = %s' % str(d_tstamp)
    h_comment = 'comment added to constants metadata, default = %s' % str(d_comment)
    h_version = 'version, default = %s' % str(d_version)
    h_ictab   = 'color table index in range [1,8], default = %d' % d_ictab

    parser = ArgumentParser(usage=USAGE, description='%s - command opens mask editor GUI' % SCRNAME)
    parser.add_argument('posargs',           default=d_posargs,    type=str,   help=h_posargs, nargs='*')
    parser.add_argument('-a', '--ndafname',  default=d_ndafname,   type=str,   help=h_ndafname)
    parser.add_argument('-g', '--geofname',  default=d_geofname,   type=str,   help=h_geofname)
    parser.add_argument('-d', '--det',       default=d_det,        type=str,   help=h_det)
    parser.add_argument('-k', '--dskwargs',  default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-L', '--logmode',   default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('--ictab',           default=d_ictab,      type=int,   help=h_ictab)
#    parser.add_argument('--dirmode',         default=d_dirmode,    type=int,   help=h_dirmode)
#    parser.add_argument('--filemode',        default=d_filemode,   type=int,   help=h_filemode)
#    parser.add_argument('--group',           default=d_group,      type=str,   help=h_group)
#    parser.add_argument('--fraclm',          default=d_fraclm,     type=float, help=h_fraclm)
#    parser.add_argument('--fraclo',          default=d_fraclo,     type=float, help=h_fraclo)
#    parser.add_argument('--frachi',          default=d_frachi,     type=float, help=h_frachi)
#    parser.add_argument('-D', '--deploy',    action='store_true',              help=h_deploy)
#    parser.add_argument('-t', '--tstamp',    default=d_tstamp,     type=int,   help=h_tstamp)
#    parser.add_argument('-C', '--comment',   default=d_comment,    type=str,   help=h_comment)
#    parser.add_argument('-v', '--version',   default=d_version,    type=str,   help=h_version)
#    parser.add_argument('-o', '--dirrepo',   default=d_dirrepo,    type=str,   help=h_dirrepo)

    return parser

if __name__ == "__main__":
    mask_editor()

# EOF
