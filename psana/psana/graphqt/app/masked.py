#!/usr/bin/env python

import sys
import psana.graphqt.MEDUtils as mu  # includes psana.detector.Utils info_dict, info_command_line, info_namespace, info_parser_arguments, str_tstamp

#from psana.detector.dir_root import DIR_DATA_TEST  # , DIR_REPO # _DARK_PROC
DIR_DATA_TEST = mu.DIR_DATA_TEST # = '/sdf/group/lcls/ds/ana/detector/data2_test'
NDAFNAME = '%s/misc/epix10kaquad-meclv2518-0101-CeO2-ave.npy' % DIR_DATA_TEST
GEOFNAME = '%s/geometry/geo-epix10kaquad-tstx00117.data' % DIR_DATA_TEST

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = 'Usage:'\
      + '\n  %s -a <fname-nda.npy> -k <DataSource-kwargs> -d <detector> -g <fname-geometry.npy> [-L <logging-mode>] [...]' % SCRNAME\
      + '\n\nHelp:\n  %s -h' % SCRNAME\
      + '\n\nExamples:'\
      + '\n  %s' % SCRNAME\
      + '\n  %s -d epix10ka_000001' % SCRNAME\
      + '\n  %s -k exp=ueddaq02' % SCRNAME\
      + '\n  %s -k exp=ueddaq02 -d epix10ka_000001' % SCRNAME\
      + '\n  %s -k exp=ueddaq02,run=5' % SCRNAME\
      + '\n  %s -k exp=ueddaq02,run=5,dir=/sdf/data/lcls/drpsrcf/ffb/ued/ueddaq02/xtc/ -d epix10ka_000002 # for now dir is not used... to access  DB' % SCRNAME\
      + '\n  %s -g %s' % (SCRNAME, GEOFNAME)\
      + '\n  %s -a %s' % (SCRNAME, NDAFNAME)\
#      + '\n  %s -k "{\'exp\':\'abcd01234\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\', ' % SCRNAME\
#      + '\'detectors\':[\'epicsinfo\', \'tmo_opal1\', \'ebeam\']}" -d tmo_opal1 -D'\
#      + '\n\nTest:'\
#      + '\n  %s -k "{\'exp\':\'tmoc00118\', \'run\':123}" -d tmoopal -o ./work' % SCRNAME\
#      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/rixl1013320-r0093-s006-c000.xtc2 -d atmopal -o ./work' % SCRNAME\

def argument_parser():
    from argparse import ArgumentParser

    d_posargs  = []        # [NDAFNAME, GEOFNAME]
    d_ndafname = 'Select'  # NDAFNAME
    d_geofname = 'Select'  # GEOFNAME
    d_dskwargs = 'Select'  #'exp=ueddaq02,run=5' # 'files=<fname.xtc>,exp=<expname>,run=<runs>,dir=<xtc-dir>, ...'
    d_detname  = 'Select'  # 'epix10ka_000001'
    d_dirrepo  = './repo-masked' # DIR_REPO  # '<DIR_ROOT>/detector/calib2'
    d_logmode  = 'INFO'
    d_dirmode  = 0o2775
    d_filemode = 0o664
    d_group    = 'ps-users'
    d_ctab     = 3

#    d_fraclm  = 0.1     # allowed fraction limit
#    d_fraclo  = 0.05    # fraction of statistics [0,1] below low limit
#    d_frachi  = 0.95    # fraction of statistics [0,1] below high limit
#    d_deploy  = False
#    d_tstamp  = None    # 20180910111049 or run number <10000
#    d_comment = 'no comment'
#    d_version = 'V2023-09-15'

    h_posargs = 'list of positional arguments: [<fname-nda.npy>] [<fname-geometry.txt>], default = %s' % d_posargs
    h_ndafname= 'image array file name*.nda, default = %s' % d_ndafname
    h_geofname= 'geometry description constants file name *.txt, *.data, default = %s' % d_geofname
    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_detname     = 'detector name, default = %s' % d_detname
    h_dirrepo = 'repository for files, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (mu.STR_LEVEL_NAMES, d_logmode)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_group   = 'group ownership for all files, default = %s' % d_group
    h_ctab    = 'color table index in range [1,8], default = %d' % d_ctab

#    h_fraclm  = 'fraction of statistics [0,1] below low or above high gate limit to assign pixel bad status, default = %f' % d_fraclm
#    h_fraclo  = 'fraction of statistics [0,1] below low  limit of the gate, default = %f' % d_fraclo
#    h_frachi  = 'fraction of statistics [0,1] above high limit of the gate, default = %f' % d_frachi
#    h_deploy  = 'deploy constants to the calibration DB, default = %s' % d_deploy
#    h_tstamp  = 'non-default time stamp in format YYYYmmddHHMMSS or run number(<10000) for constants selection in repo. '\
#                'By default run time is used, default = %s' % str(d_tstamp)
#    h_comment = 'comment added to constants metadata, default = %s' % str(d_comment)
#    h_version = 'version, default = %s' % str(d_version)

    parser = ArgumentParser(usage=USAGE, description='%s - command opens mask editor GUI' % SCRNAME)
    parser.add_argument('posargs',           default=d_posargs,    type=str,   help=h_posargs, nargs='*')
    parser.add_argument('-a', '--ndafname',  default=d_ndafname,   type=str,   help=h_ndafname)
    parser.add_argument('-d', '--detname',   default=d_detname,    type=str,   help=h_detname)
    parser.add_argument('-k', '--dskwargs',  default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-g', '--geofname',  default=d_geofname,   type=str,   help=h_geofname)
    parser.add_argument('-L', '--logmode',   default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-o', '--dirrepo',   default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('--ctab',            default=d_ctab,       type=int,   help=h_ctab)
    parser.add_argument('--dirmode',         default=d_dirmode,    type=int,   help=h_dirmode)
    parser.add_argument('--filemode',        default=d_filemode,   type=int,   help=h_filemode)
    parser.add_argument('--group',           default=d_group,      type=str,   help=h_group)

#    parser.add_argument('--fraclm',          default=d_fraclm,     type=float, help=h_fraclm)
#    parser.add_argument('--fraclo',          default=d_fraclo,     type=float, help=h_fraclo)
#    parser.add_argument('--frachi',          default=d_frachi,     type=float, help=h_frachi)
#    parser.add_argument('-D', '--deploy',    action='store_true',              help=h_deploy)
#    parser.add_argument('-t', '--tstamp',    default=d_tstamp,     type=int,   help=h_tstamp)
#    parser.add_argument('-C', '--comment',   default=d_comment,    type=str,   help=h_comment)
#    parser.add_argument('-v', '--version',   default=d_version,    type=str,   help=h_version)

    return parser

def mask_editor():
    parser = argument_parser()
    namesp = parser.parse_args()
    posargs = namesp.posargs
    kwargs = vars(namesp)
    print('parser: %s' % mu.ut.info_parser_arguments(parser))
    #if len(posargs)>0 and kwargs['ndafname'] is None: kwargs['ndafname'] = posargs[0]
    #if len(posargs)>1 and kwargs['geofname'] is None: kwargs['geofname'] = posargs[1]

    import psana.graphqt.MEDMain as mm
    mm.mask_editor(**kwargs)
    sys.exit('End of %s'%SCRNAME)

if __name__ == "__main__":
    mask_editor()
    sys.exit('END OF %s' % SCRNAME)

# EOF
