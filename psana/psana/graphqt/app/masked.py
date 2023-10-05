#!/usr/bin/env python

import sys
import psana.graphqt.MEDUtils as mu  # includes psana.detector.Utils info_dict, info_command_line, info_namespace, info_parser_arguments, str_tstamp

import logging
STR_LEVEL_NAMES = ', '.join(logging._nameToLevel.keys())
DIR_DATA_TEST = mu.DIR_DATA_TEST # = '/sdf/group/lcls/ds/ana/detector/data2_test'
NDAFNAME = '%s/misc/epix10kaquad-meclv2518-0101-CeO2-ave.npy' % DIR_DATA_TEST
GEOFNAME = '%s/geometry/geo-epix10kaquad-tstx00117.data' % DIR_DATA_TEST

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = '\n  %s -a <fname-nda.npy> -k <DataSource-kwargs> -d <detector> -g <fname-geometry.data/txt> [-L <logging-mode>] [...]' % SCRNAME\
      + '\n\nget help:'\
      + '\n  %s -h' % SCRNAME\
      + '\n\nexamples (* - recommended):'\
      + '\n *1) %s  # launch application without parameters and set them in GUI' % SCRNAME\
      + '\n *2) %s -d epix10ka_000001 -k exp=ueddaq02,run=569 -a %s  # ndarray from file, geometry from experiment DB' % (SCRNAME, NDAFNAME)\
      + '\n  3) %s -d epix10ka_000001 -a %s  # ndarray from file, geometry from detector DB' % (SCRNAME, NDAFNAME)\
      + '\n  4) %s -d epix10ka_000001                          # geometry from detector DB' % SCRNAME\
      + '\n  5) %s -d epix10ka_000001 -k exp=ueddaq02,run=569  # geometry from experiment DB' % SCRNAME\
      + '\n  6) %s -g %s  # takes geometry from file' % (SCRNAME, GEOFNAME)\
      + '\n  7) %s -a %s  # takes array for image from file' % (SCRNAME, NDAFNAME)\
      + '\n  8) %s -a %s -g %s  # ndarray and geometry from files' % (SCRNAME, NDAFNAME, GEOFNAME)\

#      + '\n  %s -k "{\'exp\':\'abcd01234\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\', ' % SCRNAME\
#      + '\'detectors\':[\'epicsinfo\', \'tmo_opal1\', \'ebeam\']}" -d tmo_opal1 -D'\
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
    d_savelog  = True

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
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_group   = 'group ownership for all files, default = %s' % d_group
    h_ctab    = 'color table index in range [1,8], default = %d' % d_ctab
    h_savelog = 'On/Off saving log file, default = %s' % d_savelog

    parser = ArgumentParser(usage=USAGE, description='%s - command launching Mask Editor GUI' % SCRNAME)
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
    parser.add_argument('--savelog',         default=d_savelog, action='store_false', help=h_savelog)

    return parser

def mask_editor():
    parser = argument_parser()
    namesp = parser.parse_args()
    posargs = namesp.posargs
    kwargs = vars(namesp)
    print('Command "%s" started with optional arguments:%s\nLaunch GUI' % (SCRNAME, mu.ut.info_parser_arguments(parser, title='')))
    #if len(posargs)>0 and kwargs['ndafname'] is None: kwargs['ndafname'] = posargs[0]
    #if len(posargs)>1 and kwargs['geofname'] is None: kwargs['geofname'] = posargs[1]

    import psana.graphqt.MEDMain as mm
    mm.mask_editor(**kwargs)

if __name__ == "__main__":
    mask_editor()
    sys.exit('END OF %s' % SCRNAME)

# EOF
