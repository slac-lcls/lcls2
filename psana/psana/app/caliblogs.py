#!/usr/bin/env python

import os
import sys
from psana.detector.Utils import str_tstamp

DIR_ATSTART = '/sdf/group/lcls/ds/ana/detector/logs/atstart'
DIR_CONSTANTS = '/sdf/group/lcls/ds/ana/detector/calib2/constants'
SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = 'Usage:'\
      + '\n%s [-d <detector-name>] [-y <year>] [-L <logging-mode>] [...]' % SCRNAME\
      + '\nTEST COMMAND:'\
      + '\n  %s -d jungfrau' % SCRNAME\
      + '\n  %s ' % SCRNAME\
      + '\nREGULAR COMMAND:'\
      + '\n  %s -d jungfrau' % SCRNAME

def argument_parser():
    from argparse import ArgumentParser

#    d_dskwargs= None
    d_detname = None # 'jungfrau'
    d_year = str_tstamp(fmt='%Y', time_sec=None) # 2025
    d_diratstart = DIR_ATSTART
    d_dirconstants = DIR_CONSTANTS
    d_show_files = False
    d_deploy  = False
    d_logmode = 'INFO'
    d_version = 'V2025-07-15'
    d_comment = 'no comment'

#    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
#                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
#                ' or <fname.xtc> or files=<fname.xtc>'\
#                ' or pythonic dict of generic kwargs, e.g.:'\
#                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_detname = 'detector name, default = %s' % d_detname
    h_year = 'year, default = %s' % d_year
    h_diratstart = 'non-default repository of atstart logfiles, default = %s' % d_diratstart
    h_dirconstants = 'non-default repository of constants logfiles, default = %s' % d_dirconstants
    h_show_files = 'show files in the directory, default = %s' % d_show_files
    h_logmode = 'logging mode, default = %s' % d_logmode
    h_version = 'constants version, default = %s' % str(d_version)
    h_comment = 'comment added to constants metadata, default = %s' % str(d_comment)

    parser = ArgumentParser(description='Deploy calibration files from repository to DB.', usage = USAGE)
#    parser.add_argument('-k', '--dskwargs',default=d_dskwargs, type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname',      default=d_detname,      type=str,   help=h_detname)
    parser.add_argument('-y', '--year',         default=d_year,         type=str,   help=h_year)
    parser.add_argument('-s', '--diratstart',   default=d_diratstart,   type=str,   help=h_diratstart)
    parser.add_argument('-c', '--dirconstants', default=d_dirconstants, type=str,   help=h_dirconstants)
    parser.add_argument('-L', '--logmode',      default=d_logmode,      type=str,   help=h_logmode)
    parser.add_argument('-v', '--version',      default=d_version,      type=str,   help=h_version)
    parser.add_argument('-C', '--comment',      default=d_comment,      type=str,   help=h_comment)
    parser.add_argument('-S', '--show_files',   action='store_true',                help=h_show_files)
#    parser.add_argument('-D', '--deploy',  action='store_true', help=h_deploy)

    return parser


def info_xtc2dirs(sep='\n  '):
    return sep + sep.join(('/sdf/data/lcls/ds/', '/sdf/data/lcls/drpsrcf/ffb/'))

def info_logfiles(topdir, detname, year, show_files=False):
    gap = '    '
    spc = ('\n'+2*gap)
    lst_dirs = [os.path.join(topdir,fname) for fname in os.listdir(topdir)\
                                   if (True if detname is None else detname in fname)]
    lst_dirs_ext = sorted([os.path.join(fname, 'logs', year) for fname in lst_dirs])
    s = ''
    for d in lst_dirs_ext:
        if os.path.exists(d):
            s += '\n%s%s/' % (gap, d)
            lst_files = [fn for fn in os.listdir(d)]
            s += ' - %d files' % len(lst_files)
            if show_files: s += spc + spc.join([fn for fn in sorted(lst_files)])
        else: continue # s += ' - does not exist'
    return s


def do_main():

    parser = argument_parser()
    args = parser.parse_args()

    print('calibration logfiles for: %s' % args.year)

    path_atstart = os.path.join(args.diratstart, args.year)
    lst_files = [os.path.join(path_atstart,fname) for fname in os.listdir(path_atstart)\
                 if (True if args.detname is None else args.detname in fname)]
    s = '\n    '.join(sorted(lst_files))
    print('\nfiles with "at start" records in %s\n    %s' % (path_atstart, s))

    s = info_logfiles(args.dirconstants, args.detname, args.year, args.show_files)
    print('\ndirs for detectors in %s   %s' %  (args.dirconstants, s))

    dirs_scripts = os.path.join(args.dirconstants, 'scripts')
    s = info_logfiles(dirs_scripts, args.detname, args.year, args.show_files)
    print('\ndirs for scripts in %s    %s' % (dirs_scripts, s))
    print('\ninstrument directories:%s' % info_xtc2dirs())
    print('\ncalib constants repository:\n  /sdf/group/lcls/ds/ana/detector/calib2/constants\n')

if __name__ == "__main__":
    do_main()
    sys.exit('End of %s'%SCRNAME)

# EOF
