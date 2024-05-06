#!/usr/bin/env python

import sys
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
from psana.detector.dir_root import DIR_REPO_EPIX10KA
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].split('/')[-1]

def do_main():

    parser = argument_parser()
    args = parser.parse_args()
    #kwa = vars(args)

    if len(sys.argv) < 4: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)

    if args.dskwargs is None : raise IOError('WARNING: option "-k <kwargs-for-DataSource>" MUST be specified.')
    if args.det      is None : raise IOError('WARNING: option "-d <detector-name>" MUST be specified.')
    #if args.idx      is None : raise IOError('WARNING: option "-i <panel index (0-3/15 for quad/epix10ka2m etc.)>" MUST be specified.')

    from time import time
    from psana.detector.UtilsEpixm320ChargeInjection import charge_injection
    t0_sec = time()
    charge_injection(parser)
    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))


USAGE = 'TEST EXAMPLE'\
      + '\n  datinfo -k exp=tstx00417,run=325,dir=/drpneh/data/tst/tstx00417/xtc -d epixm'\
      + '\n  %s -k exp=tstx00417,run=324,dir=/drpneh/data/tst/tstx00417/xtc/ -d epixm -o ./work # on s3df' % SCRNAME\
      + '\n  %s -k exp=tstx00417,run=332,dir=/sdf/data/lcls/ds/rix/rixx1005922/scratch/xtc -d epixm -o ./work -p1 # on s3df IN WROND DIRECTORY' % SCRNAME\


def argument_parser() :
    d_dskwargs = None  # exp=ascdaq18,run=171
    d_det      = None  # 'NoDetector.0:Epix10ka.3'
    d_idx      = None  # 0-15 for epix10ka2m, 0-3 for epix10kaquad
    d_nrecs    = 1000  # number of frames
    d_dirrepo  = DIR_REPO_EPIX10KA # './work'
    d_logmode  = 'INFO'
    d_stepnum  = None
    d_stepmax  = 230
    d_stepskip = 0
    d_evskip   = 0
    d_evstep   = 1
    d_nsigm    = 8
    d_pixrc    = None
    d_dirmode  = 0o2775
    d_filemode = 0o664
    d_group    = 'ps-users'
    d_slice    = '0:,0:'
    d_version  = 'V2024-04-23'
    d_deploy   = False
    d_tstamp   = None    # 20180910111049 or run number <10000
    d_run_end  = 'end'
    d_comment  = 'no comment'
    d_plotim   = 0

    h_dskwargs = 'Data source parameters; string of comma-separated (no spaces) parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <file-name.xtc> or files=<file-name.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_det      = 'detector name, default = %s' % d_det
    h_idx      = 'panel index in the detector (0-15/3 for epix10ka2m/quad), default = %s' % str(d_idx)
    h_nrecs    = 'number of records to collect, default = %s' % str(d_nrecs)
    h_dirrepo  = 'repository for results, default = %s' % d_dirrepo
    h_logmode  = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_stepnum  = 'step number to process (from 0 to %d) for debugging or all by default, default = %s' % ((d_stepmax-1), str(d_stepnum))
    h_stepmax  = 'maximal number of steps to process(for debugging), default = %s' % str(d_stepmax)
    h_stepskip = 'skip number steps in the beginning of the data file (for debugging), default = %s' % str(d_stepskip)
    h_evskip   = 'number of eveents to skip in the beginning of each step, default = %s' % str(d_evskip)
    h_evstep   = 'number of eveents per step, default = %s' % str(d_evstep)
    h_nsigm    = 'number of sigma/spread to discard outlaiers for pixel_status, default = %s' % d_nsigm
    h_pixrc    = 'selected pixel for graphics: comma separated pixel row and colon, ex. 23,234, default = %s' % d_pixrc
    h_dirmode  = 'directory mode for all created directories, default = %s' % oct(d_dirmode)
    h_filemode = 'file mode for all saved files, default = %s' % oct(d_filemode)
    h_group    = 'group ownership for all saved files, default = %s' % d_group
    h_slice    = '(str) slice of the panel image 2-d array selected for plots and pixel status, FOR DEBUGGING ONLY, ex. "0:144,0:192", default = %s' % d_slice
    h_version  = 'constants version, default = %s' % str(d_version)

    h_deploy   = 'deploy constants to the calibration DB, default = %s' % d_deploy
    h_tstamp   = 'non-default time stamp in format YYYYmmddHHMMSS or run number(<10000) for constants selection in repo. '\
                'By default run time is used, default = %s' % str(d_tstamp)
    h_run_end  = 'last run for validity range, default = %s' % str(d_run_end)
    h_comment  = 'comment added to constants metadata, default = %s' % str(d_comment)
    h_plotim   = 'plot image/s of pedestals, default = %s' % str(d_plotim)

    from argparse import ArgumentParser
    parser = ArgumentParser(usage=USAGE, description='Process charge injection data for epixm320')
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str,             help=h_dskwargs)
    parser.add_argument('-d', '--det',      default=d_det,      type=str,             help=h_det)
    parser.add_argument('-b', '--nrecs',    default=d_nrecs,    type=int,             help=h_nrecs)
    parser.add_argument('-o', '--dirrepo',  default=d_dirrepo,  type=str,             help=h_dirrepo)
    parser.add_argument('-L', '--logmode',  default=d_logmode,  type=str,             help=h_logmode)
    parser.add_argument('-c', '--stepnum',  default=d_stepnum,  type=int,             help=h_stepnum)
    parser.add_argument('-m', '--stepmax',  default=d_stepmax,  type=int,             help=h_stepmax)
    parser.add_argument('-s', '--stepskip', default=d_stepskip, type=int,             help=h_stepskip)
    parser.add_argument('-M', '--evstep',   default=d_evstep,   type=int,             help=h_evstep)
    parser.add_argument(      '--evskip',   default=d_evskip,   type=int,             help=h_evskip)
    parser.add_argument('-i', '--idx',      default=d_idx,      type=int,             help=h_idx)
    parser.add_argument('-P', '--pixrc',    default=d_pixrc,    type=str,             help=h_pixrc)
    parser.add_argument('-S', '--nsigm',    default=d_nsigm,    type=float,           help=h_nsigm)
    parser.add_argument('--dirmode',        default=d_dirmode,  type=int,             help=h_dirmode)
    parser.add_argument('--filemode',       default=d_filemode, type=int,             help=h_filemode)
    parser.add_argument('--group',          default=d_group,    type=str,             help=h_group)
    parser.add_argument('--slice',          default=d_slice,    type=str,             help=h_slice)
    parser.add_argument('-t', '--tstamp',   default=d_tstamp,   type=int,             help=h_tstamp)
    parser.add_argument('-R', '--run_end',  default=d_run_end,  type=str,             help=h_run_end)
    parser.add_argument('-C', '--comment',  default=d_comment,  type=str,             help=h_comment)
    parser.add_argument('-v', '--version',  default=d_version,  type=str,             help=h_version)
    parser.add_argument('-p', '--plotim',   default=d_plotim,   type=int,             help=h_plotim)
    parser.add_argument('-D', '--deploy',   action='store_true',                      help=h_deploy)

    return parser


if __name__ == "__main__":
    do_main()
    sys.exit('End of %s'%SCRNAME)

# EOF
