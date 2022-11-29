#!/usr/bin/env python

import sys
from time import time
from psana.detector.Utils import info_parser_arguments
from psana.detector.UtilsCalib import pedestals_calibration
from psana.detector.dir_root import DIR_REPO_DARK_PROC
from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, init_stream_handler
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = 'Usage:'\
      + '\n  %s -k <datasource-kwargs> -d <detector> [-o <output-result-directory>] [-L <logging-mode>] [...]'\
      + '\nExamples:'\
      + '\n  %s -k exp=tmox49720,run=209 -d epix100 -D' % SCRNAME\
      + '\n  %s -k exp=tmoc00318,run=10,dir=/a/b/c/xtc -d epix100 -D' % SCRNAME\
      + '\n  %s -k "{\'exp\':\'abcd01234\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\', ' % SCRNAME\
      + '\'detectors\':[\'epicsinfo\', \'tmo_opal1\', \'ebeam\']}" -d tmo_opal1 -D'\
      + '\n\nTest:'\
      + '\n  %s -k "{\'exp\':\'tmoc00118\', \'run\':123}" -d tmoopal -o ./work' % SCRNAME\
      + '\n  %s -k exp=tmoc00118,run=123 -d tmoopal -o ./work' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/tmoc00318-r0010-s000-c000.xtc2 -d epix100 -o ./work' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/tmoc00118-r0222-s006-c000.xtc2 -d tmo_atmopal -o ./work' % SCRNAME\
      + '\n  %s -k /cds/data/psdm/prj/public01/xtc/rixl1013320-r0093-s006-c000.xtc2 -d atmopal -o ./work' % SCRNAME\
      + '\n\nHelp:\n  %s -h' % SCRNAME


def do_main():

    t0_sec = time()

    parser = argument_parser()
    args = parser.parse_args()
    kwa = vars(args)
    #defs = vars(parser.parse_args([])) # dict of defaults only

    if len(sys.argv)<3: exit('\n%s\n' % USAGE)

    assert args.dsname is not None, 'WARNING: option "-k <datasource-name>" MUST be specified.'
    assert args.det is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'

    init_stream_handler(loglevel=args.logmode)

    logger.debug('%s\nIn %s' % ((50*'_'), SCRNAME))
    logger.debug('Command line:%s' % ' '.join(sys.argv))
    logger.info(info_parser_arguments(parser))

    pedestals_calibration(**kwa)

    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))



def argument_parser():
    from argparse import ArgumentParser

    d_dsname  = None    # 'files=<fname.xtc>,exp=<expname>,run=<runs>,dir=<xtc-dir>, ...'
    d_det     = None    # 'tmoopal'
    d_nrecs   = 1000    # number of records to collect and process
    d_nrecs1  = 100     # number of records to process at 1st stage
    d_dirrepo = DIR_REPO_DARK_PROC  # '<DIR_ROOT>/detector/calib2'
    d_logmode = 'INFO'
    d_errskip = True
    d_stepnum = None
    d_stepmax = 1
    d_evskip  = 0       # number of events to skip in the beginning of each step
    d_events  = 1000    # number of events to process from the beginning of each step
    d_datbits = 0x3fff
    d_dirmode = 0o2775
    d_filemode= 0o664
    d_group   = 'ps-users'
    d_int_lo  = 1       # lowest  intensity accepted for dark evaluation
    d_int_hi  = 16000   # highest intensity accepted for dark evaluation
    d_intnlo  = 6.0     # intensity ditribution number-of-sigmas low
    d_intnhi  = 6.0     # intensity ditribution number-of-sigmas high
    d_rms_lo  = 0.001   # rms ditribution low
    d_rms_hi  = 16000   # rms ditribution high
    d_rmsnlo  = 6.0     # rms ditribution number-of-sigmas low
    d_rmsnhi  = 6.0     # rms ditribution number-of-sigmas high
    d_fraclm  = 0.1     # allowed fraction limit
    d_fraclo  = 0.05    # fraction of statistics [0,1] below low limit
    d_frachi  = 0.95    # fraction of statistics [0,1] below high limit
    d_deploy  = False
    d_tstamp  = None    # 20180910111049 or run number <10000
    d_version = 'V2022-11-22'
    d_run_end = 'end'
    d_comment = 'no comment'
    d_plotim  = 0

    h_dsname  = 'str of comma-separated (no spaces) simple parameters for DataSource(**kwargs), ex: file=<fname.xtc>,exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                'or pythonic dict of generic kwargs, e.g.: \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dsname
    h_det     = 'detector name, default = %s' % d_det
    h_nrecs   = 'number of records to calibrate pedestals, default = %s' % str(d_nrecs)
    h_nrecs1  = 'number of records to process at 1st stage, default = %s' % str(d_nrecs1)
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (' '.join(DICT_NAME_TO_LEVEL.keys()), d_logmode)
    h_errskip = 'flag to skip errors and keep processing, stop otherwise, default = %s' % d_errskip
    h_stepnum = 'step number to process or None for all steps, default = %s' % str(d_stepnum)
    h_stepmax = 'maximum number of steps to process, default = %s' % str(d_stepmax)
    h_evskip  = 'number of events to skip in the beginning of each step, default = %s' % str(d_evskip)
    h_events  = 'number of events to process from the beginning of each step, default = %s' % str(d_events)
    h_datbits = 'data bits, e.g. 0x3fff is 14-bit mask for epix10ka and Jungfrau, default = %s' % hex(d_datbits)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_group   = 'group ownership for all files, default = %s' % d_group
    h_int_lo  = 'lowest  intensity accepted for dark evaluation, default = %d' % d_int_lo
    h_int_hi  = 'highest intensity accepted for dark evaluation, default = %d' % d_int_hi
    h_intnlo  = 'intensity ditribution number-of-sigmas low, default = %f' % d_intnlo
    h_intnhi  = 'intensity ditribution number-of-sigmas high, default = %f' % d_intnhi
    h_rms_lo  = 'rms ditribution low, default = %f' % d_rms_lo
    h_rms_hi  = 'rms ditribution high, default = %f' % d_rms_hi
    h_rmsnlo  = 'rms ditribution number-of-sigmas low, default = %f' % d_rmsnlo
    h_rmsnhi  = 'rms ditribution number-of-sigmas high, default = %f' % d_rmsnhi
    h_fraclm  = 'fraction of statistics [0,1] below low or above high gate limit to assign pixel bad status, default = %f' % d_fraclm
    h_fraclo  = 'fraction of statistics [0,1] below low  limit of the gate, default = %f' % d_fraclo
    h_frachi  = 'fraction of statistics [0,1] above high limit of the gate, default = %f' % d_frachi
    h_deploy  = 'deploy constants to the calibration DB, default = %s' % d_deploy
    h_tstamp  = 'non-default time stamp in format YYYYmmddHHMMSS or run number(<10000) for constants selection in repo. '\
                'By default run time is used, default = %s' % str(d_tstamp)
    h_version = 'constants version, default = %s' % str(d_version)
    h_run_end = 'last run for validity range, default = %s' % str(d_run_end)
    h_comment = 'comment added to constants metadata, default = %s' % str(d_comment)
    h_plotim  = 'plot image/s of pedestals, default = %s' % str(d_plotim)

    parser = ArgumentParser(usage=USAGE, description='%s - proceses dark run xtc raw data fro specified detector' % SCRNAME)
    parser.add_argument('-k', '--dsname',  default=d_dsname,     type=str,   help=h_dsname)
    parser.add_argument('-d', '--det',     default=d_det,        type=str,   help=h_det)
    parser.add_argument('-n', '--nrecs',   default=d_nrecs,      type=int,   help=h_nrecs)
    parser.add_argument('--nrecs1',        default=d_nrecs1,     type=int,   help=h_nrecs1)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-L', '--logmode', default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-E', '--errskip', action='store_false',             help=h_errskip)
    parser.add_argument('--stepnum',       default=d_stepnum,    type=int,   help=h_stepnum)
    parser.add_argument('--stepmax',       default=d_stepmax,    type=int,   help=h_stepmax)
    parser.add_argument('--evskip',        default=d_evskip,     type=int,   help=h_evskip)
    parser.add_argument('--events',        default=d_events,     type=int,   help=h_events)
    parser.add_argument('--datbits',       default=d_datbits,    type=int,   help=h_datbits)
    parser.add_argument('--dirmode',       default=d_dirmode,    type=int,   help=h_dirmode)
    parser.add_argument('--filemode',      default=d_filemode,   type=int,   help=h_filemode)
    parser.add_argument('--group',         default=d_group,      type=str,   help=h_group)
    parser.add_argument('--int_lo',        default=d_int_lo,     type=int,   help=h_int_lo)
    parser.add_argument('--int_hi',        default=d_int_hi,     type=int,   help=h_int_hi)
    parser.add_argument('--intnlo',        default=d_intnlo,     type=float, help=h_intnlo)
    parser.add_argument('--intnhi',        default=d_intnhi,     type=float, help=h_intnhi)
    parser.add_argument('--rms_lo',        default=d_rms_lo,     type=float, help=h_rms_lo)
    parser.add_argument('--rms_hi',        default=d_rms_hi,     type=float, help=h_rms_hi)
    parser.add_argument('--rmsnlo',        default=d_rmsnlo,     type=float, help=h_rmsnlo)
    parser.add_argument('--rmsnhi',        default=d_rmsnhi,     type=float, help=h_rmsnhi)
    parser.add_argument('--fraclm',        default=d_fraclm,     type=float, help=h_fraclm)
    parser.add_argument('--fraclo',        default=d_fraclo,     type=float, help=h_fraclo)
    parser.add_argument('--frachi',        default=d_frachi,     type=float, help=h_frachi)
    parser.add_argument('-D', '--deploy',  action='store_true',              help=h_deploy)
    parser.add_argument('-t', '--tstamp',  default=d_tstamp,     type=int,   help=h_tstamp)
    parser.add_argument('-v', '--version', default=d_version,    type=str,   help=h_version)
    parser.add_argument('-R', '--run_end', default=d_run_end,    type=str,   help=h_run_end)
    parser.add_argument('-C', '--comment', default=d_comment,    type=str,   help=h_comment)
    parser.add_argument('-p', '--plotim',  default=d_plotim,     type=int,   help=h_plotim)

    return parser


if __name__ == "__main__":
    do_main()
    exit('End of %s'%SCRNAME)

# EOF
