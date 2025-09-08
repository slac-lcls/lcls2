#!/usr/bin/env python

import sys
from psana.detector.dir_root import DIR_REPO_EPIX10KA
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]


USAGE = 'Usage:'\
      + '\n  %s -k <\"str-of-datasource-kwargs\"> -d <detector> ' % SCRNAME\
      + '\n     [-o <output-result-directory>] [-L <logging-mode>] [other-kwargs]'\
      + '\nExamples:'\
      + '\nTests:'\
      + '\n  %s -k exp=ascdaq123,run=4 -d epixm -o ./work # data on psana' % SCRNAME\
      + '\n  %s -k exp=tstx00417,run=317,dir=/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/ -d tst_epixm -o ./work # data on psana' % SCRNAME\
      + '\n  %s -k exp=tstx00417,run=324,dir=/drpneh/data/tst/tstx00417/xtc/ -d epixm -o ./work # data on drp-neh-cmp001' % SCRNAME\
      + '\n  %s -k exp=tstx00417,run=328,dir=/sdf/data/lcls/ds/rix/rixx1005922/scratch/xtc -d epixm -o ./work # on s3df IN WROND DIRECTORY' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def do_main():

    parser = argument_parser()
    args = parser.parse_args()
    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.det is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'

    from psana.detector.UtilsEpixCalib import pedestals_calibration
    from time import time
    t0_sec = time()
    #if args.det  == 'tst_epixm': sys.exit('TEST EXIT FOR %s' % args.det)
    pedestals_calibration(parser)
    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))
    sys.exit('End of %s' % SCRNAME)


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs = None
    d_det     = None # 'epixquad'
    d_nrecs   = 100  # number of records to collect and process
    d_nrecs1  = 50   # number of records to process at 1st stage
    d_idx     = None # 0-15 for epix10ka2m, 0-3 for epix10kaquad
    d_dirrepo = DIR_REPO_EPIX10KA
    d_logmode = 'INFO'
    d_errskip = True
    d_stepnum = None
    d_stepmax = 3
    d_evskip  = 0       # number of events to skip in the beginning of each step
    d_events  = 1000000 # last event number in the step to process
    d_dirmode = 0o2775
    d_filemode= 0o664
    d_group   = 'ps-users'
    d_int_lo  = 1       # lowest  intensity accepted for dark evaluation
    d_int_hi  = None    # highest intensity accepted for dark evaluation, ex: 16000
    d_intnlo  = 6.0     # intensity ditribution number-of-sigmas low
    d_intnhi  = 6.0     # intensity ditribution number-of-sigmas high
    d_rms_lo  = 0.001   # rms ditribution low
    d_rms_hi  = None    # rms ditribution high, ex: 16000
    d_rmsnlo  = 6.0     # rms ditribution number-of-sigmas low
    d_rmsnhi  = 6.0     # rms ditribution number-of-sigmas high
    d_fraclm  = 0.1     # allowed fraction limit
    d_fraclo  = 0.05    # fraction of statistics [0,1] below low limit
    d_frachi  = 0.95    # fraction of statistics [0,1] below high limit
    d_version = 'V2024-12-18'
    d_datbits = 0o77777
    d_deploy  = False
    d_plotim  = 0

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_nrecs   = 'number of records to calibrate pedestals, default = %s' % str(d_nrecs)
    h_det     = 'detector name, default = %s' % d_det
    h_nrecs1  = 'number of records to process at 1st stage, default = %s' % str(d_nrecs1)
    h_idx     = 'segment index (0-15 for epix10ka2m, 0-3 for quad) or all by default for processing, default = %s' % str(d_idx)
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_errskip = 'flag to skip errors and keep processing, stop otherwise, default = %s' % d_errskip
    h_stepnum = 'step number to process or None for all steps, default = %s' % str(d_stepnum)
    h_stepmax = 'maximum number of steps to process, default = %s' % str(d_stepmax)
    h_evskip  = 'number of events to skip in the beginning of each step, default = %s' % str(d_evskip)
    h_events  = 'number of events to process from the beginning of each step, default = %s' % str(d_events)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_int_lo  = 'lowest  intensity accepted for dark evaluation, default = %d' % d_int_lo
    h_int_hi  = 'highest intensity accepted for dark evaluation, for None derived from data_bit_mask, default = %s' % d_int_hi
    h_intnlo  = 'intensity ditribution number-of-sigmas low, default = %f' % d_intnlo
    h_intnhi  = 'intensity ditribution number-of-sigmas high, default = %f' % d_intnhi
    h_rms_lo  = 'rms ditribution low, default = %f' % d_rms_lo
    h_rms_hi  = 'rms ditribution high, for None derived from data_bit_mask, default = %s' % d_rms_hi
    h_rmsnlo  = 'rms ditribution number-of-sigmas low, default = %f' % d_rmsnlo
    h_rmsnhi  = 'rms ditribution number-of-sigmas high, default = %f' % d_rmsnhi
    h_fraclm  = 'fraction of statistics [0,1] below low or above high gate limit to assign pixel bad status, default = %f' % d_fraclm
    h_fraclo  = 'fraction of statistics [0,1] below low  limit of the gate, default = %f' % d_fraclo
    h_frachi  = 'fraction of statistics [0,1] above high limit of the gate, default = %f' % d_frachi
    h_version = 'constants version, default = %s' % str(d_version)
    h_datbits = 'data bits, e.g. 0x7fff is 15-bit mask for epixm320, default = %s' % hex(d_datbits)
    h_deploy  = 'deploy constants to the calibration DB, default = %s' % d_deploy
    h_plotim  = 'plot image/s of pedestals, default = %s' % str(d_plotim)

    parser = ArgumentParser(usage=USAGE, description='Proceses dark run xtc data for epix10ka')
    parser.add_argument('-k', '--dskwargs',default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--det',     default=d_det,        type=str,   help=h_det)
    parser.add_argument('-n', '--nrecs',   default=d_nrecs,      type=int,   help=h_nrecs)
    parser.add_argument('--nrecs1',        default=d_nrecs1,     type=int,   help=h_nrecs1)
    parser.add_argument('-i', '--idx',     default=d_idx,        type=int,   help=h_idx)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-L', '--logmode', default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-E', '--errskip', action='store_false',             help=h_errskip)
    parser.add_argument('--stepnum',       default=d_stepnum,    type=int,   help=h_stepnum)
    parser.add_argument('--stepmax',       default=d_stepmax,    type=int,   help=h_stepmax)
    parser.add_argument('--evskip',        default=d_evskip,     type=int,   help=h_evskip)
    parser.add_argument('--events',        default=d_events,     type=int,   help=h_events)
    parser.add_argument('--dirmode',       default=d_dirmode,    type=int,   help=h_dirmode)
    parser.add_argument('--filemode',      default=d_filemode,   type=int,   help=h_filemode)
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
    parser.add_argument('-v', '--version', default=d_version,    type=str,   help=h_version)
    parser.add_argument('--datbits',       default=d_datbits,    type=int,   help=h_datbits)
    parser.add_argument('-D', '--deploy',  action='store_true', help=h_deploy)
    parser.add_argument('-p', '--plotim',  default=d_plotim,     type=int,   help=h_plotim)

    return parser


if __name__ == "__main__":
    do_main()

# EOF
