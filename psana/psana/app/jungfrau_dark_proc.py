#!/usr/bin/env python

import sys
from psana.detector.dir_root import DIR_REPO_JUNGFRAU
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

#M14 = 0o37777 # 14-bits of data, 2 bits for gain mode switch
M14 = 0x3fff # 16383, 14-bit mask

USAGE = 'Usage:'\
      + '\n  %s -k <\"str-of-datasource-kwargs\"> -d <detector> ' % SCRNAME\
      + '\n     [-o <output-result-directory>] [-L <logging-mode>] [other-kwargs]'\
      + '\nExamples:'\
      + '\nTests:'\
      + '\n  datinfo -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau # test data'\
      + '\n  %s -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau -o ./work # data' % SCRNAME\
      + '\n  %s -k exp=mfxdaq23,run=7 -d jungfrau -o ./work # data' % SCRNAME\
      + '\n  %s -k exp=ascdaq023,run=37 -d jungfrau -o ./work # data' % SCRNAME\
      + '\n  %s -k exp=mfx100861624,run=30 -d jungfrau -o work --stepnum 0 --stepmax 1 --segind 7' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs= 'exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc' # None
    d_detname = 'jungfrau' #  None
    d_nrecs   = 1000  # number of records to collect and process
    d_nrecs1  = 50    # number of records to process at 1st stage
    #d_idx     = None  # 0-15 for epix10ka2m, 0-3 for epix10kaquad
    d_dirrepo = DIR_REPO_JUNGFRAU # './work'
    d_logmode = 'INFO'
    d_errskip = True
    d_stepnum = None
    d_stepmax = 3
    d_evskip  = 0       # number of events to skip in the beginning of each step
    d_events  = 1000000 # last event number in the step to process
    d_evstep  = 1000000
    d_dirmode = 0o2775
    d_filemode= 0o664
    d_group   = 'ps-users'
    d_int_lo  = 1       # lowest  intensity accepted for dark evaluation
    d_int_hi  = M14-3   # highest intensity accepted for dark evaluation, ex: 16000
    d_intnlo  = 6.0     # intensity ditribution number-of-sigmas low
    d_intnhi  = 6.0     # intensity ditribution number-of-sigmas high
    d_rms_lo  = 0.001   # rms distribution low
    d_rms_hi  = M14-3   # rms distribution high, ex: 16000
    d_rmsnlo  = 6.0     # rms distribution number-of-sigmas low
    d_rmsnhi  = 6.0     # rms distribution number-of-sigmas high
    d_fraclm  = 0.1     # allowed fraction limit
    d_fraclo  = 0.05    # fraction of statistics [0,1] below low limit
    d_frachi  = 0.95    # fraction of statistics [0,1] below high limit
    d_version = 'V2025-06-07'
    d_datbits = M14     # 14-bits, 2 bits for gain mode switch
    d_deploy  = False
    d_plotim  = 0
    d_evcode  = None
    d_segind  = None
    d_igmode  = None
    d_mpi     = False

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_nrecs   = 'number of records to calibrate pedestals, default = %s' % str(d_nrecs)
    h_detname = 'detector name, default = %s' % d_detname
    h_nrecs1  = 'number of records to process at 1st stage, default = %s' % str(d_nrecs1)
    #h_idx     = 'segment index (0-31 for jungfrau) or all by default for processing, default = %s' % str(d_idx)
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_errskip = 'flag to skip errors and keep processing, stop otherwise, default = %s' % d_errskip
    h_stepnum = 'step number to process or None for all steps, default = %s' % str(d_stepnum)
    h_stepmax = 'maximum number of steps to process, default = %s' % str(d_stepmax)
    h_evskip  = 'number of events to skip in the beginning of each step, default = %s' % str(d_evskip)
    h_events  = 'number of events to process from the beginning of each step, default = %s' % str(d_events)
    h_evstep  = 'maximal number of events to process in each step, default = %s' % d_evstep
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
    h_evcode  = 'comma separated event codes for selection as OR combination, any negative %s'%\
                'code inverts selection, default = %s'%str(d_evcode)
    h_segind  = 'segment index in det.raw.raw array to process, default = %s' % str(d_segind)
    h_igmode  = 'gainmode index FOR DEBUGGING, default = %s' % str(d_igmode)
    h_mpi     = 'use with MPI, default = %s' % d_mpi

    parser = ArgumentParser(usage=USAGE, description='Proceses dark run xtc data for epix10ka')
    parser.add_argument('-k', '--dskwargs',default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname', default=d_detname,    type=str,   help=h_detname)
    parser.add_argument('-n', '--nrecs',   default=d_nrecs,      type=int,   help=h_nrecs)
    parser.add_argument('--nrecs1',        default=d_nrecs1,     type=int,   help=h_nrecs1)
    #parser.add_argument('-i', '--idx',     default=d_idx,        type=int,   help=h_idx)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-L', '--logmode', default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-E', '--errskip', action='store_false',             help=h_errskip)
    parser.add_argument('--stepnum',       default=d_stepnum,    type=int,   help=h_stepnum)
    parser.add_argument('--stepmax',       default=d_stepmax,    type=int,   help=h_stepmax)
    parser.add_argument('--evskip',        default=d_evskip,     type=int,   help=h_evskip)
    parser.add_argument('--events',        default=d_events,     type=int,   help=h_events)
    parser.add_argument('-e', '--evstep',  default=d_evstep,     type=int,   help=h_evstep)
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
    parser.add_argument('-D', '--deploy',  action='store_true',              help=h_deploy)
    parser.add_argument('-p', '--plotim',  default=d_plotim,     type=int,   help=h_plotim)
    parser.add_argument('-c', '--evcode',  default=d_evcode,     type=str,   help=h_evcode)
    parser.add_argument('-I', '--segind',  default=d_segind,     type=int,   help=h_segind)
    parser.add_argument('-G', '--igmode',  default=d_igmode,     type=int,   help=h_igmode)
    parser.add_argument('-M', '--mpi',     action='store_true',              help=h_mpi)
    return parser


def do_main():
    from time import time

    parser = argument_parser()
    args = parser.parse_args()
    kwa = vars(args)

    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.detname  is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'
    assert args.stepnum  is not None, 'WARNING: option "--stepnum <stepnum>" MUST be specified.'

    print('use code for MPI: %s' % args.mpi)

    t0_sec = time()
    if args.mpi: from psana.detector.UtilsJungfrauCalibMPI import jungfrau_dark_proc
    else:        from psana.detector.UtilsJungfrauCalib    import jungfrau_dark_proc

    jungfrau_dark_proc(parser)
    logger.info('End of %s, consumed time %.3f sec' % (SCRNAME, time() - t0_sec))
    sys.exit(0)


if __name__ == "__main__":
    do_main()

# EOF
