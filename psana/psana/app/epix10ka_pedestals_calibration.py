#!/usr/bin/env python
#----
import os
import sys
from time import time

from psana.detector.Utils import info_command_line_arguments
from psana.detector.UtilsEpix10kaCalib import pedestals_calibration
from psana.detector.UtilsEpix10ka import GAIN_MODES_IN
from psana.detector.UtilsEpix import CALIB_REPO_EPIX10KA

import logging
logger = logging.getLogger(__name__)
DICT_NAME_TO_LEVEL = logging._nameToLevel #{'CRITICAL': 50, 'FATAL': 50, 'ERROR': 40, 'WARN': 30, 'WARNING': 30, 'INFO': 20, 'DEBUG': 10, 'NOTSET': 0}
STR_LEVEL_NAMES = ', '.join(DICT_NAME_TO_LEVEL.keys())

#scrname = sys.argv[0].rsplit('/')[-1]
scrname = os.path.basename(sys.argv[0])

#----

def do_main():

    t0_sec = time()

    parser = argument_parser()
    args = parser.parse_args()
    opts = vars(args)
    defs = vars(parser.parse_args([])) # dict of defaults only

    if len(sys.argv)<3: exit('\n%s\n' % usage())

    assert args.exp is not None, 'WARNING: option "-e <experiment>" MUST be specified.'
    assert args.det is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'
    assert args.runs is not None, 'WARNING: option "-r <run-number(s)>" MUST be specified.'

    #logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    fmt = '[%(levelname).1s] %(name)s %(message)s' if args.logmode=='DEBUG' else '[%(levelname).1s] %(message)s'
    #logging.basicConfig(filename='log.txt', filemode='w', format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])
    logging.basicConfig(format=fmt, level=DICT_NAME_TO_LEVEL[args.logmode])

    #fh = logging.FileHandler('log.txt')
    #fh.setLevel(logging.DEBUG)
    #logger.addHandler(fh)

    logger.debug('%s\nIn epix10ka_pedestals_calibration' % (50*'_'))
    logger.debug(info_command_line_arguments(parser))

    #pedestals_calibration(*args, **opts)
    pedestals_calibration(**opts)

    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))


def usage(mode=0):
    if   mode == 1: return 'Proceses dark run xtc data for epix10ka.'
    elif mode == 2: return 'Try: %s -h' % scrname
    else: return\
           '\n%s -e <experiment> [-d <detector>] [-r <run-number(s)>]' % scrname\
           + '\n     [-x <xtc-directory>] [-o <output-result-directory>] [-L <logging-mode>]'\
           + '\nTEST COMMAND:'\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -n2 -x /cds/data/psdm/ued/ueddaq02/xtc' % scrname\
           + '\nREGULAR COMMAND:'\
           + '\n  %s -e ueddaq02 -d epixquad -r83  -L DEBUG' % scrname\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -o ./work' % scrname\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -c1 -i15 -o ./work' % scrname\
           + '\n  mpirun -n 5 epix10ka_pedestals_calibration -e ueddaq02 -d epixquad -r27 -o ./work -L INFO'\
           + '\n\n  Try: %s -h' % scrname


def argument_parser():
    from argparse import ArgumentParser

    d_fname   = None # '/cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2'
    d_exp     = None # 'ueddaq02'
    d_det     = None # 'epixquad'
    d_runs    = None # 1021 or 1021,1022-1025
    d_nbs     = 1024 # number of frames 
    d_idx     = None # 0-15 for epix10ka2m, 0-3 for epix10kaquad
    d_dirxtc  = None # '/cds/data/psdm/ued/ueddaq02/xtc'
    d_dirrepo = CALIB_REPO_EPIX10KA # './myrepo' 
    d_usesmd  = True
    d_logmode = 'INFO'
    d_errskip = True
    d_stepnum    = None
    d_stepmax    = 5
    d_dirmode    = 0o777
    d_filemode   = 0o666
    d_int_lo     = 1       # lowest  intensity accepted for dark evaluation
    d_int_hi     = 16000   # highest intensity accepted for dark evaluation
    d_intnlo     = 6.0     # intensity ditribution number-of-sigmas low
    d_intnhi     = 6.0     # intensity ditribution number-of-sigmas high
    d_rms_lo     = 0.001   # rms ditribution low
    d_rms_hi     = 16000   # rms ditribution high
    d_rmsnlo     = 6.0     # rms ditribution number-of-sigmas low
    d_rmsnhi     = 6.0     # rms ditribution number-of-sigmas high
    d_fraclm     = 0.1     # allowed fraction limit
    d_nsigma     = 6.0     # number of sigmas for gated eveluation
    #d_fmt_peds   = '%.3f'
    #d_fmt_rms    = '%.3f'
    #d_fmt_status = '%4i'

    h_fname   = 'input xtc file name, default = %s' % d_fname
    h_exp     = 'experiment name, default = %s' % d_exp
    h_det     = 'detector name, default = %s' % d_det
    h_runs    = 'run number or list of runs e.g. 12,14-18, default = %s' % str(d_runs)
    h_nbs     = 'number of frames to calibrate pedestals, default = %s' % str(d_nbs)
    h_idx     = 'segment index (0-15 for epix10ka2m, 0-3 for quad) or all by default for processing, default = %s' % str(d_idx)
    h_dirxtc  = 'non-default xtc directory, default = %s' % d_dirxtc
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_usesmd  = 'add "smd" in dataset string, default = %s' % d_usesmd
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_errskip = 'flag to skip errors and keep processing, stop otherwise, default = %s' % d_errskip
    h_stepnum    = 'step number to process or None for all steps, default = %s' % str(d_stepnum)
    h_stepmax    = 'maximum number of steps to process, default = %s' % str(d_stepmax)
    h_dirmode    = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode   = 'file access mode, default = %s' % oct(d_filemode)
    h_int_lo     = 'lowest  intensity accepted for dark evaluation, default = %d' % d_int_lo
    h_int_hi     = 'highest intensity accepted for dark evaluation, default = %d' % d_int_hi
    h_intnlo     = 'intensity ditribution number-of-sigmas low, default = %f' % d_intnlo
    h_intnhi     = 'intensity ditribution number-of-sigmas high, default = %f' % d_intnhi
    h_rms_lo     = 'rms ditribution low, default = %f' % d_rms_lo
    h_rms_hi     = 'rms ditribution high, default = %f' % d_rms_hi
    h_rmsnlo     = 'rms ditribution number-of-sigmas low, default = %f' % d_rmsnlo
    h_rmsnhi     = 'rms ditribution number-of-sigmas high, default = %f' % d_rmsnhi
    h_fraclm     = 'allowed fraction limit, default = %f' % d_fraclm
    h_nsigma     = 'number of sigmas for gated eveluation, default = %f' % d_nsigma
    #h_fmt_peds   = 'format of values in pedestals file, default = %s' % str(d_fmt_peds)
    #h_fmt_rms    = 'format of values in pixel_rms file, default = %s' % str(d_fmt_rms)
    #h_fmt_status = 'format of values in pixel_status file, default = %s' % str(d_fmt_status)

    parser = ArgumentParser(description=usage(1))
    parser.add_argument('-f', '--fname',   default=d_fname,      type=str,   help=h_fname)
    parser.add_argument('-e', '--exp',     default=d_exp,        type=str,   help=h_exp)
    parser.add_argument('-d', '--det',     default=d_det,        type=str,   help=h_det)
    parser.add_argument('-r', '--runs',    default=d_runs,       type=str,   help=h_runs)
    parser.add_argument('-b', '--nbs',     default=d_nbs,        type=int,   help=h_nbs)
    parser.add_argument('-i', '--idx',     default=d_idx,        type=int,   help=h_idx)
    parser.add_argument('-x', '--dirxtc',  default=d_dirxtc,     type=str,   help=h_dirxtc)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-S', '--usesmd',  default=d_usesmd,     type=bool,  help=h_usesmd)
    parser.add_argument('-L', '--logmode', default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-E', '--errskip', default=d_errskip,    type=bool,  help=h_errskip)
    parser.add_argument('--stepnum',       default=d_stepnum,    type=int,   help=h_stepnum)
    parser.add_argument('--stepmax',       default=d_stepmax,    type=int,   help=h_stepmax)
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
    parser.add_argument('--nsigma',        default=d_nsigma,     type=float, help=h_nsigma)
    # DO NOT WORK...
    #parser.add_argument('--fmt_peds',      default=d_fmt_peds,   type=str,   help=h_fmt_peds)
    #parser.add_argument('--fmt_rms',       default=d_fmt_rms,    type=str,   help=h_fmt_rms)
    #parser.add_argument('--fmt_status',    default=d_fmt_status, type=str,   help=h_fmt_status)

    return parser

#----

if __name__ == "__main__":
    do_main()
    exit('End of %s'%scrname)

# EOF
