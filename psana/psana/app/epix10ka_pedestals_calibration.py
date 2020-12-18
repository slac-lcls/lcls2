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
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.logmode])
    logging.basicConfig(format='[%(levelname).1s] %(name)s %(message)s', level=DICT_NAME_TO_LEVEL[args.logmode])

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
           + '\n  %s -e ueddaq02 -d epixquad -r27  -L INFO' % scrname\
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
    d_ccnum   = None
    d_ccmax   = 5
    d_idx     = None # 0-15 for epix10ka2m, 0-3 for epix10kaquad
    d_dirxtc  = None # '/cds/data/psdm/ued/ueddaq02/xtc'
    d_dirrepo = CALIB_REPO_EPIX10KA # './myrepo' 
    d_usesmd  = True
    d_logmode = 'DEBUG'
    d_errskip = True

    h_fname   = 'input xtc file name, default = %s' % d_fname
    h_exp     = 'experiment name, default = %s' % d_exp
    h_det     = 'detector name, default = %s' % d_det
    h_runs    = 'run number or list of runs e.g. 12,14-18, default = %s' % str(d_runs)
    h_nbs     = 'number of frames to calibrate pedestals, default = %s' % str(d_nbs)
    h_ccnum   = 'step number 0-4 or all by default for processing, default = %s' % str(d_ccnum)
    h_ccmax   = 'maximal number of calib-cycles to process, default = %s' % str(d_ccmax)
    h_idx     = 'segment index (0-15 for epix10ka2m, 0-3 for quad) or all by default for processing, default = %s' % str(d_idx)
    h_dirxtc  = 'non-default xtc directory, default = %s' % d_dirxtc
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_usesmd  = 'add "smd" in dataset string, default = %s' % d_usesmd
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_errskip = 'flag to skip errors and keep processing, stop otherwise, default = %s' % d_errskip

    parser = ArgumentParser(description=usage(1))
    parser.add_argument('-f', '--fname',   default=d_fname,   type=str, help=h_fname)
    parser.add_argument('-e', '--exp',     default=d_exp,     type=str, help=h_exp)
    parser.add_argument('-d', '--det',     default=d_det,     type=str, help=h_det)
    parser.add_argument('-r', '--runs',    default=d_runs,    type=str, help=h_runs)
    parser.add_argument('-b', '--nbs',     default=d_nbs,     type=int, help=h_nbs)
    parser.add_argument('-c', '--ccnum',   default=d_ccnum,   type=int, help=h_ccnum)
    parser.add_argument('-m', '--ccmax',   default=d_ccmax,   type=int, help=h_ccmax)
    parser.add_argument('-i', '--idx',     default=d_idx,     type=int, help=h_idx)
    parser.add_argument('-x', '--dirxtc',  default=d_dirxtc,  type=str, help=h_dirxtc)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo, type=str, help=h_dirrepo)
    parser.add_argument('-S', '--usesmd',  default=d_usesmd,  type=bool,help=h_usesmd)
    parser.add_argument('-L', '--logmode', default=d_logmode, type=str, help=h_logmode)
    parser.add_argument('-E', '--errskip', default=d_errskip, type=bool,help=h_errskip)

    return parser

#----

if __name__ == "__main__":
    do_main()
    exit('End of %s'%scrname)

#---- EOF
