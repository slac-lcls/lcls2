#!/usr/bin/env python
#----
import os
import sys
from time import time

from psana.detector.Utils import info_command_line_arguments
from psana.detector.UtilsEpix10kaCalib import deploy_constants
from psana.detector.UtilsEpix10ka import GAIN_MODES_IN
from psana.detector.UtilsEpix import CALIB_REPO_EPIX10KA

import logging
logger = logging.getLogger(__name__)
DICT_NAME_TO_LEVEL = logging._nameToLevel
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

    #print('Arguments:\n')
    #for k,v in opts.items() : print('  %12s : %s' % (k, str(v)))

    if len(sys.argv)<3: exit('\n%s\n' % usage())

    assert args.exp is not None,  'WARNING: option "-e <experiment>" MUST be specified.'
    assert args.det is not None,  'WARNING: option "-d <detector-name>" MUST be specified.'
    assert args.runs is not None, 'WARNING: option "-r <run-number(s)>" MUST be specified.'

    #logging.basicConfig(format='%(levelname)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.logmode])
    logging.basicConfig(format='[%(levelname).1s] %(name)s %(message)s', level=DICT_NAME_TO_LEVEL[args.logmode])
    
    logger.debug('%s\nIn epix10ka_deploy_constants' % (50*'_'))
    logger.debug(info_command_line_arguments(parser))

    deploy_constants(**opts)

    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))


def usage(mode=0):
    if   mode == 1: return 'For specified run or timstamp combine gain constants from repository and deploy them in the calib directory'
    elif mode == 2: return 'Try: %s -h' % scrname
    else: return\
           '\n%s -e <experiment> [-d <detector>] [-r <run-number>] [-L <logging-mode>] [-D] [...]' % scrname\
           + '\nTEST COMMAND:'\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -t 20180910111049 -x /cds/data/psdm/ued/ueddaq02/xtc/ -o ./myrepo -c ./calib -L info -D' % scrname\
           + '\nREGULAR COMMAND:'\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -D -L INFO' % scrname\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -t 396 -o ./work -D -c ./calib # deploys 394-end.data for all calibrations found for runs <= 386' % scrname\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -o ./work -D -c ./calib' % scrname\
           + '\n  %s -e ueddaq02 -d epixquad -r27 -o ./work -D -c ./calib --proc=g --low=0.25 --medium=1 --high=1' % scrname\
           + '\n\n  Try: %s -h' % scrname

def argument_parser():
    #from optparse import OptionParser
    from argparse import ArgumentParser

    d_exp     = None # 'mfxx32516'
    d_det     = None # 'NoDetector.0:Epix10ka.3'
    d_runs    = None # 1021
    d_tstamp  = None # 20180910111049
    d_dirxtc  = None # '/reg/d/psdm/mfx/mfxx32516/scratch/gabriel/pulser/xtc/combined'
    d_dirrepo = CALIB_REPO_EPIX10KA # './myrepo' 
    d_dircalib= None # './calib
    d_deploy  = False
    d_logmode = 'DEBUG'
    d_proc    = 'psrg'
    d_paninds = None
    d_high    = 16.40 # 1.
    d_medium  = 5.466 # 0.33333
    d_low     = 0.164 # 0.01
    #Blaj, Gabriel <blaj@slac.stanford.edu> Mon 8/3/2020 6:52 PM
    #Hi, Here are some good starting values for the ADC to keV conversion:
    #High gain: 132 ADU / 8.05 keV = 16.40 ADU/keV
    #Medium gain: 132 ADU / 8.05 keV / 3 = 5.466 ADU/keV
    #Low gain: 132 ADU / 8.05 keV / 100 = 0.164 ADU/keV

    h_exp     = 'experiment name, default = %s' % d_exp
    h_det     = 'detector name, default = %s' % d_det
    h_runs    = 'run number for beginning of the validity range or list of comma-separated runs, default = %s' % str(d_runs)
    h_tstamp  = 'non-default time stamp in format YYYYmmddHHMMSS or run number(<10000) for constants selection in repo. '\
                'By default run time is used, default = %s' % str(d_tstamp)
    h_dirxtc  = 'non-default xtc directory which is used to access run start time, default = %s' % d_dirxtc
    h_dirrepo = 'non-default repository of calibration results, default = %s' % d_dirrepo
    h_dircalib= 'deployment calib directory if different from standard one, default = %s' % d_dircalib
    h_deploy  = 'deploy constants to the calib dir, default = %s' % d_deploy
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_high    = 'default high   gain ADU/keV, default = %s' % str(d_high)
    h_medium  = 'default medium gain ADU/keV, default = %s' % str(d_medium)
    h_low     = 'default low    gain ADU/keV, default = %s' % str(d_low)
    h_proc    = '(str) keyword for processing of "p"-pedestals, "r"-rms, "s"-status, "g" or "c" - gain or charge-injection gain,'\
              + '  default = %s' % d_proc
    h_paninds = 'comma-separated panel indexds to generate constants for subset of panels (ex.: quad from 2M), default = %s' % d_paninds

    parser = ArgumentParser(description=usage(1)) #, usage = usage())
    parser.add_argument('-e', '--exp',     default=d_exp,      type=str,   help=h_exp)
    parser.add_argument('-d', '--det',     default=d_det,      type=str,   help=h_det)
    parser.add_argument('-r', '--runs',    default=d_runs,     type=str,   help=h_runs)
    parser.add_argument('-t', '--tstamp',  default=d_tstamp,   type=int,   help=h_tstamp)
    parser.add_argument('-x', '--dirxtc',  default=d_dirxtc,   type=str,   help=h_dirxtc)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,  type=str,   help=h_dirrepo)
    parser.add_argument('-c', '--dircalib',default=d_dircalib, type=str,   help=h_dircalib)
    parser.add_argument('-D', '--deploy',  default=d_deploy,   type=bool,  help=h_deploy)
    parser.add_argument('-L', '--logmode', default=d_logmode,  type=str,   help=h_logmode)
    parser.add_argument(      '--high',    default=d_high,     type=float, help=h_high)
    parser.add_argument(      '--medium',  default=d_medium,   type=float, help=h_medium)
    parser.add_argument(      '--low',     default=d_low,      type=float, help=h_low)
    parser.add_argument('-p', '--proc',    default=d_proc,     type=str,   help=h_proc)
    parser.add_argument('-I', '--paninds', default=d_paninds,  type=str,   help=h_paninds)

    return parser

#----

if __name__ == "__main__":
    do_main()
    exit('End of %s'%scrname)

#---- EOF
