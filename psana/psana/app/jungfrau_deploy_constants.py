#!/usr/bin/env python

import sys
from psana.detector.dir_root import DIR_REPO_JUNGFRAU
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = 'Usage:'\
      + '\n%s -k <kwargs-for-DataSource> -d <detector-name>  [-L <logging-mode>] [-D] [...]' % SCRNAME\
      + '\nTEST COMMAND:'\
      + '\n  datinfo -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau # test data'\
      + '\n  datinfo -k exp=mfxdaq23,run=7 -d jungfrau # test data'\
      + '\n  %s -k exp=mfxdaq23,run=7 -d jungfrau -o ./work # data' % SCRNAME\
      + '\nREGULAR COMMAND:'\
      + '\n  %s -k exp=mfxdaq23,run=7 -d jungfrau -o ./work -D # data' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs = None
    d_det     = None # 'NoDetector.0:Epix10ka.3'
    d_tstamp  = None # 20180910111049
    d_dirrepo = './work' # DIR_REPO_JUNGFRAU
    d_nsegstot= 16
    d_deploy  = False
    d_logmode = 'INFO'
    d_ctdepl  = 'psrg'
    d_paninds = None
    d_high    = None #16.40 for epix10ka
    d_medium  = None #5.466
    d_low     = None #0.164
    d_version = 'V2025-02-18'
    d_run_end = 'end'
    d_comment = 'no comment'
    d_dbsuffix= ''

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_det     = 'detector name, default = %s' % d_det
    h_tstamp  = 'non-default time stamp in format YYYYmmddHHMMSS or run number(<10000) for constants selection in repo. '\
                'By default run time is used, default = %s' % str(d_tstamp)
    h_dirrepo = 'non-default repository of calibration results, default = %s' % d_dirrepo
    h_nsegstot= 'total number of segments in the detector, default = %d' % d_nsegstot
    h_deploy  = 'deploy constants to the calib dir, default = %s' % d_deploy
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_high    = 'default high   gain ADU/keV, default = %s' % str(d_high)
    h_medium  = 'default medium gain ADU/keV, default = %s' % str(d_medium)
    h_low     = 'default low    gain ADU/keV, default = %s' % str(d_low)
    h_ctdepl    = '(str) keyword for processing of "p"-pedestals, "r"-rms, "s"-status, "g" or "c" - gain or charge-injection gain,'\
              + '  default = %s' % d_ctdepl
    h_paninds = 'comma-separated panel indexds to generate constants for subset of panels (ex.: quad from 2M), default = %s' % d_paninds
    h_version = 'constants version, default = %s' % str(d_version)
    h_run_end = 'last run for validity range, default = %s' % str(d_run_end)
    h_comment = 'comment added to constants metadata, default = %s' % str(d_comment)
    h_dbsuffix= 'suffix of the PRIVATE detector db name to deploy constants, default = %s' % str(d_dbsuffix)


    parser = ArgumentParser(description='Deploy calibration files from repository to DB.', usage = USAGE)
    parser.add_argument('-k', '--dskwargs',default=d_dskwargs, type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--det',     default=d_det,      type=str,   help=h_det)
    parser.add_argument('-t', '--tstamp',  default=d_tstamp,   type=int,   help=h_tstamp)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,  type=str,   help=h_dirrepo)
    parser.add_argument('-N', '--nsegstot',default=d_nsegstot, type=int,   help=h_nsegstot)
    parser.add_argument('-L', '--logmode', default=d_logmode,  type=str,   help=h_logmode)
    parser.add_argument(      '--high',    default=d_high,     type=float, help=h_high)
    parser.add_argument(      '--medium',  default=d_medium,   type=float, help=h_medium)
    parser.add_argument(      '--low',     default=d_low,      type=float, help=h_low)
    parser.add_argument('-p', '--ctdepl',  default=d_ctdepl,   type=str,   help=h_ctdepl)
    parser.add_argument('-I', '--paninds', default=d_paninds,  type=str,   help=h_paninds)
    parser.add_argument('-v', '--version', default=d_version,  type=str,   help=h_version)
    parser.add_argument('-R', '--run_end', default=d_run_end,  type=str,   help=h_run_end)
    parser.add_argument('-C', '--comment', default=d_comment,  type=str,   help=h_comment)
    parser.add_argument('-S', '--dbsuffix',default=d_dbsuffix, type=str,   help=h_dbsuffix)
    parser.add_argument('-D', '--deploy',  action='store_true', help=h_deploy)

    return parser


def do_main():

    parser = argument_parser()
    args = parser.parse_args()
    #opts = vars(args)
    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.det is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'

    from time import time
    from psana.detector.UtilsJungfrauCalib import jungfrau_deploy_constants
    t0_sec = time()
    jungfrau_deploy_constants(parser)
    logger.info('is completed, consumed time %.3f sec' % (time() - t0_sec))


if __name__ == "__main__":
    do_main()
    sys.exit('End of %s'%SCRNAME)

# EOF
