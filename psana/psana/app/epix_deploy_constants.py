#!/usr/bin/env python

import os
import sys
from psana.detector.dir_root import DIR_REPO_EPIX
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = 'Usage:'\
      + '\n  %s -k <\"str-of-datasource-kwargs\"> -d <detector> ' % SCRNAME\
      + '\n     [-o <output-result-directory>] [-L <logging-mode>] [other-kwargs]'\
      + '\nTests:'\
      + '\n  %s -k exp=mfxdet23,run=15 -d epixuhr -o work1 # data on psana' % SCRNAME\
      + '\n  %s -k exp=mfx101628626,run=163 -d epixuhr3x2 -o work1' % SCRNAME\
      + '\n  %s -k exp=ascdaq123,run=578 -d epixuhr3x2 -o work1' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def do_main():
    parser = argument_parser()
    args = parser.parse_args()
    if len(sys.argv)<3: sys.exit('\n%s\n\nMISSING ARGUMENTS - EXIT\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.det is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'
    assert os.path.exists(args.dirrepo), f'WARNING: repository "{args.dirrepo}" MUST exist before using this script.'\
           '\n\nHINT: create/populate repository with scripts like calibrepo or epix_dark_proc\n'

    from psana.detector.UtilsEpixCalib import epix_deploy_constants
    from time import time
    t0_sec = time()
    epix_deploy_constants(parser)
    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))
    sys.exit('End of %s' % SCRNAME)


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs = None
    d_det     = None # 'epixquad'
    d_dirrepo = DIR_REPO_EPIX
    d_logmode = 'INFO'
    d_version = 'V2026-07-15'
    d_plotim  = 0
    d_deploy  = False
    d_ctdepl  = 'prs'   # for constants from dark, 'prsnxg'
    d_tstamp  = None # 20180910111049
    d_run_beg = None
    d_run_end = 'end'
    d_comment = None
    d_dbsuffix= None

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_det     = 'detector name, default = %s' % d_det
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_version = 'constants version, default = %s' % str(d_version)
    h_plotim  = 'plot image/s of pedestals, default = %s' % str(d_plotim)
    h_deploy  = 'DEPLOY: deploy constants to the calibration DB, default = %s' % d_deploy
    h_ctdepl  = 'DEPLOY: (str) keyword for deployment: "p"-pedestals, "r"-rms, "s"-status, "x" - max, "n" - min, "g" - gain, default = %s' % d_ctdepl
    h_tstamp  = 'DEPLOY: non-default time stamp in format YYYYmmddHHMMSS, if None - run time is used, default = %s' % str(d_tstamp)
    h_run_beg = 'DEPLOY: first run for validity range, if None - use first run from -k, default = %s' % str(d_run_beg)
    h_run_end = 'DEPLOY: last run for validity range, default = %s' % str(d_run_end)
    h_comment = 'DEPLOY: comment added to constants metadata, default = %s' % str(d_comment)
    h_dbsuffix= 'DEPLOY: suffix of the PRIVATE detector db name to deploy constants, default = %s' % str(d_dbsuffix)

    parser = ArgumentParser(usage=USAGE, description='Proceses dark run xtc data for epix10ka')
    parser.add_argument('-k', '--dskwargs',default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--det',     default=d_det,        type=str,   help=h_det)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-L', '--logmode', default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-v', '--version', default=d_version,    type=str,   help=h_version)
    parser.add_argument('-p', '--plotim',  default=d_plotim,     type=int,   help=h_plotim)
    parser.add_argument('-D', '--deploy',  action='store_true', help=h_deploy)
    parser.add_argument('--ctdepl',  default=d_ctdepl,   type=str,   help=h_ctdepl)
    parser.add_argument('--tstamp',  default=d_tstamp,   type=int,   help=h_tstamp)
    parser.add_argument('--run_beg', default=d_run_beg,  type=int,   help=h_run_beg)
    parser.add_argument('--run_end', default=d_run_end,  type=str,   help=h_run_end)
    parser.add_argument('--comment', default=d_comment,  type=str,   help=h_comment)
    parser.add_argument('--dbsuffix',default=d_dbsuffix, type=str,   help=h_dbsuffix)

    return parser


if __name__ == "__main__":
    do_main()

# EOF
