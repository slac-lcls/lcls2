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
      + '\n  %s -k exp=ascdaq123,run=347 -d epixm -o ./work -s psrog' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def do_main():

    parser = argument_parser()
    args = parser.parse_args()
    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.det is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'

    from psana.detector.UtilsEpixm320Calib import deploy_constants_script
    from time import time
    t0_sec = time()
    deploy_constants_script(parser)
    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))
    sys.exit('End of %s' % SCRNAME)


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs = None
    d_det      = None # 'epixquad'
    d_dirrepo  = DIR_REPO_EPIX10KA
    d_logmode  = 'INFO'
    d_dirmode  = 0o2775
    d_filemode = 0o664
    d_version  = 'V2025-11-03'
    d_deploy   = False
    d_plotim   = 0
    d_select    = 'psrog'

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_det     = 'detector name, default = %s' % d_det
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_version = 'constants version, default = %s' % str(d_version)
    h_deploy  = 'deploy constants to the calibration DB, default = %s' % d_deploy
    h_plotim  = 'plot image/s of pedestals, default = %s' % str(d_plotim)
    h_select  = '(str) keyword for selection of constants to deploy "p"-pedestals, "r"-rms, "s"-status, '\
              + '"g" - gain, "o" - offset, "x" - max, "n" - min, default = %s' % d_select

    parser = ArgumentParser(usage=USAGE, description='combine calibration constants from repository and deploy in DB')
    parser.add_argument('-k', '--dskwargs',default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--det',     default=d_det,        type=str,   help=h_det)
    parser.add_argument('-o', '--dirrepo', default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-L', '--logmode', default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('--dirmode',       default=d_dirmode,    type=int,   help=h_dirmode)
    parser.add_argument('--filemode',      default=d_filemode,   type=int,   help=h_filemode)
    parser.add_argument('-v', '--version', default=d_version,    type=str,   help=h_version)
    parser.add_argument('-D', '--deploy',  action='store_true',              help=h_deploy)
    parser.add_argument('-p', '--plotim',  default=d_plotim,     type=int,   help=h_plotim)
    parser.add_argument('-s', '--select',  default=d_select,     type=str,   help=h_select)
    return parser


if __name__ == "__main__":
    do_main()

# EOF
