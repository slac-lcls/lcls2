#!/usr/bin/env python

import sys
from psana.detector.dir_root import DIR_REPO
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

#M14 = 0o37777 # 14-bits, 2 bits for gain mode switch

USAGE = 'Usage:'\
      + '\n  %s -k <\"str-of-datasource-kwargs\"> -d <detector> ' % SCRNAME\
      + '\n     [-o <output-result-directory>] [-L <logging-mode>] [other-kwargs]'\
      + '\nExamples:'\
      + '\nTests:'\
      + '\n  datinfo -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau # test data'\
      + '\n  %s -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau -o ./work # data' % SCRNAME\
      + '\n  %s -k exp=mfxdaq23,run=7 -d jungfrau -o ./work # data' % SCRNAME\
      + '\n  %s -k exp=ascdaq023,run=37 -d jungfrau -o ./work # data' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs= 'exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc' # None
    d_detname = 'jungfrau' #  None
    d_dirrepo = DIR_REPO_JUNGFRAU # './work'
    d_ctype = None
    d_logmode = 'INFO'
    d_dirmode = 0o2775
    d_filemode= 0o664
    d_group   = 'ps-users'
    d_version = 'V2025-03-18'
    d_segind  = None
    d_gainmode  = None

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_detname = 'detector name, default = %s' % d_detname
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_ctype   = 'calibration type, one of %s, default = %s' % (STR_LEVEL_NAMES, d_ctype)
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_version = 'constants version, default = %s' % str(d_version)
    h_segind  = 'segment index to process, default = %s' % str(d_segind)
    h_gainmode  = 'gainmode index FOR DEBUGGING, default = %s' % str(d_gainmode)

    parser = ArgumentParser(usage=USAGE, description='Proceses dark run xtc data for epix10ka')
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,    type=str,   help=h_detname)
    parser.add_argument('-o', '--dirrepo',  default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-c', '--ctype',    default=d_ctype,      type=str,   help=h_ctype)
    parser.add_argument('-L', '--logmode',  default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('--dirmode',        default=d_dirmode,    type=int,   help=h_dirmode)
    parser.add_argument('--filemode',       default=d_filemode,   type=int,   help=h_filemode)
    parser.add_argument('-v', '--version',  default=d_version,    type=str,   help=h_version)
    parser.add_argument('-I', '--segind',   default=d_segind,     type=int,   help=h_segind)
    parser.add_argument('-G', '--gainmode', default=d_gainmode,     type=int,   help=h_gainmode) 
#   parser.add_argument('-D', '--deploy',  action='store_true',  help=h_deploy)
   return parser


def do_main():
    from time import time
    parser = argument_parser()
    args = parser.parse_args()
    kwa = vars(args)

    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.detname is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'
    assert args.ctype is not None, 'WARNING: opttion "-c <calib-type>" MUST be specified.'

    t0_sec = time()
    from psana.detector.UtilsCalibRepo import save_segment_constants_in_repository
    save_segment_constants_in_repository(**kwa)
    sys.exit('End of %s, consumed time %.3f sec' % (SCRNAME, time() - t0_sec))


if __name__ == "__main__":
    do_main()

# EOF
