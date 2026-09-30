#!/usr/bin/env python

import sys
from psana.detector.dir_root import DIR_REPO
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
from psana.pscalib.calib.CalibConstants import list_calib_names
from psana.detector.UtilsJungfrauCalib import DIC_GAIN_MODE
from psana.detector.UtilsEpix10ka import GAIN_MODES_IN
lst_gainmodes = list(DIC_GAIN_MODE.keys()) # ['DYNAMIC', 'FORCE_SWITCH_G1', 'FORCE_SWITCH_G2']
lst_gainmodes += list(GAIN_MODES_IN)

logger = logging.getLogger(__name__)

#print('XXX sys.argv[0]', sys.argv[0])

SCRNAME = sys.argv[0].rsplit('/')[-1]

#M14 = 0o37777 # 14-bits, 2 bits for gain mode switch

DESCRIPTION = 'add per-panel/per-gain-range constants to repository for jungfrau, epix10ka, etc'
USAGE = DESCRIPTION\
      + '\n\nUsage:'\
      + '\n  %s -k <\"str-of-datasource-kwargs\"> -d <detector> ' % SCRNAME\
      + '\n     [-o <output-result-directory>] [-L <logging-mode>] [other-kwargs]'\
      + '\nExamples:'\
      + '\nTests:'\
      + '\n  datinfo -k exp=ascdaq023,run=37,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau # test data'\
      + '\n\n  make file with constants'\
      + '\n  make_or_print_npyarray -s 512,1024 -m 10 -r 1 -t np.float64 -f test-jf-2d.npy  # random 2-d array for jungfrau'\
      + '\n  make_or_print_npyarray -s 192,384 -m 10 -r 1 -t np.float64 -f gains2d  # random 2-d array for epixm320'\
      + '\n  make_or_print_npyarray -s 4,192,384 -m 158.796 -r 0 -t np.float64 -f gains3d  # constant 3-d array array for epixm320'\
      + '\n'\
      + '\n  %s -k exp=ascdaq023,run=37 -d jungfrau -c pedestals -G g0 -I 1 -F test_2darr.npy -o ./work' % SCRNAME\
      + '\n  %s -k exp=mfx101332224,run=15 -d jungfrau -c pixel_gain -G g2 -F fake_g2.npy -I 0 -o ./work' % SCRNAME\
      + '\n  %s -k exp=ascdaq123,run=347 -d epixm -c pixel_gain -F gains3d.npy -G SL -o ./work # US IT FOR epixm DO NOT USE -I' % SCRNAME\
      + '\n  %s -k exp=ascdaq123,run=347 -d epixm -c pixel_gain -F gains2d.npy -G SL -o ./work -I 3 # WORKS FOR epixm, BUT IS NOT USED FOR DEPLOYMENT' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def argument_parser():
    from argparse import ArgumentParser

    d_dskwargs= None # 'exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc'
    d_detname = None # 'jungfrau'
    d_dirrepo = DIR_REPO # './work'
    d_ctype = None
    d_version = 'V2025-11-06'
    d_segind  = None
    d_gainmode = None
    d_logmode = 'INFO'
    d_dirmode = 0o2775
    d_filemode= 0o664
    d_fname2darr = 'test_2darr.npy' # None
    #d_group   = 'ps-users'

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_detname = 'detector name, default = %s' % d_detname
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_ctype   = 'calibration type, one of %s, default = %s' % (list_calib_names, d_ctype)
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_dirmode = 'directory access mode, default = %s' % oct(d_dirmode)
    h_filemode= 'file access mode, default = %s' % oct(d_filemode)
    h_version = 'script version, default = %s' % str(d_version)
    h_segind  = 'segment index to process, default = %s' % str(d_segind)
    h_gainmode  = 'gainmode, (detector-dependent) one of %s, default = %s' % (str(lst_gainmodes), d_gainmode)
    h_fname2darr = 'file name for 2d panel constants, default = %s' %d_fname2darr

    parser = ArgumentParser(prog=SCRNAME, usage=USAGE, description=DESCRIPTION, epilog='help: %s -h' % SCRNAME )
    parser.add_argument('-k', '--dskwargs',   default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname',    default=d_detname,    type=str,   help=h_detname)
    parser.add_argument('-o', '--dirrepo',    default=d_dirrepo,    type=str,   help=h_dirrepo)
    parser.add_argument('-c', '--ctype',      default=d_ctype,      type=str,   help=h_ctype)
    parser.add_argument('-I', '--segind',     default=d_segind,     type=int,   help=h_segind)
    parser.add_argument('-G', '--gainmode',   default=d_gainmode,   type=str,   help=h_gainmode)
    parser.add_argument('-L', '--logmode',    default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('--dirmode',          default=d_dirmode,    type=int,   help=h_dirmode)
    parser.add_argument('--filemode',         default=d_filemode,   type=int,   help=h_filemode)
    parser.add_argument('-v', '--version',    default=d_version,    type=str,   help=h_version)
    parser.add_argument('-F', '--fname2darr', default=d_fname2darr, type=str,   help=h_fname2darr)
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
