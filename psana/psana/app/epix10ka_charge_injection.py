#!/usr/bin/env python

#from Detector.UtilsEpix10kaCalib import offset_calibration, CALIB_REPO_EPIX10KA, DIR_LOG_AT_START

import sys
from time import time
#from psana.detector.UtilsEpix10kaCalib import pedestals_calibration, DIR_REPO_EPIX10KA
from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL, init_stream_handler

logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].split('/')[-1]

def do_main():

    t0_sec = time()

    parser = argument_parser()
    args = parser.parse_args()
    kwa = vars(args)
    #defs = vars(parser.parse_args([])) # dict of defaults only

    if len(sys.argv) < 4: print('\n%s\n' % USAGE)

    if args.dskwargs is None : raise IOError('WARNING: option "-k <kwargs-for-DataSource>" MUST be specified.')
    if args.det      is None : raise IOError('WARNING: option "-d <detector-name>" MUST be specified.')
    if args.idx      is None : raise IOError('WARNING: option "-i <panel index (0-15/3 for epix10ka2m/quad)>" MUST be specified.')

    init_stream_handler(loglevel=args.logmode)

    from psana.detector.Utils import info_parser_arguments
    logger.info(info_parser_arguments(parser))

    import psana.detector.UtilsEpix10kaChargeInjection as uci
    uci.charge_injection(**kwa)

    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))


USAGE = 'Usage:'\
      + '\n  %s -e <experiment> [-d <detector>] [-r <run-number>]' % SCRNAME\
      + '\n       [-i <panel-index>] [-n <nspace>] [-x <dsname-or-extension>] [-o <output-result-directory>] [-L <logging-mode>] [-p]'\
      + '\nTEST COMMAND FOR GRAPHICS:'\
      + '\n  %s -k exp=ascdaq18,run=171 -d epixhr -i0 -o ./work    # plot all' % SCRNAME\
#      + '\n  %s ... -i0 -o ./work        # plot all' % SCRNAME\
#      + '\n  %s ... -i0 -o ./work -P     # skip pedestal calib-cycles' % SCRNAME\
#      + '\n  %s ... -i0 -o ./work -O     # skip offset calib-cycles' % SCRNAME\
#      + '\n  %s ... -i0 -o ./work -s 60  # skip first 60 calib-cycles' % SCRNAME\
#      + '\nDEBUGGING AND REGULAR COMMAND FOR DATA PROCESSING:'\
#      + '\n  %s ... -i0 -o ./work -p' % SCRNAME\
#      + '\n  %s ... -i0 -o ./work -p' % SCRNAME\
#      + '\n  %s ... -i5 -p &> 2020-06-01-r52-offset-log05.txt&' % SCRNAME


def argument_parser() :
#    d_exp      = None # 'mfxx32516'
#    d_run      = None # 1021 or 1021,1022-1025
#    d_dsnamex  = None # ':dir=/reg/d/psdm/mfx/mfxx32516/scratch/.../xtc'
    d_dskwargs = None  # exp=ascdaq18,run=171
    d_det      = None  # 'NoDetector.0:Epix10ka.3'
    d_idx      = None  # 0-15 for epix10ka2m, 0-3 for epix10kaquad
    d_nbs      = 4600  # number of frames
    d_nspace   = 5     # space between charge injected picsels
    d_dirrepo  = './work' #CALIB_REPO_EPIX10KA # './myrepo'
    d_display  = True
    d_logmode  = 'INFO'
    d_dopeds   = False
    d_dooffs   = True
    d_nperiods = True
    d_npoff    = 10
    d_ccnum    = None
    d_ccmax    = 2 * d_nspace**2
    d_ccskip   = 0
    d_errskip  = False
    d_savechi2 = False
    d_pixrc    = None
    d_dirmode  = 0o2775
    d_filemode = 0o664
    d_group    = 'ps-users'

    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_det     = 'detector name, default = %s' % d_det
    h_idx     = 'panel index (0-15/3 for epix10ka2m/quad), default = %s' % str(d_idx)
    h_nbs     = 'number of frames to calibrate offsets, default = %s' % str(d_nbs)
    h_nspace  = 'space between calibrated pixels, default = %s' % str(d_nspace)
    h_dirrepo = 'repository for calibration results, default = %s' % d_dirrepo
    h_display = 'turn off graphical display, default = %s' % d_display
    h_logmode = 'logging mode, one of %s, default = %s' % (' '.join(DICT_NAME_TO_LEVEL.keys()), d_logmode)
    h_dopeds  = 'do pedestals evaluation, default = %s' % d_dopeds
    h_dooffs  = 'do offset evaluation, default = %s' % d_dooffs
    h_nperiods= 'use all found saw periods of the pulser, default = %s' % d_nperiods
    h_npoff   = 'discard in fit number of points on trace around switching point, default = %s' % str(d_npoff)
    h_ccnum   = 'calibcycle number to process [0,102] or all by default, default = %s' % str(d_ccnum)
    h_ccmax   = 'maximal number of calib-cycles to process, default = %s' % str(d_ccmax)
    h_ccskip  = 'skip number of first consecutive calibcycles for debugging, default = %s' % str(d_ccskip)
    h_errskip = 'flag to skip errors and keep processing (stop otherwise), default = %s' % d_errskip
    h_savechi2= 'save chi2 files, default = %s' % d_savechi2
    h_pixrc   = 'selected pixel for graphics: comma separated pixel row and colon, ex. 23,234, default = %s' % d_pixrc
    h_dirmode = 'mode for all mkdir, default = %s' % oct(d_dirmode)
    h_filemode= 'mode for all saved files, default = %s' % oct(d_filemode)
    h_group   = 'group ownership for all files, default = %s' % d_group

    from argparse import ArgumentParser
    parser = ArgumentParser(usage=USAGE, description='Process charge injection data for epix10ka')
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str,             help=h_dskwargs)
    parser.add_argument('-d', '--det',      default=d_det,      type=str,             help=h_det)
    parser.add_argument('-i', '--idx',      default=d_idx,      type=int,             help=h_idx)
    parser.add_argument('-b', '--nbs',      default=d_nbs,      type=int,             help=h_nbs)
    parser.add_argument('-n', '--nspace',   default=d_nspace,   type=int,             help=h_nspace)
    parser.add_argument('-o', '--dirrepo',  default=d_dirrepo,  type=str,             help=h_dirrepo)
    parser.add_argument('-L', '--logmode',  default=d_logmode,  type=str,             help=h_logmode)
    parser.add_argument('-X', '--npoff',    default=d_npoff,    type=int,             help=h_npoff)
    parser.add_argument('-c', '--ccnum',    default=d_ccnum,    type=int,             help=h_ccnum)
    parser.add_argument('-m', '--ccmax',    default=d_ccmax,    type=int,             help=h_ccmax)
    parser.add_argument('-s', '--ccskip',   default=d_ccskip,   type=int,             help=h_ccskip)
    parser.add_argument('-G', '--pixrc',    default=d_pixrc,    type=str,             help=h_pixrc)
    parser.add_argument('--dirmode',        default=d_dirmode,  type=int,             help=h_dirmode)
    parser.add_argument('--filemode',       default=d_filemode, type=int,             help=h_filemode)
    parser.add_argument('--group',          default=d_group,    type=str,             help=h_group)
    parser.add_argument('-p', '--display',  default=d_display,  action='store_false', help=h_display)
    parser.add_argument('-P', '--dopeds',   default=d_dopeds,   action='store_false', help=h_dopeds)
    parser.add_argument('-O', '--dooffs',   default=d_dooffs,   action='store_false', help=h_dooffs)
    parser.add_argument('-N', '--nperiods', default=d_nperiods, action='store_false', help=h_nperiods)
    parser.add_argument('-E', '--errskip',  default=d_errskip,  action='store_true',  help=h_errskip)
    parser.add_argument('-C', '--savechi2', default=d_savechi2, action='store_true',  help=h_savechi2)

    return parser


if __name__ == "__main__":
    do_main()
    sys.exit('End of %s'%SCRNAME)

# EOF
