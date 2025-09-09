#!/usr/bin/env python

import sys
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES
from psana.pscalib.calib.CalibConstants import list_calib_names
import psana.pscalib.calib.UtilsCalibValidity as ucv
#import logging
#logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

DESCRIPTION = '\nprints run/time validity ranges of calibration constants for specified data source and detector'
USAGE = '\n\nCLI:'\
      + '\n  %s -k <\"str-of-datasource-kwargs\"> -d <detector> -c <calib-type>'\
        '   # use psana to get detector long/short name from -d <detector>' % SCRNAME\
      + '\n  %s exp=<expname>,run=<runnum>,shortname=<det-short-name>,ctype=<calib-type>'\
        ' -s <info-to-show>   # access to DB without psana' % SCRNAME\
      + '\n\nTest examples:'\
      + '\n  %s -k exp=mfx101332224,run=66 -d jungfrau -c pedestals -s rdt' % SCRNAME\
      + '\n  %s exp=mfx101332224,run=66,shortname=jungfrau_000003,ctype=pedestals -s rdt' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def argument_parser():
    from argparse import ArgumentParser

    d_allargs = None # 'exp=mfxdaq23,run=7,shortname=jungfrau_000003,ctype=pedestals'
    d_dskwargs= None # 'exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc'
    d_detname = None # 'jungfrau'
    d_ctype   = None
    d_version = 'V2025-07-25'
    d_logmode = 'INFO'
    d_show    = 'rd'
    h_allargs = 'REPLACEMENT FOR OPTIONS -k,-d,-c if <det-short-name> is known (psana is not uesd, only DB):'\
                ' string of comma-separated (no spaces) arguments,'\
                ' ex: exp=<expname>,run=<runnum>,shortname=<det-short-name>,ctype=<calib-type>, default = %s' % d_allargs
    h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
                ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
                ' or <fname.xtc> or files=<fname.xtc>'\
                ' or pythonic dict of generic kwargs, e.g.:'\
                ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs
    h_detname = 'detector name, default = %s' % d_detname
    h_ctype   = 'calibration type, one of %s, default = %s' % (list_calib_names, d_ctype)
    h_version = 'script version, default = %s' % str(d_version)
    h_logmode = 'logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_show    = 'show/print info about calib constants: r = run ranges from experiment DB, t = time ranges from detectr DB,'\
                ' d = prtial metadata for specified run, default = %s' % d_show

    parser = ArgumentParser(prog=SCRNAME, usage=USAGE, description=DESCRIPTION, epilog='help: %s -h' % SCRNAME )
    parser.add_argument('allargs', nargs='?', default=d_allargs,    type=str,   help=h_allargs)
    parser.add_argument('-k', '--dskwargs',   default=d_dskwargs,   type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname',    default=d_detname,    type=str,   help=h_detname)
    parser.add_argument('-c', '--ctype',      default=d_ctype,      type=str,   help=h_ctype)
    parser.add_argument('-s', '--show',       default=d_show,       type=str,   help=h_show)
#    parser.add_argument('-L', '--logmode',    default=d_logmode,    type=str,   help=h_logmode)
    parser.add_argument('-v', '--version',    default=d_version,    type=str,   help=h_version)
    return parser


def do_main():
    from time import time
    parser = argument_parser()
    args = parser.parse_args()
    kwa = vars(args)

    #if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    #assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    #assert args.detname is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'
    #assert args.ctype is not None, 'WARNING: opttion "-c <calib-type>" MUST be specified.'

    import psana.pscalib.calib.UtilsCalibValidity as ucv
    from time import time
    t0_sec = time()
    #ucv._calib_validity_ranges('mfx101332224', 'jungfrau_000003', ctype='pedestals')
    ucv.calib_validity_ranges(**kwa)
    print('Consumed time (sec): %.6f' % (time()-t0_sec))


    sys.exit('End of %s, consumed time %.3f sec' % (SCRNAME, time() - t0_sec))


if __name__ == "__main__":
    do_main()

# EOF
