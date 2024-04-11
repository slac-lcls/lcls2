#!/usr/bin/env python

from psana.app.epix10ka_pedestals_calibration import *
logger = logging.getLogger(__name__)
SCRNAME = sys.argv[0].rsplit('/')[-1]


USAGE = 'Usage:'\
      + '\n  %s -k <\"str-of-datasource-kwargs\"> -d <detector> ' % SCRNAME\
      + '\n     [-o <output-result-directory>] [-L <logging-mode>] [other-kwargs]'\
      + '\nExamples:'\
      + '\nTests:'\
      + '\n  %s -k exp=tstx00417,run=317,dir=/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/ -d tst_epixm -o ./work' % SCRNAME\
      + '\n\n  Try: %s -h' % SCRNAME


def do_main():

    parser = argument_parser()
    d_datbits = 0o77777
    h_datbits = 'data bits, e.g. 0x7fff is 15-bit mask for epixm320, default = %s' % hex(d_datbits)
    parser.add_argument('--datbits', default=d_datbits,    type=int,   help=h_datbits)
    args = parser.parse_args()
    if len(sys.argv)<3: sys.exit('\n%s\n\nEXIT DUE TO MISSING ARGUMENTS\n' % USAGE)
    assert args.dskwargs is not None, 'WARNING: option "-k <DataSource-kwargs>" MUST be specified.'
    assert args.det is not None, 'WARNING: option "-d <detector-name>" MUST be specified.'

    from psana.detector.UtilsEpixm320Calib import pedestals_calibration
    from time import time
    t0_sec = time()
    #if args.det  == 'tst_epixm': sys.exit('TEST EXIT FOR %s' % args.det)
    pedestals_calibration(parser)
    logger.info('DONE, consumed time %.3f sec' % (time() - t0_sec))
    sys.exit('End of %s' % SCRNAME)


if __name__ == "__main__":
    do_main()

# EOF
