#!/usr/bin/env python

"""
Created on 2019-11-13 by Mikhail Dubrovin
"""

import os
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' # get rid of libGL error: unable to load driver: swrast_dri.so
import sys
from psana2.graphqt.H5VMain import hdf5explorer

FNAME_TEST = '/reg/g/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5'

USAGE = 'command examples for app %s\n'%sys.argv[0]\
      + '  hdf5explorer\n'\
      + '  hdf5explorer <hdf5-file-name> [options]\n'\
      + '  hdf5explorer /reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5\n'\
      + '  hdf5explorer /reg/g/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5\n'\
      + '  hdf5explorer /reg/g/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5 -l INFO'


def hdf5explorer_gui():
    parser = input_option_parser()

    if len(sys.argv) == 1:
        print(80*'_')
        #parser.print_help()
        parser.print_usage()
        print(80*'_')

    (popts, pargs) = parser.parse_args() # TRICK! this line allows -h or --help option !!!
    opts = vars(popts)
    kwargs = opts

    fname = pargs[0] if len(pargs) else FNAME_TEST
    kwargs['fname'] = fname
    kwargs['rec_at_start'] = True

    hdf5explorer(**kwargs)


def input_option_parser():

    from optparse import OptionParser

    d_loglevel   = 'INFO'
    d_logdir     = '%s/hdf5explorer-log' % os.path.expanduser('~')
    d_savelog    = False

    h_loglevel   = 'logging level, default = %s' % d_loglevel
    h_logdir     = 'logger directory, default = %s' % d_logdir
    h_savelog    = 'save log-file at exit, default = %s' % d_savelog

    parser = OptionParser(description='HDF5 Explorer GUI', usage=USAGE)

    parser.add_option('-l', '--loglevel',   default=d_loglevel,   action='store', type='string', help=h_loglevel)
    parser.add_option('-L', '--logdir',     default=d_logdir,     action='store', type='string', help=h_logdir)
    parser.add_option('-S', '--savelog',    default=d_savelog,    action='store_true',           help=h_savelog)

    return parser


if __name__ == "__main__":
    hdf5explorer_gui()
    sys.exit(0)

# EOF
