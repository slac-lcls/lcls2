#!/usr/bin/env python
"""
Created on 2021-07-13 by Mikhail Dubrovin
"""

import os
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' # get rid of libGL error: unable to load driver: swrast_dri.so

import sys
from psana.graphqt.IVMain import do_main, logging

SCRNAME = sys.argv[0]#.rsplit('/')[-1]
LEVEL_NAMES = ', '.join(list(logging._levelToName.values()))

USAGE = 'Image Viewer\n\n'\
      + 'command example:\n'\
      + '  iv /cds/group/psdm/detector/data2_test/misc/cspad2x2.1-ndarr-ave-meca6113-r0028.npy\n' #%SCRNAME


def image_viewer():
    """Image Viewer GUI
    """
    print(USAGE)

    parser = input_argument_parser()
    args = parser.parse_args() # TRICK! this line allows -h or --help potion !!!
    print('parser args:', args)
    kwargs = vars(args) # use dict in stead of Namespace
    pargs = args.posargs

    if kwargs['fname'] is None:
       kwargs['fname'] = pargs[0] if len(pargs)>0 else None

    if len(sys.argv) == 1:
        print(80*'_')
        parser.print_help()
        parser.print_usage()
        print(80*'_')

    do_main(**kwargs)


class Constants:
    #import psana.pscalib.calib.CalibConstants as cc
    d_posargs= None
    d_fname = None #'/cds/group/psdm/detector/data2_test/misc/cspad2x2.1-ndarr-ave-meca6113-r0028.npy'
    d_loglevel = 'INFO'

    h_posargs  = 'list of positional arguments, default = %s' % d_posargs
    h_fname      = 'file name for input array, default = %s' % d_fname
    h_loglevel   = 'logging level from the list (%s), default = %s' % (LEVEL_NAMES, d_loglevel)


def input_argument_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Image Viewer')
    c = Constants()
    parser.add_argument('posargs', default=c.d_posargs, type=str, help=c.h_posargs, nargs='*')
    parser.add_argument('-f', '--fname', default=c.d_fname, type=str, help=c.h_fname)
    parser.add_argument('-l', '--loglevel', default=c.d_loglevel, type=str, help=c.h_loglevel)
    return parser
  

if __name__ == "__main__":
    image_viewer()
    sys.exit('End of %s' % SCRNAME)

# EOF
