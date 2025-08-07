#!/usr/bin/env python

import os
import sys
import numpy as np
import psana.pyalgos.generic.NDArrUtils as ndu
#import psana.pyalgos.generic.NDArrGenerators as ag
from psana.detector.Utils import info_dict, info_command_line, info_namespace, info_parser_arguments, str_tstamp

SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = '\n  %s -s <shape> [kwargs]' % SCRNAME\
      + '\nCOMMAND EXAMPLES:'\
      + '\n  %s -s 512,1024 -m 10 -r 1 -t np.float64 -f nda-random.npy  # create' % SCRNAME\
      + '\n  %s -i nda-random.npy  # load and print array from file' % SCRNAME\

def make_random_nda(shape=(704, 768), mean=100, rms=10, dtype=np.float64, fname='fake.npy'):
    print('rms', rms)
    a = (mean + rms*np.random.standard_normal(size=shape).astype(dtype=dtype)) if abs(rms)>0.001 else\
         mean * np.ones(shape, dtype=dtype)
    a = a.astype(dtype=dtype)
    print(ndu.info_ndarr(a, 'save %s:'%fname, last=10))
    np.save(fname, a)

def load_print_nda(fname=None):
    assert os.path.exists(fname)
    a = np.load(fname)
    print(ndu.info_ndarr(a, 'file %s:'%fname, last=5))

def argument_parser():
    from argparse import ArgumentParser

    d_shape   = '(704,768)'
    d_mean    = 2
    d_rms     = 0
    d_dtype   = 'np.float64'
    d_fname   = 'nda-random.npy'
    d_ifname  = None

    h_shape   = 'shape of the array, comma-separated, no-spaces, e.g.: (704,768), (512,1024), default: %s' % d_shape
    h_mean    = 'central value of normal distribution, default: %0.3f' % d_mean
    h_rms     = 'rms of normal distribution or 0 for constant mean array, default: %0.3f' % d_rms
    h_dtype   = 'str data type, default: %s' % d_dtype
    h_fname   = 'output file name, default: %s' % d_fname
    h_ifname  = 'nput file name - IF NOT None - prints content, default: %s' % d_ifname

    parser = ArgumentParser(usage=USAGE,  description='creates(random)/saves/loads/prints numpy array')
    parser.add_argument('-s', '--shape',  default=d_shape, type=str,   help=h_shape)
    parser.add_argument('-m', '--mean',   default=d_mean,  type=float, help=h_mean)
    parser.add_argument('-r', '--rms',    default=d_rms,   type=float, help=h_rms)
    parser.add_argument('-t', '--dtype',  default=d_dtype, type=str,   help=h_dtype)
    parser.add_argument('-f', '--fname',  default=d_fname, type=str,   help=h_fname)
    parser.add_argument('-i', '--ifname', default=d_fname, type=str,   help=h_ifname)

    return parser


def do_main():

    parser = argument_parser()
    args = parser.parse_args()
    opts = vars(args)

    print('command line: %s' % info_command_line())
    print(info_parser_arguments(parser))

    shape = eval(args.shape)
    dtype = eval(args.dtype)

    if args.ifname is not None:
        load_print_nda(fname=args.ifname)
    else:
        make_random_nda(shape=eval(args.shape), mean=args.mean, rms=args.rms,\
                        dtype=eval(args.dtype), fname=args.fname)


if __name__ == "__main__":
    do_main()
    sys.exit('End of %s'%SCRNAME)

# EOF
