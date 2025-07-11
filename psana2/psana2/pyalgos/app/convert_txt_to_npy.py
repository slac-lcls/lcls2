#!/usr/bin/env python

import numpy as np
import sys
import os
import string

def print_exit(case, p1=None, p2=None) :
    proc  = './%s' % os.path.basename(sys.argv[0])
    usg = 'Usage  :  %s fname.txt fname.npy\n' \
          '     or:  %s fname.txt fname.npy <shape>\n' \
          'Example:  %s fname.txt fname.npy 32,185,388' % (proc, proc, proc)

    msg = 'Used command: %s' % ' '.join(sys.argv)

    if case == 1 : msg += '\nWrong number of arguments.\n%s' % usg
    if case == 2 : msg += '\nExpected extension for file %s is ".npy"' % (os.path.basename(sys.argv[2]))
    if case == 3 : msg += '\nSize from the shape: %d is not equal to the input array size: %d.\n%s' % (p1,p2,usg)
    sys.exit('%s\nCONVERSION ABORTED' % msg)

def shape_and_size_from_string(s) :
    """Converts string like '32,185,388' to tuple (32, 185, 388) and size 32*185*388
    """
    if s is None : return None, None
    shape = tuple([int(v) for v in s.split(',')])
    size = 1
    for v in shape : size*=v
    return shape, size

def parse_input_pars() :

    if len(sys.argv)<3 : print_exit(1)
    if len(sys.argv)>4 : print_exit(1)

    finp = sys.argv[1]
    fout = sys.argv[2]
    str_shape = None if len(sys.argv)==3 else sys.argv[3]

    if os.path.splitext(fout)[1] != '.npy' : print_exit(2)

    shape, size = shape_and_size_from_string(str_shape)

    return finp, fout, shape, size

def do_main() :

    finp, fout, shape, size = parse_input_pars()

    nda =  np.loadtxt(finp)
    print('loaded array shape %s' %str(nda.shape))
    if shape is not None :
        if size != nda.size : print_exit(3, size, nda.size)
        nda.shape = shape

    print('Convert %s to %s with shape=%s' % (finp, fout, str(shape)))

    np.save(fout, nda)

if __name__ == '__main__':
    do_main()
    sys.exit()

# EOF

