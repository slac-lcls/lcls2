#!/usr/bin/env python

#import h5py                    
import numpy as np             
import sys
import os

##-----------------------------------------------------

def print_exit(case) :
    msg = 'Used command: %s' % ' '.join(sys.argv)
    if case == 1 : msg += '\nExpected command: %s fname.npy fname.txt' % (os.path.basename(sys.argv[0]))
    if case == 2 : msg += '\nExpected extension for file %s is ".npy"' % (os.path.basename(sys.argv[1]))
    sys.exit('%s\nCONVERSION ABORTED' % msg)

##-----------------------------------------------------

def parse_input_pars() :
    #print('len(sys.argv)', len(sys.argv))
    #print(sys.argv)
    
    if len(sys.argv)!=3 : print_exit(1) 

    finp = sys.argv[1]
    fout = sys.argv[2]

    if os.path.splitext(finp)[1] != '.npy' : print_exit(2) 

    return finp, fout

##-----------------------------------------------------

def do_main() :

    finp, fout = parse_input_pars()

    nda =  np.load(finp)

    print('Convert %s to %s' % (finp, fout))

    if   nda.size == 32*185*388 : nda.shape = (32*185, 388)
    elif nda.size ==  2*185*388 :
        if nda.shape[-1] ==   2 : nda.shape = (185*388, 2)
        if nda.shape[-1] == 388 : nda.shape = (2, 185*388)
    else                        : nda.shape = (nda.size,)

    np.savetxt('%s' % fout, nda, fmt = "%f")

##-----------------------------------------------------

if __name__ == '__main__':
    do_main()
    sys.exit()

##-----------------------------------------------------

