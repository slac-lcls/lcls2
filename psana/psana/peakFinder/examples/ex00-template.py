#!/usr/bin/env python


import sys
#import math
#import numpy as np
#from time import time

#import psalg_ext as algos
#from psalg_ext import local_minima_1d, local_maxima_1d,\
#from psana.pyalgos.generic.NDArrUtils import print_ndarr, reshape_to_2d
#import psana.pyalgos.generic.Graphics as gr
#from psana.pyalgos.generic.NDArrGenerators import random_standard, add_random_peaks, add_ring

#----------

def ex_01(ntest) : 
    print('ntest %s' % ntest)

#----------

if __name__ == "__main__" :
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s:' % tname)
    if   tname == '1' : ex_01(tname);
    elif tname == '2' : ex_01(tname)
    else : print('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)
 
#----------
