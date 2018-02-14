#!@PYTHON@
####!/usr/bin/env python
#------------------------------
"""
:py:class:`Entropy.py` - collection of methods to evaluate data array entropy
=============================================================================

Usage::

    # Import
    # ==============
    from psana.pyalgos.generic.Entropy import *

    # import for test only
    from pyalgos.generic.NDArrGenerators import random_standard
    from pyalgos.generic.NDArrUtils import print_ndarr

    arr_float = random_standard(shape=(1000), mu=200, sigma=25, dtype=np.float)
    arr_int16 = arr_float.astype(np.int16)  
    print_ndarr(arr_int16, name='arr_int16', first=0, last=10)

    print 'entropy(arr_int16)     = %.6f' % entropy(arr_int16)
    print 'entropy_v1(arr_int16)  = %.6f' % entropy_v1(arr_int16)
    print 'entropy_cpo(arr_int16) = %.6f' % entropy_cpo(arr_int16)

See
    - :py:class:`Utils`
    - :py:class:`NDArrUtils`
    - :py:class:`Graphics`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created by Mikhail Dubrovin
"""
#------------------------------
import numpy as np
#------------------------------

def hist_values(nda) :
    """Depending on nda.dtype fills/returns 1-D 2^8(16)-bin histogram-array of 8(16)-bit values of input n-d array
    """
    #print '%s for array dtype=%s'%(FR().f_code.co_name, str(nda.dtype))

    if nda.dtype == np.uint8 :
        return np.bincount(nda.flatten(), weights=None, minlength=1<<8)

    elif nda.dtype == np.uint16 :
        return np.bincount(nda.flatten(), weights=None, minlength=1<<16)

    elif nda.dtype == np.int16 :
        unda = nda.astype(np.uint16) # int16 (1,2,-3,0,4,-5,...) -> uint16 (1,2,0,65533,4,65531,...)        
        return np.bincount(unda.flatten(), weights=None, minlength=1<<16)

    else : 
        sys.exit('method %s get unexpected nda dtype=%s. Use np.uint8 or np.(u)int16'%(FR().f_code.co_name, str(nda.dtype)))

#------------------------------

def hist_probabilities(nda) :
    """Returns histogram-array of probabilities for each of (u)int16, uint8 intensity
    """
    #print('%s for array dtype=%s'%(FR().f_code.co_name, str(nda.dtype)))

    nvals = nda.size
    ph = np.array(hist_values(nda), dtype=np.float)
    ph /= nvals
    #print('Check sum of probabilities: %.6f for number of values in array = %d' % (ph.sum(), nvals))
    return ph

#------------------------------

def entropy(nda) :
    """Evaluates n-d array entropy using formula from https://en.wikipedia.org/wiki/Entropy_%28information_theory%29
    """ 
    unda = None # histogram array indexes must be unsigned
    if   nda.dtype == np.uint8 : unda = nda
    elif nda.dtype == np.uint16: unda = nda
    elif nda.dtype == np.int16 : unda = nda.astype(np.uint16) # int16 (1,2,-3,0,4,-5,...) -> uint16 (1,2,0,65533,4,65531,...)        

    prob_h = hist_probabilities(unda)

    p_log2p_nda = [p*np.log2(p) for p in prob_h if p>0]
    ent = -np.sum(p_log2p_nda)

    #print_ndarr(hist_values(nda), name='Histogram of uint16 values', first=1500, last=1520)
    #print_ndarr(prob_h, name='Histogram of probabilities', first=1500, last=1520)
    #print_ndarr(prob_nda, name='per pixel array of probabilities\n', first=1000, last=1010)
    #print_ndarr(p_log2p_nda, name='per pixel array of P*log2(P)\n', first=1000, last=1010)
    return ent

#------------------------------
## formula in https://en.wikipedia.org/wiki/Entropy_%28information_theory%29
## sums over all (x_i) which is a set of possible values....
## this method sums over set (one entry) of probabilities 
#------------------------------

def entropy_v1(nda) :
    """The same as entropy(nda) in a single place.
    """
    #print('%s for array dtype=%s'%(FR().f_code.co_name, str(nda.dtype)))

    unda = nda 
    if   nda.dtype == np.uint8  : unda = nda
    elif nda.dtype == np.uint16 : unda = nda
    elif nda.dtype == np.int16  : unda = nda.astype(np.uint16) # int16 (1,2,-3,0,4,-5,...) -> uint16 (1,2,0,65533,4,65531,...)        
    else : sys.exit('method %s get unexpected nda dtype=%s. Use np.uint8 or np.(u)int16'%(FR().f_code.co_name, str(nda.dtype)))

    hsize = (1<<8) if nda.dtype == np.uint8 else (1<<16)
    vals_h = np.bincount(unda.flatten(), weights=None, minlength=hsize)
    prob_h = np.array(vals_h, dtype=np.float) / unda.size
    #prob_nda = prob_h[unda]    
    #p_log2p_nda = prob_nda * np.log2(prob_nda)
    #ent = -p_log2p_nda.sum()
    p_log2p_nda = [p*np.log2(p) for p in prob_h if p>0]
    ent = -np.sum(p_log2p_nda)
    return ent

#------------------------------

def entropy_cpo(signal):
    '''Entropy evaluation method found by cpo on web

    Function returns entropy of a signal, which is 1-D numpy array
    '''
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

#------------------------------
#------------------------------
#------------------------------

def test_entropy():

    print('In %s' % sys._getframe().f_code.co_name)

    from psana.pyalgos.generic.NDArrGenerators import random_standard
    from psana.pyalgos.generic.NDArrUtils import print_ndarr
    from time import time

    arr_float = random_standard(shape=(100000,), mu=200, sigma=25, dtype=np.float)
    arr_int16 = arr_float.astype(np.int16)
    
    print_ndarr(arr_int16, name='arr_int16', first=0, last=10)

    t0_sec = time()
    ent1 = entropy(arr_int16);     t1_sec = time()
    ent2 = entropy_v1(arr_int16);  t2_sec = time()
    ent3 = entropy_cpo(arr_int16); t3_sec = time()

    print('entropy(arr_int16)     = %.6f, time=%.6f sec' % (ent1, t1_sec-t0_sec))
    print('entropy_v1(arr_int16)  = %.6f, time=%.6f sec' % (ent2, t2_sec-t1_sec))
    print('entropy_cpo(arr_int16) = %.6f, time=%.6f sec' % (ent3, t3_sec-t2_sec))

#------------------------------

if __name__ == "__main__" :
    import sys; global sys

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    if   tname == '0': test_entropy()
    elif tname == '1': test_entropy()
    else : sys.exit('Test %s is not implemented' % tname)
    sys.exit('End of Test %s' % tname)

#------------------------------

