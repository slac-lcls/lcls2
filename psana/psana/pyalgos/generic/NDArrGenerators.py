####!/usr/bin/env python
#------------------------------
"""
:py:class:`NDArrGenerators` wrapping methods for numpy random array generators
==============================================================================

Usage::

    import psana.pyalgos.generic.NDArrGenerators as ag

    # Methods

    nda = ag.random_standard(shape=(40,60), mu=200, sigma=25, dtype=np.float)
    nda = ag.random_exponential(shape=(40,60), a0=100, dtype=np.float)
    nda = ag.random_one(shape=(40,60), dtype=np.float)
    nda = ag.random_256(shape=(40,60), dtype=np.uint8)
    nda = ag.random_xffffffff(shape=(40,60), dtype=np.uint32, add=0xff000000)
    nda = ag.aranged_array(shape=(40,60), dtype=np.uint32)
    ag.print_ndarr(nda, name='', first=0, last=5)
    nda = ag.ring_intensity(r, r0, sigma)
    ag.add_ring(arr2d, amp=100, row=4.3, col=5.8, rad=100, sigma=3)
    peaks = ag.add_random_peaks(arr2d, npeaks=10, amean=100, arms=50, wmean=2, wrms=0.1)

See:
  - :py:class:`graphics`
  - :py:class:`NDArrUtils`
  - :py:class:`NDArrGenerators`
  - `numpy.random.rand <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.rand.html>`_.
  - `matplotlib <https://matplotlib.org/contents.html>`_.

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Modified for LCLS2 on 2015-01-26 by Mikhail Dubrovin
"""
#-----------------------------

import numpy as np
import math
#from psana.pyalgos.generic.NDArrUtils import shape_as_2d, shape_as_3d, reshape_to_2d, reshape_to_3d
#from psana.pyalgos.generic.NDArrUtils import size_from_shape
#from psana.pyalgos.generic.NDArrUtils import print_ndarr

#-----------------------------

def set_random_state(seed=1234567) :
    """seed : {None, int, array_like} Can be any integer between 0 and 2**32 - 1"""
    #return np.random.RandomState(seed)
    np.random.seed(seed)

#-----------------------------

def random_standard(shape=(40,60), mu=200, sigma=25, dtype=np.float) :
    """Returns numpy array of requested shape and type filled with normal distribution for mu and sigma.
    """
    a = mu + sigma*np.random.standard_normal(size=shape)
    return np.require(a, dtype)

#-----------------------------

def random_exponential(shape=(40,60), a0=100, dtype=np.float) :
    """Returns numpy array of requested shape and type filled with exponential distribution for width a0.
    """
    a = a0*np.random.standard_exponential(size=shape)
    return np.require(a, dtype) 

#-----------------------------

def random_one(shape=(40,60), dtype=np.float) :
    """Returns numpy array of requested shape and type filled with random numbers in the range [0,1].
    """
    a = np.random.random(shape)
    return np.require(a, dtype) 

#-----------------------------

random_1 = random_one

#-----------------------------

def random_256(shape=(40,60), dtype=np.uint8) :
    """Returns numpy array of requested shape and type filled with random numbers in the range [0,255].
    """
    a = 255*np.random.random(shape)
    return np.require(a, dtype) 

#-----------------------------

def random_xffffffff(shape=(40,60), dtype=np.uint32, add=0xff000000) :
    """Returns numpy array of requested shape and type 
       filled with random numbers in the range [0,0xffffff] with bits 0xff000000 for alpha mask.  
    """
    a = 0xffffff*np.random.random(shape) + add
    return np.require(a, dtype)

#------------------------------

def size_from_shape(shape) :
    """Returns size from the shape sequence 
    """
    size=1
    for d in shape : size*=d
    return size

#-----------------------------

def aranged_array(shape=(40,60), dtype=np.uint32) :
    """Returns numpy array of requested shape and type filling with ascending integer numbers.
    """
    arr = np.arange(size_from_shape(shape), dtype=dtype)
    arr.shape = shape
    return arr

#-----------------------------

def ring_intensity(r, r0, sigma) :
    """returns numpy array with ring intensity distribution modulated by Gaussian(r-r0,sigma).
       Parameters
       ----------
       r : np.array - numpy array of radius (i.e. radios for each pixel) 
       r0 : float - radius of the ring
       sigma : float - width of the ring
    """
    factor = 1/ (math.sqrt(2) * sigma)
    rr = factor*(r-r0)
    return np.exp(-rr*rr)

#-----------------------------

def add_ring(arr2d, amp=100, row=4.3, col=5.8, rad=100, sigma=3) :
    """Adds peak Gaussian-shaped peak intensity to numpy array arr2d
       Parameters
       ----------
       arr2d : np.array - 2-d numpy array 
       amp : float - ring intensity
       row : float - ring center row
       col : float - ring center col
       rad : float - ring mean radius
       sigma : float - width of the peak
    """
    nsigma = 5
    rmin = max(int(math.floor(row - rad - nsigma*sigma)), 0)
    cmin = max(int(math.floor(col - rad - nsigma*sigma)), 0)
    rmax = min(int(math.floor(row + rad + nsigma*sigma)), arr2d.shape[0])
    cmax = min(int(math.floor(col + rad + nsigma*sigma)), arr2d.shape[1])
    r = np.arange(rmin, rmax, 1, dtype = np.float32) - row
    c = np.arange(cmin, cmax, 1, dtype = np.float32) - col
    CG, RG = np.meshgrid(c, r)
    R = np.sqrt(RG*RG+CG*CG)
    arr2d[rmin:rmax,cmin:cmax] += amp * ring_intensity(R, rad, sigma)

#-----------------------------

def add_random_peaks(arr2d, npeaks=10, amean=100, arms=50, wmean=2, wrms=0.1) :
    """Returns 2-d array with peaks.
    """
    shape=arr2d.shape

    rand_uni = random_1(shape=(2, npeaks))
    r0 = rand_uni[0,:]*shape[0]
    c0 = rand_uni[1,:]*shape[1]
    rand_std = random_standard(shape=(4,npeaks), mu=0, sigma=1)
    a0    = amean + arms*rand_std[0,:] 
    sigma = wmean + wrms*rand_std[0,:] 

    peaks = zip(r0, c0, a0, sigma)

    for r0, c0, a0, sigma in peaks :
        add_ring(arr2d, amp=a0, row=r0, col=c0, rad=0, sigma=sigma)

    return peaks

#-----------------------------

def cspad2x1_arr(dtype=np.float32) :
    """returns test np.array for cspad 2x1 with linear variation of intensity from corner (0,0) to (rmax,cmax)"""
    rows, cols = 185, 388
    row2x1 = np.arange(cols)
    col2x1 = np.arange(rows)
    iY, iX = np.meshgrid(row2x1, col2x1)
    arr2x1 = np.empty((rows,cols), dtype)
    arr2x1[iX,iY] = iX+iY
    return arr2x1

def cspad_ndarr(n2x1=32, dtype=np.float32) :
    """returns test np.array for cspad with linear variation of intensity in 2x1s"""
    arr2x1 = cspad2x1_arr(dtype)
    rows, cols = arr2x1.shape
    arr = np.vstack([arr2x1 for seg in range(n2x1)])
    arr.shape = [n2x1, rows, cols]
    return arr
 
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

if __name__ == '__main__':

    from psana.pyalgos.generic.NDArrUtils import print_ndarr

    set_random_state()

    print_ndarr(random_exponential(), 'random_exponential')
    print_ndarr(random_standard(), 'random_standard')
    print_ndarr(random_1(), 'random_1', last=10)
    print_ndarr(random_256(), 'random_256', last=10)
    print_ndarr(random_xffffffff(), 'random_xffffffff')
    print_ndarr(random_standard(), 'random_standard')
    print_ndarr(aranged_array(), 'aranged_array')
    #print_ndarr(, '')
    print('Test is completed')

#-----------------------------
