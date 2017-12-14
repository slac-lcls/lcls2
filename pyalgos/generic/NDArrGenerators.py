#!@PYTHON@
####!/usr/bin/env python
#------------------------------
"""
:py:class:`NDArrGenerators` wrapping methods for numpy random array generators
==============================================================================

Usage::

    import pyimgalgos.NDArrGenerators as ag

    # Methods

    v = ag.prod_of_elements(arr, dtype=np.int)
    size = ag.size_from_shape() # use nda.size()

    shape = ag.shape_as_2d(sh)
    shape = ag.shape_as_3d(sh)
    arr2d = ag.reshape_to_2d(nda)
    arr3d = ag.reshape_to_3d(nda)
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
  - :py:class:`Graphics`
  - :py:class:`GlobalGraphics`
  - :py:class:`NDArrGenerators`
  - `numpy.random.rand <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.rand.html>`_.
  - `matplotlib <https://matplotlib.org/contents.html>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on Nov 23, 2015 by Mikhail Dubrovin
"""
import numpy as np
import math

#-----------------------------

def prod_of_elements(arr, dtype=np.int) :
    """Returns product of sequence elements
    """
    return np.prod(arr,axis=None,dtype=dtype)

#-----------------------------

def size_from_shape(shape) :
    """Returns size from the shape sequence 
    """
    return prod_of_elements(shape)

#-----------------------------

def shape_as_2d(sh) :
    """Returns 2-d shape for n-d shape if n>2, otherwise returns unchanged shape.
    """
    if len(sh)<3 : return sh
    return (size_from_shape(sh)/sh[-1], sh[-1])

#-----------------------------

def shape_as_3d(sh) :
    """Returns 3-d shape for n-d shape if n>3, otherwise returns unchanged shape.
    """
    if len(sh)<4 : return sh
    return (size_from_shape(sh)/sh[-1]/sh[-2], sh[-2], sh[-1])

#-----------------------------

def reshape_to_2d(arr) :
    """Returns n-d re-shaped to 2-d
    """
    arr.shape = shape_as_2d(arr.shape)
    return arr

#-----------------------------

def reshape_to_3d(arr) :
    """Returns n-d re-shaped to 3-d
    """
    arr.shape = shape_as_3d(arr.shape)
    return arr

#-----------------------------

def random_standard(shape=(40,60), mu=200, sigma=25, dtype=np.float) :
    """Returns numpy array of requested shape and type filled with normal distribution for mu and sigma.
    """
    a = mu + sigma*np.random.standard_normal(shape)
    return np.require(a, dtype)

#-----------------------------

def random_exponential(shape=(40,60), a0=100, dtype=np.float) :
    """Returns numpy array of requested shape and type filled with exponential distribution for width a0.
    """
    a = a0*np.random.standard_exponential(size=shape)
    return np.require(a, dtype) 

#-----------------------------

def random_one(shape=(40,60), dtype=np.float) :
    """Returns numpy array of requested shape and type filled with random numbers in the range [0,255].
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

#-----------------------------

def aranged_array(shape=(40,60), dtype=np.uint32) :
    """Returns numpy array of requested shape and type filling with ascending integer numbers.
    """
    arr = np.arange(size_from_shape(shape), dtype=dtype)
    arr.shape = shape
    return arr

#-----------------------------

def print_ndarr(nda, name='', first=0, last=5) :
    """Prints array attributes, title, and a few elements in a single line. 
    """    
    if nda is None : print '%s: %s' % (name, nda)
    elif isinstance(nda, tuple) : print_ndarr(np.array(nda), 'ndarray from tuple: %s' % name)
    elif isinstance(nda, list)  : print_ndarr(np.array(nda), 'ndarray from list: %s' % name)
    elif not isinstance(nda, np.ndarray) : print '%s: %s' % (name, type(nda))
    else: print '%s:  shape:%s  size:%d  dtype:%s %s...' % \
         (name, str(nda.shape), nda.size, nda.dtype, nda.flatten()[first:last])

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
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

if __name__ == '__main__':

    print_ndarr(random_exponential(), 'random_exponential')
    print_ndarr(random_standard(), 'random_standard')
    print_ndarr(random_1(), 'random_1', last=10)
    print_ndarr(random_256(), 'random_256', last=10)
    print_ndarr(random_xffffffff(), 'random_xffffffff')
    print_ndarr(random_standard(), 'random_standard')
    print_ndarr(aranged_array(), 'aranged_array')
    #print_ndarr(, '')
    print 'Test is completed'

#-----------------------------
