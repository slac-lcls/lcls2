#!@PYTHON@
####!/usr/bin/env python
#------------------------------
"""
Class :py:class:`HSpectrum` works with spectral histogram for arbitrary shaped numpy array
==========================================================================================
Usage::

    # Import
    # ==============
    from pyimgalgos.HSpectrum import HSpectrum

    # Initialization
    # ==============
    # 1) for bins of equal size:
    range = (vmin, vmax)
    nbins = 100
    spec = HSpectrum(range, nbins)

    # 2) for variable size bins:
    bins = (v0, v1, v2, v4, v5, vN) # any number of bin edges
    spec = HSpectrum(bins)

    # Fill spectrum
    # ==============
    # nda = ... (get it for each event somehow)
    spec.fill(nda)

    # Get spectrum
    # ==============
    histarr, edges, nbins = spec.spectrum()

    # Optional
    # ==============
    spec.print_attrs()

See:
  - :py:class:`HBins`
  - :py:class:`HPolar`
  - :py:class:`HSpectrum`
  - :py:class:`NDArrSpectrum`
  - :py:class:`RadialBkgd`
  - `Radial background <https://confluence.slac.stanford.edu/display/PSDMInternal/Radial+background+subtraction+algorithm>`_.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created in 2015 by Mikhail Dubrovin
"""
#------------------------------

import sys
import numpy as np
from pyimgalgos.HBins import HBins

#------------------------------

class HSpectrum :
    def __init__(self, edges, nbins=None, pbits=0) :
        """ Constructor
        - edges - sequence of bin edges
        - nbins - number of bins in spectrum, if None - edges are used
        - pbits - print control bits; =0 - print nothing, 1 - object attributes.
        """
        self.hbins = HBins(edges, nbins, vtype=np.float32)
        self.pbits = pbits
        self.is_inited = False
        if self.pbits : self.print_attrs()


    def print_attrs(self) :
        """ Prints object essential attributes
        """
        hb = self.hbins
        print 'Class %s object attributes:' % (self.__class__.__name__)
        print 'Binning mode: %s, where True/False for equal/variable size bins' % (hb.equalbins())
        print 'Number of bins: %d' % hb.nbins()
        print 'Bin edges: %s' % str(hb.edges())
        print 'vmin = %f\nvmax = %f' % (hb.vmin(), hb.vmax())
        print 'pbits: %d' % (self.pbits)
        #self.hbins.print_attrs()
        #self.hbins.print_attrs_defined()


    def init_spectrum(self, nda) :
        """ Initialization of the spectral histogram array at 1-st entrance in fill(nda)
            - nda - numpy n-d array with intensities for spectral histogram.
        """         
        self.ashape = nda.shape
        self.asize  = nda.size
        self.hshape = (self.asize, self.hbins.nbins())
        self.histarr = np.zeros(self.hshape, dtype=np.uint16) # huge size array
        self.pix_inds = np.array(range(self.asize), dtype=np.uint32)
        if self.pbits & 1 :
            print 'n-d array shape = %s, size = %d, dtype = %s' % (str(self.ashape), self.asize, str(nda.dtype))
            print 'histogram shape = %s, size = %d, dtype = %s' % (str(self.hshape), self.histarr.size, str(self.histarr.dtype))
        self.is_inited = True


    def fill(self, nda) :
        """ Fills n-d array spectrum histogram-array
            - nda - numpy n-d array with intensities for spectral histogram.
        """         
        if not self.is_inited : self.init_spectrum(nda)

        shape_in = nda.shape
        if len(shape_in) > 1 : nda.shape = (self.asize,) # reshape to 1-d

        bin_inds = self.hbins.bin_indexes(nda, edgemode=0)
        self.histarr[self.pix_inds, bin_inds] += 1

        if len(shape_in) > 1 : nda.shape = shape_in # return original shape


    def spectrum(self) :
        """ Returns accumulated n-d array spectrum, histogram bin edges, and number of bins
        """ 
        return self.histarr, self.hbins.edges(), self.hbins.nbins()

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------

def random_standard_array(shape=(185,388), mu=50, sigma=10) :
    """Returns n-d array of specified shape with random intensities generated for Gaussian parameters.
    """
    return mu + sigma*np.random.standard_normal(shape)

#------------------------------

def example_equidistant() :
    print """Test HSpectrum for equidistant bins"""

    from time import time

    vmin, vmax, nbins = 0, 100, 50 # binning parameters
    mu, sigma = 50, 10             # parameters of random Gaussian distribution of intensities
    nevts = 10                     # number of events in this test
    ashape = (32,185,388)          # data array shape

    spec = HSpectrum((vmin, vmax), nbins, pbits=0377)

    for ev in range(nevts) :
      arr = random_standard_array(ashape, mu, sigma)
      t0_sec = time()
      spec.fill(arr)
      print 'Event:%3d, t = %10.6f sec' % (ev, time()-t0_sec)


    if True :
      import pyimgalgos.GlobalGraphics as gg
      
      histarr, edges, nbins = spec.spectrum()
      #gg.plotImageLarge(arr, amp_range=(vmin,vmax), title='random')
      gg.plotImageLarge(histarr[0:500,:], amp_range=(0,nevts/3), title='indexes')
      gg.show()

#------------------------------

def example_varsize() :
    print """Test HSpectrum for variable size bins"""

    from time import time

    edges = (0, 30, 40, 50, 60, 70, 100) # array of bin edges
    mu, sigma = 50, 10                   # parameters of random Gaussian distribution of intensities
    nevts = 10                           # number of events in this test
    ashape = (32,185,388)                # data array shape

    spec = HSpectrum(edges, pbits=0377)

    for ev in range(nevts) :
      arr = random_standard_array(ashape, mu, sigma)
      t0_sec = time()
      spec.fill(arr)
      print 'Event:%3d, t = %10.6f sec' % (ev, time()-t0_sec)


    if True :
      import pyimgalgos.GlobalGraphics as gg

      histarr, edges, nbins = spec.spectrum()
      #gg.plotImageLarge(arr, amp_range=(vmin,vmax), title='random')
      gg.plotImageLarge(histarr[0:500,:], amp_range=(0,nevts/3), title='indexes')
      gg.show()

#------------------------------

def usage() : return 'Use command: python %s <test-number [1-2]>' % sys.argv[0]

def main() :    
    print '\n%s\n' % usage()
    if len(sys.argv) != 2 : example_equidistant()
    elif sys.argv[1] == '1' : example_equidistant()
    elif sys.argv[1] == '2' : example_varsize()
    else                  : sys.exit ('Test number parameter is not recognized.\n%s' % usage())

#------------------------------

if __name__ == "__main__" :
    main()
    sys.exit('End of test')

#------------------------------
