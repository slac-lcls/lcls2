#!/usr/bin/env python

#------------------------------
""":py:class:`RadialBkgd` - radial background subtraction for imaging detector n-d array data,
   extension of the base class :py:class:`HPolar` for methods subtract_bkgd and subtract_bkgd_interpol.

Usage::

    # Import
    # ------
    from pyimgalgos.RadialBkgd import RadialBkgd

    # Initialization
    # --------------
    rb = RadialBkgd(xarr, yarr, mask=None, radedges=None, nradbins=100, phiedges=(0,360), nphibins=32)

    # Access methods
    # --------------
    orb   = rb.obj_radbins() # returns HBins object for radial bins
    opb   = rb.obj_phibins() # returns HBins object for angular bins
    rad   = rb.pixel_rad()
    irad  = rb.pixel_irad()
    phi0  = rb.pixel_phi0()
    phi   = rb.pixel_phi()
    iphi  = rb.pixel_iphi()
    iseq  = rb.pixel_iseq()
    npix  = rb.bin_number_of_pixels()
    int   = rb.bin_intensity(nda)
    arr1d = rb.bin_avrg(nda)
    arr2d = rb.bin_avrg_rad_phi(nda, do_transp=True)
    pixav = rb.pixel_avrg(nda)
    pixav = rb.pixel_avrg_interpol(nda, method='linear') # method='nearest' 'cubic'
    cdata = rb.subtract_bkgd(nda)
    cdata = rb.subtract_bkgd_interpol(nda, method='linear')

    # Print attributes and n-d arrays
    # -------------------------------
    rb.print_attrs()
    rb.print_ndarrs()

    # Global methods
    # --------------
    from pyimgalgos.RadialBkgd import polarization_factor, divide_protected, cart2polar, polar2cart, bincount

    polf = polarization_factor(rad, phi, z)
    result = divide_protected(num, den, vsub_zero=0)
    r, theta = cart2polar(x, y)
    x, y = polar2cart(r, theta)
    bin_values = bincount(map_bins, map_weights=None, length=None)

@see :py:class:`pyimgalgos.HPolar`
:py:class:`pyimgalgos.HBins`
See `Radial background <https://confluence.slac.stanford.edu/display/PSDMInternal/Radial+background+subtraction+algorithm>`_.
:py:class:`pyimgalgos.HSpectrum`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Revision: $Revision$

@version $Id$

@author Mikhail S. Dubrovin

"""
#--------------------------------
__version__ = "$Revision$"
#--------------------------------

#import math
#import numpy as np
#from pyimgalgos.HBins import HBins
from pyimgalgos.HPolar import HPolar, polarization_factor, divide_protected, cart2polar, polar2cart, bincount

#------------------------------

class RadialBkgd(HPolar) :
    def __init__(self, xarr, yarr, mask=None, radedges=None, nradbins=100, phiedges=(0,360), nphibins=32) :
        """Parameters
           - mask     - n-d array with mask
           - xarr     - n-d array with pixel x coordinates in any units
           - yarr     - n-d array with pixel y coordinates in the same units as xarr
           - radedges - radial bin edges for corrected region in the same units of xarr;
                        default=None - all radial range
           - nradbins - number of radial bins
           - phiedges - phi ange bin edges for corrected region.
                        default=(0,360)
                        Difference of the edge limits should not exceed +/-360 degree 
           - nphibins - number of angular bins
                        default=32 - bin size equal to 1 rhumb for default phiedges
        """
        HPolar.__init__(self, xarr, yarr, mask, radedges, nradbins, phiedges, nphibins)


    def subtract_bkgd(self, ndarr) :
        """Returns 1-d numpy array of per-pixel background subtracted input data."""
        shape = ndarr.shape
        nda = ndarr.flatten()
        nda -= self.pixel_avrg(nda)
        nda.shape = shape
        return nda


    def subtract_bkgd_interpol(self, ndarr, method='linear', verb=False) :
        """Returns 1-d numpy array of per-pixel interpolated-background subtracted input data."""
        shape = ndarr.shape
        nda = ndarr.flatten()
        nda -= self.pixel_avrg_interpol(nda, method, verb)
        nda.shape = shape
        return nda

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------

def data_geo(ntest) :
    """Returns test data numpy array and geometry object
    """
    from time import time
    from PSCalib.NDArrIO import save_txt, load_txt
    from PSCalib.GeometryAccess import GeometryAccess

    dir       = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-camera2-2016-02-05'
    #fname_nda = '%s/nda-water-ring-cxij4716-r0022-e000001-CxiDs2-0-Cspad-0-ave.txt' % dir
    fname_nda = '%s/nda-water-ring-cxij4716-r0022-e014636-CxiDs2-0-Cspad-0-ave.txt' % dir
    fname_geo = '%s/calib/CsPad::CalibV1/CxiDs2.0:Cspad.0/geometry/geo-cxi01516-2016-02-18-Ag-behenate-tuned.data' % dir
    #fname_geo = '%s/geo-cxi02416-r0010-2016-03-11.txt' % dir
    fname_gain = '%s/calib/CsPad::CalibV1/CxiDs2.0:Cspad.0/pixel_gain/cxi01516-r0016-2016-02-18-FeKalpha.data' % dir

    # load n-d array with averaged water ring
    arr = load_txt(fname_nda)
    #arr *= load_txt(fname_gain)
    #print_ndarr(arr,'water ring')
    arr.shape = (arr.size,) # (32*185*388,)

    # retrieve geometry
    t0_sec = time()
    geo = GeometryAccess(fname_geo)
    geo.move_geo('CSPAD:V1', 0, 1600, 0, 0)
    geo.move_geo('QUAD:V1', 2, -100, 0, 0)
    #geo.get_geo('QUAD:V1', 3).print_geo()
    print 'Time to load geometry %.3f sec from file\n%s' % (time()-t0_sec, fname_geo)

    return arr, geo

#------------------------------

def test01(ntest, prefix='fig-v01') :
    """Test for radial 1-d binning of entire image.
    """
    from time import time
    import pyimgalgos.GlobalGraphics as gg
    from PSCalib.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    t0_sec = time()
    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0377).flatten() 
    print 'Time to retrieve geometry %.3f sec' % (time()-t0_sec)

    t0_sec = time()
    rb = RadialBkgd(X, Y, mask, nradbins=500, nphibins=1) # v1
    print 'RadialBkgd initialization time %.3f sec' % (time()-t0_sec)

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 1 : nda, title = arr,                   'averaged data'
    elif ntest == 2 : nda, title = rb.pixel_rad(),        'pixel radius value'
    elif ntest == 3 : nda, title = rb.pixel_phi(),        'pixel phi value'
    elif ntest == 4 : nda, title = rb.pixel_irad() + 2,   'pixel radial bin index' 
    elif ntest == 5 : nda, title = rb.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 6 : nda, title = rb.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    elif ntest == 7 : nda, title = mask,                  'mask'
    elif ntest == 8 : nda, title = rb.pixel_avrg(nda),    'averaged radial background'
    elif ntest == 9 : nda, title = rb.subtract_bkgd(nda) * mask, 'background-subtracted data'

    else :
        t1_sec = time()
        pf = polarization_factor(rb.pixel_rad(), rb.pixel_phi(), 94e3) # Z=94mm
        print 'Time to evaluate polarization correction factor %.3f sec' % (time()-t1_sec)

        if   ntest == 10 : nda, title = pf,                    'polarization factor'
        elif ntest == 11 : nda, title = arr * pf,              'polarization-corrected averaged data'
        elif ntest == 12 : nda, title = rb.subtract_bkgd(arr * pf) * mask , 'polarization-corrected background subtracted data'
        elif ntest == 13 : nda, title = rb.pixel_avrg(arr * pf), 'polarization-corrected averaged radial background'
        elif ntest == 14 : nda, title = rb.pixel_avrg_interpol(arr * pf) * mask , 'polarization-corrected interpolated radial background'
        elif ntest == 15 : nda, title = rb.subtract_bkgd_interpol(arr * pf) * mask , 'polarization-corrected interpolated radial background-subtracted data'


        else :
            print 'Test %d is not implemented' % ntest 
            return
        
    print 'Get %s n-d array time %.3f sec' % (title, time()-t0_sec)

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (21,) else nda[100:300,:]

    da, ds = None, None
    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter'
    if ntest in (2,3,4,5,6,7) :
        da = ds = (nda.min()-1., nda.max()+1.)

    if ntest in (12,15) :
        ds = da = (-20, 20)
        colmap = 'gray'

    else :
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print 'End of test for %s' % title    

#------------------------------

def test02(ntest, prefix='fig-v01') :
    """Test for 2-d (default) binning of the rad-phi range of entire image
    """
    #from Detector.GlobalUtils import print_ndarr
    from time import time
    import pyimgalgos.GlobalGraphics as gg
    from PSCalib.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0377).flatten() 

    t0_sec = time()
    rb = RadialBkgd(X, Y, mask) # v0
    #rb = RadialBkgd(X, Y, mask, nradbins=500) # , nphibins=8, phiedges=(-20, 240), radedges=(10000,80000))
    print 'RadialBkgd initialization time %.3f sec' % (time()-t0_sec)

    #print 'npixels_per_bin:',   rb.npixels_per_bin()
    #print 'intensity_per_bin:', rb.intensity_per_bin(arr)
    #print 'average_per_bin:',   rb.average_per_bin(arr)

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 21 : nda, title = arr,                   'averaged data'
    elif ntest == 22 : nda, title = rb.pixel_rad(),        'pixel radius value'
    elif ntest == 23 : nda, title = rb.pixel_phi(),        'pixel phi value'
    elif ntest == 24 : nda, title = rb.pixel_irad() + 2,   'pixel radial bin index' 
    elif ntest == 25 : nda, title = rb.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 26 : nda, title = rb.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    elif ntest == 27 : nda, title = mask,                  'mask'
    elif ntest == 28 : nda, title = rb.pixel_avrg(nda),      'averaged radial background'
    elif ntest == 29 : nda, title = rb.subtract_bkgd(nda) * mask, 'background-subtracted data'
    elif ntest == 30 : nda, title = rb.bin_avrg_rad_phi(nda),'r-phi'
    elif ntest == 31 : nda, title = rb.pixel_avrg_interpol(nda), 'averaged radial interpolated background'
    elif ntest == 32 : nda, title = rb.subtract_bkgd_interpol(nda, method='linear', verb=True) * mask, 'interpol-background-subtracted data'
    else :
        print 'Test %d is not implemented' % ntest 
        return

    print 'Get %s n-d array time %.3f sec' % (title, time()-t0_sec)

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (30,) else nda # [100:300,:]

    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter' 'gray'

    da = (nda.min()-1, nda.max()+1)
    ds = da

    if ntest in (21,28,29,30,31) :
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    elif ntest in (32,) : 
        colmap = 'gray'
        ds = da = (-20, 20)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print 'End of test for %s' % title    

#------------------------------

def test03(ntest, prefix='fig-v01') :
    """Test for 2-d binning of the restricted rad-phi range of entire image
    """
    from time import time
    import pyimgalgos.GlobalGraphics as gg
    from PSCalib.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0377).flatten() 

    t0_sec = time()

    rb = RadialBkgd(X, Y, mask, nradbins=200, nphibins=32, phiedges=(-20, 240), radedges=(10000,80000)) if ntest in (51,52)\
    else RadialBkgd(X, Y, mask, nradbins=  5, nphibins= 8, phiedges=(-20, 240), radedges=(10000,80000))
    #rb = RadialBkgd(X, Y, mask, nradbins=3, nphibins=8, phiedges=(240, -20), radedges=(80000,10000)) # v3

    print 'RadialBkgd initialization time %.3f sec' % (time()-t0_sec)

    #print 'npixels_per_bin:',   rb.npixels_per_bin()
    #print 'intensity_per_bin:', rb.intensity_per_bin(arr)
    #print 'average_per_bin:',   rb.average_per_bin(arr)

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 41 : nda, title = arr,                   'averaged data'
    elif ntest == 42 : nda, title = rb.pixel_rad(),        'pixel radius value'
    elif ntest == 43 : nda, title = rb.pixel_phi(),        'pixel phi value'
    elif ntest == 44 : nda, title = rb.pixel_irad() + 2,   'pixel radial bin index' 
    elif ntest == 45 : nda, title = rb.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 46 : nda, title = rb.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    elif ntest == 47 : nda, title = mask,                  'mask'
    elif ntest == 48 : nda, title = rb.pixel_avrg(nda),      'averaged radial background'
    elif ntest == 49 : nda, title = rb.subtract_bkgd(nda) * mask, 'background-subtracted data'
    elif ntest == 50 : nda, title = rb.bin_avrg_rad_phi(nda),'r-phi'
    elif ntest == 51 : nda, title = rb.pixel_avrg_interpol(nda), 'averaged radial interpolated background'
    elif ntest == 52 : nda, title = rb.subtract_bkgd_interpol(nda) * mask, 'interpol-background-subtracted data'
    else :
        print 'Test %d is not implemented' % ntest 
        return

    print 'Get %s n-d array time %.3f sec' % (title, time()-t0_sec)

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (50,) else nda # [100:300,:]

    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter' 'gray'

    da = (nda.min()-1, nda.max()+1)
    ds = da

    if ntest in (41,48,49,50,51) :
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    elif ntest in (52,) : 
        colmap = 'gray'
        ds = da = (-20, 20)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print 'End of test for %s' % title    

#------------------------------

if __name__ == '__main__' :
    import sys
    ntest = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print 'Test # %d' % ntest

    prefix = 'fig-v01-cspad-RadialBkgd'

    if   ntest < 20 : test01(ntest, prefix)
    elif ntest < 40 : test02(ntest, prefix)
    elif ntest < 60 : test03(ntest, prefix)
    else : print 'Test %d is not implemented' % ntest     
    #sys.exit('End of test')
 
#------------------------------
#------------------------------
