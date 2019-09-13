####!/usr/bin/env python
#------------------------------
"""
Class :py:class:`HPolar` - makes 2-d histogram in polar (r-phi) coordinates for imaging detector n-d array data
===============================================================================================================

Usage::

    # Import
    # ------
    from psana.pyalgos.generic.HPolar import HPolar

    # Initialization
    # --------------
    hp = HPolar(xarr, yarr, mask=None, radedges=None, nradbins=100, phiedges=(0,360), nphibins=32)

    # Access methods
    # --------------
    orb   = hp.obj_radbins() # returns HBins object for radial bins
    opb   = hp.obj_phibins() # returns HBins object for angular bins
    rad   = hp.pixel_rad()
    irad  = hp.pixel_irad()
    phi0  = hp.pixel_phi0()
    phi   = hp.pixel_phi()
    iphi  = hp.pixel_iphi()
    iseq  = hp.pixel_iseq()
    npix  = hp.bin_number_of_pixels()
    int   = hp.bin_intensity(nda)
    arr1d = hp.bin_avrg(nda)
    arr2d = hp.bin_avrg_rad_phi(nda, do_transp=True)
    pixav = hp.pixel_avrg(nda)
    pixav = hp.pixel_avrg_interpol(nda, method='linear') # method='nearest' 'cubic'

    # Print attributes and n-d arrays
    # -------------------------------
    hp.print_attrs()
    hp.print_ndarrs()

    # Global methods
    # --------------
    from psana.pyalgos.generic.HPolar import polarization_factor, divide_protected, cart2polar, polar2cart, bincount

    polf = polarization_factor(rad, phi, z)
    result = divide_protected(num, den, vsub_zero=0)
    r, theta = cart2polar(x, y)
    x, y = polar2cart(r, theta)
    bin_values = bincount(map_bins, map_weights=None, length=None)

See:
  - :py:class:`HBins`
  - :py:class:`HPolar`
  - :py:class:`HSpectrum`
  - :py:class:`RadialBkgd`
  - `Radial background <https://confluence.slac.stanford.edu/display/PSDMInternal/Radial+background+subtraction+algorithm>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created in 2015 by Mikhail Dubrovin
"""
#------------------------------

import math
import numpy as np
from psana.pyalgos.generic.HBins import HBins

from psana.pyalgos.generic.NDArrUtils import print_ndarr
#------------------------------

def divide_protected(num, den, vsub_zero=0) :
    """Returns result of devision of numpy arrays num/den with substitution of value vsub_zero for zero den elements.
    """
    pro_num = np.select((den!=0,), (num,), default=vsub_zero)
    pro_den = np.select((den!=0,), (den,), default=1)
    return pro_num / pro_den


def cart2polar(x, y) :
    """For numpy arrays x and y returns the numpy arrays of r and theta 
    """
    r = np.sqrt(x*x + y*y)
    theta = np.rad2deg(np.arctan2(y, x)) #[-180,180]
    #theta0 = np.select([theta<0, theta>=0],[theta+360,theta]) #[0,360]
    return r, theta


def polar2cart(r, theta) :
    """For numpy arryys r and theta returns the numpy arrays of x and y 
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def bincount(map_bins, map_weights=None, length=None) :
    """Wrapper for numpy.bincount with protection of weights and alattening numpy arrays
    """
    weights = None if map_weights is None else map_weights.flatten()
    return np.bincount(map_bins.flatten(), weights, length)


def polarization_factor(rad, phi_deg, z) :
    """Returns per-pixel polarization factors, assuming that detector is perpendicular to Z.
    """
    phi = np.deg2rad(phi_deg)
    ones = np.ones_like(rad)
    theta = np.arctan2(rad, z)
    sxc = np.sin(theta)*np.cos(phi)
    pol = 1 - sxc*sxc
    return divide_protected(ones, pol, vsub_zero=0)

#------------------------------

class HPolar() :
    def __init__(self, xarr, yarr, mask=None, radedges=None, nradbins=100, phiedges=(0,360), nphibins=32) :
        """Parameters
           - mask     - n-d array with mask
           - xarr     - n-d array with pixel x coordinates in any units
           - yarr     - n-d array with pixel y coordinates in the same units as xarr
           - radedges - radial bin edges for corrected region in the same units of xarr;
                        default=None - all radial range
           - nradbins - number of radial bins
           - phiedges - phi angle bin edges for corrected region.
                        default=(0,360)
                        Difference of the edge limits should not exceed +/-360 degree 
           - nphibins - number of angular bins
                        default=32 - bin size equal to 1 rhumb for default phiedges
        """
        self.rad, self.phi0 = cart2polar(xarr, yarr)
        self.shapeflat = (self.rad.size,)
        self.rad.shape = self.shapeflat
        self.phi0.shape = self.shapeflat
        self.mask = mask

        phimin = min(phiedges[0], phiedges[-1])

        self.phi = np.select((self.phi0<phimin, self.phi0>=phimin), (self.phi0+360.,self.phi0))

        self._set_rad_bins(radedges, nradbins)
        self._set_phi_bins(phiedges, nphibins)
        
        npbins = self.pb.nbins()
        nrbins = self.rb.nbins()
        self.ntbins = npbins*nrbins # total number of bins in r-phi array
        
        self.irad = self.rb.bin_indexes(self.rad, edgemode=1)  
        self.iphi = self.pb.bin_indexes(self.phi, edgemode=1)

        cond = np.logical_and(\
               np.logical_and(self.irad > -1, self.irad < nrbins),
               np.logical_and(self.iphi > -1, self.iphi < npbins)
               )

        if mask is not None : 
            cond = np.logical_and(cond, mask.astype(np.bool).flatten())

        # index ntbins stands for overflow bin
        self.iseq = np.select((cond,), (self.iphi*nrbins + self.irad,), self.ntbins).flatten() 

        #self.npix_per_bin = np.bincount(self.iseq, weights=None, minlength=None)
        self.npix_per_bin = np.bincount(self.iseq, weights=None, minlength=self.ntbins+1)

        self.griddata = None


    def _set_rad_bins(self, radedges, nradbins) :
        rmin = math.floor(np.amin(self.rad)) if radedges is None else radedges[0]
        rmax = math.ceil (np.amax(self.rad)) if radedges is None else radedges[-1]
        if rmin<1 : rmin = 1
        self.rb = HBins((rmin, rmax), nradbins)


    def _set_phi_bins(self, phiedges, nphibins) :
        if phiedges[-1] > phiedges[0]+360\
        or phiedges[-1] < phiedges[0]-360:
            raise ValueError('Difference between angular edges should not exceed 360 degree;'\
                             ' phiedges: %.0f, %.0f' % (phiedges[0], phiedges[-1]))        
        self.pb = HBins(phiedges, nphibins)
        phi1, phi2 = self.pb.limits()
        self.is360 = math.fabs(math.fabs(phi2-phi1)-360) < 1e-3


    def print_attrs(self) :
        print('%s attrbutes:' % self.__class__.__name__)
        print(self.pb.strrange(fmt='Phi bins:  min:%8.1f  max:%8.1f  nbins:%5d'))
        print(self.rb.strrange(fmt='Rad bins:  min:%8.1f  max:%8.1f  nbins:%5d'))


    def print_ndarrs(self) :
        print('%s n-d arrays:' % self.__class__.__name__)
        print_ndarr(self.rad, '  rad')
        print_ndarr(self.phi, '  phi')
        print_ndarr(self.mask,'  mask')
        #print('Phi limits: ', phiedges[0], phiedges[-1])


    def obj_radbins(self) :
        """Returns HBins object for radial bins."""
        return self.rb


    def obj_phibins(self) :
        """Returns HBins object for angular bins."""
        return self.pb


    def pixel_rad(self) :
        """Returns 1-d numpy array of pixel radial parameters."""
        return self.rad


    def pixel_irad(self) :
        """Returns 1-d numpy array of pixel radial indexes."""
        return self.irad


    def pixel_phi0(self) :
        """Returns 1-d numpy array of pixel angules in the range [-180,180] degree."""
        return self.phi0


    def pixel_phi(self) :
        """Returns 1-d numpy array of pixel angules in the range [phi_min, phi_min+360] degree."""
        return self.phi


    def pixel_iphi(self) :
        """Returns 1-d numpy array of pixel angular indexes."""
        return self.iphi


    def pixel_iseq(self) :
        """Returns 1-d numpy array of sequentially (in rad and phi) numerated pixel indexes."""
        return self.iseq


    def bin_number_of_pixels(self) :
        """Returns 1-d numpy array of number of accounted pixels per bin."""
        return self.npix_per_bin


    def _flatten_(self, nda) :
        if len(nda.shape)>1 :
            #nda.shape = self.shapeflat
            return nda.flatten() # return flatten copy in order to preserve input array shape
        return nda


    def bin_intensity(self, nda) :
        """Returns 1-d numpy array of total pixel intensity per bin for input array nda."""
        #return np.bincount(self.iseq, weights=self._flatten_(nda), minlength=None)
        return np.bincount(self.iseq, weights=self._flatten_(nda), minlength=self.ntbins+1) # +1 for overflow bin


    def bin_avrg(self, nda) :
        """Returns 1-d numpy array of averaged in bin intensity for input array nda."""
        num = self.bin_intensity(self._flatten_(nda))
        den = self.bin_number_of_pixels()
        #print_ndarr(nda, name='ZZZ bin_avrg: nda', first=0, last=5)
        #print_ndarr(num, name='ZZZ bin_avrg: num', first=0, last=5)
        #print_ndarr(den, name='ZZZ bin_avrg: den', first=0, last=5)
        return divide_protected(num, den, vsub_zero=0)


    def bin_avrg_rad_phi(self, nda, do_transp=True) :
        """Returns 2-d (rad,phi) numpy array of averaged in bin intensity for input array nda."""
        arr_rphi = self.bin_avrg(self._flatten_(nda))[:-1] # -1 removes overflow bin
        arr_rphi.shape = (self.pb.nbins(), self.rb.nbins())
        return np.transpose(arr_rphi) if do_transp else arr_rphi


    def pixel_avrg(self, nda) :
        """Returns 1-d numpy array of per-pixel background for input array nda."""
        bin_avrg= self.bin_avrg(self._flatten_(nda))
        return np.array([bin_avrg[i] for i in self.iseq])


    def pixel_avrg_interpol(self, nda, method='linear', verb=False) : # 'nearest' 'cubic'
        """Returns 1-d numpy array of per-pixel interpolated background for averaged input data."""

        #if not is360 : raise ValueError('Interpolation works for 360 degree coverage ONLY') 

        if self.griddata is None :
            from scipy.interpolate import griddata
            self.griddata = griddata

        # 1) get values in bin centers
        binv = self.bin_avrg_rad_phi(self._flatten_(nda), do_transp=False)

        # 2) add values in bin edges
        
        if verb : print('binv.shape: ', binv.shape)
        vrad_a1,  vrad_a2 = binv[0,:], binv[-1,:]
        if self.is360 :
            vrad_a1 = vrad_a2 = 0.5*(binv[0,:] + binv[-1,:]) # [iphi, irad]
        nodea = np.vstack((vrad_a1, binv, vrad_a2))
        
        vang_rmin, vang_rmax = nodea[:,0], nodea[:,-1]
        vang_rmin.shape = vang_rmax.shape = (vang_rmin.size, 1) # it should be 2d for hstack
        val_nodes = np.hstack((vang_rmin, nodea, vang_rmax))
        if verb : print('nodear.shape: ', val_nodes.shape)

        # 3) extend bin-centers by limits        
        bcentsr = self.rb.bincenters()
        bcentsp = self.pb.bincenters()
        blimsr  = self.rb.limits()
        blimsp  = self.pb.limits()

        rad_nodes = np.concatenate(((blimsr[0],), bcentsr, (blimsr[1],)))
        phi_nodes = np.concatenate(((blimsp[0],), bcentsp, (blimsp[1],)))
        if verb : print('rad_nodes.shape', rad_nodes.shape)
        if verb : print('phi_nodes.shape', phi_nodes.shape)

        # 4) make point coordinate and value arrays
        points_rad, points_phi = np.meshgrid(rad_nodes, phi_nodes)
        if verb : print('points_phi.shape', points_phi.shape)
        if verb : print('points_rad.shape', points_rad.shape)
        points = np.array(zip(points_phi.flatten(), points_rad.flatten())) 
        if verb : print('points.shape', points.shape)

        values = val_nodes.flatten()
        if verb : print('values.shape', values.shape)

        # 4) return interpolated data on (phi, rad) grid
        grid_vals = self.griddata(points, values, (self.phi, self.rad), method=method)
        return np.select((self.iseq<self.pb.nbins()*self.rb.nbins(),), (grid_vals,), default=0)

#------------------------------
#------------------------------
#----------- TEST -------------
#------------------------------
#------------------------------


def data_geo(ntest) :
    """Method for tests: returns test data numpy array and geometry object
    """
    from time import time
    from psana.pscalib.calib.NDArrIO import save_txt, load_txt
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess

    dir       = '/reg/g/psdm/detector/alignment/cspad/calib-cxi-camera2-2016-02-05'
    #fname_nda = '%s/nda-water-ring-cxij4716-r0022-e000001-CxiDs2-0-Cspad-0-ave.txt' % dir
    #fname_nda = '%s/nda-water-ring-cxij4716-r0022-e014636-CxiDs2-0-Cspad-0-ave.txt' % dir
    #fname_nda = '%s/nda-lysozyme-cxi02416-r0010-e052421-CxiDs2-0-Cspad-0-max.txt' % dir
    fname_nda = '%s/nda-lysozyme-cxi01516-r0026-e093480-CxiDs2-0-Cspad-0-max.txt'%dir if ntest in (21,28,29,30)\
                else '%s/nda-water-ring-cxij4716-r0022-e014636-CxiDs2-0-Cspad-0-ave.txt'%dir
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
    print('Time to load geometry %.3f sec from file\n%s' % (time()-t0_sec, fname_geo))

    return arr, geo

#------------------------------

def usage(ntest=None) :
    s = ''
    if ntest is None      : s+='\n Tests for radial 1-d binning of entire image'
    if ntest in (None, 1) : s+='\n  1 - averaged data'
    if ntest in (None, 2) : s+='\n  2 - pixel radius value'
    if ntest in (None, 3) : s+='\n  3 - pixel phi value'
    if ntest in (None, 4) : s+='\n  4 - pixel radial bin index'
    if ntest in (None, 5) : s+='\n  5 - pixel phi bin index'
    if ntest in (None, 6) : s+='\n  6 - pixel sequential (rad and phi) bin index'
    if ntest in (None, 7) : s+='\n  7 - mask'
    if ntest in (None, 8) : s+='\n  8 - averaged radial intensity'
    if ntest in (None, 9) : s+='\n  9 - interpolated radial intensity'

    if ntest is None      : s+='\n Test for 2-d (default) binning of the rad-phi range of entire image'
    if ntest in (None,21) : s+='\n 21 - averaged data'
    if ntest in (None,24) : s+='\n 24 - pixel radial bin index'
    if ntest in (None,25) : s+='\n 25 - pixel phi bin index'
    if ntest in (None,26) : s+='\n 26 - pixel sequential (rad and phi) bin index'
    if ntest in (None,28) : s+='\n 28 - averaged radial intensity'
    if ntest in (None,29) : s+='\n 29 - averaged radial interpolated intensity'
    if ntest in (None,30) : s+='\n 30 - r-phi'
                                     
    if ntest is None      : s+='\n Test for 2-d binning of the restricted rad-phi range of entire image'
    if ntest in (None,41) : s+='\n 41 - averaged data'
    if ntest in (None,44) : s+='\n 44 - pixel radial bin index'
    if ntest in (None,45) : s+='\n 45 - pixel phi bin index'
    if ntest in (None,46) : s+='\n 46 - pixel sequential (rad and phi) bin index'
    if ntest in (None,48) : s+='\n 48 - averaged radial intensity'
    if ntest in (None,49) : s+='\n 49 - averaged radial interpolated intensity'
    if ntest in (None,50) : s+='\n 50 - r-phi'

    return s


#------------------------------

def test01(ntest, prefix='fig-v01') :
    """Test for radial 1-d binning of entire image.
    """
    from time import time
    import psana.pyalgos.generic.Graphics as gg
    from psana.pscalib.geometry.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    t0_sec = time()
    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0o377).flatten() 
    print('Time to retrieve geometry %.3f sec' % (time()-t0_sec))

    t0_sec = time()
    hp = HPolar(X, Y, mask, nradbins=500, nphibins=1) # v1
    print('HPolar initialization time %.3f sec' % (time()-t0_sec))

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 1 : nda, title = arr,                   'averaged data'
    elif ntest == 2 : nda, title = hp.pixel_rad(),        'pixel radius value'
    elif ntest == 3 : nda, title = hp.pixel_phi(),        'pixel phi value'
    elif ntest == 4 : nda, title = hp.pixel_irad() + 2,   'pixel radial bin index' 
    elif ntest == 5 : nda, title = hp.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 6 : nda, title = hp.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    elif ntest == 7 : nda, title = mask,                  'mask'
    elif ntest == 8 : nda, title = hp.pixel_avrg(nda),    'averaged radial intensity'
    elif ntest == 9 : nda, title = hp.pixel_avrg_interpol(arr) * mask , 'interpolated radial intensity'
    else :
        print('Test %d is not implemented' % ntest)
        return
        
    print('Get %s n-d array time %.3f sec' % (title, time()-t0_sec))

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (21,) else nda[100:300,:]

    da, ds = None, None
    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter'
    if ntest in (2,3,4,5,6,7) :
        da = ds = (nda.min()-1., nda.max()+1.)

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

    print('End of test for %s' % title) 

#------------------------------

def test02(ntest, prefix='fig-v01') :
    """Test for 2-d (default) binning of the rad-phi range of entire image
    """
    #from Detector.GlobalUtils import print_ndarr
    from time import time
    import psana.pyalgos.generic.Graphics as gg
    from psana.pscalib.geometry.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0o377).flatten() 

    t0_sec = time()
    #hp = HPolar(X, Y, mask) # v0
    hp = HPolar(X, Y, mask, nradbins=500) # , nphibins=8, phiedges=(-20, 240), radedges=(10000,80000))
    print('HPolar initialization time %.3f sec' % (time()-t0_sec))

    #print('bin_number_of_pixels:',   hp.bin_number_of_pixels())
    #print('bin_intensity:', hp.bin_intensity(arr))
    #print('bin_avrg:',   hp.bin_avrg(arr))

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 21 : nda, title = arr,                   'averaged data'
    elif ntest == 24 : nda, title = hp.pixel_irad() + 2,   'pixel radial bin index' 
    elif ntest == 25 : nda, title = hp.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 26 : nda, title = hp.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    #elif ntest == 27 : nda, title = mask,                  'mask'
    elif ntest == 28 : nda, title = hp.pixel_avrg(nda),    'averaged radial intensity'
    elif ntest == 29 : nda, title = hp.pixel_avrg_interpol(nda) * mask, 'averaged radial interpolated intensity'
    elif ntest == 30 : nda, title = hp.bin_avrg_rad_phi(nda),'r-phi'
    else :
        print('Test %d is not implemented' % ntest)
        return

    print('Get %s n-d array time %.3f sec' % (title, time()-t0_sec))

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (30,) else nda # [100:300,:]

    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter' 'gray'

    da = (nda.min()-1, nda.max()+1)
    ds = da

    if ntest in (21,28,29,30) :
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print('End of test for %s' % title)

#------------------------------

def test03(ntest, prefix='fig-v01') :
    """Test for 2-d binning of the restricted rad-phi range of entire image
    """
    from time import time
    import psana.pyalgos.generic.Graphics as gg
    from psana.pscalib.geometry.GeometryAccess import img_from_pixel_arrays

    arr, geo = data_geo(ntest)

    iX, iY = geo.get_pixel_coord_indexes()
    X, Y, Z = geo.get_pixel_coords()
    mask = geo.get_pixel_mask(mbits=0o377).flatten() 

    t0_sec = time()

    #hp = HPolar(X, Y, mask, nradbins=5, nphibins=8, phiedges=(-20, 240), radedges=(10000,80000))
    hp = HPolar(X, Y, mask, nradbins=3, nphibins=8, phiedges=(240, -20), radedges=(80000,10000)) # v3

    print('HPolar initialization time %.3f sec' % (time()-t0_sec))

    #print('bin_number_of_pixels:',   hp.bin_number_of_pixels())
    #print('bin_intensity:', hp.bin_intensity(arr))
    #print('bin_avrg:',   hp.bin_avrg(arr))

    t0_sec = time()
    nda, title = arr, None
    if   ntest == 41 : nda, title = arr,                   'averaged data'
    elif ntest == 44 : nda, title = hp.pixel_irad() + 2,   'pixel radial bin index' 
    elif ntest == 45 : nda, title = hp.pixel_iphi() + 2,   'pixel phi bin index'
    elif ntest == 46 : nda, title = hp.pixel_iseq() + 2,   'pixel sequential (rad and phi) bin index'
    #elif ntest == 47 : nda, title = mask,                  'mask'
    elif ntest == 48 : nda, title = hp.pixel_avrg(nda),    'averaged radial intensity'
    elif ntest == 49 : nda, title = hp.pixel_avrg_interpol(nda) * mask, 'averaged radial interpolated intensity'
    elif ntest == 50 : nda, title = hp.bin_avrg_rad_phi(nda),'r-phi'
    else :
        print('Test %d is not implemented' % ntest)
        return

    print('Get %s n-d array time %.3f sec' % (title, time()-t0_sec))

    img = img_from_pixel_arrays(iX, iY, nda) if not ntest in (50,) else nda # [100:300,:]

    colmap = 'jet' # 'cubehelix' 'cool' 'summer' 'jet' 'winter' 'gray'

    da = (nda.min()-1, nda.max()+1)
    ds = da

    if ntest in (41,48,49,50) :
        ave, rms = nda.mean(), nda.std()
        da = ds = (ave-2*rms, ave+3*rms)

    gg.plotImageLarge(img, amp_range=da, figsize=(14,12), title=title, cmap=colmap)
    gg.save('%s-%02d-img.png' % (prefix, ntest))

    gg.hist1d(nda, bins=None, amp_range=ds, weights=None, color=None, show_stat=True, log=False, \
           figsize=(6,5), axwin=(0.18, 0.12, 0.78, 0.80), \
           title=None, xlabel='Pixel value', ylabel='Number of pixels', titwin=title)
    gg.save('%s-%02d-his.png' % (prefix, ntest))

    gg.show()

    print('End of test for %s' % title)

#------------------------------

if __name__ == '__main__' :
    import sys

    #if len(sys.argv) == 1 : print(usage())
    print(usage())

    ntest = int(sys.argv[1]) if len(sys.argv)>1 else 1
    print('Test # %d: %s' % (ntest, usage(ntest)))

    prefix = 'fig-v01-cspad-HPolar'

    if   ntest<20 : test01(ntest, prefix)
    elif ntest<40 : test02(ntest, prefix)
    elif ntest<60 : test03(ntest, prefix)
    else : print('Test %d is not implemented' % ntest)
    #sys.exit('End of test')
 
#------------------------------
#------------------------------
