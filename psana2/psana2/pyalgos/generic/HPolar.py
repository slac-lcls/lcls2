
"""
Class :py:class:`HPolar` - makes 2-d histogram in polar (r-phi) coordinates for imaging detector n-d array data
===============================================================================================================

Usage::

    # Import
    # ------
    from psana2.pyalgos.generic.HPolar import HPolar

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
    pixav = hp.pixel_avrg(nda, subs_value=0)
    pixav = hp.pixel_avrg_interpol(nda, method='linear') # method='nearest' 'cubic'

    # Print attributes and n-d arrays
    # -------------------------------
    hp.info_attrs()
    hp.print_attrs()
    hp.print_ndarrs()

    # Global methods
    # --------------
    from psana2.pyalgos.generic.HPolar import polarization_factor, divide_protected, cart2polar, polar2cart, bincount

    polf = polarization_factor(rad, phi, z, vertical=False)
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

import math
import numpy as np
from psana2.pyalgos.generic.HBins import HBins
#from psana2.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr


def info_ndarr(nda, name='', first=0, last=5):
    _name = '%s '%name if name!='' else name
    s = ''
    gap = '\n' if (last-first)>10 else ' '
    if nda is None: s = '%sNone' % _name
    elif isinstance(nda, tuple): s += info_ndarr(np.array(nda), 'ndarray from tuple: %s' % name)
    elif isinstance(nda, list):  s += info_ndarr(np.array(nda), 'ndarray from list: %s' % name)
    elif not isinstance(nda, np.ndarray):
        s = '%s%s' % (_name, type(nda))
    else:
        a = '' if last == 0 else\
            '%s%s' % (str(nda.ravel()[first:last]).rstrip(']'), '...]' if nda.size>last else ']')
        s = '%sshape:%s size:%d dtype:%s%s%s' % (_name, str(nda.shape), nda.size, nda.dtype, gap, a)
    return s

def print_ndarr(nda, name=' ', first=0, last=5):
    print(info_ndarr(nda, name, first, last))


def divide_protected(num, den, vsub_zero=0):
    """Returns result of devision of numpy arrays num/den with substitution of value vsub_zero for zero den elements.
    """
    pro_num = np.select((den!=0,), (num,), default=vsub_zero)
    pro_den = np.select((den!=0,), (den,), default=1)
    return pro_num / pro_den


def cart2polar(x, y):
    """For numpy arrays x and y returns the numpy arrays of r and theta.
    """
    r = np.sqrt(x*x + y*y)
    theta = np.rad2deg(np.arctan2(y, x)) #[-180,180]
    #theta0 = np.select([theta<0, theta>=0],[theta+360,theta]) #[0,360]
    return r, theta


def polar2cart(r, theta):
    """For numpy arryys r and theta returns the numpy arrays of x and y.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def bincount(map_bins, map_weights=None, length=None):
    """Wrapper for numpy.bincount with protection of weights and alattening numpy arrays.
    """
    weights = None if map_weights is None else map_weights.ravel()
    return np.bincount(map_bins.ravel(), weights, length)


def polarization_factor(rad, phi_deg, z, vertical=False):
    """Returns per-pixel polarization factors, assuming that detector is perpendicular to Z.
    """
    _phi_deg = np.array(phi_deg + 90) if vertical else phi_deg
    phi = np.deg2rad(_phi_deg)
    ones = np.ones_like(rad)
    theta = np.arctan2(rad, z)
    sxc = np.sin(theta)*np.cos(phi)
    pol = 1 - sxc*sxc
    return divide_protected(ones, pol, vsub_zero=0)


class HPolar():
    def __init__(self, xarr, yarr, mask=None, radedges=None, nradbins=100, phiedges=(0,360), nphibins=32):
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

        self.cond = np.logical_and(\
               np.logical_and(self.irad > -1, self.irad < nrbins),
               np.logical_and(self.iphi > -1, self.iphi < npbins)
               )

        if mask is not None:
            self.cond = np.logical_and(self.cond, mask.astype(np.dtype(bool)).ravel())

        # index ntbins stands for overflow bin
        self.iseq = np.select((self.cond,), (self.iphi*nrbins + self.irad,), self.ntbins).ravel()

        #self.npix_per_bin = np.bincount(self.iseq, weights=None, minlength=None)
        self.npix_per_bin = np.bincount(self.iseq, weights=None, minlength=self.ntbins+1)

        self.griddata = None


    def _set_rad_bins(self, radedges, nradbins):
        rmin = math.floor(np.amin(self.rad)) if radedges is None else radedges[0]
        rmax = math.ceil (np.amax(self.rad)) if radedges is None else radedges[-1]
        if rmin<1: rmin = 1
        self.rb = HBins((rmin, rmax), nradbins)


    def _set_phi_bins(self, phiedges, nphibins):
        if phiedges[-1] > phiedges[0]+360\
        or phiedges[-1] < phiedges[0]-360:
            raise ValueError('Difference between angular edges should not exceed 360 degree;'\
                             ' phiedges: %.0f, %.0f' % (phiedges[0], phiedges[-1]))
        self.pb = HBins(phiedges, nphibins)
        phi1, phi2 = self.pb.limits()
        self.is360 = math.fabs(math.fabs(phi2-phi1)-360) < 1e-3


    def info_attrs(self):
        return '%s attrbutes:' % self.__class__.__name__\
          + self.pb.strrange(fmt='\nPhi bins:  min:%8.1f  max:%8.1f  nbins:%5d')\
          + self.rb.strrange(fmt='\nRad bins:  min:%8.1f  max:%8.1f  nbins:%5d')


    def print_attrs(self):
        print(self.info_attrs())


    def print_ndarrs(self):
        print('%s n-d arrays:' % self.__class__.__name__)
        print('  rad  shape=', str(self.rad.shape))
        print('  phi  shape=', str(self.phi.shape))
        print('  mask shape=', str(self.mask.shape))
        #print('Phi limits: ', phiedges[0], phiedges[-1])


    def obj_radbins(self):
        """Returns HBins object for radial bins."""
        return self.rb


    def obj_phibins(self):
        """Returns HBins object for angular bins."""
        return self.pb


    def pixel_rad(self):
        """Returns 1-d numpy array of pixel radial parameters."""
        return self.rad


    def pixel_irad(self):
        """Returns 1-d numpy array of pixel radial indexes [-1,nrbins] - extended edgemode."""
        return self.irad


    def pixel_phi0(self):
        """Returns 1-d numpy array of pixel angules in the range [-180,180] degree."""
        return self.phi0


    def pixel_phi(self):
        """Returns 1-d numpy array of pixel angules in the range [phi_min, phi_min+360] degree."""
        return self.phi


    def pixel_iphi(self):
        """Returns 1-d numpy array of pixel angular indexes [-1,npbins] - extended edgemode."""
        return self.iphi


    def pixel_iseq(self):
        """Returns 1-d numpy array of sequentially (in rad and phi) numerated pixel indexes [0,ntbins].
           WARNING: pixels outside the r-phi region of interest marked by the index ntbins,
                    ntbins - total number of r-phi bins, which exceeds allowed range of r-phi indices...
        """
        return self.iseq


    def bin_number_of_pixels(self):
        """Returns 1-d numpy array of number of accounted pixels per bin."""
        return self.npix_per_bin


    def _ravel_(self, nda):
        if len(nda.shape)>1:
            #nda.shape = self.shapeflat
            return nda.ravel() # return ravel copy in order to preserve input array shape
        return nda


    def bin_intensity(self, nda):
        """Returns 1-d numpy array of total pixel intensity per bin for input array nda."""
        #return np.bincount(self.iseq, weights=self._ravel_(nda), minlength=None)
        return np.bincount(self.iseq, weights=self._ravel_(nda), minlength=self.ntbins+1) # +1 for overflow bin


    def bin_avrg(self, nda):
        """Returns 1-d numpy array of averaged in r-phi bin intensities for input image array nda.
           WARNING array range [0, nrbins*npbins + 1], where +1 bin intensity is for all off ROI pixels.
        """
        num = self.bin_intensity(self._ravel_(nda))
        den = self.bin_number_of_pixels()
        #print_ndarr(nda, name='ZZZ bin_avrg: nda', first=0, last=5)
        #print_ndarr(num, name='ZZZ bin_avrg: num', first=0, last=5)
        #print_ndarr(den, name='ZZZ bin_avrg: den', first=0, last=5)
        return divide_protected(num, den, vsub_zero=0)


    def bin_avrg_rad_phi(self, nda, do_transp=True):
        """Returns 2-d (rad,phi) numpy array of averaged in bin intensity for input array nda."""
        arr_rphi = self.bin_avrg(self._ravel_(nda))[:-1] # -1 removes off ROI bin
        arr_rphi.shape = (self.pb.nbins(), self.rb.nbins())
        return np.transpose(arr_rphi) if do_transp else arr_rphi


    def pixel_avrg(self, nda, subs_value=0):
        """Makes r-phi histogram of intensities from input image array and
           projects r-phi averaged intensities back to image.
           Returns ravel 1-d numpy array of per-pixel intensities taken from r-phi histogram.
           - nda - input (2-d or 1-d-ravel) pixel array.
           - subs_value - value sabstituted for pixels out of ROI defined by the min/max in r-phi.
        """
        bin_avrg= self.bin_avrg(self._ravel_(nda))
        return np.select((self.cond,), (bin_avrg[self.iseq],), subs_value).ravel()
        #return np.array([bin_avrg[i] for i in self.iseq]) # iseq may be outside the bin_avrg range


    def pixel_avrg_interpol(self, nda, method='linear', verb=False, subs_value=0): # 'nearest' 'cubic'
        """Makes r-phi histogram of intensities from input image and
           projects r-phi averaged intensities back to image with per-pixel interpolation.
           Returns 1-d numpy array of per-pixel interpolated intensities taken from r-phi histogram.
           - subs_value - value sabstituted for pixels out of ROI defined by the min/max in r-phi.
        """

        #if not is360: raise ValueError('Interpolation works for 360 degree coverage ONLY')

        if self.griddata is None:
            from scipy.interpolate import griddata
            self.griddata = griddata

        # 1) get values in bin centers
        binv = self.bin_avrg_rad_phi(self._ravel_(nda), do_transp=False)

        # 2) add values in bin edges

        if verb: print('binv.shape: ', binv.shape)
        vrad_a1,  vrad_a2 = binv[0,:], binv[-1,:]
        if self.is360:
            vrad_a1 = vrad_a2 = 0.5*(binv[0,:] + binv[-1,:]) # [iphi, irad]
        nodea = np.vstack((vrad_a1, binv, vrad_a2))

        vang_rmin, vang_rmax = nodea[:,0], nodea[:,-1]
        vang_rmin.shape = vang_rmax.shape = (vang_rmin.size, 1) # it should be 2d for hstack
        val_nodes = np.hstack((vang_rmin, nodea, vang_rmax))
        if verb: print('nodear.shape: ', val_nodes.shape)

        # 3) extend bin-centers by limits
        bcentsr = self.rb.bincenters()
        bcentsp = self.pb.bincenters()
        blimsr  = self.rb.limits()
        blimsp  = self.pb.limits()

        rad_nodes = np.concatenate(((blimsr[0],), bcentsr, (blimsr[1],)))
        phi_nodes = np.concatenate(((blimsp[0],), bcentsp, (blimsp[1],)))
        if verb: print('rad_nodes.shape', rad_nodes.shape)
        if verb: print('phi_nodes.shape', phi_nodes.shape)

        # 4) make point coordinate and value arrays
        points_rad, points_phi = np.meshgrid(rad_nodes, phi_nodes)
        if verb: print('points_phi.shape', points_phi.shape)
        if verb: print('points_rad.shape', points_rad.shape)
        points = np.vstack((points_phi.ravel(), points_rad.ravel())).T
        values = val_nodes.ravel()
        if verb:
            #print('points:', points)
            print('points.shape', points.shape)
            print('values.shape', values.shape)

        # 5) return interpolated data on (phi, rad) grid
        grid_vals = self.griddata(points, values, (self.phi, self.rad), method=method)
        return np.select((self.iseq<self.ntbins,), (grid_vals,), default=subs_value)

# EOF
