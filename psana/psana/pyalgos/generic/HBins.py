
####!/usr/bin/env python
"""
Class :py:class:`HBins` histogram-style bin parameters holder
=============================================================

Usage::

    from psana.pyalgos.generic.HBins import HBins

    # Equal bins constructor
    hb = HBins((1,6), nbins=5)

    # Variable bins constructor
    hb = HBins((1,2,4,6,10))

    # Access methods
    nbins         = hb.nbins()         # returns int input parameter - number of bins
    edges         = hb.edges()         # returns np.array input list of bin edges
    vmin          = hb.vmin()          # returns vtype minimal value of bin edges
    vmax          = hb.vmax()          # returns vtype maximal value of bin edges
    vtype         = hb.vtype()         # returns np.dtype - type of bin edge values
    equalbins     = hb.equalbins()     # returns bool True/False for equal/variable size bins

    limits        = hb.limits()        # returns np.array of limits (vmin, vmax)
    binedges      = hb.binedges()      # returns np.array with bin edges of size nbins+1
    binedgesleft  = hb.binedgesleft()  # returns np.array with bin left edges of size nbins
    binedgesright = hb.binedgesright() # returns np.array with bin rignt edges of size nbins
    bincenters    = hb.bincenters()    # returns np.array with bin centers of size nbins
    binwidth      = hb.binwidth()      # returns np.array with bin widths of size nbins or scalar bin width for equal bins
    halfbinw      = hb.halfbinw()      # returns np.array with half-bin widths of size nbins or scalar bin half-width for equal bins
    strrange      = hb.strrange(fmt)   # returns str of formatted vmin, vmax, nbins ex: 1-6-5

    ind     = hb.bin_index(value, edgemode=0)    # returns bin index [0,nbins) for value.
    indarr  = hb.bin_indexes(valarr, edgemode=0) # returns array of bin index [0,nbins) for array of values
    hisarr  = hb.bin_count(valarr, edgemode=0)   # returns array of bin counts [0,nbins) for array of values (histogram value per bin)
    # edgemode - defines what to do with underflow overflow indexes;
    #          = 0 - use indexes  0 and nbins-1 for underflow overflow, respectively
    #          = 1 - use extended indexes -1 and nbins for underflow overflow, respectively

    hb.set_bin_data(data, dtype=np.float) # adds bin data to the HBins object. data size should be equal to hb.nbins()
    data = bin_data(dtype=np.float)       # returns numpy array of data associated with HBins object.
    data = hb.set_bin_data_from_array(self, arr, dtype=np.float64, edgemode=0) # set bin data from array (like image or ndarray)
    mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w = hb.histogram_statistics()

    # Print methods
    hb.print_attrs_defined()
    hb.print_attrs()
    hb.print_attrs_and_methods()

See:
  - :py:class:`HBins`
  - :py:class:`HPolar`
  - :py:class:`HSpectrum`
  - :py:class:`NDArrSpectrum`
  - :py:class:`RadialBkgd`
  - `Radial background <https://confluence.slac.stanford.edu/display/PSDMInternal/Radial+background+subtraction+algorithm>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-01-15 by Mikhail Dubrovin
"""

import math
import numpy as np
sqrt = math.sqrt

#import logging
#logger = logging.getLogger(__name__)

class HBins():
    """Hystogram-style bin parameters holder
    """
    def __init__(self, edges, nbins=None, vtype=np.float32):
        """Class constructor for
           - equal bins,       ex: hb = HBins((1,6), nbins=5)
           - or variable bins, ex: hb = HBins((1,2,4,6,10))

           Parameters:
           - edges - sequence of two or more bin edges
           - nbins - (int) number of bins for equal size bins
           - vtype - numpy type of bin values (optional parameter)
        """
        self._name       = self.__class__.__name__
        self._vtype      = vtype
        self._set_valid_edges(edges)
        self._set_valid_nbins(nbins)

        self._vmin       = min(self._edges)
        self._vmax       = max(self._edges)
        self._equalbins  = len(self._edges)==2 and nbins is not None
        self._ascending  = self._edges[0] < self._edges[-1]
        self._bin_data   = None

        # dynamic parameters
        self._limits     = None
        self._binwidth   = None
        self._halfbinw   = None
        self._binedges   = None
        self._bincenters = None
        self._inds       = None
        self._indedges   = None
        self._indcenters = None
        self._strrange   = None


    def _set_valid_edges(self, edges):

        if not isinstance(edges,(tuple,list,np.array)):
            raise ValueError('Parameter edges is not a tuple or list: '\
                             'edges=%s' % str(edges))

        if len(edges)<2:
            raise ValueError('Sequence of edges should have at least two values: '\
                             'edges=%s' % str(edges))

        if not all([isinstance(v,(int, float, np.generic)) for v in tuple(edges)]):
            raise ValueError('Sequence of edges has a wrong type value: '\
                             'edges=%s' % str(edges))

        if edges[0]==edges[-1]:
            raise ValueError('Sequence of edges has equal limits: '\
                             'edges=%s' % str(edges))

        if len(edges)>2:
            if edges[0]<edges[-1] and not all([x<y for x,y in zip(edges[:-1], edges[1:])]):
                raise ValueError('Sequence of edges is not monotonically ascending: '\
                                 'edges=%s' % str(edges))

            if edges[0]>edges[-1] and not all([x>y for x,y in zip(edges[:-1], edges[1:])]):
                raise ValueError('Sequence of edges is not monotonically descending: '\
                                 'edges=%s' % str(edges))

        self._edges = np.array(edges, dtype=self._vtype)


    def _set_valid_nbins(self, nbins):

        if nbins is None:
            self._nbins = len(self._edges)-1
            return

        if not isinstance(nbins, int):
            raise ValueError('nbins=%s has a wrong type. Expected integer.' % str(nbins))

        if nbins < 1:
            raise ValueError('nbins=%d should be positive.' % nbins)

        self._nbins = nbins


    def edges(self):
        """Returns input sequence of edges"""
        return self._edges


    def vmin(self):
        """Returns minimal value of the range"""
        return self._vmin


    def vmax(self):
        """Returns miximal value of the range"""
        return self._vmax


    def nbins(self):
        """Returns number of bins"""
        return self._nbins


    def vtype(self):
        """Returns npumpy datatype for bin values"""
        return self._vtype


    def equalbins(self):
        return self._equalbins


    def ascending(self):
        return self._ascending


    def limits(self):
        """Returns np.array of two ordered limits (vmin, vmax)"""
        if self._limits is None:
            self._limits = np.array((self._edges[0], self._edges[-1]), dtype=self._vtype)
        return self._limits


    def binedges(self):
        """Returns np.array of nbins+1 values of bin edges"""
        if self._binedges is None:
            if self._equalbins:
                self._binedges = np.linspace(self._edges[0], self._edges[-1], self._nbins+1, endpoint=True, dtype=self._vtype)
            else:
                self._binedges = self._edges
        return self._binedges


    def binedgesleft(self):
        """Returns np.array of nbins values of bin left edges"""
        return self.binedges()[:-1]


    def binedgesright(self):
        """Returns np.array of nbins values of bin right edges"""
        return self.binedges()[1:]


    def binwidth(self):
        """Returns np.array of nbins values of bin widths"""
        if self._binwidth is None:
            if self._equalbins:
                self._binwidth = float(self._edges[-1]-self._edges[0])/self._nbins
            else:
                self._binwidth = self.binedgesright() - self.binedgesleft()
        return self._binwidth


    def halfbinw(self):
        """Returns np.array of nbins values of bin half-widths"""
        if self._halfbinw is None:
                self._halfbinw = 0.5 * self.binwidth()
        return self._halfbinw


    def bincenters(self):
        """Returns np.array of nbins values of bin centers"""
        if self._bincenters is None:
            self._bincenters = self.binedgesleft() + self.halfbinw()
        return self._bincenters


    def _set_limit_indexes(self, edgemode):
        """Returns limit bin indexes for underflow and overflow values"""
        if   edgemode==0: return  0, self._nbins-1
        elif edgemode==1: return -1, self._nbins


    def bin_index(self, v, edgemode=0):
        """Returns bin index for scalar value"""

        indmin, indmax = self._set_limit_indexes(edgemode)
        if self._ascending:
            if v< self._edges[0] : return indmin
            if v>=self._edges[-1]: return indmax
        else:
            if v> self._edges[0] : return indmin
            if v<=self._edges[-1]: return indmax

        if self._equalbins:
            return int(math.floor((v-self._edges[0])/self.binwidth()))


        if self._ascending:
            for ind, edgeright in enumerate(self.binedgesright()):
                if v<edgeright:
                    return ind
        else:
            for ind, edgeright in enumerate(self.binedgesright()):
                if v>edgeright:
                    return ind


    def bin_indexes(self, arr, edgemode=0):

        indmin, indmax = self._set_limit_indexes(edgemode)

        if self._equalbins:
            factor = float(self._nbins)/(self._edges[-1]-self._edges[0])
            nbins1 = self._nbins-1
            nparr = (np.array(arr, dtype=self._vtype)-self._edges[0])*factor
            ind = np.array(np.floor(nparr), dtype=np.int32)
            return np.select((ind<0, ind>nbins1), (indmin, indmax), default=ind)

        else:
            conds = None
            if self._ascending:
                conds = np.array([arr<edge for edge in self.binedges()], dtype=np.bool)
            else:
                conds = np.array([arr>edge for edge in self.binedges()], dtype=np.bool)

            inds1d = list(range(-1, self._nbins))
            inds1d[0] = indmin # re-define index for underflow
            inds = np.array(len(arr)*inds1d, dtype=np.int32)
            inds.shape = (len(arr),self._nbins+1)
            inds = inds.transpose()
            #print('indmin, indmax = ', indmin, indmax)
            #print('XXX conds:\n', conds)
            #print('XXX inds:\n', inds)
            return np.select(conds, inds, default=indmax)


    def bin_count(self, arr, edgemode=0, weights=None):
        indarr = self.bin_indexes(arr.ravel(), edgemode)
        return np.bincount(indarr, weights, self.nbins())


    def set_bin_data(self, data, dtype=np.float64):
        assert len(data)==self.nbins()
            #self._bin_data = None
            #return
        self._bin_data = np.array(data, dtype)


    def bin_data(self, dtype=np.float64):
        return self._bin_data.astype(dtype) if self._bin_data is not None else None


    def bin_data_max(self):
        return self._bin_data.max() if self._bin_data is not None else None


    def bin_data_min(self):
        return self._bin_data.min() if self._bin_data is not None else None


    def set_bin_data_from_array(self, arr, dtype=np.float64, edgemode=0):
        aravel = arr.ravel()
        hisarr = self.bin_count(aravel, edgemode=edgemode)
        self.set_bin_data(hisarr, dtype=dtype)
        #return hisarr


    def strrange(self, fmt='%.0f-%.0f-%d'):
        """Returns string of range parameters"""
        if self._strrange is None:
            self._strrange =fmt % (self._edges[0], self._edges[-1], self._nbins)
        return self._strrange


    def print_attrs(self):
        print('Attributes of the %s object' % self._name)
        for k,v in self.__dict__.items():
            print('  %s: %s' % (k.ljust(16), str(v)))


    def print_attrs_defined(self):
        print('Attributes (not None) of the %s object' % self._name)
        for k,v in self.__dict__.items():
            if v is None: continue
            print('  %s: %s' % (k.ljust(16), str(v)))


    def print_attrs_and_methods(self):
        print('Methods & attributes of the %s object' % self._name)
        for m in dir(self):
            print('  %s' % (str(m).ljust(16)))


    def histogram_statistics(self, vmin=None, vmax=None):
        ibeg, iend = self.bin_indexes((vmin,vmax), edgemode=0)
        #logger.debug('histogram_statistics between bins %d : %d'%(ibeg, iend))
        center = self.bincenters()[ibeg:iend]
        weights = self.bin_data()[ibeg:iend]
        if weights is None: weights = np.ones_like(center)

        sum_w  = weights.sum()
        if sum_w <= 0: return  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        sum_w2 = (weights*weights).sum()
        neff   = sum_w*sum_w/sum_w2 if sum_w2>0 else 0
        sum_1  = (weights*center).sum()
        mean = sum_1/sum_w
        d      = center - mean
        d2     = d * d
        wd2    = weights*d2
        m2     = (wd2)   .sum() / sum_w
        m3     = (wd2*d) .sum() / sum_w
        m4     = (wd2*d2).sum() / sum_w

        #sum_2  = (weights*center*center).sum()
        #err2 = sum_2/sum_w - mean*mean
        #err  = sqrt(err2)

        rms  = sqrt(m2) if m2>0 else 0
        rms2 = m2

        err_mean = rms/sqrt(neff)
        err_rms  = err_mean/sqrt(2)

        skew, kurt, var_4 = 0, 0, 0

        if rms>0 and rms2>0:
            skew  = m3/(rms2 * rms)
            kurt  = m4/(rms2 * rms2) - 3
            var_4 = (m4 - rms2*rms2*(neff-3)/(neff-1))/neff if neff>1 else 0
        err_err = sqrt(sqrt(var_4)) if var_4>0 else 0
        #print  'mean:%f, rms:%f, err_mean:%f, err_rms:%f, neff:%f' % (mean, rms, err_mean, err_rms, neff)
        #print  'skew:%f, kurt:%f, err_err:%f' % (skew, kurt, err_err)
        return mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w, ibeg, iend

# EOF
