#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`pypsalgos` provides access to C++ algorithms from python.

Usage::

    # IMPORT MISC
    # ===========
    import numpy as np
    from pyimgalgos.GlobalUtils import print_ndarr

    # INPUT PARAMETERS
    # ================

    shape = (32,185,388) # e.g.
    ave, rms = 200, 25
    data = np.array(ave + rms*np.random.standard_normal(shape), dtype=np.double)
    mask = np.ones(shape, dtype=np.uint16)

    #mask = det.mask()             # see class Detector.PyDetector
    #mask = np.loadtxt(fname_mask) # 

    # 2-D IMAGE using cython object directly
    # ======================================

    import psalgos

    alg = psalgos.peak_finder_algos(seg=0, pbits=0)
    alg.set_peak_selection_parameters(npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=6)

    peaks = alg.peak_finder_v3r3_d2(data, mask, rank=5, r0=7, dr=2, nsigm=5)
    peaks = alg.peak_finder_v4r3_d2(data, mask, thr_low=20, thr_high=50, rank=5, r0=7, dr=2)

    peaks = alg.list_of_peaks_selected()
    peaks_all = alg.list_of_peaks()

    p = algo.peak(0)
    p = algo.peak_selected(0)

    for p in peaks : print '  row:%4d, col:%4d, npix:%4d, son::%4.1f' % (p.row, p.col, p.npix, p.son)
    for p in peaks : print p.parameters()

    map_u2 = alg.local_maxima()
    map_u2 = alg.local_minima()
    map_u4 = alg.connected_pixels()

    import psalgos.pypsalgos as algos

    n = algos.local_minima_2d(data, mask, rank, extrema) # 0.019(sec) for shape=(1000, 1000) on psanaphi110
    n = algos.local_maxima_2d(data, mask, rank, extrema) # 0.019(sec)
    n = algos.local_maxima_rank1_cross_2d(data, mask, extrema) # 0.014(sec)
    n = algos.threshold_maxima_2d(data, mask, rank, thr_low, thr_high, extrema) # 0.0017(sec)

    # DIRECT CALL PEAKFINDERS
    # =======================

    from psalgos.pypsalgos import peaks_adaptive, peaks_droplet, peaks_adaptive_2d, peaks_droplet_2d

    # data and mask are 2-d numpy arrays of the same shape    
    peaks = peaks_adaptive_2d(data, mask, rank=5, r0=7.0, dr=2.0, nsigm=5, seg=0, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8)

    peaks = peaks_droplet_2d(data, mask=None, thr_low, thr_high, rank=5, r0=7.0, dr=2.0, seg=0, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8)

    # data and mask are N-d numpy arrays or list of 2-d numpy arrays of the same shape    
    peaks = peaks_adaptive(data, mask, rank=5, r0=7.0, dr=2.0, nsigm=3, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8)

    # data and mask are N-d numpy arrays or list of 2-d numpy arrays of the same shape    
    peaks = peaks_droplet(data, mask=None, thr_low, thr_high, rank=5, r0=7.0, dr=2.0, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8)

    # convert peaks (list of peak objects) to somethong else:
    from psalgos.pypsalgos import list_of_peak_parameters, numpy_2d_arr_of_peak_parameters

    lst_peak_pars = list_of_peak_parameters(peaks)         # returns list of tuples, where tuple consists of float peak parameters 
    arr_peak_pars = numpy_2d_arr_of_peak_parameters(peaks) # returns 2d numpy array of float peak parameters

    # Backward compatability support for ImgAlgos.PyAlgos
    # ===================================================

    # import:
    from psalgos.pypsalgos import PyAlgos # replacement for: from ImgAlgos.PyAlgos import PyAlgos

    # create object:
    alg = PyAlgos(mask=mask, pbits=0)

    # set peak-selector parameters:
    alg.set_peak_selection_pars(npix_min=1, npix_max=5000, amax_thr=0, atot_thr=0, son_min=8)

    # call peakfinders ATTENTION TO REVISION NUMBER: use r3 in stead of r2

    # v3 stands for "ranker" a.k.a. "adaptive", r3 stands for revision 
    peaks = alg.peak_finder_v3r3(nda, rank=2, r0=7.0, dr=2.0, nsigm=0) #, mask=...

    # v4 stands for "droplet" a.k.a. "two threshold", r3 stands for revision
    peaks = alg.peak_finder_v4r3(nda, thr_low=10, thr_high=50, rank=5, r0=7.0, dr=2.0) #, mask=...

    # OPTIONAL for backward compatability
    # set mask
    alg.set_mask(mask)

    # set parameters for S/N evaluation algorithm
    alg.set_son_pars(r0=7, dr=2)

    # print info
    alg.print_attributes() # mask, r0, dr, peak selection parameters


Created: 2017-08-10 by Mikhail Dubrovin

--------
"""

#------------------------------
#
#from pyimgalgos.GlobalUtils import print_ndarr
#

import psalgos
import numpy as np

#------------------------------

def shape_as_2d(sh) :
    """Returns 2-d shape for n-d shape if n>2, otherwise returns unchanged shape.
    """
    if len(sh)<3 : return sh
    return (size_from_shape(sh)/sh[-1], sh[-1])

#------------------------------

def shape_as_3d(sh) :
    """Returns 3-d shape for n-d shape if n>3, otherwise returns unchanged shape.
    """
    if len(sh)<4 : return sh
    return (size_from_shape(sh)/sh[-1]/sh[-2], sh[-2], sh[-1])

#------------------------------

def reshape_to_2d(arr) :
    """Returns n-d re-shaped to 2-d
    """
    arr.shape = shape_as_2d(arr.shape)
    return arr

#------------------------------

def reshape_to_3d(arr) :
    """Returns n-d re-shaped to 3-d
    """
    arr.shape = shape_as_3d(arr.shape)
    return arr

#------------------------------

def local_minima_1d(data, mask=None, rank=3, extrema=None) :
    """Finds local minima of specified rank in 1-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return psalgos.local_minima_1d(data, _mask, rank, extrema)

#local_minimums_1d = local_minima_1d

#------------------------------

def local_maxima_1d(data, mask=None, rank=3, extrema=None) :
    """Finds local maxima of specified rank in 1-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return psalgos.local_maxima_1d(data, _mask, rank, extrema)

#local_maximums_1d = local_maxima_1d

#------------------------------

def local_minima_2d(data, mask=None, rank=3, extrema=None) :
    """Finds local minima of specified rank in 2-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return psalgos.local_minimums(data, _mask, rank, extrema)

#local_minimums_2d = local_minima_2d

#------------------------------

def local_maxima_2d(data, mask=None, rank=3, extrema=None) :
    """Finds local maxima of specified rank in 2-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return psalgos.local_maximums(data, _mask, rank, extrema)

#local_maximums_2d = local_maxima_2d

#------------------------------

def local_maxima_rank1_cross_2d(data, mask=None, extrema=None) :
    """Finds local maxima rank=1 cross in 2-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return psalgos.local_maximums_rank1_cross(data, _mask, extrema)

#local_maximums_rank1_cross_2d = local_maxima_rank1_cross_2d

#------------------------------

def threshold_maxima_2d(data, mask=None, rank=3, thr_low=None, thr_high=None, extrema=None) :
    """Finds local maxima using 2-threshold algorithm in 2-d array of data with mask and returns results in array extrema.
    """
    if (thr_low is None) or (thr_high is None) or (extrema is None) : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return psalgos.threshold_maximums(data, _mask, rank, thr_low, thr_high, extrema)

#threshold_maximums_2d = threshold_maxima_2d

#------------------------------

def print_matrix_of_diag_indexes(rank=5) : 
    """Prints matrix of diagonal indexes for local_maxima_2d and local_minima_2d algorithms.
    """
    psalgos.print_matrix_of_diag_indexes(rank)

#------------------------------

def print_vector_of_diag_indexes(rank=5) : 
    """Prints vector of diagonal indexes for local_maxima_2d and local_minima_2d algorithms.
    """
    psalgos.print_vector_of_diag_indexes(rank)

#------------------------------

def peak_finder_alg_2d(seg=0, pbits=0) :
    """Creates and returns algorithm object.
    """
    return psalgos.peak_finder_algos(seg, pbits)

#------------------------------

def peaks_adaptive_2d(data, mask=None, rank=5, r0=7.0, dr=2.0, nsigm=5,\
                      seg=0, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8) :
    """Peak finder for 2-d numpy arrays of data and mask of the same shape
    """
    #from time import time

    #t0_sec = time()
    # pbits NONE=0, DEBUG=1, INFO=2, WARNING=4, ERROR=8, CRITICAL=16

    o = psalgos.peak_finder_algos(seg, pbits=0) #377) # ~17 microsecond
    _npix_max = npix_max if npix_max is not None else (2*rank+1)*(2*rank+1)
    o.set_peak_selection_parameters(npix_min, _npix_max, amax_thr, atot_thr, son_min)
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)

    peaks = o.peak_finder_v3r3_d2(data, _mask, rank, r0, dr, nsigm)

    #print 'peak_finder_v3r3_d2: total time = %.6f(sec)' % (time()-t0_sec)
    #print_ndarr(data,'XXX:data')
    #print_ndarr(_mask,'XXX:_mask')

    return peaks # or o.list_of_peaks_selected()

#------------------------------

def peaks_droplet_2d(data, mask=None, thr_low=None, thr_high=None, rank=5, r0=7.0, dr=2.0,\
                     seg=0, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8) :
    """ peak finder for 2-d numpy arrays of data and mask of the same shape
    """
    #from time import time

    #t0_sec = time()
    # pbits NONE=0, DEBUG=1, INFO=2, WARNING=4, ERROR=8, CRITICAL=16

    if None in (thr_low, thr_high) : return None

    o = psalgos.peak_finder_algos(seg, pbits=0) #377) # ~17 microsecond
    _npix_max = npix_max if npix_max is not None else (2*rank+1)*(2*rank+1)
    o.set_peak_selection_parameters(npix_min, _npix_max, amax_thr, atot_thr, son_min)
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)

    peaks = o.peak_finder_v4r3_d2(data, _mask, thr_low, thr_high, rank, r0, dr)

    #print 'peak_finder_v3r3_d2: total time = %.6f(sec)' % (time()-t0_sec)
    #print_ndarr(data,'XXX:data')
    #print_ndarr(_mask,'XXX:_mask')

    return peaks # or o.list_of_peaks_selected()

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def peaks_adaptive(data, mask, rank=5, r0=7.0, dr=2.0, nsigm=5,\
                   npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8) :
    """Wrapper for liast of 2-d arrays or >2-d arrays 
       data and mask are N-d numpy arrays or list of 2-d numpy arrays of the same shape
    """
    if isinstance(data, list) :
        peaks=[]
        if mask is None :
            for seg, d2d in enumerate(data) :
                peaks += peaks_adaptive_2d(d2d, mask, rank, r0, dr, nsigm,\
                                           seg, npix_min, npix_max, amax_thr, atot_thr, son_min)
        else :
            for seg, (d2d, m2d) in enumerate(zip(data,mask)) :
                peaks += peaks_adaptive_2d(d2d, m2d, rank, r0, dr, nsigm,\
                                           seg, npix_min, npix_max, amax_thr, atot_thr, son_min)
        return peaks

    elif isinstance(data, np.ndarray) :
        if data.ndim==2 :
            seg=0
            return peaks_adaptive_2d(data, mask, rank, r0, dr, nsigm,\
                                     seg, npix_min, npix_max, amax_thr, atot_thr, son_min)

        elif data.ndim>2 :
            shape_in = data.shape
            data.shape = shape_as_3d(shape_in)
            peaks=[]
            if mask is None :
                _mask = np.ones((shape_in[-2], shape_in[-1]), dtype=np.uint16)
                for seg in range(data.shape[0]) :
                    peaks += peaks_adaptive_2d(data[seg,:,:], _mask, rank, r0, dr, nsigm,\
                                               seg, npix_min, npix_max, amax_thr, atot_thr, son_min)
            else :
                mask.shape = data.shape
                for seg in range(data.shape[0]) :
                    peaks += peaks_adaptive_2d(data[seg,:,:], mask[seg,:,:], rank, r0, dr, nsigm,\
                                               seg, npix_min, npix_max, amax_thr, atot_thr, son_min)

                mask.shape = shape_in
            data.shape = shape_in
            return peaks

        else : raise IOError('pypsalgos.peak_finder_v3r3: wrong data.ndim %s' % str(data.ndim))

    else : raise IOError('pypsalgos.peak_finder_v3r3: unexpected object type for data: %s' % str(data))

#------------------------------

def peaks_droplet(data, mask, thr_low, thr_high, rank=5, r0=7.0, dr=2.0,\
                  npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8) :
    """Wrapper for liast of 2-d arrays or >2-d arrays 
       data and mask are N-d numpy arrays or list of 2-d numpy arrays of the same shape
    """
    if isinstance(data, list) :
        peaks=[]
        if mask is None :
            for seg, d2d in enumerate(data) :
                peaks += peaks_droplet_2d(d2d, mask, thr_low, thr_high, rank, r0, dr,\
                                          seg, npix_min, npix_max, amax_thr, atot_thr, son_min)
        else :
            for seg, (d2d, m2d) in enumerate(zip(data,mask)) :
                peaks += peaks_droplet_2d(d2d, m2d, thr_low, thr_high, rank, r0, dr,\
                                          seg, npix_min, npix_max, amax_thr, atot_thr, son_min)
        return peaks

    elif isinstance(data, np.ndarray) :
        if data.ndim==2 :
            seg=0
            return peaks_droplet_2d(data, mask, thr_low, thr_high, rank, r0, dr,\
                                    seg, npix_min, npix_max, amax_thr, atot_thr, son_min)

        elif data.ndim>2 :
            shape_in = data.shape
            data.shape = shape_as_3d(shape_in)
            peaks=[]
            if mask is None :
                _mask = np.ones((shape_in[-2], shape_in[-1]), dtype=np.uint16)
                for seg in range(data.shape[0]) :
                    peaks += peaks_droplet_2d(data[seg,:,:], _mask, thr_low, thr_high, rank, r0, dr,\
                                              seg, npix_min, npix_max, amax_thr, atot_thr, son_min)
            else :
                mask.shape = data.shape
                for seg in range(data.shape[0]) :
                    peaks += peaks_droplet_2d(data[seg,:,:], mask[seg,:,:], thr_low, thr_high, rank, r0, dr,\
                                              seg, npix_min, npix_max, amax_thr, atot_thr, son_min)

                mask.shape = shape_in
            data.shape = shape_in
            return peaks

        else : raise IOError('pypsalgos.peak_finder_v3r3: wrong data.ndim %s' % str(data.ndim))

    else : raise IOError('pypsalgos.peak_finder_v3r3: unexpected object type for data: %s' % str(data))

#------------------------------

def list_of_peak_parameters(peaks) :
    """Converts list of peak objects to the (old style) list of peak parameters.
    """    
    return [p.parameters() for p in peaks]

#------------------------------

def numpy_2d_arr_of_peak_parameters(peaks) :
    """Converts list of peak objects to the (old style) numpy array of peak parameters.
    """    
    return np.array(list_of_peak_parameters(peaks), dtype=np.float)

#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------

class PyAlgos :
    """Backward compatability support for ImgAlgos.PyAlgos.
    """
    def __init__(self, windows=None, mask=None, pbits=0) :
        """Costructor

        :param numpy.array windows: is not used.
        :param numpy.array mask: of the same shape as data or None.
        :param uint pbits: level of verbosity bitword: NONE=0, DEBUG=1, INFO=2, WARNING=4, ERROR=8, CRITICAL=16
        """
        self.mask = mask
        self.r0 = 7
        self.dr = 2
        self.set_peak_selection_pars()
        #self.alg = psalgos.peak_finder_algos(seg, pbits)


    def set_peak_selection_pars(self, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=6) :
        """Sets peak selection parameters.
        """
        self.npix_min = npix_min
        self.npix_max = npix_max
        self.amax_thr = amax_thr
        self.atot_thr = atot_thr
        self.son_min  = son_min


    def peak_finder_v3r3(self, data, rank=5, r0=7, dr=2, nsigm=5, mask=None) :
        """Runs "ranker" peakfinder and returns the list of peak parameters.

        :param np.array data: n-d numpy array or list of 2-d numpy arrays of data 
        :param int rank: radial size or the region [row-rank:row+rank, col-rank:col+rank] where central pixel is a local maximum
        :param float r0: radius [in pixels] of the ring for background evaluation
        :param float dr: width [in pixels] of the ring for background evaluation
        :param float nsigm: threshold on intensity to include pixel in connected group in terms of number of sigma (rms) estimated for pixels in the ring defined by r0 and dr
        :param np.array mask: mask uint16 shaped as data
        :return: np.array peaks: 2-d numpy.array of float peak parameters; 
                 each row contains (seg, row, col, npix, amp_max, amp_tot, 
                 row_cgrav, col_cgrav, row_sigma, col_sigma, row_min, row_max, col_min, col_max, 
                 bkgd, noise, son)
        :rtype numpy.array: ndim=2, dtype=float 
        """
        _mask = mask if mask is not None else self.mask
        _r0 = r0 if r0 is not None else self.r0
        _dr = dr if dr is not None else self.dr
        peaks = peaks_adaptive(data, _mask, rank, _r0, _dr, nsigm,\
                               self.npix_min, self.npix_max, self.amax_thr, self.atot_thr, self.son_min)

        return numpy_2d_arr_of_peak_parameters(peaks)
        #return list_of_peak_parameters(peaks)


    def peak_finder_v4r3(self, data, thr_low=20, thr_high=50, rank=5, r0=7, dr=2, mask=None) :
        """Runs "droplet-finder" and returns the list of peak parameters.

        :param np.array data: n-d numpy array or list of 2-d numpy arrays of data 
        :param float thr_low: low threshold [ADU] - used to include pixel in connected group and select pixels to estimate rms and mean in the ring
        :param float thr_high: high threshold [ADU] - used to search for local maxima candidate
        :param int rank: radial size or the region [row-rank, row+rank] where central pixel is a local maximum
        :param float r0: radius [in pixels] of the ring for background evaluation
        :param float dr: width [in pixels] of the ring for background evaluation
        :param np.array mask: mask uint16 shaped as data
        :return: np.array peaks: 2-d numpy.array of float peak parameters; 
                 each row contains (seg, row, col, npix, amp_max, amp_tot, 
                 row_cgrav, col_cgrav, row_sigma, col_sigma, row_min, row_max, col_min, col_max, 
                 bkgd, noise, son)
        :rtype numpy.array: ndim=2, dtype=float - array of peak parameters.
        """
        _mask = mask if mask is not None else self.mask
        _r0 = r0 if r0 is not None else self.r0
        _dr = dr if dr is not None else self.dr
        peaks = peaks_droplet(data, _mask, thr_low, thr_high, rank, _r0, _dr,\
                               self.npix_min, self.npix_max, self.amax_thr, self.atot_thr, self.son_min)

        return numpy_2d_arr_of_peak_parameters(peaks)
        #return list_of_peak_parameters(peaks)

    def set_son_pars(self, r0=5, dr=0.05) :
        self.r0 = r0 
        self.dr = dr 

    def set_mask(self, mask=None) :
        self.mask = mask

    def set_windows(self, winds=None) :
        pass

    def maps_of_maps_of_pixel_status(self) :
        #self.alg.... works for 2d only
        return None

    def maps_of_local_maximums(self) :
        #self.alg.... works for 2d only
        return None

    def maps_of_connected_pixels(self) :
        #self.alg.... works for 2d only
        return None

    def print_attributes(self) :
        print 'mask    ', self.mask
        print 'r0      ', self.r0
        print 'dr      ', self.dr
        print 'npix_min', self.npix_min
        print 'npix_max', self.npix_max
        print 'amax_thr', self.amax_thr
        print 'atot_thr', self.atot_thr
        print 'son_min ', self.son_min 

    def print_input_pars(self) :
        self.print_attributes()

#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :
    print 'See tests in examples, e.g.,\n  python psalgos/examples/ex-02-localextrema.py 3'

#------------------------------
