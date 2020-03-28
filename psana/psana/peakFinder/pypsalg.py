#!/usr/bin/env python
#------------------------------
"""
Class :py:class:`pypsalg` convenience python wrapper for C++/cython peak-finding algorithms.

Usage::

    # IMPORT MISC
    # ===========
    import numpy as np
    from psana.pyalgos.generic.NDArrUtils import print_ndarr
    #from psana.peakFinder.pypsalg import peaks_adaptive, peaks_droplet

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

    import psana.peakFinder.pypsalg as algos

    alg = algos.peak_finder_algos(seg=0, pbits=0)
    alg.set_peak_selection_parameters(npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=6)

    peaks = alg.peak_finder_v3r3_d2(data, mask, rank=5, r0=7, dr=2, nsigm=5)
    peaks = alg.peak_finder_v4r3_d2(data, mask, thr_low=20, thr_high=50, rank=5, r0=7, dr=2)

    peaks = alg.list_of_peaks_selected()
    peaks_all = alg.list_of_peaks()

    p = algo.peak(0)
    p = algo.peak_selected(0)

    for p in peaks : print('  row:%4d, col:%4d, npix:%4d, son::%4.1f' % (p.row, p.col, p.npix, p.son))
    for p in peaks : print(p.parameters())

    map_u2 = alg.local_maxima()
    map_u2 = alg.local_minima()
    map_u4 = alg.connected_pixels()

    import psana.peakFinder.pypsalg as algos

    n = algos.local_minima_2d(data, mask, rank, extrema) # 0.019(sec) for shape=(1000, 1000) on psanaphi110
    n = algos.local_maxima_2d(data, mask, rank, extrema) # 0.019(sec)
    n = algos.local_maxima_rank1_cross_2d(data, mask, extrema) # 0.014(sec)
    n = algos.threshold_maxima_2d(data, mask, rank, thr_low, thr_high, extrema) # 0.0017(sec)

    # DIRECT CALL PEAKFINDERS
    # =======================

    from psana.peakFinder.pypsalg import peaks_adaptive, peaks_droplet, peaks_adaptive_2d, peaks_droplet_2d

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

Created: 2017-08-10 by Mikhail Dubrovin
         2020-03-05 adapted as lcls/2psana/psana/peakFinder/pypsalg.py
"""

#------------------------------
#from psalg_ext import peak_finder_algos
import psalg_ext as algos
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
    return algos.local_minima_1d(data, _mask, rank, extrema)

#local_minimums_1d = local_minima_1d

#------------------------------

def local_maxima_1d(data, mask=None, rank=3, extrema=None) :
    """Finds local maxima of specified rank in 1-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return algos.local_maxima_1d(data, _mask, rank, extrema)

#local_maximums_1d = local_maxima_1d

#------------------------------

def local_minima_2d(data, mask=None, rank=3, extrema=None) :
    """Finds local minima of specified rank in 2-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return algos.local_minimums(data, _mask, rank, extrema)

#local_minimums_2d = local_minima_2d

#------------------------------

def local_maxima_2d(data, mask=None, rank=3, extrema=None) :
    """Finds local maxima of specified rank in 2-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return algos.local_maximums(data, _mask, rank, extrema)

#local_maximums_2d = local_maxima_2d

#------------------------------

def local_maxima_rank1_cross_2d(data, mask=None, extrema=None) :
    """Finds local maxima rank=1 cross in 2-d array of data with mask and returns results in array extrema.
    """
    if extrema is None : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return algos.local_maximums_rank1_cross(data, _mask, extrema)

#local_maximums_rank1_cross_2d = local_maxima_rank1_cross_2d

#------------------------------

def threshold_maxima_2d(data, mask=None, rank=3, thr_low=None, thr_high=None, extrema=None) :
    """Finds local maxima using 2-threshold algorithm in 2-d array of data with mask and returns results in array extrema.
    """
    if (thr_low is None) or (thr_high is None) or (extrema is None) : return
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)
    return algos.threshold_maximums(data, _mask, rank, thr_low, thr_high, extrema)

#threshold_maximums_2d = threshold_maxima_2d

#------------------------------

def print_matrix_of_diag_indexes(rank=5) : 
    """Prints matrix of diagonal indexes for local_maxima_2d and local_minima_2d algorithms.
    """
    algos.print_matrix_of_diag_indexes(rank)

#------------------------------

def print_vector_of_diag_indexes(rank=5) : 
    """Prints vector of diagonal indexes for local_maxima_2d and local_minima_2d algorithms.
    """
    algos.print_vector_of_diag_indexes(rank)

#------------------------------

def peak_finder_alg_2d(seg=0, pbits=0) :
    """Creates and returns algorithm object.
    """
    return algos.peak_finder_algos(seg, pbits)

#------------------------------

def peaks_adaptive_2d(data, mask=None, rank=5, r0=7.0, dr=2.0, nsigm=5,\
                      seg=0, npix_min=1, npix_max=None, amax_thr=0, atot_thr=0, son_min=8) :
    """Peak finder for 2-d numpy arrays of data and mask of the same shape
    """
    #from time import time

    #t0_sec = time()
    # pbits NONE=0, DEBUG=1, INFO=2, WARNING=4, ERROR=8, CRITICAL=16

    o = algos.peak_finder_algos(seg, pbits=0) #377) # ~17 microsecond
    _npix_max = npix_max if npix_max is not None else (2*rank+1)*(2*rank+1)
    o.set_peak_selection_parameters(npix_min, _npix_max, amax_thr, atot_thr, son_min)
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)

    peaks = o.peak_finder_v3r3_d2(data, _mask, rank, r0, dr, nsigm)

    #print('peak_finder_v3r3_d2: total time = %.6f(sec)' % (time()-t0_sec))
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

    o = algos.peak_finder_algos(seg, pbits=0) #377) # ~17 microsecond
    _npix_max = npix_max if npix_max is not None else (2*rank+1)*(2*rank+1)
    o.set_peak_selection_parameters(npix_min, _npix_max, amax_thr, atot_thr, son_min)
    _mask = mask if mask is not None else np.ones(data.shape, dtype=np.uint16)

    peaks = o.peak_finder_v4r3_d2(data, _mask, thr_low, thr_high, rank, r0, dr)

    #print('peak_finder_v3r3_d2: total time = %.6f(sec)' % (time()-t0_sec))
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

        else : raise IOError('algos.peak_finder_v3r3: wrong data.ndim %s' % str(data.ndim))

    else : raise IOError('algos.peak_finder_v3r3: unexpected object type for data: %s' % str(data))

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

        else : raise IOError('algos.peak_finder_v3r3: wrong data.ndim %s' % str(data.ndim))

    else : raise IOError('algos.peak_finder_v3r3: unexpected object type for data: %s' % str(data))

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

if __name__ == "__main__" :
    print('See tests in examples, e.g.,\n  python psalgos/examples/ex-02-localextrema.py 3')

#------------------------------
