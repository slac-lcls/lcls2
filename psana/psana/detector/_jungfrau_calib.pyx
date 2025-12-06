# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython helpers for Jungfrau calibration math.
"""

import numpy as np
cimport numpy as cnp

ctypedef cnp.uint16_t u16
ctypedef cnp.float32_t f32

cdef inline int _gain_index(u16 raw_value) nogil:
    cdef int g = raw_value >> 14  # gain bits in the two MSBs
    if g == 3:
        return 2
    if g == 0 or g == 1:
        return g
    # Combination 2 (10b) is treated as default/raw.
    return -1


cpdef void calibrate_panels(cnp.ndarray[u16, ndim=3] raw,
                            cnp.ndarray[f32, ndim=4] pedoffs,
                            cnp.ndarray[f32, ndim=4] gains,
                            object mask,
                            cnp.ndarray[f32, ndim=3] out):
    """
    Calibrate Jungfrau panels by applying pedestals, offsets, gains, and mask.
    The pedoffs/gains arrays must already be ordered to match the panel order
    present in ``raw`` (shape: (3, n_panels, 512, 1024)).
    """

    cdef bint use_mask = mask is not None
    cdef cnp.ndarray[f32, ndim=3] mask_arr
    if use_mask:
        mask_arr = <cnp.ndarray[f32, ndim=3]>mask

    cdef Py_ssize_t nseg = raw.shape[0]
    cdef Py_ssize_t rows = raw.shape[1]
    cdef Py_ssize_t cols = raw.shape[2]
    cdef Py_ssize_t seg, row, col
    cdef u16 rawv
    cdef int gidx
    cdef f32 ped, gain, mval, result

    for seg in range(nseg):
        for row in range(rows):
            for col in range(cols):
                rawv = raw[seg, row, col]
                gidx = _gain_index(rawv)
                if gidx >= 0:
                    ped = pedoffs[gidx, seg, row, col]
                    gain = gains[gidx, seg, row, col]
                else:
                    ped = 0.0
                    gain = 1.0

                result = ((rawv & 0x3fff) - ped) * gain
                if use_mask:
                    mval = mask_arr[seg, row, col]
                    result *= mval
                out[seg, row, col] = result

