#
# cythonize utils from modules of lcls2/psana/psana/pycalgos/
#    UtilsDetector.cc/hh etc.

# command to build this module only, see lcls2/psana/setup.py
# ./build_all.sh -m -d -b PYCALGOS

# import psana.pycalgos.utilsdetector_ext as pcud

import numpy as np
cimport numpy as cnp
cimport libc.stdint as si
#from libcpp.string cimport string
#from libcpp.vector cimport vector
#from libcpp cimport bool

#ctypedef fused dtypes1d :
#    cnp.ndarray[cnp.float64_t,ndim=1]
#    cnp.ndarray[cnp.float32_t,ndim=1]
#    cnp.ndarray[cnp.int64_t,  ndim=1]
#    cnp.ndarray[cnp.int32_t,  ndim=1]
#    cnp.ndarray[cnp.int16_t,  ndim=1]

#ctypedef fused dtypes1d :
#    cnp.ndarray[cnp.double_t, ndim=2]
#    cnp.ndarray[cnp.float_t,  ndim=2]
#    cnp.ndarray[cnp.uint16_t, ndim=2]
#    cnp.ndarray[cnp.uint32_t, ndim=2]
#    cnp.ndarray[cnp.uint64_t, ndim=2]

#ctypedef si.uint32_t  size_t
#ctypedef unsigned     size_t;

ctypedef si.uint8_t    mask_t;
ctypedef cnp.uint16_t  rawd_t;
ctypedef cnp.float32_t peds_t;
ctypedef cnp.float32_t gain_t;
ctypedef cnp.float32_t cc_t;
ctypedef cnp.float32_t out_t;

#cdef extern from "pycalgos/UtilsDetector.hh" namespace "utilsdetector":
#  cdef struct ccst:
#    peds_t pedestal
#    gain_t gain


cdef extern from "pycalgos/UtilsDetector.hh" namespace "utilsdetector":
    cnp.double_t calib_std(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask,
                           const size_t& size, const rawd_t databits, out_t *out) except +


cdef extern from "pycalgos/UtilsDetector.hh" namespace "utilsdetector":
    cnp.double_t calib_jungfrau_v0(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask,
                           const size_t& size, out_t *out) except +


cdef extern from "pycalgos/UtilsDetector.hh" namespace "utilsdetector":
    cnp.double_t calib_jungfrau_v1(const rawd_t *raw, const cc_t *cc,
                           const size_t& size, const size_t& size_blk, out_t *out) except +


def cy_calib_std(cnp.ndarray[rawd_t, ndim=1] raw,
                 cnp.ndarray[peds_t, ndim=1] peds,
		 cnp.ndarray[gain_t, ndim=1] gain,
		 cnp.ndarray[mask_t, ndim=1] mask,
		 size_t size,
		 rawd_t databits,
		 cnp.ndarray[out_t, ndim=1] out):
    return calib_std(&raw[0], &peds[0], &gain[0], &mask[0], size, databits, &out[0]) # returns double time, us


def cy_calib_jungfrau_v0(cnp.ndarray[rawd_t, ndim=1] raw,
                 cnp.ndarray[peds_t, ndim=1] peds,
		 cnp.ndarray[gain_t, ndim=1] gain,
		 cnp.ndarray[mask_t, ndim=1] mask,
		 size_t size,
		 cnp.ndarray[out_t, ndim=1] out):
    return calib_jungfrau_v0(&raw[0], &peds[0], &gain[0], &mask[0], size, &out[0]) # returns double time, us


def cy_calib_jungfrau_v1(cnp.ndarray[rawd_t, ndim=1] raw,
                 cnp.ndarray[peds_t, ndim=1] cc,
		 size_t size,
		 size_t size_blk,
		 cnp.ndarray[out_t, ndim=1] out):
    return calib_jungfrau_v1(&raw[0], &cc[0], size, size_blk, &out[0]) # returns double time, us


#    print('raw.dtype =', str(raw.dtype))

# EOF
