
# Example:

#from cpython.bool cimport *
#from cpython.float cimport *
#from cpython.int cimport *
#from cpython.list cimport *
#from cpython.object cimport *
#from cpython.ref cimport *
#from cpython.string cimport *

from libcpp.string cimport string
#from libcpp.vector cimport vector
#from libcpp cimport bool

from libc.time cimport time_t, ctime

import numpy as np
cimport numpy as np

#np.import_array()

#from cython.operator cimport dereference as deref

#------------------------------

cdef extern from "<stdint.h>" nogil:
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long  uint64_t


ctypedef fused dtypes2d :
    np.ndarray[np.double_t, ndim=2]
    np.ndarray[np.int16_t,  ndim=2]
    np.ndarray[np.uint16_t, ndim=2]

#------------------------------
#------------------------------
#------------------------------
# TESTS
#------------------------------
#------------------------------
#------------------------------

def met1():
    """cython method for test call from python"""
    cdef int nelem = 1
    print "HELL of cython is here"


#cdef extern from "psalg/hexanode/cfib.hh":
#cdef extern from "/reg/neh/home/dubrovin/LCLS/con-lcls2/lcls2/install/include/psalg/hexanode/cfib.hh" namespace "psalgos":
#cdef extern from "/reg/neh/home/dubrovin/LCLS/con-lcls2/lcls2/install/include/psalg/hexanode/cfib.hh":
cdef extern from "psalg/hexanode/cfib.hh":
    double cfib(int n)


def fib(n):
    """Returns the nth Fibonacci number"""
    return cfib(n)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

