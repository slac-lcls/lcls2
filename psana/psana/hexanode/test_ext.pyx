
# Example:

#from libcpp.string cimport string
#from libc.time cimport time_t, ctime
#import numpy as np
#cimport numpy as np

#------------------------------

#cdef extern from "<stdint.h>" nogil:
#    ctypedef   signed char  int8_t
#    ctypedef   signed short int16_t
#    ctypedef   signed int   int32_t
#    ctypedef   signed long  int64_t
#    ctypedef unsigned char  uint8_t
#    ctypedef unsigned short uint16_t
#    ctypedef unsigned int   uint32_t
#    ctypedef unsigned long  uint64_t

#ctypedef fused dtypes2d :
#    np.ndarray[np.double_t, ndim=2]
#    np.ndarray[np.int16_t,  ndim=2]
#    np.ndarray[np.uint16_t, ndim=2]

#------------------------------
#------------------------------
#------------------------------
# TESTS
#------------------------------
#------------------------------
#------------------------------

#def met1():
#    """cython method for test call from python"""
#    cdef int nelem = 1
#    print "HELL of cython is here"


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

