#
# cythonize class NDArray
#
#----------

####distutils: language = c++
####distutils: extra_compile_args = ["-std=c++11",] # "-g"]

#----------

#cimport cython

import numpy as np
cimport numpy as cnp
cimport libc.stdint as si
from libcpp.string cimport string

#from libc.time cimport time_t, ctime
#from libc.string cimport memcpy

#----------

ctypedef si.uint32_t shape_t
ctypedef si.uint32_t size_t

#----------

ctypedef fused dtypes2d :
    cnp.ndarray[cnp.double_t, ndim=2]
    cnp.ndarray[cnp.float_t,  ndim=2]
    cnp.ndarray[cnp.int16_t,  ndim=2]
    cnp.ndarray[cnp.int32_t,  ndim=2]
    cnp.ndarray[cnp.int64_t,  ndim=2]
    cnp.ndarray[cnp.uint16_t, ndim=2]
    cnp.ndarray[cnp.uint32_t, ndim=2]
    cnp.ndarray[cnp.uint64_t, ndim=2]

#----------

cdef extern from "psalg/hexanode/ctest_utils.hh" namespace "psalg":
    void ctest_nda[T](T *arr, int r, int c) except +

def test_nda_fused(dtypes2d nda):
    print('nda.dtype =', str(nda.dtype))
    ctest_nda(&nda[0,0], nda.shape[0], nda.shape[1])

#----------

from libcpp.vector cimport vector
ctypedef vector[double] cvector_t

#ctypedef fused cvector_t :
#    vector[double]
#    vector[float]
#    vector[int]

cdef extern from "psalg/hexanode/ctest_utils.hh" namespace "psalg":
    double ctest_vector(cvector_t &v)

def py_ctest_vector(v):
    ctest_vector(v)

#----------

cdef extern from "xtcdata/xtc/Array.hh" namespace "XtcData":
    cdef cppclass Array[T]:
        Array() except+
        Array(void *data, shape_t *shape, size_t rank) except+
        Array(const Array& o) except+
        Array& operator=(Array& o) except+

        T& operator()(unsigned i)
        T& operator()(unsigned i,unsigned j)
        T& operator()(unsigned i,unsigned j,unsigned k)
        T& operator()(unsigned i,unsigned j,unsigned k,unsigned l)
        T& operator()(unsigned i,unsigned j,unsigned k,unsigned l,unsigned m)
        cnp.uint32_t rank() const
        shape_t* shape() const
        T* data()
        const T* const_data() const
        cnp.uint64_t num_elem()
        void shape(size_t a, size_t b=0, size_t c=0, size_t d=0, size_t e=0)
        void set_rank(size_t rank)
        void set_shape(shape_t *shape)
        void set_data(void *data)

#----------

cdef extern from "psalg/calib/NDArray.hh" namespace "psalg":
    cdef cppclass NDArray[T](Array[T]):
        NDArray() except+
        NDArray(const shape_t* sh, const size_t ndim, void *buf_ext=0) except+
        NDArray(const shape_t* sh, const size_t ndim, const void *buf_ext) except+
        NDArray(const NDArray[T]& o) except+

        cnp.uint32_t* shape()
        cnp.uint32_t ndim()
        T* data()
        const T* const_data()
        void set_shape(const shape_t* shape=0, const size_t ndim=0)
        void set_shape_string(const string&)
        void reshape(const shape_t*, const size_t)
        size_t size() const
        void set_ndarray(Array[T]&)
        void set_ndarray(NDArray[T]&)
        void set_const_data_buffer(const void *buf)
        void set_data_buffer(void *buf_ext)
        void set_data_copy(const void *buf)
        void reserve_data_buffer(const size_t)

#----------

#ctypedef fused ndatypes:
#    NDArray[cnp.double_t]
#    NDArray[cnp.float_t]
#    NDArray[cnp.int16_t]
#    NDArray[cnp.int32_t]
#    NDArray[cnp.int64_t]
#    NDArray[cnp.uint16_t]
#    NDArray[cnp.uint32_t]
#    NDArray[cnp.uint64_t]

#----------

cdef class py_ndarray_double:
    """ Python wrapper for C++ class NDArray. 
    """

    cdef NDArray[double]* cptr # holds a C++ instance

    def __cinit__(self):
        print("In py_ndarray_double.__cinit__")
        self.cptr = new NDArray[double]();

    def set_nda(self, dtypes2d nda):
        print("In py_ndarray_double.set_nda")
        print('nda.dtype =', str(nda.dtype))

    def __dealloc__(self):
        print("In py_ndarray_double.__dealloc__")
        if self.cptr is not NULL :
            del self.cptr
            self.cptr = NULL

#----------
