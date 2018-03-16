# TODO
# ** should outputs be arrays instead of Peak class
# ** psalg should be built with cmake to produce a .so
# ** the cython .pyx should be in psana and built in psana
# ** what array class are we going to use?
# *  remove main class from pebble, so it only gets created once
# remove the "if" statement around placement-new and non-placement-new
# every method gets a drp-pointer
# calling push_back could malloc?
# pebble overwrite protection (bounds check) if arrays get too big

import sys # flush

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

################# Psana Array ######################

cdef extern from "../../../psalg/psalg/include/AllocArray.hh" namespace "psalg":
    cdef cppclass Array[T]:
        Array() except+
        cnp.uint32_t *shape()
        T *data()
        void incRefCnt()

cdef extern from "../../../psalg/psalg/include/Allocator.hh":
    cdef cppclass Allocator:
        pass
    cdef cppclass Heap(Allocator):
        pass

################# PyAllocArray1D ######################

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class PyAllocArray1D:
    cdef void* shape_ptr
    cdef void* data_ptr
    cdef int size
    cdef int typenum

    cdef init(self, void* buffer_ptr, void* data_ptr, int size, int typenum):
        self.shape_ptr = buffer_ptr
        self.data_ptr  = data_ptr
        self.size = size
        self.typenum = typenum

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = cnp.PyArray_SimpleNewFromData(1, shape, self.typenum, self.data_ptr)

        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(self)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.shape_ptr)

################# Peak Finder ######################

from libcpp.vector cimport vector
cimport libc.stdint as si

cdef extern from "../../../psalg/psalg/include/PeakFinderAlgos.h" namespace "psalgos":
    cdef cppclass Peak :
        Peak() except +
        Peak(const Peak& o) except +
        Peak operator=(const Peak& rhs) except +
        float seg
        float row
        float col
        float npix
        float amp_max
        float amp_tot
        float row_cgrav 
        float col_cgrav
        float row_sigma
        float col_sigma
        float row_min
        float row_max
        float col_min
        float col_max
        float bkgd
        float noise
        float son

cdef extern from "../../../psalg/psalg/include/Types.h" namespace "types":
    ctypedef unsigned shape_t
    ctypedef si.uint16_t mask_t
    ctypedef si.uint16_t extrim_t
    ctypedef si.uint32_t conmap_t

ctypedef fused nptype2d :
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"]
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]
    cnp.ndarray[cnp.int16_t,   ndim=2, mode="c"]
    cnp.ndarray[cnp.int32_t,   ndim=2, mode="c"]
    cnp.ndarray[cnp.int64_t,   ndim=2, mode="c"]
    cnp.ndarray[cnp.uint16_t,  ndim=2, mode="c"]
    cnp.ndarray[cnp.uint32_t,  ndim=2, mode="c"]
    cnp.ndarray[cnp.uint64_t,  ndim=2, mode="c"]

cdef extern from "../../../psalg/psalg/include/PeakFinderAlgos.h" namespace "psalgos":
    cdef cppclass PeakFinderAlgos:
         Array[float] rows
         Array[float] cols
         Array[float] intens
         unsigned numPeaksSelected

         PeakFinderAlgos(Allocator *allocator, const size_t& seg, const unsigned& pbits, const size_t& lim_rank, const size_t& lim_peaks) except +

         void setPeakSelectionPars(const float& npix_min
                                  ,const float& npix_max
                                  ,const float& amax_thr
                                  ,const float& atot_thr
                                  ,const float& son_min)

         void peakFinderV3r3[T](const T *data
                               ,const mask_t *mask
                               ,const size_t& rows
                               ,const size_t& cols
                               ,const size_t& rank
                               ,const double& r0
                               ,const double& dr
                               ,const double& nsigm)
 
         void printParameters();

         void localMaxima    (extrim_t *arr2d, const size_t& rows, const size_t& cols)

         void localMinima    (extrim_t *arr2d, const size_t& rows, const size_t& cols)

         void connectedPixels(conmap_t *arr2d, const size_t& rows, const size_t& cols)

cdef class py_peak :
    cdef Peak* cptr  # holds a C++ pointer to instance

    def __cinit__(self, _make_obj=True):
        #print "In py_peak.__cinit__"
        if _make_obj:
            self.cptr = new Peak()

    def __dealloc__(self):
        #print "In py_peak.__dealloc__"
        if self.cptr is not NULL :
            del self.cptr
            self.cptr = NULL


    # https://groups.google.com/forum/#!topic/cython-users/39Nwqsksdto
    @staticmethod
    cdef factory(Peak cpp_obj):
        py_obj = py_peak.__new__(py_peak, _make_obj=False)
        (<py_peak> py_obj).cptr = new Peak(cpp_obj) # C++ copy constructor
        return py_obj

    def parameters(self) : #, cpp_obj=None) : 
        p = self.cptr     # if cpp_obj is None else <Peak*> &cpp_obj
        return (p.seg, p.row, p.col, p.npix, p.amp_max, p.amp_tot,\
                p.row_cgrav, p.col_cgrav, p.row_sigma, p.col_sigma,\
                p.row_min, p.row_max, p.col_min, p.col_max, p.bkgd, p.noise, p.son)

    @property
    def seg      (self) : return self.cptr.seg       

    @property
    def row      (self) : return self.cptr.row       

    @property
    def col      (self) : return self.cptr.col       

    @property
    def npix     (self) : return self.cptr.npix      

    @property
    def amp_max  (self) : return self.cptr.amp_max   

    @property
    def amp_tot  (self) : return self.cptr.amp_tot   

    @property
    def row_cgrav(self) : return self.cptr.row_cgrav 

    @property
    def col_cgrav(self) : return self.cptr.col_cgrav 

    @property
    def row_sigma(self) : return self.cptr.row_sigma 

    @property
    def col_sigma(self) : return self.cptr.col_sigma 

    @property
    def row_min  (self) : return self.cptr.row_min   

    @property
    def row_max  (self) : return self.cptr.row_max   

    @property
    def col_min  (self) : return self.cptr.col_min   

    @property
    def col_max  (self) : return self.cptr.col_max   

    @property
    def bkgd     (self) : return self.cptr.bkgd      

    @property
    def noise    (self) : return self.cptr.noise     

    @property
    def son      (self) : return self.cptr.son    

#------------------------------

cdef class peak_finder_algos :
    """ Python wrapper for C++ class. 
    """
    cdef PeakFinderAlgos* cptr  # holds a C++ pointer to instance
    cdef si.uint16_t rows, cols
    cdef Heap heap
    cdef Heap *hptr

    def __cinit__(self, seg=0, pbits=0, lim_rank=50, lim_peaks=4096):
        self.hptr = &self.heap
        self.cptr = new PeakFinderAlgos(self.hptr, seg, pbits, lim_rank, lim_peaks)

    def __dealloc__(self):
        del self.cptr

    def set_peak_selection_parameters(self\
                                     ,const float& npix_min\
                                     ,const float& npix_max\
                                     ,const float& amax_thr\
                                     ,const float& atot_thr\
                                     ,const float& son_min) :
        self.cptr.setPeakSelectionPars(npix_min, npix_max, amax_thr, atot_thr, son_min)

    def print_attributes(self) : self.cptr.printParameters()

    def peak_finder_v3r3_d2(self\
                           ,nptype2d data\
                           ,cnp.ndarray[mask_t, ndim=2, mode="c"] mask\
                           ,const size_t& rank\
                           ,const double& r0\
                           ,const double& dr\
                           ,const double& nsigm) :
        self.cptr.peakFinderV3r3(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, r0, dr, nsigm)
        return self.getPeaks()

    def getPeaks(self):
        cdef cnp.ndarray rows_cgrav, cols_cgrav, intens # make readonly

        arr0 = PyAllocArray1D()
        arr0.init(<void*> self.cptr.rows.shape(), <void*> self.cptr.rows.data(), self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        rows_cgrav = np.array(arr0, copy=False)
        rows_cgrav.base = <PyObject*> arr0
        self.cptr.rows.incRefCnt() # increment ref count so that C++ doesn't delete array

        arr1 = PyAllocArray1D()
        arr1.init(<void*> self.cptr.cols.shape(), <void*> self.cptr.cols.data(), self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        cols_cgrav = np.array(arr1, copy=False)
        cols_cgrav.base = <PyObject*> arr1
        self.cptr.cols.incRefCnt()

        arr2 = PyAllocArray1D()
        arr2.init(<void*> self.cptr.intens.shape(), <void*> self.cptr.intens.data(), self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        intens     = np.array(arr2, copy=False)
        intens.base     = <PyObject*> arr2
        self.cptr.intens.incRefCnt()

        return rows_cgrav, cols_cgrav, intens

    def local_maxima(self) :
        sh = (self.rows, self.cols)
        cdef cnp.ndarray[extrim_t, ndim=2] arr2d = np.ascontiguousarray(np.empty(sh, dtype=np.uint16))
        self.cptr.localMaxima(&arr2d[0,0], self.rows, self.cols)
        return arr2d


    def local_minima(self) :
        sh = (self.rows, self.cols)
        cdef cnp.ndarray[extrim_t, ndim=2] arr2d = np.ascontiguousarray(np.empty(sh, dtype=np.uint16))
        self.cptr.localMinima(&arr2d[0,0], self.rows, self.cols)
        return arr2d


    def connected_pixels(self) :
        sh = (self.rows, self.cols)
        cdef cnp.ndarray[conmap_t, ndim=2] arr2d = np.ascontiguousarray(np.empty(sh, dtype=np.uint32))
        self.cptr.connectedPixels(&arr2d[0,0], self.rows, self.cols)
        return arr2d







