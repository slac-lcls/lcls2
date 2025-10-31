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

cdef extern from "psalg/alloc/AllocArray.hh" namespace "psalg":
    cdef cppclass Array[T]:
        Array() except+
        cnp.uint32_t *shape()
        T *data()
        cnp.uint32_t& _refCnt()
        T& operator()(unsigned i)
    cdef cppclass AllocArray[T](Array[T]):
        pass
    cdef cppclass AllocArray1D[T](AllocArray[T]):
        pass

cdef extern from "psalg/alloc/Allocator.hh":
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

# An AllocArray1D python class that deallocates our C AllocArray1D
# (by decrementing its reference count) when the Python numpy array (returned
# from init()) is deleted.

ctypedef fused arrf:
    AllocArray1D[float]
    AllocArray1D[cnp.uint8_t]
    AllocArray1D[cnp.uint16_t]

# this follows the example here:
# https://gist.github.com/GaelVaroquaux/1249305
cdef class PyAllocArray1D:
    cdef cnp.uint32_t* refcnt_ptr
    cdef void* shape_ptr
    cdef void* data_ptr
    cdef int size
    cdef int typenum
    cdef cnp.ndarray arr

    cdef init(self, arrf* carr, int size, int typenum):
        self.refcnt_ptr = &(carr._refCnt())
        self.shape_ptr = carr.shape()
        self.data_ptr  = carr.data()
        self.size = size
        self.typenum = typenum
        # Increment the reference count of C array
        self.refcnt_ptr[0] += 1
        # Convert to numpy array, which calls __array__
        arr = np.asarray(self)
        return arr

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = cnp.PyArray_SimpleNewFromData(1, shape, self.typenum, self.data_ptr)

        # Increment the reference count, as the above assignment was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(self)

        return ndarray

    def __dealloc__(self):
        """ This is called by Python when all the
        references to the object are gone. """
        # Decrement the reference count of C array
        self.refcnt_ptr[0] -= 1

################# Peak Finder ######################

from libcpp.vector cimport vector
cimport libc.stdint as si

cdef extern from "include/PeakFinderAlgos.hh" namespace "psalgos":
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

cdef extern from "include/Types.hh" namespace "types":
    ctypedef unsigned shape_t
    ctypedef si.uint16_t mask_t
    ctypedef si.uint16_t extrim_t
    ctypedef si.uint32_t conmap_t

ctypedef fused nptype1d :
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"]
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]
    cnp.ndarray[cnp.int16_t,   ndim=1, mode="c"]
    cnp.ndarray[cnp.int32_t,   ndim=1, mode="c"]
    cnp.ndarray[cnp.int64_t,   ndim=1, mode="c"]
    cnp.ndarray[cnp.uint16_t,  ndim=1, mode="c"]
    cnp.ndarray[cnp.uint32_t,  ndim=1, mode="c"]
    cnp.ndarray[cnp.uint64_t,  ndim=1, mode="c"]

ctypedef fused nptype2d :
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"]
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]
    cnp.ndarray[cnp.int16_t,   ndim=2, mode="c"]
    cnp.ndarray[cnp.int32_t,   ndim=2, mode="c"]
    cnp.ndarray[cnp.int64_t,   ndim=2, mode="c"]
    cnp.ndarray[cnp.uint16_t,  ndim=2, mode="c"]
    cnp.ndarray[cnp.uint32_t,  ndim=2, mode="c"]
    cnp.ndarray[cnp.uint64_t,  ndim=2, mode="c"]

cdef extern from "include/PeakFinderAlgos.hh" namespace "psalgos":
    cdef cppclass PeakFinderAlgos:
         AllocArray1D[float] rows
         AllocArray1D[float] cols
         AllocArray1D[float] intens
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
        rows_cgrav = arr0.init(&self.cptr.rows, self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        cnp.PyArray_SetBaseObject(rows_cgrav, arr0)

        arr1 = PyAllocArray1D()
        cols_cgrav = arr1.init(&self.cptr.cols, self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        cnp.PyArray_SetBaseObject(cols_cgrav, arr1)

        arr2 = PyAllocArray1D()
        intens = arr2.init(&self.cptr.intens, self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        cnp.PyArray_SetBaseObject(intens, arr2)

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

#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------
#------------------------------

cdef extern from "include/LocalExtrema.hh" namespace "localextrema":

    size_t localMinima1d[T](const T *data
                           ,const mask_t *mask
                           ,const size_t& cols
                           ,const size_t& stride
                           ,const size_t& rank
                           ,extrim_t *arr1d
                           )

    size_t localMaxima1d[T](const T *data
                           ,const mask_t *mask
                           ,const size_t& cols
                           ,const size_t& stride
                           ,const size_t& rank
                           ,extrim_t *arr1d
                           )

    size_t mapOfLocalMinimums[T](const T *data
                                ,const mask_t *mask
                                ,const size_t& rows
                                ,const size_t& cols
                                ,const size_t& rank
                                ,extrim_t *arr2d
                                )

    size_t mapOfLocalMaximums[T](const T *data
                                ,const mask_t *mask
                                ,const size_t& rows
                                ,const size_t& cols
                                ,const size_t& rank
                                ,extrim_t *arr2d
                                )

    size_t mapOfLocalMaximumsRank1Cross[T](const T *data
                                          ,const mask_t *mask
                                          ,const size_t& rows
                                          ,const size_t& cols
                                          ,extrim_t *arr2d
                                          )

    size_t mapOfThresholdMaximums[T](const T *data
                                    ,const mask_t *mask
                                    ,const size_t& rows
                                    ,const size_t& cols
                                    ,const size_t& rank
		                    ,const double& thr_low 
		                    ,const double& thr_high 
                                    ,extrim_t *arr2d
                                    )

    void printMatrixOfDiagIndexes(const size_t& rank)
    void printVectorOfDiagIndexes(const size_t& rank)


def local_minima_1d(nptype1d data,\
                    cnp.ndarray[mask_t, ndim=1, mode="c"] mask,\
                    si.int32_t rank,\
                    cnp.ndarray[extrim_t, ndim=1, mode="c"] arr1d\
                   ):
    stride = 1;
    #print 'XXX psalgos_ext.py - data.size: %d   rank: %d' % (data.size, rank)
    return localMinima1d(&data[0], &mask[0], data.size, stride, rank, &arr1d[0])


def local_maxima_1d(nptype1d data,\
                    cnp.ndarray[mask_t, ndim=1, mode="c"] mask,\
                    si.int32_t rank,\
                    cnp.ndarray[extrim_t, ndim=1, mode="c"] arr1d\
                   ):
    stride = 1;
    #print 'XXX psalgos_ext.py - data.size: %d   rank: %d' % (data.size, rank)
    return localMaxima1d(&data[0], &mask[0], data.size, stride, rank, &arr1d[0])


def local_minimums(nptype2d data,\
                   cnp.ndarray[mask_t, ndim=2, mode="c"] mask,\
                   si.int32_t rank,\
                   cnp.ndarray[extrim_t, ndim=2, mode="c"] arr2d\
                  ): 
    return mapOfLocalMinimums(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, &arr2d[0,0])


def local_maximums(nptype2d data,\
                   cnp.ndarray[mask_t, ndim=2, mode="c"] mask,\
                   si.int32_t rank,\
                   cnp.ndarray[extrim_t, ndim=2, mode="c"] arr2d\
                  ): 
    return mapOfLocalMaximums(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, &arr2d[0,0])


def local_maximums_rank1_cross(nptype2d data,\
                   cnp.ndarray[mask_t, ndim=2, mode="c"] mask,\
                   cnp.ndarray[extrim_t, ndim=2, mode="c"] arr2d\
                  ): 
    return mapOfLocalMaximumsRank1Cross(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], &arr2d[0,0])


def threshold_maximums(nptype2d data,\
                       cnp.ndarray[mask_t, ndim=2, mode="c"] mask,\
                       si.int32_t rank,\
                       double thr_low,\
                       double thr_high,\
                       cnp.ndarray[extrim_t, ndim=2, mode="c"] arr2d\
                      ) : 
    return mapOfThresholdMaximums(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], rank, thr_low, thr_high, &arr2d[0,0])


def print_matrix_of_diag_indexes(si.int32_t& rank) : printMatrixOfDiagIndexes(rank)


def print_vector_of_diag_indexes(si.int32_t& rank) : printVectorOfDiagIndexes(rank)

#------------------------------




