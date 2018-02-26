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

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

################# Array Wrapper ######################

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size
    cdef int typenum

    cdef set_data(self, void* data_ptr, int size, int typenum):
        """ Set the data of the array

        This cannot be done in the constructor as it must recieve C-level
        arguments.

        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data

        """
        self.data_ptr = data_ptr
        self.size = size
        self.typenum = typenum

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = cnp.PyArray_SimpleNewFromData(1, shape, self.typenum, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        print("Dealloc memory")
        free(<void*>self.data_ptr)

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
         unsigned ps_row
         unsigned ps_col
         float *rows
         float *cols
         float *intens
         unsigned numPeaksSelected

         PeakFinderAlgos(const size_t& seg, const unsigned& pbits, si.uint8_t* drpPtr) except +

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

         void peakFinderV4r3[T](const T *data
                               ,const mask_t *mask
                               ,const size_t& rows
                               ,const size_t& cols
                               ,const double& thr_low
                               ,const double& thr_high
                               ,const size_t& rank
	                       ,const double& r0
	                       ,const double& dr
                               )
 
         void printParameters();

         const Peak& peak(const int& i)

         const Peak& peakSelected(const int& i)

         const vector[Peak]& vectorOfPeaks()

         const vector[Peak]& vectorOfPeaksSelected()

         void localMaxima    (extrim_t *arr2d, const size_t& rows, const size_t& cols)

         void localMinima    (extrim_t *arr2d, const size_t& rows, const size_t& cols)

         void connectedPixels(conmap_t *arr2d, const size_t& rows, const size_t& cols)

         const vector[vector[float]] peaksSelected()
         float *convPeaksSelected()

cdef class py_peak :
    cdef Peak* cptr  # holds a C++ pointer to instance

    def __cinit__(self, _make_obj=True):
        #print "In py_peak.__cinit__"
        if _make_obj:
            self.cptr = new Peak()

    def __dealloc__(self):
        print "In py_peak.__dealloc__"
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

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class peak_finder_algos :
    """ Python wrapper for C++ class. 
    """
    cdef PeakFinderAlgos* cptr  # holds a C++ pointer to instance
    cdef si.uint16_t rows, cols
    cdef si.uint8_t* drpPtr

    def __cinit__(self, seg=0, pbits=0):
        #print "In peak_finder_algos.__cinit__"
        self.drpPtr = NULL
        #self.drpPtr = <si.uint8_t*>PyMem_Malloc(10240*100000)
        self.cptr = new PeakFinderAlgos(seg, pbits, self.drpPtr)

    def __dealloc__(self):
        #print "In peak_finder_algos.__dealloc__"
        del self.cptr
        PyMem_Free(self.drpPtr) # no-op if self.drpPtr is NULL

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
        #self.rows = data.shape[0]
        #self.cols = data.shape[1]
        #return self.list_of_peaks_selected()


    def peak_finder_v4r3_d2(self\
                           ,nptype2d data\
                           ,cnp.ndarray[mask_t, ndim=2, mode="c"] mask\
                           ,const double& thr_low
                           ,const double& thr_high
                           ,const size_t& rank\
                           ,const double& r0\
                           ,const double& dr) :
        self.cptr.peakFinderV4r3(&data[0,0], &mask[0,0], data.shape[0], data.shape[1], thr_low, thr_high, rank, r0, dr)
        self.rows = data.shape[0]
        self.cols = data.shape[1]
        return self.list_of_peaks_selected()


    def list_of_peaks_selected(self) :
        cdef vector[Peak] peaks = self.cptr.vectorOfPeaksSelected()
        return [py_peak.factory(p) for p in peaks]


    def list_of_peaks(self) :
        cdef vector[Peak] peaks = self.cptr.vectorOfPeaks()	
        return [py_peak.factory(p) for p in peaks]


    def peak(self, int i=0) : 
        return py_peak.factory(self.cptr.peak(i))


    def peak_selected(self, int i=0) : 
        return py_peak.factory(self.cptr.peakSelected(i))

    def peaks_selected(self):
        temp = np.asarray(self.cptr.peaksSelected()) # This makes a copy, 5e-5 sec
        return temp
        #cdef vector[vector[float]] peaks = self.cptr.peaksSelected() # This makes a copy, 5e-5 sec
        #return np.asarray(peaks)

    def convPeaksSelected(self):
        # ps_row and ps_col should become properties of array
        # TODO: test when no peaks are found
        cdef float[:, ::1] mv = <float[:self.cptr.ps_row, :self.cptr.ps_col]>self.cptr.convPeaksSelected() # No copy, 3e-5 sec
        cdef float[:] rows = mv[:, 0]
        cdef float[:] cols = mv[:, 1]
        cdef float[:] intens = mv[:, 2]
        return np.asarray(rows), np.asarray(cols), np.asarray(intens)

    def getPeaksSelected(self):
        cdef float[::1] rows = <float[:self.cptr.numPeaksSelected]>self.cptr.rows
        cdef float[::1] cols = <float[:self.cptr.numPeaksSelected]>self.cptr.cols
        cdef float[::1] intens = <float[:self.cptr.numPeaksSelected]>self.cptr.intens
        return np.asarray(rows), np.asarray(cols), np.asarray(intens)

    def getPeaks(self):
        cdef cnp.ndarray rows, cols, intens
        # Call the C function
        arr0 = ArrayWrapper()
        arr1 = ArrayWrapper()
        arr2 = ArrayWrapper()
        arr0.set_data(<void*> self.cptr.rows, self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        arr1.set_data(<void*> self.cptr.cols, self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        arr2.set_data(<void*> self.cptr.intens, self.cptr.numPeaksSelected, cnp.NPY_FLOAT)
        rows = np.array(arr0, copy=False)
        cols = np.array(arr1, copy=False)
        intens = np.array(arr2, copy=False)
        # Assign our object to the 'base' of the ndarray object
        rows.base = <PyObject*> arr0
        cols.base = <PyObject*> arr1
        intens.base = <PyObject*> arr2
        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(arr0)
        Py_INCREF(arr1)
        Py_INCREF(arr2)
        return rows, cols, intens

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







