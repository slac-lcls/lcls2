# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

import sys # ref count

include "../peakFinder/dgramCreate.pyx"
include "../peakFinder/peakFinder.pyx"

################# High Speed Digitizer #################

ctypedef AllocArray1D[cnp.uint16_t*] arrp

cdef extern from "../../../psalg/psalg/include/hsd.hh" namespace "Pds::HSD":

    cdef cppclass HsdEventHeaderV1:
        void printVersion()
        unsigned samples()

    cdef cppclass Hsd_v1_2_3(HsdEventHeaderV1):
        int parseChan(const cnp.uint8_t* data, const unsigned chanNum) except +
        AllocArray1D[unsigned] numPixels
        AllocArray1D[cnp.uint16_t*] rawPtr
        AllocArray1D[AllocArray1D[cnp.uint16_t]] sPosx
        AllocArray1D[AllocArray1D[cnp.uint16_t]] lenx
        AllocArray1D[AllocArray1D[cnp.uint16_t]] fexPos
        AllocArray1D[arrp] fexPtr
        AllocArray1D[unsigned] numFexPeaksx

    cdef cppclass Client:
         Client(Allocator *allocator, const char* version, const unsigned nChan) except +
         HsdEventHeaderV1* getHsd() except +

def hsd(version, list data=None):
    myhsd = 'hsd_v'+version.replace('.','_')
    if data is None:
        return eval(myhsd+'(version)')
    else:
        return eval(myhsd+'(version, data)')

# TODO: check time / valgrind
cdef class hsd_v1_2_3:
    """ Python wrapper for C++ class.
    """
    cdef Client* cptr  # holds a C++ pointer to instance
    cdef HsdEventHeaderV1* hptr
    cdef Heap heap
    cdef Heap *ptr
    cdef Hsd_v1_2_3* ptr123
    cdef list chans

    def __cinit__(self, version, list chans):
        self.ptr = &self.heap
        self.cptr = new Client(self.ptr, version.encode("UTF-8"), len(chans))
        self.hptr = self.cptr.getHsd()

    def __dealloc__(self):
        del self.cptr

    def __init__(self, version, list chans):
        self.chans = chans
        self.ptr123 = <Hsd_v1_2_3*>self.hptr # cast once, and reuse _ptr
        cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] _arr
        for i in range(len(self.chans)): # TODO: disable bounds, wraparound check
            _arr = self.chans[i]
            self.ptr123.parseChan(&_arr[0], i)

    def printVersion(self):
        self.hptr.printVersion()

    def waveform(self):
        cdef list waveform
        waveform = []
        cdef cnp.ndarray w
        cdef cnp.npy_intp shape[1]
        for i in range(len(self.chans)): # TODO: disable bounds, wraparound check
            shape[0] = <cnp.npy_intp> self.ptr123.numPixels(i)
            w = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16,
                            <cnp.uint16_t*>self.ptr123.rawPtr(i))
            w.base = <PyObject*> self.chans[i]
            Py_INCREF(self.chans[i])
            waveform.append(w)
        return waveform

    def peaks(self, chanNum):
        cdef list listOfPeaks, sPos # TODO: check whether this helps with speed
        listOfPeaks = []
        sPos = []
        cdef cnp.ndarray peak
        cdef cnp.npy_intp shape[1]
        for i in range(self.ptr123.numFexPeaksx(chanNum)): # TODO: disable bounds, wraparound check
            shape[0] = <cnp.npy_intp> self.ptr123.lenx(chanNum)(i) # len
            peak = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16,
                               <cnp.uint16_t*>self.ptr123.fexPtr(chanNum)(i))
            peak.base = <PyObject*> self.chans[i]
            Py_INCREF(self.chans[i])
            listOfPeaks.append(peak)
            sPos.append(self.ptr123.sPosx(chanNum)(i))
        return listOfPeaks, sPos
