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

    cdef cppclass Hsd_v1_0_0(HsdEventHeaderV1):
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
cdef class hsd_v1_0_0:
    """ Python wrapper for C++ class.
    """
    cdef Client* cptr  # holds a C++ pointer to instance
    cdef HsdEventHeaderV1* hptr
    cdef Heap heap
    cdef Heap *ptr
    cdef Hsd_v1_0_0* ptr100
    cdef cnp.uint8_t* dptr
    cdef list chans

    def __cinit__(self, version, list data):
        self.ptr = &self.heap
        self.cptr = new Client(self.ptr, version.encode("UTF-8"), len(data))
        self.hptr = self.cptr.getHsd()

    def __dealloc__(self):
        del self.cptr

    def __init__(self, version, list data):
        self.ptr100 = <Hsd_v1_0_0*>self.hptr # cast once, and reuse _ptr
        cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] _arr
        self.chans = data
        for i in range(len(self.chans)): # TODO: disable bounds, wraparound check
            _arr = self.chans[i]
            self.ptr100.parseChan(&_arr[0], i)

    def printVersion(self):
        self.hptr.printVersion()

    def raw(self):
        cdef list waveform
        waveform = []
        cdef cnp.ndarray w
        cdef cnp.npy_intp shape[1]
        for i in range(len(self.chans)): # TODO: disable bounds, wraparound check
            shape[0] = <cnp.npy_intp> self.ptr100.numPixels(i)
            print("waveform length: {}".format(shape[0]))
            w = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16,
                            <cnp.uint16_t*>self.ptr100.rawPtr(i))
            w.base = <PyObject*> self.chans[i]
            Py_INCREF(self.chans[i])
            waveform.append(w)
        return waveform

    def fex(self, chanNum):
        cdef list listOfPeaks, sPos # TODO: check whether this helps with speed
        listOfPeaks = []
        sPos = []
        cdef cnp.ndarray peak
        cdef cnp.npy_intp shape[1]
        for i in range(self.ptr100.numFexPeaksx(chanNum)): # TODO: disable bounds, wraparound check
            shape[0] = <cnp.npy_intp> self.ptr100.lenx(chanNum)(i) # len
            peak = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16,
                               <cnp.uint16_t*>self.ptr100.fexPtr(chanNum)(i))
            peak.base = <PyObject*> self.chans[i]
            Py_INCREF(self.chans[i])
            listOfPeaks.append(peak)
            sPos.append(self.ptr100.sPosx(chanNum)(i))
        return listOfPeaks, sPos
