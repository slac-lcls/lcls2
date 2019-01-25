# Import the Python-level symbols of numpy
import numpy as np
from psana.detector.detectors import DetectorImpl

# Import the C-level symbols of numpy
cimport numpy as cnp

import sys # ref count

include "../peakFinder/peakFinder.pyx"  # defines Allocator, PyAlloArray1D

################# High Speed Digitizer #################

ctypedef AllocArray1D[cnp.uint16_t*] arrp

cimport libc.stdint as si
ctypedef si.uint32_t env_t
ctypedef si.uint8_t chan_t

cdef extern from "xtcdata/xtc/Dgram.hh" namespace "XtcData":
    cdef cppclass Dgram:
        pass

cdef extern from "psalg/digitizer/Hsd.hh" namespace "Pds::HSD":
    cdef cppclass HsdEventHeaderV1:
        void printVersion()
        unsigned samples()
        unsigned streams()
        unsigned channels()
        unsigned sync()
        bint raw()
        bint fex()

    cdef cppclass Hsd_v1_2_3(HsdEventHeaderV1):
        Hsd_v1_2_3(Allocator *allocator)
        void init(env_t *e)
        void printVersion()

    cdef cppclass Channel:
        Channel(Allocator *allocator, Hsd_v1_2_3 *vHsd, const si.uint8_t *data)
        unsigned npeaks()
        unsigned numPixels
        AllocArray1D[cnp.uint16_t] waveform
        unsigned numFexPeaks
        AllocArray1D[cnp.uint16_t] sPos, len, fexPos
        AllocArray1D[arrp] fexPtr


class hsd_hsd_1_2_3(cyhsd_hsd_1_2_3, DetectorImpl):

    def __init__(self, dgramlist, configs, calibs):
        DetectorImpl.__init__(self, dgramlist, configs, calibs)
        cyhsd_hsd_1_2_3.__init__(self)

class waveform:
    """
    This is a waveform object.

    Possible attributes (if the data exists in the event):
    ch0:    waveform intensity from channel 0
    ch1:    waveform intensity from channel 1
    ...
    ch16:   waveform intensity from channel 16
    times:  time axis (s)
    """

    def __init__(self):
        pass

class peaks:
    """
    This is a peaks object.

    Possible attributes (if the data exists in the event):
    ch0:    tuple of beginning of peak and array of peak intensities from channel 0
    ch1:    tuple of beginning of peak and array of peak intensities from channel 1
    ...
    ch16:   tuple of beginning of peak and array of peak intensities from channel 16
    assemble0:  peak intensities assembled into a waveform from channel 0
    assemble1:  peak intensities assembled into a waveform from channel 1
    ...
    assemble16: peak intensities assembled into a waveform from channel 16
    """

    def __init__(self):
        pass

cdef class cyhsd_hsd_1_2_3:
    cdef HsdEventHeaderV1* hptr
    cdef Hsd_v1_2_3* cptr
    cdef Heap heap
    cdef Heap *ptr
    cdef Dgram *dptr
    cdef Channel *chptr[16] # Maximum channels: 16
    cdef dict wvDict
    cdef list chanList
    cdef unsigned chanCounter
    cdef list fexPeaks
    cdef dict fexDict

    def __cinit__(self):
        self.ptr = &self.heap
        self.cptr = new Hsd_v1_2_3(self.ptr)

    def __init__(self): # dgramlist: evt_dgram.xpphsd.hsd which has chan00, chan01, chan02, chan03
        self.wvDict = {}
        self.chanList = []
        self.chanCounter = 0
        self.fexDict = {}

        self._setEnv(self._dgramlist[-1].env)
        for chanNum in xrange(16): # Maximum channels: 16
            chanName = 'chan'+'{num:02d}'.format(num=chanNum) # chan00
            if hasattr(self._dgramlist[-1], chanName):
                chan = eval('self._dgramlist[-1].'+chanName)
                if chan.size > 0:
                    chanName = 'ch'+str(chanNum) # ch0
                    self._setChan(chanName, chan)

        self._genWaveformAttr()
        self._genPeaksAttr()

    def __dealloc__(self):
        del self.cptr
        for x in xrange(len(self.chanList)):
            del self.chptr[x]

    def _setEnv(self, cnp.ndarray[env_t, ndim=1, mode="c"] env):
        self.cptr.init(&env[0])

    def _setChan(self, chanName, cnp.ndarray[chan_t, ndim=1, mode="c"] chan):
        self.chptr[self.chanCounter] = new Channel(self.ptr, self.cptr, &chan[0])
        self.chanList.append(chanName)
        self.chanCounter += 1

    def _samples(self):
        return self.cptr.samples()

    def _streams(self):
        return self.cptr.streams()

    def _channels(self):
        return self.cptr.channels()

    def _sync(self):
        return self.cptr.sync()

    def _raw(self):
        return self.cptr.raw()

    def _fex(self):
        return self.cptr.fex()

    def _genWaveformAttr(self):
        """Return a dictionary of available waveforms in the event."""
        cdef cnp.ndarray wv # TODO: make readonly

        for i, chanName in enumerate(self.chanList):
            if self.chptr[i].numPixels:
                arr0 = PyAllocArray1D()
                wv = arr0.init(&self.chptr[i].waveform, self.chptr[i].numPixels, cnp.NPY_UINT16)
                wv.base = <PyObject*> arr0
                # export waveform.chX
                if not hasattr(self, 'waveform'):
                    setattr(self, 'waveform', waveform()) #setattr(self, 'waveform', types.SimpleNamespace())
                    setattr(self.waveform, 'times', np.arange(self.chptr[i].numPixels)) # FIXME: placeholder for times
                setattr(self.waveform, chanName, wv)

    def _genPeaksAttr(self):
        for i, chanName in enumerate(self.chanList):
            self.fexDict[chanName] = self._channelPeaks(chanName)
            if len(self._channelPeaks(chanName)[0]):
                if not hasattr(self, 'peaks'):
                    setattr(self, 'peaks', peaks())
                setattr(self.peaks, chanName, self._channelPeaks(chanName))
                setattr(self.peaks, 'assemble'+chanName, np.arange(self.chptr[i].numPixels)) # FIXME: placeholder for assembled waveform

    def _channelPeaks(self, chanName):
        cdef list listOfPeaks, listOfPos # TODO: check whether this helps with speed
        listOfPeaks = []
        listOfPos = []
        cdef cnp.ndarray peak
        cdef cnp.npy_intp shape[1]
        try:
            chanNum = self.chanList.index(chanName)
            for i in range(self.chptr[chanNum].numFexPeaks): # TODO: disable bounds, wraparound check
                shape[0] = <cnp.npy_intp> self.chptr[chanNum].len(i) # len
                peak = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16,
                                   <cnp.uint16_t*>self.chptr[chanNum].fexPtr(i))
                peak.base = <PyObject*> self.fexPeaks
                Py_INCREF(self.fexPeaks)
                listOfPeaks.append(peak)
                listOfPos.append(self.chptr[chanNum].sPos(i))
        except ValueError:
            print("No such channel exists")
            listOfPeaks = None
            listOfPos = None
        return (listOfPos, listOfPeaks)

"""
    def _waveforms(self):
        cdef cnp.ndarray wv

        if not self.wvDict:
            for i, chanName in enumerate(self.chanList):
                arr0 = PyAllocArray1D()
                wv = arr0.init(&self.chptr[i].waveform, self.chptr[i].numPixels, cnp.NPY_UINT16)
                wv.base = <PyObject*> arr0
                self.wvDict[chanName] = wv

        return self.wvDict

    def _peaks(self):
        for chanName in self.chanList:
            self.fexDict[chanName] = self._channelPeaks(chanName)

        return self.fexDict
"""
