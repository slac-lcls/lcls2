# Import the Python-level symbols of numpy
import numpy as np
np.set_printoptions(threshold=np.inf)
from psana.detector.detector_impl import DetectorImpl

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
        Channel(Allocator *allocator, Hsd_v1_2_3 *vHsd, const env_t *evtheader, const si.uint8_t *data)
        unsigned npeaks()
        unsigned numPixels
        AllocArray1D[cnp.uint16_t] waveform
        unsigned numFexPeaks
        AllocArray1D[cnp.uint16_t] sPos, len, fexPos
        AllocArray1D[arrp] fexPtr

class hsd_hsd_1_2_3(cyhsd_base_1_2_3, DetectorImpl):

    def __init__(self, *args):
        DetectorImpl.__init__(self, *args)
        cyhsd_base_1_2_3.__init__(self)

cdef class cyhsd_base_1_2_3:
    cdef HsdEventHeaderV1* hptr
    cdef Hsd_v1_2_3* cptr
    cdef Heap heap
    cdef Heap *ptr
    cdef Dgram *dptr
    cdef Channel *chptr[16] # Maximum channels: 16
    cdef dict _wvDict
    cdef list _chanList
    cdef unsigned _chanCounter
    cdef list _fexPeaks
    cdef dict _peaksDict

    def __cinit__(self):
        self.ptr = &self.heap
        self.cptr = new Hsd_v1_2_3(self.ptr)

    def __init__(self): # dgramlist: evt_dgram.xpphsd.hsd which has chan00, chan01, chan02, chan03
        self._wvDict = {}
        self._chanList = []
        self._chanCounter = 0
        self._evt = None
        self._det_dgrams = None
        self._peaksDict = {}
        self._fexPeaks = []

    def _setEnv(self, cnp.ndarray[env_t, ndim=1, mode="c"] env):
        self.cptr.init(&env[0])

    def _setChan(self, chanName, cnp.ndarray[env_t, ndim=1, mode="c"] evtheader, cnp.ndarray[chan_t, ndim=1, mode="c"] chan):
        self.chptr[self._chanCounter] = new Channel(self.ptr, self.cptr, &evtheader[0], &chan[0])
        self._chanList.append(chanName)
        self._chanCounter += 1

    def _isNewEvt(self, evt):
        if self._evt == None or not (evt._nanoseconds == self._evt._nanoseconds and evt._seconds == self._evt._seconds):
            return True
        else:
            return False

    def _parseEvt(self, evt):
        self._wvDict = {}
        self._chanList = []
        self._chanCounter = 0
        self._peaksDict = {}
        self._fexPeaks = []
        self._det_dgrams = self.dgrams(evt)
        self._setEnv(self._det_dgrams[0].env)
        for chanNum in xrange(16): # Maximum channels: 16
            chanName = 'chan'+'{num:02d}'.format(num=chanNum) # e.g. chan16
            if hasattr(self._det_dgrams[0], chanName):
                chan = eval('self._det_dgrams[0].'+chanName)
                if chan.size > 0:
                    chanName = str(chanNum) # e.g. 16
                    self._setChan(chanName, self._det_dgrams[0].env, chan)
        self._evt = evt

    def __dealloc__(self):
        del self.cptr
        for x in xrange(len(self._chanList)):
            del self.chptr[x]

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

    def waveforms(self, evt):
        """Return a dictionary of available waveforms in the event.
        0:    raw waveform intensity from channel 0
        1:    raw waveform intensity from channel 1
        ...
        16:   raw waveform intensity from channel 16
        times:  time axis (s)
        """
        # TODO: compare self.evt and evt
        cdef cnp.ndarray wv # TODO: make readonly
        if self._isNewEvt(evt):
            self._parseEvt(evt)
            for i, chanName in enumerate(self._chanList):
                if self.chptr[i].numPixels:
                    arr0 = PyAllocArray1D()
                    wv = arr0.init(&self.chptr[i].waveform, self.chptr[i].numPixels, cnp.NPY_UINT16)
                    wv.base = <PyObject*> arr0
                    self._wvDict[chanName] = wv
            self._wvDict["times"] = np.arange(self.chptr[i].numPixels) # FIXME: placeholder for times
        return self._wvDict

    def _channelPeaks(self, chanName):
        cdef list listOfPeaks, listOfPos # TODO: check whether this helps with speed
        listOfPeaks = []
        listOfPos = []
        cdef cnp.ndarray peak
        cdef cnp.npy_intp shape[1]
        try:
            chanNum = self._chanList.index(chanName)
            for i in range(self.chptr[chanNum].numFexPeaks): # TODO: disable bounds, wraparound check
                shape[0] = <cnp.npy_intp> self.chptr[chanNum].len(i) # len
                peak = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16,
                                   <cnp.uint16_t*>self.chptr[chanNum].fexPtr(i))
                peak.base = <PyObject*> self._fexPeaks
                Py_INCREF(self._fexPeaks)
                listOfPeaks.append(peak)
                listOfPos.append(self.chptr[chanNum].sPos(i))
        except ValueError:
            print("No such channel exists")
            listOfPeaks = None
            listOfPos = None
        return (listOfPos, listOfPeaks)

    def peaks(self, evt):
        """Return a dictionary of available peaks found in the event.
        0:    tuple of beginning of peaks and array of peak intensities from channel 0
        1:    tuple of beginning of peaks and array of peak intensities from channel 1
        ...
        16:   tuple of beginning of peaks and array of peak intensities from channel 16
        """
        # TODO: compare self.evt and evt
        if self._isNewEvt(evt):
            self._parseEvt(evt)
            for i, chanName in enumerate(self._chanList):
                self._peaksDict[chanName] = self._channelPeaks(chanName)
        return self._peaksDict

    def assem(self, evt):
        """Return a dictionary of available peaks assembled as waveforms.
        0:  peak intensities assembled into a waveform from channel 0
        1:  peak intensities assembled into a waveform from channel 1
        ...
        16: peak intensities assembled into a waveform from channel 16
        times:  time axis (s)
        """
        # TODO: compare self.evt and evt
        # TODO: assemble peaks into waveforms
        return self._assemDict
