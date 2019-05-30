# Import the Python-level symbols of numpy
import numpy as np
from psana.detector.detector_impl import DetectorImpl

# Import to use cython decorators
cimport cython
# Import the C-level symbols of numpy
cimport numpy as cnp

import sys # ref count
from amitypes import HSDWaveforms, HSDPeaks

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
        AllocArray1D[cnp.uint16_t] sPos, len
        AllocArray1D[arrp] fexPtr

class hsd_hsd_1_2_3(cyhsd_base_1_2_3, DetectorImpl):

    def __init__(self, *args):
        DetectorImpl.__init__(self, *args)
        cyhsd_base_1_2_3.__init__(self)
    def _seg_chans(self):
        """
        returns a dictionary with segment numbers as the key, and a list
        of active channel numbers for each segment.
        """
        channels = {}
        for config in self._configs:
            seg_dict = getattr(config,self._det_name)
            for seg,seg_config in seg_dict.items():
                # currently the hsd only has 1 channel (zero)
                # will have to revisit in the future
                enable = getattr(getattr(getattr(seg_config,'hsdConfig'),'enable'),'value')
                if enable: channels[seg] = [0]
        return channels

cdef class cyhsd_base_1_2_3:
    cdef HsdEventHeaderV1* hptr
    cdef Hsd_v1_2_3* cptr
    cdef Heap heap
    cdef Heap *ptr
    cdef Dgram *dptr
    cdef Channel *chptr[16*16] # Maximum channels: 16, maximum segments: 16
    cdef dict _wvDict
    cdef list _chanList
    cdef list _fexPeaks
    cdef dict _peaksDict

    def __cinit__(self):
        self.ptr = &self.heap
        self.cptr = new Hsd_v1_2_3(self.ptr)

    def __init__(self): # dgramlist: evt_dgram.xpphsd.hsd which has chan00, chan01, chan02, chan03
        self._wvDict = {}
        self._chanList = []
        self._evt = None
        self._hsdsegments = None
        self._peaksDict = {}
        self._fexPeaks = []

    def _setEnv(self, cnp.ndarray[env_t, ndim=1, mode="c"] env):
        self.cptr.init(&env[0])

    def _setChan(self, iseg, chanNum, cnp.ndarray[env_t, ndim=1, mode="c"] evtheader, cnp.ndarray[chan_t, ndim=1, mode="c"] chan):
        self.chptr[iseg*16+chanNum] = new Channel(self.ptr, self.cptr, &evtheader[0], &chan[0])
        self._chanList.append((iseg,chanNum))

    def _isNewEvt(self, evt):
        if self._evt == None or not (evt._nanoseconds == self._evt._nanoseconds and evt._seconds == self._evt._seconds):
            return True
        else:
            return False

    def _parseEvt(self, evt):
        self._wvDict = {}
        self._chanList = []
        self._peaksDict = {}
        self._fexPeaks = []
        self._hsdsegments = self._segments(evt)
        self._setEnv(self._hsdsegments[0].env)
        self._evt = evt
        for iseg in self._hsdsegments:
            for chanNum in xrange(16): # Maximum channels: 16
                chanName = 'chan'+'{num:02d}'.format(num=chanNum) # e.g. chan16
                if hasattr(self._hsdsegments[iseg], chanName):
                    chan = eval('self._hsdsegments[iseg].'+chanName)
                    if chan.size > 0:
                        self._setChan(iseg,chanNum, self._hsdsegments[iseg].env, chan)

    def __dealloc__(self):
        del self.cptr
        for (iseg,chanNum) in self._chanList:
            del self.chptr[iseg*16+chanNum]

    def _samples(self):
        return self.cptr.samples()

    def _streams(self):
        return self.cptr.streams()

    def _channelMask(self):
        return self.cptr.channels()

    def _sync(self):
        return self.cptr.sync()

    def _raw(self):
        return self.cptr.raw()

    def _fex(self):
        return self.cptr.fex()

    # adding this decorator allows access to the signature information of the function in python
    # this is used for AMI type safety
    @cython.binding(True)
    def waveforms(self, evt) -> HSDWaveforms:
        """Return a dictionary of available waveforms in the event.
        0:    raw waveform intensity from channel 0
        1:    raw waveform intensity from channel 1
        ...
        16:   raw waveform intensity from channel 16
        times:  time axis (s)
        """
        cdef cnp.ndarray wv # TODO: make readonly
        if self._isNewEvt(evt):
            self._parseEvt(evt)
        for (iseg, chanNum) in self._chanList:
            first_chan_in_seg = True
            if self.chptr[iseg*16+chanNum].numPixels:
                if first_chan_in_seg:
                    self._wvDict[iseg] = {}
                    self._wvDict[iseg]["times"] = np.arange(self.chptr[iseg*16+chanNum].numPixels)
                arr0 = PyAllocArray1D()
                wv = arr0.init(&self.chptr[iseg*16+chanNum].waveform, self.chptr[iseg*16+chanNum].numPixels, cnp.NPY_UINT16)
                wv.base = <PyObject*> arr0
                self._wvDict[iseg][chanNum] = wv
        return self._wvDict

    def _channelPeaks(self, iseg, chanNum):
        cdef list listOfPeaks, listOfPos # TODO: check whether this helps with speed
        listOfPeaks = []
        listOfPos = []
        cdef cnp.ndarray peak
        cdef cnp.npy_intp shape[1]
        try:
            for i in range(self.chptr[iseg*16+chanNum].numFexPeaks): # TODO: disable bounds, wraparound check
                shape[0] = <cnp.npy_intp> self.chptr[iseg*16+chanNum].len(i) # len
                peak = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16,
                                   <cnp.uint16_t*>self.chptr[iseg*16+chanNum].fexPtr(i))
                peak.base = <PyObject*> self._fexPeaks
                Py_INCREF(self._fexPeaks)
                listOfPeaks.append(peak)
                listOfPos.append(self.chptr[iseg*16+chanNum].sPos(i))
        except ValueError:
            print("No such channel exists")
            listOfPeaks = None
            listOfPos = None
        return (listOfPos, listOfPeaks)

    # adding this decorator allows access to the signature information of the function in python
    # this is used for AMI type safety
    @cython.binding(True)
    def peaks(self, evt) -> HSDPeaks:
        """Return a dictionary of available peaks found in the event.
        0:    tuple of beginning of peaks and array of peak intensities from channel 0
        1:    tuple of beginning of peaks and array of peak intensities from channel 1
        ...
        16:   tuple of beginning of peaks and array of peak intensities from channel 16
        """
        if self._isNewEvt(evt):
            self._parseEvt(evt)
        for (iseg, chanNum) in self._chanList:
            if iseg not in self._peaksDict:
                self._peaksDict[iseg]={}
            self._peaksDict[iseg][chanNum] = self._channelPeaks(iseg,chanNum)
        return self._peaksDict

    def assem(self, evt) -> HSDWaveforms:
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
