#
# See psalg/digitizer/Hsd.hh for a summary of hsd software design ideas.
#
import time

import numpy as np
from psana.detector.detector_impl import DetectorImpl
from cpython cimport PyObject, Py_INCREF

errprint=True

# Import to use cython decorators
cimport cython
# Import the C-level symbols of numpy
cimport numpy as cnp

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

import sys # ref count
from amitypes import HSDWaveforms, HSDPeaks, HSDAssemblies, HSDPeakTimes

################# High Speed Digitizer #################

cimport libc.stdint as si
ctypedef si.uint32_t evthdr_t
ctypedef si.uint8_t chan_t

# cdef extern from "xtcdata/xtc/Dgram.hh" namespace "XtcData":
#     cdef cppclass Dgram:
#         pass

cdef extern from "HsdPython.hh" namespace "Pds::HSD":
    cdef cppclass ChannelPython:
        ChannelPython()
        ChannelPython(const evthdr_t *evtheader, const si.uint8_t *data)
        si.uint16_t* waveform(unsigned &numsamples)
        #si.uint16_t* sparse(unsigned &numsamples)
        unsigned next_peak(unsigned &sPos, si.uint16_t** peakPtr)

cdef class PyChannelPython:
    cdef public cnp.ndarray waveform
    #cdef public cnp.ndarray sparse
    cdef public list peakList
    cdef public list startPosList
    def __init__(self, cnp.ndarray[evthdr_t, ndim=1, mode="c"] evtheader, cnp.ndarray[chan_t, ndim=1, mode="c"] chan, dgram):
        cdef cnp.npy_intp shape[1]
        cdef si.uint16_t* wf_ptr
        cdef ChannelPython chanpy
        cdef unsigned numsamples = 0
        cdef unsigned startSample = 0
        cdef unsigned startPos = 0
        cdef unsigned peakLen = 0
        cdef si.uint16_t* peakPtr = <si.uint16_t*>0
        cdef cnp.ndarray peak

        chanpy = ChannelPython(&evtheader[0], &chan[0])

        wf_ptr = chanpy.waveform(numsamples)
        shape[0] = numsamples
        if numsamples:
            self.waveform = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16, wf_ptr)
            self.waveform.base = <PyObject*> dgram
            Py_INCREF(dgram)
        else:
            self.waveform = None

#        wf_ptr = chanpy.sparse(numsamples)
#        shape[0] = numsamples
#        if numsamples:
#            self.sparse = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16, wf_ptr)
#            self.sparse.base = <PyObject*> dgram
#            Py_INCREF(dgram)
#        else:
#            self.sparse = None

        self.peakList = None
        self.startPosList = None
        while True:
            shape[0] = chanpy.next_peak(startPos,&peakPtr)
            if not shape[0]: break
            if self.peakList is None:
                # we've found a peak so initialize the arrays
                self.peakList = []
                self.startPosList = []
            peak = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_UINT16, peakPtr)
            peak.base = <PyObject*> dgram
            Py_INCREF(dgram)
            self.peakList.append(peak)
            self.startPosList.append(startPos)

class hsd_hsd_1_2_3(cyhsd_base_1_2_3, DetectorImpl):

    def __init__(self, *args):
        DetectorImpl.__init__(self, *args)
        cyhsd_base_1_2_3.__init__(self)
        self._load_config()

    def _load_config(self):
        for config in self._configs:
            if not hasattr(config,self._det_name):
                continue
            seg_dict = getattr(config,self._det_name)
            for seg,seg_config in seg_dict.items():
                self._padValue[seg] = (seg_config.config.fex.ymin+
                                       seg_config.config.fex.ymax)//2
                self._padLength[seg] = seg_config.config.fex.gate*40
        
    # this routine is used by ami/data.py
    def _seg_chans(self):
        """
        returns a dictionary with segment numbers as the key, and a list
        of active channel numbers for each segment.
        """
        channels = {}
        for config in self._configs:
            if not hasattr(config,self._det_name):
                continue
            seg_dict = getattr(config,self._det_name)
            for seg,seg_config in seg_dict.items():
                # currently the hsd only has 1 channel (zero)
                # will have to revisit in the future
                channels[seg] = [0]

        return channels

cdef class cyhsd_base_1_2_3:
    cdef dict _wvDict
    cdef list _fexPeaks
    cdef dict _peaksDict
    cdef dict _padDict

    def __cinit__(self):
        pass

    def __init__(self):
        self._wvDict = {}
        self._spDict = {}
        self._fexPeaks = []
        self._peaksDict = {}
        self._peakTimesDict = {}
        self._evt = None
        self._hsdsegments = None

        self._padDict = {}
        self._padValue = {}
        self._padLength = {}

    def _isNewEvt(self, evt):
        if self._evt == None or not (evt._nanoseconds == self._evt._nanoseconds and evt._seconds == self._evt._seconds):
            return True
        else:
            return False

    def _padEvt(self):
        global errprint
        """ Slow padding routine currently only needed by AMI"""
        if not self._pychansegs: return
        
        cdef int iseg
        for iseg, (chanNum, pychan) in self._pychansegs.items():
            # Skip padding for empty peak segments
            if pychan.peakList is None: continue

            times = []
            for start, peak in zip(pychan.startPosList, pychan.peakList):
                times.append(np.arange(start, start+len(peak)) * 1/(6.4*1e9*13/14))

            if iseg not in self._peakTimesDict:
                self._peakTimesDict[iseg]={}
            self._peakTimesDict[iseg][chanNum] = times

            padvalues = np.zeros(self._padLength[iseg])+self._padValue[iseg]
            for ipeak, (start, peak) in enumerate(zip(pychan.startPosList, pychan.peakList)):
                if start+len(peak)>len(padvalues):
                    if errprint:
                        print('*** Skipping hsd FEX peak out of range. Start:',start,'Length:',len(peak),'Array size:',len(padvalues))
                        print('*** Suppressing duplicate error messages.')
                        errprint = False
                else:
                    padvalues[start:start+len(peak)]=peak

            if iseg not in self._padDict:
                self._padDict[iseg]={}
            self._padDict[iseg][chanNum] = padvalues
            self._padDict[iseg]["times"] = np.arange(self._padLength[iseg]) * 1/(6.4e9*13/14)

    def _parseEvt(self, evt):
        self._wvDict = {}
        self._spDict = {}
        self._peaksDict = {}
        self._padDict = {}
        self._fexPeaks = []
        self._hsdsegments = self._segments(evt)
        if self._hsdsegments is None: return # no segments at all
        self._evt = evt
        #seglist = [] # not used at the moment

        # Keep segment-pychan data for slow padding routine when asked
        self._pychansegs = {}

        cdef int iseg
        for iseg in self._hsdsegments:
            #seglist.append(iseg) # not used at the moment

            chans = [seg_key for seg_key in self._hsdsegments[iseg].__dict__.keys() \
                    if seg_key.startswith('chan')]
            if not chans: continue

            # Only permit one channel per segment
            chanName = chans[0]
            chanNum = int(chanName[4:])
            
            chan = getattr(self._hsdsegments[iseg], chanName)
            if chan.size > 0:
                pychan = PyChannelPython(self._hsdsegments[iseg].eventHeader, chan, self._hsdsegments[iseg])
                self._pychansegs[iseg] = (chanNum, pychan)
                if pychan.waveform is not None:
                    if iseg not in self._wvDict.keys():
                        self._wvDict[iseg] = {}
                        # FIXME: this needs to be put in units of seconds
                        # perhaps both for 5GHz and 6GHz models
                        self._wvDict[iseg]["times"] = np.arange(len(pychan.waveform)) * 1/(6.4*1e9*13/14)
                    self._wvDict[iseg][chanNum] = pychan.waveform
#                        if pychan.sparse is not None:
#                            if iseg not in self._spDict.keys():
#                                self._spDict[iseg] = {}
#                                self._spDict[iseg][chanNum] = pychan.sparse
                if pychan.peakList is not None:
                    if iseg not in self._peaksDict.keys():
                        self._peaksDict[iseg]={}
                    self._peaksDict[iseg][chanNum] = (pychan.startPosList,pychan.peakList)

        # maybe check that we have all segments in the event?
        # FIXME: also check that we have all the channels we expect?
        # unclear how to flag this.  maybe return None to the user
        # from the det xface?
        #seglist.sort()
        #if seglist != self._config_segments: 


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
        if not self._wvDict:
            return None
        else:
            return self._wvDict

    # adding this decorator allows access to the signature information of the function in python
    # this is used for AMI type safety
#    @cython.binding(True)
#    def sparse(self, evt) -> HSDWaveforms:
#        """Return a dictionary of available waveforms in the event.
#        0:    raw waveform intensity from channel 0
#        1:    raw waveform intensity from channel 1
#        ...
#        16:   raw waveform intensity from channel 16
#        times:  time axis (s)
#        """
#        cdef cnp.ndarray wv # TODO: make readonly
#        if self._isNewEvt(evt):
#            self._parseEvt(evt)
#        if not self._spDict:
#            return None
#        else:
#            return self._spDict

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
        if not self._peaksDict:
            return None
        else:
            return self._peaksDict

    @cython.binding(True)
    def peak_times(self, evt) -> HSDPeakTimes:
        """Return a dictionary of available times of peaks found in the event.
        0:    tuple of beginning of peaks and array of peak intensities from channel 0
        1:    tuple of beginning of peaks and array of peak intensities from channel 1
        ...
        16:   tuple of beginning of peaks and array of peak intensities from channel 16
        """
        if self._isNewEvt(evt):
            self._parseEvt(evt)
        if not self._peakTimesDict:
            return None
        else:
            return self._peakTimesDict

    # adding this decorator allows access to the signature information of the function in python
    # this is used for AMI type safety
    @cython.binding(True)
    def padded(self, evt) -> HSDWaveforms:
        """Return a dictionary of available padded waveforms in the event.
        0:    reconstructed waveform intensity from channel 0
        1:    reconstructed waveform intensity from channel 1
        ...
        16:   reconstructed waveform intensity from channel 16
        times:  time axis (s)
        """
        cdef cnp.ndarray wv # TODO: make readonly

        # Padding data is slow so it's not done when _parseEvt is called and
        # so to check if has been done already, we have to check the _padDict
        # variable and CANNOT rely on _isNewEvt.
        if self._isNewEvt(evt):
            self._parseEvt(evt)
            self._padEvt()
        else:
            if not self._padDict:
                self._padEvt()
        
        if not self._padDict:
            return None
        else:
            return self._padDict


class hsd_raw_2_0_0(hsd_hsd_1_2_3):

    def __init__(self, *args):
        hsd_hsd_1_2_3.__init__(self, *args)

    def _load_config(self):
        for config in self._configs:
            if not hasattr(config,self._det_name):
                continue
            seg_dict = getattr(config,self._det_name)
            for seg,seg_config in seg_dict.items():
                self._padValue[seg] = (seg_config.config.user.fex.ymin+
                                       seg_config.config.user.fex.ymax)//2
                self._padLength[seg] = int(seg_config.config.user.fex.gate_ns*0.160*13/14)*40
