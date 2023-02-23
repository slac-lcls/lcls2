import numpy as np
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array1d, Array2d

class wave8_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(wave8_raw_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

    def raw_all(self,evt) -> Array2d:
        # only return a 2D array if all channels present
        # have the same length
        waveforms = []
        for i in range(8):
            func = getattr(self,'raw_%d'%i, None)
            if func:
                wf = func(evt)
                if wf is None: continue # missing waveform
                if len(waveforms)>0:
                    if wf.shape != waveforms[-1].shape:
                        return None # waveforms not the same shape
                waveforms.append(wf)
        if len(waveforms)==0: return None # no waveforms present
        return np.stack(waveforms)

class wave8_fex_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(wave8_fex_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

    def base_all(self,evt) -> Array1d:
        vals = []
        for i in range(8):
            func = getattr(self,'base_%d'%i, None)
            if func:
                val = func(evt)
                if val is not None: vals.append(val)
        if len(vals)==0: return None # no data
        return np.array(vals)

    def integral_all(self,evt) -> Array1d:
        vals = []
        for i in range(8):
            func = getattr(self,'integral_%d'%i, None)
            if func:
                val = func(evt)
                if val is not None: vals.append(val)
        if len(vals)==0: return None # no data
        return np.array(vals)
