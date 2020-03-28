import numpy as np
from psana.detector.detector_impl import DetectorImpl

class wave8_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(wave8_raw_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

class wave8_fex_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(wave8_fex_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

