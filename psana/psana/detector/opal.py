import numpy as np
from psana.detector.detector_impl import DetectorImpl

class opal_raw_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super(opal_raw_2_0_0, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

