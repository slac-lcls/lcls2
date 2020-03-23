import numpy as np
from psana.detector.detector_impl import DetectorImpl
# for AMI
import typing
import inspect

class hpsex_hpsex_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(hpsex_hpsex_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None

        return segments[0]

class hpscp_hpscp_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(hpscp_hpscp_0_0_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        return segments[0]

class ebeam_ebeamAlg_0_7_1(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_ebeamAlg_0_7_1, self).__init__(*args)

        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        return segments[0]

class ebeam_raw_2_0_0(ebeam_ebeamAlg_0_7_1):
    def __init__(self, *args):
        super().__init__(*args)
