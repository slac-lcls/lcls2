"""Data access VERSIONS
   for composite detectors made of epix10kt segments/panels.
"""
import numpy as np
from amitypes import Array2d
#from psana.detector.detector_impl import DetectorImpl
from psana.detector.epix_base import epix_base, logging
logger = logging.getLogger(__name__)

class epixhr2x2_raw_2_0_1(epix_base):
    def __init__(self, *args, **kwargs):
        epix_base.__init__(self, *args, **kwargs)

    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nx = segs[0].raw.shape[1]
            ny = segs[0].raw.shape[0]
            f = segs[0].raw & 0x7fff
        return f


