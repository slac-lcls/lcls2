#from psana.detector.detector_impl import DetectorImpl
from psana.detector.areadetector import DetectorImpl, AreaDetectorRaw
from amitypes import Array1d

class piranha4_raw_2_0_0(AreaDetectorRaw):
    def __init__(self, *args, **kwa):
        super().__init__(*args)

    def raw(self,evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        return segs[0].image[:,0]

class piranha4_raw_2_1_0(AreaDetectorRaw):
    def __init__(self, *args, **kwa):
        super().__init__(*args)

    def raw(self,evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        return segs[0].image

class piranha4_ttfex_1_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(piranha4_ttfex_1_0_0, self).__init__(*args)
        self._add_fields()

class piranha4_ttfex_1_0_1(piranha4_ttfex_1_0_0):
    """Algorithm version 1.0.1 - Address potential race condition.

    Note from GD - 2025/10/22:
    We believe there may have been a race condition which could invalidate results
    stored in the FEX. We think it was possible for parallel threads to modify
    the stored FEX results in `m_flt_position` etc, before the write or caput by
    a competing thread could be done. There were no semaphores or other synchronization
    mechanisms guarding writes and reads to/from these shared member attributes.

    To address this possibility we changed the Piranha4TTFex::analyze function to return
    the results to the caller instead of store them on member attributes.

    This increment in algorithm indicates that this new approach is being used. There is
    no difference in the structure/format of the data from algorithm 1.0.0.
    """
    def __init__(self, *args, **kwa):
        super().__init__(*args, **kwa)

class piranha4_ttavg_1_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(piranha4_ttavg_1_0_0, self).__init__(*args)
        self._add_fields()

class piranha4_simfex_1_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(piranha4_simfex_1_0_0, self).__init__(*args, **kwa)
        self._add_fields()
