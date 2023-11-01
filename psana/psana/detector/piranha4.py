from psana.detector.detector_impl import DetectorImpl
from amitypes import Array1d

class piranha4_raw_2_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super().__init__(*args)

    def raw(self,evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        return segs[0].image[:,0]

class piranha4_raw_2_1_0(DetectorImpl):
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

class piranha4_ttavg_1_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(piranha4_ttavg_1_0_0, self).__init__(*args)
        self._add_fields()


class piranha4_simfex_1_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(piranha4_simfex_1_0_0, self).__init__(self, *args, **kwa)
        self._add_fields()
