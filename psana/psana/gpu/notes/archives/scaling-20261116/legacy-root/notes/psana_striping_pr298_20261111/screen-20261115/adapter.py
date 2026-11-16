"""Process-local sim10 detector registration; no expansion or calibration."""
from psana.detector.detector_impl import DetectorImpl
class jungfrau_sim10_1_0_0(DetectorImpl):
    def seed(self,evt)->dict:
        s=self._segments(evt)
        return None if s is None else {p:v.seed for p,v in s.items()}
def register_reader():
    from psana.detector import detectors
    name='jungfrau_sim10_1_0_0';old=getattr(detectors,name,None)
    if old is not None and old is not jungfrau_sim10_1_0_0:raise ValueError('Conflicting sim10 adapter')
    setattr(detectors,name,jungfrau_sim10_1_0_0)
