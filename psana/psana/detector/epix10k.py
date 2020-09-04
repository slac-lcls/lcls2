from psana.detector.areadetector import AreaDetector

class epix10k_raw_0_0_1(AreaDetector):
    def __init__(self, *args):
        super().__init__(*args)
    def raw(self,evt):
        data = {}
        segments = self._segments(evt)
        if segments is None: return None
        for segment,val in segments.items():
            data[segment]=val.raw
        return data
