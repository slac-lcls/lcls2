from psana.detector.areadetector import AreaDetector

class epix10k_raw_0_0_1(AreaDetector):
    def __init__(self, *args):
        super().__init__(*args)
    def raw(self,evt):
        data = {}
        for segment,val in self._segments(evt).items():
            data[segment]=val.raw
        return data
