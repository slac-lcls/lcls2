
import numpy as np
import psana.detector.areadetector as ad

class epix100hw_raw_2_0_1(ad.AreaDetector):

    def __init__(self, *args, **kwa):
        ad.AreaDetector.__init__(self, *args, **kwa)
        self._seg_geo = ad.sgs.Create(segname='EPIX100:V1')

#    def image(self,evt):
#        segments = self._segments(evt)
#        return segments[0].raw

class epix100_raw_2_0_1(ad.AreaDetector):

    def __init__(self, *args, **kwa):
        ad.AreaDetector.__init__(self, *args, **kwa)
        self._seg_geo = ad.sgs.Create(segname='EPIX100:V1')
