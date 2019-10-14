import numpy as np
from psana.detector.detector_impl import DetectorImpl

class timetool_timetool_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(timetool_timetool_0_0_1, self).__init__(*args)

    def image(self,evt):
        # check for missing data
        if self._segments(evt) is None: return None
        # seems reasonable to assume that all timetool  data comes from one segment
        segment = getattr(evt._dgrams[0],self._det_name)[0]
        
        return segment.timetool.data

    def header(self,evt):                           #take out.  scientists don't care about this. 
        # check for missing data
        if self._segments(evt) is None: return None
        # seems reasonable to assume that all timetool data comes from one segment
        segment = getattr(evt._dgrams[0],self._det_name)[0]
        
        return segment.timetool.data[:16]

    #def edge_value(self,evt):              #speak with damiani for how to send back this data to ami
    #def edge_uncertainty(self,evt):
