import numpy as np
from psana.detector.detector_impl import DetectorImpl

class tt_detector_type_placeholder_tt_algorithm_placeholder_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(tt_detector_type_placeholder_tt_algorithm_placeholder_0_0_1, self).__init__(*args)

    def image(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all timetool data comes from one segment
        return segments[0].data

    def header(self,evt):                           #take out.  scientists don't care about this. 
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all timetool data comes from one segment
        return segments[0].data[:16]

    #def edge_value(self,evt):              #speak with damiani for how to send back this data to ami
    #def edge_uncertainty(self,evt):
