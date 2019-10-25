import numpy as np
from psana.detector.detector_impl import DetectorImpl
import TimeToolDev.eventBuilderParser as timeToolParser


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
    
    def parsed_frame(self,evt):
        pFrame = timeToolParser.timeToolParser()
        p = self._segments(evt)[0].data.tobytes()
        pFrame.parseData(p)

        return pFrame

    #def edge_value(self,evt):              #speak with damiani for how to send back this data to ami
    #def edge_uncertainty(self,evt):        #needs _(underscore) to hide things AMI doesn't need to see.
    #
