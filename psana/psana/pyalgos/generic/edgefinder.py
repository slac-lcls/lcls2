import numpy as np

class EdgeFinder(object):

    def __init__(self, myrun):
        self._run = myrun
        self.kernel = self.get_fir_coefficients()
        self.tt_det = myrun.Detector('tmott')

    def __call__(self, evt):
        firmware_edge = None
        software_edge = None
        my_kernel = self.kernel
        tt_detector_object = self.tt_det

        parsed_frame_object = tt_detector_object.ttalg.parsed_frame(evt)
        image = tt_detector_object.ttalg._image(evt)
        
        #known good length values that a raw frame can take
        assert image.shape == (2208,) or image.shape == (144,) or image.shape == (4272,)
        
        #make sure the edge position falls within the pixel range
        assert parsed_frame_object.edge_position >=0 and parsed_frame_object.edge_position <= 2047
        
        #validating that we're getting the correct number of pixels
        if(parsed_frame_object.prescaled_frame is not None):
            assert len(parsed_frame_object.prescaled_frame) == 2048

            firmware_edge = parsed_frame_object.edge_position
            software_edge = np.argmax(np.convolve(parsed_frame_object.prescaled_frame,my_kernel))

        if(parsed_frame_object.background_frame is not None):
            #validating that we're getting the correct number of pixels from the firmware
            assert len(parsed_frame_object.background_frame) == 2048
        
        return firmware_edge, software_edge


    def twos_complement(self, hexstr,bits):
        value = int(hexstr,16)
        if value & (1 << (bits-1)):
            value -= 1 << bits
        return value

    def get_fir_coefficients(self):
        myrun = self._run
        my_kernel = []
        my_dict = myrun.configs[0].tmotimetool[0].timetoolConfig.cl.Application.AppLane0.Fex.FIR.__dict__
        for key in my_dict:
            mystring = my_dict[key]
            my_kernel.extend([self.twos_complement(mystring[i:i+2],8) for i in range(0,len(mystring),2)])
        return my_kernel


