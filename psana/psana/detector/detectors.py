from hsd import *  # cython hsd

class hsd_raw_0_0_0(Detector):
    def calib(self):
        print('hsd raw:',[data.array0Pgp for data in self._dgramlist])

class hsd_fex_4_5_6(Detector):
    def calib(self):
        pass

class cspad_raw_2_3_42(Detector):
    def calib(self):
        pass
class cspad_raw_2_3_43(cspad_raw_2_3_42):
    def calib(self):
        pass
