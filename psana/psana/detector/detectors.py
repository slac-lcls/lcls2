from hsd import *  # cython hsd
import numpy as np

class hsd_raw_0_0_0(Detector):
    @property
    def calib(self):
        return np.zeros((5))
        #print('hsd raw:',[data.array0Pgp for data in self._dgramlist])

class hsd_fex_4_5_6(Detector):
    @property
    def calib(self):
        return np.zeros((5))
        #pass

class cspad_raw_2_3_42(Detector):
    @property
    def raw(self):
        return self._dgramlist[0].arrayRaw
    @property
    def mysum(self):
        return self._dgramlist[0].arrayRaw.sum()

class cspad_raw_2_3_43(cspad_raw_2_3_42):
    @property
    def raw(self):
        raise NotImplementedError()
