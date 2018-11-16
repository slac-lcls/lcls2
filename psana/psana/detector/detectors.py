from hsd import *  # cython hsd
import numpy as np

class hsd_raw_0_0_0(Detector):
    def calib(self):
        return np.empty((5))
        #print('hsd raw:',[data.array0Pgp for data in self._dgramlist])

class hsd_fex_4_5_6(Detector):
    def calib(self):
        return np.empty((5))
        #pass

class cspad_raw_2_3_42(Detector):
    @property
    def raw(self):
        return self._dgramlist[0].arrayRaw

class cspad_raw_2_3_43(cspad_raw_2_3_42):
    @property
    def raw(self):
        raise NotImplementedError()
