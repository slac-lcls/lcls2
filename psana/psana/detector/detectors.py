
class Detector(object):

    def __init__(self):
        # the list of the data objects
        self._datalist = []
        return

    def _append_dgram(self, drp_class_dgram):
        self._datalist.append(drp_class_dgram)


class hsd_raw_0_0_0(Detector):
    def calib(self):
        print('hsd raw:',[data.array0Pgp for data in self._datalist])
        # FIXME need pass in calibs & configs in the same way ^

class hsd_fex_4_5_6(Detector):
    def calib(self):
        pass
        #print('hsd fex',[data.arrayFex for data in self._datalist])

class hsd_hsd_1_2_3(Detector):
    def calib(self):
        pass
        #print('*** in hsd_hsd_1_2_3 calib method')


class cspad_raw_2_3_42(Detector):
    def calib(self):
        pass
class cspad_raw_2_3_43(cspad_raw_2_3_42):
    def calib(self):
            pass

