import numpy as np

class DetectorImpl(object):
    def __init__(self, dgramlist, configs, calibs):
        self._dgramlist = dgramlist
        self._configs   = configs
        self._calibs    = calibs
        return

class hsd_raw_0_0_0(DetectorImpl):
    def __init__(self, dgramlist, configs, calibs):
        super(hsd_raw_0_0_0, self).__init__(dgramlist, configs, calibs)
    @property
    def calib(self):
        return np.zeros((5))

class hsd_fex_4_5_6(DetectorImpl):
    def __init__(self, dgramlist, configs, calibs):
        super(hsd_fex_4_5_6, self).__init__(dgramlist, configs, calibs)
    @property
    def calib(self):
        return np.zeros((6))

class cspad_raw_2_3_42(DetectorImpl):
    def __init__(self, dgramlist, configs, calibs):
        super(cspad_raw_2_3_42, self).__init__(dgramlist, configs, calibs)
    @property
    def raw(self):
        return self._dgramlist[0].arrayRaw
    @property
    def mysum(self):
        return self._dgramlist[0].arrayRaw.sum()

class cspad_raw_2_3_43(cspad_raw_2_3_42):
    def __init__(self, dgramlist, configs, calibs):
        super(cspad_raw_2_3_43, self).__init__(dgramlist, configs, calibs)
    @property
    def raw(self):
        raise NotImplementedError()

# for early cctbx/psana2 development
class cspad_raw_1_2_3(DetectorImpl):
    def __init__(self, dgramlist, configs, calibs):
        super(cspad_raw_1_2_3, self).__init__(dgramlist, configs, calibs)
    @property
    def raw(self):
        quad0 = self._dgramlist[0].quads0_data
        quad1 = self._dgramlist[0].quads1_data
        quad2 = self._dgramlist[0].quads2_data
        quad3 = self._dgramlist[0].quads3_data
        return np.concatenate((quad0, quad1, quad2, quad3), axis=0)

    @property
    def photonEnergy(self):
        return self._dgramlist[0].photonEnergy

    def calib(self, evt, verbose=0): 
        fake_run = -1 # FIXME: in current psana2 RunHelper, calib constants are setup according to that run.
        data = self.raw(evt)
        data = data.astype(np.float64) # convert raw photon counts to float for other math operations.
        data -= self.pedestals(fake_run)
        self.common_mode_apply(fake_run, data, None)
        gain_mask = self.gain_mask(fake_run)
        if gain_mask is not None:
            data *= gain_mask
        data *= self.gain(fake_run)
        return data

    def image(self, data, verbose=0): print("cspad.image")

    def _fetch(self, key):
        val = None
        if self._calib:
            if key in self._calib:
                val = self._calib[key]
        return val

    def pedestals(self, run):
        return self._fetch('pedestals')

    def gain_mask(self, run, gain=0):
        return self._fetch('gain_mask')

    def common_mode_apply(self, run, data, common_mode):
        # FIXME: apply common_mode
        return data

    def gain(self, run):
        # default gain is set to 1.0 (FIXME)
        return 1.0

    def geometry(self, run):
        geometry_string = self._fetch('geometry_string')
        geometry_access = None
        if geometry_string is not None:
            geometry_access = GeometryAccess()
            geometry_access.load_pars_from_str(geometry_string)
        return geometry_access
