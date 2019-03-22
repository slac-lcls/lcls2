import numpy as np
from psana.detector.detector_impl import DetectorImpl

class hsd_raw_0_0_0(DetectorImpl):
    def __init__(self, *args):
        super(hsd_raw_0_0_0, self).__init__(*args)
    def calib(self, evt):
        return np.zeros((5))

class hsd_fex_4_5_6(DetectorImpl):
    def __init__(self, *args):
        super(hsd_fex_4_5_6, self).__init__(*args)
    def calib(self, evt):
        return np.zeros((6))

class cspad_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(cspad_raw_2_3_42, self).__init__(*args)
    def raw(self, evt):
        # an example of how to handle multiple segments
        segs = self.segments(evt)
        return np.stack([segs[i].arrayRaw for i in range(len(segs))])
    def calib(self, evt):
        return self.raw(evt)
    def image(self, evt):
        segs = self.segments(evt)
        return np.vstack([segs[i].arrayRaw for i in range(len(segs))])

class ebeam_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_ebeamalg_2_3_42, self).__init__(*args)
    def energy(self, evt):
        return self.segments(evt)[0].energy

class cspad_raw_2_3_43(cspad_raw_2_3_42):
    def __init__(self, *args):
        super(cspad_raw_2_3_43, self).__init__(*args)
    def raw(self, evt):
        raise NotImplementedError()

class ebeam_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_raw_2_3_42, self).__init__(*args)
    def energy(self, evt):
        return self.segments(evt)[0].energy

class ebeam_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_raw_2_3_42, self).__init__(*args)
    def energy(self, evt):
        return self.segments(evt)[0].energy

class laser_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(laser_raw_2_3_42, self).__init__(*args)
    def laserOn(self, evt):
        return self.segments(evt)[0].laserOn

class hsd_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(hsd_raw_2_3_42, self).__init__(*args)
    def waveform(self, evt):
        # example of how to check for missing detector in event
        if self.segments(evt) is None:
            return None
        else:
            return self.segments(evt)[0].waveform

class epics_fuzzy_0_0_0(DetectorImpl):
    def __init__(self, *args):
        det_name, drp_class_name, configs, calibs, self._epics_store = args
        super(epics_fuzzy_0_0_0, self).__init__(det_name, drp_class_name, configs, calibs)
    def __call__(self, evt):
        epics_dicts = self._epics_store.checkout_by_events([evt])
        if not epics_dicts:
            return None

        return epics_dicts[0][self._det_name]

# for early cctbx/psana2 development
class cspad_raw_1_2_3(DetectorImpl):
    def __init__(self, *args):
        super(cspad_raw_1_2_3, self).__init__(*args)
    def raw(self, evt):
        quad0 = self.segments(evt)[0].quads0_data
        quad1 = self.segments(evt)[0].quads1_data
        quad2 = self.segments(evt)[0].quads2_data
        quad3 = self.segments(evt)[0].quads3_data
        return np.concatenate((quad0, quad1, quad2, quad3), axis=0)

    def photonEnergy(self, evt):
        return self.segments(evt)[0].photonEnergy

    def calib(self, evt):
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
