import numpy as np
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array1d, Array2d, Array3d

class hsd_raw_0_0_0(DetectorImpl):
    def __init__(self, *args):
        super(hsd_raw_0_0_0, self).__init__(*args)
    def calib(self, evt) -> Array1d:
        return np.zeros((5))

class hexanode_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def waveforms(self, evt):
        segments = self._segments(evt)
        return segments[0].waveforms[2:7,...]
    def times(self, evt):
        segments = self._segments(evt)
        return segments[0].times[2:7,...]

class hsd_fex_4_5_6(DetectorImpl):
    def __init__(self, *args):
        super(hsd_fex_4_5_6, self).__init__(*args)
    def calib(self, evt) -> Array1d:
        return np.zeros((6))

class cspad_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(cspad_raw_2_3_42, self).__init__(*args)
    def raw(self, evt) -> Array3d:
        # an example of how to handle multiple segments
        segs = self._segments(evt)
        return np.stack([segs[i].arrayRaw for i in range(len(segs))])
    def calib(self, evt) -> Array3d:
        return self.raw(evt)
    def image(self, evt) -> Array2d:
        segs = self._segments(evt)
        return np.vstack([segs[i].arrayRaw for i in range(len(segs))])

class ebeam_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_ebeamalg_2_3_42, self).__init__(*args)
    def energy(self, evt):
        return self._segments(evt)[0].energy

class cspad_raw_2_3_43(cspad_raw_2_3_42):
    def __init__(self, *args):
        super(cspad_raw_2_3_43, self).__init__(*args)
    def raw(self, evt) -> None:
        raise NotImplementedError()

class ebeam_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_raw_2_3_42, self).__init__(*args)
    def energy(self, evt):
        return self._segments(evt)[0].energy

class ebeam_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_raw_2_3_42, self).__init__(*args)
    def energy(self, evt) -> float:
        return self._segments(evt)[0].energy

class laser_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(laser_raw_2_3_42, self).__init__(*args)
    def laserOn(self, evt) -> int:
        return self._segments(evt)[0].laserOn

class hsd_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(hsd_raw_2_3_42, self).__init__(*args)
    def waveform(self, evt) -> Array1d:
        # example of how to check for missing detector in event
        if self._segments(evt) is None:
            return None
        else:
            return self._segments(evt)[0].waveform

class EnvImpl(DetectorImpl):
    def __init__(self, *args):
        det_name, self._var_name, drp_class_name, configs, calibs, self._env_store = args
        super(EnvImpl, self).__init__(det_name, drp_class_name, configs, calibs)
    def __call__(self, events):
        if isinstance(events, list):
            return self._env_store.values(events, self._var_name)
        else:
            env_values = self._env_store.values([events], self._var_name)
            return env_values[0]

class epics_epics_0_0_0(EnvImpl):
    def __init__(self, *args):
        super(epics_epics_0_0_0, self).__init__(*args)

class scanDet_scan_2_3_42(EnvImpl):
    def __init__(self, *args):
        super(scanDet_scan_2_3_42, self).__init__(*args)

# for early cctbx/psana2 development
class cspad_raw_1_2_3(DetectorImpl):
    def __init__(self, *args):
        super(cspad_raw_1_2_3, self).__init__(*args)
    def raw(self, evt):
        #quad0 = self._segments(evt)[0].quads0_data
        #quad1 = self._segments(evt)[0].quads1_data
        #quad2 = self._segments(evt)[0].quads2_data
        #quad3 = self._segments(evt)[0].quads3_data
        #return np.concatenate((quad0, quad1, quad2, quad3), axis=0)
        return self._segments(evt)[0].raw

    def raw_data(self, evt):
        return self.raw(evt)

    def photonEnergy(self, evt):
        return self._segments(evt)[0].photonEnergy

    def calib(self, evt):
        data = self.raw(evt)
        data = data.astype(np.float64) # convert raw photon counts to float for other math operations.
        data -= self.pedestals()
        self.common_mode_apply(data, None)
        gain_mask = self.gain_mask()
        if gain_mask is not None:
            data *= gain_mask
        data *= self.gain()
        return data

    def image(self, data, verbose=0): print("cspad.image")

    def _fetch(self, key):
        val = None
        if key in self._calibconst:
            val, _ = self._calibconst[key]
        return val

    def pedestals(self):
        return self._fetch('pedestals')

    def gain_mask(self, gain=0):
        return self._fetch('gain_mask')

    def common_mode_apply(self, data, common_mode):
        # FIXME: apply common_mode
        return data

    def gain(self):
        # default gain is set to 1.0 (FIXME)
        return 1.0

    def geometry(self):
        geometry_string = self._fetch('geometry')
        geometry_access = None
        if geometry_string is not None:
            from psana.pscalib.geometry.GeometryAccess import GeometryAccess
            geometry_access = GeometryAccess()
            geometry_access.load_pars_from_str(geometry_string)
        return geometry_access
