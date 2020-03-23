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

# for the fake cameras in the teststand
class cspad_cspadRawAlg_1_2_3(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def raw(self, evt) -> Array2d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].array_raw
class fakecam_raw_2_0_0(cspad_cspadRawAlg_1_2_3):
    def __init__(self, *args):
        super().__init__(*args)

# for the pva detector in the teststand
class pva_pvaAlg_1_2_3(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def raw(self, evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].value

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


class eventid_valseid_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(eventid_valseid_0_0_1, self).__init__(*args)
    def experiment(self, evt):
        return self._segments(evt)[0].experiment if self._segments(evt) is not None else None
    def run(self, evt):
        return self._segments(evt)[0].run        if self._segments(evt) is not None else None
    def fiducials(self, evt):
        return self._segments(evt)[0].fiducials  if self._segments(evt) is not None else None
    def time(self, evt): 
        return self._segments(evt)[0].time       if self._segments(evt) is not None else None


class gasdetector_valsgd_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(gasdetector_valsgd_0_0_1, self).__init__(*args)
    def f_11_ENRC(self, evt):
        return self._segments(evt)[0].f_11_ENRC if self._segments(evt) is not None else None
    def f_12_ENRC(self, evt):
        return self._segments(evt)[0].f_12_ENRC if self._segments(evt) is not None else None
    def f_21_ENRC(self, evt):
        return self._segments(evt)[0].f_21_ENRC if self._segments(evt) is not None else None
    def f_22_ENRC(self, evt):
        return self._segments(evt)[0].f_22_ENRC if self._segments(evt) is not None else None
    def f_63_ENRC(self, evt):
        return self._segments(evt)[0].f_63_ENRC if self._segments(evt) is not None else None
    def f_64_ENRC(self, evt):
        return self._segments(evt)[0].f_64_ENRC if self._segments(evt) is not None else None


class ebeam_valsebm_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_valsebm_0_0_1, self).__init__(*args)
    def Charge(self, evt):
        return self._segments(evt)[0].Charge     if self._segments(evt) is not None else None
    def DumpCharge(self, evt):
        return self._segments(evt)[0].DumpCharge if self._segments(evt) is not None else None
    def XTCAVAmpl(self, evt):
        return self._segments(evt)[0].XTCAVAmpl  if self._segments(evt) is not None else None
    def XTCAVPhase(self, evt):
        return self._segments(evt)[0].XTCAVPhase if self._segments(evt) is not None else None
    def PkCurrBC2(self, evt):
        return self._segments(evt)[0].PkCurrBC2  if self._segments(evt) is not None else None
    def L3Energy(self, evt):
        return self._segments(evt)[0].L3Energy   if self._segments(evt) is not None else None


class ebeam_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(ebeam_raw_2_3_42, self).__init__(*args)
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

class tt_tt_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def image(self, evt) -> Array1d:
        return self._segments(evt)[0].image

class hsd_raw_2_3_42(DetectorImpl):
    def __init__(self, *args):
        super(hsd_raw_2_3_42, self).__init__(*args)
    def waveform(self, evt) -> Array1d:
        # example of how to check for missing detector in event
        if self._segments(evt) is None:
            return None
        else:
            return self._segments(evt)[0].waveform

class camera_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(camera_raw_0_0_1, self).__init__(*args)
    def array(self, evt) -> Array2d:
        if self._segments(evt) is None:
            return None
        else:
            return self._segments(evt)[0].array
    def __call__(self, evt) -> Array2d:
        """Alias for self.raw(evt)"""
        return self.array(evt)

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
