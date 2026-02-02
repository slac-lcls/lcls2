import numpy as np
from psana.detector.detector_impl import hiddenmethod, DetectorImpl
from amitypes import Array1d, Array2d, Array3d

# For testing quadanode (this uses Cython for speed)
from quadanode import waveforms, times

class hsd_raw_0_0_0(DetectorImpl):
    def __init__(self, *args):
        super(hsd_raw_0_0_0, self).__init__(*args)
    def calib(self, evt) -> Array1d:
        return np.zeros((5))

class pnccd_raw_1_2_3(DetectorImpl):
    """Test detector produced by dgrampy"""
    def __init__(self, *args):
        super(pnccd_raw_1_2_3, self).__init__(*args)
    def calib(self, evt) -> Array3d:
        segs = self._segments(evt)
        return segs[0].calib
    def photon_energy(self, evt):
        segs = self._segments(evt)
        return segs[0].photon_energy

class hexanode_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def waveforms(self, evt):
        segments = self._segments(evt)
        return segments[0].waveforms[2:7,...]
    def times(self, evt):
        segments = self._segments(evt)
        return segments[0].times[2:7,...]

# for tmo-fex simulated data test
class quadanode_fex_4_5_6(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)

    def waveforms(self, evt, n_dets=1):
        """ Returns list of fex waveforms for `segid` channel.

        List of segments/ channels:
        MCP, X1, X2, Y1, Y2

        Converts 1D waveform into a list of waveforms using lengths array
        as delimiters.
        """
        return waveforms(evt, self, n_dets=n_dets, n_chans_per_det=5)

        #wf1d = self._segments(evt)[segid].waveforms
        #lengths = self._segments(evt)[segid].lengths
        #n_wf = len(lengths)
        #waveforms = []
        #for i in range(n_wf):
        #    st = np.sum(lengths[:i])
        #    en = st + lengths[i]
        #    waveforms.append(wf1d[st:en])
        #return waveforms

    def times(self, evt, n_dets=1):
        return times(evt, self, n_dets=n_dets, n_chans_per_det=5)

        #startpos = self._segments(evt)[segid].startpos
        #return startpos

# for dgrampy test
class hsd_fex_4_5_6(DetectorImpl):
    def __init__(self, *args):
        super(hsd_fex_4_5_6, self).__init__(*args)
    def calib(self, evt) -> Array1d:
        return np.zeros((6))
    @hiddenmethod
    def show(self, evt):
        segs = self._segments(evt)
        print(f'valFex: {segs[0].valFex}')
        print(f'strFex: {segs[0].strFex}')
        print(f'arrayFex0: {segs[0].arrayFex0}')
        print(f'arrayFex1: {segs[0].arrayFex1}')
        print(f'arrayFex2: {segs[0].arrayFex2}')
        print(f'arrayFex3: {segs[0].arrayFex3}')
        print(f'arrayFex4: {segs[0].arrayFex4}')
        print(f'arrayFex5: {segs[0].arrayFex5}')
        print(f'arrayFex6: {segs[0].arrayFex6}')
        print(f'arrayFex7: {segs[0].arrayFex7}')
        print(f'arrayFex8: {segs[0].arrayFex8}')
        print(f'arrayFex9: {segs[0].arrayFex9}')
        #print(f'arrayString: {segs[0].arrayString}')

# for integrating detector test (1) hsd/fast (2) andor/slow
class hsd_raw_0_0_2(DetectorImpl):
    def __init__(self, *args):
        super(hsd_raw_0_0_2, self).__init__(*args)
    def calib(self, evt):
        segs = self._segments(evt)
        return segs[0].calib
class andor_raw_0_0_2(DetectorImpl):
    def __init__(self, *args):
        super(andor_raw_0_0_2, self).__init__(*args)
    def calib(self, evt):
        segs = self._segments(evt)
        if segs is None:
            return None
        if 1 in segs:
            return segs[1].calib
        else:
            return segs[0].calib
class epix_raw_0_0_2(DetectorImpl):
    def __init__(self, *args):
        super(epix_raw_0_0_2, self).__init__(*args)
    def calib(self, evt):
        segs = self._segments(evt)
        if segs is None:
            return None
        return segs[2].calib

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
class fakecam_cube_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def bin(self, evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].bin
    def entries(self, evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].entries
    def value(self, evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].value
    def array(self, evt) -> Array3d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].array

# for the pva detector in the teststand
class pva_pvaAlg_1_2_3(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def raw(self, evt) -> Array1d:
        segs = self._segments(evt)
        if segs is None: return None
        if segs[0] is None: return None
        return segs[0].value
class andor_raw_0_0_1(pva_pvaAlg_1_2_3):
    def __init__(self, *args):
        super().__init__(*args)

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

class ele_opal_raw_1_2_3(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def image(self, evt) -> Array2d:
        return self._segments(evt)[0].img

class eventid_valseid_0_0_1(DetectorImpl):
    def __init__(self, *args):
        DetectorImpl.__init__(self, *args)
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
        DetectorImpl.__init__(self, *args)
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


class xtcavpars_valsxtp_0_0_1(DetectorImpl):
    def __init__(self, *args):
        DetectorImpl.__init__(self, *args)
    def XTCAV_Analysis_Version      (self, evt): return self._segments(evt)[0].XTCAV_Analysis_Version       if self._segments(evt) is not None else None
    def XTCAV_ROI_sizeX             (self, evt): return self._segments(evt)[0].XTCAV_ROI_sizeX              if self._segments(evt) is not None else None
    def XTCAV_ROI_sizeY             (self, evt): return self._segments(evt)[0].XTCAV_ROI_sizeY              if self._segments(evt) is not None else None
    def XTCAV_ROI_startX            (self, evt): return self._segments(evt)[0].XTCAV_ROI_startX             if self._segments(evt) is not None else None
    def XTCAV_ROI_startY            (self, evt): return self._segments(evt)[0].XTCAV_ROI_startY             if self._segments(evt) is not None else None
    def XTCAV_calib_umPerPx         (self, evt): return self._segments(evt)[0].XTCAV_calib_umPerPx          if self._segments(evt) is not None else None
    def OTRS_DMP1_695_RESOLUTION    (self, evt): return self._segments(evt)[0].OTRS_DMP1_695_RESOLUTION     if self._segments(evt) is not None else None
    def XTCAV_strength_par_S        (self, evt): return self._segments(evt)[0].XTCAV_strength_par_S         if self._segments(evt) is not None else None
    def OTRS_DMP1_695_TCAL_X        (self, evt): return self._segments(evt)[0].OTRS_DMP1_695_TCAL_X         if self._segments(evt) is not None else None
    def XTCAV_Amp_Des_calib_MV      (self, evt): return self._segments(evt)[0].XTCAV_Amp_Des_calib_MV       if self._segments(evt) is not None else None
    def SIOC_SYS0_ML01_AO214        (self, evt): return self._segments(evt)[0].SIOC_SYS0_ML01_AO214         if self._segments(evt) is not None else None
    def XTCAV_Phas_Des_calib_deg    (self, evt): return self._segments(evt)[0].XTCAV_Phas_Des_calib_deg     if self._segments(evt) is not None else None
    def SIOC_SYS0_ML01_AO215        (self, evt): return self._segments(evt)[0].SIOC_SYS0_ML01_AO215         if self._segments(evt) is not None else None
    def XTCAV_Beam_energy_dump_GeV  (self, evt): return self._segments(evt)[0].XTCAV_Beam_energy_dump_GeV   if self._segments(evt) is not None else None
    def REFS_DMP1_400_EDES          (self, evt): return self._segments(evt)[0].REFS_DMP1_400_EDES           if self._segments(evt) is not None else None
    def XTCAV_calib_disp_posToEnergy(self, evt): return self._segments(evt)[0].XTCAV_calib_disp_posToEnergy if self._segments(evt) is not None else None
    def SIOC_SYS0_ML01_AO216        (self, evt): return self._segments(evt)[0].SIOC_SYS0_ML01_AO216         if self._segments(evt) is not None else None


class ebeam_valsebm_0_0_1(DetectorImpl):
    def __init__(self, *args):
        DetectorImpl.__init__(self, *args)
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
        DetectorImpl.__init__(self, *args)
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

class sfx_raw_1_2_3(DetectorImpl):
    """Test detector produced by dgrampy"""
    def __init__(self, *args):
        super(sfx_raw_1_2_3, self).__init__(*args)
    def calib(self, evt) -> Array3d:
        segs = self._segments(evt)
        return segs[0].calib
    def photon_energy(self, evt):
        segs = self._segments(evt)
        return segs[0].photon_energy

    ## peak finder
    def npeaks(self,evt):
        segs = self._segments(evt)
        return segs[0].npeaks
    def seg(self,evt):
        segs = self._segments(evt)
        return segs[0].seg
    def row(self,evt):
        segs = self._segments(evt)
        return segs[0].row
    def col(self,evt):
        segs = self._segments(evt)
        return segs[0].col
    def npix(self,evt):
        segs = self._segments(evt)
        return segs[0].npix
    def amax(self,evt):
        segs = self._segments(evt)
        return segs[0].amax
    def atot(self,evt):
        segs = self._segments(evt)
        return segs[0].atot

class epixhremu_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    #def raw(self, evt) -> Array2d:
    #    segs = self._segments(evt)
    #    if segs is None: return None
    #    if segs[0] is None: return None
    #    return segs[0].raw
    def raw(self, evt) -> Array3d:
        # an example of how to handle multiple segments
        segs = self._segments(evt)
        return np.stack([segs[i].raw for i in sorted(segs.keys())])
    def calib(self, evt) -> Array3d:
        return self.raw(evt)
    def image(self, evt) -> Array2d:
        segs = self._segments(evt)
        return np.vstack([segs[i].raw for i in sorted(segs.keys())])
