from psana.detector.detector import DetectorBase
import numpy as np
import os
from psana.pscalib.geometry.GeometryAccess import GeometryAccess

class CSPAD(DetectorBase):

    def __init__(self, name, config, dgramInd, calib):
        super(CSPAD, self).__init__(name, config, dgramInd, calib)
        self.name = name
        self.ind = dgramInd
        config = config[self.ind]
        detcfg = getattr(config.software, self.name)
        assert detcfg.dettype == 'cspad'

    class _Factory:
        def create(self, name, config, dgramInd, calib): 
            return CSPAD(name, config, dgramInd, calib)

    def photonEnergy(self, evt):
        return eval('evt.dgrams[self.ind].' + self.name + '.raw.photonEnergy')

    def raw(self, evt, verbose=0):
        quad0 = eval('evt.dgrams[self.ind].'+self.name).raw.quads0_data
        quad1 = eval('evt.dgrams[self.ind].'+self.name).raw.quads1_data
        quad2 = eval('evt.dgrams[self.ind].'+self.name).raw.quads2_data
        quad3 = eval('evt.dgrams[self.ind].'+self.name).raw.quads3_data
        return np.concatenate((quad0, quad1, quad2, quad3), axis=0)

    def raw_data(self, evt):
        # compatible with current cctbx code
        return self.raw(evt)
    
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

