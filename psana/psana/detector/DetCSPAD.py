from psana.detector.detector import DetectorBase
import numpy as np
import os

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
    
    def calib(self, data, verbose=0): print("cspad.calib")

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

