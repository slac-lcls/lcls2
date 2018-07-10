from psana.detector.detector import DetectorBase
import numpy as np
import os

class CSPAD(DetectorBase):

    def __init__(self, name, config, dgramInd):
        super(CSPAD, self).__init__(name, config, dgramInd)
        self.name = name
        self.ind = dgramInd
        config = config[self.ind]
        detcfg = getattr(config.software, self.name)
        assert detcfg.dettype == 'cspad'

    class _Factory:
        def create(self, name, config, dgramInd): return CSPAD(name, config, dgramInd)

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

    def pedestals(self, run):
        try:
            dir = os.environ['DEMO_CXID9114']
        except:
            dir = '/reg/common/package/temp'
        return np.load(os.path.join(dir,'cxid9114_r0089_pedestals.npy'))