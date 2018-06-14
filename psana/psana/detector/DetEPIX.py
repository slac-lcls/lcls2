from psana.detector.detector import DetectorBase
import psana.detector.opaque as opaque

class EPIX(DetectorBase):
    def __init__(self, name, config, dgramInd):
        super(EPIX, self).__init__(name, config, dgramInd)
        det = getattr(config.software, name) # FIXME: index config with dgramInd
        try:
            detcfg = getattr(det, 'epix')
        except:
            return None

    class _Factory:
        def create(self, name, config, dgramInd): return EPIX(name, config, dgramInd)

    def raw(self, evt, verbose=0):
        print("epix.raw")

    def calib(self, data, verbose=0):
        print("epix.calib")

    def image(self, data, verbose=0):
        print("epix.image")