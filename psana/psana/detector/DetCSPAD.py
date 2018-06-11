from psana.detector.detector import DetectorBase
import psana.detector.opaque as opaque

class CSPAD(DetectorBase):

    def __init__(self, name, config):
        super(CSPAD, self).__init__(name, config)
        self.name = name
        detcfg = getattr(config.software, self.name)
        assert detcfg.dettype == 'cspad'
        assert detcfg.raw.software == 'raw'
        assert detcfg.raw.version == (2, 3, 42)

    class _Factory:
        def create(self, name, config): return CSPAD(name, config)

    def raw(self, evt, verbose=0):
        det = getattr(evt, self.name)
        evtAttr = det.raw.arrayRaw
        evtAttr = evtAttr.reshape((2, 3, 3))  # This mini cspad has 2x3x3 pixels
        return evtAttr

    def calib(self, data, verbose=0): print("cspad.calib")

    def image(self, data, verbose=0): print("cspad.image")
