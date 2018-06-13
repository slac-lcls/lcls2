from psana.detector.detector import DetectorBase
import psana.detector.opaque as opaque

class HSD(DetectorBase):
    def __init__(self, name, config, dgramInd):
        super(HSD, self).__init__(name, config, dgramInd)
        config = config[self.dgramInd[0]]
        det = getattr(config.software, name)
        try:
            detcfg = getattr(det, 'hsd')
        except:
            return None

        chans = [x for x in dir(detcfg) if x.startswith('chan')]
        self.numChans = len(chans)
        setattr(config, 'nchan', self.numChans)
        self.rawData = opaque.OpaqueRawData(name, config)

    class _Factory:
        def create(self, name, config, dgramInd): return HSD(name, config, dgramInd)

    def __call__(self, evt):
        return self.rawData(evt)
