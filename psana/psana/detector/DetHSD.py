from psana.detector.detector import DetectorBase
import psana.detector.opaque as opaque

class HSD(DetectorBase):
    def __init__(self, name, config):
        super(HSD, self).__init__(name, config)
        det = getattr(config.software, name)
        try:
            detcfg = getattr(det, 'hsd')
        except:
            return None
        chans = [x for x in dir(detcfg) if x.startswith('chan')]
        self.numChans = len(chans)
        assert detcfg.chan0.software == 'fpga'
        assert detcfg.chan0.version == (1, 2, 3)
        setattr(config, 'nchan', self.numChans)
        self.rawData = opaque.OpaqueRawData(name, config)

    class _Factory:
        def create(self, name, config): return HSD(name, config)

    def __call__(self, evt):
        return self.rawData(evt)

    def waveform(self, evt):
        myrawhsd = self.rawData(evt)
        return myrawhsd.raw()

    def peaks(self, evt, chan):
        myrawhsd = self.rawData(evt)
        listOfPeaks, sPos = myrawhsd.fex(chan)
        return listOfPeaks, sPos