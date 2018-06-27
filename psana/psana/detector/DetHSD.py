from psana.detector.detector import DetectorBase
import psana.detector.opaque as opaque

class HSD(DetectorBase):
    def __init__(self, name, config, dgramInd):
        super(HSD, self).__init__(name, config, dgramInd)
        config = config[self.dgramInd[0]] # FIXME: eliminate ind, there aren't two
        det = getattr(config.software, name)
        try:
            detcfg = getattr(det, 'hsd')
        except:
            return None

        chans = [x for x in dir(detcfg) if x.startswith('chan')]
        self.numChans = len(chans)
        setattr(config, 'nchan', self.numChans)
        self._rawData = opaque.OpaqueRawData(name, config)
        self._evt = None
        self._waveforms = None
        self._peaks = None
        self._startPos = None

    class _Factory:
        def create(self, name, config, dgramInd): return HSD(name, config, dgramInd)

    def __call__(self, evt):
        return self._rawData(evt)

    def waveforms(self, evt):
        if (self._evt is not evt):
            self._evt = evt
            self._waveforms = self._rawData(evt).waveform()
        return self._waveforms

    def peaks(self, evt, chanNum):
        if (self._evt is not evt):
            self._evt = evt
            self._peaks, self._startPos = self._rawData(evt).peaks(chanNum)
        return self._peaks, self._startPos