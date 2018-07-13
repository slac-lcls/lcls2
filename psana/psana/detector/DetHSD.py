from psana.detector.detector import DetectorBase
import psana.detector.opaque as opaque

class HSD(DetectorBase):
    def __init__(self, name, config, dgramInd, calib):
        super(HSD, self).__init__(name, config, dgramInd, calib)
        config = config[self.dgramInd]
        det = getattr(config.software, name)
        try:
            detcfg = getattr(det, 'hsd')
        except:
            return None

        chans = [x for x in dir(detcfg) if x.startswith('chan')]
        self.numChans = len(chans)
        setattr(config, 'nchan', self.numChans)
        self._rawData = opaque.OpaqueRawData(name, config)
        self._evtW = None
        self._evtP = None
        self._waveforms = None
        self._peaks = None
        self._startPos = None

    class _Factory:
        def create(self, name, config, dgramInd, calib): 
            return HSD(name, config, dgramInd, calib)

    def __call__(self, evt):
        return self._rawData(evt)

    def waveforms(self, evt):
        if (self._evtW is not evt):
            self._evtW = evt
            self._waveforms = self._rawData(evt).waveform()
        return self._waveforms

    def peaks(self, evt, chanNum):
        if (self._evtP is not evt):
            self._evtP = evt
            self._peaks, self._startPos = self._rawData(evt).peaks(chanNum)
        return self._peaks, self._startPos
