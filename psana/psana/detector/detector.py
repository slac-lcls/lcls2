# Polymorphic factory methods.
from __future__ import generators
import psana.detector.opaque as opaque

class Detector():
    def __init__(self, config):
        self.config = config

    def __call__(self, name):
        det = getattr(self.config.software, name)
        df = eval(str(det.dettype).upper() + '._Factory()') # HSD._Factory()
        return df.create(name, self.config)

class DetectorBase(object):
    def __init__(self, name, config):
        self.detectorName = name
        self.config = config

class HSD(DetectorBase):
    """
    High Speed Digitizer (HSD) reader
    """
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
        assert detcfg.chan0.version  == (1,2,3)

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

class EPIX(DetectorBase):
    """
    epix reader
    """
    def __init__(self, name, config):
        super(EPIX, self).__init__(name, config)
        det = getattr(config.software, name)
        try:
            detcfg = getattr(det, 'epix')
        except:
            return None

    class _Factory:
        def create(self, name, config): return EPIX(name, config)

    def raw(self, evt, verbose=0):    print("epix.raw")
    def calib(self, data, verbose=0): print("epix.calib")
    def image(self, data, verbose=0): print("epix.image")

class CSPAD(DetectorBase):
    """
    cspad reader
    """
    def __init__(self, name, config):
        super(CSPAD, self).__init__(name, config)
        self.name = name
        detcfg = getattr(config.software, self.name)
        assert detcfg.dettype      == 'cspad'
        assert detcfg.raw.software == 'raw'
        assert detcfg.raw.version  == (2,3,42)

    class _Factory:
        def create(self, name, config): return CSPAD(name, config)

    def raw(self, evt, verbose=0):
        det = getattr(evt,self.name)
        evtAttr = det.raw.arrayRaw
        evtAttr = evtAttr.reshape((2,3,3)) # This mini cspad has 2x3x3 pixels
        return evtAttr

    def calib(self, data, verbose=0): print("cspad.calib")
    def image(self, data, verbose=0): print("cspad.image")
