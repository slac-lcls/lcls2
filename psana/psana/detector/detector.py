# Polymorphic factory methods.
class Detector:
    def __init__(self, config, calib=None):
        self.config = config
        self._calib = calib

    def searchDgramName(self, name):
        """Search datagrams for name and return a list of dgram indices."""
        dgramInd = None
        for i, dgram in enumerate(self.config):
           if hasattr(dgram, "software"):
               if hasattr(dgram.software, name): dgramInd = i # TODO: throw error if two the same
        return dgramInd

    def __call__(self, name):
        dgramInd = self.searchDgramName(name)
        det = getattr(self.config[dgramInd].software, name)
        df = eval(str(det.dettype).upper() + '._Factory()')  # HSD._Factory()
        return df.create(name, self.config, dgramInd, self._calib)

class DetectorBase(object):
    def __init__(self, name, config, dgramInd, calib):
        self.detectorName = name
        self.config = config
        self.dgramInd = dgramInd
        self._calib = calib


from psana.detector.DetHSD import HSD
from psana.detector.DetEPIX import EPIX
from psana.detector.DetCSPAD import CSPAD

