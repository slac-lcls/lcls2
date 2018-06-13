# Polymorphic factory methods.
class Detector:
    def __init__(self, config):
        self.config = config

    def searchDgramName(self, name):
        """Search datagrams for name and return a list of dgram indices."""
        dgramInd = []
        for i, dgram in enumerate(self.config):
           if hasattr(dgram.software, name): dgramInd.append(i)
        return dgramInd

    def __call__(self, name):
        dgramInd = self.searchDgramName(name)
        det = getattr(self.config[dgramInd[0]].software, name)
        df = eval(str(det.dettype).upper() + '._Factory()')  # HSD._Factory()
        return df.create(name, self.config, dgramInd)

class DetectorBase(object):
    def __init__(self, name, config, dgramInd):
        self.detectorName = name
        self.config = config
        self.dgramInd = dgramInd

from psana.detector.DetHSD import HSD
from psana.detector.DetEPIX import EPIX
from psana.detector.DetCSPAD import CSPAD
