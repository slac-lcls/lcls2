# Polymorphic factory methods.
class Detector:
    def __init__(self, config):
        self.config = config

    def __call__(self, name):
        det = getattr(self.config.software, name)
        df = eval(str(det.dettype).upper() + '._Factory()')  # HSD._Factory()
        return df.create(name, self.config)

class DetectorBase(object):
    def __init__(self, name, config):
        self.detectorName = name
        self.config = config

from psana.detector.DetHSD import HSD
from psana.detector.DetEPIX import EPIX
from psana.detector.DetCSPAD import CSPAD
