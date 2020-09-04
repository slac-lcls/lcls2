# placeholder for common behavior, e.g. algorithms like common mode?
# but an opal doesn't have common mode so perhaps need more subclassing.

from psana.detector.detector_impl import DetectorImpl
class AreaDetector(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    # example of some possible common behavior
    def _common_mode(self):
        pass
