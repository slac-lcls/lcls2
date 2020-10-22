"""Data access methods common for all AREA DETECTORS.
"""

from psana.detector.detector_impl import DetectorImpl

import logging
logger = logging.getLogger(__name__)

class AreaDetector(DetectorImpl):

    def __init__(self, *args, **kwargs):
        logger.debug('AreaDetector.__init__') #  self.__class__.__name__
        DetectorImpl.__init__(self, *args, **kwargs)

    # example of some possible common behavior
    def _common_mode(self, *args, **kwargs):
        logger.debug('in %s._common_mode' % self.__class__.__name__)
        pass

    def raw(self,evt):
        data = {}
        segments = self._segments(evt)
        if segments is None: return None
        for segment,val in segments.items():
            data[segment]=val.raw
        return data

#----

if __name__ == "__main__":
    import sys
    sys.exit('See example in test_%s if available...' % sys.argv[0].split('/')[-1])

#----
