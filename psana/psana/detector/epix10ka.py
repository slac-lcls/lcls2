"""Data access VERSIONS
   for composite detectors made of epix10ka panels.
"""
from psana.detector.epix10ka_base import epix10ka_base, logging
logger = logging.getLogger(__name__)

class epix10k_raw_0_0_1(epix10ka_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epix10k_raw_0_0_1.__init__')
        epix10ka_base.__init__(self, *args, **kwargs)

#----
