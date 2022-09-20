import numpy as np
from psana.detector.piranha4_base import piranha4_base, logging
from psana.detector.detector_impl import DetectorImpl

logger = logging.getLogger(__name__)

class piranha4_raw_2_0_0(piranha4_base):
    def __init__(self, *args, **kwa):
        piranha4_base.__init__(self, *args, **kwa)
        #self._add_fields() < overrides det.raw.image(...)


class piranha4_ttfex_2_0_0(piranha4_base):
    def __init__(self, *args, **kwa):
        piranha4_base.__init__(self, *args, **kwa)
        self._add_fields()

#  removed some fields
class piranha4_ttfex_2_1_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(piranha4_ttfex_2_1_0, self).__init__(*args)
        self._add_fields()

class piranha4_ttproj_2_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(piranha4_ttproj_2_0_0, self).__init__(*args)
        self._add_fields()


class piranha4_simfex_2_0_0(piranha4_base):
    def __init__(self, *args, **kwa):
        piranha4_base.__init__(self, *args, **kwa)
        self._add_fields()


class piranha4_simfex_2_1_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        piranha4_base.__init__(self, *args, **kwa)
        self._add_fields()


class piranha4_ref_2_0_0(piranha4_base):
    def __init__(self, *args, **kwa):
        piranha4_base.__init__(self, *args, **kwa)
        self._add_fields()

# EOF
