
import numpy as np
from psana.detector.opal_base import opal_base, logging
from psana.detector.detector_impl import DetectorImpl

logger = logging.getLogger(__name__)

class opal_raw_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        #self._add_fields() < overrides det.raw.image(...)


class opal_ttfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

#  removed some fields
class opal_ttfex_2_1_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(opal_ttfex_2_1_0, self).__init__(*args)
        self._add_fields()

class opal_ttfex_2_1_1(opal_ttfex_2_1_0):
    """Algorithm version 2.1.1 - Address potential race condition.

    Note from GD - 2025/10/22:
    We believe there may have been a race condition which could invalidate results
    stored in the FEX. We think it was possible for parallel threads to modify
    the stored FEX results in `m_flt_position` etc, before the write or caput by
    a competing thread could be done. There were no semaphores or other synchronization
    mechanisms guarding writes and reads to/from these shared member attributes.

    To address this possibility we changed the OpalTTFex::analyze function to return
    the results to the caller instead of store them on member attributes.

    This increment in algorithm indicates that this new approach is being used. There is
    no difference in the structure/format of the data from algorithm 2.1.0.
    """
    def __init__(self, *args, **kwa):
        super().__init__(*args, **kwa)

class opal_ttproj_2_0_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        super(opal_ttproj_2_0_0, self).__init__(*args)
        self._add_fields()


class opal_simfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()


class opal_simfex_2_1_0(DetectorImpl):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()


class opal_ref_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

# EOF


