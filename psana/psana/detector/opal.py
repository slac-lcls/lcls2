
import numpy as np
from psana.detector.opal_base import opal_base, logging
logger = logging.getLogger(__name__)

class opal_raw_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        #self._add_fields() < overrides det.raw.image(...)


class opal_ttfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()


class opal_simfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()


class opal_simfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()


class opal_ref_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

# EOF


