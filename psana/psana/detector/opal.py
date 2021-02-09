import numpy as np
#from psana.detector.detector_impl import DetectorImpl
#from amitypes import Array2d
from psana.detector.opal_base import opal_base, logging
logger = logging.getLogger(__name__)

#class opal_raw_2_0_0(DetectorImpl):
#    def __init__(self, *args, **kwa_base):
#        super(opal_raw_2_0_0, self).__init__(*args, **kwa)

class opal_raw_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

class opal_ttfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        #super(opal_ttfex_2_0_0, self).__init__(*args, **kwa)
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

class opal_simfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        #super(opal_simfex_2_0_0, self).__init__(*args, **kwa)
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

class opal_simfex_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        #super(opal_simfex_2_0_0, self).__init__(*args, **kwa)
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

class opal_ref_2_0_0(opal_base):
    def __init__(self, *args, **kwa):
        #super(opal_ref_2_0_0, self).__init__(*args, **kwa)
        opal_base.__init__(self, *args, **kwa)
        self._add_fields()

