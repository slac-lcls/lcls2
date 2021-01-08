"""Data access VERSIONS
   for composite detectors made of epix10ka panels.
"""
import numpy as np
from amitypes import Array2d
from psana.detector.epix10ka_base import epix10ka_base, logging
logger = logging.getLogger(__name__)

class epix10k_raw_0_0_1(epix10ka_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epix10k_raw_0_0_1.__init__')
        epix10ka_base.__init__(self, *args, **kwargs)


class epix_raw_2_0_1(epix10ka_base):
    def __init__(self, *args, **kwargs):
        epix10ka_base.__init__(self, *args, **kwargs)
    def array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nsegs = len(segs)
            if nsegs==4:
                nx = segs[0].raw.shape[1]
                ny = segs[0].raw.shape[0]
                f = np.zeros((ny*2,nx*2), dtype=segs[0].raw.dtype)
                xa = [nx, 0, nx, 0]
                ya = [ny, ny, 0, 0]
                for i in range(4):
                    x = xa[i]
                    y = ya[i]
                    f[y:y+ny,x:x+nx] = segs[i].raw & 0x3fff
        return f
    def __call__(self, evt) -> Array2d:
        """Alias for self.raw(evt)"""
        return self.array(evt)


#class epixquad_raw_2_0_0(epix10ka_base):
#    def __init__(self, *args, **kwargs):
#        epix10ka_base.__init__(self, *args, **kwargs)
#        self._add_fields()
#    def _info(self,evt):
#        # check for missing data
#        segments = self._segments(evt)
#        if segments is None: return None
#        return segments[0]

# EOF
