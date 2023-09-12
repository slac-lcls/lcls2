

"""Class :py:class:`MEDUtils` - utilities for Mask Editor
=========================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDUtils.py

    from psana.graphqt.MEDUtils import *

Created on 2023-09-07 by Mikhail Dubrovin
"""
import os
import logging
logger = logging.getLogger(__name__)


import psana.pyalgos.generic.PSUtils as psu
from psana.pyalgos.generic.NDArrUtils import reshape_to_2d, info_ndarr, np

#char_expand    = u' \u25BC' # down-head triangle
#char_shrink    = u' \u25B2' # solid up-head triangle

def image_from_ndarray(nda):
    if nda is None:
       logger.warning('nda is None - return None for image')
       return None

    if not isinstance(nda, np.ndarray):
       logger.warning('nda is not np.ndarray, type(nda): %s - return None for image' % type(nda))
       return None

    img = psu.table_nxn_epix10ka_from_ndarr(nda) if (nda.size % (352*384) == 0) else\
          psu.table_nxm_jungfrau_from_ndarr(nda) if (nda.size % (512*1024) == 0) else\
          psu.table_nxm_cspad2x1_from_ndarr(nda) if (nda.size % (185*388) == 0) else\
          reshape_to_2d(nda)
    logger.debug(info_ndarr(img,'img'))
    return img

def random_image(shape=(10,10)):
    import psana.pyalgos.generic.NDArrGenerators as ag
    return ag.random_standard(shape, mu=0, sigma=10)

def image_from_kwargs(**kwa):
    ndafname = kwa.get('ndafname', None)

    if not os.path.lexists(ndafname):
        logger.warning('ndarray file %s not found - use random image' % ndafname)
        return ndarandom_image()

    nda = np.load(ndafname)
    if nda is None:
        logger.warning('can not load ndarray from file: %s - use random image' % ndafname)
        return ndarandom_image()

    geofname = kwa.get('geofname', None)
    if not os.path.lexists(geofname):
        logger.warning('geometry file %s not found - use ndarray without geometry' % geofname)
        return image_from_ndarray(nda)

    from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays
    geo = GeometryAccess(geofname)

    irows, icols = geo.get_pixel_coord_indexes(do_tilt=True, cframe=0)
    return img_from_pixel_arrays(irows, icols, W=nda)


def color_table(ict=2):
    import psana.graphqt.ColorTable as ct
    return ct.next_color_table(ict)  # OR ct.color_table_monochr256()

# EOF
