

"""Class :py:class:`MEDUtils` - utilities for Mask Editor
=========================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDUtils.py

    from psana.graphqt.MEDUtils import *

Created on 2023-09-07 by Mikhail Dubrovin
"""
char_expand    = u' \u25BC' # down-head triangle
char_shrink    = u' \u25B2' # solid up-head triangle


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

# EOF
