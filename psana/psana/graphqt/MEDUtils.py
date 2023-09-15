

"""Class :py:class:`MEDUtils` - utilities for Mask Editor
=========================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDUtils.py

    from psana.graphqt.MEDUtils import *
    #     image_from_ndarray, random_image, image_from_kwargs, mask_ndarray_from_2d,
    #     color_table, list_of_instruments, list_of_experiments
    v = image_from_ndarray(nda)
    v = random_image(shape=(64,64))
    v = image_from_kwargs(**kwa)
    v = mask_ndarray_from_2d(mask2d, geo)
    v = color_table(ict=2)
    v = list_of_instruments()
    v = list_of_experiments(instr, fltr='cdb_')

Created on 2023-09-07 by Mikhail Dubrovin
"""
import os
import logging
logger = logging.getLogger(__name__)

import psana.pyalgos.generic.PSUtils as psu
from psana.pyalgos.generic.NDArrUtils import reshape_to_2d, reshape_to_3d, info_ndarr, np
from psana.detector.dir_root import DIR_FFB, DIR_DATA # DIR_FFB='/sdf/data/lcls/drpsrcf/ffb' or DIR_DATA='/sdf/data/lcls/ds/'

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

def random_image(shape=(64,64)):
    import psana.pyalgos.generic.NDArrGenerators as ag
    return ag.random_standard(shape, mu=0, sigma=10)

def image_from_kwargs(**kwa):
    """returns 2-d image array and geo (GeometryAccess) of available, otherwise None"""
    ndafname = kwa.get('ndafname', None)

    if not os.path.lexists(ndafname):
        logger.warning('ndarray file %s not found - use random image' % ndafname)
        return random_image(), None

    nda = np.load(ndafname)
    if nda is None:
        logger.warning('can not load ndarray from file: %s - use random image' % ndafname)
        return ndarandom_image(), None

    geofname = kwa.get('geofname', None)
    if not os.path.lexists(geofname):
        logger.warning('geometry file %s not found - use ndarray without geometry' % geofname)
        return image_from_ndarray(nda), None

    from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays
    geo = GeometryAccess(geofname)
    irows, icols = geo.get_pixel_coord_indexes(do_tilt=True, cframe=0)
    return img_from_pixel_arrays(irows, icols, W=nda), geo

def mask_ndarray_from_2d(mask2d, geo):
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess, convert_mask2d_to_ndarray # GeometryAccess, img_from_pixel_arrays
    assert isinstance(geo, GeometryAccess)
    irows, icols = geo.get_pixel_coord_indexes(do_tilt=True, cframe=0)
    irows = reshape_to_3d(irows)
    icols = reshape_to_3d(icols)
    #print(info_ndarr(irows, 'XXX irows'))
    #print(info_ndarr(icols, 'XXX icols'))
    logger.debug(info_ndarr(mask2d, 'input 2-d mask'))
    mask_nda = convert_mask2d_to_ndarray(mask2d, irows, icols) # , dtype=np.uint8)
    mask_nda.shape = irows.shape
    logger.debug(info_ndarr(mask_nda, 'output 3-d mask'))
    return mask_nda

def color_table(ict=2):
    import psana.graphqt.ColorTable as ct
    return ct.next_color_table(ict)  # OR ct.color_table_monochr256()

def list_of_instruments():
    #logger.debug(sys._getframe().f_code.co_name)
    dirins = DIR_DATA
    logger.debug('list_of_instruments in %s' % dirins)
    if os.path.lexists(dirins):
        return sorted(set([s.lower() for s in os.listdir(dirins) if len(s)==3]))
    else:
        logger.warning('list_of_instruments: DIRECTORY %s IS UNAVAILABLE - use default list' % dirins)
        return ['cxi', 'dia', 'mec', 'mfx', 'rix', 'tmo', 'tst', 'txi', 'ued', 'xcs', 'xpp']

def list_of_experiments(instr, fltr='cdb_'):
    direxp = '%s/%s' % (DIR_DATA, instr)
    logger.debug('list_of_experiments in %s' % direxp)
    if os.path.lexists(direxp):
        return psu.list_of_experiments(direxp=direxp)
    else:
        logger.warning('list_of_experiments: %s IS EMPTY' % direxp)
        return []
# EOF
