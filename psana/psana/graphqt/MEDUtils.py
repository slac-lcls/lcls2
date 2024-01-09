#!/usr/bin/env python

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

    Methods on 2023-10-04:

    is_none(v, msg='value is None', meth=logger.debug):
    image_from_ndarray(nda):
    random_image(shape=(64,64)):
    experiment_from_dskwargs(s):
    dbname_colname(**kwa):
    geo_text_and_meta_from_db(**kwa):
    image_from_kwargs(**kwa):
    image_from_geo_and_nda(geo, nda, vbase=0):
    mask_ndarray_from_2d(mask2d, geo):
    color_table(ict=2):
    list_of_instruments():
        logger.warning('list_of_instruments: DIRECTORY %s IS UNAVAILABLE - use default list' % dirins)
    list_of_experiments(instr): #  fltr='cdb_'
    db_names(fltr=None):
    db_namesroot(dbnames=None, fltr=None):
    db_expnames(dbnames=None, fltr=None):
    db_detnames(dbnames=None, fltr=None):
    db_instruments(dbnames=None):
    db_dettypes(dbnames=None):
    ds_run_from_str_dskwargs(dskwargs='exp=uedcom103,run=812'):
    data_detnames(dskwargs='exp=tmoc00321,run=3'): #,dir=/sdf/data/lcls/ds/ued/uedcom103/xtc/
    data_det_uniqueid(dskwargs='exp=uedcom103,run=812', detname_daq='epixquad'):
    datasource_kwargs_from_string(s):
    datasource_kwargs_to_string(**kwargs):


Created on 2023-09-07 by Mikhail Dubrovin
"""
import os
import logging
logger = logging.getLogger(__name__)

#from time import time
import psana.pyalgos.generic.PSUtils as psu
from psana.pyalgos.generic.NDArrUtils import reshape_to_2d, reshape_to_3d, info_ndarr, np
import psana.pscalib.calib.MDBWebUtils as mdbwu
from psana.detector.dir_root import DIR_FFB, DIR_DATA, DIR_DATA_TEST #, DIR_FFB='/sdf/data/lcls/drpsrcf/ffb' or DIR_DATA='/sdf/data/lcls/ds/'
import psana.detector.Utils as ut  # info_dict, info_command_line, info_namespace, info_parser_arguments, str_tstamp

#char_expand    = u' \u25BC' # down-head triangle
#char_shrink    = u' \u25B2' # solid up-head triangle

collection_names, find_doc, find_docs, get_data_for_doc =\
    mdbwu.collection_names, mdbwu.find_doc, mdbwu.find_docs, mdbwu.get_data_for_doc

def is_none(v, msg='value is None', meth=logger.debug):
    r = v is None
    if r: meth(msg)
    return r # True or False

def image_from_ndarray(nda):
    if nda is None:
       logger.debug('nda is None - return None for image')
       return None

    if not isinstance(nda, np.ndarray):
       logger.debug('nda is not np.ndarray, type(nda): %s - return None for image' % type(nda))
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

def experiment_from_dskwargs(s):
    """returns experiment from string dskwargs"""
    dskwa = datasource_kwargs_from_string(s)
    logger.debug('dskwargs: %s' % str(dskwa))
    exp = dskwa.get('exp', None)
    return dskwa.get('exp', None)

def dbname_colname(**kwa):
    """returns dbname, colname from **kwa"""
    logger.debug('**kwa: %s' % str(kwa))
    s   = kwa.get('dskwargs', None)
    exp = experiment_from_dskwargs(s)
    colname = kwa.get('detname', None)
    if colname == 'Selected':
        logger.warning('DETECTOR IS NOT SELECTED')
        return exp, None
    dbname = 'cdb_%s' % (colname if exp is None else exp)
    return dbname, colname

def geo_text_and_meta_from_db(**kwa):
    s = kwa.get('dskwargs', None)
    if is_none(s, msg='str dskwargs is None'): return None, None
    dskwa = datasource_kwargs_from_string(s)
    dbname, colname = dbname_colname(**kwa)
    if is_none(dbname, msg='dbname is None'): return None, None
    if is_none(colname, msg='colname is None'): return None, None
    run = dskwa.get('run', 9999)
    query = {'ctype':'geometry', 'run':{'$lte':run}}
    logger.debug('\n  set geometry for dbname: %s colname: %s\n     query: %s' % (dbname, colname, query))
    doc = find_doc(dbname, colname, query=query)
    if is_none(doc, msg='doc is None'): return None, None
    geo_txt = get_data_for_doc(dbname, doc)
    logger.debug('geometry constants from DB:\n%s' % geo_txt)
    return geo_txt, doc

def image_from_kwargs(**kwa):
    """returns 2-d image array, geo (GeometryAccess) of available, otherwise None, and geo_doc if from DB or None"""
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess

    img, geo = None, None
    geo_doc  = None
    nda      = kwa.get('nda', None)
    geo_txt  = kwa.get('geo_txt', None)
    ndafname = kwa.get('ndafname', None)
    geofname = kwa.get('geofname', None)

    logger.debug(info_ndarr(nda, 'nda'))
    logger.debug('geo_txt: %s' % ('None' if geo_txt is None else geo_txt[:100]))
    logger.debug('ndafname: %s' % ndafname)
    logger.debug('geofname: %s' % geofname)

    # get nda  from (1) nda or (2) ndafname
    if nda is None:
        if ndafname in (None, 'Select') or not os.path.lexists(ndafname):
            logger.debug('nda is None and ndafname %s not found' % ndafname)
        else:
            nda = np.load(ndafname)
            if nda is None:
                logger.debug('can not load ndarray from file: %s' % ndafname)
            else:
                logger.info('ndarray of shape %s for image is loaded from file: %s' % (nda.shape, ndafname))

    # get geo from (1) geo_txt, (2) geofname, (3) DB
    if geo_txt is None:
        if geofname is None or not os.path.lexists(geofname):
            logger.debug('geo_txt is None and geometry file %s not found - try to find in DB' % geofname)
            geo_txt, geo_doc = geo_text_and_meta_from_db(**kwa)
            if geo_txt is not None:
                d = geo_doc
                logger.info('geometry text is loaded from DB for exp: %s det: %s run: %s' % (d['experiment'], d['detector'], str(d['run'])))
                geo = GeometryAccess()
                geo.load_pars_from_str(geo_txt)
        else:
            logger.info('geometry is loaded from file: %s' % str(geofname))
            geo = GeometryAccess(geofname)
    else:
        geo = GeometryAccess()
        geo.load_pars_from_str(geo_txt)
        logger.info('geometry is set for geo_txt: %s ...' % geo_txt[:200])

    return image_from_geo_and_nda(geo, nda), geo, geo_doc

def image_from_geo_and_nda(geo, nda, vbase=0):
    """returns 2-d image array and GeometryAccess object for geo, nda"""
    from psana.pscalib.geometry.GeometryAccess import img_from_pixel_arrays
    if geo is None:
        return random_image() if nda is None else\
               image_from_ndarray(nda)
    else:
        irows, icols = geo.get_pixel_coord_indexes(do_tilt=True, cframe=0)
        return img_from_pixel_arrays(irows, icols, W=nda, vbase=vbase)

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

def list_of_experiments(instr): #  fltr='cdb_'
    direxp = '%s/%s' % (DIR_DATA, instr)
    logger.debug('list_of_experiments in %s' % direxp)
    if os.path.lexists(direxp):
        return psu.list_of_experiments(direxp=direxp)
    else:
        logger.warning('list_of_experiments: %s IS EMPTY' % direxp)
        return []

def db_names(fltr=None):
    dbnames = mdbwu.database_names()
    return dbnames if fltr is None else\
           [n for n in mdbwu.database_names() if fltr in n]

def db_namesroot(dbnames=None, fltr=None):
    _dbnames = [n.strip('cdb_') for n in (db_names() if dbnames is None else dbnames)]
    return _dbnames if fltr is None else\
          [n for n in _dbnames if fltr in n]

def db_expnames(dbnames=None, fltr=None):
    return [n for n in db_namesroot(dbnames, fltr) if not '_' in n and (len(n)>7) and (len(n)<12)]

def db_detnames(dbnames=None, fltr=None):
    return [n for n in db_namesroot(dbnames, fltr) if '_' in n]

def db_instruments(dbnames=None):
    return sorted(set([n[:3] for n in db_expnames(dbnames)]))

def db_dettypes(dbnames=None):
    return sorted(set([n.split('_')[0] for n in db_detnames(dbnames)]))

def ds_run_from_str_dskwargs(dskwargs='exp=uedcom103,run=812'):
    from psana import DataSource # dt(sec)=0.000004
    from psana.detector.utils_psana import datasource_kwargs_from_string # dt(sec)=0.000612
    kwargs = datasource_kwargs_from_string(dskwargs)
    ods = DataSource(**kwargs)
    #t0_sec = time()
    orun = next(ods.runs()) # dt(sec)=4.244698
    #print('next(ods.runs()) time = %.6f' % (time() - t0_sec))
    #det = orun.Detector(detname)
    return ods, orun

def data_detnames(dskwargs='exp=tmoc00321,run=3'): #,dir=/sdf/data/lcls/ds/ued/uedcom103/xtc/
    ods, orun = ds_run_from_str_dskwargs(dskwargs)
    return orun.detnames

def detector_uniqueid_namedb(dskwargs='exp=uedcom103,run=812', detname_daq='epixquad'):
    """Returns odet.raw._uniqueid (lond name recognizable in DB) for (str) DataSource dskwargs.
       Dataset test cmd: detnames exp=uedcom103,run=812,dir=/sdf/data/lcls/ds/ued/uedcom103/xtc/ epixquad
    """
    ods, orun = ds_run_from_str_dskwargs(dskwargs) #  dt(sec)=4.2
    odet = orun.Detector(detname_daq) # dt(sec)=0.007404
    #return orun.detnames
    #odet.raw._uniqueid   # epixhremu_00cafe0002-0000000000-0000000000-0000000000-...
    #odet.raw._det_name   # epixhr_emu
    #odet.raw._dettype    # epixhremu
    uniqueid = odet.raw._uniqueid
    return uniqueid, mdbwu.pro_detector_name(uniqueid)

def datasource_kwargs_from_string(s):
    from psana.psexp.utils import datasource_kwargs_from_string
    return datasource_kwargs_from_string(s)

def datasource_kwargs_to_string(**kwargs):
    from psana.psexp.utils import datasource_kwargs_to_string
    return datasource_kwargs_to_string(**kwargs)


if __name__ == "__main__":
    print('\n=== list_of_instruments:\n%s' % str(list_of_instruments()))
    dbnames = db_names()
    print('\n=== db_names:\n%s' % str(dbnames))
    print('\n=== db_namesroot:\n%s' % str(db_namesroot(dbnames)))
    print('\n=== db_expnames:\n%s' % str(db_expnames(dbnames)))
    print('\n=== db_detnames:\n%s' % str(db_detnames(dbnames)))
    print('\n=== db_instruments:\n%s' % str(db_instruments(dbnames)))
    print('\n=== db_dettypes:\n%s' % str(db_dettypes(dbnames)))
    print('\n=== db_expnames(fltr="tmo"):\n%s' % str(db_expnames(dbnames, fltr='tmo')))
    print('\n=== db_detnames(fltr="epixhr2x2"):\n%s' % str(db_detnames(dbnames, fltr='epixhr2x2')))
    #print('\n=== data_detnames:\n%s' % str(data_detnames(dskwargs='exp=tmoc00321,run=3')))
    uniqueid, detnamedb = detector_uniqueid_namedb(dskwargs='exp=uedcom103,run=812', detname_daq='epixquad')
    print('\n=== detector_uniqueid_namedb:\n  %s\n  %s' % (uniqueid, detnamedb))

# EOF
