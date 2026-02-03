"""
:py:class:`UtilsCalibRepo`
==========================

Usage::
    import psana.detector.UtilsCalibRepo as ucr

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@datetime 2025-03-19
@author Mikhail Dubrovin
@see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/...>`_.
"""

import os
import sys
import numpy as np

import logging
logger = logging.getLogger(__name__)

import psana.detector.Utils as uts
import psana.detector.UtilsCalib as uc
import psana.detector.utils_psana as ups
import psana.pscalib.calib.CalibConstants as cc
from psana.detector.NDArrUtils import info_ndarr, save_2darray_in_textfile, save_ndarray_in_textfile
from psana.detector.RepoManager import set_repoman_and_logger, fname_prefix, calib_file_name

def save_constants_in_repository(dic_consts, **kwa):
    """dic_consts = {<ctype>: <nda-for-ctype>,...}"""

    #CTYPE_DTYPE = cc.dic_calib_name_to_dtype # {'pedestals': np.float32,...}
    #repoman  = kwa.get('repoman', None)
    expname  = kwa.get('exp', None)
    #detname  = kwa.get('detname', None)
    dettype  = kwa.get('dettype', None)
    #deploy   = kwa.get('deploy', False)
    #dirrepo  = kwa.get('dirrepo', './work')
    #dirmode  = kwa.get('dirmode',  0o2775)
    filemode = kwa.get('filemode', 0o664)
    #group    = kwa.get('group', 'ps-users')
    #tstamp   = kwa.get('tstamp', '2010-01-01T00:00:00')
    tsshort  = kwa.get('tsshort', '20100101000000')
    runnum   = kwa.get('run_orig', None)
    #uniqueid = kwa.get('uniqueid', 'not-def-id')
    segids   = kwa.get('segment_ids', [])  # self._uniqueid.split('_')[1]
    segnums  = kwa.get('segment_inds', []) # self._sorted_segment_inds # _segment_numbers in entire det
    segind   = kwa.get('segind', None) # segment index in det.raw.raw
    gainmode = kwa.get('gainmode', None)
    #longname = kwa.get('longname', 'non-def-longname') # odet.raw._uniqueid
    shortname= kwa.get('shortname', 'non-def-shortname') # uc.detector_name_short(longname)

    repoman = set_repoman_and_logger(kwa)
    repoman.makedir_dettype(dettype)

    d = ups.dict_filter(kwa, list_keys=('dskwargs', 'dirrepo', 'ctype',\
                                        'dettype', 'tsshort', 'detname', 'longname', 'shortname',\
                                        'gainmode', 'segment_ids', 'segment_inds', 'version'))
    logger.debug('save_constants_in_repository kwa: %s' % uts.info_dict(kwa))
    logger.info('essential kwargs:%s' % uts.info_dict(d, fmt='  %12s: %s', sep='\n'))

    dic_ctype_fmt = uc.dic_ctype_fmt(**kwa)

    for i,(segnum,segid) in enumerate(zip(segnums, segids)):

      if segind is not None and i != segind:
          logger.debug('---- skip daq segment:%02d, segnum:%02d id:%s   save only --segind %d' % (i, segnum, segid, segind))
          continue

      logger.info('%s next segment\n   save segment constants for gain mode: %s in repo for raw ind:%02d segment num:%02d id: %s'%\
                  (20*'-', gainmode, i, segnum, segid))

      for ctype, nda in dic_consts.items():
        dir_ct = repoman.makedir_ctype(segid, ctype)
        fprefix = fname_prefix(shortname, segnum, tsshort, expname, runnum, dir_ct)
        fname = calib_file_name(fprefix, ctype, gainmode)
        fmt = dic_ctype_fmt.get(ctype,'%.5f')
        arr2d = None
        if nda.ndim == 2:
          arr2d = nda
          save_2darray_in_textfile(arr2d, fname, filemode, fmt)
        elif dettype == 'jungfrau': # DO THIS in order to not possibly brake it for other detectors...
          arr2d = nda[i,:]
          save_2darray_in_textfile(arr2d, fname, filemode, fmt)
        else:
          save_ndarray_in_textfile(nda, fname, filemode, fmt) # save nda AS IS 3-d ex. for epixm ?????
          logger.warning('IS THIS CODE FOR N-d ARRAY IS USED ANYWHERE ?\n' + info_ndarr(nda, 'array of %s' % ctype))

        logger.debug(info_ndarr(arr2d, 'array of %s' % ctype))
        logger.info('saved: %s' % fname)


def _set_segment_ind_and_id(kwa):
    """re-define lists of detector segment_inds and segment_ids for specified detector or segind"""
    dettype  = kwa.get('dettype', None)
    segind   = kwa.get('segind', None)

    if dettype == 'epixm320':

        d = ups.dict_filter(kwa, list_keys=('segind', 'segment_inds', 'segment_ids', 'longname'))
        logger.info('INPUT kwargs:%s' % uts.info_dict(d, fmt='  %12s: %s', sep='\n'))

        if segind is None:
            longname = kwa.get('longname', 'UNDEFINED')
            panel_id = longname.split('_')[1] # 0016778240-0176075265-0452984854-4021594881-1962934296-0177446913-0402653206
            kwa['segment_ids'] = [panel_id,]
        else:
            segment_numbers = kwa.get('segment_numbers', []) # [0, 1, 2, 3]
            assert segind in segment_numbers,\
                 'specified segment index "--segind %d" is not available in the list of det.raw._segment_numbers: %s'%\
                 (segind, str(segment_numbers))
            kwa['segment_inds'] = segment_numbers

        d = ups.dict_filter(kwa, list_keys=('segind', 'segment_inds', 'segment_ids'))
        logger.info('RE-DEFINED kwargs:%s' % uts.info_dict(d, fmt='  %12s: %s', sep='\n'))

    #sys.exit('TEST EXIT')


def _check_gainmode_with_assert(gainmode, lst_gainmodes, dettype):
    try:
        assert gainmode in lst_gainmodes
    except AssertionError:
        print('WARNING: specified --gainmode %s is missing in the list of gainmodes %s for dettype: %s'%\
              (gainmode, str(lst_gainmodes), dettype))
        sys.exit(1)


def _check_gainmode(**kwa):
    gainmode = kwa.get('gainmode', None)
    dettype  = kwa.get('dettype', None)
    if dettype == 'jungfrau':
        from psana.detector.UtilsJungfrauCalib import DIC_GAIN_MODE
        lst_gainmodes = DIC_GAIN_MODE.keys() # ['DYNAMIC', 'FORCE_SWITCH_G1', 'FORCE_SWITCH_G2']
        _check_gainmode_with_assert(gainmode, lst_gainmodes, dettype)
    elif dettype == 'epix10ka':
        from psana.detector.UtilsEpix10ka import GAIN_MODES #, GAIN_MODES_IN
        lst_gainmodes = GAIN_MODES # ['FH','FM','FL','AHL-H','AML-M','AHL_L','AML_L']
        _check_gainmode_with_assert(gainmode, lst_gainmodes, dettype)
#    return gmode


def _check_ctype(kwa):
    from psana.pscalib.calib.CalibConstants import list_calib_names
    ctype = kwa.get('ctype', None)
    try:
        assert ctype in list_calib_names
    except AssertionError:
        print('WARNING: calibration type "%s" is missing in the list of known calib types:\n%s'%\
              (ctype, str(list_calib_names)))
        sys.exit(1)
#    return ctype


def _segcons_from_file(**kwa):
    fname = kwa.get('fname2darr', None)
    assert os.path.exists(fname), 'MISSING file --fname2darr %s' % fname
    arr = np.load(fname)
    seggeo_shape = kwa.get('seggeo_shape', None)
    try:
        assert arr.shape == seggeo_shape
    except AssertionError:
        logger.warning('calibration constants from file: %s ' % fname \
            +'have shape: %s ' % str(arr.shape)\
                     +'different from the segment geometry shape: %s ' % str(seggeo_shape))
        #sys.exit(1)
    logger.info(info_ndarr(arr, 'constants from file: %s' % fname, last=6))
    return arr


def _set_tstamp(kwa):
    ts = kwa.get('tstampbegin', None)
    if ts is None: return None
    try:
        its = int(ts)
    except ValueError:
        print('WARNING: timestamp "%s" can not be convertrd to digital' % ts)
        sys.exit(1)
    assert 19900701000000 < its
    assert its < 21000101000000
    kwa['tsshort'] = ts
#    return ts


def save_segment_constants_in_repository(**kwa):

    repoman = set_repoman_and_logger(kwa)

    from psana.detector.Utils import info_dict
    #print('input kwargs: %s' % info_dict(kwa))
    from psana import DataSource
    dskwargs = ups.data_source_kwargs(**kwa)
    logger.debug('DataSource kwargs: %s' % str(dskwargs))

    ds = DataSource(**dskwargs)
    orun = next(ds.runs())
    odet = orun.Detector(kwa.get('detname', None))

    kwa_save = uc.add_metadata_kwargs(orun, odet, **kwa)
    _set_segment_ind_and_id(kwa_save) # pass kwa_save without **, as mutable
    _check_gainmode(**kwa_save)
    _check_ctype(kwa_save)
    _set_tstamp(kwa_save)

    logger.info('kwa_save:%s' % info_dict(kwa_save))
    print(50*'=')
    arr_seg = _segcons_from_file(**kwa_save)
    ctype = kwa_save.get('ctype', None)

    save_constants_in_repository({ctype: arr_seg}, **kwa_save)


if __name__ == "__main__":

    def nda_random(mu=100, sigma=5, shape=(512,1024), dtype=np.float32):
        a = mu + sigma*np.random.standard_normal(size=shape)
        return (a).astype(dtype=dtype)

    kwa = {'dskwargs'    : 'exp=ascdaq023,run=37',\
           'detname'     : 'jungfrau',\
           'dirrepo'     : './work',\
           'ctype'       : 'pedestals',\
           'gainmode'    : 'g0',\
           'segind'      : 1,\
           'tstampbegin' : '20250101000000',\
           'fname2darr'  : 'test_2darr.npy',\
           'version'     : '2025-03-14',\
    }

    np.save(kwa['fname2darr'], nda_random())

    save_segment_constants_in_repository(**kwa)

    sys.exit('End of %s' % sys.argv[0])

# EOF
