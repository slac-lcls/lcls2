#!/usr/bin/env python

import sys
import logging
logger = logging.getLogger(__name__)

#import psana.pyalgos.generic.Graphics as gg
#from psana.pscalib.geometry.SegGeometry import *
#from psana.detector.epix10k import DetectorImpl

from psana.pyalgos.generic.NDArrUtils import info_ndarr # print_ndarr
#----------

fname0 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2'
fname1 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005-seg1and3.xtc2'

#----------

def print_det_raw_attrs(det):
    print('dir(det):', dir(det))
    print('det._dettype:', det._dettype)
    print('det._detid:', det._detid)
    print('det.calibconst:', det.calibconst)
    r = det.raw
    print('dir(det.raw):', dir(r))
    print('r._add_fields:', r._add_fields)
    print('r._calibconst:', r._calibconst)
    print('r._common_mode:', r._common_mode)
    print('r._configs:', r._configs)
    print('r._det_name:', r._det_name)
    print('r._dettype:', r._dettype)
    print('r._drp_class_name:', r._drp_class_name)
    print('r._env_store:', r._env_store)
    print('r._info:', r._info)
    print('r._return_types:', r._return_types)
    print('r._segments:', r._segments)
    print('r._sorted_segment_ids:', r._sorted_segment_ids)
    print('r._uniqueid:', r._uniqueid)
    print('r._var_name:', r._var_name)
    print('r.raw:', r.raw)
    #print('r.dtype:', r.dtype)

#----------

def test_raw(fname, args):
    logger.info('in test_raw data from file:\n  %s' % fname)

    from psana import DataSource
    ds = DataSource(files=fname)
    orun = next(ds.runs())
    det = orun.Detector('epix10k2M')

    if args.pattrs:
      print('dir(orun):', dir(orun))
      print('dir(det):', dir(det))
      print_det_raw_attrs(det)

    from psana.pyalgos.generic.Utils import str_attributes
    print(str_attributes(orun, cmt='\nattributes of orun %s:'% str(orun), fmt=', %s'))

    oraw = det.raw
    detnameid = oraw._uniqueid
    expname = orun.expt if orun.expt is not None else 'mfxc00318'
    runnum = orun.runnum
    print('expname:', expname)
    print('runnum:', runnum)
    #print('detname:', oraw._det_name)
    print('detname:', det._det_name)
    print('split detnameid:', '\n'.join(detnameid.split('_')))

    #sys.exit('TEST EXIT')


    calib_const = det.calibconst if hasattr(det,'calibconst') else None
    print('det.calibconst', calib_const.keys())


    from psana.pscalib.calib.MDBWebUtils import calib_constants

    logger.info('call calib_constants directly')

    pedestals, _ = calib_constants(detnameid, exp=expname, ctype='pedestals',    run=runnum)
    gain, _      = calib_constants(detnameid, exp=expname, ctype='pixel_gain',   run=runnum)
    rms, _       = calib_constants(detnameid, exp=expname, ctype='pixel_rms',    run=runnum)
    status, _    = calib_constants(detnameid, exp=expname, ctype='pixel_status', run=runnum)

    logger.info(info_ndarr(pedestals, 'pedestals'))
    logger.info(info_ndarr(gain,      'gain     '))
    logger.info(info_ndarr(rms,       'rms      '))
    logger.info(info_ndarr(status,    'status   '))

    myrun = next(ds.runs())
    for evnum,evt in enumerate(myrun.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        raw = det.raw.raw(evt)
        for segment,panel in raw.items():
            print(segment,panel.shape)
    print(50*'-')

#----------

if __name__ == "__main__":

    SCRNAME = sys.argv[0].rsplit('/')[-1]
    DICT_NAME_TO_LEVEL = logging._nameToLevel # {'INFO': 20, 'WARNING': 30, 'WARN': 30,...
    LEVEL_NAMES = [k for k in DICT_NAME_TO_LEVEL.keys() if isinstance(k,str)]
    STR_LEVEL_NAMES = ', '.join(LEVEL_NAMES)

    usage =\
        '\n  python %s <test-name> [optional-arguments]' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - test_raw("%s")'%fname0\
      + '\n    1 - test_raw("%s")'%fname1\

    d_loglev  = 'INFO' #'INFO' #'DEBUG'
    d_pattrs  = False

    import argparse

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('tname', type=str, help='test name')
    parser.add_argument('-l', '--loglev', default=d_loglev, type=str, help='logging level name, one of %s, def=%s' % (STR_LEVEL_NAMES, d_loglev))
    parser.add_argument('-P', '--pattrs', default=d_pattrs, action='store_true', help='print objects attrubutes, def=%s' % d_pattrs)

    args = parser.parse_args()
    kwa = vars(args)
    s = '\nArguments:'
    for k,v in kwa.items(): s += '\n %8s: %s' % (k, str(v))

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.loglev])
    #logger.setLevel(intlevel)

    logger.info(s)

    tname = args.tname
    if   tname=='0': test_raw(fname0, args)
    elif tname=='1': test_raw(fname1, args)
    else: logger.warning('NON-IMPLEMENTED TEST: %s' % tname)

    sys.exit('END OF %s' % SCRNAME)

#----------

