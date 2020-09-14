
import sys
import logging
logger = logging.getLogger(__name__)

#from time import time
#import psana.pyalgos.generic.Graphics as gg
#from psana.pscalib.geometry.SegGeometry import *
#from psana.detector.epix10k import DetectorImpl

#----------
#fname0 = '/reg/neh/home/cpo/git/psana_cpo/epix.xtc2'
#fname1 = '/reg/neh/home/cpo/git/psana_cpo/epix_2seg.xtc2'
fname0 = '/reg/g/psdm/detector/data2_test/xtc/data-mfxc00318-r0013-epix10kaquad-e000005.xtc2'
fname1 = '/reg/g/psdm/detector/data2_test/xtc/data-mfxc00318-r0013-epix10kaquad-e000005-seg1and3.xtc2'

#----------

def test_raw(fname):
    logger.info('in test_raw data from file:\n  %s' % fname)

    from psana import DataSource
    ds = DataSource(files=fname)
    myrun = next(ds.runs())
    det = myrun.Detector('epix10k2M')

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
    #print('r.dtype:', r.dtype)
    print('r.raw:', r.raw)

    #calib_const = det._calibconst
    #print('calib_const', calib_const)

    detname = r._uniqueid

    from psana.pscalib.calib.MDBWebUtils import calib_constants
    pedestals, _ = calib_constants(detname, exp='mfxc00318', ctype='pedestals', run=13)

    print('pedestals',pedestals)

    for evt in ds.events():
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
        '\n  python %s <test-name>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - test_raw("%s")'%fname0\
      + '\n    1 - test_raw("%s")'%fname1\

    d_loglev  ='DEBUG'

    import argparse

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('tname', type=str, help='test name')
    parser.add_argument('-l', '--loglev', default=d_loglev, type=str, help='logging level name, one of %s, def=%s' % (STR_LEVEL_NAMES, d_loglev))

    args = parser.parse_args()
    s = '\nArguments:'
    for k,v in vars(args).items(): s += '\n %8s: %s' % (k, str(v))

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.loglev])
    #logger.setLevel(intlevel)

    logger.info(s)

    tname = args.tname
    if   tname=='0': test_raw(fname0)
    elif tname=='1': test_raw(fname1)
    else: logger.warning('NON-IMPLEMENTED TEST: %s' % tname)

    sys.exit('END OF %s' % SCRNAME)

#----------

