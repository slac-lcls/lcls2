#!/usr/bin/env python

import sys
import logging
logger = logging.getLogger(__name__)

from time import time
t0_sec = time()
import numpy as np
dt_sec = time()-t0_sec
print('np import consumed time (sec) = %.6f' % dt_sec)

from psana.pyalgos.generic.NDArrUtils import info_ndarr
from psana import DataSource

#----

fname0 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2'
fname1 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005-seg1and3.xtc2'


#on daq-det-drp01:
fname2 = '/u2/lcls2/tst/tstx00117/xtc/tstx00117-r0147-s000-c000.xtc2'
detname='epixquad'

#----

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


def test_calib_constants_directly(expname, runnum, detnameid):
    logger.info('in test_calib_constants_directly')
    from psana.pscalib.calib.MDBWebUtils import calib_constants

    pedestals, _ = calib_constants(detnameid, exp=expname, ctype='pedestals',    run=runnum)
    gain, _      = calib_constants(detnameid, exp=expname, ctype='pixel_gain',   run=runnum)
    rms, _       = calib_constants(detnameid, exp=expname, ctype='pixel_rms',    run=runnum)
    status, _    = calib_constants(detnameid, exp=expname, ctype='pixel_status', run=runnum)

    logger.info(info_ndarr(pedestals, 'pedestals'))
    logger.info(info_ndarr(gain,      'gain     '))
    logger.info(info_ndarr(rms,       'rms      '))
    logger.info(info_ndarr(status,    'status   '))


def det_calib_constants(det, ctype):
    calib_const = det.calibconst if hasattr(det,'calibconst') else None

    if calib_const is not None:
      logger.info('det.calibconst.keys(): ' + str(calib_const.keys()))
      cdata, cmeta = calib_const[ctype]
      logger.info('%s meta: %s' % (ctype, str(cmeta)))
      logger.info(info_ndarr(cdata, '%s data'%ctype))
      return cdata, cmeta
    else:
      logger.warning('det.calibconst is None')
      return None, None


def datasource_run(**kwa):
    ds = DataSource(**kwa)
    return ds, next(ds.runs())


def datasource_run_det(**kwa):
    ds = DataSource(**kwa)
    print('\nXXX dir(ds):', dir(ds))
    
    run = next(ds.runs())
    print('\nXXX dir(run):', dir(run))

    det = run.Detector(kwa.get('detname','opal'))
    print('\nXXX dir(det):', dir(det))
    
    return ds, run, det


def ds_run_det(fname, args):
    logger.info('ds_run_det input file:\n  %s' % fname)

    ds = DataSource(files=fname)
    orun = next(ds.runs())
    det = orun.Detector(args.detname)

    if args.pattrs:
      print('dir(orun):', dir(orun))
      print('dir(det):', dir(det))
      print_det_raw_attrs(det)

    from psana.pyalgos.generic.Utils import str_attributes
    print(str_attributes(orun, cmt='\nattributes of orun %s:'% str(orun), fmt=', %s'))

    oraw = det.raw
    detnameid = oraw._uniqueid
    expname = orun.expt if orun.expt is not None else args.expname # 'mfxc00318'
    runnum = orun.runnum
    print('expname:', expname)
    print('runnum :', runnum)
    #print('detname:', oraw._det_name)
    print('detname:', det._det_name)
    print('split detnameid:', '\n'.join(detnameid.split('_')))

    print(50*'=')
    test_calib_constants_directly(expname, runnum, detnameid)

    peds_data, peds_meta = det_calib_constants(det, 'pedestals')

    #sys.exit('TEST EXIT')
    return ds, orun, det


def test_raw(fname, args):
    logger.info('in test_raw data from file:\n  %s' % fname)
    ds, run, det = ds_run_det(fname, args)

    for evnum,evt in enumerate(run.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        segs = det.raw.segments(evt)
        raw  = det.raw.raw(evt)
        logger.info(info_ndarr(segs, 'segs '))
        logger.info(info_ndarr(raw,  'raw  '))
    print(50*'-')


def test_calib(fname, args):
    from time import time

    logger.info('in test_raw data from file:\n  %s' % fname)

    ds = DataSource(files=fname, detname=args.detname)

    run = next(ds.runs())
    #print('\nXXX dir(run):', dir(run))
    print('XXX runnum       : ', run.runnum)   # 147
    print('XXX run.detnames : ', run.detnames) # {'epixquad'}
    print('XXX run.expt     : ', run.expt)     # tstx00117
    print('XXX run.id       : ', run.id)       # 0
    print('XXX run.timestamp: ', run.timestamp)# 4190613356186573936 (int)
    t_sec = int(time())
    print('XXX tnow time %d sec as tstamp: %d' % (t_sec,t_sec<<32))
    print('XXX run.stepinfo : ', run.stepinfo) # {('epixquad', 'step'): ['value', 'docstring'], ('epixquadhw', 'step'): ['value', 'docstring']}

    #ts = run.timestamp
    #sec = float(ts >> 32) #& 0xffffffff
    #nsec = ts & 0xffffffff
    #print('XXX timestamp: %d sec %d nsec'%(sec,nsec))# 4190613356186573936 today sec:1607015429

    det = run.Detector(args.detname)
    print('XXX det.calibconst.keys(): ', det.calibconst.keys()) # dict_keys(['geometry'])
    #print(det.calibconst)
    print('XXX det._det_name: ', det._det_name) # epixquad
    print('XXX det._dettype : ', det._dettype)  # epix
    print('XXX det._detid   : ', det._detid)    # -

    print('det._configs:', det._configs)        # [<dgram.Dgram object at 0x7f7794082d40>]???? WHY IT IS A LIST? HOW TO GET LIST INDEX FOR DETECTOR?
    cfg = det._configs[0]
    print('\ndir(cfg):', dir(cfg))              # [..., '_dgrambytes', '_file_descriptor', '_offset', '_size', '_xtc', 'epixquad', 'epixquadhw', 'service', 'software', 'timestamp']

    print('\ndir(det.raw):', dir(det.raw))      # [..., '_add_fields', '_calibconst', '_common_mode', '_configs', '_det_name', '_dettype', '_drp_class_name', '_env_store', '_info', '_return_types', '_seg_configs', '_segments', '_sorted_segment_ids', '_uniqueid', '_var_name', 'array', 'cached_pixel_coord_indexes', 'calib', 'det_calibconst', 'det_geo', 'det_geotxt_and_meta', 'geo', 'image', 'interpol_pars', 'pix_rc', 'pix_xyz', 'pixel_coord_indexes', 'pixel_coords', 'raw', 'segments']

    seg_cfgs = det.raw._seg_configs()
    print('det.raw._seg_configs():', det.raw._seg_configs())

    for i,scfg in seg_cfgs.items():
        print('\n== Segment %d'%i)
        #print('  scfg', scfg) # container.Container object
        #print('  dir(scfg.config):', dir(scfg.config)) # [..., 'asicPixelConfig', 'trbit']
        print(info_ndarr(scfg.config.asicPixelConfig, '  scfg.config.asicPixelConfig: ')) # shape:(4, 178, 192) size:136704 dtype:uint8 [12 12 12 12 12]...
        print('  scfg.config.trbit:', scfg.config.trbit) # [1 1 1 1]

#    exit('TEST EXIT')
#################


    for stepnum,step in enumerate(run.steps()):

        print('%s\nStep %1d' % (50*'_',stepnum))
        #print('STEP dir(step):', dir(step))   # [..., 'esm', 'events', 'evt', 'evt_iter']            #print('STEP dir(det.step):', dir(det.step)) #'dgrams', 'docstring', 'env_store', 'value']

        #continue


        for evnum,evt in enumerate(step.events()):
            if evnum>5 and evnum%200!=0: continue
            print('%s\nStep %1d Event %04d' % (50*'_',stepnum, evnum))
            segs = det.raw._segments(evt)
            raw  = det.raw.raw(evt)
            logger.info('segs: %s' % str(segs))
            logger.info(info_ndarr(raw,  'raw  '))

            print('STEP det.step.docstring(evt):', det.step.docstring(evt))
            print('STEP det.step.value(evt):', det.step.value(evt))


        print(50*'-')


def test_image(fname, args):

    logger.info('in test_image data from file:\n  %s' % fname)
    ds, run, det = ds_run_det(fname, args)

    flimg = None

    for evnum,evt in enumerate(run.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        arr = det.raw.calib(evt)
        logger.info(info_ndarr(arr, 'arr   '))
        if arr is None: continue

        #=======================
        #arr = np.ones_like(arr)
        #=======================


        t0_sec = time()

        img = det.raw.image(evt, nda=arr, pix_scale_size_um=args.pscsize, mapmode=args.mapmode)
        #img = det.raw.image(evt)
        dt_sec = time()-t0_sec
        logger.info('image composition time = %.6f sec ' % dt_sec)

        logger.info(info_ndarr(img, 'image '))
        if img is None: continue

        alimits = (img.min(),img.max()) if args.mapmode == 4 else\
                  None if args.mapmode else\
                  (0,4)

        if args.dograph:

            if flimg is None:
                from UtilsGraphics import gr, fleximage
                flimg = fleximage(img, arr=arr, h_in=8, nneg=1, npos=3, alimits=alimits) #, cmap='jet')
                #flimg = fleximage(img, h_in=8, alimits=(0,4)) #, cmap='jet')
            else:
                flimg.update(img, arr=arr)

            gr.show(mode=1)

    if args.dograph:
        gr.show()
        if args.ofname is not None:
            gr.save_fig(flimg.fig, fname=args.ofname, verb=True)

    print(50*'-')

#----

if __name__ == "__main__":

    SCRNAME = sys.argv[0].rsplit('/')[-1]
    DICT_NAME_TO_LEVEL = logging._nameToLevel # {'INFO': 20, 'WARNING': 30, 'WARN': 30,...
    LEVEL_NAMES = [k for k in DICT_NAME_TO_LEVEL.keys() if isinstance(k,str)]
    STR_LEVEL_NAMES = ', '.join(LEVEL_NAMES)


    tname = sys.argv[1] if len(sys.argv)>1 else '100'
    usage =\
        '\n  python %s <test-name> [optional-arguments]' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - test_raw("%s")'%fname0\
      + '\n    1 - test_raw("%s")'%fname1\
      + '\n    2 - test_image("%s")'%fname0\
      + '\n    3 - test_image("%s")'%fname1\
      + '\n    4 - test_calib("%s") DATA FILE IS AVAILABLE ON daq-det-drp01 ONLY'%fname2\
      + '\n ==== '\
      + '\n    ./%s 2 -m0 -s101' % SCRNAME\
      + '\n    ./%s 2 -m1' % SCRNAME\
      + '\n    ./%s 2 -m2 -lDEBUG' % SCRNAME\
      + '\n    ./%s 2 -m3 -s101 -o img.png' % SCRNAME\
      + '\n    ./%s 2 -m4' % SCRNAME\

    d_loglev  = 'INFO' #'INFO' #'DEBUG'
    d_pattrs  = False
    d_dograph = True
    d_detname = 'epixquad'  if tname=='4' else 'epix10k2M'
    d_expname = 'tstx00117' if tname=='4' else 'mfxc00318'
    d_ofname  = None
    d_mapmode = 1
    d_pscsize = 100

    h_loglev  = 'logging level name, one of %s, def=%s' % (STR_LEVEL_NAMES, d_loglev)
    h_mapmode = 'multi-entry pixels image mappimg mode 0/1/2/3 = statistics of entries/last pix intensity/max/mean, def=%s' % d_mapmode

    import argparse

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('tname', type=str, help='test name')
    parser.add_argument('-l', '--loglev',  default=d_loglev,  type=str, help=h_loglev)
    parser.add_argument('-d', '--detname', default=d_detname, type=str, help='detector name, def=%s' % d_detname)
    parser.add_argument('-e', '--expname', default=d_expname, type=str, help='experiment name, def=%s' % d_expname)
    parser.add_argument('-P', '--pattrs',  default=d_pattrs,  action='store_true',  help='print objects attrubutes, def=%s' % d_pattrs)
    parser.add_argument('-G', '--dograph', default=d_dograph, action='store_false', help='plot graphics, def=%s' % d_pattrs)
    parser.add_argument('-o', '--ofname',  default=d_ofname,  type=str, help='output image file name, def=%s' % d_ofname)
    parser.add_argument('-m', '--mapmode', default=d_mapmode, type=int, help=h_mapmode)
    parser.add_argument('-s', '--pscsize', default=d_pscsize, type=float, help='pixel scale size [um], def=%.1f' % d_pscsize)

    args = parser.parse_args()
    kwa = vars(args)
    s = '\nArguments:'
    for k,v in kwa.items(): s += '\n %8s: %s' % (k, str(v))

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.loglev])
    #logger.setLevel(intlevel)

    logger.info(s)

    tname = args.tname
    if   tname=='0': test_raw  (fname0, args)
    elif tname=='1': test_raw  (fname1, args)
    elif tname=='2': test_image(fname0, args)
    elif tname=='3': test_image(fname1, args)
    elif tname=='4': test_calib(fname2, args)
    else: logger.warning('NON-IMPLEMENTED TEST: %s' % tname)

    exit('END OF %s' % SCRNAME)

#----

