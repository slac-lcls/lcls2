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


#print('DATA FILE IS AVAILABLE ON daq-det-drp01 ONLY')
#fname2 = '/u2/lcls2/tst/tstx00117/xtc/tstx00117-r0147-s000-c000.xtc2'
"""
Wed 12/9/2020 8:13 PM
Runs 27 and 28 are properly configured pedestal calibrations.  Run 29 is a partial charge injection calibration (for the usual reasons).
-Matt
"""
#print('DATA FILE IS AVAILABLE ON drp-ued-cmp001 ONLY')
#fname2 = '/u2/pcds/pds/ued/ueddaq02/xtc/ueddaq02-r0028-s000-c000.xtc2' #dark
#fname2 = '/reg/d/psdm/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2' #dark
fname2 = '/cds/data/psdm/ued/ueddaq02/xtc/ueddaq02-r0027-s000-c000.xtc2' #dark
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
    print('r._common_mode_:', r._common_mode_)
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
    #'pixel_rms', 'pixel_status', 'pedestals', 'pixel_gain', 'geometry'
    pedestals, _ = calib_constants(detnameid, exp=expname, ctype='pedestals',    run=runnum)
    gain, _      = calib_constants(detnameid, exp=expname, ctype='pixel_gain',   run=runnum)
    rms, _       = calib_constants(detnameid, exp=expname, ctype='pixel_rms',    run=runnum)
    status, _    = calib_constants(detnameid, exp=expname, ctype='pixel_status', run=runnum)

    logger.info(info_ndarr(pedestals, 'pedestals'))
    logger.info(info_ndarr(gain,      'gain     '))
    logger.info(info_ndarr(rms,       'rms      '))
    logger.info(info_ndarr(status,    'status   '))


def det_calib_constants(det, ctype):
    #ctype = 'pixel_rms', 'pixel_status', 'pedestals', 'pixel_gain', 'geometry'
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


def ds_run_det(args):

    logger.info('ds_run_det input file:\n  %s' % args.fname)

    kwa = {'files':args.fname,} if args.fname is not None else\
          {'exp':args.expname,'run':int(args.runs.split(',')[0])}
    #ds = DataSource(exp=args.expt, run=args.run, dir=f'/cds/data/psdm/{args.expt[:3]}/{args.expt}/xtc')
    ds = DataSource(**kwa)
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

    print('run.detnames : ', orun.detnames) # {'epixquad'}
    print('run.expt     : ', orun.expt)     # tstx00117
    print('run.id       : ', orun.id)       # 0
    print('run.timestamp: ', orun.timestamp)# 4190613356186573936 (int)

    print('fname:', args.fname)
    print('expname:', expname)
    print('runnum :', runnum)
    print('detname:', det._det_name)
    print('split detnameid:', '\n'.join(detnameid.split('_')))
    print(50*'=')
    test_calib_constants_directly(expname, runnum, detnameid)

    peds_data, peds_meta = det_calib_constants(det, 'pedestals')

    return ds, orun, det


def selected_record(nrec):
    return nrec<5\
       or (nrec<50 and not nrec%10)\
       or (nrec<500 and not nrec%100)\
       or (not nrec%1000)


def test_raw(args):

    ds, run, det = ds_run_det(args)

    for stepnum,step in enumerate(run.steps()):
      print('%s\nStep %1d' % (50*'_',stepnum))

      for evnum,evt in enumerate(step.events()):
        if evnum>args.evtmax: exit('exit by number of events limit %d' % args.evtmax)
        if not selected_record(evnum): continue
        print('%s\nEvent %04d' % (50*'_',evnum))
        segs = det.raw.segments(evt)
        raw  = det.raw.raw(evt)
        logger.info(info_ndarr(segs, 'segs '))
        logger.info(info_ndarr(raw,  'raw  '))
    print(50*'-')


def test_calib(args):
    from time import time

    ds, run, det = ds_run_det(args)

    t_sec = int(time())
    print('XXX tnow time %d sec as tstamp: %d' % (t_sec,t_sec<<32))
    #print('XXX run.stepinfo : ', run.stepinfo) # {('epixquad', 'step'): ['value', 'docstring'], ('epixquadhw', 'step'): ['value', 'docstring']}

    #ts = run.timestamp
    #sec = float(ts >> 32) #& 0xffffffff
    #nsec = ts & 0xffffffff
    #print('XXX timestamp: %d sec %d nsec'%(sec,nsec))# 4190613356186573936 today sec:1607015429

    #det = run.Detector(args.detname)
    #det.raw._det_at_raw = det # TEMPORARY SOLUTION

    print('XXX det.calibconst.keys(): ', det.calibconst.keys()) # dict_keys(['geometry'])
    #print(det.calibconst)
    print('XXX det._det_name: ', det._det_name) # epixquad
    print('XXX det._dettype : ', det._dettype)  # epix
    print('XXX det._detid   : ', det._detid)    # -
    print('XXX det.raw._det_name: ', det.raw._det_name) # epixquad
    print('XXX det.raw._dettype : ', det.raw._dettype)  # epix
    print('XXX det.raw._calibconst.keys(): ', det.raw._calibconst.keys()) # dict_keys(['geometry'])
    print('XXX det.raw._seg_configs(): ', det.raw._seg_configs()) 
    print('XXX det.raw._uniqueid: ', det.raw._uniqueid)
    print('XXX det.raw._sorted_segment_ids: ', det.raw._sorted_segment_ids) # [0, 1, 2, 3]

    #'pixel_rms', 'pixel_status', 'pedestals', 'pixel_gain', 'geometry'

    #peds   =  det.raw._calibconst['pedestals'][0]
    #status =  det.raw._calibconst['pixel_status'][0]
    #rms    =  det.raw._calibconst['pixel_rms'][0]
    #gain   =  det.raw._calibconst['pixel_gain'][0]
    #geom   =  det.raw._calibconst['geometry'][0]

    peds   = det.raw._pedestals()
    status = det.raw._status() 
    rms    = det.raw._rms()
    gain   = det.raw._gain()

    print(info_ndarr(peds  , 'pedestals    '))
    print(info_ndarr(status, 'pixel_status '))
    print(info_ndarr(rms   , 'pixel_rms    '))
    print(info_ndarr(gain  , 'pixel_gain   '))
    #print('geometry', geom[:200], '...')

    #exit('TEST EXIT')

    print('det._configs:', det._configs)        # [<dgram.Dgram object at 0x7f7794082d40>]???? WHY IT IS A LIST? HOW TO GET LIST INDEX FOR DETECTOR?
    cfg = det._configs[0]
    print('\ndir(cfg):', dir(cfg))              # [..., '_dgrambytes', '_file_descriptor', '_offset', '_size', '_xtc', 'epixquad', 'epixquadhw', 'service', 'software', 'timestamp']

    print('\ndir(det.raw):', dir(det.raw))      # [..., '_add_fields', '_calibconst', '_common_mode_', '_configs', '_det_name', '_dettype', '_drp_class_name', '_env_store', '_info', '_return_types', '_seg_configs', '_segments', '_sorted_segment_ids', '_uniqueid', '_var_name', 'array', '_cached_pixel_coord_indexes', 'calib', '_det_calibconst', '_det_geo', '_det_geotxt_and_meta', '_geo_', 'image', '_interpol_pars_', '_pix_rc_', '_pix_xyz_', '_pixel_coord_indexes', '_pixel_coords', 'raw', 'segments']

    seg_cfgs = det.raw._seg_configs()
    print('det.raw._seg_configs():', det.raw._seg_configs())

    for i,scfg in seg_cfgs.items():
        print('\n== Segment %d'%i)
        #print('  scfg', scfg) # container.Container object
        #print('  dir(scfg.config):', dir(scfg.config)) # [..., 'asicPixelConfig', 'trbit']
        print(info_ndarr(scfg.config.asicPixelConfig, '  scfg.config.asicPixelConfig: ')) # shape:(4, 178, 192) size:136704 dtype:uint8 [12 12 12 12 12]...
        print('  scfg.config.trbit:', scfg.config.trbit) # [1 1 1 1]

    for stepnum,step in enumerate(run.steps()):
        print('%s\nStep %1d' % (50*'_',stepnum))
        for evnum,evt in enumerate(step.events()):
            if evnum>args.evtmax: exit('exit by number of events limit %d' % args.evtmax)
            if not selected_record(evnum): continue
            print('%s\nStep %1d Event %04d' % (50*'_',stepnum, evnum))
            #segs = det.raw._segments(evt)
            #raw  = det.raw.raw(evt)
            #logger.info('segs: %s' % str(segs))
            #logger.info(info_ndarr(raw,  'raw: '))
            t0_sec = time()

            calib  = det.raw.calib(evt)
            ###########################
            print('det.raw.calib(evt) time (sec) = %.6f' % (time()-t0_sec))
            logger.info(info_ndarr(det.raw._pedestals(), 'peds  '))
            logger.info(info_ndarr(det.raw.raw(evt),    'raw   '))
            logger.info(info_ndarr(calib,               'calib '))

        print(50*'-')

    print('\ndir(det): ', dir(det))
    print('\ndir(det.raw): ', dir(det.raw))


def test_image(args):

    ds, run, det = ds_run_det(args)

    flimg = None

    for stepnum,step in enumerate(run.steps()):
      print('%s\nStep %1d' % (50*'_',stepnum))

      for evnum,evt in enumerate(step.events()):
        if evnum>args.evtmax:
            exit('exit by number of events limit %d' % args.evtmax)
            #break
        if evnum>2 and evnum%500!=0: continue
        print('%s\nStep %1d Event %04d' % (50*'_',stepnum, evnum))

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
        print('image composition time = %.6f sec ' % dt_sec)

        logger.info(info_ndarr(img, 'image '))
        if img is None: continue

        alimits = (img.min(),img.max()) if args.mapmode == 4 else\
                  None if args.mapmode else\
                  (0,4)

        if args.dograph:

            if flimg is None:
                from psana.detector.UtilsGraphics import gr, fleximage
                flimg = fleximage(img, arr=arr, h_in=8, nneg=1, npos=3, alimits=alimits) #, cmap='jet')
                #flimg = fleximage(img, h_in=8, alimits=(0,4)) #, cmap='jet')    

            else:
                flimg.update(img, arr=arr)
                flimg.fig.canvas.set_window_title('Event %d' % evnum)

            gr.show(mode=1)

    print('\n  !!! TO EXIT - close graphical window - click on [x] in the window corner')

    if args.dograph:
        gr.show()
        if args.ofname is not None:
            gr.save_fig(flimg.fig, fname=args.ofname, verb=True)

    print(50*'-')


def test_mask(args):
    ds, run, det = ds_run_det(args)
    mask = det.raw._mask_from_status()
    print(info_ndarr(mask, 'mask '))

    if args.dograph:
        from psana.detector.UtilsGraphics import gr, fleximage
        evnum,evt = None, None
        for evnum,evt in enumerate(run.events()):
            if evt is None: print('Event %d is None' % evnum); continue
            print('Found non-empty event %d' % evnum); break
        if evt is None: exit('ALL events are None')

        #arr = det.raw.raw(evt)
        arr = mask + 1
        img = det.raw.image(evt, nda=arr, pix_scale_size_um=args.pscsize, mapmode=args.mapmode)
        flimg = fleximage(img, arr=arr, h_in=8, nneg=1, npos=3)#, alimits=alimits) #, cmap='jet')
        gr.show()
        if args.ofname is not None:
            gr.save_fig(flimg.fig, fname=args.ofname, verb=True)

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
      + '\n    0 - test_raw("fname=%s")'%fname0\
      + '\n    1 - test_raw("fname=%s")'%fname1\
      + '\n    4 - test_calib("fname=%s")'%fname1\
      + '\n    5 - test_image("fname=%s")'%fname1\
      + '\n    raw   - test_raw  (args)'\
      + '\n    calib - test_calib(args)'\
      + '\n    image - test_image(args)'\
      + '\n ==== '\
      + '\n    ./%s raw -e ueddaq02 -d epixquad -r66 # raw' % SCRNAME\
      + '\n    ./%s calib -e ueddaq02 -d epixquad -r66 # calib' % SCRNAME\
      + '\n    ./%s image -e ueddaq02 -d epixquad -r66 -N100000 # image' % SCRNAME\
      + '\n    ./%s mask -e ueddaq02 -d epixquad -r66 # mask' % SCRNAME\

      #+ '\n ==== '\
      #+ '\n    ./%s 2 -m0 -s101' % SCRNAME\
      #+ '\n    ./%s 2 -m1' % SCRNAME\
      #+ '\n    ./%s 2 -m2 -lDEBUG' % SCRNAME\
      #+ '\n    ./%s 2 -m3 -s101 -o img.png' % SCRNAME\
      #+ '\n    ./%s 2 -m4' % SCRNAME\
      #+ '\n    2 - does not contain config for calib....test_image("%s")'%fname0\
      #+ '\n    3 - does not contain config for calib....test_image("%s")'%fname1\

    d_fname   = fname0 if tname in ('0','2') else\
                fname1 if tname in ('1','3') else\
                fname2 if tname in ('4','5') else\
                None
    d_loglev  = 'INFO' #'INFO' #'DEBUG'
    d_pattrs  = False
    d_dograph = True
    d_detname = 'epix10k2M' if tname in ('0','1','2','3') else 'epixquad'
    d_expname = 'ueddaq02' # None #'ueddaq02' if tname=='4' else 'mfxc00318'
    d_runs    = '66' # '27,29'
    d_ofname  = None
    d_mapmode = 1
    d_pscsize = 100
    d_evtmax  = 1000

    h_loglev  = 'logging level name, one of %s, def=%s' % (STR_LEVEL_NAMES, d_loglev)
    h_mapmode = 'multi-entry pixels image mappimg mode 0/1/2/3 = statistics of entries/last pix intensity/max/mean, def=%s' % d_mapmode

    import argparse

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('tname', type=str, help='test name')
    parser.add_argument('-f', '--fname',   default=d_fname,   type=str, help='xtc file name, def=%s' % d_fname)
    parser.add_argument('-l', '--loglev',  default=d_loglev,  type=str, help=h_loglev)
    parser.add_argument('-d', '--detname', default=d_detname, type=str, help='detector name, def=%s' % d_detname)
    parser.add_argument('-e', '--expname', default=d_expname, type=str, help='experiment name, def=%s' % d_expname)
    parser.add_argument('-r', '--runs',    default=d_runs,    type=str, help='run or comma separated list of runs, def=%s' % d_runs)
    parser.add_argument('-P', '--pattrs',  default=d_pattrs,  action='store_true',  help='print objects attrubutes, def=%s' % d_pattrs)
    parser.add_argument('-G', '--dograph', default=d_dograph, action='store_false', help='plot graphics, def=%s' % d_pattrs)
    parser.add_argument('-o', '--ofname',  default=d_ofname,  type=str, help='output image file name, def=%s' % d_ofname)
    parser.add_argument('-m', '--mapmode', default=d_mapmode, type=int, help=h_mapmode)
    parser.add_argument('-N', '--evtmax',  default=d_evtmax,  type=int, help='maximal number of events, def=%s' % d_evtmax)
    parser.add_argument('-s', '--pscsize', default=d_pscsize, type=float, help='pixel scale size [um], def=%.1f' % d_pscsize)

    args = parser.parse_args()
    kwa = vars(args)
    s = '\nArguments:'
    for k,v in kwa.items(): s += '\n %8s: %s' % (k, str(v))

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.loglev])
    #logger.setLevel(intlevel)

    logger.info(s)

    tname = args.tname
    if   tname=='0': test_raw(args)
    elif tname=='1': test_raw(args)
    #elif tname=='2': test_image(args)
    #elif tname=='3': test_image(args))
    elif tname=='4':     test_calib(args)
    elif tname=='5':     test_image(args)
    elif tname=='raw':   test_raw  (args)
    elif tname=='calib': test_calib(args)
    elif tname=='image': test_image(args)
    elif tname=='mask':  test_mask(args)
    else: logger.warning('NON-IMPLEMENTED TEST: %s' % tname)

    exit('END OF %s' % SCRNAME)

# EOF

