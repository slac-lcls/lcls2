#!/usr/bin/env python
"""
"""
import sys
import logging
logger = logging.getLogger(__name__)

from time import time
t0_sec = time()
import numpy as np
print('np import consumed time (sec) = %.6f' % (time()-t0_sec))

from psana.pyalgos.generic.NDArrUtils import info_ndarr, divide_protected
from psana import DataSource
from psana.detector.UtilsMask import CC, DTYPE_MASK

SCRNAME = sys.argv[0].rsplit('/')[-1]

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
      logger.debug('det.calibconst is None')
      return None, None


def datasource_run(**kwa):
    ds = DataSource(**kwa)
    return ds, next(ds.runs())


def datasource_run_det(**kwa):
    ds = DataSource(**kwa)
    print('\n  dir(ds):', dir(ds))

    run = next(ds.runs())
    print('\n  dir(run):', dir(run))

    det = run.Detector(kwa.get('detname','opal'))
    print('\n  dir(det):', dir(det))

    return ds, run, det


def ds_run_det(args):

    logger.info('ds_run_det input file:\n  %s' % args.fname)

    kwa = {'files':args.fname,} if args.fname is not None else\
          {'exp':args.expname,'run':int(args.runs.split(',')[0])}
    if args.dirxtc is not None: kwa['dir'] = args.dirxtc

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
        if evnum>args.events: sys.exit('exit by number of events limit %d' % args.events)
        if not selected_record(evnum): continue
        print('%s\nEvent %04d' % (50*'_',evnum))
        segs = det.raw._segment_numbers(evt)
        raw  = det.raw.raw(evt)
        logger.info(info_ndarr(segs, 'segs '))
        logger.info(info_ndarr(raw,  'raw  '))
    print(50*'-')


def test_calib(args):
    from time import time

    ds, run, det = ds_run_det(args)

    t_sec = int(time())
    print('XXX tnow time %d sec as tstamp: %d' % (t_sec,t_sec<<32))
    print('XXX det.calibconst.keys(): ', det.calibconst.keys()) # dict_keys(['geometry'])
    print('XXX det._det_name: ', det._det_name) # epixquad
    print('XXX det._dettype : ', det._dettype)  # epix
    print('XXX det._detid   : ', det._detid)    # -
    print('XXX det.raw._det_name: ', det.raw._det_name) # epixquad
    print('XXX det.raw._dettype : ', det.raw._dettype)  # epix
    print('XXX det.raw._calibconst.keys(): ', det.raw._calibconst.keys()) # dict_keys(['geometry'])
    print('XXX det.raw._seg_configs(): ', det.raw._seg_configs())
    print('XXX det.raw._uniqueid: ', det.raw._uniqueid)
    print('XXX det.raw._sorted_segment_ids: ', det.raw._sorted_segment_ids) # [0, 1, 2, 3]

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

    print('det._configs:', det._configs)
    cfg = det._configs[0]
    print('\ndir(cfg):', dir(cfg))

    print('\ndir(det.raw):', dir(det.raw))

    seg_cfgs = det.raw._seg_configs()
    print('det.raw._seg_configs():', det.raw._seg_configs())

    for i,scfg in seg_cfgs.items():
        print('\n== Segment %d'%i)
        #print('  scfg', scfg) # container.Container object
        #print('  dir(scfg.config):', dir(scfg.config)) # [..., 'asicPixelConfig', 'trbit']
        print(info_ndarr(scfg.config.asicPixelConfig, '  scfg.config.asicPixelConfig: ')) # shape:(4, 178, 192) dtype:uint8
        print('  scfg.config.trbit:', scfg.config.trbit) # [1 1 1 1]

    for stepnum,step in enumerate(run.steps()):
        print('%s\nStep %1d' % (50*'_',stepnum))
        for evnum,evt in enumerate(step.events()):
            if evnum>args.events: sys.exit('exit by number of events limit %d' % args.events)
            if not selected_record(evnum): continue
            print('%s\nStep %1d Event %04d' % (50*'_',stepnum, evnum))
            #segs = det.raw._segments(evt)
            #raw  = det.raw.raw(evt)
            #logger.info('segs: %s' % str(segs))
            #logger.info(info_ndarr(raw,  'raw: '))
            t0_sec = time()

            calib  = det.raw.calib(evt)

            print('det.raw.calib(evt) time (sec) = %.6f' % (time()-t0_sec))
            logger.info(info_ndarr(det.raw._pedestals(), 'peds  '))
            logger.info(info_ndarr(det.raw.raw(evt),     'raw   '))
            logger.info(info_ndarr(calib,                'calib '))

        print(50*'-')

    print('\ndir(det): ', dir(det))
    print('\ndir(det.raw): ', dir(det.raw))


def test_image(args):

    import psana.detector.UtilsEpix10ka as ue
    from psana.detector.UtilsGraphics import gr, fleximage, flexhist, fleximagespec

    dograph = args.dograph.lower()
    flimg, flspe, flims = None, None, None

    ds, run, det = ds_run_det(args)
    peds = det.raw._pedestals() if args.grindex is None else det.raw._pedestals()[args.grindex,:]

    is_epix10ka = False if det is None else det.raw._dettype == 'epix10ka'
    is_epixhr2x2 = False if det is None else det.raw._dettype == 'epixhr2x2'
    dcfg = ue.config_object_det(det) if is_epix10ka else None

    geo = det.raw._det_geo()
    pscsize = geo.get_pixel_scale_size()
    #print('pscsize', pscsize)

    break_event_loop = False

    nframes = 0
    sum_arr, sum_sta = None, None
    med_vs_evt = np.zeros(args.events-args.evskip+10, dtype=np.float64)
    nrec_med = 0

    MDBITS = det.raw._data_bit_mask # 0x7fff  # 32767

    for stepnum,step in enumerate(run.steps()):
      print('%s\nStep %1d' % (50*'_',stepnum))

      if args.stepnum is not None and stepnum != args.stepnum:
          print('  skip - step selected in option -M is %1d' % (args.stepnum))
          continue
      print('%s\n  begin event loop' % (50*'_'))
      for evnum,evt in enumerate(step.events()):
        if evnum<args.evskip:
            print('Step %1d Event %04d - skip first %04d events' % (stepnum, evnum, args.evskip),\
                   end=('\r' if evnum<args.evskip-1 else '\n'))
            continue

        if evnum>args.events:
            print('break by number of events limit %d set in option -N' % args.events)
            break_event_loop = True
            break
        if evnum>2 and evnum%args.evjump!=0: continue
        print('%s\nStep %1d Event %04d' % (50*'_',stepnum, evnum))

        if dcfg is not None:
            s = '    gain mode fractions for: FH       FM       FL'\
                '       AHL-H    AML-M    AHL-L    AML-L\n%s' % (29*' ')
            #ue.info_pixel_gain_mode_for_fractions(dcfg, data=det.raw.raw(evt), msg=s))
            gmfracs = ue.pixel_gain_mode_fractions(det.raw, evt)
            print(ue.info_pixel_gain_mode_for_fractions(gmfracs, msg=s))
            gmind = ue.gain_mode_index_from_fractions(gmfracs)
            gmname = ue.gain_mode_name_for_index(gmind).upper()
            print('  == major gain mode %d : %s' % (gmind, gmname))
            #print('  == gain mode: %s' % ue.find_gain_mode(det.raw, evt).upper())

            if peds is None: peds = det.raw._pedestals()[gmind,:]

        #user_mask = np.ones_like(det.raw.raw(evt), dtype=DTYPE_MASK) #np.uint8
        #user_mask[0,100:150,200:250] = 0
        user_mask = None

        arr = None
        if args.show == 'raw-peds-med':
           arr = (det.raw.raw(evt) & MDBITS) - peds
           med = np.median(arr)
           print('XXX from raw-peds subtract its median = %.3f' % med)
           arr -= med

        if arr is None:
           arr = det.raw.calib(evt, cmpars=(7,7,100,10),\
                            mbits=0o7, mask=user_mask, edge_rows=10, edge_cols=10, center_rows=5, center_cols=5)\
                                                 if args.show == 'calibcm'  else\
              det.raw.calib(evt, cmpars=(8,7,10,10))\
                                                 if args.show == 'calibcm8' else\
              det.raw.calib(evt)                 if args.show == 'calib'    else\
              peds                               if args.show == 'peds'     else\
              det.raw._gain_range_index(evt)     if args.show == 'grind'    else\
              (det.raw.raw(evt) & MDBITS) - peds if args.show == 'raw-peds' else\
              (det.raw.raw(evt) & MDBITS)        if args.show == 'rawbm'    else\
               np.ones_like(det.raw.raw(evt))    if args.show == 'ones'     else\
              (det.raw.raw(evt) & args.bitmask)

        #if args.show == 'calibcm': arr += 1 # to see panel edges

        logger.info(info_ndarr(arr, 'arr '))
        if arr is None: continue

        med = np.median(arr)
        med_vs_evt[nrec_med] = med; nrec_med+=1

        if args.cumulat:
          if (med > args.thrmin) and (med < args.thrmax):
            nframes +=1
            cond = arr > args.thrpix
            if nframes != 1:
                _ = np.add(sum_arr[cond], arr[cond], out=sum_arr[cond])
                sum_sta[cond] += 1
            else:
                sum_arr = np.array(arr,dtype=np.float64)
                sum_sta = np.zeros_like(arr,dtype=np.uint64)

            #if nframes > 1: arr = sum_arr/float(nframes)
            if nframes > 1: arr = divide_protected(sum_arr, sum_sta)
            print('Step %1d event:%04d nframes:%04d arr median:%.3f' % (stepnum, evnum, nframes, med))
          else:
            continue

        t0_sec = time()

        img = det.raw.image(evt, nda=arr, pix_scale_size_um=pscsize, mapmode=args.mapmode)
        print('image composition time = %.6f sec ' % (time()-t0_sec))

        logger.info(info_ndarr(img, 'img '))
        logger.info(info_ndarr(arr, 'arr '))
        if img is None: continue

        title = '%s %s run:%s step:%d ev:%d' % (args.detname, args.expname, args.runs, stepnum, evnum)

        if 'i' in dograph:
            if flimg is None:
                flimg = fleximage(img, arr=arr, fraclo=0.05, frachi=0.95)
                flimg.move(10,20)
            else:
                flimg.update(img, arr=arr)
                flimg.fig.canvas.set_window_title(title)
                flimg.axtitle(title)

        if 'h' in dograph:
            if flspe is None:
                flspe = flexhist(arr, bins=50, color='green', fraclo=0.001, frachi=0.999)
                flspe.move(800,20)
            else:
                flspe.update(arr, bins=50, color='green', fraclo=0.001, frachi=0.999)
                flspe.fig.canvas.set_window_title(title)
                flspe.axtitle(title)

        if 'c' in dograph:
            if flims is None:
                flims = fleximagespec(img, arr=arr, bins=100, color='lightgreen',\
                                      amin=args.gramin,   amax=args.gramax,\
                                      nneg=args.grnneg,   npos=args.grnpos,\
                                      fraclo=args.grfrlo, frachi=args.grfrhi,\
                                      w_in=args.figwid, h_in=args.fighig,\
                )
                flims.move(10,20)
            else:
                #print(info_ndarr(arr, 'YYY before update arr: ', last=5))
                flims.update(img, arr=arr)
                flims.axtitle(title)

            gr.show(mode=1)

      if break_event_loop: break

    med_vs_evt = med_vs_evt[:nrec_med]
    med = np.median(med_vs_evt)
    q05 = np.quantile(med_vs_evt, 0.05, interpolation='linear')
    q95 = np.quantile(med_vs_evt, 0.95, interpolation='linear')

    print(info_ndarr(med_vs_evt, 'per event median  ', last=nrec_med-1))
    print('  median over %d event-records: %.3f' % (nrec_med, med))
    print('  quantile(med_vs_evt, 0.05): %.3f' % q05)
    print('  quantile(med_vs_evt, 0.95): %.3f' % q95)

    if args.dograph:
        print('\n  !!! TO EXIT - close graphical window(s) - click on [x] in the window corner')
        gr.show()
        if args.ofname is not None:
            if 'i' in dograph: gr.save_fig(flimg.fig, fname=args.ofname+'-img', verb=True)
            if 'h' in dograph: gr.save_fig(flspe.fig, fname=args.ofname+'-spe', verb=True)
            if 'c' in dograph: gr.save_fig(flims.fig, fname=args.ofname+'-imgspe', verb=True)

    print(50*'-')


def test_single_image(args):
    ds, run, det = ds_run_det(args)
    #mask = det.raw._mask_from_status()
    #raw = det.raw.raw()
    #mask = det.raw._mask_calib()
    #mask_edges = det.raw._mask_edges(mask, edge_rows=20, edge_cols=10)
    #mask = det.raw._mask_edges(edge_rows=20, edge_cols=10, center_rows=4, center_cols=2)
    #mask = det.raw._mask(calib=True, status=True, edges=True,\
    #                     edge_rows=20, edge_cols=10, center_rows=4, center_cols=2)

    tname = args.tname
    grindex = 0 if args.grindex is None else args.grindex

    arr, amin, amax, title = None, None, None, '%s for --grindex=%d ' % (tname, grindex)
    if tname == 'mask':

        umask = np.ones((4, 352, 384), dtype=np.uint8)
        umask[3, 100:120, 160:200] = 0
        arr = 1 + det.raw._mask(status=True, status_bits=0xffff, gain_range_inds=(0,1,2,3,4),\
                                neighbors=True, rad=5, ptrn='r',\
                                edges=True, edge_rows=10, edge_cols=5,\
                                center=True, center_rows=5, center_cols=3,\
                                calib=False,\
                                umask=umask,\
                                force_update=False)
        amin, amax = 0, 2
        title = 'mask+1'
    elif tname=='peds':    arr = det.raw._pedestals()[grindex,:]
    elif tname=='status':  arr = det.raw._status()[grindex,:]
    elif tname=='rms':     arr = det.raw._rms()[grindex,:]
    elif tname=='gain':    arr = det.raw._gain()[grindex,:]
    elif tname=='xcoords': arr = det.raw._pixel_coords(do_tilt=True, cframe=args.cframe)[0]
    elif tname=='ycoords': arr = det.raw._pixel_coords(do_tilt=True, cframe=args.cframe)[1]
    elif tname=='zcoords': arr = det.raw._pixel_coords(do_tilt=True, cframe=args.cframe)[2]
    else: arr = det.raw._mask_calib()

    geo = det.raw._det_geo()
    pscsize = geo.get_pixel_scale_size()

    print(info_ndarr(arr, 'array for %s' % tname))

    if args.dograph:
        from psana.detector.UtilsGraphics import gr, fleximage
        evnum,evt = None, None
        for evnum,evt in enumerate(run.events()):
            if evt is None: print('Event %d is None' % evnum); continue
            print('Found non-empty event %d' % evnum); break
        if evt is None: sys.exit('ALL events are None')

        img = det.raw.image(evt, nda=arr, pix_scale_size_um=pscsize, mapmode=args.mapmode)
        print(info_ndarr(img, 'image for %s' % tname))

        flimg = fleximage(img, arr=arr, h_in=args.fighig, w_in=args.figwid, amin=amin, amax=amax)#, cmap='jet')
        flimg.fig.canvas.set_window_title(title)
        flimg.move(10,20)
        gr.show()
        if args.ofname is not None:
            gr.save_fig(flimg.fig, fname=args.ofname, verb=True)

def do_main():

    DICT_NAME_TO_LEVEL = logging._nameToLevel # {'INFO': 20, 'WARNING': 30, 'WARN': 30,...
    LEVEL_NAMES = [k for k in DICT_NAME_TO_LEVEL.keys() if isinstance(k,str)]
    STR_LEVEL_NAMES = ', '.join(LEVEL_NAMES)

    tname = sys.argv[1] if len(sys.argv)>1 else '100'
    usage =\
        '\n  %s [test-name] [optional-arguments]' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    raw   - test_raw W/O GRAPHICS'\
      + '\n    calib - test_calib W/O GRAPHICS'\
      + '\n    [image] - test_image WITH GRAPHICS - optional default'\
      + '\n    mask, peds, rms, status, gain, z/y/xcoords - test_single_image WITH GRAPHICS'\
      + '\n ==== '\
      + '\n    %s raw -e ueddaq02 -d epixquad -r66 # raw' % SCRNAME\
      + '\n    %s calib -e ueddaq02 -d epixquad -r66 # calib' % SCRNAME\
      + '\n    %s mask  -e ueddaq02 -d epixquad -r66 # mask' % SCRNAME\
      + '\n    %s -e ueddaq02 -d epixquad -r66 -N1000 # image' % SCRNAME\
      + '\n    %s -e ueddaq02 -d epixquad -r108 -N1 -S grind' % SCRNAME\
      + '\n    %s -e ueddaq02 -d epixquad -r140 -N100 -M2 -S calibcm8' % SCRNAME\
      + '\n    %s -e ueddaq02 -d epixquad -r140 -N100 -M2 -S calibcm8 -o img-ueddaq02-epixquad-r140-ev0002-cm8-7-100-10.png -N3' % SCRNAME\
      + '\n    %s -e ueddaq02 -d epixquad -r211 -N1 -M0 -Speds -g0 # - plot pedestals for gain group 0/FH' % SCRNAME\
      + '\n    %s -e ueddaq02 -d epixquad -r211 -N100 -Sraw-peds -M2 -g2 # - plot calib[step=2] - pedestals[gain group 2]' % SCRNAME\
      + '\n    %s -e rixx45619 -d epixhr -r118 --gramin 1 --gramax 32000 -Sraw' % SCRNAME\
      + '\n    %s -e rixx45619 -d epixhr -r118 --gramin 1 --gramax 32000 -Speds -g0' % SCRNAME\
      + '\n    %s -e rixx45619 -d epixhr -r118 --gramin -100 --gramax 100 -Sraw-peds -g0' % SCRNAME\
      + '\n    %s -e rixx45619 -d epixhr -r119 -Scalib' % SCRNAME\
      + '\n    %s -e rixx45619 -d epixhr -r119 -Sones' % SCRNAME\
      + '\n    %s -e rixx45619 -d epixhr -N10000 -J200 --gramin 0 --gramax 10 -Sgrind' % SCRNAME\
      + '\n    %s mask -e rixx45619 -d epixhr -r119' % SCRNAME\
      + '\n    %s peds -e rixx45619 -d epixhr -r119 -g1' % SCRNAME\
      + '\n    %s gains -e rixx45619 -d epixhr -r119 -g1' % SCRNAME\
      + '\n    %s xcoords -e rixx45619 -d epixhr -r119 --cframe=1' % SCRNAME\

    d_loglev  = 'INFO'
    d_fname   = None
    d_pattrs  = False
    d_dograph = 'c' # 'ihc'
    d_cumulat = False
    d_show    = 'calibcm'
    d_detname = 'epixquad'
    d_expname = 'ueddaq02'
    d_runs    = '1'
    d_dirxtc  = None # '/cds/data/psdm/ued/ueddaq02/xtc'
    d_ofname  = None
    d_mapmode = 1
    d_events  = 1000
    d_evskip  = 0
    d_evjump  = 100
    d_stepnum = None
    d_bitmask = 0xffff
    d_grindex = None
    d_thrmin  = -0.344 # -0.598 as in dark 211, -0.344 r134
    d_thrmax  = 0.582 # 0.582 r134
    d_thrpix  = -10000
    d_gramin  = None
    d_gramax  = None
    d_grnneg  = None
    d_grnpos  = None
    d_grfrlo  = 0.01
    d_grfrhi  = 0.99
    d_cframe  = 0
    d_figwid  = 14
    d_fighig  = 12

    h_loglev  = 'logging level name, one of %s, def=%s' % (STR_LEVEL_NAMES, d_loglev)
    h_mapmode = 'multi-entry pixels image mappimg mode 0/1/2/3 = statistics of entries/last pix intensity/max/mean, def=%s' % d_mapmode
    h_show = 'select image correction from raw/calib/calibcm/calibcm8/grind/rawbm/raw-peds/raw-peds-med/peds/ones, def=%s' % d_show
    h_dirxtc  = 'non-default xtc directory, default = %s' % d_dirxtc
    import argparse

    parser = argparse.ArgumentParser(usage=usage, description='%s - test epix10ka data'%SCRNAME)
    #parser.add_argument('posargs', nargs='*', type=str, help='test name and other positional arguments')
    parser.add_argument('tname', nargs='?', type=str, help='test name and other positional arguments')
    parser.add_argument('-f', '--fname',   default=d_fname,   type=str, help='xtc file name, def=%s' % d_fname)
    parser.add_argument('-l', '--loglev',  default=d_loglev,  type=str, help=h_loglev)
    parser.add_argument('-d', '--detname', default=d_detname, type=str, help='detector name, def=%s' % d_detname)
    parser.add_argument('-e', '--expname', default=d_expname, type=str, help='experiment name, def=%s' % d_expname)
    parser.add_argument('-r', '--runs',    default=d_runs,    type=str, help='run or comma separated list of runs, def=%s' % d_runs)
    parser.add_argument('-x', '--dirxtc',  default=d_dirxtc,  type=str, help=h_dirxtc)
    parser.add_argument('-P', '--pattrs',  default=d_pattrs,  action='store_true',  help='print objects attrubutes, def=%s' % d_pattrs)
    parser.add_argument('-G', '--dograph', default=d_dograph, type=str, help='plot i/h/c=image/hist/comb, def=%s' % d_dograph)
    parser.add_argument('-C', '--cumulat', default=d_cumulat, action='store_true', help='plot cumulative image, def=%s' % d_cumulat)
    parser.add_argument('-S', '--show',    default=d_show,    type=str, help=h_show)
    parser.add_argument('-o', '--ofname',  default=d_ofname,  type=str, help='output image file name, def=%s' % d_ofname)
    parser.add_argument('-m', '--mapmode', default=d_mapmode, type=int, help=h_mapmode)
    parser.add_argument('-N', '--events',  default=d_events,  type=int, help='maximal number of events, def=%s' % d_events)
    parser.add_argument('-K', '--evskip',  default=d_evskip,  type=int, help='number of events to skip in the beginning of run, def=%s' % d_evskip)
    parser.add_argument('-J', '--evjump',  default=d_evjump,  type=int, help='number of events to jump, def=%s' % d_evjump)
    parser.add_argument('-B', '--bitmask', default=d_bitmask, type=int,   help='bitmask for raw MDBITS=16383/0x7fff=32767, def=%s' % hex(d_bitmask))
    parser.add_argument('-M', '--stepnum', default=d_stepnum, type=int,   help='step selected to show or None for all, def=%s' % d_stepnum)
    parser.add_argument('-g', '--grindex', default=d_grindex, type=int,   help='gain range index [0,6] for peds, def=%s' % str(d_grindex))
    parser.add_argument('-t', '--thrmin',  default=d_thrmin,  type=float, help='minimal threshold on median to accumulate events with -C, def=%f' % d_thrmin)
    parser.add_argument('-T', '--thrmax',  default=d_thrmax,  type=float, help='maximal threshold on median to accumulate events with -C, def=%f' % d_thrmax)
    parser.add_argument('--thrpix',        default=d_thrpix,  type=float, help='per pixel intensity threshold to accumulate events with -C, def=%f' % d_thrpix)
    parser.add_argument('--gramin',        default=d_gramin,  type=float, help='minimal intensity for grphics, def=%s' % str(d_gramin))
    parser.add_argument('--gramax',        default=d_gramax,  type=float, help='maximal intensity for grphics, def=%s' % str(d_gramax))
    parser.add_argument('--grnneg',        default=d_grnneg,  type=float, help='number of mean deviations of intensity negative limit for grphics, def=%s' % str(d_grnneg))
    parser.add_argument('--grnpos',        default=d_grnpos,  type=float, help='number of mean deviations of intensity negative limit for grphics, def=%s' % str(d_grnpos))
    parser.add_argument('--grfrlo',        default=d_grfrlo,  type=float, help='fraqction of statistics [0,1] below low  limit for grphics, def=%s' % str(d_grfrlo))
    parser.add_argument('--grfrhi',        default=d_grfrhi,  type=float, help='fraqction of statistics [0,1] below high limit for grphics, def=%s' % str(d_grfrhi))
    parser.add_argument('--cframe',        default=d_cframe,  type=int, help='coordinate frame for images 0/1 for psana/LAB, def=%s' % str(d_cframe))
    parser.add_argument('--figwid',        default=d_figwid,  type=float, help='figure width, inch, def=%f' % d_figwid)
    parser.add_argument('--fighig',        default=d_fighig,  type=float, help='figure width, inch, def=%f' % d_fighig)

    args = parser.parse_args()
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(name)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.loglev])
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger.debug('parser.parse_args: %s' % str(args))

    kwa = vars(args)
    s = '\nArguments:'
    for k,v in kwa.items(): s += '\n %8s: %s' % (k, str(v))

    logger.info(s)

    if len(sys.argv)<2:
        print(usage)
        sys.exit('MISSING ARGUMENTS - EXIT')

    tname = args.tname
    if tname is None: tname = 'image'
    print('tname:', tname)

    if   tname=='raw':   test_raw  (args)
    elif tname=='calib': test_calib(args)
    elif tname=='image': test_image(args)
    elif tname in ('mask', 'peds', 'status', 'rms', 'gain', 'xcoords', 'ycoords', 'zcoords'): test_single_image(args)
    else: logger.warning('NON-IMPLEMENTED TEST: %s' % tname)


if __name__ == "__main__":
    do_main()
    sys.exit('END OF %s' % SCRNAME)

# EOF

