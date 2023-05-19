#!/usr/bin/env python
"""./lcls2/psana/psana/detector/test_issues_2023.py 1
"""

import sys
import logging
SCRNAME = sys.argv[0].rsplit('/')[-1]
STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)


def ds_run_det(exp='ascdaq18', run=171, detname='epixhr', **kwa):
    from psana import DataSource
    ds = DataSource(exp=exp, run=run, **kwa)
    orun = next(ds.runs())
    det = orun.Detector(detname)
    return ds, orun, det


def issue_2023_01_03():
    """epixhr calib method with common mode correction using standard detector interface
       datinfo -k exp=rixx45619,run=119 -d epixhr
    """
    import psana.pyalgos.generic.PSUtils as psu
    from psana.detector.NDArrUtils import info_ndarr
    from time import time

    #ds, orun, det = ds_run_det(exp='rixx45619',run=121, detname='epixhr', dir='/cds/data/psdm/prj/public01/xtc')
    ds, orun, det = ds_run_det(exp='ueddaq02',run=569, detname='epixquad', dir='/cds/data/psdm/prj/public01/xtc')

    print('common mode parameters from DB', det.raw._common_mode())

    from psana.detector.UtilsGraphics import gr, fleximage, arr_median_limits
    flimg = None
    for nevt,evt in enumerate(orun.events()):
        print('== Event %03d ==' % nevt)
        t0_sec_tot = time()
        arr = det.raw.calib(evt, cmpars=(0,7,300,10))
        logger.info('calib consumed time = %.6f sec' % (time()-t0_sec_tot))

        #arr = det.raw.calib(evt, cmpars=0)
        #arr = det.raw.calib(evt)
        #arr = det.raw.raw(evt)
        if nevt>29: break
        if arr is None: continue

        #print(info_ndarr(arr,'arr:'))
        #sh = img.shape # shape:(1, 288, 384)
        #img = arr[0,144:,:192] # cut off a single ASIC with meaningfull data
        img = psu.table_nxn_epix10ka_from_ndarr(arr)
        print(info_ndarr(img,'img:'))

        if flimg is None:
           flimg = fleximage(img, arr=None, h_in=8, w_in=11, nneg=1, npos=3)
        gr.set_win_title(flimg.fig, titwin='Event %d' % nevt)
        flimg.update(img, arr=None)
        gr.show(mode='DO NOT HOLD')
    gr.show()


def issue_2023_01_06():
    """test utils_calib_components.py with direct acces to _calibconst and _config_object in the calib method
    """
    import psana.detector.utils_calib_components as ucc
    from psana.detector.NDArrUtils import info_ndarr
    from time import time
    import numpy as np

    # ds, orun, det = ds_run_det(exp='rixx45619', run=121, detname='epixhr', dir='/cds/data/psdm/prj/public01/xtc')
    ds, orun, det = ds_run_det(exp='ueddaq02', run=569, detname='epixquad', dir='/cds/data/psdm/prj/public01/xtc')

    config = det.raw._config_object()
    calibc = det.raw._calibconst

    logger.debug('calibc: %s' % str(calibc))

    cc = ucc.calib_components_epix(calibc, config)

    print('calib_types: ', cc.calib_types())
    print('config - number of panels: ', cc.number_of_panels())
    print('dettype: ', cc.dettype())

    from psana.detector.UtilsGraphics import gr, fleximage, arr_median_limits
    flimg = None
    for nevt,evt in enumerate(orun.events()):
        print('== Event %03d ==' % nevt)
        t0_sec_tot = time()
        raw = det.raw.raw(evt) # , cmpars=(0,7,300,10))
        pedest  = cc.event_pedestals(raw)

        #arr = np.array(raw & cc.data_bit_mask(), dtype=np.float32) - pedest
        calib = cc.calib(raw, cmpars=(0,7,300,10))  # **kwa
        arr = cc.common_mode_correction(raw, cmpars=(0,7,300,10))  # **kwa
        print(info_ndarr(calib,'calib:'))
        print(info_ndarr(arr,'cmcorr:'))

        logger.info('time consumption to make 3-d array for imaging = %.6f sec' % (time()-t0_sec_tot))

        if nevt>29: break
        if arr is None: continue

        #print(info_ndarr(arr,'arr:'))
        #sh = img.shape # shape:(1, 288, 384)
        #img = arr[0,144:,:192] # cut off a single ASIC with meaningfull data
        img = ucc.psu.table_nxn_epix10ka_from_ndarr(arr, gapv=0)
        print(info_ndarr(img,'img:'))

        if flimg is None:
           flimg = fleximage(img, arr=None, h_in=8, w_in=11, nneg=1, npos=3)
        gr.set_win_title(flimg.fig, titwin='Event %d' % nevt)
        flimg.update(img, arr=None)
        gr.show(mode='DO NOT HOLD')
    gr.show()


def issue_2023_01_10():
    """test for of the 1st charge injection for epixhr
       datinfo -k exp=ascdaq18,run=170 -d epixhr  # 'scantype': 'pedestal'
       datinfo -k exp=ascdaq18,run=171 -d epixhr  # 'scantype': 'chargeinj'
    """
    import psana.detector.utils_calib_components as ucc
    import psana.detector.UtilsEpix10kaChargeInjection as ueci

    from psana.detector.NDArrUtils import info_ndarr
    from time import time
    import numpy as np
    # dir='/cds/data/psdm/asc/ascdaq18/xtc/' # default
    # dir='/cds/data/psdm/prj/public01/xtc') # preserved
    ds, orun, det = ds_run_det(exp='ascdaq18', run=171, detname='epixhr', dir='/cds/data/psdm/asc/ascdaq18/xtc/')

    config = det.raw._config_object()
    calibc = det.raw._calibconst

    logger.debug('calibc: %s' % str(calibc))

    cc = ucc.calib_components_epix(calibc, config)
    data_bit_mask = cc.data_bit_mask() # 0o77777 for epixhr
    pedestals = cc.pedestals()

    ones = np.ones_like(pedestals, dtype=np.float32)

    print('calib_types: ', cc.calib_types())
    print('config - number of panels: ', cc.number_of_panels())
    print('dettype: ', cc.dettype())
    print('calib_metadata: ', cc.calib_metadata(ctype='pedestals'))
    print(info_ndarr(pedestals,'pedestals:'))
    print('data_bit_mask:', oct(data_bit_mask))

    #sys.exit('TEST EXIT')

    from psana.detector.UtilsGraphics import gr, fleximage, arr_median_limits
    flimg = None

    nstep_sel = 2
    space = 5
    databitw = 0o037777

    for nstep, step in enumerate(orun.steps()):
      #if nstep<nstep_sel: continue
      #elif nstep>nstep_sel: break
      if nstep>10: break

      irow, icol = ueci.injection_row_col(nstep, space)

      s = '== Step %02d irow %03d icol %03d ==' % (nstep, irow, icol)
      print(s)


      for nevt,evt in enumerate(step.events()):
        #if nevt>1000: break
        if nevt%100: continue

        #print('== Step %02d Event %03d irow %03d icol %03d ==' % (nstep, nevt, irow, icol))

        #t0_sec_tot = time()
        raw = det.raw.raw(evt)
        if raw is None: continue

        peds = cc.event_pedestals(raw)
        #arr = peds
        arr = np.array(raw & data_bit_mask, dtype=np.float32) - peds

        #gmaps = cc.gain_maps_epix(raw)
        #arr = ucc.event_constants_for_gmaps(gmaps, ones, default=0)
        #arr = ucc.map_gain_range_index_for_gmaps(gmaps, default=10) # stack bits...
        #arr = np.array(raw & 0o100000, dtype=np.int) # 0o77777 # behaves ok
        arr1 = np.array(arr[0,irow,100:120], dtype=np.int16) & databitw
        print(info_ndarr(arr1,'%s  arr1:' % s, first=0, last=10), '  v[col]=%5d' % arr1[icol])

        #logger.info('time consumption to make 3-d array for imaging = %.6f sec' % (time()-t0_sec_tot))
        #pedestals: shape:(7, 1, 288, 384)
        #img = cc.pedestals()[1,0,:150,:200]
        #img = arr[0,:150,:200] # cut off a single ASIC with meaningfull data
        img = arr[0,:144,:192] # cut off a single ASIC with meaningfull data
        #img = arr[0,60:144,110:192] # cut off a single ASIC with meaningfull data
        #img = arr[0,0:20,100:120] # cut off a single ASIC with meaningfull data
        #img = arr[0,:,:] # cut off a single ASIC with meaningfull data
        #img = ucc.psu.table_nxn_epix10ka_from_ndarr(arr, gapv=0)
        #print(info_ndarr(img,'img:'))

        if flimg is None:
           flimg = fleximage(img, arr=None, h_in=8, w_in=11, nneg=1, npos=3)
        gr.set_win_title(flimg.fig, titwin='Step %02d Event %d' % (nstep,nevt))
        flimg.update(img, arr=None, amin=0, amax=databitw)
        gr.show(mode='DO NOT HOLD')
    gr.show()


def issue_2023_02_07():
    #ds, orun, det = ds_run_det(exp='ascdaq18', run=171, detname='epixhr', dir='/cds/data/psdm/asc/ascdaq18/xtc/')
    from psana import DataSource
    for runnum in (170, 171):
      ds = DataSource(exp='ascdaq18', run=runnum)  # , dir='/cds/data/psdm/asc/ascdaq18/xtc/')
      orun = next(ds.runs())
      print('runnum: %d timestamp: %d' % (orun.runnum, orun.timestamp))


def issue_2023_04_27():
    from psana.detector.NDArrUtils import print_ndarr, info_ndarr
    #from psana import DataSource
    #ds, detname = DataSource(exp='tstx00417', run=277, dir='/cds/data/drpsrcf/tst/tstx00417/xtc'), 'epixhr_emu'
    ##ds, detname = DataSource(exp='ascdaq18', run=171, dir='/cds/data/psdm/asc/ascdaq18/xtc/'), 'epixhr'
    #orun = next(ds.runs())
    #odet = orun.Detector(detname)

    #ds, orun, odet = ds_run_det(exp='tstx00417',run=277, detname='epixhr_emu', dir='/cds/data/drpsrcf/tst/tstx00417/xtc')
    ds, orun, odet = ds_run_det(exp='ascdaq18',run=171, detname='epixhr', dir='/cds/data/psdm/asc/ascdaq18/xtc/')

    print('run.runnum: %d detnames: %s expt: %s' % (orun.runnum, str(orun.detnames), orun.expt))
    print('odet.raw._det_name: %s' % odet.raw._det_name) # epixquad
    print('odet.raw._dettype : %s' % odet.raw._dettype)  # epix

    for nstep, step in enumerate(orun.steps()):
        print('\n==== step:%02d' %nstep)
        if nstep>4: break
        for k,v in odet.raw._seg_configs().items():
            cob = v.config
            print_ndarr(cob.asicPixelConfig, 'seg:%02d trbits: %s asicPixelConfig:'%(k, str(cob.trbit)))

        for nev, evt in enumerate(step.events()):
            if nev>5: break
            raw = odet.raw.raw(evt)
            print(info_ndarr(raw, 'Event:%02d raw:'%nev))


def issue_2023_04_28():
    """epixhremu default geometry implementation
    """
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess
    from psana.detector.NDArrUtils import info_ndarr
    ds, orun, det = ds_run_det(exp='tstx00417',run=277, detname='epixhr_emu', dir='/cds/data/drpsrcf/tst/tstx00417/xtc')
    #ds, orun, det = ds_run_det(exp='ascdaq18',run=171, detname='epixhr', dir='/cds/data/psdm/asc/ascdaq18/xtc/')

    x,y,z = det.raw._pixel_coords()
    print(info_ndarr(x,'det.raw.pixel_coords x:'))

#    for nevt,evt in enumerate(orun.events()):
#        geotxt = det.raw._det_geotxt_default()
#        print('_det_geotxt_default:\n%s' % geotxt)
#        o = GeometryAccess()
#        o.load_pars_from_str(geotxt)
#        x,y,z = o.get_pixel_coords()
#        print(info_ndarr(x,'x:'))
#        if det.raw.image(evt) is not None: break

def issue_2023_05_19():
    """
    """
    from psana import DataSource
    ds = DataSource(exp='tmoc00118', run=222, dir='/cds/data/psdm/prj/public01/xtc')
    for run in ds.runs():
      opal = run.Detector('tmo_opal1')
      for nev,evt in enumerate(run.events()):
        if nev>5: break
        img = opal.raw.image(evt)
        print('ev:%04d  evt.timestamp: %d image.shape: %s' % (nev, evt.timestamp, str(img.shape)))


def issue_2023_mm_dd():
    print('template')

USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel-e.g.-DEBUG-or-INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - issue_2023_01_03 - test epixhr, calib and common mode correction'\
      + '\n    2 - issue_2023_01_06 - test utils_calib_components.py'\
      + '\n    3 - issue_2023_01_10 - test for of the 1st charge injection for epixhr'\
      + '\n    4 - issue_2023_02_07 - test timestamp for exp=ascdaq18,run=170/1 for epixhr'\
      + '\n    5 - issue_2023_04_27 - test configuration for Ric generated epixhremu exp=tstx00417,run=277'\
      + '\n    6 - issue_2023_04_28 - test epixhremu - load default geometry'\
      + '\n    7 - issue_2023_05_19 - test opal.raw.image for Mona'\

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('0',): issue_2023_mm_dd()
elif TNAME in  ('1',): issue_2023_01_03()
elif TNAME in  ('2',): issue_2023_01_06()
elif TNAME in  ('3',): issue_2023_01_10()
elif TNAME in  ('4',): issue_2023_02_07()
elif TNAME in  ('5',): issue_2023_04_27()
elif TNAME in  ('6',): issue_2023_04_28()
elif TNAME in  ('7',): issue_2023_05_19()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

#if __name__ == "__main__":
