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


def issue_2023_07_25():
    """datinfo -k exp=tstx00417,run=287,dir=/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc/ -d epixhr_emu
       on pcds use psffb, which sees data
       or use s3dff
    """
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage#, arr_median_limits
    flimg = None

    #ds = DataSource(exp='tstx00417', run=286, dir='/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc') # on s3df 3-panel run
    ds = DataSource(exp='tstx00417', run=286, dir='/cds/data/drpsrcf/tst/tstx00417/xtc') # on pcds 3-panel run
    #ds = DataSource(exp='tstx00417', run=287, dir='/cds/data/drpsrcf/tst/tstx00417/xtc') # on pcds 20-panel run detectors=['epixhr_emu'])
    for run in ds.runs():
      det = run.Detector('epixhr_emu')
      for nev,evt in enumerate(run.events()):
        if nev>10000: break
        arr = det.fex.calib(evt)
        if arr is None: continue
        print(info_ndarr(arr, '==== ev:%05d  evt.timestamp: %d arr:' % (nev, evt.timestamp)))

        img = det.fex.image(evt, value_for_missing_segments=800)
        #img = det.fex.image(evt)
        print(info_ndarr(img, 43*' ' + 'image:'))

        if flimg is None:
           #flimg = fleximage(img, arr=None, h_in=8, w_in=11, amin=9000, amax=11000) #, nneg=1, npos=3)
           flimg = fleximage(img, arr=None, h_in=8, w_in=11, amin=700, amax=800) #, nneg=1, npos=3)
        gr.set_win_title(flimg.fig, titwin='Event %d' % nev)
        flimg.update(img, arr=None)
        gr.show(mode='DO NOT HOLD')
    gr.show()


def issue_2023_07_26():
    """
    """
    from time import time
    import psana.pscalib.calib.MDBWebUtils as wu
    t0_sec = time()
    det_name = 'epixhremu_00cafe000c-0000000000-0000000000-0000000000-0000000000-0000000000-0000000000'
    #det_name = 'epixhremu_000002' # works
    #det_name = 'epixhr_emu' # non-populated  in db
    calib_const = wu.calib_constants_all_types(det_name, exp='tstx00417', run=0)
    print('consumed time = %.6f sec' % (time()-t0_sec))
    print('calib_const.keys:', calib_const.keys())


def issue_2023_07_27():
    """curl commands to access data from GridFS equivalent to previous script:
        2.031110 sec - get data for ctype: pixel_status
        2.059888 sec - get data for ctype: geometry
        2.929590 sec - get data for ctype: pixel_gain
        3.967860 sec - get data for ctype: pedestals
        4.862775 sec - get data for ctype: pixel_rms
    """
    from requests import get
    from time import time

    requests = [\
             'https://pswww.slac.stanford.edu/calib_ws/cdb_tstx00417/gridfs/6450604a3a7ab9e8b9dc63b2',
             'https://pswww.slac.stanford.edu/calib_ws/cdb_tstx00417/gridfs/64514377ff231530db74d279',
             'https://pswww.slac.stanford.edu/calib_ws/cdb_tstx00417/gridfs/64506059a42d4cbaddd64eff',
             'https://pswww.slac.stanford.edu/calib_ws/cdb_tstx00417/gridfs/64506032c67f3e7208e57b33',
             'https://pswww.slac.stanford.edu/calib_ws/cdb_tstx00417/gridfs/6450604110a2e726664e6a54',
            ]

    curls = ['curl -s "%s"' % r for r in requests]

    print('\n'.join([s for s in curls]))
    query=None
    for url in requests:
        t0_sec = time()
        print('====\n  in python: r = get("%s", query=None, timeout=180)' % url)
        print('  equivalent in os: curl -s "%s"' % url)

        r = get(url, query, timeout=180)

        if r.ok:
            s = 'content[0:30]: %s...' % r.content[0:30]\
              + '\nconsumed time = %.6f sec' % (time()-t0_sec)
            #mu.object_from_data_string(s, doc)

        else:
            s = 'get url: %s query: %s\n  response status: %s status_code: %s reason: %s'%\
                (url, str(query), r.ok, r.status_code, r.reason)
            s += '\nTry command: curl -s "%s"' % url
        print(s)


def issue_2023_10_04():
    """ detnames exp=tstx00417,run=286,dir=/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc
    Name       | Data Type
    ----------------------
    epixhr_emu | raw ...
    """
    ds, orun, odet = ds_run_det(exp='tstx00417', run=286, detname='epixhr_emu', dir='/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc')
    print('odet.raw._uniqueid', odet.raw._uniqueid) # epixhremu_00cafe0002-0000000000-0000000000-0000000000-...
    print('odet.raw._det_name', odet.raw._det_name) # epixhr_emu
    print('odet.raw._dettype',  odet.raw._dettype)  # epixhremu

    detname = longname = odet.raw._uniqueid
    #detname = 'epixhremu_000002' # DB name
    import psana.pscalib.calib.MDBWebUtils as wu
    calib_const = wu.calib_constants_all_types(detname, exp='tstx00417', run=9999)
    #calib_const = wu.calib_constants_all_types(detname, run=9999)
    print('calib_const.keys:', calib_const.keys())

def issue_2023_10_05():
    from time import time
    from psana import DataSource
    ds = DataSource(exp='uedcom103',run=812)
    t0_sec = time()
    orun = next(ds.runs()) # 4.2 sec !!!!
    print('next(ods.runs()) time = %.6f' % (time() - t0_sec))
    uniqueid = orun.Detector('epixquad').raw._uniqueid
    print('uniqueid:', uniqueid)


def issue_2023_10_26():

    from psana import DataSource
    from psana.detector.NDArrUtils import info_ndarr

    ds = DataSource(exp='rixx1003721', run=200, intg_det='epixhr')

    M14 =  0x3fff  # 16383
    M15 =  0x7fff  # 32767

    print('data bits: %d' % M15)

    for irun,orun in enumerate(ds.runs()):
      print('\n\n== run:%d' % irun)
      det = orun.Detector('epixhr')
      for istep,ostep in enumerate(orun.steps()):
        print('\n==== step:%d' % istep)
        for ievt,evt in enumerate(ostep.events()):

          if evt is None:
              continue
          raw = det.raw.raw(evt)
          if raw is None:
              continue
          #a = raw[:,:144,:192] # min:0 max:0
          #a = raw[:,:144,193:] # min:0 max:0
          #a = raw[:,145:,193:] # min:0 max:0
          a = raw[:,145:,:192] & M15  # min:4811 max:10054
          print(info_ndarr(a,'raw:%4d' % ievt), 'min:%6d max:%6d' % (a.min(), a.max()), end='\r')


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
      + '\n    8 - issue_2023_07_25 - test epixhremu.fex.image for Ric'\
      + '\n    9 - issue_2023_07_26 - test calib_constants_all_types for Ric'\
      + '\n   10 - issue_2023_07_27 - test calib_constants_all_types for Ric - curl commands'\
      + '\n   11 - issue_2023_10_04 - test calib constants for detector names'\
      + '\n   12 - issue_2023_10_05 - test orun = next(ds.runs()) dt = 4.2 sec !!!'\
      + '\n   13 - issue_2023_10_26 - issue reported by philip'\

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('0',): issue_2023_mm_dd()
elif TNAME in  ('1',): issue_2023_01_03()
elif TNAME in  ('2',): issue_2023_01_06()
elif TNAME in  ('3',): issue_2023_01_10()
elif TNAME in  ('4',): issue_2023_02_07()
elif TNAME in  ('5',): issue_2023_04_27()
elif TNAME in  ('6',): issue_2023_04_28()
elif TNAME in  ('7',): issue_2023_05_19()
elif TNAME in  ('8',): issue_2023_07_25()
elif TNAME in  ('9',): issue_2023_07_26()
elif TNAME in ('10',): issue_2023_07_27()
elif TNAME in ('11',): issue_2023_10_04()
elif TNAME in ('12',): issue_2023_10_05()
elif TNAME in ('13',): issue_2023_10_26()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

#if __name__ == "__main__":
