#!/usr/bin/env python
"""./lcls2/psana/psana/detector/test_issues_2024.py <TNAME>
"""

import sys
import logging

SCRNAME = sys.argv[0].rsplit('/')[-1]
global STRLOGLEV # sys.argv[2] if len(sys.argv)>2 else 'INFO'
global INTLOGLEV  # logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
#logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)


def ds_run_det(exp='ascdaq18', run=171, detname='epixhr', **kwa):
    from psana import DataSource
    ds = DataSource(exp=exp, run=run, **kwa)
    orun = next(ds.runs())
    det = orun.Detector(detname)
    return ds, orun, det


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



def issue_2023_10_26():

    from psana import DataSource
    from psana.detector.NDArrUtils import info_ndarr

    ds = DataSource(exp='rixx1003721', run=200, intg_det='epixhr')

    M14 =  0x3fff  # 16383
    M15 =  0x7fff  # 32767

    print('data bits: %d' % M14)

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
          a = raw[:,145:,:192] & M14  # min:4811 max:10054
          print(info_ndarr(a,'raw:%4d' % ievt), 'min:%6d max:%6d' % (a.min(), a.max()), end='\r')


def issue_2024_mm_dd():
    print('template')


def issue_2024_02_14():
  """ISSUE: zero pedestals due to two versions of pedestals in the DB for bad and good dark processing
  """
  ds, orun, det = ds_run_det(exp='ascdaq18',run=407, detname='epixhr') #, dir='/cds/data/drpsrcf/tst/tstx00417/xtc')
  while True:
    evt = next(orun.events())
    if evt is None:
        continue
    raw = det.raw.raw(evt)
    if raw is None:
        continue
    cal = det.raw.calib(evt)
    status = det.raw._status()
    peds = det.raw._pedestals()
    print("raw:", raw)
    #print("cal:", cal)
    print("cal max, min:", cal.max(), cal.min())
    #print("status:", status)
    print("status max, min:", status.max(), status.min())
    #print("peds:", peds)
    print("peds max, min:", peds.max(), peds.min())
    break


def issue_2024_02_22():
    """ detnames exp=tstx00417,run=305,dir=/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc
    ---------------------------
    Name            | Data Type
    ---------------------------
    pvdetinfo_imNkM | pvdetinfo
    sp1k4           | raw
    im6k4           | raw
    im5k4           | raw
    triginfo        | triginfo
    timing          | raw
    """
    ds, orun, odet = ds_run_det(exp='tstx00417', run=305, detname='epixhr_emu', dir='/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc')
    print('odet.raw._uniqueid', odet.raw._uniqueid) # epixhremu_00cafe0002-0000000000-0000000000-0000000000-...
    print('odet.raw._det_name', odet.raw._det_name) # epixhr_emu
    print('odet.raw._dettype',  odet.raw._dettype)  # epixhremu

    detname = longname = odet.raw._uniqueid
    #detname = 'epixhremu_000002' # DB name
    import psana.pscalib.calib.MDBWebUtils as wu
    calib_const = wu.calib_constants_all_types(detname, exp='tstx00417', run=9999)
    #calib_const = wu.calib_constants_all_types(detname, run=9999)
    print('calib_const.keys:', calib_const.keys())


def issue_2024_03_14():
    """
    O'Grady, Paul Christopher <cpo@slac.stanford.edu>
​    Dubrovin, Mikhail
    Hi Mikhail,
    This archon calibration is working really well (and calibman is a nice tool for looking at constants).
    There is one small issue where it would be good to brainstorm about what is best:
    rixc00121 run 140 has the archon in “full vertical binning” mode where the det.raw.raw data shape is (1,4800),
    but det.calibconst[‘pedestals’][0] returns shape (4800).
    Would it be possible (and a good idea?) to have det_dark_proc store the pedestals with the same shape as the data (1,4800)?
    Thanks for any thoughts,
    chris

    datinfo -k "{'exp':'rixc00121','run':140,'dir':'/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc','detectors':['archon']}" -d archon
    det_dark_proc -k "{'exp':'rixc00121','run':140,'dir':'/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc','detectors':['archon']}" -d archon -o work
    """
    #ds, orun, det = ds_run_det(exp='rixc00121', run=140, detname='archon', dir='/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc')
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    ds = DataSource(exp='rixc00121',run=140, dir='/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc',detectors=['archon']) # raw data shape=(1,4800)
    #ds = DataSource(exp='rixc00121',run=142,dir='/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc',detectors=['archon']) # raw data shape=(75,4800)
    orun = next(ds.runs())
    det = orun.Detector('archon')
    evt = next(orun.events())
    print(info_ndarr(det.raw.raw(evt), 'det.raw.raw(evt)'))
    peds = det.calibconst['pedestals'][0]
    print(info_ndarr(peds, 'peds'))
    print('\ndet.calibconst["pedestals"]', det.calibconst['pedestals'])


def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc'  # None
    d_detname  = 'archon' # None
    d_loglevel = 'INFO' # 'DEBUG'
    h_tname    = 'test name, usually numeric number, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME, usage=USAGE())
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    return parser


def USAGE():
    import inspect
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
    + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "TNAME in" in s])


def selector():

    parser = argument_parser()
    args = parser.parse_args()
    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)
    #basic_config(format='[%(levelname).1s] L%(lineno)04d: %(filename)s %(message)s', int_loglevel=None, str_loglevel=args.loglevel)
    #logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) # get rid of messages
    #STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'
    #TNAME = sys.argv[1] if len(sys.argv)>1 else None

    TNAME = args.tname  # sys.argv[1] if len(sys.argv)>1 else '0'

    if   TNAME in  ('0',): issue_2024_mm_dd() # template
    elif TNAME in  ('1',): issue_2023_05_19() # opal.raw.image for Mona'
    elif TNAME in  ('2',): issue_2023_07_25() # epixhremu.fex.image for Ric
    elif TNAME in  ('3',): issue_2023_07_26() # calib_constants_all_types for Ric
    elif TNAME in  ('4',): issue_2023_07_27() # calib_constants_all_types for Ric - curl commands
    elif TNAME in  ('5',): issue_2024_02_14() # ascdaq18,run=407, epixhr - zero pedestals for Philip
    elif TNAME in  ('6',): issue_2024_02_22() # calib for epixm320 tstx00417,run=308 for Ric
    elif TNAME in  ('7',): issue_2024_03_14() # new detector archon exp=rixc00121,run=142 for Chris
    else:
        print(USAGE())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%TNAME)

    exit('END OF TEST %s'%TNAME)


if __name__ == "__main__":
    selector()

# EOF
