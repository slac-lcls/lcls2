#!/usr/bin/env python
"""./lcls2/psana/psana/detector/test_issues_2024.py <TNAME>
"""

import sys
import logging

SCRNAME = sys.argv[0].rsplit('/')[-1]
global STRLOGLEV # sys.argv[2] if len(sys.argv)>2 else 'INFO'
global INTLOGLEV # logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)


def ds_run_det(exp='ascdaq18', run=171, detname='epixhr', **kwa):
    from psana import DataSource
    ds = DataSource(exp=exp, run=run, **kwa)
    orun = next(ds.runs())
    det = orun.Detector(detname)
    return ds, orun, det


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
    Dubrovin, Mikhail
    Hi Mikhail,
    This archon calibration is working really well (and calibman is a nice tool for looking at constants).
    There is one small issue where it would be good to brainstorm about what is best:
    rixc00121 run 140 has the archon infull vertical binning mode where the det.raw.raw data shape is (1,4800),
    but det.calibconstpedestals[0] returns shape (4800).
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


def issue_2024_03_19():
    """config scan for epixm
       datinfo -k exp=tstx00417,run=309,dir=/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/ -d tst_epixm
    """
    from psana import DataSource
    import sys
    from psana.detector.NDArrUtils import info_ndarr
    from psana.detector.UtilsGraphics import gr, fleximage

    ds = DataSource(exp='tstx00417',run=317,dir='/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/')
    orun = next(ds.runs())
    det = orun.Detector('tst_epixm')
    calibconst = det.calibconst#['pedestals'][0]
    print('calibconst.keys():', calibconst.keys())
    #print('dir(det.raw):', dir(det.raw))
    geo = det.raw._det_geo()
    print('geo:', geo)
    peds = det.raw._pedestals()
    print(info_ndarr(peds, 'peds'))

    step_value = orun.Detector('step_value')
    step_docstring = orun.Detector('step_docstring')
    flimg = None
    for nstep, step in enumerate(orun.steps()):
        print('step:', nstep, step_value(step), step_docstring(step))
        for nevt,evt in enumerate(step.events()):
            if nevt==3: print('evt3 nstep:', nstep, ' step_value:', step_value(evt), ' step_docstring:', step_docstring(evt))

        for k,v in det.raw._seg_configs().items():
            cob = v.config
            print('dir(cob) w/o underscores:', [v for v in tuple(dir(cob)) if v[0]!='_'])
            print('  cob.CompTH_ePixM', cob.CompTH_ePixM)
            print('  cob.Precharge_DAC_ePixM', cob.Precharge_DAC_ePixM)

        #print(info_ndarr(det.raw.raw(evt), 'raw'))
        print(info_ndarr(det.raw.calib(evt), 'calib'))
        img = det.raw.image(evt)
        print(info_ndarr(img, 'image'))

        if flimg is None:
           flimg = fleximage(img, arr=None, h_in=5, w_in=10, nneg=1, npos=3)
        gr.set_win_title(flimg.fig, titwin='Event %d' % nevt)
        flimg.update(img, arr=None)
        gr.show(mode='DO NOT HOLD')
    gr.show()


def issue_2024_03_26():
    """generate and save text ndarray
    """
    import numpy as np
    #shape, dtype, fname = (4, 192, 384), np.float32, 'nda-epixm-gains.txt'
    shape, dtype, fname = (3, 4, 192, 384), np.float32, 'nda-epixm-peds.txt'
    import psana.pyalgos.generic.NDArrUtils as au
    nda = np.zeros(shape, dtype=dtype)
    #nda = np.ones(shape, dtype=dtype)
    from psana.detector.NDArrUtils import info_ndarr
    print(info_ndarr(nda, 'ones'))
    au.save_ndarray_in_textfile(nda, fname, 0o664, ' %d', umask=0o0, group='ps-users')
    print('saved in file %s' % fname)


def issue_2024_04_02():
    """junk.py from Chris and Ric
       datinfo -k exp=tstx00417,run=320,dir=/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc/ -d epixm

    """
    from psana import DataSource
    ds = DataSource(exp='tstx00417',run=320,dir='/reg/neh/operator/tstopr/data/drp/tst/tstx00417/xtc')
    myrun = next(ds.runs())
    det = myrun.Detector('epixm')
    for nevt,evt in enumerate(myrun.events()):
        raw = det.raw.raw(evt)
        print(raw.shape)
        print(det.raw.image(evt))
        if nevt>3: break


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

    TNAME = args.tname  # sys.argv[1] if len(sys.argv)>1 else '0'

    if   TNAME in  ('0',): issue_2024_mm_dd() # template
    elif TNAME in  ('1',): issue_2024_02_14() # ascdaq18,run=407, epixhr - zero pedestals for Philip
    elif TNAME in  ('2',): issue_2024_02_22() # calib for epixm320 tstx00417,run=308 for Ric
    elif TNAME in  ('3',): issue_2024_03_14() # new detector archon exp=rixc00121,run=142 for Chris
    elif TNAME in  ('4',): issue_2024_03_19() # Ric and Chris - config scan for epixm
    elif TNAME in  ('5',): issue_2024_03_26() # generate and save text ndarray
    elif TNAME in  ('6',): issue_2024_04_02() # junk.py from Chris and Ric
    else:
        print(USAGE())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%TNAME)

    exit('END OF TEST %s'%TNAME)


if __name__ == "__main__":
    selector()

# EOF
