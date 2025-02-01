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


def issue_2025_mm_dd():
    print('template')


def issue_2025_01_29():
    """test for common mode in det.raw.calib/image implementation for archon
       datinfo -k exp=rixx1016923,run=118 -d archon
    """
    #ds, orun, det = ds_run_det(exp='rixc00121', run=140, detname='archon', dir='/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc')
    import numpy as np
    from time import time
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage

    #ds = DataSource(exp='rixc00121',run=154, dir='/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc',detectors=['archon']) # raw data shape=(1200,4800), >200 evts
    ds = DataSource(exp='rixx1016923',run=118, detectors=['archon'])
    orun = next(ds.runs())
    det = orun.Detector('archon', gainfact=2, cmpars=(1,0,0))

    flimg = None
    events = 2
    evsel = 0

    for nev, evt in enumerate(orun.events()):
       #print(info_ndarr(det.raw.raw(evt), '%3d: det.raw.raw(evt)' % nev))
       raw = det.raw.raw(evt)
       if raw is None:
           #print('evt:%3d - raw is None' % nev, end='\r')
           continue
       evsel += 1

       if evsel>events:
           print('BREAK for nev>%d' % events)
           break

       print('==== evt/sel: %4d/%4d' % (nev,evsel))

       t0_sec = time()

       #img  = det.raw.image(evt)
       #clb  = det.raw.calib(evt)

       img = raw
       #img = clb
       #img = det.raw._mask_fake(raw.shape)
       #img = det.raw._arr_to_image(clb)

       dt_sec = (time() - t0_sec)*1000
       print(info_ndarr(img, 'evt:%3d dt=%.3f msec  det.raw.img(evt)' % (nev, dt_sec)))
       print('det.raw._tstamp_raw(raw): ', det.raw._tstamp_raw(raw))

       if img.ndim==2 and img.shape[0] == 1:
           img = np.stack(1000*tuple(img))

       #img[0:100,0:100] = 0
       print(info_ndarr(img, 'img ')) #, last=100))

       if flimg is None:
          flimg = fleximage(img, h_in=5, w_in=16, nneg=1, npos=3) # arr=arr_img

       flimg.update(img) #, arr=arr_img
       flimg.fig.suptitle('Event %d: det.raw.raw' % nev, fontsize=16)
       gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
       gr.show(mode='DO NOT HOLD')

    gr.show()


def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc'  # None
    d_detname  = 'archon' # None
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    h_tname    = 'test name, usually numeric number, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME, usage=USAGE())
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest', default=d_subtest, type=str, help=h_subtest)
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

    if   TNAME in  ('0',): issue_2025_mm_dd() # template
    elif TNAME in  ('1',): issue_2025_01_29() # archon V2 common mode
    else:
        print(USAGE())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%TNAME)

    exit('END OF TEST %s'%TNAME)


if __name__ == "__main__":
    selector()

# EOF
