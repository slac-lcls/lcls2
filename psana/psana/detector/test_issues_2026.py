#!/usr/bin/env python
"""./lcls2/psana/psana/detector/test_issues_2026.py <TNAME> [-h]"""

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


def issue_2026_02_04():
    """jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 50 --nrecs1 50 --stepnum 1
       ISSUE: continue seems does not work in the step loop
       PROBLEM: loop is terminated earlier due to max_events=100
       FIX: increase 'max_events':100000
    """
    from psana import DataSource
    #ds = DataSource(exp='mfx100848724', run=49)
    ds = DataSource(exp='mfx100848724', run=49, **{'max_events':100000})
    orun = next(ds.runs())
    det = orun.Detector('jungfrau')
    for istep, step in enumerate(orun.steps()):
        print('begin step: %d' % istep)
        if istep == 0:
            continue
        print('end of step: %d' % istep)
    print('end of step loop')


def issue_2026_02_26():
    """ISSUE: command
              jungfrau_dark_proc -k exp=xppc00125,run=6 -d jungfrau1M -o work1 --stepnum 0
              does not work (work in old release) because of mpi features
       PROBLEM: dskwargs={...,'smd_callback': None}
       FIX: remove 'smd_callback': None from dskwargs
    """
    if False:
        import os
        os.environ['PS_EB_NODES']='1'
        os.environ['PS_SRV_NODES']='1'

        import psutil
        cpu_num = psutil.Process().cpu_num()

        from psana.detector.Utils import get_hostname
        hostname = get_hostname()

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        s_rsch = 'rank:%03d/%03d-cpu:%03d-%s' % (rank, size, cpu_num, hostname)
        print('>>>> %s sys.argv: %s' % (s_rsch, sys.argv))

    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    #[I] UtilsJungfrauCalib.py L0163 DataSource dskwargs: {'exp': 'xppc00125', 'run': 6, 'max_events': 10000, 'batch_size': 2, 'smd_callback': None}
    #dskwargs={'exp': 'xppc00125', 'run': 6, 'max_events': 10000, 'batch_size': 2, 'smd_callback': None}
    dskwargs={'exp': 'xppc00125', 'run': 6, 'max_events': 10000, 'batch_size': 2}
    ds = DataSource(**dskwargs)
    #run = next(ds.runs())
    for irun, run in enumerate(ds.runs()):
      logger.info('\n%s runnum %d %s' % (20*'=', run.runnum, 20*'='))
      det = run.Detector('jungfrau1M')
      for istep, step in enumerate(run.steps()):
        print('begin step: %d' % istep)
        for ievt, evt in enumerate(step.events()):
            if ievt<10:
                raw = det.raw.raw(evt)
                print(info_ndarr(raw, 'ev: %03d raw' % ievt))
        print('end of step: %d' % istep)
    print('end of step loop')


def issue_2026_03_16():
    """ISSUE: Hi Mikhail, I looked into the problem Valerio/Chris reported and traced it to the Jungfrau pedestals
         deployed for mfx100848724, starting from run 49. It looks like the deployed pedestal constants have
         the wrong dtype: float64 instead of the expected float32. Because of that, det.raw.calib(evt) fails
         for runs that resolve to this pedestal. I reproduced the failure for mfx100848724, run 51.
       PROBLEM:
       FIX:
    """
    import psana
    ds = psana.DataSource(exp="mfx100848724", run=51, dir="/sdf/data/lcls/ds/prj/public01/xtc", max_events=10)
    run = next(ds.runs())
    det = run.Detector("jungfrau")

    for evt in run.events():
        data = det.raw.raw(evt)  # raw data
        print(f"Extracted raw for {evt}")
        data = det.raw.calib(evt)  # calibrated data
        print(f"Extracted calib for {evt}")



def issue_2026_03_27(subtest='0o7777'):
    """ISSUE: 2026-03-26 3:20PM O'Grady, Paul Christopher
              Dubrovin, Mikhail?; Vincent Esposito <vincent.esposito@stanford.edu>?
              Hi Mikhail, Vincent sees unusual looking images for the jungfrau 1M in xppc00125 run 77 (see attached).
              I think Vincent deployed a dark using run 76.  Can you have a look to see if you see anything wrong?
       REASON: detector does not work. Dark raw intensity jumps between 3k and 50k
       FIX: Vincent will meet with detector grp

       datinfo -k exp=xppc00125,run=77 -d jungfrau1M
    """
    import os
    import numpy as np
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage, fleximagespec
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d
    #from time import sleep

    events = 20

    ds = DataSource(exp='xppc00125', run=77, **{'max_events':events})
    run = next(ds.runs())
    det = run.Detector('jungfrau1M')

    peds = det.raw._pedestals()
    mask = det.raw._mask();
    mask_default = det.raw._mask_default()
    mask_calib_or_default = det.raw._mask_calib_or_default()
    mask_edges = det.raw._mask_edges(edge_rows=0, edge_cols=0)
    mask_center = det.raw._mask_center()
    mask_neighbors = det.raw._mask_neighbors(mask_default)
    stat = det.raw._status() #; stat.shape = (3*512, 1024)
    mask_stat = det.raw._mask_from_status(gain_range_inds=(0,)) #; mask_stat.shape = (512, 1024)
    print(ndu.info_ndarr(peds,                  'XXX peds', last=10))
    print(ndu.info_ndarr(stat,                  'XXX stat', last=10))
    print(ndu.info_ndarr(mask_stat,             'XXX _mask_stat', last=10))
    print(ndu.info_ndarr(mask_default,          'XXX _mask_default', last=10))
    print(ndu.info_ndarr(mask_calib_or_default, 'XXX _mask_calib_or_default', last=10))
    print(ndu.info_ndarr(mask_edges,            'XXX _mask_edges', last=10))
    print(ndu.info_ndarr(mask_center,           'XXX _mask_center', last=10))
    print(ndu.info_ndarr(mask_neighbors,        'XXX _mask_neighbors', last=10))
    print(ndu.info_ndarr(mask,                  'XXX _mask', last=10))

    plot_image = True # False

    if True:
        flimg = None
        for nevt,evt in enumerate(run.events()):
            #if nevt>events-1: break
            raw   = det.raw.raw(evt)
            if raw is None: continue

            print('=== evt:%03d'%nevt)
            print(ndu.info_ndarr(raw, '=== evt:%03d raw' % nevt, last=10))
            img = raw[0,:]
            #img = raw[1,:]
            #img.shape = (2*512, 1024)
            #calib = det.raw.calib(evt)
            #print(ndu.info_ndarr(calib, '    calib', last=10))
            #print('>> end calib')
            #img = det.raw.image(evt)
            #img = calib
            #img = mask
            #img = mask_edges
            #img = stat
            #img = det.raw.image(evt, nda=calib)
            #img = calib; img.shape = (512, 1024)
            #img = raw; img.shape = (512, 1024)
            #img = ndu.reshape_to_3d(peds)
            #img.shape = (512, 1024)
            #img = peds[0,0,:]
            #img = peds; img.shape = (3*512, 1024)
            #print('evt:', nevt)
            #print('    raw  :', raw.shape)
            #print('    calib:', calib.shape)

            print(ndu.info_ndarr(img, '            img', last=10))

            if plot_image:
                if flimg is None:
                    #flimg = fleximage(img, h_in=10, w_in=12)
                    flimg = fleximagespec(img, h_in=7, w_in=20)
                    gr.plt.ion()
                #print('    image:', img.shape)
                title = 'evt %02d test image'%nevt
                #flimg.fig.suptitle(title, fontsize=16)
                gr.set_win_title(flimg.fig, titwin=title)
                flimg.update(img)
                #gr.draw_fig(flimg.fig)
                #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
                #gr.draw_fig(flimg.fig)
                #flimg.fig.canvas.draw_idle()
                gr.show(mode='DO NOT HOLD', pause_sec=1)
                #gr.plt.show(block=False)
                #gr.plt.pause(0.01)
        if plot_image: gr.show()


def issue_2026_04_01(subtest='0o7777'):
    """ISSUE:
       PROBLEM:
       FIX:

    datinfo -k exp=tstx00117,run=330,dir=/sdf/data/lcls/drpsrcf/ffb/users/tst_data/tst/tstx00117/xtc/ -d epixuhr3x2

    """
    print('test of emulated data for epixuhr3x2')

    import os
    import numpy as np
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage, fleximagespec
    import psana.detector.NDArrUtils as ndu
    import psana.detector.UtilsEpixUHR as ueu
    #from time import sleep

    events = 20

    ds = DataSource(exp='tstx00117', run=330, dir='/sdf/data/lcls/drpsrcf/ffb/users/tst_data/tst/tstx00117/xtc/', **{'max_events':events})
    run = next(ds.runs())
    det = run.Detector('epixuhr3x2')

    #peds = det.raw._pedestals()
    #mask = det.raw._mask();
    #print('type(det)', type(det))
    print('det.raw._segment_numbers', det.raw._segment_numbers)

    #cfgs = getattr(det, 'config')._seg_configs() # dict for {<segnum>: <psana.container.Container object>}
    cfgs = det.config._seg_configs()

    isubset = 0o7777 if subtest is None else int(subtest)
    if isubset & 1:
        print('cfgs:', cfgs)
        print('type(cfgs):', type(cfgs))
        print('cfgs[0]:', cfgs[0])
        print('dir(cfgs[0]):', dir(cfgs[0]))
        print('cfgs[0].config:', cfgs[0].config)

    if isubset & 2:
        from psana.app.config_dump import dump
        attrlist=[]
        for myobj in cfgs.values():
            dump(myobj, attrlist)

    if isubset & 4:
      for segnum in det.raw._segment_numbers:
        cfg = cfgs.get(segnum, None)
        if cfg is None:
            print('segment %d configuration is None' % segnum)
            continue
        segcfg = cfg.config
        #print('segment %d configuration: %s' % (segnum, str(dir(cfg.config))))
        print('gainAsic:', segcfg.gainAsic)
        print(ndu.info_ndarr(segcfg.gainCSVAsic, 'gainCSVAsic:'))
      print(ndu.info_ndarr(ueu.cbits_epixuhr(det), '== cbits_epixuhr(det):'))
      print(ndu.info_ndarr(ueu.gains_epixuhr(det), '== gains_epixuhr(det):'))

    if isubset & 8:
        cbits = ueu.cbits_epixuhr(det)
        print(ndu.info_ndarr(ueu.cbits_epixuhr(det), 'cbits_epixuhr(det):'))
        print(ndu.info_ndarr(ueu.gains_epixuhr(det), 'gains_epixuhr(det):'))
        gmaps = ueu.gain_maps_epixuhr(cbits)
        print(ueu.info_gain_mode_arrays(gmaps, first=0, last=5, spacer='\n    ', cmt='gain mode arrays:'))


def issue_2026_04_10(subtest='0o7777'):
    """ISSUE: Hart, Philip Adam, Apr 10, 2026, at 10:50AM
              O'Grady, Paul Christopher; Dubrovin, Mikhail; Uervirojnangkoorn, Monarin; Siddiqui, Khalid
              Hi, it's a bit hard to demonstrate the pedestal claim clearly since the raw data doesn't match the calib ordering (so I claim) but
              python -i /sdf/home/p/philiph/repos/beamtime-calibration-suite/standalone_scripts/uedPedCheck.py
              plots a dark frame image and the pixels that Mona has set low don't get pedestal subtracted.
              Thanks,Philip

              datinfo -k exp=ued1016014,run=51 -d epixquad1kfps
              datinfo -k exp=ued1015999,run=185 -d epixquad1kfps
              ('epixquad1kfps', 'raw') : <class 'psana.detector.epix10ka.epix10ka_raw_2_0_1'>
       PROBLEM:
       FIX:
    """
    from psana import DataSource
    from time import time
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import psana.detector.NDArrUtils as ndu

    def plot_image(image, title='image', call_clf=True, figsize=None, block_show=True,\
                   pos=(10,10), vmin=None, vmax=None, fname=None):
        #if call_clf: plt.clf()
        fig = plt.figure(figsize=figsize)
        axim = fig.add_axes((0.08, 0.05, 0.89, 0.93)) #, **kwa)
        imsh = axim.imshow(image, vmin=vmin, vmax=vmax)
        fig.canvas.manager.set_window_title(title)
        #plt.title(title)
        fig.canvas.manager.window.move(pos[0], pos[1])
        plt.show(block=block_show)
        if fname is not None:
            print(f'save plot in file: {fname}')
            fig.savefig(fname) #, **kwa)

    isubset = 0o7777 if subtest is None else int(subtest)

    exp, run, detName = 'ued1016014', 50, 'epixquad1kfps'
    if isubset & 16: exp, run, detName = 'ued1015999', 185, 'epixquad1kfps'

    ds = DataSource(exp=exp, run=run, detectors=[detName])
    myrun = next(ds.runs())
    det = myrun.Detector(detName)

    if isubset & 1:
        while True:
            evt = next(myrun.events())
            #det.raw._raw_buf = None
            t0_sec = time()
            raw = det.raw.raw(evt) #, copy=False)
            print('XXX det.raw.raw(evt) time: %.6f sec' % (time() - t0_sec))

            cal = det.raw.calib(evt)
            print('XXX AreaDetectorRaw.raw IS CALLED TWISE')

            rimg = det.raw.image(evt, raw)
            dimg = det.raw.image(evt)
            #peds = det.calibconst["pedestals"][0] # shape (7, 4, 352, 384)
            #pimg = det.raw.image(evt, peds[0])
            cimg = det.raw.image(evt, cal)
            break

        print(ndu.info_ndarr(raw, 'raw:'))
        print(ndu.info_ndarr(cal, 'cal:'))
        #print(ndu.info_ndarr(pimg, 'pimg:'))
        print(ndu.info_ndarr(dimg, 'dimg:'))
        print(ndu.info_ndarr(rimg, 'rimg:'))
        print(ndu.info_ndarr(cimg, 'cimg:'))

        #plot_image(pimg, title='peds', call_clf=False)
        #plot_image(dimg.clip(-50, 500), title='default image, clipped', call_clf=True)
        plot_image(rimg, title='raw, image', block_show=False, pos=(10,0), fname='img-phil-raw.png')
        plot_image(cimg.clip(-50, 500), title='calib, clipped', block_show=True, pos=(400,0), fname='img-phil-calib.png')

    if isubset & 2:
        #while True:
            #evt = next(myrun.events())
        for nevt,evt in enumerate(myrun.events()):
            #det.raw._raw_buf = None
            arr = det.raw._array(evt)
            raw = det.raw.raw(evt) #, copy=False)
            cal = det.raw.calib(evt)
            print('XXX === evt: %03d' % nevt)
            print(ndu.info_ndarr(raw, 'raw'))
            print(ndu.info_ndarr(cal, 'cal'))
            #print(ndu.info_ndarr(arr, 'arr'))
            if nevt>1: break
        #import matplotlib
        #print(matplotlib.get_backend())
        #plot_image(arr, title='det.raw._array', figsize=(8,8), block_show=False, pos=(500,0))
        if isubset & 4:
          plot_image(ndu.reshape_to_2d(raw), title='raw as 2d', figsize=(4,8), block_show=False, pos=(10,0))
          plot_image(ndu.reshape_to_2d(cal), title='cal as 2d', figsize=(4,8), block_show=False, pos=(350,0), vmin=0, vmax=5)
          plot_image(ndu.reshape_to_2d(cal), title='cal as 2d re-scailed', figsize=(4,8), block_show=True,  pos=(700,0), vmin=-5, vmax=5)
        if isubset & 8:
           plot_image(det.raw.image(evt, raw), title='det.raw.image(evt,raw)', figsize=(8,8), block_show=False, pos=(10,0))
           plot_image(det.raw.image(evt, cal), title='det.raw.image(evt,cal)', figsize=(8,8), block_show=False, pos=(350,0), vmin=0, vmax=5)
           plot_image(det.raw.image(evt, cal), title='det.raw.image(evt,cal)', figsize=(8,8), block_show=True,  pos=(700,0), vmin=-5, vmax=5, fname='img-my-calib.png')

    if isubset & 16:
        #while True:
            evt = next(myrun.events())
            raw = det.raw.raw(evt) & 0x3fff
            peds = det.raw._pedestals()
            raw_peds = raw - peds[2,:]
            print(ndu.info_ndarr(raw_peds, 'raw - peds'))
            plot_image(det.raw.image(evt, raw_peds), title='det.raw.image(evt,raw_peds)', figsize=(8,8), block_show=True, pos=(10,10), vmin=0, vmax=20, fname='img-my-raw-peds.png')




def issue_2026_04_15(args):
    """ISSUE: jungfrau 1M in xppc00125 run 77.
       REASON:
       FIX:
       Mona: You can try ued1015999 run 185. Here's a full frame image of one shot sent by UED.
       datinfo -k exp=ued1015999,run=185 -d epixquad1kfps
       datinfo -k exp=xppc00125,run=148 -d jungfrau1M
    """
    import os
    import numpy as np
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage, fleximagespec
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d
    #from time import sleep

    events = args.events
    isubset = 0o7777 if args.subtest is None else int(args.subtest)

    if isubset & 1: expname, runnum, detname = 'xppc00125', 148, 'jungfrau1M'
    if isubset & 2: expname, runnum, detname = 'ued1015999', 185, 'epixquad1kfps' # gain mode FL

    amin, amax = None, None
    if isubset & 1: amin, amax = 0, 15

    ds = DataSource(exp=expname, run=runnum, **{'max_events':events})
    run = next(ds.runs())
    det = run.Detector(detname)

    geo_meta = det.calibconst['geometry']
    if geo_meta is not None: print('geometry metadata:', geo_meta[1])

    peds = det.raw._pedestals()
    print(ndu.info_ndarr(peds, 'det.raw._pedestals()', last=10))

    if True:
        plot_image = True
        flimg = None
        for nevt,evt in enumerate(run.events()):
            raw = det.raw.raw(evt)
            print(ndu.info_ndarr(raw, '=== evt:%03d raw' % nevt, last=10))
            #arr = np.array(raw, dtype=np.float32) & 0x7fff - peds[2,:] if isubset & 2 else raw
            arr = det.raw.calib(evt)
            #img = ndu.reshape_to_2d(raw)
            img = det.raw.image(evt, arr)

            print(ndu.info_ndarr(img, '             img', last=10))

            if plot_image:
                if flimg is None:
                    flimg = fleximagespec(img, h_in=8, w_in=12, amin=amin, amax=amax)
                    gr.plt.ion()
                title = 'evt %02d test image'%nevt
                #flimg.fig.suptitle(title, fontsize=16)
                gr.set_win_title(flimg.fig, titwin=title)
                flimg.update(img)
                gr.show(mode='DO NOT HOLD', pause_sec=1)
        if plot_image: gr.show()

#===

def issue_2026_MM_DD(subtest='0o7777'):
    """ISSUE:
       PROBLEM:
       FIX:
    """
    print('template')

#===
    

def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc'  # None
    d_detname  = 'archon' # None
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    d_events   = 5
    h_tname    = 'test name, usually numeric number 0,...,>20, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    h_events   = '(int) number of events, default = %d' % d_events
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME,\
                            usage='for list of implemented tests run it without parameters')
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
    parser.add_argument('-n', '--events',   default=d_events,   type=int, help=h_events)
    return parser


def selector():
    parser = argument_parser()
    args = parser.parse_args()
    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    TNAME = args.tname # sys.argv[1] if len(sys.argv)>1 else '0'

    if   TNAME in ('0',): issue_2026_mm_dd() # template
    elif TNAME in ('1',): issue_2026_02_04()
    elif TNAME in ('2',): issue_2026_02_26()
    elif TNAME in ('3',): issue_2026_03_16()
    elif TNAME in ('4',): issue_2026_03_27(args.subtest)
    elif TNAME in ('5',): issue_2026_04_01(args.subtest)
    elif TNAME in ('6',): issue_2026_04_10(args.subtest)
    elif TNAME in ('7',): issue_2026_04_15(args) # various images selected by -s [#]
    elif TNAME in ('99',):issue_2026_MM_DD(args.subtest)
    else:
        print(USAGE())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%TNAME)
    exit('END OF TEST %s'%TNAME)


def USAGE():
    import inspect
    #return '\n  TEST'
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
         + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "TNAME in" in s])\
         + '\n\nHELP:\n  list of parameters: ./%s -h\n  list of tests:      ./%s' % (SCRNAME, SCRNAME)


if __name__ == "__main__":
    if len(sys.argv)==1:
        print(USAGE())
        exit('\nMISSING ARGUMENTS -> EXIT')
    selector()

# EOF
