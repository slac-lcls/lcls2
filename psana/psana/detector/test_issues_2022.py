#!/usr/bin/env python

import sys
import logging
SCRNAME = sys.argv[0].rsplit('/')[-1]
STRLOGLEV = sys.argv[2] if len(sys.argv)>2 else 'INFO'
INTLOGLEV = logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)


def issue_2022_01_21():
    """O'Grady, Paul Christopher <cpo@slac.stanford.edu> Wed 1/19/2022 9:08 PM
       Hi Mikhail,
       I took exp=tmoc00318,run=8 that has some epix100 data.  Some caveats, however.
       There are only 4 events. The detector name/type is wrong.
       There is no good detector interface yet, but I kludged one that you can see in
       ~cpo/git/lcls2/psana/psana/detector/ in the files epix100.py and one added line at the bottom of detectors.py.
       With that the script below works for me.
       chris
    """
    from psana import DataSource
    ds = DataSource(exp='tmoc00318',run=8)
    orun = next(ds.runs())
    det = orun.Detector('epix100hw')
    for nevt,evt in enumerate(orun.events()):
        print('det.raw.raw(evt).shape  :', det.raw.raw(evt).shape)
        print('det.raw.calib(evt).shape:', det.raw.calib(evt).shape)
        print('det.raw.image(evt).shape:', det.raw.image(evt).shape)


def issue_2022_01_26():
    """The same as issue_2022_01_21 but for run 10, print ndarray, access constants.
    """
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    ds = DataSource(exp='tmoc00318',run=10)
    orun = next(ds.runs())
    det = orun.Detector('epix100')

    print('dir(det.raw):', dir(det.raw))
    print()
    print(info_ndarr(det.raw._pedestals(),   'det.raw._pedestals()  '))
    print(info_ndarr(det.raw._gain(),        'det.raw._gain()'))
    print(info_ndarr(det.raw._rms(),         'det.raw._rms()'))
    print(info_ndarr(det.raw._status(),      'det.raw._status()'))
    print(info_ndarr(det.raw._mask_calib(),  'det.raw._mask_calib()'))
    print(info_ndarr(det.raw._mask_from_status(),  'det.raw._mask_from_status()'))
    print(info_ndarr(det.raw._mask_edges(),  'det.raw._mask_edges()'))
    print(info_ndarr(det.raw._common_mode(), 'det.raw._common_mode()'))
    #print(info_ndarr(det.raw.,   'det.raw.'))
    print(info_ndarr(det.raw._pixel_coords(do_tilt=True, cframe=0), 'det.raw._pixel_coords(...)'))

    print()

    for nevt,evt in enumerate(orun.events()):
        if nevt>10:
            print('event loop is terminated by maximal number of events')
            break
        print(info_ndarr(det.raw.raw(evt),   'det.raw.raw(evt)  '))
        print(info_ndarr(det.raw.calib(evt), 'det.raw.calib(evt)'))


def issue_2022_02_08():
    """test copy xtc2 file to .../public01/xtc/
    cd /cds/data/psdm/prj/public01/xtc/
    cp /cds/data/psdm/tmo/tmoc00318/xtc/tmoc00318-r0010-s000-c000.xtc2 .
    sudo chown psdatmgr tmoc00318-r0010-s000-c000.xtc2
    the same for smalldata/
    """
    from time import time
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource

    ds = DataSource(exp='tmoc00318',run=10, dir='/cds/data/psdm/prj/public01/xtc')
    orun = next(ds.runs())
    det = orun.Detector('epix100')

    for i,evt in enumerate(orun.events()):
        if i>20: break
        t0_sec = time()
        arr = det.raw.calib(evt, cmpars=(0,7,100,10)) # None or 0/1/2/4/7 : dt=0.02/0.036/0.049/0.016/0.90 sec
        #arr = det.raw._common_mode_increment(evt, cmpars=(0,7,100,10))
        print(info_ndarr(arr, 'Ev.%3d dt=%.3f sec  det.raw.calib(evt, cmpars=(0,7,100,10)): '%(i, time()-t0_sec)))

def issue_2022_02_15():
    """O'Grady, Paul Christopher <cpo@slac.stanford.edu> Tue 2/15/2022 3:32 PM
       Hi Mikhail,
       I?m having trouble getting det.raw.image to work from the head of git.
       I ran pedestals as shown in ~cpo/junk.txt, but det.raw.image returns None even though det.raw.raw and det.raw.calib show data.
       So it feels like it?s a geometry issue?  Would you have some advice?  Thank you?
       chris

       *** event 1404 (1, 704, 768) (1, 704, 768) None
       *** event 1405 (1, 704, 768) (1, 704, 768) None
       *** event 1406 (1, 704, 768) (1, 704, 768) None
    """
    from psana import DataSource
    ds = DataSource(exp='tmox49720',run=209)
    myrun = next(ds.runs())
    det = myrun.Detector('epix100')
    for nevt,evt in enumerate(myrun.events()):
        print('*** event',nevt,det.raw.raw(evt).shape,det.raw.calib(evt).shape,det.raw.image(evt))

def issue_2022_03_01():
    """O'Grady, Paul Christopher <cpo@slac.stanford.edu> Mon 2/28/2022 11:44 PM
    Hi Mikhail, Today Matt upgraded the firmware for the ued epix camera.
    At the moment det.raw.calib works but det.raw.image returns None for exp=detdaq02,run=569,
    which suggests there is no geometry.  I?m guessing this may have happened because
    the firmware version is part of the detector ?unique id? and the firmware changed today.
    You can see that ?detnames -i? returns different id?s for runs 500 (a while ago) and 569 (today)
    """
    from psana import DataSource
    ds = DataSource(exp='ueddaq02',run=569)
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')
    for nevent,evt in enumerate(myrun.events()):
        print(det.raw.calib(evt).shape,det.raw.image(evt))
        if nevent>10: break

def issue_2022_03_02():
    """epix100 default geometry implementation
    """
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource

    #ds = DataSource(exp='tmox49720',run=209)
    #orun = next(ds.runs())
    #det = orun.Detector('epix100')

    ds = DataSource(exp='rixx45619',run=119)
    orun = next(ds.runs())
    det = orun.Detector('epixhr')

    for nevt,evt in enumerate(orun.events()):
        geotxt = det.raw._det_geotxt_default()
        print('_det_geotxt_default:\n%s' % geotxt)
        o = GeometryAccess()
        o.load_pars_from_str(geotxt)
        x,y,z = o.get_pixel_coords()
        print(info_ndarr(x,'x:'))
        if det.raw.image(evt) is not None: break


def issue_2022_03_08():
    """test generic access to calibconst for the detector interface
    copy xtc2 file to .../public01/xtc/
    cd /cds/data/psdm/prj/public01/xtc/
    cp /cds/data/psdm/tmo/tmoc00318/xtc/tmoc00318-r0010-s000-c000.xtc2 .
    sudo chown psdatmgr tmoc00318-r0010-s000-c000.xtc2
    the same for smalldata/
    """
    from psana import DataSource
    ds = DataSource(exp='tmoc00318',run=10, dir='/cds/data/psdm/prj/public01/xtc')
    orun = next(ds.runs())
    det = orun.Detector('epix100')
    peds, meta = det.calibconst['pedestals']
    print('det.calibconst["pedestals"] constants\n', peds, '\nmetadata\n', meta)


def issue_2022_03_16():
    """Uervirojnangkoorn, Monarin <monarin@slac.stanford.edu> Wed 3/16/2022 12:25 PM
    Hi Mikhail, I mentioned that I tried to use calibcfg and calibtab for amox27716 run 85.
    I can see the contents of both variables from det.calibconst but they are (I believe)
    not compatible for DLDProcess.
    Here?s the script that works (this script uses calibcfg and calibtab from txt files)
    /cds/home/m/monarin/psana-nersc/psana2/dgrampy/ex-01-conv-raw-to-fex-wf.py
    If you uncomment line 144 (kwargs.update(cc)), you will see the error that I showed earlier.
    The cc has calibcfg and calibtab contents.
    My question is Is there a way to pass calibcfg and calibtab contents instead of the file paths.
    Thank you in advance!
    Mona
    """
    from psana import DataSource

    ds    = DataSource(files='/sdf/group/lcls/ds/ana/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e001000.xtc2')
    orun  = next(ds.runs())
    det   = orun.Detector('tmo_quadanode')
    cc    = det.calibconst

    print('cc.keys():', cc.keys())
    for k in cc.keys():
        data, meta = cc[k]
        print(k, 'meta:\n', meta)
        print(k, 'type(data):\n', type(data))
        print(k, 'data[:500]:\n', data[:500])

def issue_2022_04_06():
    """O'Grady, Paul Christopher <cpo@slac.stanford.edu> Wed 4/6/2022 6:01 PM
    Hi Mikhail,
    I started looking at this, but I think you will be more efficient than me. since you are more expert on calibration.
    The script below has a det.raw.calib that always returns 0.
    I think it might be because all pixels are marked bad in pixel_status with values of 42
    (the corresponding pedestal run is 397 I believe).  A later run 463 (which uses pedestal run 420) is fine.
    So it feels like there is some issue generating the constants with run 397?
    Would you have time to see what is going wrong with the run 397 pedestals (I think)?
    If you don?t have time I will continue looking.  Thanks!
    chris
    """
    import psana
    import numpy as np
    expname = 'uedcom103'
    runnum = 419
    ds = psana.DataSource(exp=expname,run=runnum,detectors=['epixquad','epicsinfo'])
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')
    print(det.calibconst.keys(),det.calibconst)
    for nevent,evt in enumerate(myrun.events()):
        print(det.raw.calib(evt))
        break

def issue_2022_06_17():
    """Default common_mode parameters
    """
    import psana
    import numpy as np
    expname = 'uedcom103'
    runnum = 419
    ds = psana.DataSource(exp=expname,run=runnum,detectors=['epixquad','epicsinfo'])
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')
    #print(det.calibconst.keys(),det.calibconst)
    print(det.calibconst.keys())

    #cmpars, meta = det.calibconst['common_mode']
    cmpars = det.raw._common_mode()
    print('cmpars',cmpars)

    #for nevent,evt in enumerate(myrun.events()):
    #    print(det.raw.calib(evt))
    #    break


def issue_2022_07_12():
    """
    """
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    ds = DataSource(exp='uedcom103',run=7, dir='/cds/data/psdm/prj/public01/xtc')
    orun = next(ds.runs())
    det = orun.Detector('epixquad')
    peds, meta = det.calibconst['pedestals']
    print('\nmetadata\n', meta)
    print(info_ndarr(peds, '\npedestals'))

    #mask = test_mask_select(tname, det)  # [0,:]
    evt = next(orun.events())

    print(info_ndarr(det.raw._mask_from_status(), 'det.raw._mask_from_status'))
    print(info_ndarr(det.raw._mask_comb(), 'det.raw._mask_comb'))

    #for nevent,evt in enumerate(orun.events()):
    #    print('=========', info_ndarr(det.raw.calib(evt), '\nEvt %03d det.raw.calib' % nevent))
    #    if nevent > 3: break


def issue_2022_11_17():
    """O'Grady, Paul Christopher <cpo@slac.stanford.edu>
    Dubrovin, Mikhail, Ulmer, Anatoli
    Hi Mikhail,
    Anatoli (cc'd) noticed that det.raw.image() for LCLS2 epix100 seems to be broken.
    It breaks for me in both the production release (4.4.11) and our latest development release.
    I believe it works in 4.4.10.
    My quick guess is that this is perhaps caused by the changes that have been made
    to make psana2 work at S3DF?
    Would you be able to have a look? Thanks! chris
    """
    import psana as ps
    ds = ps.DataSource(exp='tmolv1720', run=[180], detectors=['epix100', 'hsd', 'timing'])
    ds_run = next(ds.runs())
    epix_det = ds_run.Detector('epix100')
    evt = next(ds_run.events())
    raw = epix_det.raw.raw(evt)
    image = epix_det.raw.image(evt, nda=raw)

    from psana.detector.NDArrUtils import info_ndarr
    print(info_ndarr(raw,   'raw  '))
    print(info_ndarr(image, 'image'))

    calib = epix_det.raw.calib(evt)
    print(info_ndarr(calib, 'calib'))



def issue_2022_01_dd():
    print('template')

USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel-e.g.-DEBUG-or-INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - issue_2022_01_21 - test epix100hw raw, calib, image'\
      + '\n    2 - issue_2022_01_26 - test epix100 raw, calib, image and calib constants'\
      + '\n    3 - issue_2022_02_08 - test copy xtc2 file to .../public01/xtc/, epix100 common mode timing'\
      + '\n    4 - issue_2022_02_15 - test epix100 cpo - missing geometry'\
      + '\n    5 - issue_2022_03_01 - test epixquad cpo - copy constants'\
      + '\n    6 - issue_2022_03_02 - test epix100 - default geometry'\
      + '\n    7 - issue_2022_03_08 - test det.calibconst as generic access to cc'\
      + '\n    8 - issue_2022_03_16 - test calibconst for Mona'\
      + '\n    9 - issue_2022_04_06 - test epixquad cpo - constants'\
      + '\n   10 - issue_2022_06_17 - test default common_mode parameters'\
      + '\n   11 - issue_2022_07_12 - test det.raw.calib for refactored code for CalibConstants'\
      + '\n   12 - issue_2022_11_17 - test det.raw.calib epix100 object has no attribute _gain_factor_'\


TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('0',): issue_2022_01_dd()
elif TNAME in  ('1',): issue_2022_01_21()
elif TNAME in  ('2',): issue_2022_01_26()
elif TNAME in  ('3',): issue_2022_02_08()
elif TNAME in  ('4',): issue_2022_02_15()
elif TNAME in  ('5',): issue_2022_03_01()
elif TNAME in  ('6',): issue_2022_03_02()
elif TNAME in  ('7',): issue_2022_03_08()
elif TNAME in  ('8',): issue_2022_03_16()
elif TNAME in  ('9',): issue_2022_04_06()
elif TNAME in ('10',): issue_2022_06_17()
elif TNAME in ('11',): issue_2022_07_12()
elif TNAME in ('12',): issue_2022_11_17()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

#if __name__ == "__main__":
