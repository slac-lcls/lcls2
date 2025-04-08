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
       datinfo -k exp=rixx1016923,run=119 -d archon
       datinfo -k exp=rixx1017523,run=395 -d archon - (1, 4800) 20kevts
       datinfo -k exp=rixx1017523,run=396 -d archon   (600, 4800) 1836 evts

      rhttps://pswww.slac.stanford.edu/lgbk/lgbk/rixx1017523/eLog
      ixx1017523, DARK runs: 393, 394, 396, 397, 401, 402, 403, 404, 406, 410, 411
    """
    import numpy as np
    from time import time
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage

    #ds, orun, det = ds_run_det(exp='rixc00121', run=140, detname='archon', dir='/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc')
    #ds = DataSource(exp='rixc00121',run=154, dir='/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc',detectors=['archon']) # raw shape=(1200,4800), >200 evts
    #ds = DataSource(exp='rixx1016923',run=118, detectors=['archon'])
    #ds = DataSource(exp='rixx1016923',run=119, detectors=['archon'])
    #ds = DataSource(exp='rixx1017523',run=395, detectors=['archon']) # (1, 4800)
    #ds = DataSource(exp='rixx1017523',run=396, detectors=['archon']) # (600, 4800)
    ds = DataSource(exp='rixx1017523',run=418, detectors=['archon']) # (600, 4800)
    orun = next(ds.runs())
    det = orun.Detector('archon', gainfact=1) # , cmpars=(1,0,0)) #(1,0,0))

    flimg = None
    events = 10
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

       #img = np.array(raw)[0,:]
       #img.shape = (1,4800)
       #img,  title  = det.raw._calibconst['pedestals'][0], 'pedestals'
       #img, title  = det.raw.raw(evt), 'raw'
       #img, title  = det.raw.calib(evt), 'calib'
       img, title  = det.raw.image(evt), 'image'
       img = np.copy(img)

       #img = (raw.copy()/1000).astype(dtype=np.float64) # np.uint16)
       #img = clb
       #img = det.raw._mask_fake(raw.shape)
       #img = det.raw._arr_to_image(clb)

       dt_sec = (time() - t0_sec)*1000
       print(info_ndarr(img, 'evt:%3d dt=%.3f msec  image' % (nev, dt_sec), last=10))
       #print('det.raw._tstamp_raw(raw): ', det.raw._tstamp_raw(raw))

       if img.ndim==2 and img.shape[0] == 1:
           img = np.stack(1000*tuple(img))

       #img[0:100,0:100] = 2
       #print(info_ndarr(img, 'img ', last=10))

       #np.save('archon_raw.npy', img)
       #fraclo, frachi, fracme = 0.1, 0.9, 0.5
       #arr = img[0,:]
       #qlo = np.quantile(arr, fraclo, method='linear')
       #qhi = np.quantile(arr, frachi, method='linear')
       #qme = np.quantile(arr, fracme, method='linear')
       #print('qlo, qhi, qme, mean, max, min:', qlo, qhi, np.median(arr), np.mean(arr), np.max(arr), np.min(arr))


       if flimg is None:
          flimg = fleximage(img, h_in=5, w_in=20) # arr=arr_img)#, amin=0, amax=20), nneg=1, npos=3

       flimg.update(img)
       flimg.fig.suptitle('Event %d: %s' % (nev, title), fontsize=16)
       gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
       gr.show(mode='DO NOT HOLD')

    gr.show()


def issue_2025_01_31():
    """datinfo -k exp=mfxdaq23,run=4,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc -d jungfrau
    """
    import psana
    from psana.detector.NDArrUtils import info_ndarr
    from psana.detector.UtilsGraphics import gr, fleximage

    flimg = None

    #ds = psana.DataSource(exp="mfxdaq23", run=4, dir="/cds/data/drpsrcf/mfx/mfxdaq23/xtc")
    ds = psana.DataSource(exp='mfxdaq23', run=7, dir='/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc')
    run = next(ds.runs())
    evt = next(run.events())
    det = run.Detector('jungfrau')  # jungfrau  | jungfrauemu | raw       | 0_1_0
    #det.raw._path_geo_default = 'pscalib/geometry/data/geometry-def-jungfrau4M.data'
    #print('det.raw._path_geo_default', det.raw._path_geo_default)

    segs = det.raw._segments(evt)
    raw = det.raw.raw(evt) #####(8, 512, 1024)

    print(info_ndarr(raw, 'raw', last=10))
    print('det.raw._uniqueid', det.raw._uniqueid)
    print('det.raw._dettype', det.raw._dettype)
    print('det.raw._det_name', det.raw._det_name)
    print('det.raw._seg_geo.shape():', det.raw._seg_geo.shape())
    print('det.raw._sorted_segment_inds', det.raw._sorted_segment_inds)
    print('det.raw._segment_numbers', det.raw._segment_numbers)
    #print('det.raw._calibconst.keys()', det.raw._calibconst.keys())
    print(info_ndarr(raw, 'raw', last=10))

    img = det.raw.image(evt)
    print(info_ndarr(img, 'image', last=10))


    if flimg is None:
        flimg = fleximage(img, h_in=5, w_in=16, nneg=1, npos=3) # arr=arr_img

    flimg.update(img)
    flimg.fig.suptitle('Event XX: det.raw.raw', fontsize=16)
    gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
    gr.show(mode='DO NOT HOLD')

    gr.show()


def issue_2025_02_05():
    """fleximage does not show image"""
    #import numpy as np
    from psana.detector.NDArrUtils import info_ndarr
    #from psana.detector.UtilsGraphics import np, gr, fleximage
    import psana.detector.UtilsGraphics as ug
    np, gr, fleximage = ug.np, ug.gr, ug.fleximage
    fname = 'archon_raw.npy'
    arr = np.load(fname)
    print(arr)
    print(info_ndarr(arr, 'arr from %s' % fname, last=10))
    fraclo, frachi, fracme = 0.1, 0.9, 0.5
    qlo = np.quantile(arr, fraclo, method='linear')
    qhi = np.quantile(arr, frachi, method='linear')
    qme = np.quantile(arr, fracme) #, method='nearest')
    print('qlo, qhi, qme, mean, max, min:', qlo, qhi, np.median(arr), np.mean(arr), np.max(arr), np.min(arr))


def issue_2025_02_21():
    """Access to jungfrau panel configuration object

       mfxdaq23 > ascdaq023
       datinfo -k exp=ascdaq023,run=31 -d jungfrau # ,dir=/sdf/data/lcls/drpsrcf/ffb/asc/ascdaq023/xtc
       datinfo -k exp=ascdaq023,run=31 -d jungfrau # ,dir=/sdf/data/lcls/ds/asc/ascdaq023/xtc

    [dorlhiac@drp-det-cmp003 ~/cnfs]$> config_dump exp=ascdaq023,run=31,dir=/cds/home/d/dorlhiac/xtc jungfrau config
    config.expert.PauseThreshold: 16
    config.expert.TriggerDelay: 42
    config.firmwareBuild: Lcls2XilinxKcu1500Udp_10GbE: Vivado v2023.1, rdsrv403 (Ubuntu 22.04.5 LTS), Built Thu Feb  6 04:55:59 PM PST 2025 by ruckman
    config.firmwareVersion: 50331648
    config.user.bias_voltage_v: 200
    config.user.exposure_period: 0.2
    config.user.exposure_time_s: 1e-05
    config.user.gain0.value: 0
    config.user.gainMode.value: 3
    config.user.jungfrau_ip: 10.0.0.15
    config.user.jungfrau_mac: aa:bb:cc:dd:ee:ff
    config.user.kcu_ip: 10.0.0.10
    config.user.kcu_mac: 08:00:56:00:00:00
    config.user.port: 32410
    config.user.speedLevel.value: 1
    config.user.trigger_delay_s: 0.000238
    """

    import psana
    from psana.detector.NDArrUtils import info_ndarr
    from psana.detector.UtilsGraphics import gr, fleximage

    from psana.psexp.utils import DataSourceFromString
    import numpy as np
    import argparse

    def dump(obj, attrlist):
      allattrs = dir(obj)
      usefulattrs=[attr for attr in allattrs if (not attr.startswith('_') and attr != 'help')]
      for attr in usefulattrs:
        val = getattr(obj, attr)
        attrlist.append(attr)
        if type(val) in [int, float, np.ndarray, str]:
            print('.'.join(attrlist)+':', val)
        elif type(val) is psana.container.Container:
            dump(val, attrlist)
        attrlist.pop(-1)

    #ds = psana.DataSource(exp='ascdaq023', run=31, dir='/sdf/data/lcls/ds/asc/ascdaq023/xtc')
    ds = psana.DataSource(exp='ascdaq023', run=34, dir='/sdf/data/lcls/ds/asc/ascdaq023/xtc')

    run = next(ds.runs())
    evt = next(run.events())
    det = run.Detector('jungfrau')

    print('dir(det.raw):', dir(det.raw))
    print('det.raw._segment_ids():', det.raw._segment_ids())
    print('det.raw._segment_numbers:', det.raw._segment_numbers)
    ind = 0
    c0 = det.raw._seg_configs()[ind].config
    print('dir(det.raw._seg_configs()[ind].config):', dir(c0))

    cfg0_user = det.raw._seg_configs()[ind].config.user
    print('dir(user):', dir(cfg0_user))

    attrs = [a for a in dir(cfg0_user) if (not a.startswith('_') and a != 'help')]
    #attrs = ['bias_voltage_v', 'exposure_period', 'exposure_time_s', 'gain0', 'gainMode',\
    #          'jungfrau_ip', 'jungfrau_mac', 'kcu_ip', 'kcu_mac', 'port', 'speedLevel', 'trigger_delay_s']
    for a in attrs:
        v = getattr(cfg0_user, a)
        sa,sv = (a+'.value', str(v.value)) if type(v) is psana.container.Container else (a, str(v))
        print('   user.%s: %s' % (sa, sv))
    #dump(cfg0_user, attrs)

    print('gain0   :', cfg0_user.gain0.value)
    print('gainMode:', cfg0_user.gainMode.value)

    scfgs = det.raw._seg_configs_user()


def issue_2025_02_25():
    """post(url, headers=headers, data=d) does not save big array like (3,16,512,1024) float32...
    """
    print('\n\nLOAD BIG DATA FROM FILE\n')
    from psana.detector.NDArrUtils import info_ndarr
    from psana.pscalib.calib.NDArrIO import load_txt
    fname = '/sdf/home/d/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/detector/'\
            'work/jungfrauemu/merge_tmp/jungfrauemu_000001-20250203095124-mfxdaq23-r0007-pedestals.txt'
    print('load nda from: %s' %fname )
    nda = load_txt(fname)  # nda shape:(3, 16, 512, 1024) size:25165824 dtype:float32

    #nda = nda[:,1,:,:]
    print(info_ndarr(nda, 'nda'))

    print('\n\nPUSH BIG DATA TO DB\n')
    import io
    from krtc import KerberosTicket
    from urllib.parse import urlparse
    import psana.pscalib.calib.MDBUtils as mu
    from requests import get, post, delete

    URL_KRB = 'https://pswww.slac.stanford.edu/ws-kerb/calib_ws/'
    KRBHEADERS = KerberosTicket("HTTP@" + urlparse(URL_KRB).hostname).getAuthHeaders()

    headers = dict(KRBHEADERS)
    headers['Content-Type'] = 'application/octet-stream'
    f = io.BytesIO(nda.tobytes())
    d = f.read()

    url = 'https://pswww.slac.stanford.edu/ws-kerb/calib_ws/cdb_test/gridfs/'
    print('url:', url)
    print('headers:', headers)
    resp = post(url, headers=headers, data=d)
    print('post resp.text:', resp.text)


def issue_2025_02_27():
    """det_dark_proc -d archon -k exp=rixx1017523,run=393 -D -o work issues
    """
    import psana
    ds = psana.DataSource(exp='rixx1017523', run=393) #, dir='/sdf/data/lcls/ds/asc/ascdaq023/xtc')
    orun = next(ds.runs())
    odet = orun.Detector('archon')
    v = getattr(odet.raw,'_segment_ids', None) # odet.raw._segment_ids()
    v = None if v is None else v()
    print('odet.raw._segment_ids():', v)
    print('odet.raw._sorted_segment_inds', odet.raw._sorted_segment_inds)
    print('dir(odet)', dir(odet))
    print('dir(odet.raw)', dir(odet.raw))
    print('odet.calibconst.keys()', odet.calibconst.keys())
    print('odet.raw._calibconst.keys()', odet.raw._calibconst.keys())


def issue_2025_03_06():
    """jungfrau
       datinfo -k exp=ascdaq023,run=37 -d jungfrau
       jungfrau_dark_proc -k exp=ascdaq023,run=37 -d jungfrau
    """
    import numpy as np
    from time import time
    from psana.detector.NDArrUtils import info_ndarr, divide_protected, reshape_to_2d
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    from psana.detector.UtilsJungfrau import info_gainbits

    ds = DataSource(exp='ascdaq023',run=37) #, detectors=['jungfrau']) # (600, 4800)
    orun = next(ds.runs())
    det = orun.Detector('jungfrau', gainfact=1) # , cmpars=(1,0,0)) #(1,0,0))

    flimg = None
    events = 10
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

       #print('==== evt/sel: %4d/%4d' % (nev,evsel))

       t0_sec = time()

       #img,  title  = det.raw._calibconst['pedestals'][0], 'pedestals'
       #img, title  = det.raw.raw(evt), 'raw'
       img, title  = det.raw.calib(evt, cmpars=(1,7,1000)), 'calib'
       #img, title  = det.raw.image(evt), 'image'

       dt_sec = (time() - t0_sec)*1000
       print(info_ndarr(img, '==== evt/sel:%3d/%3d dt=%.3f msec  image' % (nev, evsel, dt_sec), last=10))
       #print('det.raw._tstamp_raw(raw): ', det.raw._tstamp_raw(raw))

       print('     info_gainbits', info_gainbits(raw))

       if img.ndim==2 and img.shape[0] == 1:
           img = np.stack(1000*tuple(img))

       #np.save('archon_raw.npy', img)
       #fraclo, frachi, fracme = 0.1, 0.9, 0.5
       #arr = img[0,:]
       #qlo = np.quantile(arr, fraclo, method='linear')
       #qhi = np.quantile(arr, frachi, method='linear')
       #qme = np.quantile(arr, fracme, method='linear')
       #print('qlo, qhi, qme, mean, max, min:', qlo, qhi, np.median(arr), np.mean(arr), np.max(arr), np.min(arr))
       img = reshape_to_2d(img)

       if flimg is None:
          flimg = fleximage(img, h_in=5, w_in=20) # arr=arr_img)#, amin=0, amax=20), nneg=1, npos=3

       flimg.update(img)
       flimg.fig.suptitle('Event %d: %s' % (nev, title), fontsize=16)
       #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
       gr.show(mode='DO NOT HOLD')

    gr.show()


def issue_2025_03_18():
    """Silke - direct access to calibration constants
       datinfo -k exp=ued1006477,run=15 -d epixquad
    """
    from psana.pscalib.calib.MDBWebUtils import calib_constants
    from psana import DataSource
    ds = DataSource(exp='ued1006477',run=15)
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')
    print('\n\n\n  det.calibconst:',det.calibconst['pedestals'][0].shape)

    detrawid = det.raw._uniqueid
    print('\n\n\n  det.raw._uniqueid:', detrawid)
    peds = calib_constants(detrawid, exp='ued1006477', ctype="pedestals", run=15)[0]
    print('calib_constants uniqueid:',peds.shape)
    import psana.detector.UtilsCalib as uc
    shortname = uc.detector_name_short(detrawid)
    print('\n\n\n  shortname', shortname)
    peds = calib_constants(shortname, exp='ued1006477', ctype="pedestals", run=15)[0]
    print('calib_constants shortname',peds.shape)
    # eventually silke would like to get the constants for another run like this:
    #peds = calib_constants(shortname, exp='ued1006477', ctype="pedestals", run=17)[0]
    #print('calib_constants shortname',peds)

 
def issue_2025_03_19():
    """Silke -  It appears to me that she is picking up constants from February 12th for uedc00104 and we don?t understand why
       datinfo -k exp=uedc00104,run=177 -d epixquad
       REASON: recent mess with detector names, in passing as metadata kwa['detector'] = detname, should be shortname
    """
    from psana.pscalib.calib.MDBWebUtils import calib_constants
    from psana import DataSource
    ds = DataSource(exp='uedc00104',run=177)
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')
    print('det.calibconst:',det.calibconst['pedestals'][1])


def issue_2025_03_27():
    """direct access to calibration constants for jungfrau16M
       datinfo -k exp=ascdaq023,run=37 -d jungfrau
    """
    from time import time
    from psana.pscalib.calib.MDBWebUtils import calib_constants
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource

    t0_sec = time()
    ds = DataSource(exp='ascdaq023',run=37)
    myrun = next(ds.runs())
    det = myrun.Detector('jungfrau')
    print('\n\ntime for det.calibconst (sec): %.3f' % (time()-t0_sec))
    print('det.calibconst:',det.calibconst['pedestals'][0].shape)

    shortname = 'jungfrau_000001' # uc.detector_name_short(detrawid)
    print('\n\nshortname', shortname)
    t0_sec = time()
    nda = calib_constants(shortname, exp='ascdaq023', ctype='pedestals', run=37)[0] # 'pixel_status'
    print('time for calib_constants (sec): %.3f' % (time()-t0_sec))
    print(info_ndarr(nda,'calib_constants shortname', last=10))
#    print(info_ndarr(nda[1,1,:],'nda[1,1,:]', last=10))


def issue_2025_03_28():
    """see cpo email
       datinfo -k exp=mfxdaq23,run=11,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfxdaq23/xtc/ -d jungfrau
       datinfo -k exp=mfxdaq23,run=11,dir=/sdf/data/lcls/ds/mfx/mfxdaq23/xtc/ -d jungfrau
       datinfo -k exp=mfxdaq23,run=11 -d jungfrau
    """
    from psana import DataSource
    ds = DataSource(exp='mfxdaq23',run=11,dir='/sdf/data/lcls/ds/mfx/mfxdaq23/xtc')
    myrun = next(ds.runs())
    det = myrun.Detector('jungfrau')
    for nevt,evt in enumerate(myrun.events()):
      print('evt:', nevt)
      print('    raw  :', det.raw.raw(evt).shape)
      print('    calib:', det.raw.calib(evt).shape)
      print('    image:', det.raw.image(evt).shape)
      if nevt>4: break


def issue_2025_04_02():
    """
    """
    import os
    #from psana.pscalib.geometry.GeometryAccess import GeometryAccess, img_from_pixel_arrays
    import psana.pscalib.geometry.GeometryAccess as ga # import GeometryAccess, img_from_pixel_arrays
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d
    import psana.pyalgos.generic.NDArrGenerators as ag

    SCRDIR = os.path.dirname(os.path.realpath(__file__))

    #fname_geo = os.path.join(SCRDIR, '../pscalib/geometry/data/geometry-def-jungfrau16M.data')
    #fname_geo = os.path.join(SCRDIR, '../pscalib/geometry/data/geometry-def-epixuhr.data')
    fname_geo = os.path.join(SCRDIR, '../pscalib/geometry/data/geometry-def-epixm320.data')
    logger.info('fngeo: %s' % fname_geo)
    assert os.path.exists(fname_geo)

    geo = ga.GeometryAccess(fname_geo)
    rows, cols = geo.get_pixel_coord_indexes()

    print(ndu.info_ndarr(rows, 'rows'))
    sh3d = ndu.shape_nda_as_3d(rows) # i.e. (1,1,512,1024) > (1,512,1024)
    rows.shape = cols.shape = sh3d
    print(ndu.info_ndarr(rows, 'rows'))
    arr = ag.arr3dincr(sh3d)
    arr1 = ag.arr2dincr(sh3d[1:]) # gg.np.array(arr[0,:])
    for n in range(sh3d[0]):
        arr[n,:] += (10+n)*arr1

    img = ga.img_from_pixel_arrays(rows, cols, W=arr)

    if False:
        import psana.pyalgos.generic.Graphics as gg # for test purpose
        gg.plotImageLarge(img) #, amp_range=amp_range)
        gg.move(500,10)
        gg.show()
        #gg.save_plt(fname='img.png')

    if True:
        flimg = None
        from psana.detector.UtilsGraphics import gr, fleximage

        if flimg is None:
          flimg = fleximage(img, h_in=11, w_in=11) # arr=arr_img)#, amin=0, amax=20), nneg=1, npos=3
        else:
          flimg.update(img)
        flimg.fig.suptitle('test of geometry', fontsize=16)
        #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
        # gr.show(mode='DO NOT HOLD')

        gr.show()


def issue_2025_04_03(args):
    """https://confluence.slac.stanford.edu/spaces/LCLSIIData/pages/267391733/psana#psana-PublicPracticeData
       datinfo -k exp=ued1010667,run=181,dir=/sdf/data/lcls/ds/prj/public01/xtc -d epixquad
       shape:(4, 352, 384)
    """
    import os
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    ds = DataSource(exp='ued101066', run=181, dir='/sdf/data/lcls/ds/prj/public01/xtc')
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')

    if True:
        det_geo = det.raw._det_geo()
        seg_geo = det.raw._seg_geo
        top_geo = det_geo.get_top_geo()
        #print('det_geo:%s\n%s' % (type(det_geo), str(dir(det_geo))))
        #print('seg_geo:%s\n%s' % (type(seg_geo), str(dir(seg_geo))))
        #print('top_geo:%s\n%s' % (type(top_geo), str(dir(top_geo))))

        print(ndu.info_ndarr(seg_geo.pixel_size_array(axis='X'), 'seg_geo.pixel_size_array(axis="X")'))
        print(ndu.info_ndarr(seg_geo.pixel_size_array(axis='Y'), 'seg_geo.pixel_size_array(axis="Y")'))
        print(ndu.info_ndarr(det_geo.get_pixel_coords(), 'det_geo.get_pixel_coords'))

        def info_recurs_geo(geo):
            geo.print_geo()
            for o in geo.get_list_of_children():
                info_recurs_geo(o)

        info_recurs_geo(top_geo)


    if False:
        flimg = None
        for nevt,evt in enumerate(myrun.events()):
            raw   = det.raw.raw(evt)
            calib = det.raw.calib(evt)
            img   = det.raw.image(evt)
            print('evt:', nevt)
            print('    raw  :', raw.shape)
            print('    calib:', calib.shape)
            if nevt>10: break
            if flimg is None:
                flimg = fleximage(img, h_in=11, w_in=11)
            print('    image:', img.shape)
            flimg.update(img)
            flimg.fig.suptitle('test of geometry', fontsize=16)
            #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
            # gr.show(mode='DO NOT HOLD')
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
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
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
    elif TNAME in  ('2',): issue_2025_01_31() # emulated jungfrauemu
    elif TNAME in  ('3',): issue_2025_02_05() # fleximage does not show image
    elif TNAME in  ('4',): issue_2025_02_21() # access to jungfrau panel configuration object
    elif TNAME in  ('5',): issue_2025_02_25() # test saving BIG 32-segment (3,16,512,1024) float32 jungfrau calib constants in DB
    elif TNAME in  ('6',): issue_2025_02_27() # det_dark_proc -d archon -k exp=rixx1017523,run=393 -D -o work issue
    elif TNAME in  ('7',): issue_2025_03_06() # jungfrau
    elif TNAME in  ('8',): issue_2025_03_18() # Silke - direct access to calibration constants
    elif TNAME in  ('9',): issue_2025_03_19() # Silke - picking up wrong constants from Feb 12
    elif TNAME in ('10',): issue_2025_03_27() # me - direct access to calibration constants for jungfrau16M
    elif TNAME in ('11',): issue_2025_03_28() # cpo - jungfrau issues
    elif TNAME in ('12',): issue_2025_04_02() # Aaron Brewster - acces to jungfrau geometry from file
    elif TNAME in ('13',): issue_2025_04_03(args) # Aaron Brewster - acces to jungfrau geometry from det._calibconst
    else:
        print(USAGE())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%TNAME)

    exit('END OF TEST %s'%TNAME)


if __name__ == "__main__":
    selector()

# EOF
