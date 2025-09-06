#!/usr/bin/env python
"""./lcls2/psana/psana/detector/test_issues_2024.py <TNAME>"""

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
    #ds = DataSource(exp='ascdaq023',run=37)
    ds = DataSource(exp='mfx101332224',run=66)
    myrun = next(ds.runs())
    det = myrun.Detector('jungfrau')
    print('\n\ntime for det.calibconst (sec): %.3f' % (time()-t0_sec))
    print('det.calibconst.keys:',det.calibconst.keys())
    print('det.calibconst:',det.calibconst['pedestals'][0].shape)

    shortname = 'jungfrau_000003' # uc.detector_name_short(detrawid)
    print('\n\nshortname', shortname)
    t0_sec = time()
    #nda = calib_constants(shortname, exp='ascdaq023', ctype='pedestals', run=37)[0] # 'pixel_status'
    nda = calib_constants(shortname, exp='mfx101332224', ctype='pedestals', run=66)[0] # 'pixel_status'
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


def issue_2025_04_03():
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


def issue_2025_04_09():
    """Philip: det.calibconst.keys()
          dict_keys(['pixel_status', 'pedestals', 'pixel_min', 'pixel_max', 'pixel_rms'])
       datinfo -k exp=mfx101332224,run=15 -d jungfrau
    """
    from psana import DataSource
    ds = DataSource(exp='mfx101332224',run=15)
    myrun = next(ds.runs())
    det = myrun.Detector('jungfrau')
    print('det.calibconst.keys():', det.calibconst.keys())
    gain = det.calibconst['pixel_gain'][0]
    import psana.pyalgos.generic.NDArrUtils as ndu
    print(ndu.info_ndarr(gain, 'gain:', last=10))


def issue_2025_04_10():
    """epixquad det.raw.image timing
       datinfo -k exp=ued1010667,run=181,dir=/sdf/data/lcls/ds/prj/public01/xtc -d epixquad
       shape:(4, 352, 384)

       export OPENBLAS_NUM_THREADS=1
       echo $OPENBLAS_NUM_THREADS

       by default, OPENBLAS_NUM_THREADS=1
       median dt, msec: 16.220, 16.394, 16.230, 16.623, 16.154
       export OPENBLAS_NUM_THREADS=0
       median dt, msec: 16.423, 16.230, 16.477, 16.071, 16.722
    """
    import os
    import numpy as np
    from time import time
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    ds = DataSource(exp='ued101066', run=181, dir='/sdf/data/lcls/ds/prj/public01/xtc')
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')
    events = 100
    arrdt = np.empty(events, dtype=np.float64)
    if True:
        flimg = None
        for nevt,evt in enumerate(myrun.events()):
            if nevt>events-1: break
            raw   = det.raw.raw(evt)
            calib = det.raw.calib(evt)
            t0_sec = time()
            img   = det.raw.image(evt)
            dt_sec = (time() - t0_sec)*1000
            #print('evt:', nevt)
            arrdt[nevt] = dt_sec
            print('evt:%3d dt=%.3f msec  raw+calib+image' % (nevt, dt_sec))
            print('    raw  :', raw.shape)
            print('    calib:', calib.shape)
            if flimg is None:
                flimg = fleximage(img, h_in=11, w_in=11)
            print('    image:', img.shape)
            flimg.update(img)
            flimg.fig.suptitle('test of geometry', fontsize=16)
            #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
            # gr.show(mode='DO NOT HOLD')
        gr.show()
        print(ndu.info_ndarr(arrdt, 'arrdt', last=events))
        print('median dt, msec: %.3f' % np.median(arrdt))


def issue_2025_04_11():
    """access to multiple calibration constants
       datinfo -k exp=mfx101332224,run=7 -d epix100
    """
    #from psana.pscalib.calib.MDBWebUtils import calib_constants
    from psana.pscalib.calib.MDBUtils import timestamp_id, sec_and_ts_from_id
    from psana import DataSource
    ds = DataSource(exp='mfx101332224',run=7)
    myrun = next(ds.runs())
    det = myrun.Detector('epix100')
    cc = det.calibconst['pedestals']
    print('\n\n== pedestals:', cc[0].shape)
    print('\n\n== matadata:',  cc[1])
    id_doc = cc[1]['id_data'] # '_id']
    print('id_doc', id_doc)
    print('tsDB', timestamp_id(id_doc))
    print('id -> sec:', sec_and_ts_from_id(id_doc))


def make_random_nda(shape=(704, 768), mu=100, sigma=10, fname='fake.npy'):
    import numpy as np
    #import psana.pyalgos.generic.NDArrGenerators as ag
    import psana.pyalgos.generic.NDArrUtils as ndu
    a = mu + sigma*np.random.standard_normal(size=shape).astype(dtype=np.float64)
    print(ndu.info_ndarr(a, 'save %s:'%fname, last=10))
    np.save(fname, a)

def issue_2025_nda():
    for sh in ((704, 768), (512,1024)):
        make_random_nda(shape=sh, mu=2, sigma=0.1, fname='fake_%dx%d.npy' % sh)


def issue_2025_04_17():
    """shape: (32, 512, 1024)

       export OPENBLAS_NUM_THREADS=1
       echo $OPENBLAS_NUM_THREADS

       by default, OPENBLAS_NUM_THREADS=1
       median dt, msec: 13.677, 13.700

       export OPENBLAS_NUM_THREADS=0
       median dt, msec: 14.030

       export OPENBLAS_NUM_THREADS=10
       median dt, msec: 13.638
    """
    import os
    import numpy as np
    from time import time
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    shape = (32, 512, 1024)
    mu_g, sigma_g, dtype_g =   10,   1, np.float32
    mu_p, sigma_p, dtype_p = 1000,  10, np.float32
    mu,   sigma,   dtype   = 1100, 100, np.float32

    gain = mu_g + sigma_g*np.random.standard_normal(size=shape).astype(dtype=dtype_g)
    peds = mu_p + sigma_p*np.random.standard_normal(size=shape).astype(dtype=dtype_p)
    print(ndu.info_ndarr(gain, 'gain'))
    print(ndu.info_ndarr(peds, 'peds'))

    nloops = 100
    arrdt = np.empty(nloops, dtype=np.float32)

    for n in range(nloops):
        nda = mu + sigma*np.random.standard_normal(size=shape).astype(dtype=dtype)
        t0_sec = time()
        #result = nda-peds
        result = (nda-peds) * gain
        dt_sec = (time() - t0_sec)*1000
        arrdt[n] = dt_sec
        print('%02d dt, msec: %.3f %s' % (n, dt_sec, ndu.info_ndarr(result, 'result')))

    print('median dt, msec: %.3f' % np.median(arrdt))


def issue_2025_04_21():
    """The same as issue_2025_04_10, but for jungfrau 16M
       epixquad det.raw.image timing
       datinfo -k exp=mfx101332224,run=9999,dir=/sdf/data/lcls/ds/xpp/xpptut15/scratch/cpo -d jungfrau
       shape:(3,32,512,1024)

       export OPENBLAS_NUM_THREADS=1
       echo $OPENBLAS_NUM_THREADS

       by default, OPENBLAS_NUM_THREADS=1
       median dt, msec: 265.418
       export OPENBLAS_NUM_THREADS=0
       median dt, msec: 264.812
       export OPENBLAS_NUM_THREADS=10
       median dt, msec: 264.504
       export OPENBLAS_NUM_THREADS=64
       median dt, msec: 262.617
    """
    import os
    import numpy as np
    from time import time
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    ds = DataSource(exp='mfx101332224', run=9999, dir='/sdf/data/lcls/ds/xpp/xpptut15/scratch/cpo')
    myrun = next(ds.runs())
    det = myrun.Detector('jungfrau')
    events = 100
    arrdt = np.zeros(events, dtype=np.float64)
    if True:
        flimg = None
        for nevt,evt in enumerate(myrun.events()):
            if nevt>events-1: break
            raw   = det.raw.raw(evt)
            if raw is None: continue
            calib = det.raw.calib(evt)
            t0_sec = time()
            img   = det.raw.image(evt)
            dt_sec = (time() - t0_sec)*1000
            #print('evt:', nevt)
            arrdt[nevt] = dt_sec
            #print('evt:%3d dt=%.3f msec  raw+calib+image' % (nevt, dt_sec))
            print('evt:%3d dt=%.3f msec det.raw.image' % (nevt, dt_sec))
            print('    raw  :', raw.shape)
            print('    calib:', calib.shape)
            if flimg is None:
                flimg = fleximage(img, h_in=11, w_in=11)
            print('    image:', img.shape)
            flimg.update(img)
            flimg.fig.suptitle('test of geometry', fontsize=16)
            #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
            # gr.show(mode='DO NOT HOLD')
        gr.show()
        print(ndu.info_ndarr(arrdt, 'arrdt', last=events))
        print('median dt, msec: %.3f' % np.median(arrdt))


def issue_2025_04_22():
    """epixquad det.raw.calib/image - zeros
       datinfo -k exp=ued1006419,run=2 -d epixquad
       shape:(4, 352, 384)
    """
    import os
    import numpy as np
    from time import time
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    ds = DataSource(exp='ued1006419', run=2) #, dir='/sdf/data/lcls/ds/prj/public01/xtc')
    myrun = next(ds.runs())
    det = myrun.Detector('epixquad')
    events = 10
    arrdt = np.empty(events, dtype=np.float64)
    if True:
        flimg = None
        for nevt,evt in enumerate(myrun.events()):
            if nevt>events-1: break
            raw   = det.raw.raw(evt)
            calib = det.raw.calib(evt)
            t0_sec = time()
            img   = det.raw.image(evt)
            dt_sec = (time() - t0_sec)*1000
            #print('evt:%3d' % nevt)
            arrdt[nevt] = dt_sec
            #print('evt:%3d dt=%.3f msec  raw+calib+image' % (nevt, dt_sec))
            print(ndu.info_ndarr(calib, 'evt:%3d calib'% nevt, last=10))
            print('min: %f max: %f' % (np.min(calib),np.max(calib)))
            if flimg is None:
                flimg = fleximage(img, h_in=11, w_in=11)
            #print('    image:', img.shape)
            flimg.update(img)
            flimg.fig.suptitle('test of geometry', fontsize=16)
            #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
            gr.show(mode='DO NOT HOLD')
        gr.show()
        print(ndu.info_ndarr(arrdt, 'arrdt', last=events))
        print('median dt, msec: %.3f' % np.median(arrdt))



def issue_2025_04_23(DO_WITH_PUBLISH=True):
    """cpo - jungfrau16M image for 4 drp panels
       datinfo -k exp=mfx101332224,run=167 -d jungfrau
       shape:(4, 512, 1024)
    """
    from psana import DataSource
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    if DO_WITH_PUBLISH:
        from psmon.plots import Image
        from psmon import publish
        publish.local = True
        publish.plot_opts.aspect = 1

    ds = DataSource(exp='mfx101332224',run=167)
    myrun = next(ds.runs())
    jf = myrun.Detector('jungfrau')
    for nevent,evt in enumerate(myrun.events()):
        if nevent > 10: break
        raw = jf.raw.raw(evt)
        print(ndu.info_ndarr(raw, 'ev:%02d raw:' % nevent))
        if raw is None: continue
        image = jf.raw.image(evt)
        print(ndu.info_ndarr(image, '    image:'))

        if DO_WITH_PUBLISH:
            imgsend = Image(nevent,"Random",image)
            publish.send('image',imgsend)
            input("hit cr")

def issue_2025_04_29():
    """
       datinfo -k exp=mfx101332224,run=204 -d jungfrau
       shape:(19, 512, 1024)
    """
    from psana import DataSource
    ds = DataSource(exp='mfx101332224',run=204)
    myrun = next(ds.runs())
    det = myrun.Detector('jungfrau')
    for nevt,evt in enumerate(myrun.events()):
        print(det.raw.image(evt).shape)
        break


def issue_2025_05_07():
    """test of the detector axis, shape:(796, 6144)
       det_dark_proc -k exp=rix101333324,run=46 -d axis_svls
       datinfo -k exp=rix101333324,run=46 -d axis_svls
    """
    import os
    import numpy as np
    from time import time
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    ds = DataSource(exp='rix101333324', run=46)
    myrun = next(ds.runs())
    det = myrun.Detector('axis_svls')
    events = 5
    arrdt = np.empty(events, dtype=np.float64)
    if True:
        flimg = None
        for nevt,evt in enumerate(myrun.events()):
            if nevt>events-1: break
            raw   = det.raw.raw(evt)
            calib = det.raw.calib(evt)
            t0_sec = time()
            #img = det.raw.image(evt, nda=raw/2) # raw
            img = det.raw.image(evt, nda=None) # raw
            dt_sec = (time() - t0_sec)*1000
            #print('evt:', nevt)
            arrdt[nevt] = dt_sec
            print('evt:%3d dt=%.3f msec for det.raw.image(evt)' % (nevt, dt_sec))
            print(ndu.info_ndarr(raw,   '  raw  :'))
            print(ndu.info_ndarr(calib, '  calib:'))
            print(ndu.info_ndarr(img,   '  img  :'))
            if flimg is None:
                flimg = fleximage(img, h_in=2.5, w_in=15)
            flimg.update(img)
            flimg.fig.suptitle('evt: %d test of detector axis' % nevt, fontsize=16)
            #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
            gr.show(mode='DO NOT HOLD')
        gr.show()
        print(ndu.info_ndarr(arrdt, 'arrdt', last=events))
        print('median dt, msec: %.3f' % np.median(arrdt))


def issue_2025_05_14():
    """test QComboBox for control_gui"""

    from PyQt5.QtWidgets import QComboBox, QMainWindow, QApplication
    from PyQt5.QtCore import Qt
    import sys

    class CustomQComboBox(QComboBox):
        def __init__(self, parent=None):
            super().__init__(parent)

        def keyPressEvent(self, event):
            #print('event.key():', event.key())
            if event.key() in (Qt.Key_Up, Qt.Key_Down):
                event.ignore()  # Ignore up and down arrow keys
            else:
                super().keyPressEvent(event) # Default behavior for other keys

        def wheelEvent(self, event):
            event.ignore()
            #print('event:', dir(event), '\n')
            #print('event.angleDelta().y():', event.angleDelta().y(), '\n')
            #super().wheelEvent(event) # Default behavior
            #if event.angleDelta().y() in (120, -120):
            #    event.ignore()  # Ignore up and down arrow keys
            #else:
            #    super().keyPressEvent(event) # Default behavior for other keys

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            combobox = CustomQComboBox()
            combobox.addItems(['One', 'Two', 'Three', 'Four'])
            # Connect signals to the methods.
            combobox.activated.connect(self.activated)
            combobox.currentTextChanged.connect(self.text_changed)
            combobox.currentIndexChanged.connect(self.index_changed)

            self.setCentralWidget(combobox)

        def activated(Self, index):
            print("Activated index:", index)

        def text_changed(self, s):
            print("Text changed:", s)

        def index_changed(self, index):
            print("Index changed", index)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()


def issue_2025_05_16(fname='test-array.npy', USE_GZIP=True): #False):
    """test gzip.open save and load"""
    from time import time
    import numpy as np
    from psana.detector.NDArrUtils import info_ndarr

    if USE_GZIP:
        import gzip
        fname += '.gz'

    # Create a sample numpy array
    arr = np.random.rand(32,3,512,1024).astype(np.float32)
    print(info_ndarr(arr, 'random array'))

    t0_sec = time()
    #f = gzip.open(fname, 'wb')
    #np.save(f, arr)
    #f.close()
    if USE_GZIP:
      with gzip.open(fname, 'wb') as f:
        np.save(f, arr)
    else:
        np.save(fname, arr)

    print('dt=%.3f us USE_GZIP=%s save' % (time() - t0_sec, str(USE_GZIP)))
    # USE_GZIP=True:  8.836us - 176532 test-array.npy.gz
    # USE_GZIP=False: 0.232us - 196612 test-array.npy compression factor = 0.9

    t0_sec = time()
    #f = gzip.open(fname, 'rb')
    #loaded_arr = np.load(f)
    #f.close()

    loaded_arr = None
    if USE_GZIP:
      with gzip.open(fname, 'rb') as f:
        loaded_arr = np.load(f)
    else:
        loaded_arr = np.load(fname)
    print('dt=%.3f us USE_GZIP=%s load' % (time() - t0_sec, str(USE_GZIP)))
    # USE_GZIP=True: dt=1.286
    # USE_GZIP=False: dt=0.486 us
    # Verify that the loaded array is the same as the original array
    np.testing.assert_array_equal(arr, loaded_arr)


def issue_2025_06_05():
    """test for excessive output
       epixquad det.raw.calib/image - zeros
       datinfo -k exp=mfxdet23,run=120 -d epixuhr
    """
    import os
    import numpy as np
    from time import time
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    #ds = DataSource(exp='mfxdet23', run=120)
    ds = DataSource(exp='mfxdet23', run=9)
    myrun = next(ds.runs())
    det = myrun.Detector('epixuhr')
    events = 10
    arrdt = np.empty(events, dtype=np.float64)
    if True:
        flimg = None
        for nevt,evt in enumerate(myrun.events()):
            if nevt>events-1: break
            raw   = det.raw.raw(evt)
            if raw is None: continue
            calib = det.raw.calib(evt)
            t0_sec = time()
            img   = det.raw.image(evt)
            dt_sec = (time() - t0_sec)*1000
            #print('evt:%3d' % nevt)
            arrdt[nevt] = dt_sec
            #print('evt:%3d dt=%.3f msec  raw+calib+image' % (nevt, dt_sec))
            print(ndu.info_ndarr(calib, 'evt:%3d calib'% nevt, last=10))
            print('min: %f max: %f' % (np.min(calib),np.max(calib)))
            if flimg is None:
                flimg = fleximage(img, h_in=11, w_in=11)
            #print('    image:', img.shape)
            flimg.update(img)
            flimg.fig.suptitle('test of geometry', fontsize=16)
            #gr.save_fig(flimg.fig, fname='img_det_raw_raw.png', verb=True)
            gr.show(mode='DO NOT HOLD')
        gr.show()
        print(ndu.info_ndarr(arrdt, 'arrdt', last=events))
        print('median dt, msec: %.3f' % np.median(arrdt))


def issue_2025_06_06():
    """timing of jungfrau_dark_proc with split for steps and panels
       datinfo -k exp=mfx100852324,run=7 -d jungfrau
       sbatch -p milano --account lcls:mfx100852324 --mem 8GB --cpus-per-task 5 -o work1
       cmd: jungfrau_dark_proc -o /sdf/data/lcls/ds/mfx/mfx100852324/results/calib_view -d jungfrau -k exp=mfx100852324,run=6,dir=/sdf/data/lcls/drpsrcf/ffb/mfx/mfx100852324/xtc --stepnum 2 --stepmax 3 -L INFO --segind 9
    """
    import os
    import numpy as np
    from time import time

    for step in range(3):
        for panel in range(32):
            print('step:%d panel:%02d' % (step, panel))


def issue_2025_06_17():
    """       datinfo -k exp=mfx100852324,run=7 -d epix100_0
    """
    import os
    import numpy as np
    from time import time
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    ds = DataSource(exp='mfx100852324', run=13)
    myrun = next(ds.runs())
    det = myrun.Detector('epix100_0')
    #peds = det.raw._pedestals()
    calibc = det.calibconst
    print('det.calibconst:', calibc['pedestals'][1])
    print(ndu.info_ndarr(calibc['pedestals'][0], 'peds:'))

    events = 5
    for nevt,evt in enumerate(myrun.events()):
            if nevt>events-1: break
            t0_sec = time()
            raw   = det.raw.raw(evt)
            dt_sec = (time() - t0_sec)*1000
            print(ndu.info_ndarr(raw,   'evt:%3d dt=%.3f msec for det.raw.raw(evt):' % (nevt, dt_sec)))


def issue_2025_06_25(subtest=1):
    """test for detectors=['archon',]
       datinfo -k exp=rixx1017523,run=418 -d archon   (600, 4800) 1836 evts
    """
    import numpy as np
    from time import time
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.utils_psana as up

    dskwargs = None
    if subtest in (None, '1'):
        dskwargs = {'exp':'rixx1017523', 'run':418}
        #ds = DataSource(exp='rixx1017523',run=418)
    elif subtest == '2':
        dskwargs = {'exp':'rixx1017523', 'run':418, 'detectors':['archon']}
        #ds = DataSource(exp='rixx1017523',run=418, detectors=['archon'])
    elif subtest == '3':
        str_dskwargs = 'exp=rixx1017523,run=418'
        detname = 'archon'
        dskwargs = up.datasource_kwargs_from_string(str_dskwargs, detname=detname)
    elif subtest == '4':
        kwa = {'dskwargs':'exp=rixx1017523,run=418',
               'det':'archon'}
        dskwargs = up.data_source_kwargs(**kwa)
    else:
        kwa = {'dskwargs':"{'exp':'rixx1017523', 'run':418, 'detectors':['archon']}",
               'det':'archon'}
        dskwargs = up.data_source_kwargs(**kwa)

    print('subtest:%s dskwargs:%s' % (subtest, str(dskwargs)))
    ds = DataSource(**dskwargs)

    orun = next(ds.runs())
    det = orun.Detector('archon', gainfact=1) # , cmpars=(1,0,0)) #(1,0,0))

    events = 10
    evsel = 0

    for nev, evt in enumerate(orun.events()):
       t0_sec = time()
       raw = det.raw.raw(evt)
       dt_sec = (time() - t0_sec)*1000

       if raw is None:
           print('evt:%3d - raw is None' % nev, end='\r')
           continue
       evsel += 1

       if evsel>events:
           print('BREAK for nev>%d' % events)
           break

       print(info_ndarr(raw, 'evt/sel:%6d/%4d dt=%.3f msec  raw' % (nev, evsel, dt_sec), last=10))


def issue_2025_06_26(args):
    """epixm raw/image/calib
       datinfo -k exp=rix101332624,run=208 -d c_epixm
    """
    import numpy as np
    from time import time
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.utils_psana as up
    subtest = args.subtest if args.subtest is not None else '1'

    ds = DataSource(exp='rix101332624',run=208)

    orun = next(ds.runs())
    det = orun.Detector('c_epixm') # , cmpars=(1,0,0)) #(1,0,0))

    events = 10
    evsel = 0

    for nev, evt in enumerate(orun.events()):
       raw = det.raw.raw(evt)

       if raw is None:
           print('evt:%3d - raw is None' % nev, end='\r')
           continue

       evsel += 1
       if evsel>events:
           print('BREAK for nev>%d' % events)
           break


       t0_sec = time()
       nda = det.raw.calib(evt)
       #nda = det.raw.image(evt)
       dt_sec = (time() - t0_sec)*1000

       nda = raw
       #nda = calib
       #nda = image
       print(info_ndarr(nda, 'evt/sel:%6d/%4d dt=%.3f msec  nda' % (nev, evsel, dt_sec), last=5))


def issue_2025_06_27(args):
    """epixm raw/image/calib
       datinfo -k exp=rix101332624,run=208 -d c_epixm
    """
    import os
    import numpy as np
    from psana.detector.NDArrUtils import info_ndarr

    fname = '/sdf/home/p/philiph/psana/jungfrau/psana2/gains/pixel_offset.npy'

    a = np.load(fname)
    print(info_ndarr(a,'nda:'))


def issue_2025_06_30(args):
    """2025_06_30 5:13PM
       Hi Mikhail,
       For the epixm in exp=rix100837624,run=34 would you be able to deploy pedestals/offsets to zero and gains to 1
       so that det.calib values are the same as det.raw?  It would make Alex’s life easier for the beamtime starting tomorrow.
       We may also need to do it for rix101332624 tomorrow, but I have the impression
       the constants will automatically propagate there?  Let me know if not…
       Thanks!
       chris

       run test_issues_2025.py for issue_2025_06_30 to make epixm-zeros.data and epixm-ones.data
       datinfo -k exp=rix100837624,run=34 -d c_epixm  # shape:(4, 192, 384)

       calibman
       cdb add -e rix100837624 -d epixm320_000006 -c pedestals    -r 1 -f epixm-zeros.data
       cdb add -e rix100837624 -d epixm320_000006 -c pixel_offset -r 1 -f epixm-zeros.data
       cdb add -e rix100837624 -d epixm320_000006 -c pixel_gain   -r 1 -f epixm-ones.data
       calibman
    """
    import numpy as np
    from psana.detector.NDArrUtils import info_ndarr
    from psana.pscalib.calib.NDArrIO import save_txt, load_txt

    sh = (4, 192, 384)
    fname1 = 'epixm-ones.data'
    fname0 = 'epixm-zeros.data'
    a0 = np.zeros(sh, dtype=np.float32)
    a1 = np.ones(sh, dtype=np.float32)
    save_txt(fname0, a0, fmt='%.1f')
    save_txt(fname1, a1, fmt='%.1f')
    print('saved %s'% fname0, info_ndarr(a0,'a0:'))
    print('saved %s'% fname1, info_ndarr(a1,'a1:'))


def issue_2025_07_01(args):
    """test that issue_2025_06_30 is resolved - print pedestals, pixel_offset, pixel_gain for epixm320
       datinfo -k exp=rix100837624,run=34 -d c_epixm  # shape:(4, 192, 384)
    """
    import numpy as np
    from time import time
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.utils_psana as up
    subtest = args.subtest if args.subtest is not None else '1'

    ds = DataSource(exp='rix100837624',run=34)

    orun = next(ds.runs())
    det = orun.Detector('c_epixm') # , cmpars=(1,0,0)) #(1,0,0))

    print(info_ndarr(det.raw._pedestals(), 'pedestals', last=5))
    print(info_ndarr(det.raw._gain(), 'gain', last=5))
    #print(info_ndarr(det.raw._offset(), 'offset', last=5))

    events = 10
    evsel = 0

    for nev, evt in enumerate(orun.events()):
       raw = det.raw.raw(evt)

       if raw is None:
           print('evt:%3d - raw is None' % nev, end='\r')
           continue

       evsel += 1
       if evsel>events:
           print('BREAK for nev>%d' % events)
           break

       calib = det.raw.calib(evt)
       print(info_ndarr(raw,   'evt/sel:%6d/%4d raw' % (nev, evsel), last=5))
       print(info_ndarr(calib, 18*' '+'calib', last=5))


def issue_2025_07_16():
    """test info calibration validity"""
    import psana.pscalib.calib.UtilsCalibValidity as ucv
    from time import time
    t0_sec = time()
    ucv._calib_validity_ranges('mfx101332224', 'jungfrau_000003', ctype='pedestals')
    print('Consumed time (sec): %.6f' % (time()-t0_sec))


def issue_2025_07_22():
    """test time to get shortname from longname
    """
    from time import time
    import psana.detector.UtilsCalib as uc
    #longname = odet.raw._uniqueid
    longname = 'jungfrau_9000000000000-230921-3007f0217_9000000000000-230921-3001c01f5_9000000000000-230921-3007f0239_9000000000000-230921-3007f01c7'
    t0_sec = time()
    shortname = uc.detector_name_short(longname)
    print('uc.detector_name_short(longname) consumed time (sec): %.6f' % (time()-t0_sec))
    print('longname: %s\nshortname: %s' % (longname, shortname))


def issue_2025_07_23():
    """ detnames exp=mfx101332224,run=66 #,dir=/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc
        datinfo -k exp=mfx101332224,run=66 -d jungfrau
    """
    expname = 'mfx101332224'
    runnum = 66
    detname = 'jungfrau'

    ds, orun, odet = ds_run_det(exp=expname, run=runnum, detname=detname) #, dir='/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc')
    print('odet.raw._uniqueid', odet.raw._uniqueid) # epixhremu_00cafe0002-0000000000-0000000000-0000000000-...
    print('odet.raw._det_name', odet.raw._det_name) # epixhr_emu
    print('odet.raw._dettype',  odet.raw._dettype)  # epixhremu

    longname = odet.raw._uniqueid
    import psana.pscalib.calib.MDBWebUtils as wu
    calib_const = wu.calib_constants_all_types(longname, exp=expname, run=runnum)
    #calib_const = wu.calib_constants_all_types(longname, run=runnum)
    print('calib_const.keys:', calib_const.keys())

def issue_2025_07_29():
    """ cpo - epix10ka missing geometry

        datinfo -k exp=ascdaq123,run=192 -d epix10ka  # raw  shape:(4, 352, 384)
    """
    from psana import DataSource
    ds = DataSource(exp='ascdaq123',run=192)
    myrun = next(ds.runs())
    epix = myrun.Detector('epix10ka')
    for nevt,evt in enumerate(myrun.events()):
        print(epix.raw.raw(evt).shape)
        print(epix.raw.image(evt))
        if nevt>10: break



def issue_2025_08_19():
    """Philip - pixel_gain are not deployed/used?
       datinfo -k exp=mfxdaq23,run=31 -d epix100_0
    """
    expname = 'mfxdaq23'
    runnum  = 31
    detname = 'epix100_0'

    ds, orun, odet = ds_run_det(exp=expname, run=runnum, detname=detname) #, dir='/sdf/data/lcls/drpsrcf/ffb/tst/tstx00417/xtc')
    print('odet.raw._uniqueid', odet.raw._uniqueid) # epixhremu_00cafe0002-0000000000-0000000000-0000000000-...
    print('odet.raw._det_name', odet.raw._det_name) # epixhr_emu
    print('odet.raw._dettype',  odet.raw._dettype)  # epixhremu

    longname = odet.raw._uniqueid
    import psana.pscalib.calib.MDBWebUtils as wu
    calib_const = wu.calib_constants_all_types(longname, exp=expname, run=runnum)
    #calib_const = wu.calib_constants_all_types(longname, run=runnum)
    print('calib_const.keys:', calib_const.keys())

    import psana.pscalib.calib.MDBWebUtils as wu
#    docs = wu.find_docs('cdb_mfxdaq23', 'epix100_000005', query={'ctype':'pixel_gain'})
#    docs = wu.find_docs('cdb_mfxdaq23', 'epix100_000005', query={"detector": "epix100_000005", "run": 31})
    docs = wu.find_docs('cdb_mfxdaq23', 'epix100_000005', query={"detector": "epix100_000005", "run":{"$lte": 31}})
    print(docs)


def issue_2025_08_26():
    """       datinfo -k exp=ascdaq023,run=43 -d jungfrau
    """
    from psana import DataSource
    ds = DataSource(exp='ascdaq023',run=43)
    myrun = next(ds.runs())
    det = myrun.Detector('jungfrau')

    #print('XXX', det.raw._segment_numbers)
    for nevt,evt in enumerate(myrun.events()):
      if nevt>10: break
      calib = det.raw.calib(evt)
      if calib is None:
        print('none')
        continue
      print(nevt,calib.shape)


def issue_2025_08_28(subtest='0o7777'):
    """test det.raw.calib()
       datinfo -k exp=mfx100852324,run=7 -d epix100_0
    """
    import os
    import numpy as np
    from time import time
    from psana import DataSource
    from psana.detector.UtilsGraphics import gr, fleximage
    import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d

    ds = DataSource(exp='mfx100852324', run=13)
    myrun = next(ds.runs())
#    det = myrun.Detector('epix100_0', logmet_init=logger.info)
    det = myrun.Detector('epix100_0')

    isubset = 0o7777 if subtest is None else int(subtest)
    if isubset & 1:
        #peds = det.raw._pedestals()
        calibc = det.calibconst
        print('===\ndet.calibconst:', calibc['pedestals'][1])
        print(ndu.info_ndarr(calibc['pedestals'][0], 'peds:'))
        print('===\n', det.raw._info_calibconst(), '\n===\n')

    if isubset & 2:
        print('=== test of mask methods ===')
        mask_def = det.raw._mask_default()
        print(ndu.info_ndarr(mask_def,                          '     det.raw._mask_default()         :'))
        print(ndu.info_ndarr(det.raw._mask_calib_or_default(),  '     det.raw._mask_calib_or_default():'))
        print(ndu.info_ndarr(det.raw._mask_from_status(),       '     det.raw._mask_from_status()     :'))
        print(ndu.info_ndarr(det.raw._mask_neighbors(mask_def), '     det.raw._mask_neighbors()       :'))
        print(ndu.info_ndarr(det.raw._mask_edges(),             '     det.raw._mask_edges()           :'))
        print(ndu.info_ndarr(det.raw._mask_center(),            '     det.raw._mask_center()          :'))
        print(ndu.info_ndarr(det.raw._mask_comb(),              '     det.raw._mask_comb()            :'))
        print(ndu.info_ndarr(det.raw._mask(),                   '     det.raw._mask()                 :'))
        print('===\n')

#    kwa = {'status':True, 'neighbors':True, 'edges':True, 'center':True, 'calib':True, 'umask':None}
    events = 5
    for nevt,evt in enumerate(myrun.events()):
        if nevt>events-1: break
        t0_sec = time()
        #raw   = det.raw.raw(evt)
        nda   = det.raw.calib(evt) #, **kwa)
        dt_sec = (time() - t0_sec)*1000
        print(ndu.info_ndarr(nda,   'evt:%3d dt=%.3f msec for det.raw.calib(evt):' % (nevt, dt_sec)))


#===
    
#===

def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc'  # None
    d_detname  = 'archon' # None
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    h_tname    = 'test name, usually numeric number 0,...,>20, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME,\
                            usage='for list of implemented tests run it without parameters')
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
    return parser


def selector():
    parser = argument_parser()
    args = parser.parse_args()
    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    TNAME = args.tname # sys.argv[1] if len(sys.argv)>1 else '0'

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
    elif TNAME in ('13',): issue_2025_04_03() # Aaron Brewster - acces to jungfrau geometry from det._calibconst
    elif TNAME in ('14',): issue_2025_04_09() # Philip: det.calibconst.keys() - missing pixel_gain
    elif TNAME in ('15',): issue_2025_nda()   # me - generate and save random numpy array in file
    elif TNAME in ('16',): issue_2025_04_10() # cpo - epixquad det.raw.image timing with OPENBLAS_NUM_THREADS=1/0
    elif TNAME in ('17',): issue_2025_04_11() # me - access to multiple calibconst
    elif TNAME in ('18',): issue_2025_04_17() # cpo - timing for large np.array with OPENBLAS_NUM_THREADS=1/0 - resulting time difference 2.5%
    elif TNAME in ('19',): issue_2025_04_21() # me - timing of jungfrau 16M, the same as issue_2025_04_10, but for jungfrau 16M
    elif TNAME in ('20',): issue_2025_04_22() # cpo - epixquad calib/image are 0. epix10ka_deploy_constants deployed zeros - misidentified "gain"
    elif TNAME in ('21',): issue_2025_04_23() # cpo - jungfrau16M image for 4 drp panels
    elif TNAME in ('22',): issue_2025_04_29() # Philip/cpo - jungfrau16M constants shape:(3, 19, 512, 1024).
    elif TNAME in ('23',): issue_2025_05_07() # me - test of the detector axis shape:(796, 6144)
    elif TNAME in ('24',): issue_2025_05_14() # me - test QComboBox for control_gui
    elif TNAME in ('25',): issue_2025_05_16(USE_GZIP=True)  # me - test with gzip save and load
    elif TNAME in ('26',): issue_2025_05_16(USE_GZIP=False) # me - test  w/o gzip save and load
    elif TNAME in ('27',): issue_2025_06_05() # Seshu & Chris - too many messages epixuhr | INFO ] TBD, psana.detector.calibconstants | WARNING ]
    elif TNAME in ('28',): issue_2025_06_06() # Chris - jungfrau_dark_proc split for steps and panels
    elif TNAME in ('29',): issue_2025_06_17() # Chris - calibconstants run range
    elif TNAME in ('30',): issue_2025_06_25(args.subtest) # Patrik - test for detectors=['archon',]
    elif TNAME in ('31',): issue_2025_06_26(args) # me - epixm raw/image/calib
    elif TNAME in ('32',): issue_2025_06_27(args) # philip - calibrepo
    elif TNAME in ('33',): issue_2025_06_30(args) # cpo - epixm add to DB pedestals=0, pixel_offset=0, pixel_gain=1
    elif TNAME in ('34',): issue_2025_07_01(args) # test that issue_2025_06_30 is resolved - print pedestals, pixel_offset, pixel_gain for epixm320
    elif TNAME in ('35',): issue_2025_07_16() # test for runs validity tool
    elif TNAME in ('36',): issue_2025_07_22() # test time to get shortname from longname
    elif TNAME in ('37',): issue_2025_07_23() # test wu.calib_constants_all_types
    elif TNAME in ('38',): issue_2025_07_29() # Chris - epix10ka missing geometry
    elif TNAME in ('39',): issue_2025_08_19() # Philip - epix100 pixel_gain are not deployed/used?
    elif TNAME in ('40',): issue_2025_08_26() # Chris - jf - fix det.raw.calib(evt) in case if pedestals are missing?
    elif TNAME in ('41',): issue_2025_08_28(args.subtest) # me - epix100 test det.raw.calib(evt)
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
