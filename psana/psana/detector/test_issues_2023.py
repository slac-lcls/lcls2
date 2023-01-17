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


def issue_2023_01_03():
    """epixhr calib method with common mode correction using standard detector interface
       datinfo -k exp=rixx45619,run=119 -d epixhr
    """
    import psana.pyalgos.generic.PSUtils as psu
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from time import time

    #ds = DataSource(exp='rixx45619',run=121, dir='/cds/data/psdm/prj/public01/xtc')
    #orun = next(ds.runs())
    #det = orun.Detector('epixhr')

    ds = DataSource(exp='ueddaq02',run=569, dir='/cds/data/psdm/prj/public01/xtc')
    orun = next(ds.runs())
    det = orun.Detector('epixquad')

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
    from psana import DataSource
    from time import time
    import numpy as np

    #ds = DataSource(exp='rixx45619',run=121, dir='/cds/data/psdm/prj/public01/xtc')
    #orun = next(ds.runs())
    #det = orun.Detector('epixhr')

    ds = DataSource(exp='ueddaq02',run=569, dir='/cds/data/psdm/prj/public01/xtc')
    orun = next(ds.runs())
    det = orun.Detector('epixquad')

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
    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    from time import time
    import numpy as np
    # dir='/cds/data/psdm/asc/ascdaq18/xtc/' # default
    # dir='/cds/data/psdm/prj/public01/xtc') # preserved
    #ds = DataSource(exp='ascdaq18',run=170)
    ds = DataSource(exp='ascdaq18',run=171)
    orun = next(ds.runs())
    det = orun.Detector('epixhr')

    config = det.raw._config_object()
    calibc = det.raw._calibconst

    logger.debug('calibc: %s' % str(calibc))

    cc = ucc.calib_components_epix(calibc, config)
    data_bit_mask = cc.data_bit_mask()
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
    for nstep,step in enumerate(orun.steps()):
      if nstep>5: break
      for nevt,evt in enumerate(step.events()):
        #if nevt>1000: break
        if nevt%100: continue

        print('== Step %02d Event %03d ==' % (nstep,nevt))

        #t0_sec_tot = time()
        raw = det.raw.raw(evt)
        if raw is None: continue

        #peds = cc.event_pedestals(raw)
        #arr = None
        #arr2 = np.array(raw & data_bit_mask, dtype=np.float32) - peds

        gmaps = cc.gain_maps_epix(raw)
        #arr = ucc.event_constants_for_gmaps(gmaps, ones, default=0)
        #arr = ucc.event_constants_for_gmaps(gmaps, ones, default=0)
        arr = ucc.map_gain_range_index_for_gmaps(gmaps, default=10)

        print(info_ndarr(arr,'arr:'))

        #logger.info('time consumption to make 3-d array for imaging = %.6f sec' % (time()-t0_sec_tot))

        #img = cc.pedestals()[1,0,:150,:200]
        #img = arr[0,144:,:192] # cut off a single ASIC with meaningfull data
        #img = arr[0,:143,:192] # cut off a single ASIC with meaningfull data
        img = arr[0,:150,:200] # cut off a single ASIC with meaningfull data
        #img = arr[0,:,:] # cut off a single ASIC with meaningfull data
        #img = ucc.psu.table_nxn_epix10ka_from_ndarr(arr, gapv=0)
        #print(info_ndarr(img,'img:'))

        if flimg is None:
           flimg = fleximage(img, arr=None, h_in=8, w_in=11, nneg=1, npos=3)
        gr.set_win_title(flimg.fig, titwin='Step %02d Event %d' % (nstep,nevt))
        flimg.update(img, arr=None)
        gr.show(mode='DO NOT HOLD')
    gr.show()


def issue_2023_01_dd():
    print('template')

USAGE = '\nUsage:'\
      + '\n  python %s <test-name> <loglevel-e.g.-DEBUG-or-INFO>' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - print usage'\
      + '\n    1 - issue_2023_01_03 - test epixhr, calib and common mode correction'\
      + '\n    2 - issue_2023_01_06 - test utils_calib_components.py'\
      + '\n    3 - issue_2023_01_10 - test for of the 1st charge injection for epixhr'\

TNAME = sys.argv[1] if len(sys.argv)>1 else '0'

if   TNAME in  ('0',): issue_2023_01_dd()
elif TNAME in  ('1',): issue_2023_01_03()
elif TNAME in  ('2',): issue_2023_01_06()
elif TNAME in  ('3',): issue_2023_01_10()
else:
    print(USAGE)
    exit('TEST %s IS NOT IMPLEMENTED'%TNAME)

exit('END OF TEST %s'%TNAME)

#if __name__ == "__main__":
