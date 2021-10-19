#!/usr/bin/env python
"""
2021-10-18:
Xiaozhe complains that too many pixels outside signal region in ueddaq02 r401 shows up in selection of intensities between 100 and 500 keV.
See:
  - `github: <https://github.com/slac-lcls/lcls2>`_.
  - `confluence: <https://confluence.slac.stanford.edu/display/PSDM/EPIXQUAD+ueddaq02+r401+issue+calib+hot+banks+2021-10-18>`_.

Created on 2021-10-18 by Mikhail Dubrovin
"""
import sys
import numpy as np
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=logging.INFO)
from psana.pyalgos.generic.NDArrUtils import info_ndarr
from psana import DataSource
from psana.detector.UtilsGraphics import gr, fleximagespec#, fleximage, flexhist

tname = sys.argv[1] if len(sys.argv) > 1 else '0'
flims  = None
fname = 'ims.png'
EVENTS = 5
THRMIN = 100
THRMAX = 500
AMIN   = 1
AMAX   = 200
M14 = 0o37777
GAINF = 1./0.164
CROP1_IMG = False
CROP2_IMG = False

def selection(arr): return np.where((arr>THRMIN) & (arr<THRMAX), arr, 0)

ds = DataSource(exp='ueddaq02',run=401)
orun = next(ds.runs())
det = orun.Detector('epixquad')

prefix = 'ims-%s-r%04d' % (orun.expt, orun.runnum)

print('*** det.raw._calibconst.keys:', det.raw._calibconst.keys())
print('*** pedestal metadata:', det.raw._calibconst['pedestals'][1])
print('*** gain metadata:', det.raw._calibconst['pixel_gain'][1])
print('*** rms metadata:', det.raw._calibconst['pixel_rms'][1])
print('*** status metadata:', det.raw._calibconst['pixel_status'][1])

peds  = det.raw._calibconst['pedestals'][0]
gains = det.raw._calibconst['pixel_gain'][0]
rms   = det.raw._calibconst['pixel_rms'][0]
print(info_ndarr(peds,'pedestals'))
print(info_ndarr(gains,'gain'))
print(info_ndarr(rms,'rms'))

arr, img = None, None
suffix = ''

for nevt,evt in enumerate(orun.events()):

    if nevt>EVENTS:
        print('break at nevt %d' % nevt)
        break

    if tname=='1':
        suffix = 'segment-nums'
        ones = np.ones((352,384))
        arr = np.stack([ones-0.7, ones, ones*2, ones*3])
        AMIN = 0
        AMAX =3

    elif tname=='2':
        suffix = 'e%04d-gain-range-index' % nevt
        arr = det.raw._gain_range_index(evt)
        AMIN = 0
        AMAX = 15

    elif tname=='3':
        suffix = 'gain'
        arr = gains[2,:]   #(7, 4, 352, 384)
        AMIN = 0
        AMAX = 1

    elif tname=='4':
        suffix = 'pedestals'
        arr = peds[2,:]   #(7, 4, 352, 384)
        AMIN = 2000
        AMAX = 4000

    elif tname=='5':
        suffix = 'rms'
        arr = rms[2,:]   #(7, 4, 352, 384)
        AMIN = 0
        AMAX = 6

    elif tname=='6':
        suffix = 'e%04d-raw' % nevt
        arr = det.raw.raw(evt) & M14
        AMIN = 2000
        AMAX = 4000

    elif tname=='7':
        suffix = 'e%04d-raw-peds' % nevt
        arr = (det.raw.raw(evt) & M14) - peds[2,:]
        AMIN = -17
        AMAX = 17

    elif tname=='8':
        suffix = 'e%04d-raw-peds-x-gain' % nevt
        arr = ((det.raw.raw(evt) & M14) - peds[2,:])*GAINF
        AMIN = -100
        AMAX = 100

    elif tname=='9':
        suffix = 'e%04d-raw-peds-x-gain-region-hot' % nevt
        arr = np.array(((det.raw.raw(evt) & M14) - peds[2,:])*GAINF)
        CROP1_IMG = True
        AMIN = -100
        AMAX = 100

    elif tname=='10':
        suffix = 'e%04d-raw-peds-x-gain-region-cold' % nevt
        arr = np.array(((det.raw.raw(evt) & M14) - peds[2,:])*GAINF)
        CROP2_IMG = True
        AMIN = -100
        AMAX = 100

    elif tname=='11':
        suffix = 'e%04d-calib' % nevt
        arr = det.raw.calib(evt)
        AMIN = -100
        AMAX = 100

    else:
        suffix = 'e%04d-calib-issue-with-thresholds' % nevt
        arr = det.raw.calib(evt)
        arr = selection(arr)
        AMIN = 50
        AMAX = 200

    print(info_ndarr(arr,'Event %d det.raw.calib'%nevt))

    img = det.raw.image(evt, nda=arr)

    if CROP1_IMG:
        img0 = np.zeros_like(img)
        img0[:352,600:] = img[:352,600:]
        img = img0
        arr = img[:352,600:]

    if CROP2_IMG:
        img0 = np.zeros_like(img)
        img0[:352,:192] = img[:352,:192]
        img = img0
        arr = img[:352,:192]

    print(info_ndarr(img,'  img'))

    if flims is None:
        flims = fleximagespec(img, arr=arr, bins=100, w_in=11, h_in=8, amin=AMIN, amax=AMAX)#fraclo=0.01, frachi=0.99
        flims.move(10,20)
    else:
        fname = '%s-%s.png' % (prefix, suffix)
        flims.update(img, arr=arr, amin=AMIN, amax=AMAX)
        flims.axtitle('Event %d %s'%(nevt,fname))

    gr.show(mode=1)

    if tname=='0':
        flims.save(fname)

gr.show()

flims.save(fname)
