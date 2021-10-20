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
import math
import numpy as np
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=logging.INFO)
from psana.pyalgos.generic.NDArrUtils import info_ndarr, divide_protected
from psana import DataSource
from psana.detector.UtilsGraphics import gr, fleximagespec#, fleximage, flexhist
import argparse


SCRNAME = sys.argv[0].rsplit('/')[-1]
USAGE = '\n    %s -r554 -t1' % SCRNAME\
      + '\n    %s -e ueddaq02 -d epixquad -rr554 -t1' % SCRNAME\

d_tname   = '0'
d_detname = 'epixquad'
d_expname = 'ueddaq02'
d_run     = 554
d_events  = 5
d_evskip  = 0
d_saveims = False
d_grindex = 2
d_amin    = -100
d_amax    = 100

parser = argparse.ArgumentParser(usage=USAGE, description='%s - test issue 2021-10-18 with epixquad data'%SCRNAME)
parser.add_argument('-t', '--tname',   default=d_tname,   type=str, help='test name, def=%s' % d_tname)
parser.add_argument('-d', '--detname', default=d_detname, type=str, help='detector name, def=%s' % d_detname)
parser.add_argument('-e', '--expname', default=d_expname, type=str, help='experiment name, def=%s' % d_expname)
parser.add_argument('-r', '--run',     default=d_run,     type=int, help='run number, def=%s' % d_run)
parser.add_argument('-N', '--events',  default=d_events,  type=int, help='maximal number of events, def=%s' % d_events)
parser.add_argument('-K', '--evskip',  default=d_evskip,  type=int, help='number of events to skip in the beginning of run, def=%s' % d_evskip)
parser.add_argument('-S', '--saveims', default=d_saveims, action='store_true', help='save image in file, def=%s' % d_saveims)
parser.add_argument('-g', '--grindex', default=d_grindex, type=int, help='gain range index [0,6] for peds, gains etc., def=%s' % str(d_grindex))
parser.add_argument('--amin',          default=d_amin,    type=float, help='spectrum minimal value, def=%s' % str(d_amin))
parser.add_argument('--amax',          default=d_amax,    type=float, help='spectrum maximal value, def=%s' % str(d_amax))

args = parser.parse_args()
print('*** parser.parse_args: %s' % str(args))

tname = args.tname # sys.argv[1] if len(sys.argv) > 1 else '0'
THRMIN = 100
THRMAX = 500
AMIN   = 1
AMAX   = 200
M14 = 0o37777
GAINF = 1./0.164
CROP1_IMG = False
CROP2_IMG = False

flims  = None
fname = 'ims.png'

def selection(arr): return np.where((arr>THRMIN) & (arr<THRMAX), arr, 0)

ds = DataSource(exp=args.expname, run=args.run)
orun = next(ds.runs())
det = orun.Detector(args.detname)

prefix = 'ims-%s-r%04d' % (orun.expt, orun.runnum)

print('*** det.raw._calibconst.keys:', det.raw._calibconst.keys())
print('*** pedestal metadata:', det.raw._calibconst['pedestals'][1])
print('*** gain metadata:', det.raw._calibconst['pixel_gain'][1])
print('*** rms metadata:', det.raw._calibconst['pixel_rms'][1])
print('*** status metadata:', det.raw._calibconst['pixel_status'][1])

peds  = det.raw._calibconst['pedestals'][0]
gain = det.raw._calibconst['pixel_gain'][0]
rms   = det.raw._calibconst['pixel_rms'][0]
print(info_ndarr(peds,'pedestals'))
print(info_ndarr(rms,'rms'))
print(info_ndarr(gain,'gain'))

arr, img = None, None
suffix = ''

for nevt,evt in enumerate(orun.events()):

    if nevt>args.events:
        print('break at nevt %d' % nevt)
        break

    if nevt<args.evskip:
        print('skip nevt %d' % nevt)
        continue

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
        arr = gain[args.grindex,:]   #(7, 4, 352, 384)
        AMIN = 0
        AMAX =math.ceil(np.amax(arr))

    elif tname=='4':
        suffix = 'pedestals'
        arr = peds[args.grindex,:]   #(7, 4, 352, 384)
        AMIN = 2000
        AMAX = 4000

    elif tname=='5':
        suffix = 'rms'
        arr = rms[args.grindex,:]   #(7, 4, 352, 384)
        AMIN = 0
        AMAX = 8 #math.ceil(np.amax(arr))

    elif tname=='6':
        suffix = 'e%04d-raw' % nevt
        arr = det.raw.raw(evt) & M14
        AMIN = 2000
        AMAX = 4000

    elif tname=='7':
        suffix = 'e%04d-raw-peds' % nevt
        arr = (det.raw.raw(evt) & M14) - peds[args.grindex,:]
        AMIN = -17
        AMAX = 17

    elif tname=='8':
        suffix = 'e%04d-raw-peds-x-gain' % nevt
        arr = ((det.raw.raw(evt) & M14) - peds[args.grindex,:])*GAINF
        AMIN = -100
        AMAX = 100

    elif tname=='9':
        suffix = 'e%04d-raw-peds-x-gain-region-hot' % nevt
        arr = np.array(((det.raw.raw(evt) & M14) - peds[args.grindex,:])*GAINF)
        CROP1_IMG = True
        AMIN = -100
        AMAX = 100

    elif tname=='10':
        suffix = 'e%04d-raw-peds-x-gain-region-cold' % nevt
        arr = np.array(((det.raw.raw(evt) & M14) - peds[args.grindex,:])*GAINF)
        CROP2_IMG = True
        AMIN = -100
        AMAX = 100

    elif tname=='11':
        suffix = 'calib-e%04d' % nevt
        arr = det.raw.calib(evt)
        AMIN = args.amin #-100
        AMAX = args.amax # 100

    elif tname=='12':
        suffix = 'gain-factor'
        gfac = divide_protected(np.ones_like(gain), gain)
        print(info_ndarr(gfac,'gfac'))
        from psana.detector.UtilsEpix10ka import gain_maps_epix10ka_any
        gmaps = gain_maps_epix10ka_any(det.raw, evt)
        assert gmaps is not None
        #gr0, gr1, gr2, gr3, gr4, gr5, gr6 = gmaps
        factor = np.select(gmaps,\
                       (gfac[0,:], gfac[1,:], gfac[2,:], gfac[3,:],\
                        gfac[4,:], gfac[5,:], gfac[6,:]), default=1) # 2msec
        print(info_ndarr(factor, 'factor'))
        #arr = gain[grind,:]   #(7, 4, 352, 384)
        arr = factor
        AMIN = 0
        AMAX =math.ceil(np.amax(factor))

    else:
        suffix = 'calib-issue-with-thresholds-e%04d' % nevt
        arr = selection(det.raw.calib(evt))
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

    if tname in ('0','11') and args.saveims:
        flims.save(fname)

gr.show()

if args.saveims: flims.save(fname)

sys.exit('END OF %s -t %s' % (SCRNAME, tname))
