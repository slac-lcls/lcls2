#!/usr/bin/env python
"""
This script grew-up from test of specific issue -
2021-10-18:
Xiaozhe complains that too many pixels outside signal region in ueddaq02 r401 shows up in selection of intensities between 100 and 500 keV.
See:
  - `github: <https://github.com/slac-lcls/lcls2>`_.
  - `confluence: <https://confluence.slac.stanford.edu/display/PSDM/EPIXQUAD+ueddaq02+r401+issue+calib+hot+banks+2021-10-18>`_.

Created on 2021-10-18 by Mikhail Dubrovin
"""
import sys

SCRNAME = sys.argv[0].rsplit('/')[-1]
if len(sys.argv)<2: sys.exit( '\nMISSING PARAMETERS\nTry: %s -h' % SCRNAME)

import math
import numpy as np
import logging
logger = logging.getLogger(__name__)
DICT_NAME_TO_LEVEL = logging._nameToLevel # {'INFO': 20, 'WARNING': 30, 'WARN': 30,...

from psana.detector.Utils import info_parser_arguments
from psana.pyalgos.generic.NDArrUtils import info_ndarr, divide_protected, reshape_to_2d
from psana import DataSource
from psana.detector.UtilsGraphics import gr, fleximagespec#, fleximage, flexhist
from psana.detector.utils_psana import datasource_kwargs_from_string

from psana.detector.UtilsEpix10ka  import event_constants
import argparse

USAGE = '\n    %s -h' % SCRNAME\
      + '\n    %s -r554 -t1' % SCRNAME\
      + '\n    %s -k /cds/data/psdm/prj/public01/xtc/ueddaq02-r0569-s001-c000.xtc2 -d epixquad -t1' % SCRNAME\
      + '\n    -t, --tname - test name/number:'\
      + '\n      1 - segment numeration'\
      + '\n      2 - gain range index'\
      + '\n      3 - gain, ADU/keV'\
      + '\n      4 - pedestals'\
      + '\n      5 - rms'\
      + '\n      6 - raw'\
      + '\n      7 - raw-peds'\
      + '\n      8 - (raw-peds)/gain, keV'\
      + '\n      9 - calib, keV'\
      + '\n     10 - status'\
      + '\n     11 - gain factor = 1/gain, keV/ADU'\
      + '\n     ----'\
      + '\n     21 - run 401 two-threshold selection issue'\
      + '\n     22 - (raw-peds)/gain, keV hot - specific isuue test'\
      + '\n     23 - (raw-peds)/gain, keV cold - specific isuue test'\
      + '\n'

d_dskwargs = None
d_detname = 'epixquad'
d_tname   = '0'
d_events  = 5
d_evskip  = 0
d_stepnum = None
d_saveimg = False
d_grindex = None
d_amin    = None
d_amax    = None
d_cframe  = 0
d_loglev  = 'INFO'

h_dskwargs= 'string of comma-separated (no spaces) simple parameters for DataSource(**kwargs),'\
            ' ex: exp=<expname>,run=<runs>,dir=<xtc-dir>, ...,'\
            ' or <fname.xtc> or files=<fname.xtc>'\
            ' or pythonic dict of generic kwargs, e.g.:'\
            ' \"{\'exp\':\'tmoc00318\', \'run\':[10,11,12], \'dir\':\'/a/b/c/xtc\'}\", default = %s' % d_dskwargs

parser = argparse.ArgumentParser(usage=USAGE, description='%s - test per-event components of the det.raw.calib method'%SCRNAME)
parser.add_argument('-k', '--dskwargs', type=str, help=h_dskwargs)
parser.add_argument('-d', '--detname', default=d_detname, type=str, help='detector name, def=%s' % d_detname)
parser.add_argument('-t', '--tname',   default=d_tname,   type=str, help='test name, def=%s' % d_tname)
parser.add_argument('-N', '--events',  default=d_events,  type=int, help='maximal number of events, def=%s' % d_events)
parser.add_argument('-K', '--evskip',  default=d_evskip,  type=int, help='number of events to skip in the beginning of run, def=%s' % d_evskip)
parser.add_argument('-s', '--stepnum', default=d_stepnum, type=int, help='step number counting from 0 or None for all steps, def=%s' % d_stepnum)
parser.add_argument('-S', '--saveimg', default=d_saveimg, action='store_true', help='save image in file, def=%s' % d_saveimg)
parser.add_argument('-g', '--grindex', default=d_grindex, type=int, help='gain range index [0,6] for peds, gains etc., def=%s' % str(d_grindex))
parser.add_argument('-l', '--loglev',  default=d_loglev,  type=str, help='logger level (DEBUG, INFO, WARNING, etc.), def.=%s' % str(d_loglev))
parser.add_argument('--amin',          default=d_amin,    type=float, help='spectrum minimal value, def=%s' % str(d_amin))
parser.add_argument('--amax',          default=d_amax,    type=float, help='spectrum maximal value, def=%s' % str(d_amax))
parser.add_argument('--cframe',        default=d_cframe,  type=int, help='coordinate frame for images 0/1 for psana/LAB, def=%s' % str(d_cframe))

args = parser.parse_args()
logger.info('*** parser.parse_args: %s' % str(args))

logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=DICT_NAME_TO_LEVEL[args.loglev])
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('psana.psexp.event_manager').setLevel(logging.INFO)

logger.info(info_parser_arguments(parser))

tname = args.tname # sys.argv[1] if len(sys.argv) > 1 else '0'
THRMIN = 100
THRMAX = 500
AMIN   = 1
AMAX   = 200
CROP1_IMG = False
CROP2_IMG = False

flims  = None
fname = 'ims.png'

def selection(arr): return np.where((arr>THRMIN) & (arr<THRMAX), arr, 0)

def amin_amax(args, amin_def=None, amax_def=None):
    return args.amin if args.amin else amin_def,\
           args.amax if args.amax else amax_def

ds = DataSource(**datasource_kwargs_from_string(args.dskwargs))
orun = next(ds.runs())
det = orun.Detector(args.detname)

MDB = det.raw._data_bit_mask # M14 if det.raw._dettype == 'epix10ka' else M15
prefix = 'ims-%s-r%04d' % (orun.expt, orun.runnum)

keys = det.raw._calibconst.keys()

print('*** det.raw._data_bit_mask_: %s' % oct(MDB))
print('*** det.raw._calibconst.keys:', keys)
print('*** pedestal metadata:', det.raw._calibconst['pedestals'][1] if 'pedestals' in keys else None)
print('*** gain metadata:', det.raw._calibconst['pixel_gain'][1] if 'pixel_gain' in keys else None)
#print('*** rms metadata:', det.raw._calibconst['pixel_rms'][1])
#print('*** status metadata:', det.raw._calibconst['pixel_status'][1])

peds   = det.raw._calibconst['pedestals'][0]    if 'pedestals'    in keys else None
gain   = det.raw._calibconst['pixel_gain'][0]   if 'pixel_gain'   in keys else None
rms    = det.raw._calibconst['pixel_rms'][0]    if 'pixel_rms'    in keys else None
status = det.raw._calibconst['pixel_status'][0] if 'pixel_status' in keys else None
print(info_ndarr(peds,'pedestals'))
print(info_ndarr(rms,'rms'))
print(info_ndarr(gain,'gain, ADU/keV'))

arr, img = None, None
suffix = ''
evt_peds, evt_gfac = None, None

for nstep,step in enumerate(orun.steps()):

  if args.stepnum is not None and nstep<args.stepnum:
    print('skip nstep %d < stepnum=%d' % (nstep, args.stepnum))
    continue

  if args.stepnum is not None and nstep>args.stepnum:
    print('break at nstep %d > stepnum=%d' % (nstep, args.stepnum))
    break

  print('=== Step %d' % nstep)

  for nevt,evt in enumerate(step.events()):

    if nevt>args.events:
        print('break at nevt %d' % nevt)
        break

    if nevt<args.evskip:
        print('skip nevt %d' % nevt)
        continue

    if tname in ('4', '7', '8', '22', '23'):
        evt_peds = peds[args.grindex,:] if args.grindex is not None else\
                   event_constants(det.raw, evt, peds) #(7, 4, 352, 384) -> (4, 352, 384)
        print(info_ndarr(evt_peds,'evt_peds'))

    if tname in ('8', '11', '22', '23'):
        gfac = divide_protected(np.ones_like(gain), gain)
        evt_gfac = gfac[args.grindex,:] if args.grindex is not None else\
                   event_constants(det.raw, evt, gfac) #(7, 4, 352, 384) -> (4, 352, 384)
        print(info_ndarr(evt_gfac,'evt_gfac, keV/ADU'))

    step_evt = 's%02d-e%04d' % (nstep, nevt)

    if tname=='1':
        suffix = 'segment-nums'
        ones = np.ones(det.raw._seg_geo.shape()) # (352,384)
        seginds = det.raw._segment_indices() #_segments(evt)
        print('seginds', seginds)
        arr = np.stack([ones*i for i in seginds])
        AMIN, AMAX = amin_amax(args, amin_def=-1, amax_def=4)

    elif tname=='2':
        suffix = 'gain-range-index-%s' % step_evt
        arr = det.raw._gain_range_index(evt)
        AMIN, AMAX = amin_amax(args, amin_def=-1, amax_def=8)

    elif tname=='3':
        suffix = 'gain-%s' % step_evt
        arr = event_constants(det.raw, evt, gain) #(4, 352, 384)
        AMIN, AMAX = amin_amax(args, amin_def=0, amax_def=20)

    elif tname=='4':
        suffix = 'pedestals-%s' % step_evt
        arr = evt_peds
        AMIN, AMAX = amin_amax(args, amin_def=2000, amax_def=4000)

    elif tname=='5':
        suffix = 'rms-%s' % step_evt
        arr = rms[args.grindex,:] if args.grindex is not None else\
              event_constants(det.raw, evt, rms) #(4, 352, 384)
        AMIN, AMAX = amin_amax(args, amin_def=0, amax_def=8)

    elif tname=='6':
        suffix = 'raw-%s' % step_evt
        arr = det.raw.raw(evt) & MDB
        AMIN, AMAX = None, None # amin_amax(args, amin_def=2000, amax_def=4000)
        ofname = suffix + '.npy'
        logger.info('saved file %s' % ofname)
        np.save(ofname, arr)

    elif tname=='7':
        suffix = 'raw-peds-%s' % step_evt
        arr = (det.raw.raw(evt) & MDB)
        if evt_peds is None: logger.warning('evt_peds is None - DO NOT SUBTRACT')
        else: arr = arr.astype(evt_peds.dtype) - evt_peds
        AMIN, AMAX = amin_amax(args, amin_def=-40, amax_def=40)

    elif tname=='8':
        suffix = 'raw-peds-x-gain-%s' % step_evt
        arr = (det.raw.raw(evt) & MDB)
        if evt_peds is None: logger.warning('evt_peds is None - DO NOT SUBTRACT')
        else: arr = arr.astype(evt_peds.dtype) - evt_peds
        if evt_gfac is None: logger.warning('evt_gfac is None - DO NOT MULTIPLY')
        else: arr *= evt_gfac
        AMIN, AMAX = amin_amax(args, amin_def=-5, amax_def=5)

    elif tname=='9':
        suffix = 'calib-%s' % step_evt
        arr = det.raw.calib(evt)
        AMIN, AMAX = amin_amax(args, amin_def=-5, amax_def=5)

    elif tname=='10':
        suffix = 'status-%s' % step_evt
        arr = event_constants(det.raw, evt, status) #(4, 352, 384)
        AMIN, AMAX = amin_amax(args, amin_def=0, amax_def=32)

    elif tname=='11':
        suffix = 'gain-factor-%s' % step_evt
        arr = evt_gfac
        AMIN, AMAX = amin_amax(args, amin_def=0, amax_def=20)

    elif tname=='21':
        suffix = 'calib-issue-with-thresholds-%s' % step_evt
        arr = selection(det.raw.calib(evt))
        AMIN, AMAX = amin_amax(args, amin_def=50, amax_def=200)

    elif tname=='22':
        suffix = 'raw-peds-x-gain-region-hot-%s' % step_evt
        arr = np.array(((det.raw.raw(evt) & MDB) - evt_peds)*evt_gfac)
        CROP1_IMG = True
        AMIN, AMAX = amin_amax(args, amin_def=-5, amax_def=5)

    elif tname=='23':
        suffix = 'raw-peds-x-gain-region-cold-%s' % step_evt
        arr = np.array(((det.raw.raw(evt) & MDB) - evt_peds)*evt_gfac)
        CROP2_IMG = True
        AMIN, AMAX = amin_amax(args, amin_def=-5, amax_def=5)

    else:
        suffix = 'calib-%s' % step_evt
        arr = det.raw.calib(evt)
        AMIN, AMAX = amin_amax(args, amin_def=-100, amax_def=100)

    logger.info(info_ndarr(arr,'Event %d: %s' % (nevt, suffix)))

    img = det.raw.image(evt, nda=arr, vbase=-1, cframe=args.cframe)

    if img is None and arr is not None:
       logger.warning('image is None, but arr is not... > convert to 2d')
       #from psana.detector.NDArrUtils import info_ndarr, divide_protected, reshape_to_2d, save_ndarray_in_textfile
       img = reshape_to_2d(arr)


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

    logger.info(info_ndarr(img,'  img'))

    if flims is None:
        if img is None:
           sys.exit('EXIT - image is None - NOTHING TO DRAW')
        sh = img.shape
        h = 6
        w = min(sh[1]/sh[0]*h, 20)
        flims = fleximagespec(img, arr=arr, bins=100, w_in=w, h_in=h, amin=AMIN, amax=AMAX) #fraclo=0.01, frachi=0.99
        flims.move(10,20)
    else:
        fname = '%s-%s.png' % (prefix, suffix)
        flims.update(img, arr=arr, amin=AMIN, amax=AMAX)
        flims.axtitle('Event %d %s'%(nevt,fname))

    gr.show(mode=1)

    if tname in ('0','9') and args.saveimg:
        flims.save(fname)

gr.show()

if args.saveimg: flims.save(fname)

sys.exit('END OF %s -t %s' % (SCRNAME, tname))

# EOF
