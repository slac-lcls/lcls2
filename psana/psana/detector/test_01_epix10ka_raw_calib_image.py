#!/usr/bin/env python

import sys
import logging
logger = logging.getLogger(__name__)

from time import time
t0_sec = time()
import numpy as np
dt_sec = time()-t0_sec
print('np import consumed time (sec) = %.6f' % dt_sec)

from psana.pyalgos.generic.NDArrUtils import info_ndarr

#exit('TEST EXIT')
#----

fname0 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2'
fname1 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005-seg1and3.xtc2'

#----

def print_det_raw_attrs(det):
    print('dir(det):', dir(det))
    print('det._dettype:', det._dettype)
    print('det._detid:', det._detid)
    print('det.calibconst:', det.calibconst)
    r = det.raw
    print('dir(det.raw):', dir(r))
    print('r._add_fields:', r._add_fields)
    print('r._calibconst:', r._calibconst)
    print('r._common_mode:', r._common_mode)
    print('r._configs:', r._configs)
    print('r._det_name:', r._det_name)
    print('r._dettype:', r._dettype)
    print('r._drp_class_name:', r._drp_class_name)
    print('r._env_store:', r._env_store)
    print('r._info:', r._info)
    print('r._return_types:', r._return_types)
    print('r._segments:', r._segments)
    print('r._sorted_segment_ids:', r._sorted_segment_ids)
    print('r._uniqueid:', r._uniqueid)
    print('r._var_name:', r._var_name)
    print('r.raw:', r.raw)
    #print('r.dtype:', r.dtype)


def test_calib_constants_directly(expname, runnum, detnameid):
    logger.info('in test_calib_constants_directly')
    from psana.pscalib.calib.MDBWebUtils import calib_constants

    pedestals, _ = calib_constants(detnameid, exp=expname, ctype='pedestals',    run=runnum)
    gain, _      = calib_constants(detnameid, exp=expname, ctype='pixel_gain',   run=runnum)
    rms, _       = calib_constants(detnameid, exp=expname, ctype='pixel_rms',    run=runnum)
    status, _    = calib_constants(detnameid, exp=expname, ctype='pixel_status', run=runnum)

    logger.info(info_ndarr(pedestals, 'pedestals'))
    logger.info(info_ndarr(gain,      'gain     '))
    logger.info(info_ndarr(rms,       'rms      '))
    logger.info(info_ndarr(status,    'status   '))


def det_calib_constants(det, ctype):
    calib_const = det.calibconst if hasattr(det,'calibconst') else None

    if calib_const is not None:
      logger.info('det.calibconst.keys(): ' + str(calib_const.keys()))
      cdata, cmeta = calib_const[ctype]
      logger.info('%s meta: %s' % (ctype, str(cmeta)))
      logger.info(info_ndarr(cdata, '%s data'%ctype))
      return cdata, cmeta
    else:
      logger.warning('det.calibconst is None')
      return None, None


def ds_run_det(fname, args):
    logger.info('ds_run_det input file:\n  %s' % fname)

    from psana import DataSource
    ds = DataSource(files=fname)
    orun = next(ds.runs())
    det = orun.Detector(args.detname)

    if args.pattrs:
      print('dir(orun):', dir(orun))
      print('dir(det):', dir(det))
      print_det_raw_attrs(det)

    from psana.pyalgos.generic.Utils import str_attributes
    print(str_attributes(orun, cmt='\nattributes of orun %s:'% str(orun), fmt=', %s'))

    oraw = det.raw
    detnameid = oraw._uniqueid
    expname = orun.expt if orun.expt is not None else args.expname # 'mfxc00318'
    runnum = orun.runnum
    print('expname:', expname)
    print('runnum:', runnum)
    #print('detname:', oraw._det_name)
    print('detname:', det._det_name)
    print('split detnameid:', '\n'.join(detnameid.split('_')))

    print(50*'=')
    test_calib_constants_directly(expname, runnum, detnameid)

    peds_data, peds_meta = det_calib_constants(det, 'pedestals')

    #sys.exit('TEST EXIT')
    return ds, orun, det


def test_raw(fname, args):
    logger.info('in test_raw data from file:\n  %s' % fname)
    ds, run, det = ds_run_det(fname, args)

    for evnum,evt in enumerate(run.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        segs = det.raw.segments(evt)
        raw  = det.raw.raw(evt)
        logger.info(info_ndarr(segs, 'segs '))
        logger.info(info_ndarr(raw,  'raw  '))
    print(50*'-')


def test_image(fname, args):

    logger.info('in test_image data from file:\n  %s' % fname)
    ds, run, det = ds_run_det(fname, args)

    gr, fig, axim, axcb = None, None, None, None
    imsh, cbar = None, None
    h_in = 8 # inch figure height

    if args.dograph:
        import psana.pyalgos.generic.Graphics as gg
        from UtilsGraphics import gr
        fig = gr.figure(figsize=(9,8), title='Image', dpi=80, facecolor='w', edgecolor='w', frameon=True, move=None)# **kwargs)
        fig, axim, axcb = gr.fig_img_cbar_axes(fig=fig, win_axim=(0.05,0.03,0.87,0.93), win_axcb=(0.923,0.03,0.02,0.93))

    for evnum,evt in enumerate(run.events()):
        print('%s\nEvent %04d' % (50*'_',evnum))
        img = det.raw.image(evt)
        logger.info(info_ndarr(img, 'image '))

        if args.dograph:
            arr = det.raw.calib(evt)
            #ave, rms = arr.mean(), arr.std()
            med = np.median(arr)
            spr = np.median(np.abs(arr-med))
            amin, amax = med-1*spr, med+3*spr
            print('median:%.1f spread:%.1f amin:%.1f amax:%.1f' % (med, spr, amin, amax))

            if imsh is None :
              asp_ratio = float(img.shape[1])/img.shape[0]
              fig.set_size_inches(h_in*asp_ratio, h_in)
              imsh, cbar = gr.imshow_cbar(fig, axim, axcb, img, amin=amin, amax=amax, extent=None,\
                interpolation='nearest', aspect='auto', origin='upper',\
                orientation='vertical', cmap='inferno') #, **kwargs)
            else :
              imsh.set_data(img)
              imsh.set_clim(amin, amax)

        gr.show(mode=1)

    if args.dograph: gr.show()

    print(50*'-')

#----

if __name__ == "__main__":

    SCRNAME = sys.argv[0].rsplit('/')[-1]
    DICT_NAME_TO_LEVEL = logging._nameToLevel # {'INFO': 20, 'WARNING': 30, 'WARN': 30,...
    LEVEL_NAMES = [k for k in DICT_NAME_TO_LEVEL.keys() if isinstance(k,str)]
    STR_LEVEL_NAMES = ', '.join(LEVEL_NAMES)

    usage =\
        '\n  python %s <test-name> [optional-arguments]' % SCRNAME\
      + '\n  where test-name: '\
      + '\n    0 - test_raw("%s")'%fname0\
      + '\n    1 - test_raw("%s")'%fname1\
      + '\n    2 - test_image("%s")'%fname0\
      + '\n    3 - test_image("%s")'%fname1\

    d_loglev  = 'INFO' #'INFO' #'DEBUG'
    d_pattrs  = False
    d_dograph = True
    d_detname = 'epix10k2M'
    d_expname = 'mfxc00318'

    h_loglev  = 'logging level name, one of %s, def=%s' % (STR_LEVEL_NAMES, d_loglev)

    import argparse

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('tname', type=str, help='test name')
    parser.add_argument('-l', '--loglev', default=d_loglev, type=str, help=h_loglev)
    parser.add_argument('-P', '--pattrs', default=d_pattrs, action='store_true', help='print objects attrubutes, def=%s' % d_pattrs)
    parser.add_argument('-G', '--dograph', default=d_dograph, action='store_false', help='plot graphics, def=%s' % d_pattrs)
    parser.add_argument('-d', '--detname', default=d_detname, type=str, help='detector name, def=%s' % d_detname)
    parser.add_argument('-e', '--expname', default=d_expname, type=str, help='experiment name, def=%s' % d_expname)

    args = parser.parse_args()
    kwa = vars(args)
    s = '\nArguments:'
    for k,v in kwa.items(): s += '\n %8s: %s' % (k, str(v))

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=DICT_NAME_TO_LEVEL[args.loglev])
    #logger.setLevel(intlevel)

    logger.info(s)

    tname = args.tname
    if   tname=='0': test_raw  (fname0, args)
    elif tname=='1': test_raw  (fname1, args)
    elif tname=='2': test_image(fname0, args)
    elif tname=='3': test_image(fname1, args)
    else: logger.warning('NON-IMPLEMENTED TEST: %s' % tname)

    sys.exit('END OF %s' % SCRNAME)

#----

