#!/usr/bin/env python
"""
Test datasets:
       datinfo -k exp=ued1015999,run=185 -d epixquad1kfps
       datinfo -k exp=xppc00125,run=148 -d jungfrau1M
Example:
       detimage -k exp=ued1015999,run=185 -d epixquad1kfps
       detimage -k exp=xppc00125,run=148 -d jungfrau1M
"""
import sys
from psana.detector.UtilsLogging import logging, STR_LEVEL_NAMES, DICT_NAME_TO_LEVEL
logger = logging.getLogger(__name__)
import psana.detector.NDArrUtils as ndu # info_ndarr, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d


SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = '%s -k <dataset-kwargs> -d <detector-name> -n <number-of-events> -m <number-events-to-skip> -M <mode-r/c/p> -L <log-level-str>'%SCRNAME\
      + '\n  Examples:'\
      + '\n  %s -k exp=xppc00125,run=148 -d jungfrau1M --figsize 16,10 --aminmax 0,10'%SCRNAME\
      + '\n  %s -k exp=ued1015999,run=185 -d epixquad1kfps -M c'%SCRNAME\
      + '\n  %s -k exp=ued1015999,run=185 -d epixquad1kfps -M p --gmind 2 --databits 0x3fff'%SCRNAME\
      + '\n  %s -k exp=ued1016014,run=50 -d epixquad1kfps -M p --gmind 2 --databits 0x3fff'%SCRNAME\
      + '\n  %s -k exp=ued1016014,run=50 -d epixquad1kfps -M c --aminmax "(-10,10) --as2d"'%SCRNAME\

def argument_parser():
    import argparse

    d_dskwargs = 'exp=ued1015999,run=185'
    d_detname  = 'epixquad1kfps'
    d_events   = 5
    d_evskip   = 0
    d_logmode  = 'INFO'
    d_fnprefix = 'img'
    d_figsize  = '12,8'
    d_aminmax  = 'None, None'
    d_aslice   = ':'
    d_mode     = 'p'
    d_as2d     = False
    d_segind   = None
    d_gmind    = 0
    d_databits = '0x7fff'

    h_dskwargs = '(str) dataset name, default = %s' % d_dskwargs
    h_detname  = '(str) detector name, default = %s' % d_detname
    h_events   = '(int) number of events to collect, default = %s' % d_events
    h_evskip   = '(int) number of events to skip, default = %s' % d_evskip
    h_logmode  = '(str) logging mode, one of %s, default = %s' % (STR_LEVEL_NAMES, d_logmode)
    h_fnprefix = '(str) output file name fnprefix, default = %s' % str(d_fnprefix)
    h_figsize  = '(str) figure size in inch, width, hight, default = %s' % str(d_figsize)
    h_aminmax  = '(str) intensity min, max values or None, default = %s' % str(d_aminmax)
    h_aslice   = '(str) slice applied to image, e.g. 0:180,620:, default = %s' % str(d_aslice)
    h_mode     = '(str) mode of data: r/c/p : raw/calib/raw-peds default = %s' % str(d_aslice)
    h_as2d     = '(bool) plot nd-array as 2d, else as assembled image, default = %s' % str(d_as2d)
    h_segind   = '(int) segment index to mark or None, default = %s' % str(d_segind)
    h_gmind    = '(int) gain mode index to subtract pedestals for --mode p, default = %d' % d_gmind
    h_databits = '(str/hex) data bits mask, default = %s' % str(d_databits)

    parser = argparse.ArgumentParser(usage=USAGE, description='Draws area detector image in the event loop')
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str,   help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str,   help=h_detname)
    parser.add_argument('-n', '--events',   default=d_events,   type=int,   help=h_events)
    parser.add_argument('-m', '--evskip',   default=d_evskip,   type=int,   help=h_evskip)
    parser.add_argument('-L', '--logmode',  default=d_logmode,  type=str,   help=h_logmode)
    parser.add_argument('-f', '--fnprefix', default=d_fnprefix, type=str,   help=h_fnprefix)
    parser.add_argument(      '--figsize',  default=d_figsize,  type=str,   help=h_figsize)
    parser.add_argument(      '--aminmax',  default=d_aminmax,  type=str,   help=h_aminmax)
    parser.add_argument('-S', '--aslice',   default=d_aslice,   type=str,   help=h_aslice)
    parser.add_argument('-M', '--mode',     default=d_mode,     type=str,   help=h_mode)
    parser.add_argument(      '--as2d',     action='store_true',            help=h_as2d)
    parser.add_argument(      '--segind',   default=d_segind,   type=int,   help=h_segind)
    parser.add_argument(      '--gmind',    default=d_gmind,    type=int,   help=h_gmind)
    parser.add_argument(      '--databits', default=d_databits, type=str,   help=h_databits)

    return parser

import numpy as np
def seg_sunrise(sh=(512, 1024), dtype=np.int32, vmax=None):
    nrows, ncols = sh
    rows = np.arange(nrows, dtype=dtype)
    cols = np.arange(ncols, dtype=dtype)
    cs, rs = np.meshgrid(cols, rows)
    arr2d = cs + rs
    if vmax is not None: arr2d *= (vmax/(cs[-1]+rs[-1]))
    print('XXX vmax', vmax)
    print('arr2d.shape:', arr2d.shape)
    return arr2d

def segments_with_gaps(a, gappix=10, gapval=0):
    if a.ndim < 3: return a
    a3d = ndu.reshape_to_3d(a)
    n,r,c = a3d.shape
    gap = gapval * np.ones((gappix, c), dtype=a3d.dtype)
    lstarrs = [a3d[0,:],]
    for i in range(1,n):
        lstarrs.append(gap)
        lstarrs.append(a3d[i,])
    return np.vstack(lstarrs)


def detimage():
    """
    """
    import os
    import numpy as np
    from psana import DataSource
    import psana.detector.utils_psana as up
    from psana.detector.UtilsGraphics import gr, fleximage, fleximagespec
    #from time import sleep

    parser = argument_parser()
    args = parser.parse_args()

    kwa = vars(args)

    print('parser.parse_args: %s' % str(args))

    #if len(sys.argv)<3: sys.exit('%s\n EXIT - MISSING ARGUMENTS\n' % USAGE)

    str_dskwargs = args.dskwargs
    detname  = args.detname
    events   = args.events
    evskip   = args.evskip
    logmode  = args.logmode
    fnprefix = args.fnprefix
    figsize  = eval(args.figsize)
    aminmax  = eval(args.aminmax)
    aslice   = eval('np.s_[%s]' % args.aslice)
    mode     = args.mode
    amin, amax = eval(args.aminmax)
    as2d     = args.as2d
    segind   = args.segind
    gmind    = args.gmind
    aslice   = args.aslice
    databits = int(args.databits, 16)
    _slice = eval('np.s_[%s]' % aslice)
    gappix = kwa.get('gappix', 10)
    gapval = kwa.get('gapval', -100)

    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=DICT_NAME_TO_LEVEL[logmode])

    dskwargs = up.datasource_kwargs_from_string(str_dskwargs, detname=detname)
    dskwargs['max_events'] = events
    logger.info('DataSource kwargs: %s' % str(dskwargs))

    try:
      ds = DataSource(**dskwargs)
    except Exception as err:
      logger.error('DataSource(**dskwargs) does not work:\n    %s' % err)
      sys.exit('Exit processing')

    expname = dskwargs.get('exp', None)
    runnum  = dskwargs.get('run', None)

    run = next(ds.runs())

    if expname is None: expname = run.expt
    if runnum is None: runnum = run.runnum

    logger.info('\n==== run: %d exp: %s' % (runnum, expname))
    logger.info(up.info_run(run, cmt='run info:\n    ', sep='\n    ', verb=3))

    det = run.Detector(detname)

    geo_meta = det.calibconst['geometry']
    if geo_meta is not None: print('geometry metadata:', geo_meta[1])

    peds = det.raw._pedestals()
    print(ndu.info_ndarr(peds, 'det.raw._pedestals()', last=10))

    if True:
        plot_image = True
        flimg = None
        for nevt,evt in enumerate(run.events()):
            if nevt<evskip: continue
            raw = det.raw.raw(evt)
            print(ndu.info_ndarr(raw, '=== evt:%03d raw' % nevt, last=5))

            _raw = raw & databits if databits < 0xffff else raw

            arr = det.raw.calib(evt) if mode == 'c' else\
                  _raw

            if mode == 'p':
                _peds = peds if peds.ndim < 4 else peds[gmind,:] # assuming shape like: (3, <n-panels>, 512, 1024)
                print(ndu.info_ndarr(_peds, '             peds[%d,:]' % gmind, last=5))
                arr = arr.astype(peds.dtype) - _peds

            if segind is not None:
                sh = arr.shape
                arr[segind,:] = seg_sunrise(sh=(sh[-2],sh[-1]), dtype=arr.dtype, vmax=np.quantile(arr, 0.98, method='linear'))
            #img = ndu.reshape_to_2d(arr) if as2d else\
            img = segments_with_gaps(arr, gappix=gappix, gapval=-gapval) if as2d else\
                  det.raw.image(evt, arr)

            if not aslice in (':', None):
                img = img[_slice]

            print(ndu.info_ndarr(img, '             img', last=5))

            if plot_image:
                if flimg is None:
                    flimg = fleximagespec(img, h_in=figsize[1], w_in=figsize[0], amin=aminmax[0], amax=aminmax[1])
                    gr.plt.ion()
                title = 'evt %02d test image'%nevt
                #flimg.fig.suptitle(title, fontsize=16)
                gr.set_win_title(flimg.fig, titwin=title)
                flimg.update(img)
                gr.show(mode='DO NOT HOLD', pause_sec=1)
        if plot_image: gr.show()

        if fnprefix:
            _fnprefix = '%s-%s-%s-r%04d-e%06d-mode-%s' % (fnprefix, detname, expname, runnum, nevt, mode)
            gr.save_fig(flimg.fig, fname=f'{_fnprefix}-img.png') #, prefix=None, suffix='.png')


#if __name__ == "__main__":
detimage()
sys.exit('END OF %s' % SCRNAME)

# EOF
