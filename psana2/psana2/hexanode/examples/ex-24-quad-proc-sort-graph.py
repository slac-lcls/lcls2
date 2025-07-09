#!/usr/bin/env python

"""
    See example in proc_data(**kwargs)

    - loop over events in psana dataset from xtc2 file
          ds = DataSource(files=DSNAME)
          orun  = next(ds.runs())
          for nev,evt in enumerate(orun.events()):

    - use Detector interface to get acqiris raw waveforms and sampling times
          det = orun.Detector(DETNAME)
          wts = det.raw.times(evt)
          wfs = det.raw.waveforms(evt)

    - process all Quad-DLD chanel waveforms using peakfinder
          peaks = WFPeaks(**kwargs)
          nhits, pkinds, pkvals, pktsec = peaks(wfs,wts)

    - process found peaktimes using Roentdec library algorithms,
      get and use sorted hit information
          proc = DLDProcessor(**kwargs)
          for x,y,r,t in proc.xyrt_list(nev, nhits, pktsec):

    - accumulate per event DLDProcessor internal info in
          stats = DLDStatistics(proc,**kwargs)
          stats.fill_data(nhits, pktsec)

    - plot DLDStatistics arrays at the end of the event loop
          from psana2.hexanode.DLDGraphics import draw_plots
          draw_plots(stats, prefix=OFPREFIX, do_save=True, hwin_x0y0=(0,10))
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys
from time import time

from psana import DataSource
from psana2.hexanode.WFPeaks import WFPeaks
from psana2.hexanode.DLDProcessor  import DLDProcessor
from psana2.hexanode.DLDStatistics import DLDStatistics
from psana2.hexanode.DLDGraphics   import draw_plots

from psana2.pyalgos.generic.NDArrUtils import print_ndarr
from psana2.pyalgos.generic.Utils import str_kwargs, do_print
from psana2.hexanode.examples.ex_test_data import DIR_DATA_TEST

DIR_ABSPATH = os.path.abspath(os.path.dirname(__file__)) # absolute path to .../psana/hexanode/examples
FNAME = '%s/%s' % (DIR_DATA_TEST, 'data-amox27716-r0100-acqiris-e001000.xtc2')
USAGE = 'Usage: python %s' % sys.argv[0]


def proc_data(**kwargs):

    logger.info(str_kwargs(kwargs, title='Input parameters:'))

    DSNAME       = kwargs.get('dsname', FNAME)
    DETNAME      = kwargs.get('detname','tmo_quadanode')
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 100) + EVSKIP
    OFPREFIX     = kwargs.get('ofprefix','./')
    VERBOSE      = kwargs.get('verbose', False)

    ds    = DataSource(files=DSNAME)
    orun  = next(ds.runs())
    det   = orun.Detector(DETNAME)

    print('XXX det.calibconst', det.calibconst)

    kwargs['consts'] = det.calibconst if det.calibconst else None

    peaks = WFPeaks(**kwargs)
    proc  = DLDProcessor(**kwargs)
    stats = DLDStatistics(proc,**kwargs)

    for nev,evt in enumerate(orun.events()):
        if nev<EVSKIP: continue
        if nev>EVENTS: break

        if do_print(nev): logger.info('Event %3d'%nev)
        t0_sec = time()

        wts = det.raw.times(evt)
        wfs = det.raw.waveforms(evt)

        nhits, pkinds, pkvals, pktsec = peaks(wfs,wts)

        if VERBOSE:
            print("  waveforms processing time = %.6f sec" % (time()-t0_sec))
            print_ndarr(wfs,    '  waveforms     : ', last=4)
            print_ndarr(wts,    '  times         : ', last=4)
            print_ndarr(nhits,  '  number_of_hits: ')
            print_ndarr(pktsec, '  peak_times_sec: ', last=4)

        proc.event_proc(nev, nhits, pktsec)

        stats.fill_data(nhits, pktsec)

        if VERBOSE:
            for i,(x,y,r,t) in enumerate(proc.xyrt_list(nev, nhits, pktsec)):
                 print('    hit:%2d x:%7.3f y:%7.3f t:%10.5g r:%7.3f' % (i,x,y,t,r))

    draw_plots(stats, prefix=OFPREFIX, do_save=True, hwin_x0y0=(0,10))


if __name__ == "__main__":

    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'
    logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('%s\nTEST %s' % (50*'_', tname))

    kwargs = {'dsname'   : FNAME,
              'detname'  : 'tmo_quadanode',
              'numchs'   : 5,
              'numhits'  : 16,
              'evskip'   : 0,
              'events'   : 1000,
              'ofprefix' : 'figs-DLD/plot',
              'run'      : 100,
              'exp'      : 'amox27716',
              'calibcfg' : '%s/configuration_quad.txt' % DIR_ABSPATH,
              'calibtab' : '%s/calibration_table_data.txt' % DIR_ABSPATH,
              'verbose'  : False,
             }

    # Parameters of the CFD descriminator for hit time finding algotithm
    cfdpars= {'cfd_base'       :  0.,
              'cfd_thr'        : -0.05,
              'cfd_cfr'        :  0.85,
              'cfd_deadtime'   :  10.0,
              'cfd_leadingedge':  True,
              'cfd_ioffsetbeg' :  1000,
              'cfd_ioffsetend' :  2000,
              'cfd_wfbinbeg'   :  6000,
              'cfd_wfbinend'   : 22000,
             }

    # On/Off statistical parameters
    statpars={'STAT_NHITS'         : True,
              'STAT_TIME_CH'       : False,
              'STAT_REFLECTIONS'   : False,
              'STAT_UVW'           : False,
              'STAT_TIME_SUMS'     : True,
              'STAT_CORRELATIONS'  : False,
              'STAT_XY_COMPONENTS' : False,
              'STAT_XY_2D'         : False,
              'STAT_PHYSICS'       : True,
              'STAT_MISC'          : False,
             }

    kwargs.update(cfdpars)
    kwargs.update(statpars)

    proc_data(**kwargs)

    print('\n', USAGE)
    sys.exit('End of %s' % sys.argv[0])

# EOF
