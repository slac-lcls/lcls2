#!/usr/bin/env python

"""
    See example in proc_data(**kwargs)

    - loop over records in hdf5 file
      f = open_input_h5file(IFNAME, **kwargs)
      while f.next_event():
          nev = f.event_number()

    - get peak info
          nhits, pktsec = f.peak_arrays()
          #nhits  = f.number_of_hits()
          #pktsec = f.tdcsec()

    - process peaktimes using Roentdec library algorithms,
      get and use sorted hit information
          proc = DLDProcessor(**kwargs)
          for x,y,r,t in proc.xyrt_list(nev, nhits, pktsec):

    - accumulate per event DLDProcessor internal info in
          stats = DLDStatistics(proc,**kwargs)
          stats.fill_data(nhits, pktsec)

    - plot DLDStatistics arrays at the end of the event loop
          from psana.hexanode.DLDGraphics import draw_plots
          draw_plots(stats, prefix=OFPREFIX, do_save=True, hwin_x0y0=(0,10))
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys
from time import time

from psana.hexanode.WFHDF5IO import open_input_h5file
from psana.hexanode.DLDProcessor  import DLDProcessor
from psana.hexanode.DLDStatistics import DLDStatistics
from psana.hexanode.DLDGraphics   import draw_plots

from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.pyalgos.generic.Utils import str_kwargs, do_print
from psana.hexanode.examples.ex_test_data import DIR_DATA_TEST

DIR_ABSPATH = os.path.abspath(os.path.dirname(__file__)) # absolute path to .../psana/hexanode/examples
#FNAME = '/sdf/group/lcls/ds/ana/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5'
FNAME = '%s/%s' % (DIR_DATA_TEST, '../../data_test/hdf5/amox27716-r0100-e060000-single-node.h5')
USAGE = 'Use command: python %s' % sys.argv[0]


def proc_data(**kwargs):

    logger.info(str_kwargs(kwargs, title='Input parameters:'))

    IFNAME       = kwargs.get('ifname', FNAME)
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 10) + EVSKIP
    OFPREFIX     = kwargs.get('ofprefix','./')
    VERBOSE      = kwargs.get('verbose', False)

    proc  = DLDProcessor(**kwargs)
    stats = DLDStatistics(proc,**kwargs)

    f = open_input_h5file(IFNAME, **kwargs)
    print('  file: %s\n  number of events in file %d' % (IFNAME, f.events_in_h5file()))

    t0_sec = time()
    nev=0
    while f.next_event():
        nev = f.event_number()

        if nev<EVSKIP: continue
        if nev>EVENTS: break

        if do_print(nev): logger.info('Event %3d'%nev)

        nhits, pktsec = f.peak_arrays()

        if VERBOSE:
            print_ndarr(nhits,  '  number_of_hits: ')
            print_ndarr(pktsec, '  peak_times_sec: ', last=4)

        proc.event_proc(nev, nhits, pktsec)

        stats.fill_data(nhits, pktsec)

        if VERBOSE:
            for i,(x,y,r,t) in enumerate(proc.xyrt_list(nev, nhits, pktsec)):
                 print('    hit:%2d x:%7.3f y:%7.3f t:%10.5g r:%7.3f' % (i,x,y,t,r))

    dt = time()-t0_sec
    print('%d events processing time = %.3f sec or %.6f sec/event or %.3f Hz' % (nev, dt, dt/nev, nev/dt))

    draw_plots(stats, prefix=OFPREFIX, do_save=True, hwin_x0y0=(0,10))


if __name__ == "__main__":

    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'
    logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('%s\nTEST %s' % (50*'_', tname))

    kwargs = {'ifname': FNAME,
              'numchs'  : 5,
              'numhits' : 16,
              'evskip'  : 7,
              'events'  : 60000,
              'ofprefix': 'figs-DLD/plot',
              'run'     : 100,
              'exp'     : 'amox27716',
              'calibcfg' : '%s/configuration_quad.txt' % DIR_ABSPATH,
              'calibtab' : '%s/calibration_table_data.txt' % DIR_ABSPATH,
              'verbose' :  False,
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
             }\
        if False else\
             {'STAT_NHITS'         : True,
              'STAT_TIME_CH'       : True,
              'STAT_REFLECTIONS'   : True,
              'STAT_UVW'           : True,
              'STAT_TIME_SUMS'     : True,
              'STAT_CORRELATIONS'  : True,
              'STAT_XY_COMPONENTS' : True,
              'STAT_XY_2D'         : True,
              'STAT_PHYSICS'       : True,
              'STAT_MISC'          : True,
             }

    kwargs.update(cfdpars)
    kwargs.update(statpars)

    proc_data(**kwargs)

    print('\n', USAGE)
    sys.exit('End of %s' % sys.argv[0])

# EOF
