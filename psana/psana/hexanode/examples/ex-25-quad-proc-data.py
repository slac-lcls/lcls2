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
          kwargs = {'consts':det.calibconst, 'events':1500, ...}
          proc = DLDProcessor(**kwargs)
          for x,y,r,t in proc.xyrt_list(nev, nhits, pktsec) :
"""

#----------

import logging
logger = logging.getLogger(__name__)

import sys
from time import time

from psana import DataSource
from psana.hexanode.WFPeaks import WFPeaks
from psana.hexanode.DLDProcessor import DLDProcessor

from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.pyalgos.generic.Utils import str_kwargs, do_print

#----------

USAGE = 'Use command: python %s [test-number]' % sys.argv[0]

#----------

def proc_data(**kwargs):

    logger.info(str_kwargs(kwargs, title='Input parameters:'))

    DSNAME       = kwargs.get('dsname', '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
    DETNAME      = kwargs.get('detname','tmo_quadanode')
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 10) + EVSKIP
    VERBOSE      = kwargs.get('verbose', False)

    ds    = DataSource(files=DSNAME)
    orun  = next(ds.runs())

    print('\nruninfo expt: %s  runnum: %d' % (orun.expt, orun.runnum))

    det   = orun.Detector(DETNAME)
    #kwargs['detobj'] = det
    kwargs['consts'] = det.calibconst

    peaks = WFPeaks(**kwargs)
    proc  = DLDProcessor(**kwargs) #detobj=det to get cfg/calib constants

    for nev,evt in enumerate(orun.events()):
        if nev<EVSKIP : continue
        if nev>EVENTS : break

        if do_print(nev) : logger.info('Event %3d'%nev)
        t0_sec = time()

        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)

        nhits, pkinds, pkvals, pktsec = peaks(wfs,wts) # ACCESS TO PEAK INFO

        if VERBOSE :
            print("  waveforms processing time = %.6f sec" % (time()-t0_sec))
            print_ndarr(wfs,    '  waveforms      : ', last=4)
            print_ndarr(wts,    '  times          : ', last=4)
            print_ndarr(nhits,  '  number_of_hits : ')
            print_ndarr(pktsec, '  peak_times_sec : ', last=4)

        for i,(x,y,r,t) in enumerate(proc.xyrt_list(nev, nhits, pktsec)) :
            print('    hit:%2d x:%7.3f y:%7.3f t:%10.5g r:%7.3f' % (i,x,y,t,r))

#----------

if __name__ == "__main__" :

    logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('%s\nTEST %s' % (50*'_', tname))

    kwargs = {#'dsname'   : '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2',
              'dsname'   : '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e001000.xtc2',
              'detname'  : 'tmo_quadanode',
              'numchs'   : 5,
              'numhits'  : 16,
              'evskip'   : 0,
              'events'   : 10,
              'ofprefix' : './',
              'run'      : 100,
              'exp'      : 'amox27716',
              'calibcfg' : '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/configuration_quad.txt',
              'calibtab' : '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/calibration_table_data.txt',
              'verbose'  : False,
              'command'  : 1, # if != 1 - overrides command from configuration file
              'detobj'   : None # get cfg & calib constants from detector object if specified
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

    kwargs.update(cfdpars)

    proc_data(**kwargs)

    print('\n%s' % USAGE)
    sys.exit('End of %s' % sys.argv[0])

#----------
