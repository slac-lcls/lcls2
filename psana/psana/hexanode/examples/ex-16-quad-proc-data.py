#!/usr/bin/env python

"""
    Example for methods:
   
    peaks = WFPeaks(**kwargs)
    proc  = DLDProcessor(**kwargs)
    # in event loop : 
        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)
        nhits, pkinds, pkvals, pktns = peaks(wfs,wts)
        for x,y,r,t in proc.xyrt_list(nhits, pktns) :

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

def usage(): return 'Use command: python %s [test-number]' % sys.argv[0]

#----------

def proc_data(**kwargs):

    logger.info(str_kwargs(kwargs, title='Input parameters:'))

    DSNAME       = kwargs.get('dsname', '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
    DETNAME      = kwargs.get('detname','tmo_hexanode')
    NUM_CHANNELS = kwargs.get('numchs', 5)
    NUM_HITS     = kwargs.get('numhits', 16)
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 10) + EVSKIP
    EXP          = kwargs.get('exp', 'amox27716')
    RUN          = kwargs.get('run', 100)
    OFPREFIX     = kwargs.get('ofprefix','./')

    peaks = WFPeaks(**kwargs)
    proc  = DLDProcessor(**kwargs)

    ds    = DataSource(files=DSNAME)
    orun  = next(ds.runs())
    det   = orun.Detector(DETNAME)

    for nevt,evt in enumerate(orun.events()):
        if nevt<EVSKIP : continue
        if nevt>EVENTS : break

        if do_print(nevt) : logger.info('Event %3d'%nevt)
        t0_sec = time()

        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)

        nhits, pkinds, pkvals, pktns = peaks(wfs,wts) # ACCESS TO PEAK INFO

        print("  waveforms processing time = %.6f sec" % (time()-t0_sec))
        print_ndarr(wfs,   '  waveforms      : ', last=4)
        print_ndarr(wts,   '  times          : ', last=4)
        print_ndarr(nhits, '  number_of_hits : ')
        print_ndarr(pktns, '  peak_times_ns  : ', last=4)
        print('XXX 1')
        proc.event_proc(nevt, nhits, pktns)
        print('XXX 2')

        #for i,(x,y,r,t) in enumerate(proc.xyrt_list(nevt, nhits, pktns)) :
        #    print('  hit:%1d  x:%5.3f  y:%5.3f  r:%5.3f  t:%5.3f' % (i,x,y,r,t))

#----------
#----------
#----------
#----------

if __name__ == "__main__" :

    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'
    logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('%s\nTEST %s' % (50*'_', tname))

    kwargs = {'dsname'   : '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2',
              'detname'  : 'tmo_hexanode',
              'numchs'   : 5,
              'numhits'  : 16,
              'evskip'   : 0,
              'events'   : 10,
              'ofprefix' : './',
              'run'      : 100,
              'exp'      : 'amox27716',
              'calibcfg' : '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/configuration_quad.txt',
              'calibtab' : '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/calibration_table_data.txt',
             }

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

    print('\n',usage())
    sys.exit('End of %s' % sys.argv[0])

#----------
