#!/usr/bin/env python
#----------
"""1. In psana event loop access acqiris waveforms - 2d array shape = (NUM_CHANNELS, NUM_HITS),
   2. applys CFD method to all channel waveforms and 
      find number of hits[NUM_CHANNELS] and array of hit times[NUM_CHANNELS,NUM_HITS]
   3. saves CFD results in hdf5 file per event,
   4. at the end saves in hdf5 file a few summary parameters.
"""
#----------

import logging
logger = logging.getLogger(__name__)

import os
import sys
from time import time

from psana import DataSource
from psana.hexanode.WFPeaks import WFPeaks
from psana.hexanode.WFHDF5IO import open_output_h5file

from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.pyalgos.generic.Utils import str_kwargs

#----------

def usage():
    return 'Use command: python %s [test-number]' % sys.argv[0]

#----------

def proc_data(**kwargs):

    DSNAME       = kwargs.get('dsname', '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
    DETNAME      = kwargs.get('detname','tmo_hexanode')
    NUM_CHANNELS = kwargs.get('numchs', 5)
    NUM_HITS     = kwargs.get('numhits', 16)
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 10) + EVSKIP
    EXP          = kwargs.get('exp', 'amox27716')
    RUN          = kwargs.get('run', 100)
    OFPREFIX     = kwargs.get('ofprefix','./')
    ofname       = '%s%s-r%04d-e%06d-single-node.h5' % (OFPREFIX, EXP, int(RUN), EVENTS)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('TEST %s' % tname)

    logger.info(str_kwargs(kwargs, title='Input parameters:'))

    peaks = WFPeaks(**kwargs)
    ofile = open_output_h5file(ofname, peaks, **kwargs)

    ds = DataSource(files=DSNAME)
    orun = next(ds.runs())
    det = orun.Detector(DETNAME)

    for nevt,evt in enumerate(orun.events()):
        if nevt<EVSKIP : continue
        if nevt>EVENTS : break
        print('Event %3d'%nevt)
        wts = det.raw.times(evt)
        wfs = det.raw.waveforms(evt)

        t0_sec = time()

        # ACCESS FOR TEST 1:
        if tname == '1' :
            nhits, pkinds, pkvals, pktns = peaks(wfs,wts)

        # ALTERNATIVE ACCESS:
        else :
            peaks.proc_waveforms(wfs, wts) 
            nhits  = peaks.number_of_hits(wfs, wts)
            pktns  = peaks.peak_times_ns(wfs, wts)
            pkvals = peaks.peak_values(wfs, wts)
            pkinds = peaks.peak_indexes(wfs, wts)

        print("  waveforms processing time = %.6f sec" % (time() - t0_sec))
        print_ndarr(wfs,   '  waveforms      : ', last=4)
        print_ndarr(nhits, '  number_of_hits : ')
        print_ndarr(wts,   '  times          : ', last=4)
        print_ndarr(pktns, '  peak_times_ns  : ', last=4)

        ofile.add_event_to_h5file()

    print('\n',usage())
    sys.exit('TEST %s EXIT' % tname)

#----------

if __name__ == "__main__" :

    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'
    logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

    print(50*'_')
    kwargs = {'dsname'   : '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2',
              'detname'  : 'tmo_hexanode',
              'numchs'   : 5,
              'numhits'  : 16,
              'evskip'   : 0,
              'events'   : 5,
              'ofprefix' : './',
              'run'      : 100,
              'exp'      : 'amox27716',
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

    sys.exit('End of %s' % sys.argv[0])

#----------
