#!/usr/bin/env python
"""
   Reads xtc2 processes waveforms and saves peak info in hdf5.

   The same as ex-25-quad-proc-data.py, 
   BUT:
       - do not use DLDProcessor
       + saves wf times and nhits in HDF5 file
   peak time units: [sec]
"""

#----------

import logging
logger = logging.getLogger(__name__)

import sys
from time import time

from psana import DataSource
from psana.hexanode.WFPeaks import WFPeaks
from psana.hexanode.WFHDF5IO import open_output_h5file

from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.pyalgos.generic.Utils import str_kwargs, do_print

#----------

USAGE = 'Use command: python %s [test-number]' % sys.argv[0]

#----------

def proc_data(**kwargs):

    logger.info(str_kwargs(kwargs, title='Input parameters:'))

    DSNAME       = kwargs.get('dsname', '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e001000.xtc2')
    DETNAME      = kwargs.get('detname','tmo_quadanode')
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 10) + EVSKIP
    EXP          = kwargs.get('exp', 'amox27716')
    RUN          = kwargs.get('run', 100)
    VERBOSE      = kwargs.get('verbose', False)
    OFPREFIX     = kwargs.get('ofprefix','./')
    ofname       = '%s%s-r%04d-e%06d-ex-22_lowerthresh.h5' % (OFPREFIX, EXP, int(RUN), EVENTS)

    peaks = WFPeaks(**kwargs)
    ofile = open_output_h5file(ofname, peaks, **kwargs)

    ds    = DataSource(files=DSNAME)
    orun  = next(ds.runs())
    det   = orun.Detector(DETNAME)

    for nev,evt in enumerate(orun.events()):
    
        if nev<EVSKIP : continue
        if nev>EVENTS : break

        if do_print(nev) : logger.info('Event %4d'%nev)
        t0_sec = time()

        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)

        nhits, pkinds, pkvals, pktsec = peaks(wfs,wts) # ACCESS TO PEAK INFO

        if VERBOSE :
            print("  ev:%4d waveforms processing time = %.6f sec" % (nev, time()-t0_sec))
            print_ndarr(wfs,    '    waveforms      : ', last=4)
            print_ndarr(wts,    '    times          : ', last=4)
            print_ndarr(nhits,  '    number_of_hits : ')
            print_ndarr(pktsec, '    peak_times_sec : ', last=4)

        ofile.add_event_to_h5file()

#----------

if __name__ == "__main__" :

    logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('%s\nTEST %s' % (50*'_', tname))

    kwargs = {'dsname'   : '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e001000.xtc2',
              'detname'  : 'tmo_quadanode',
              'numchs'   : 5,
              'numhits'  : 16,
              'evskip'   : 0,
              'events'   : 10000,
              'ofprefix' : './',
              'run'      : 100,
              'exp'      : 'amox27716',
              'version'  : 4,
              'DLD': True,
              'paramsCFD' : {0: {'channel': 'mcp',
                              'delay': 3.068e-09,
                              'fraction': 0.35,
                              'offset': 0.054470544805354439,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.056374120466532174,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                              1: {'channel': 'x1',
                              'delay': 3.997500000000001e-09,
                              'fraction': 0.35,
                              'offset': 0.032654320557811034,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.048439800379417808,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                             2: {'channel': 'x2',
                               'delay': 4.712500000000001e-09,
                              'fraction': 0.35,
                              'offset': 0.058295909692775157,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.062173077232695384,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                             3: {'channel': 'y1',
                              'delay': 4.5435e-09,
                              'fraction': 0.35,
                              'offset': 0.01740340726630819,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.035850750860370109,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                             4: {'channel': 'y2',
                              'delay': 4.140500000000001e-09,
                              'fraction': 0.35,
                              'offset': 0.0088379291811293368,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.035254198205580331,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0}}}
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
