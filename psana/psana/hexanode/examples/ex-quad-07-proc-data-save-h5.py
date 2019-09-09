#!/usr/bin/env python
#------------------------------
"""1. Reads in loop events with hexanode-acqiris waveforms,
   2. applys CFD method to 5 waveforms (MCP, and 2 delay line outputs) and find number of hits and array of hit times,
   3. saves per channel number of hits and array of hit times in hdf5 file,
   4. at the end saves in hdf5 file a few summary parameters.
"""
#------------------------------

import os
import sys
from time import time
from expmon.HexDataIO import HexDataIO, do_print

from expmon.PSUtils import exp_run_from_dsname # event_time
#from pyimgalgos.GlobalUtils import print_ndarr

#------------------------------

def usage():
    return 'Use command: python hexanode/examples/ex-quad-07-proc-data-save-h5.py'

#------------------------------

def proc_data(**kwargs):

    print usage()
    SRCCHS       = kwargs.get('srcchs', {'AmoEndstation.0:Acqiris.1':(2,3,4,5,6)})    # ,'AmoITOF.0:Acqiris.0':(0,)})
    DSNAME       = kwargs.get('dsname', 'exp=amox27716:run=100') #'exp=amox27716:run=100:smd'
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 1200) + EVSKIP
    OFPREFIX     = kwargs.get('ofprefix','./')
    NUM_CHANNELS = kwargs.get('numchs', 5)
    NUM_HITS     = kwargs.get('numhits', 16)

    print 'Input parameters:'
    for k,v in kwargs.iteritems() : print '%20s : %s' % (k,str(v))

    DIO = HexDataIO(**kwargs) # alternative transition of parameters
    DIO.open_input_dataset(DSNAME, pbits=0, do_mpids=False)

    DIO.print_wf_hit_finder_parameters()

    exp, run = exp_run_from_dsname(DSNAME)
    ofname = '%s%s-r%04d-e%06d-single-node.h5' % (OFPREFIX, exp, int(run), EVENTS)
    DIO.open_output_h5file(ofname)

    print 'DIO experiment: %s' % DIO.experiment()
    print 'DIO run       : %s' % DIO.run()
    print 'DIO start time: %s' % DIO.start_time()
    print 'DIO stop time : %s' % DIO.stop_time()
    print 'DIO tdc_resolution    : %.3f ns' % DIO.tdc_resolution()
    print 'DIO number of channels: %2d' % DIO.get_number_of_channels()

    t0_sec = time()
    t1_sec = time()

    while DIO.read_next_event():
        evnum = DIO.event_number()
        if evnum < EVSKIP : continue

        if do_print(evnum) :
            t1 = time()
            print 'Event: %06d, dt(sec): %.3f' % (evnum, t1 - t1_sec), 
            t1_sec = t1

            #DIO.print_tdc_data()
            #DIO.print_times()
            print 'nhits:', DIO.get_number_of_hits_array()
            #gu.print_ndarr(nhits, '    nhits', first=0, last=7)

        DIO.add_event_to_h5file()

        if evnum > EVENTS-2 : break

    print "%d events processed, consumed time = %.6f sec\n" % (DIO.nevents_processed(), time() - t0_sec)
    DIO.close_output_h5file(pbits=1)

#------------------------------

if __name__ == "__main__" :

    print 50*'_'
    #exp=amox27716:run=91
    #exp=amox27716:run=100
    kwargs = {'srcchs'   : {'AmoEndstation.0:Acqiris.1':(2,3,4,5,6)},
              'numchs'   : 5,
              'numhits'  : 16,
              'dsname'   : 'exp=amox27716:run=91',
              'evskip'   : 0,
              'events'   : 60000,
              'ofprefix' : './',
             }

    cfdpars= {'cfd_base'       :  0.,
              'cfd_thr'        : -0.04,
              'cfd_cfr'        :  0.9,
              'cfd_deadtime'   :  5.0,
              'cfd_leadingedge':  True,
              'cfd_ioffsetbeg' :  1000,
              'cfd_ioffsetend' :  2000,
              'cfd_wfbinbeg'   :  6000,
              'cfd_wfbinend'   : 22000,
             }

    kwargs.update(cfdpars)

    proc_data(**kwargs)

    sys.exit('End of %s' % sys.argv[0])

#------------------------------
