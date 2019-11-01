#!/usr/bin/env python

""" 
Module :py:class:`HexDataPreProc` - hexanode LCLS data processing using MPI and creating "small data" hdf5 file
===============================================================================================================

Created on 2017-12-08 by Mikhail Dubrovin
"""
#------------------------------

import sys
from time import time
import psana
from expmon.HexDataIO import HexDataIO, do_print
from expmon.PSUtils import event_time, exp_run_from_dsname
from pyimgalgos.GlobalUtils import print_ndarr, str_tstamp

#------------------------------

def usage():
    return 'Use command: mpirun -n 8 python hexanode/examples/ex-08-proc-MPIDS-save-h5.py\n'\
           'or:\n'\
           '   bsub -o log-mpi-n16-%J.log -q psnehq -n 16 mpirun python hexanode/examples/ex-08-proc-MPIDS-save-h5.py'

#------------------------------

def preproc_data(**kwargs):

    DSNAME       = kwargs.get('dsname', 'exp=xpptut15:run=390:smd')
    SRCCHS       = kwargs.get('srcchs', {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)})
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 1000000) + EVSKIP
    OFPREFIX     = kwargs.get('ofprefix','./')
    NUM_CHANNELS = kwargs.get('numchs', 7)
    NUM_HITS     = kwargs.get('numhits', 16)

    DIO = HexDataIO(srcchs=SRCCHS, numchs=NUM_CHANNELS, numhits=NUM_HITS)
    #DIO = HexDataIO(**kwargs) # alternative transition of parameters
    DIO.open_input_dataset(DSNAME, pbits=0, do_mpids=True)
    DIO.set_wf_hit_finder_parameters(**kwargs)

    exp, run = exp_run_from_dsname(DSNAME)
    ofname = '%s%s-r%04d-e%06d-n%02d-mpi.h5' % (OFPREFIX, exp, int(run), EVENTS, DIO.ds.size)

    if DIO.ds.master : # rank==0
        print usage()

        print 'Input parameters:'
        for k,v in kwargs.iteritems() : print '%20s : %s' % (k,str(v))

        print 'DIO dataset start time: %s' % DIO.start_time()
        print "number of MPI cores: %d" % DIO.ds.size

        DIO.print_wf_hit_finder_parameters()

    smldata = DIO.ds.small_data(ofname, keys_to_save=['nhits','tdcns','event_number'], gather_interval=4096)

    t0_sec = t1_sec = time()
    stop_time = start_time = None
    nev = 0
    for evt in DIO.events():

        nev = DIO.ds.event_number()

        if nev < EVSKIP: continue
        if nev >= EVENTS: break

	if do_print(nev) :
            t1 = time()
            print 'Rank: %d event: %06d, dt(sec): %.3f' % (DIO.ds.rank, nev, t1-t1_sec)
            t1_sec = t1

        if evt is None :
            print '  Event: %4d WARNING: evt is None, rank: %d' % (nev, DIO.ds.rank)
            continue

        if start_time is None :
            start_time = event_time(evt)
        stop_time = event_time(evt)

        DIO.proc_waveforms_for_evt(evt)

        #print_ndarr(DIO._number_of_hits, '    number of hits', first=0, last=7)
        #print_ndarr(DIO._tdc_ns, '    TDC[ns]', first=0, last=5)
 
        # save in hdf5 per-event data
        smldata.event(nhits=DIO._number_of_hits, tdcns=DIO._tdc_ns, event_number=nev)

    # get "summary" data
    # run_sum = smldata.sum(partial_run_sum)
    # save HDF5 file, including summary data
    dt = time() - t0_sec
    if DIO.ds.master : # rank==0
        print ''
        print "run start time(sec) = %.1f %s" % (start_time.seconds(), str_tstamp(time_sec=start_time.seconds()))
        print "run stop time (sec) = %.1f %s" % (stop_time.seconds(),  str_tstamp(time_sec=stop_time.seconds()))
        print "number of processing nodes: %d" % DIO.ds.size
        print "%d events processed" % nev
        print "consumed time (sec) = %.6f" % dt
        print "processing rate (Hz) = %.3f\n" % (float(nev+1)/dt)

    # save in hdf5 a few values at the end
    smldata.save(nevents=nev,\
                 start_time=start_time.seconds(),\
                 stop_time=stop_time.seconds(),\
                 proc_time_sec=dt,\
                 tdc_res_ns=DIO.tdc_resolution(),\
                 run=run,\
                 experiment=exp)

#------------------------------

if __name__ == "__main__" :
    print 50*'_'
    print 'See example in hexanode/examples/ex-08-proc-MPIDS-save-h5.py'\
          '\nand application expmon/app/hex_data_proc'

    #kwargs = {'events':1500,}
    #preproc_data(**kwargs)

#------------------------------
