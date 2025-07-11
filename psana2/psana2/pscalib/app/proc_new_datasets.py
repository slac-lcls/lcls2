#####!/usr/bin/env python
#------------------------------

import os
import sys
from time import time, sleep

import psana2.pscalib.proc.RunProcUtils as rpu
from   psana.pscalib.proc.SubprocUtils import subproc, number_of_batch_jobs, batch_job_ids, batch_job_kill
#import psana.pscalib.proc.SubprocUtils as spu
#import psana.pyalgos.generic.Utils as gu


#import PSCalib.RunProcUtils as rpu
#from PSCalib.SubprocUtils import subproc, number_of_batch_jobs, batch_job_ids, batch_job_kill
##import PSCalib.SubprocUtils as spu
##import PSCalib.GlobalUtils as gu

#------------------------------

def subprocess_command(exp='xpptut15', run='0260', procname='pixel_status', qname='psnehq',\
                       dt_sec=60, sources='cspad,opal,epix100,pnccd,princeton,andor') :
    """Returns command like 'proc_control -e xpptut15 -r 0260 -p pixel_status -q psnehq -t 60 -s cspad,pnccd'
    """
    if procname == 'pixel_status' :
        return 'proc_control -e %s -r %s -p %s -q %s -t %d -s %s' %\
                             (exp, str(run), procname, qname, dt_sec, sources)
    else :
        return None

#------------------------------

def proc_exp_runs(exp_runs, procname='pixel_status', do_proc=False, qname='psnehq', njobs=5,\
                       dt_sec=60, sources='cspad,opal,epix100,pnccd,princeton,andor') :

    njobs_in_queue = number_of_batch_jobs(qname=qname)
    print('%d jobs found in queue %s' % (njobs_in_queue, qname))

    for kstatus in ('SSUSP','UNKWN') :
      for id in batch_job_ids(status=kstatus, user=None, qname=qname) : 
        print('  kill %s job %s' % (kstatus, id))
        batch_job_kill(id, user=None, qname=qname, addopts='-r')

    for i,(exp,run) in enumerate(exp_runs) :
        dsname = 'exp=%s:run=%s'%(exp, run.lstrip('0'))
        logname = rpu.log_file(exp, procname)
        print('%4d %s %4s %s %s'%(i+1, exp.ljust(10), run, dsname.ljust(22), logname))
        #--------------
        gap = 5*' '
        if do_proc and i<njobs and njobs_in_queue<njobs: 
            cmd = subprocess_command(exp, run, procname, qname, dt_sec, sources)
            if cmd is None : raise IOError('ERROR: batch submission command is None...')

            print('%sStart subprocess: %s' % (gap, cmd))
            t0_sec = time()
            out, err = subproc(cmd, env=None, shell=False, do_wait=False)
            print('%sSubprocess starting time dt=%7.3f sec' % (gap, time()-t0_sec))
            #print('%sSubprocess starting response:\n       out: %s\n       err: "%s"'%\
            #      (gap, out, err.strip('\n')))

            # mark dataset as processed in any case
            #==============================================
            rpu.append_log_file(exp, procname, [str(run),])
            #==============================================
        #--------------
    print('%d new runs found' % (len(exp_runs)))

#------------------------------
#------------------------------

def proc_new_datasets(parser) :
    """Finds and processes new datasets.

       - Finds list of non-processed experimental runs
       - Submit jobs for specified number of subprocesses
    """

    (popts, pargs) = parser.parse_args()
    procname = popts.pro
    qname    = popts.que
    submit   = popts.sub
    njobs    = popts.njb
    mode     = popts.mod
    dt_sec   = popts.dts
    sources  = popts.srs

    #tstamp = gu.str_tstamp('%Y-%m-%dT%H:%M:%S', time())

    #print('default command: %s' % subprocess_command())
    #print_datasets_new_under_control(procname, add_to_log=False)

    rpu.print_explogs_under_control(procname)

    exp_runs = rpu.exp_run_new(None, procname) if mode == 'ALL' else\
               rpu.exp_run_new_under_control(procname)
    proc_exp_runs(exp_runs, procname, submit, qname, njobs, dt_sec, sources)

#------------------------------

def usage() : 
    return "\n%prog [-p <process-name> -q <queue-name> -m <mode> -s]"\
           "\n  Ex.1: %prog -p pixel_status -q psnehq -m ALL -s"\
           "\n  Ex.2: %prog     <--- lists non-processed datasets"\
           "\n  Ex.3: %prog -S  <--- submits non-processed datasets for processing"

#------------------------------

def input_option_parser() :

    from optparse import OptionParser
    d_pro = 'pixel_status'
    d_que = 'psnehq'
    d_njb = 2
    d_mod = 'LIST'
    d_sub = False
    d_dts = 60 # sec
    d_srs = 'cspad,opal,epix100,pnccd,princeton,andor'

    h_pro = 'data processor name, default = %s' % d_pro
    h_que = 'batch queue name, default = %s' % d_que
    h_njb = 'maximal number of subprocesses/batch jobs to submit, default = %d' % d_njb
    h_mod = 'processing mode ALL or LIST, default = %s' % d_mod
    h_sub = 'start subrocess, default = %s' % d_sub
    h_dts = 'wait time (sec) between job status check, default = %.1f' % d_dts
    h_srs = 'sources of data (detector types), default = %s' % d_srs

    parser = OptionParser(description='Finds non-processed datasets and begin processing.', usage=usage())
    parser.add_option('-p', '--pro', default=d_pro, action='store', type='string', help=h_pro)
    parser.add_option('-q', '--que', default=d_que, action='store', type='string', help=h_que)
    parser.add_option('-n', '--njb', default=d_njb, action='store', type='int',    help=h_njb)
    parser.add_option('-m', '--mod', default=d_mod, action='store', type='string', help=h_mod)
    parser.add_option('-S', '--sub', default=d_sub, action='store_true',           help=h_sub)
    parser.add_option('-t', '--dts', default=d_dts, action='store', type='float',  help=h_dts)
    parser.add_option('-s', '--srs', default=d_srs, action='store', type='string', help=h_srs)
 
    return parser #, parser.parse_args()
  
#------------------------------

def do_main() :

    parser = input_option_parser()

    if len(sys.argv) == 1 : 
        parser.print_help()
        #sys.exit('WARNING: using ALL default parameters...')
        print('WARNING: using ALL default parameters...')

    proc_new_datasets(parser)
    sys.exit(0)

#------------------------------

if __name__ == "__main__" :
    do_main()

#------------------------------
