#####!/usr/bin/env python
#------------------------------

import os
import sys
from time import time, sleep

import psana2.pscalib.proc.SubprocUtils as spu
import psana2.pscalib.proc.RunProcUtils as rpu
import psana2.pyalgos.generic.Utils as gu

#------------------------------

def _batch_submit_command(exp='xpptut15', run='0260', procname='pixel_status', qname='psnehq', logfname='log.txt',\
                          ofprefix='work/nda-#exp-#run-#src-#evts-#type.txt',\
                          sources='cspad,opal,epix100,pnccd,princeton,andor') :
    dsname = rpu.dsname(exp, run)

    if procname == 'pixel_status' :
        #return 'bsub -q %s -o log-%%J.txt event_keys -d %s' % (qname, dsname)
        #return 'bsub -q %s -o %s event_keys -d %s' % (qname, logfname, dsname)
        #return 'bsub -q %s -o %s which det_ndarr_data_status' % (qname, logfname)
        #return 'bsub -q %s -o %s env' % (qname, logfname)
        #return 'bsub -q %s -o %s det_ndarr_data_status -d %s -S 4' % (qname, logfname, dsname)
        return 'bsub -q %s -o %s det_ndarr_data_status -d %s -s %s -f %s -S 4 -u 4' %\
                     (qname, logfname, dsname, sources, ofprefix)
    else :
        return None

#------------------------------

def proc_control(parser) :
    """Dataset processing module

       - Submits job in batch for data processing
       - Checks in loop batch job status
       - Marks in DB that job is processed
       - Save common log file for submission and processing
    """

    (popts, pargs) = parser.parse_args()
    exp      = popts.exp
    run      = popts.run
    procname = popts.pro
    qname    = popts.que
    dt_sec   = popts.dts
    sources  = popts.srs

    tstamp = gu.str_tstamp('%Y-%m-%dT%H:%M:%S', time())

    logpref = rpu.log_batch_prefix(exp, run, procname)
    logfname = '%s-%s.txt' % (logpref, tstamp)

    for i in range(10) :
        if gu.create_path(logfname, depth=6, mode=0o774, verb=False) : continue

    gu.save_textfile('\nCreated path: %s' % logfname, logfname, mode='a')
    os.chmod(logfname, 0o664)

    ofprefix='%s/nda-#exp-#run-#src-#evts-#type.txt' % rpu.work_dir(exp, procname)
    gu.save_textfile('\nOutput work files: %s' % ofprefix, logfname, mode='a')

    for i in range(5) :
        if gu.create_path(ofprefix, depth=8, mode=0o774, verb=False) : continue

    msg = '\nproc_control exp: %s run: %s procname: %s qname: %s logfname %s'%\
          (exp, str(run), procname, qname, logfname)
    gu.save_textfile(str(msg), logfname, mode='a')

    cmd = _batch_submit_command(exp, run, procname, qname, logfname, ofprefix, sources)
    if cmd is None : raise IOError('ERROR: batch submission command is None...')

    t0_sec = time()
    out, err, jobid = spu.batch_job_submit(cmd, env=None, shell=False)
    msg = 'bsub subproc time dt=%7.3f sec' % (time()-t0_sec)
    gu.save_textfile(msg, logfname, mode='a')

    #if 'submitted without an AFS token' in err : 
    #    print '  Tip: to get rid of error message use commands: kinit; aklog'

    rec   = gu.log_rec_on_start()
    msg = '%s\nSubmitted batch job %s to %s\n  cmd: %s\n  out: %s\n  err: "%s"\n%s\n'%\
          (rec, jobid, qname, cmd, out, err.strip('\n'), 80*'_')
    #print msg
    gu.save_textfile(msg, logfname, mode='a')

    # check batch job status in loop
    status = None
    counter = 0
    while status in (None, 'RUN', 'PEND') :
        counter+=1
        out, err, status = spu.batch_job_status(jobid, user=None, qname=qname)
        ts = gu.str_tstamp('%Y-%m-%dT%H:%M:%S', time())
        msg = '%4d %s %s job %s status %s' % (counter, ts, qname, jobid, status)
        #print msg
        gu.save_textfile('%s\n'%msg, logfname, mode='a')
        if status in ('EXIT', 'DONE') : break
        sleep(dt_sec)

    # change log file name in case of bad status
    if status != 'DONE' : 
        logfname_bad = '%s-%s' % (logfname, str(status))
        #cmd = 'mv %s %s' % (logfname, logfname_bad)
        #os.system(cmd)
        os.rename(logfname, logfname_bad)

#------------------------------

def usage() : 
    return "\n%prog -e <experiment> -r <run-number> [-p <process-name> -q <queue-name> -t <wait-sec> -s <sources>]"\
           "\n  Ex.: %prog -e xpptut15 -r 0260 -p pixel_status -q psnehq -t 5 -s cspad,pnccd"

#------------------------------

def input_option_parser() :

    from optparse import OptionParser
    d_exp = 'xpptut15'
    d_run = '0260'
    d_pro = 'pixel_status'
    d_que = 'psnehq'
    d_dts = 60 # sec
    d_srs = 'cspad,opal,epix100,pnccd,princeton,andor'

    h_exp = 'experiment name, default = %s' % d_exp
    h_run = 'run number, default = %s' % d_run
    h_pro = 'data processor name, default = %s' % d_pro
    h_que = 'batch queue name, default = %s' % d_que
    h_dts = 'wait time (sec) between job status check, default = %.1f' % d_dts
    h_srs = 'sources of data (detector types), default = %s' % d_srs

    parser = OptionParser(description='Control process for dataset processing.', usage=usage())
    parser.add_option('-e', '--exp', default=d_exp, action='store', type='string', help=h_exp)
    parser.add_option('-r', '--run', default=d_run, action='store', type='string', help=h_run)
    parser.add_option('-p', '--pro', default=d_pro, action='store', type='string', help=h_pro)
    parser.add_option('-q', '--que', default=d_que, action='store', type='string', help=h_que)
    parser.add_option('-t', '--dts', default=d_dts, action='store', type='float',  help=h_dts)
    parser.add_option('-s', '--srs', default=d_srs, action='store', type='string', help=h_srs)

    return parser #, parser.parse_args()
  
#------------------------------

def do_main() :

    parser = input_option_parser()

    if len(sys.argv) < 5 : 
        parser.print_help()
        sys.exit('WARNING: check input parameters')

    proc_control(parser)
    sys.exit(0)

#------------------------------

if __name__ == "__main__" :
    do_main()

#------------------------------
