#!/usr/bin/env python
#------------------------------

import os
import sys
from time import time
from expmon.EMUtils import list_of_runs_in_xtc_dir #, list_of_files_in_dir_for_ext 
import PSCalib.GlobalUtils as gu

"""
:py:class:`NewRunFinderUtils` contains a set of utilities helping find new runs in xtc directory for data processing
====================================================================================================================

Content of files in the xtc directory is conmpared with log file in static DIR_LOG place 
for specified process name. New runs available in the xtc directory and not listed in the log file 
can be retrieved. After new run(s) processing log record(s) should be appended.

Usage::

    # Import
    import PSCalib.RunProcUtils as rpu

    # Parameters
    exp = 'xpptut15'
    run = '0059'
    procname = 'pixel_status'

    # Methods
    dsn   = rpu.dsname(exp='xpptut15', run='0001')
    fname = rpu.control_file(procname='pixel_status')
    fname = rpu.log_file(exp='xpptut15', procname='pixel_status')
    fname = rpu.arc_file(exp='xpptut15', procname='pixel_status')
    dname = rpu.xtc_dir(exp='xpptut15')
    dname = rpu.work_dir(exp='xpptut15', procname='pixel_status')
    dname = rpu.instrument_dir(ins='CXI')
    runs  = rpu.runs_in_xtc_dir(exp='xpptut15', verb=0)
    recs  = rpu.recs_in_log_file(exp='xpptut15', procname='pixel_status', verb=0)
    runs  = rpu.runs_in_log_file(exp='xpptut15', procname='pixel_status')
    text  = rpu.msg_to_log(runs=[])
    rpu.append_log_file(exp='xpptut15', procname='pixel_status', runs=[], verb=0)
    rpu.move_recs_to_archive(procname, exp, runs)
    runs  = rpu.runs_new_in_exp(exp='xpptut15', procname='pixel_status', verb=0)
    runs  = rpu.runs_old_in_exp(exp='xpptut15', procname='pixel_status', verb=0)
    exps  = rpu.experiments(ins='CXI')
    exps  = rpu.experiments_under_control(procname='pixel_status')
    exp_run = rpu.exp_run_new_under_control(procname='pixel_status')
    exp_run = rpu.exp_run_new(ins='CXI', procname='pixel_status')
    exp_run = rpu.exp_run_old(ins='CXI', procname='pixel_status')
    d_exp_run = rpu.dict_exp_run_old(ins='CXI', procname='pixel_status')

    # Example methods
    rpu.print_new_runs(exp='xpptut15', procname='pixel_status', verb=1)
    rpu.print_experiments_under_control(procname='pixel_status')
    rpu.print_experiments_all(procname='pixel_status', ins=None)
    rpu.print_experiments(ins='CXI')
    rpu.print_experiments_under_control(procname='pixel_status')
    rpu.print_all_experiments()
    rpu.print_exp_runs(exp_runs, procname='pixel_status', add_to_log=False)
    rpu.print_datasets_new(ins='CXI', procname='pixel_status', add_to_log=False)
    rpu.print_datasets_new_under_control(procname='pixel_status', add_to_log=False)

    rpu.print_exp_runs_old(dic_exp_runs, procname='pixel_status', move_to_archive=False)
    rpu.print_datasets_old(ins='CXI', procname='pixel_status', move_to_archive=False)

Methods 
  * :meth:`dsname`
  * :meth:`log_file`
  * :meth:`xtc_dir`
  * :meth:`work_dir`
  * :meth:`runs_in_xtc_dir`
  * :meth:`recs_in_log_file`
  * :meth:`runs_in_log_file`
  * :meth:`runs_new_in_exp`
  * :meth:`runs_old_in_exp`
  * :meth:`append_log_file`
  * ...

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-10-05 by Mikhail Dubrovin
"""
#------------------------------

INSTRUMENTS = ('SXR', 'AMO', 'XPP', 'XCS', 'CXI', 'MEC', 'MFX', 'DET', 'MOB', 'USR', 'DIA')
DIR_INS = '/reg/d/psdm'
DIR_LOG = '/reg/g/psdm/logs/run_proc'

#------------------------------

def dsname(exp='xpptut15', run='0001') :
    """Returns (str) control file name, e.g. 'exp=xpptut15:run=1' for (str) exp and (str) of (int) run
    """
    if   isinstance(run, str) : return 'exp=%s:run=%s' % (exp, run.lstrip('0'))
    elif isinstance(run, int) : return 'exp=%s:run=%d' % (exp, run)
    else : return None

#------------------------------

def control_file(procname='pixel_status') :
    """Returns (str) control file name, e.g. '/reg/g/psdm/logs/run_proc/pixel_status/proc_control.txt'
    """
    return '%s/%s/experiments.txt' % (DIR_LOG, procname)

#------------------------------

def log_file(exp='xpptut15', procname='pixel_status') :
    """Returns (str) log file name, e.g. '/reg/g/psdm/logs/run_proc/pixel_status/XPP/xpptut15-proc-runs.txt'
    """
    return '%s/%s/%s/%s-proc-runs.txt' % (DIR_LOG, procname, exp[:3].upper(), exp)

#------------------------------

def log_batch_prefix(exp='xpptut15', run='0001', procname='pixel_status') :
    """Returns (str) log file template, e.g. '/reg/g/psdm/logs/run_proc/pixel_status/XPP/xpptut15/log-xpptut15-r0001'
    """
    return '%s/%s/%s/%s/log-%s-r%s' % (DIR_LOG, procname, exp[:3].upper(), exp, exp, run)

#------------------------------

def arc_file(exp='xpptut15', procname='pixel_status') :
    """Returns (str) arcive log file name, e.g. '/reg/g/psdm/logs/run_proc/pixel_status/XPP/xpptut15-proc-arch.txt'
    """
    return '%s/%s/%s/%s-proc-arch.txt' % (DIR_LOG, procname, exp[:3].upper(), exp)

#------------------------------

def xtc_dir(exp='xpptut15') :
    """Returns (str) xtc directory name, e.g. '/reg/d/psdm/XPP/xpptut15/xtc'
    """
    return '%s/%s/%s/xtc' % (DIR_INS, exp[:3].upper(), exp)

#------------------------------

def work_dir(exp='xpptut15', procname='pixel_status') :
    """Returns (str) work directory, e.g. '/reg/g/psdm/logs/run_proc/pixel_status/XPP/xpptut15/work'
    """
    return '%s/%s/%s/%s/work' % (DIR_LOG, procname, exp[:3].upper(), exp)

#------------------------------

def instrument_dir(ins='CXI') :
    """Returns (str) instrument directory, e.g. '/reg/g/psdm/logs/run_proc/pixel_status/CXI'
    """
    _ins = ins.upper()
    if not(_ins in INSTRUMENTS): raise IOError('Unknown instrument "%s"' % ins)
    return '%s/%s' % (DIR_INS, _ins)

#------------------------------

def runs_in_xtc_dir(exp='xpptut15', verb=0) :
    """Returns sorted list of (str) runs in xtc directory name, e.g. ['0059', '0060',...]
    """
    dirxtc = xtc_dir(exp)
    if not os.path.lexists(dirxtc) : return []
        #raise IOError('Directory %s is not available' % dirxtc)
    if verb : print 'Scan directory: %s' % dirxtc
    return sorted(list_of_runs_in_xtc_dir(dirxtc))

#------------------------------

def recs_in_log_file(exp='xpptut15', procname='pixel_status', verb=0) :
    """Returns list of (str) records in the log file for specified experiment and process name.
       E.g. of one record: '0151 2017-10-05T15:19:21'
    """
    fname_log = log_file(exp, procname)
    if verb : print 'Log file: %s' % fname_log
    if not os.path.lexists(fname_log) : 
        if verb : print 'Log file "%s" does not exist' % fname_log
        return []
    recs = gu.load_textfile(fname_log).split('\n')
    return recs # list of records, each record is '0059 <time-stamp>'

#------------------------------

def runs_in_log_file(exp='xpptut15', procname='pixel_status') :
    """Returns list of (4-char str) runs in the log file for specified experiment and process name.
       E.g. ['0059', '0060',...]
    """
    runs = [rec.split(' ')[0] for rec in recs_in_log_file(exp, procname) if rec]
    return runs

#------------------------------

def msg_to_log(runs=[]) :
    """Returns (str) message to the log file for list of (str) runs.
    """
    if len(runs)==0 : return None
    tstamp = gu.str_tstamp('%Y-%m-%dT%H:%M:%S', time())
    login  = gu.get_login()
    cwd    = gu.get_cwd()
    host   = gu.get_hostname()
    cmd    = sys.argv[0].split('/')[-1]
    recs = ['%s %s %s %s cwd:%s cmd:%s'%(s, tstamp, login, host, cwd, cmd) for s in runs]
    text = '\n'.join(recs)
    return text+'\n'
    #return text if len(runs)>1 else text+'\n'

#------------------------------

def append_log_file(exp='xpptut15', procname='pixel_status', runs=[], verb=0) :
    """Appends records in the log file for list of (str) runs for specified experiment and process name.
    """
    fname_log = log_file(exp, procname)
    if verb : print 'Append log file: %s' % fname_log
    gu.create_path(fname_log, depth=6, mode=0774, verb=False)
    text = msg_to_log(runs)
    if text is None : return
    #print 'Save in file text "%s"' % text
    gu.save_textfile(text, fname_log, mode='a')
    os.chmod(fname_log, 0664)

#------------------------------

def move_recs_to_archive(procname, exp, runs) :
    """Move expired run records from log file to archive file.
    """
    fname_log = log_file(exp, procname)
    fname_arc = arc_file(exp, procname)
    print 'Move records for old runs to archive file: %s\n' % fname_arc
    recs = gu.load_textfile(fname_log).split('\n')

    recs_log = [rec for rec in recs if not(rec[:4] in runs)]
    recs_arc = [rec for rec in recs if     rec[:4] in runs]

    text_log = '\n'.join(recs_log)
    text_log+='\n'
    #if len(runs)==1 : text_log+='\n'

    text_arc = '\n'.join(recs_arc)
    text_arc+='\n'
    #if len(runs)==1 : text_arc+='\n'

    #print '  ==> log\n%s' % text_log
    #print '  ==> arc\n%s' % text_arc

    gu.save_textfile(text_log, fname_log, mode='w')
    os.chmod(fname_log, 0664)

    gu.save_textfile(text_arc, fname_arc, mode='a')
    os.chmod(fname_log, 0664)

#------------------------------

def runs_new_in_exp(exp='xpptut15', procname='pixel_status', verb=0) :
    """Returns list of (4-char str) runs which are found in xtc directory and not yet listed in the log file,
       e.g. ['0059', '0060',...]
    """
    runs_log = runs_in_log_file(exp, procname)
    runs_xtc = runs_in_xtc_dir(exp)
    runs_new = [s for s in runs_xtc if not(s in runs_log)]

    if verb & 2:
        for srun in runs_xtc :
            if srun in runs_new : print '%s - new' % srun
            else :                print '%s - processed %s' % (srun, dsname(exp, srun))

    if verb :
        print '\nScan summary for exp=%s process="%s"' % (exp, procname)
        print '%4d runs in xtc dir  : %s' % (len(runs_xtc), xtc_dir(exp)),\
              '\n%4d runs in log file : %s' % (len(runs_log), log_file(exp, procname)),\
              '\n%4d runs NEW in xtc directory' % len(runs_new)

    return runs_new

#------------------------------

def runs_old_in_exp(exp='xpptut15', procname='pixel_status', verb=0) :
    """Returns list of (4-char str) runs which are found in the log file and are not listed in xtc directory,
       e.g. ['0059', '0060',...]
    """
    runs_log = runs_in_log_file(exp, procname)
    runs_xtc = runs_in_xtc_dir(exp)
    runs_old = [s for s in runs_log if not(s in runs_xtc)]

    if verb & 2:
        for srun in runs_log :
            if srun in runs_old : print '%s - old' % srun
            else :                print '%s - processed %s' % (srun, dsname(exp, srun))

    if verb :
        print '\nScan summary for exp=%s process="%s"' % (exp, procname)
        print '%4d runs in xtc dir  : %s' % (len(runs_xtc), xtc_dir(exp)),\
              '\n%4d runs in log file : %s' % (len(runs_log), log_file(exp, procname)),\
              '\n%4d runs OLD in log file' % len(runs_old)

    return runs_old

#------------------------------

def experiments(ins='CXI') :
    """Returns list of (8,9-char-str) experiment names for specified 3-char (str) instrument name,
       or all experiments is ins=None,
       e.g. ['mfxo1916', 'mfxn8416', 'mfxlq3915',...]
    """
    if ins is None :
        list_of_exps=[]
        for i in INSTRUMENTS : 
            list_of_exps += experiments(ins=i)
        return list_of_exps
            
    else :
        dirname = instrument_dir(ins)
        _ins = ins.lower()
        return [fname for fname in os.listdir(dirname) if (fname[:3]==_ins and len(fname)<10)]

#------------------------------

def experiments_under_control(procname='pixel_status') :
    """Returns list of (str) experiment names from control file.
    """
    fname = control_file(procname)
    if not os.path.lexists(fname) : 
        #raise IOError('Control file "%s" does not exist' % fname)
        print 'WARNING: control file "%s" does not exist' % fname
        return []
    recs = gu.load_textfile(fname).split('\n')
    return [rec for rec in recs if (rec and (rec[0]!='#'))] # skip empty and commented records

#------------------------------

def exp_run_new(ins='CXI', procname='pixel_status') :
    """Returns new list of tuples (exp,run) for specified instrument (=None for all) and procname,
       e.g. [('mfx13016','0005'), ('mfx15070','0008'),...]
    """
    exp_run = []
    for exp in experiments(ins) :
        exp_run += [(exp,run) for run in runs_new_in_exp(exp, procname, verb=0)]
    return exp_run

#------------------------------

def exp_run_new_under_control(procname='pixel_status') :
    """Returns new list of tuples (exp,run) for specified procname for experiments from control file,
       e.g. [('mfx13016','0005'), ('mfx15070','0008'),...]
    """
    exp_run = []
    for exp in experiments_under_control(procname) :
        exp_run += [(exp,run) for run in runs_new_in_exp(exp, procname, verb=0)]
    return exp_run

#------------------------------

def exp_run_old(ins='CXI', procname='pixel_status') :
    """Returns old list of tuples (exp,run) for specified instrument (=None for all) and procname,
       e.g. [('mfx13016','0005'), ('mfx15070','0008'),...]
    """
    exp_run = []
    for exp in experiments(ins) :
        exp_run += [(exp,run) for run in runs_old_in_exp(exp, procname, verb=0)]
    return exp_run

#------------------------------

def dict_exp_run_old(ins='CXI', procname='pixel_status') :
    """Returns dict {exp:list_of_runs} for specified instrument (=None for all) and procname,
       e.g. {'mfx13016':['0005','0006'], 'mfx15070':['0003','0004','0005'],...]
    """
    exp_run = {}
    for exp in experiments(ins) :
        runs = [run for run in runs_old_in_exp(exp, procname, verb=0)]
        if len(runs)!=0 :
            exp_run[exp] = runs
    return exp_run

#------------------------------
#------------------------------
#--------  EXAMPLES  ----------
#------------------------------
#------------------------------

def print_new_runs(exp='xpptut15', procname='pixel_status', verb=1) :
    runs_new = runs_new_in_exp(exp, procname, verb)
    if len(runs_new) :
        print 'New runs found in %s for process %s:' % (exp, procname)
        for srun in runs_new :
            print srun,
        print ''
        append_log_file(exp, procname, runs_new)
    else :
        print 'No new runs found in %s for process %s' % (exp, procname)

        #ctime_sec = os.path.getctime(fname)
        #ctime_str = gu.str_tstamp('%Y-%m-%dT%H:%M:%S', ctime_sec)
        #print ctime_str, fname

#------------------------------

def print_experiments_under_control(procname='pixel_status') :
    for exp in experiments_under_control(procname) :
        print '%s\nProcess new runs for %s' % (50*'=', exp)
        print_new_runs(exp, procname)

#------------------------------

def print_experiments_all(procname='pixel_status', ins=None) :
    for exp in experiments(ins) :
        print '%s\nProcess new runs for %s' % (50*'=', exp)
        print_new_runs(exp, procname)

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def print_experiments(ins='CXI') :
    exps = experiments(ins)
    for exp in exps :
        print exp
    dname = '%s/<all-ins>/<all-exp>/'%DIR_INS if ins is None else instrument_dir(ins)
    print '%d experiments found in %s' % (len(exps), dname)

#------------------------------

def print_experiments_count_runs() : # ins='CXI'
    d_ins_nruns = {}
    d_ins_nexps = {}
    nruns_tot = 0
    nexps = 0
    for ins in INSTRUMENTS :
        nruns_ins = 0
        exps = experiments(ins)
        nexps += len(exps)
        for exp in exps :
            runs = runs_in_xtc_dir(exp)
            nruns = len(runs)
            nruns_ins += nruns
            nruns_tot += nruns
            print '  %10s  nruns:%4d' % (exp, nruns)
        d_ins_nruns[ins] = nruns_ins
        d_ins_nexps[ins] = len(exps)

    print '\nSummary on %s\n%s'%(gu.str_tstamp('%Y-%m-%dT%H:%M:%S', time()), 40*'_')
    for ins,nruns in d_ins_nruns.iteritems() :
        print '%6d runs in %4d experiments of %s' % (nruns, d_ins_nexps[ins], ins)

    dname = '%s/<all-ins>/<all-exp>/'%DIR_INS
    print '%s\n%6d runs in %4d experiments of %s' % (40*'_', nruns_tot, nexps, dname)

#------------------------------

def print_explogs_under_control(procname='pixel_status') :
    print '%s\nExperiments under control:' % (110*'_')
    for i, exp in enumerate(experiments_under_control(procname)) :
        print '%4d %s %s'%(i+1, exp.ljust(10), log_file(exp, procname))

#------------------------------

def print_experiments_under_control(procname='pixel_status') :
    for exp in experiments_under_control(procname) :
        print exp

#------------------------------

def print_all_experiments() :
    tot_nexps=0
    ins_nexps={}
    for ins in INSTRUMENTS : 
        exps = experiments(ins)
        for exp in exps :
            print exp
            tot_nexps += 1
        print '%d experiments found in %s\n' % (len(exps), instrument_dir(ins))
        ins_nexps[ins] = len(exps)

    print 'Number of expriments per instrument'
    #print '%d experiments found in %s' % (len(exps), ''.join(INSTRUMENTS))
    for ins in INSTRUMENTS : print '%s : %4d' % (ins, ins_nexps[ins])
    print 'Total number of expriments %d' % tot_nexps

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def print_exp_runs(exp_runs, procname='pixel_status', add_to_log=False) :
    for i,(exp,run) in enumerate(exp_runs) :
        dsname = 'exp=%s:run=%s'%(exp, run.lstrip('0'))
        logname = log_file(exp, procname)
        print '%4d %s %4s %s %s'%(i+1, exp.ljust(10), run, dsname.ljust(22), logname)
        #--------------
        if add_to_log : append_log_file(exp, procname, [run,])
        #--------------
    print '%d new runs found' % (len(exp_runs))

#------------------------------

def print_datasets_new(ins='CXI', procname='pixel_status', add_to_log=False) :
    exp_runs = exp_run_new(ins, procname)
    print_exp_runs(exp_runs, procname, add_to_log)

#------------------------------

def print_datasets_new_under_control(procname='pixel_status', add_to_log=False) :
    exp_runs = exp_run_new_under_control(procname)
    print_exp_runs(exp_runs, procname, add_to_log)

#------------------------------

def print_exp_runs_old(dic_exp_runs, procname='pixel_status', move_to_archive=False) :
    nruns = 0
    for exp,runs in dic_exp_runs.iteritems() :
        #dsname = 'exp=%s:run=%s'%(exp, run.lstrip('0'))
        logname = log_file(exp, procname)
        print '%s%s\n  '%(exp.ljust(10), logname),
        for i, run in enumerate(runs) : 
           print run,
           if i and ((i+1)%10)==0 : print '\n  ',
        print '\n'
        nruns += len(runs) 
        #--------------
        if move_to_archive : move_recs_to_archive(procname, exp, runs)
        #--------------
    print '%d old runs found in logs which are missing in xtc directories' % nruns

#------------------------------

def print_datasets_old(ins='CXI', procname='pixel_status', move_to_archive=False) :
    dic_exp_runs = dict_exp_run_old(ins, procname)
    print_exp_runs_old(dic_exp_runs, procname, move_to_archive)

#------------------------------
#------------------------------
#------------------------------

def usage() :
    return  'python PSCalib/src/RunProcUtils.py <test_name>\n'\
           +'       <test_name> = 1  - print new files in experiments listed in control file\n'\
           +'                   = 10 - the same as 1 and save record for each new run in log file\n'\
           +'                   = 2  - print new files in all experiments\n'\
           +'                   = 20 - the same as 2 and save record for each new run in log file\n'\
           +'                   = 3  - print all experiments\n'\
           +'                   = 4  - print old (available in log but missing in xtc-dir) files for all experiments\n'\
           +'                   = 40 - the same as 4 and move old run records from log to archive file\n'\
           +'                   = 5  - print statistics of all instruments, experiments/runs in xtc directories\n'

#------------------------------

if __name__ == "__main__" :
    print 80*'_'
    tname = sys.argv[1] if len(sys.argv)>1 else '5' # 'CXI'
    cname = tname.upper()
    t0_sec = time()

    if cname in INSTRUMENTS : 
        print_experiments(ins=cname)
        print_datasets_new(ins=cname, procname='pixel_status')

    elif tname=='0' : print_all_experiments()

    elif tname=='1' : print_datasets_new_under_control(procname='pixel_status')
    elif tname=='10': print_datasets_new_under_control(procname='pixel_status', add_to_log=True)

    elif tname=='2' : print_datasets_new(ins=None, procname='pixel_status')
    elif tname=='20': print_datasets_new(ins=None, procname='pixel_status', add_to_log=True)

    elif tname=='3' : print_experiments(ins=None) # all

    elif tname=='4' : print_datasets_old(ins=None, procname='pixel_status')
    elif tname=='40': print_datasets_old(ins=None, procname='pixel_status', move_to_archive=True)

    elif tname=='5' : print_experiments_count_runs()

    else : sys.exit ('Not recognized test name: "%s"' % tname)
    print 'Test %s time (sec) = %.3f' % (tname, time()-t0_sec)

    if len(sys.argv)<2 : print usage()

    sys.exit ('End of %s' % sys.argv[0])

#------------------------------
