#!/usr/bin/env python

import sys
SCRNAME = sys.argv[0].rsplit('/')[-1]

USAGE = '\n  %s --cmd <"command"> --opt <optional-paraneter> --vals <parameter-values> [<other-kwargs>]' % SCRNAME\
      + '\n  %s --cmd "epix10ka_pedestals_calibration -e ueddaq02 -d epixquad -r 422" --opt stepnum --vals 0,1,2,3,4' % SCRNAME

import logging
logger = logging.getLogger(__name__)

import subprocess # for subprocess.Popen
from time import time, sleep

def argument_parser():
    from argparse import ArgumentParser

    d_cmd     = 'epix10ka_pedestals_calibration -e ueddaq02 -d epixquad -r 422 -o ./work'
    d_opt     = 'stepnum'
    d_vals    = '0,1,2,3,4'
    d_logpref = './log'
    d_dtsec   = 5
    d_nchk    = 100

    h_cmd     = 'command prefix for subprocesses, default = %s' % d_cmd
    h_opt     = 'option name to subprocess, default = %s' % d_opt
    h_vals    = 'comma separated option values to subprocess, default = %s' % d_vals
    h_logpref = 'log name prefix, default = %s' % d_logpref
    h_dtsec   = 'time interval (sec) to check if job is done, default = %d' % d_dtsec
    h_nchk    = 'number of checks if job is done until forced exit from this app, default = %d' % d_nchk

    parser = ArgumentParser(description='Split job for subprocesses using one of the command line optional parameters')
    parser.add_argument('--cmd',     default=d_cmd,        type=str,   help=h_cmd)
    parser.add_argument('--opt',     default=d_opt,        type=str,   help=h_opt)
    parser.add_argument('--vals',    default=d_vals,       type=str,   help=h_vals)
    parser.add_argument('--logpref', default=d_logpref,    type=str,   help=h_logpref)
    parser.add_argument('--dtsec',   default=d_dtsec,      type=int,   help=h_dtsec)
    parser.add_argument('--nchk',    default=d_nchk,       type=int,   help=h_nchk)

    return parser


def subproc_open(command_seq, logname, env=None, shell=False): # e.g, command_seq=['bsub', '-q', cp.batch_queue, '-o', 'log-ls.txt', 'ls -l']
    log = open(logname, 'w')
    return subprocess.Popen(command_seq, stdout=log, stderr=log, env=env, shell=shell) #, stdin=subprocess.STDIN


def do_main():

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.INFO)

    #from psana.detector.RepoManager import RepoManager
    #RepoManager(...).save_record_at_start(procname)#, tsfmt='%Y-%m-%dT%H:%M:%S%z')

    dirrepo = './'
    procname = 'parallel_proc'
    from psana.detector.Utils import save_log_record_at_start
    save_log_record_at_start(dirrepo, procname, dirmode=0o777, filemode=0o666, tsfmt='%Y-%m-%dT%H:%M:%S%z', umask=0o0)

    print('Usage:%s\n' % USAGE)

    parser = argument_parser()
    args = parser.parse_args()

    if True:
        kwa = vars(args)
        print('kwa:', str(kwa))

    t0_sec = time()
    lognames = []
    subprocesses = []
    for v in args.vals.split(','):
       cmd = '%s --%s %s' % (args.cmd, args.opt, v)
       logname = '%s-%s-%s-%s.txt' % (args.logpref, args.cmd.split(' ',1)[0], args.opt, v)
       lognames.append(logname)
       subprocesses.append(subproc_open(cmd.split(' '), logname, env=None, shell=False))
       print('\n== creates subprocess for command: %s\n   see results in %s' % (cmd, logname))

    print('\n')
    break_check_loop = False
    for n in range(args.nchk):
       print('working for %d sec,' % (time()-t0_sec), end=' ')
       pstatus = [p.poll() for p in subprocesses]
       print('subprocesses poll status: %s%s' % (str(pstatus),50*' '), end='\r')
       if all(v==0 for v in pstatus):
           print('\nALL SUBPROCESSES COMPLETED')
           break_check_loop = True
           break
       for i,v in enumerate(pstatus):
           if v is not None and v>0:
               print('\nSUBPROCESSES ERROR, see log file: %s' % lognames[i])
               break_check_loop = True
               break
       if break_check_loop: break
       sleep(args.dtsec)


if __name__ == "__main__":
    do_main()
    sys.exit('End of %s'%SCRNAME)

# EOF
