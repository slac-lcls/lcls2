#!/usr/bin/env python
"""./lcls2/psana/psana/detector/test_issues_2024.py <TNAME>"""

import sys
import logging

SCRNAME = sys.argv[0].rsplit('/')[-1]
global STRLOGLEV # sys.argv[2] if len(sys.argv)>2 else 'INFO'
global INTLOGLEV # logging._nameToLevel[STRLOGLEV]
logger = logging.getLogger(__name__)


def ds_run_det(exp='ascdaq18', run=171, detname='epixhr', **kwa):
    from psana import DataSource
    ds = DataSource(exp=exp, run=run, **kwa)
    orun = next(ds.runs())
    det = orun.Detector(detname)
    return ds, orun, det


def issue_2026_mm_dd():
    """ISSUE:
       PROBLEM:
       FIX:
    """
    print('template')


def issue_2026_02_04():
    """jungfrau_dark_proc -k exp=mfx100848724,run=49 -d jungfrau -o ./work1 --nrecs 50 --nrecs1 50 --stepnum 1
       ISSUE: continue seems does not work in the step loop
       PROBLEM: loop is terminated earlier due to max_events=100
       FIX: increase 'max_events':100000
    """
    from psana import DataSource
    #ds = DataSource(exp='mfx100848724', run=49)
    ds = DataSource(exp='mfx100848724', run=49, **{'max_events':100000})
    orun = next(ds.runs())
    det = orun.Detector('jungfrau')
    for istep, step in enumerate(orun.steps()):
        print('begin step: %d' % istep)
        if istep == 0:
            continue
        print('end of step: %d' % istep)
    print('end of step loop')


def issue_2026_02_26():
    """ISSUE: command
              jungfrau_dark_proc -k exp=xppc00125,run=6 -d jungfrau1M -o work1 --stepnum 0
              does not work (work in old release) because of mpi features
       PROBLEM: dskwargs={...,'smd_callback': None}
       FIX: remove 'smd_callback': None from dskwargs
    """
    if False:
        import os
        os.environ['PS_EB_NODES']='1'
        os.environ['PS_SRV_NODES']='1'

        import psutil
        cpu_num = psutil.Process().cpu_num()

        from psana.detector.Utils import get_hostname
        hostname = get_hostname()

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        s_rsch = 'rank:%03d/%03d-cpu:%03d-%s' % (rank, size, cpu_num, hostname)
        print('>>>> %s sys.argv: %s' % (s_rsch, sys.argv))

    from psana.detector.NDArrUtils import info_ndarr
    from psana import DataSource
    #[I] UtilsJungfrauCalib.py L0163 DataSource dskwargs: {'exp': 'xppc00125', 'run': 6, 'max_events': 10000, 'batch_size': 2, 'smd_callback': None}
    #dskwargs={'exp': 'xppc00125', 'run': 6, 'max_events': 10000, 'batch_size': 2, 'smd_callback': None}
    dskwargs={'exp': 'xppc00125', 'run': 6, 'max_events': 10000, 'batch_size': 2}
    ds = DataSource(**dskwargs)
    #run = next(ds.runs())
    for irun, run in enumerate(ds.runs()):
      logger.info('\n%s runnum %d %s' % (20*'=', run.runnum, 20*'='))
      det = run.Detector('jungfrau1M')
      for istep, step in enumerate(run.steps()):
        print('begin step: %d' % istep)
        for ievt, evt in enumerate(step.events()):
            if ievt<10:
                raw = det.raw.raw(evt)
                print(info_ndarr(raw, 'ev: %03d raw' % ievt))
        print('end of step: %d' % istep)
    print('end of step loop')


def issue_2026_03_16():
    """ISSUE: Hi Mikhail, I looked into the problem Valerio/Chris reported and traced it to the Jungfrau pedestals
         deployed for mfx100848724, starting from run 49. It looks like the deployed pedestal constants have
         the wrong dtype: float64 instead of the expected float32. Because of that, det.raw.calib(evt) fails
         for runs that resolve to this pedestal. I reproduced the failure for mfx100848724, run 51.
       PROBLEM:
       FIX:
    """
    import psana
    ds = psana.DataSource(exp="mfx100848724", run=51, dir="/sdf/data/lcls/ds/prj/public01/xtc", max_events=10)
    run = next(ds.runs())
    det = run.Detector("jungfrau")

    for evt in run.events():
        data = det.raw.raw(evt)  # raw data
        print(f"Extracted raw for {evt}")
        data = det.raw.calib(evt)  # calibrated data
        print(f"Extracted calib for {evt}")

#===
    
#===

def argument_parser():
    from argparse import ArgumentParser
    d_tname = '0'
    d_dskwargs = 'exp=rixc00121,run=140,dir=/sdf/data/lcls/drpsrcf/ffb/rix/rixc00121/xtc'  # None
    d_detname  = 'archon' # None
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    h_tname    = 'test name, usually numeric number 0,...,>20, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME,\
                            usage='for list of implemented tests run it without parameters')
    parser.add_argument('tname',            default=d_tname,    type=str, help=h_tname)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
    return parser


def selector():
    parser = argument_parser()
    args = parser.parse_args()
    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)

    TNAME = args.tname # sys.argv[1] if len(sys.argv)>1 else '0'

    if   TNAME in ('0',): issue_2026_mm_dd() # template
    elif TNAME in ('1',): issue_2026_02_04()
    elif TNAME in ('2',): issue_2026_02_26()
    elif TNAME in ('3',): issue_2026_03_16()
    elif TNAME in ('99',): issue_2026_02_04(args.subtest)
    else:
        print(USAGE())
        exit('\nTEST "%s" IS NOT IMPLEMENTED'%TNAME)
    exit('END OF TEST %s'%TNAME)


def USAGE():
    import inspect
    #return '\n  TEST'
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
         + '\n'.join([s for s in inspect.getsource(selector).split('\n') if "TNAME in" in s])\
         + '\n\nHELP:\n  list of parameters: ./%s -h\n  list of tests:      ./%s' % (SCRNAME, SCRNAME)


if __name__ == "__main__":
    if len(sys.argv)==1:
        print(USAGE())
        exit('\nMISSING ARGUMENTS -> EXIT')
    selector()

# EOF
