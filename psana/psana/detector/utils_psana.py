"""
:py:class:`utils_psana` psana-specific utilities
================================================

Usage::

    from psana.detector.utils_psana import seconds, timestamp_run, datasource_kwargs, info_run

    t_sec = seconds(ts, epoch_offset_sec=631152000) #Converts LCLS2 timestamp to unix epoch time
    ts = timestamp_run(run, fmt='%Y-%m-%dT%H:%M:%S')
    kwa = datasource_arguments(args) # re-define kwargs for DataSource,  args as a namespace
    kwa = datasource_kwargs(**kwa) # re-define kwargs for DataSource

    s = info_run_dsparms_det_classes(run, cmt='run.dsparms.det_classes:\n ', sep='\n ')
    s = info_run(run, cmt='run info:', sep='\n    ', verb=0o377)
    s = info_detector(det, cmt='detector info:', sep='\n    ')
    s = info_uniqueid(det, cmt='det.raw._uniqueid.split("_"):', sep='\n ')

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-02-16 by Mikhail Dubrovin
"""

from psana.detector.Utils import str_tstamp
from psana.detector.Utils import info_dict #, info_namespace, info_command_line

def seconds(ts, epoch_offset_sec=631152000) -> float:
    """
    Converts LCLS2 (int) timestamp to unix epoch time.
    The epoch used is 1-Jan-1990 rather than 1970. -Matt

    Receives  ts = orun.timestamp  # 4193682596073796843 relative to 1990-01-01
    Returns unix epoch time in sec # 1607569818.532117 sec

    import datetime
    epoch_offset_sec=(datetime.datetime(1990, 1, 1)-datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)

    see lcls2/psana/psana/event.py
    """
    assert isinstance(ts, int)
    return float(ts>>32) + float(ts&0xffffffff)*1.e-9 + epoch_offset_sec


def timestamp_run(run, fmt='%Y-%m-%dT%H:%M:%S'):
    """converts LCLS2 run.timestamp to human readable timestamp, e.g. 2020-10-27T12:54:47
    """
    return str_tstamp(fmt=fmt, time_sec=seconds(run.timestamp))


def datasource_kwargs_from_string(s):
    """ Parses string parameters like "exp=<exp-name>,run=<comma-separated-run-numbers>,dir=<xtc-files-directory>,max_events=<number-of-events>"
        to dict keyward arguments.
    See: https://confluence.slac.stanford.edu/display/LCLSIIData/psana

    files: str - xtc2 file name
    exp: str - experiment name
    run: int run number or str with comma-separated run numbers, list of runs ???? THIS WOULD NOT WORK
    dir: str - xtc2 directory name
    max_events: int - maximal number of events to process
    live: True
    timestamp = np.array([4194783241933859761,4194783249723600225,4194783254218190609,4194783258712780993], dtype=np.uint64)??? list of ints?
    intg_det = 'andor'
    batch_size = 1
    detectors = ['epicsinfo', 'tmo_opal1', 'ebeam'] - only reads these detectors (faster)  ???? THIS WOULD NOT WORK
    smd_callback= smd_callback,                     - smalldata callback (see notes above)
    small_xtc   = ['tmo_opal1'],                    - detectors to be used in smalldata callback ???? THIS WOULD NOT WORK
    shmem='tmo' or 'rix',...

    Returns
    -------
    kwargs for DataSource
    """
    from psana.psexp.utils import datasource_kwargs_from_string  # DataSourceFromString

    return datasource_kwargs_from_string(s)


def datasource_arguments(args):
    """
    Parameters
    ----------
    args.fname: str - xtc2 file name
    args.expname: str - experiment name
    args.runs: int run number or str with comma-separated run numbers
    args.detname: str - detector name
    args.evtmax: int - maximal number of events to process

    Returns
    -------
    dict of DataSource keyword arguments
    """
    assert args.fname is not None\
           or None not in (args.expname, args.runs), 'experiment name and run or xtc2 file name need to be specified for DataSource'
    kwa = {'files':args.fname,} if args.fname is not None else\
          {'exp':args.expname,'run':[int(v) for v in args.runs.split(',')]}
    if args.evtmax: kwa['max_events'] = args.evtmax
    return kwa


def datasource_kwargs(**kwargs):
    """ The same as datasource_arguments, but use python standard kwargs as input parameters
    Returns
    -------
    - kwa: dict of DataSource keyword arguments
    """
    fname   = kwargs.get('fname', None)
    exp     = kwargs.get('exp', None)
    runs    = kwargs.get('runs', None)
    events  = kwargs.get('events', 0)

    assert fname is not None\
           or None not in (exp, runs), 'experiment name and run or xtc2 file name need to be specified for DataSource'
    kwa = {'files':fname,} if fname is not None else\
          {'exp':exp,'run':[int(v) for v in runs.split(',')]}
    #if events: kwa['max_events'] = events
    return kwa


def info_run_dsparms_det_classes_v1(run, cmt='run.dsparms.det_classes:', sep='\n '):
    return cmt + sep.join(['%8s : %s' % (str(k),str(v)) for k,v in run.dsparms.det_classes.items()])


def info_run_dsparms_det_classes(run, cmt='run.dsparms.det_classes:\n ', sep='\n '):
    return cmt + info_dict(run.dsparms.det_classes, fmt='%10s : %s', sep=sep)


def info_run(run, cmt='run info:', sep='\n    ', verb=0o377):
    t_sec = seconds(run.timestamp)
    ts_run = timestamp_run(run, fmt='%Y-%m-%dT%H:%M:%S')
    #ts_run = str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=t_sec)
    return cmt\
      +   'run.runnum    : %d' % (run.runnum)\
      + '%srun.expt      : %s' % (sep, run.expt)\
      + '%srun.detnames  : %s' % (sep, ' '.join(run.detnames))\
      + '%srun.timestamp : %s -> %s' % (sep, run.timestamp, ts_run)\
      +('%srun.id        : %s' % (sep, run.id) if verb & 1 else '')\
      +('%s%s' % (sep, info_run_dsparms_det_classes(run, cmt='run.dsparms.det_classes:', sep=sep+'   ')) if verb & 2 else '')


def info_detnames(run, cmt='command: '):
    #import subprocess
    #cmd = 'detnames -r exp=%s,run=%d' % (run.expt, run.runnum)
    #p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #out = str(p.stdout.read())
    #return fmt%(cmt, cmd, out)
    from subprocess import getoutput
    cmd = 'detnames exp=%s,run=%d -r' % (run.expt, run.runnum)
    return cmt + cmd + '\n' + getoutput(cmd)


def info_detnames_for_dskwargs(str_kwa, cmt='command: '):
    from subprocess import getoutput
    cmd = 'detnames %s -r' % (str_kwa)
    return cmt + cmd + '\n' + getoutput(cmd)


def print_detnames(run, cmt='command: '):
    import os
    cmd = 'detnames exp=%s,run=%d -r' % (run.expt, run.runnum)
    print(cmt + cmd)
    os.system(cmd)


def info_detector(det, cmt='detector info:', sep='\n    '):
    return cmt\
        +  'det.raw._det_name   : %s' % (det.raw._det_name)\
        +'%sdet.raw._dettype    : %s' % (sep, det.raw._dettype)\
        +'%s_sorted_segment_ids : %s' % (sep, str(det.raw._sorted_segment_ids))\
        +'%sdet.raw._uniqueid   : %s' % (sep, det.raw._uniqueid)\
        +'%s%s' % (sep, info_uniqueid(det, cmt='det.raw._uniqueid.split("_"):%s     '%sep, sep=sep+'     '))\
        +'%sdet methods vbisible: %s' % (sep, ' '.join([v for v in dir(det) if v[0]!='_']))\
        +'%s             _hidden: %s' % (sep, ' '.join([v for v in dir(det) if (v[0]=='_' and v[1]!='_')]))\
        +'%sdet.raw     vbisible: %s' % (sep, ' '.join([v for v in dir(det.raw) if v[0]!='_']))\
        +'%s             _hidden: %s' % (sep, ' '.join([v for v in dir(det.raw) if (v[0]=='_' and v[1]!='_')]))\
        +'%sdet.raw._calibconst.keys(): %s' % (sep, ', '.join(det.raw._calibconst.keys()))


def info_uniqueid(det, cmt='det.raw._uniqueid.split("_"):', sep='\n '):
    return cmt + sep.join(det.raw._uniqueid.split('_'))

#EOF
