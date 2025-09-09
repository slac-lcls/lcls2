
"""calibration validity info

   import psana.pscalib.calib.UtilsCalibValidity as ucv
"""
import sys
import psana.pscalib.calib.MDBUtils as mu
import psana.pscalib.calib.MDBWebUtils as wu
from time import time, gmtime, localtime, strftime
#import logging
#logger = logging.getLogger(__name__)

def dict_filter(d, keys=('experiment', 'detname', 'detector', 'shortname', 'ctype', 'run', 'run_orig',\
                         'run_beg', 'run_end', 'time_stamp', 'tstamp_orig', 'dettype', 'version', 'longname'), ordered=False):
    import psana.detector.utils_psana as ups # seconds, data_source_kwargs
    return ups.dict_filter(d, list_keys=keys, ordered=ordered)

def tstamp_from_sec(tsec, fmt='%Y%m%d_%H%M%S', gmt=False):
    return strftime(fmt, gmtime(tsec) if gmt else localtime(tsec))

def add_run_from_doc(runs, d, irun_max=9999):
    for r in (d['run'], d['run_end']):
        runs.add(r if r != 'end' else irun_max)

def add_time_from_doc(times_sec, time_stamps, d):
    times_sec.add(d['time_sec'])
    time_stamps.add(d['time_stamp'])

def select_doc_for_tsec(sorted_docs, tsec, key='time_sec'):
    for i,d in enumerate(sorted_docs):
        if d[key] >= tsec:
            #print('XXX selected sorted document #%d' % i)
            return d

def info_run_validity_ranges(dbname, colname, ctype='pedestals'):
    """works with EXPERIMENT DB
       1) extend documents with tsec_id, tstamp_id evaluated form _id
       2) make list of begin and end runs
       2) select doc for each run range interval
    """
    docs = wu.find_docs(dbname, colname, query={'ctype':ctype}) # , url=cc.URL)
    s = '\nlist of constants for dbname:%s colname:%s ctype:%s' % (dbname, colname, ctype)
    run_boardes = set()
    for d in docs:
      d['tsec_id'], d['tstamp_id'] = mu.sec_and_ts_from_id(d['_id'], fmt='%Y%m%d_%H%M%S', gmt=False)
      s += '\ntsecDB: %d tstampDB: %s run: %4d run_orig: %4d run_beg: %4s run_end: %4s'%\
            (d['tsec_id'], d['tstamp_id'], d['run'], d['run_orig'], str(d.get('run_beg',None)), str(d['run_end']))
      add_run_from_doc(run_boardes, d)

    sorted_runs = sorted(run_boardes)
    s += '\nsorted validity boarder runs: %s' % str(sorted(sorted_runs))
    s += '\n\nrun ranges for EXPERIMENT DB dbname:%s colname:%s ctype:%s' % (dbname, colname, ctype)
    for rbeg, rend in zip(sorted_runs[:-1], sorted_runs[1:]):
        doc = wu.select_doc_in_run_range(docs, rbeg)
        s += '\nrun range: %4d -%4d tstampDB: %s tsecDB: %d run_orig: %4d'%\
            (rbeg, rend-1, doc['tstamp_id'], doc['tsec_id'], doc['run_orig'])
    return s

def info_time_validity_ranges(shortname, ctype='pedestals', fmt='%Y%m%d_%H%M%S', gmt=False):
    """works with detector DB
    """
    dbname = 'cdb_%s' % shortname
    colname = shortname
    docs = wu.find_docs(dbname, colname, query={'ctype':ctype}) # , url=cc.URL)
    s = '\nDB document keys: %s' % str(docs[0].keys())
    s += '\n\nlist of constants for dbname:%s colname:%s ctype:%s' % (dbname, colname, ctype)
    times_sec = set()
    time_stamps = set()
    sorted_docs = sorted(docs, key=lambda x: x['time_sec'])
    for d in sorted_docs:
      s += '\ntime_sec: %d time_stamp: %s tstamp_orig: %s run_orig: %4d experiment: %12s'%\
            (d['time_sec'], d['time_stamp'], d['tstamp_orig'], d['run_orig'], d['experiment'])
      add_time_from_doc(times_sec, time_stamps, d)

    sorted_times = sorted(times_sec)
    s += '\n\nsorted validity boarder times (sec): %s' % str(sorted(sorted_times))
    s += '\n\ntime ranges for DETECTOR DB dbname:%s colname:%s ctype:%s' % (dbname, colname, ctype)
    for tbeg, tend in zip(sorted_times[:-1], sorted_times[1:]):
        d = select_doc_for_tsec(sorted_docs, tbeg)
        s += '\ntime stamp range: [%s - %s) sec: %d - %d  exp: %s run_orig: %4d'%\
              (tstamp_from_sec(tbeg), tstamp_from_sec(tend), tbeg, tend, d['experiment'], d['run_orig'])
    return s


def print_calib_constants_for_ctype(expname, detlongname, ctype='pedestals', run=None, time_sec=None):
    resp = wu.calib_constants_for_ctype(detlongname, exp=expname, ctype=ctype, run=run, time_sec=time_sec, vers=None, dbsuffix='')
    if resp is not None:
        d = dict_filter(resp[1])
        print('selected constants in calib_constants_for_ctype:\n%s\n' % str(d))


def _calib_validity_ranges(exp, shortname, ctype='pedestals', show='rtd'):
    dbname = 'cdb_%s' % exp
    colname = shortname
    if ('t' in show): print(info_time_validity_ranges(shortname, ctype))
    if ('r' in show): print(info_run_validity_ranges(dbname, colname, ctype))


def dict_from_str(s):
    """converts str of arguments like
       exp=mfx101332224,run=66,shortname=jungfrau_000003,ctype=pedestals
       to dict = {'exp':'mfx101332224', 'run'=66, 'shortname'='jungfrau_000003', 'ctype'='pedestals'}
    """
    d = {}
    if s is not None:
        flds = s.split(',')
        for f in flds:
            k,v = f.split('=')
            d[k] = v
        run = d.get('run', None)
        if run is not None:
            d[run] = int(run) # str to int
    return d


def calib_validity_ranges(**kwa):

    print('parameters: %s' % str(kwa))

    exp = None
    shortname = None
    ctype   = kwa.get('ctype', 'pedestals')
    allargs = kwa.get('allargs', None)
    show    = kwa.get('show', 'rtd')
    if allargs is None:
        from psana import DataSource
        import psana.detector.UtilsCalib as uc
        import psana.detector.utils_psana as up
        dskwargs = up.data_source_kwargs(**kwa)
        print('dskwargs:%s' % str(dskwargs))
        ds = DataSource(**dskwargs)
        orun = next(ds.runs())
        runnum=orun.runnum
        try:
            odet = orun.Detector(kwa['detname'])
        except Exception as err:
            print('Detector("%s") is not available for %s.\n    %s'%\
                  (kwa['detname'], str(dskwargs), err))
            sys.exit('Exit processing')

        longname = odet.raw._uniqueid
        shortname = uc.detector_name_short(longname)
        exp = dskwargs['exp']
        if ('d' in show):
            print_calib_constants_for_ctype(exp, longname, ctype, run=runnum, time_sec=time())
    else:
        d = dict_from_str(allargs)
        exp       = d.get('exp', None)
        shortname = d.get('shortname', None)
        ctype     = d.get('ctype', 'pedestals')
        runnum    = d.get('run', None)
        #print_calib_constants_for_ctype(exp, longname, ctype, run=runnum, time_sec=time())

    print('exp: %s shortname: %s ctype: %s' % (exp, shortname, ctype))
    _calib_validity_ranges(exp, shortname, ctype=ctype, show=show)


if __name__ == "__main__":

    _calib_validity_ranges('mfx101332224', 'jungfrau_000002', ctype='pedestals')

    sys.exit('\nuse calibvalidity CLI')
# EOF
