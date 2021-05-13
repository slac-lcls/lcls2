from __future__ import print_function

"""
:py:class:`UtilsCalib`
==============================

Usage::
    from Detector.UtilsCalib import proc_block, DarkProc, evaluate_limits
    from Detector.UtilsCalib import tstamps_run_and_now, tstamp_for_dataset
    gate_lo, gate_hi, arr_med, arr_abs_dev = proc_block(block, **kwa)

    lo, hi = evaluate_limits(arr, nneg=5, npos=5, lim_lo=1, lim_hi=1000, cmt='')
    ts_run, ts_now = tstamps_run_and_now(env, fmt=TSTAMP_FORMAT)
    ts_run = tstamp_for_dataset(dsname, fmt=TSTAMP_FORMAT)

    save_log_record_on_start(dirrepo, fname, fac_mode=0o777)
    fname = find_file_for_timestamp(dirname, pattern, tstamp)

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-04-05 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from time import time, strftime, localtime
from psana import EventId, DataSource
from PSCalib.GlobalUtils import log_rec_on_start, create_directory, save_textfile, dic_det_type_to_calib_group
from Detector.GlobalUtils import info_ndarr, divide_protected #reshape_to_2d#print_ndarr
from PSCalib.UtilsPanelAlias import alias_for_id #, id_for_alias
from PSCalib.NDArrIO import save_txt

TSTAMP_FORMAT = '%Y%m%d%H%M%S'

def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None):
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    return strftime(fmt, localtime(time_sec))


def evt_time(evt):
    """Returns event (double) time for input psana.Event object.
    """
    evid = evt.get(EventId)
    ttuple = evid.time()
    #logger.debug('evt_time %s', str(ttuple))
    return float(ttuple[0]) + float(ttuple[1])*1e-9


def env_time(env):
    """Returns event (double) time for input psana.Env object.
    """
    evid = env.configStore().get(EventId)
    ttuple = evid.time()
    #logger.debug('env_time %s' % str(ttuple))
    return float(ttuple[0]) + float(ttuple[1])*1e-9


def dataset_time(dsname):
    """Returns event (double) time for input dsname "exp=xcsx35617:run=6".
    """
    ds = DataSource(dsname)
    return env_time(ds.env())


def tstamps_run_and_now(env, fmt=TSTAMP_FORMAT):
    """Returns tstamp_run, tstamp_now
    """
    time_run = env_time(env)
    ts_run = str_tstamp(fmt=fmt, time_sec=time_run)
    ts_now = str_tstamp(fmt=fmt, time_sec=None)

    logger.debug('tstamps_run_and_now:'
                 + ('\n  run time stamp      : %s' % ts_run)\
                 + ('\n  current time stamp  : %s' % ts_now))
    return ts_run, ts_now


def tstamp_for_dataset(dsname, fmt=TSTAMP_FORMAT):
    """Returns tstamp_run for dataset dsname, e.g. "exp=xcsx35617:run=6".
    """
    tsec = dataset_time(dsname)
    return str_tstamp(fmt=fmt, time_sec=tsec)


def rundescriptor_in_dsname(dsname):
    """Returns (str) run-descriptor flom dsname, e.g. "6-12" from dsname="exp=xcsx35617:run=6-12".
    """
    for fld in dsname.split(':'):
        if fld[:4]=='run=': return fld.split('=')[-1]
    return None


def is_single_run_dataset(dsname):
    return rundescriptor_in_dsname(dsname).isdigit()
    

def evaluate_limits(arr, nneg=5, npos=5, lim_lo=1, lim_hi=16000, cmt='') :
    """Moved from Detector.UtilsEpix10kaCalib
       Evaluates low and high limit of the array, which are used to find bad pixels.
    """
    ave, std = (arr.mean(), arr.std())
    lo = ave-nneg*std if nneg>0 else lim_lo
    hi = ave+npos*std if npos>0 else lim_hi
    lo, hi = max(lo, lim_lo), min(hi, lim_hi)
    logger.info('evaluate_limits %s: ave=%.3f std=%.3f  limits low=%.3f high=%.3f'%\
                (cmt, ave, std, lo, hi)) # sys._getframe().f_code.co_name
    return lo, hi


def save_log_record_on_start(dirrepo, fname, fac_mode=0o774):
    """Adds record on start to the log file <dirlog>/logs/log-<fname>-<year>.txt
    """
    rec = log_rec_on_start()
    repoman = RepoManager(dirrepo, filemode=fac_mode)
    logfname = repoman.logname_on_start(fname)
    fexists = os.path.exists(logfname)
    save_textfile(rec, logfname, mode='a')
    if not fexists: os.chmod(logfname, fac_mode)
    logger.debug('record on start: %s' % rec)
    logger.info('saved:  %s' % logfname)


def save_2darray_in_textfile(nda, fname, fmode, fmt):
    fexists = os.path.exists(fname)
    np.savetxt(fname, nda, fmt=fmt)
    if not fexists: os.chmod(fname, fmode)
    logger.info('saved:  %s' % fname)


def save_ndarray_in_textfile(nda, fname, fmode, fmt):
    fexists = os.path.exists(fname)
    save_txt(fname=fname, arr=nda, fmt=fmt)
    if not fexists: os.chmod(fname, fmode)
    logger.debug('saved: %s fmode: %s fmt: %s' % (fname, oct(fmode), fmt))

 
def file_name_prefix(panel_type, panel_id, tstamp, exp, irun, fname_aliases):
    panel_alias = alias_for_id(panel_id, fname=fname_aliases, exp=exp, run=irun)
    return '%s_%s_%s_%s_r%04d' % (panel_type, panel_alias, tstamp, exp, irun), panel_alias


class RepoManager(object):
    """Supports repository directories/files naming structure
       <dirrepo>/<panel_id>/<constant_type>/<files-with-constants>
       <dirrepo>/logs/<year>/<log-files>
       <dirrepo>/logs/log-<fname>-<year>.txt # file with log_rec_on_start()
       e.g.: dirrepo = '/reg/g/psdm/detector/gains/epix10k/panels'

    Usage::

      from Detector.UtilsCalib import RepoManager
      repoman = RepoManager(dirrepo)
      d = repoman.dir_logs()
      d = repoman.makedir_logs()
    """

    def __init__(self, dirrepo, **kwa):
        self.dirrepo = dirrepo.rstrip('/')
        self.dirmode     = kwa.get('dirmode',  0o774)
        self.filemode    = kwa.get('filemode', 0o664)
        self.dirname_log = kwa.get('dirname_log', 'logs')


    def makedir(self, d):
        """create and return directory d with mode defined in object property
        """
        create_directory(d, self.dirmode)
        return d


    def dir_in_repo(self, name):
        """return directory <dirrepo>/<name>
        """
        return os.path.join(self.dirrepo, name)


    def makedir_in_repo(self, name):
        """create and return directory <dirrepo>/<name>
        """
        return self.makedir(self.dir_in_repo(name))


    def dir_logs(self):
        """return directory <dirrepo>/logs
        """
        return self.dir_in_repo(self.dirname_log)


    def makedir_logs(self):
        """create and return directory <dirrepo>/logs
        """
        return self.makedir(self.dir_logs())


    def dir_logs_year(self, year=None):
        """return directory <dirrepo>/logs/<year>
        """
        _year = str_tstamp(fmt='%Y') if year is None else year
        return os.path.join(self.dir_logs(), _year)


    def makedir_logs_year(self, year=None):
        """create and return directory <dirrepo>/logs/<year>
        """
        return self.makedir(self.dir_logs_year(year))


    def dir_merge(self, dname='merge_tmp'):
        return self.dir_in_repo(dname)


    def makedir_merge(self, dname='merge_tmp'):
        return self.makedir(self.dir_merge(dname))


    def dir_panel(self, panel_id):
        """returns path to panel directory like <dirrepo>/<panel_id>
        """
        return os.path.join(self.dirrepo, panel_id)


    def makedir_panel(self, panel_id):
        """create and returns path to panel directory like <dirrepo>/<panel_id>
        """
        return self.makedir(self.dir_panel(panel_id))


    def dir_type(self, panel_id, ctype): # ctype='pedestals'
        """returns path to the directory like <dirrepo>/<panel_id>/<ctype>
        """
        return '%s/%s' % (self.dir_panel(panel_id), ctype)


    def makedir_type(self, panel_id, ctype): # ctype='pedestals'
        """create and returns path to the directory like <dirrepo>/<panel_id>/<ctype>
        """
        return self.makedir(self.dir_type(panel_id, ctype))


    def dir_types(self, panel_id, subdirs=('pedestals', 'rms', 'status', 'plots')):
        """define structure of subdirectories in calibration repository under <dirrepo>/<panel_id>/...
        """
        return ['%s/%s'%(self.dir_panel(panel_id), name) for name in subdirs]


    def makedir_types(self, panel_id, subdirs=('pedestals', 'rms', 'status', 'plots')):
        """create structure of subdirectories in calibration repository under <dirrepo>/<panel_id>/...
        """
        dirs = self.dir_types(panel_id, subdirs=subdirs)
        for d in dirs: self.makedir(d)
        return dirs


    def logname_on_start(self, scrname, year=None):
        _year = str_tstamp(fmt='%Y') if year is None else str(year)
        return '%s/%s_log_%s.txt' % (self.makedir_logs(), _year, scrname)


    def logname(self, scrname):
        tstamp = str_tstamp(fmt='%Y-%m-%dT%H%M%S')
        return '%s/%s_log_%s.txt' % (self.makedir_logs_year(), tstamp, scrname)


def proc_dark_block(block, **kwa):
    """Copied and modified from UtilsEpix10kaCalib
       Assumes that ALL dark events are in the block - returns ALL arrays
       
       Returns per-panel (352, 384) arrays of mean, rms, ...
       block.shape = (nrecs, 352, 384), where nrecs <= 1024
    """
    exp        = kwa.get('exp', None)
    detname    = kwa.get('det', None)
    int_lo     = kwa.get('int_lo', 1)       # lowest  intensity accepted for dark evaluation
    int_hi     = kwa.get('int_hi', 16000)   # highest intensity accepted for dark evaluation
    intnlo     = kwa.get('intnlo', 6.0)     # intensity ditribution number-of-sigmas low
    intnhi     = kwa.get('intnhi', 6.0)     # intensity ditribution number-of-sigmas high
    rms_lo     = kwa.get('rms_lo', 0.001)   # rms ditribution low
    rms_hi     = kwa.get('rms_hi', 16000)   # rms ditribution high
    rmsnlo     = kwa.get('rmsnlo', 6.0)     # rms ditribution number-of-sigmas low
    rmsnhi     = kwa.get('rmsnhi', 6.0)     # rms ditribution number-of-sigmas high
    fraclm     = kwa.get('fraclm', 0.1)     # allowed fraction limit
    fraclo     = kwa.get('fraclo', 0.05)    # fraction of statistics below low gate limit
    frachi     = kwa.get('frachi', 0.95)    # fraction of statistics below high gate limit
    frac05     = 0.5
    nrecs1     = kwa.get('nrecs1', None)    # number of records for the 1st stage processing

    logger.debug('in proc_dark_block for exp=%s det=%s, block.shape=%s' % (exp, detname, str(block.shape)))
    logger.info(info_ndarr(block, 'begin pricessing of the data block:\n    ', first=100, last=105))
    logger.debug('fraction of statistics for gate limits low: %.3f high: %.3f' % (fraclo, frachi))

    t0_sec = time()

    nrecs, ny, nx = block.shape
    shape = (ny, nx)
    if nrecs1 is None or nrecs1>nrecs: nrecs1 = nrecs

    arr1_u16 = np.ones(shape, dtype=np.uint16)
    arr1     = np.ones(shape, dtype=np.uint64)

    t1_sec = time()

    """
    NOTE:
    - our data is uint16.
    - np.median(block, axis=0) or np.quantile(...,interpolation='linear') return result rounded to int
    - in order to return interpolated float values apply the trick:
      data_block + random [0,1)-0.5
    - this would distort data in the range [-0.5,+0.5) ADU, but would allow 
      to get better interpolation for median and quantile values
    - use nrecs1 (< nrecs) due to memory and time consumption
    """
    #blockf64 = np.random.random((nrecs1, ny, nx)) - 0.5 + block[:nrecs1,:]
    #logger.debug(info_ndarr(blockf64, '1-st stage conversion uint16 to float64,'\
    #                                 +' add random [0,1)-0.5 time = %.3f sec '%\
    #                                  (time()-t1_sec), first=100, last=105))

    blockf64 = block[:nrecs1,:]
    #arr_med = np.median(block, axis=0)
    arr_med = np.quantile(blockf64, frac05, axis=0, interpolation='linear')
    arr_qlo = np.quantile(blockf64, fraclo, axis=0, interpolation='lower')
    arr_qhi = np.quantile(blockf64, frachi, axis=0, interpolation='higher')
    logger.debug('block array median/quantile(0.5) for med, qlo, qhi time = %.3f sec' % (time()-t1_sec))

    med_med = np.median(arr_med)
    med_qlo = np.median(arr_qlo)
    med_qhi = np.median(arr_qhi)

    arr_dev_3d = block[:,] - arr_med # .astype(dtype=np.float64)
    arr_abs_dev = np.median(np.abs(arr_dev_3d), axis=0)
    med_abs_dev = np.median(arr_abs_dev)

    logger.info(info_ndarr(arr_med,     '    arr_med[100:105] ', first=100, last=105))
    logger.info(info_ndarr(arr_qlo,     '    arr_qlo[100:105] ', first=100, last=105))
    logger.info(info_ndarr(arr_qhi,     '    arr_qhi[100:105] ', first=100, last=105))
    logger.info(info_ndarr(arr_abs_dev, '    abs_dev[100:105] ', first=100, last=105))

    s = 'data-block pre-processing time %.3f sec' % (time()-t0_sec)\
      + '\nresults for median over pixels intensities:'\
      + '\n    %.3f fraction of the event spectrum is below %.3f ADU - pedestal estimator' % (frac05, med_med)\
      + '\n    %.3f fraction of the event spectrum is below %.3f ADU - gate low limit' % (fraclo, med_qlo)\
      + '\n    %.3f fraction of the event spectrum is below %.3f ADU - gate upper limit' % (frachi, med_qhi)\
      + '\n    event spectrum spread    median(abs(raw-med)): %.3f ADU - spectral peak width estimator' % med_abs_dev
    logger.info(s)

    #sys.exit('TEST EXIT')

    logger.debug(info_ndarr(arr_med, '1st iteration proc time = %.3f sec arr_av1' % (time()-t0_sec)))
    #gate_half = nsigma*rms_ave
    #logger.debug('set gate_half=%.3f for intensity gated average, which is %.3f * sigma' % (gate_half,nsigma))
    #gate_half = nsigma*abs_dev_med
    #logger.debug('set gate_half=%.3f for intensity gated average, which is %.3f * abs_dev_med' % (gate_half,nsigma))

    # 2nd loop over recs in block to evaluate gated parameters
    logger.debug('begin 2nd iteration')

    sta_int_lo = np.zeros(shape, dtype=np.uint64)
    sta_int_hi = np.zeros(shape, dtype=np.uint64)

    arr_max = np.zeros(shape, dtype=block.dtype)
    arr_min = np.ones (shape, dtype=block.dtype) * 0x3fff

    gate_lo    = arr1_u16 * int_lo
    gate_hi    = arr1_u16 * int_hi

    #gate_hi = np.minimum(arr_av1 + gate_half, gate_hi).astype(dtype=block.dtype)
    #gate_lo = np.maximum(arr_av1 - gate_half, gate_lo).astype(dtype=block.dtype)
    gate_lo = np.maximum(arr_qlo, gate_lo).astype(dtype=block.dtype)
    gate_hi = np.minimum(arr_qhi, gate_hi).astype(dtype=block.dtype)
    cond = gate_hi>gate_lo
    gate_hi[np.logical_not(cond)] +=1
    #gate_hi = np.select((cond, np.logical_not(cond)), (gate_hi, gate_hi+1), 0)

    logger.debug(info_ndarr(gate_lo, '    gate_lo '))
    logger.debug(info_ndarr(gate_hi, '    gate_hi '))

    arr_sum0 = np.zeros(shape, dtype=np.uint64)
    arr_sum1 = np.zeros(shape, dtype=np.float64)
    arr_sum2 = np.zeros(shape, dtype=np.float64)

    #blockdbl = np.array(block, dtype=np.float64)

    for nrec in range(nrecs):
        raw    = block[nrec,:]
        rawdbl = raw.astype(dtype=np.uint64) # blockdbl[nrec,:]

        logger.debug('nrec:%03d median(raw-ave): %f' % (nrec, np.median(raw.astype(dtype=np.float64) - arr_med)))
        #logger.debug('nrec:%03d median(raw-ave): %.6f' % (nrec, np.median(raw.astype(dtype=np.float64) - arr_med)))
        #logger.debug(info_ndarr(raw, '  raw     '))
        #logger.debug(info_ndarr(arr_med, '  arr_med '))

        condlist = (np.logical_not(np.logical_or(raw<gate_lo, raw>gate_hi)),)

        arr_sum0 += np.select(condlist, (arr1,), 0)
        arr_sum1 += np.select(condlist, (rawdbl,), 0)
        arr_sum2 += np.select(condlist, (np.square(rawdbl),), 0)

        sta_int_lo += np.select((raw<int_lo,), (arr1,), 0)
        sta_int_hi += np.select((raw>int_hi,), (arr1,), 0)

        arr_max = np.maximum(arr_max, raw)
        arr_min = np.minimum(arr_min, raw)

    arr_av1 = divide_protected(arr_sum1, arr_sum0)
    arr_av2 = divide_protected(arr_sum2, arr_sum0)

    frac_int_lo = np.array(sta_int_lo/nrecs, dtype=np.float32)
    frac_int_hi = np.array(sta_int_hi/nrecs, dtype=np.float32)

    arr_rms = np.sqrt(arr_av2 - np.square(arr_av1))
    #rms_ave = arr_rms.mean()
    rms_ave = mean_constrained(arr_rms, rms_lo, rms_hi)

    rms_min, rms_max = evaluate_limits(arr_rms, rmsnlo, rmsnhi, rms_lo, rms_hi, cmt='RMS')
    ave_min, ave_max = evaluate_limits(arr_av1, intnlo, intnhi, int_lo, int_hi, cmt='AVE')

    arr_sta_rms_hi = np.select((arr_rms>rms_max,),    (arr1,), 0)
    arr_sta_rms_lo = np.select((arr_rms<rms_min,),    (arr1,), 0)
    arr_sta_int_hi = np.select((frac_int_hi>fraclm,), (arr1,), 0)
    arr_sta_int_lo = np.select((frac_int_lo>fraclm,), (arr1,), 0)
    arr_sta_ave_hi = np.select((arr_av1>ave_max,),    (arr1,), 0)
    arr_sta_ave_lo = np.select((arr_av1<ave_min,),    (arr1,), 0)

    logger.info('Bad pixel status:'\
               +'\n  status  1: %8d pixel rms       > %.3f' % (arr_sta_rms_hi.sum(), rms_max)\
               +'\n  status  2: %8d pixel rms       < %.3f' % (arr_sta_rms_lo.sum(), rms_min)\
               +'\n  status  4: %8d pixel intensity > %g in more than %g fraction of events' % (arr_sta_int_hi.sum(), int_hi, fraclm)\
               +'\n  status  8: %8d pixel intensity < %g in more than %g fraction of events' % (arr_sta_int_lo.sum(), int_lo, fraclm)\
               +'\n  status 16: %8d pixel average   > %g'   % (arr_sta_ave_hi.sum(), ave_max)\
               +'\n  status 32: %8d pixel average   < %g'   % (arr_sta_ave_lo.sum(), ave_min)\
               )

    #0/1/2/4/8/16/32 for good/hot-rms/cold-rms/saturated/cold/average above limit/average below limit, 
    arr_sta = np.zeros(shape, dtype=np.uint64)
    arr_sta += arr_sta_rms_hi    # hot rms
    arr_sta += arr_sta_rms_lo*2  # cold rms
    arr_sta += arr_sta_int_hi*4  # satturated
    arr_sta += arr_sta_int_lo*8  # cold
    arr_sta += arr_sta_ave_hi*16 # too large average
    arr_sta += arr_sta_ave_lo*32 # too small average

    absdiff_av1_med = np.abs(arr_av1-arr_med)
    logger.debug(info_ndarr(absdiff_av1_med, 'np.abs(arr_av1-arr_med)', first=100, last=105))
    logger.info('estimator of difference between gated average and median np.median(np.abs(arr_av1-arr_med)): %.3f' % np.median(absdiff_av1_med))

    cond = absdiff_av1_med > med_abs_dev
    arr_av1[cond] = arr_med[cond]

    arr_sta_bad = np.select((cond,), (arr1,), 0)
    frac_bad = arr_sta_bad.sum()/float(arr_av1.size)
    logger.debug('fraction of panel pixels with gated average deviated from and replaced by median: %.6f' % frac_bad)

    #logger.info('data block processing time = %.3f sec' % (time()-t0_sec))
    #logger.debug(info_ndarr(arr_av1, 'arr_av1     [100:105] ', first=100, last=105))
    #logger.debug(info_ndarr(arr_rms, 'pixel_rms   [100:105] ', first=100, last=105))
    #logger.debug(info_ndarr(arr_sta, 'pixel_status[100:105] ', first=100, last=105))
    #logger.debug(info_ndarr(arr_med, 'arr mediane [100:105] ', first=100, last=105))

    return arr_av1, arr_rms, arr_sta

#===
#===
#===

def proc_block(block, **kwa):
    """Dark data 1st stage processing to define gate limits.
       block.shape = (nrecs, <raw-detector-shape>), 
       where <raw-detector-shape> can be per segment (352, 384) or per detector (nsegs, 352, 384)
       Returns segment/detector shaped arrays of gate_lo, gate_hi, arr_med, arr_abs_dev
    """
    exp        = kwa.get('exp', None)
    detname    = kwa.get('det', None)
    int_lo     = kwa.get('int_lo', 1)       # lowest  intensity accepted for dark evaluation
    int_hi     = kwa.get('int_hi', 16000)   # highest intensity accepted for dark evaluation
    #intnlo     = kwa.get('intnlo', 6.0)     # intensity ditribution number-of-sigmas low
    #intnhi     = kwa.get('intnhi', 6.0)     # intensity ditribution number-of-sigmas high
    #rms_lo     = kwa.get('rms_lo', 0.001)   # rms ditribution low
    #rms_hi     = kwa.get('rms_hi', 16000)   # rms ditribution high
    #rmsnlo     = kwa.get('rmsnlo', 6.0)     # rms ditribution number-of-sigmas low
    #rmsnhi     = kwa.get('rmsnhi', 6.0)     # rms ditribution number-of-sigmas high
    #fraclm     = kwa.get('fraclm', 0.1)     # allowed fraction limit
    fraclo     = kwa.get('fraclo', 0.05)    # fraction of statistics below low gate limit
    frachi     = kwa.get('frachi', 0.95)    # fraction of statistics below high gate limit
    frac05     = 0.5
    #nrecs1     = kwa.get('nrecs1', None)    # number of records for the 1st stage processing

    logger.debug('in proc_dark_block for exp=%s det=%s, block.shape=%s' % (exp, detname, str(block.shape)))
    logger.info(info_ndarr(block, 'begin pricessing of the data block', first=100, last=105))

    t0_sec = time()

    #nrecs1, ny, nx = block.shape[0]
    nrecs1= block.shape[0]
    shape = block.shape[1:] #(ny, nx)
    #if nrecs1 is None or nrecs1>nrecs: nrecs1 = nrecs

    arr1_u16 = np.ones(shape, dtype=np.uint16)
    arr1     = np.ones(shape, dtype=np.uint64)

    t1_sec = time()

    """
    NOTE:
    - our data is uint16.
    - np.median(block, axis=0) or np.quantile(...,interpolation='linear') return result rounded to int
    - in order to return interpolated float values apply the trick:
      data_block + random [0,1)-0.5
    - this would distort data in the range [-0.5,+0.5) ADU, but would allow 
      to get better interpolation for median and quantile values
    - use nrecs1 (< nrecs) due to memory and time consumption
    """
    #blockf64 = np.random.random(block.shape) - 0.5 + block
    #logger.debug(info_ndarr(blockf64, '1-st stage conversion uint16 to float64,'\
    #                                 +' add random [0,1)-0.5 time = %.3f sec'%\
    #                                  (time()-t1_sec), first=100, last=105))

    blockf64 = block
    #arr_med = np.median(block, axis=0)
    arr_med = np.quantile(blockf64, frac05, axis=0, interpolation='linear')
    arr_qlo = np.quantile(blockf64, fraclo, axis=0, interpolation='lower')
    arr_qhi = np.quantile(blockf64, frachi, axis=0, interpolation='higher')

    logger.debug('block array median/quantile(frac) for med, qlo, qhi time = %.3f sec' % (time()-t1_sec))

    med_med = np.median(arr_med)
    med_qlo = np.median(arr_qlo)
    med_qhi = np.median(arr_qhi)

    arr_dev_3d = block[:,] - arr_med # .astype(dtype=np.float64)
    arr_abs_dev = np.median(np.abs(arr_dev_3d), axis=0)
    med_abs_dev = np.median(arr_abs_dev)

    s = 'proc_block pre-processing time %.3f sec' % (time()-t0_sec)\
      + '\n    results for median over pixels intensities:'\
      + '\n    %.3f fraction of the event spectrum is below %.3f ADU - pedestal estimator' % (frac05, med_med)\
      + '\n    %.3f fraction of the event spectrum is below %.3f ADU - gate low limit' % (fraclo, med_qlo)\
      + '\n    %.3f fraction of the event spectrum is below %.3f ADU - gate upper limit' % (frachi, med_qhi)\
      + '\n    event spectrum spread    median(abs(raw-med)): %.3f ADU - spectral peak width estimator' % med_abs_dev
    logger.info(s)

    gate_lo    = arr1_u16 * int_lo
    gate_hi    = arr1_u16 * int_hi

    gate_lo = np.maximum(np.floor(arr_qlo), gate_lo).astype(dtype=block.dtype)
    gate_hi = np.minimum(np.ceil(arr_qhi),  gate_hi).astype(dtype=block.dtype)
    cond = gate_hi>gate_lo
    gate_hi[np.logical_not(cond)] +=1

    logger.debug('proc_block results'\
                +info_ndarr(arr_med,     '\n    arr_med[100:105]', first=100, last=105)\
                +info_ndarr(arr_abs_dev, '\n    abs_dev[100:105]', first=100, last=105)\
                +info_ndarr(gate_lo,     '\n    gate_lo[100:105]', first=100, last=105)\
                +info_ndarr(gate_hi,     '\n    gate_hi[100:105]', first=100, last=105))
                #+info_ndarr(arr_qlo,     '\n    arr_qlo[100:105]', first=100, last=105)\
                #+info_ndarr(arr_qhi,     '\n    arr_qhi[100:105]', first=100, last=105)\

    return gate_lo, gate_hi, arr_med, arr_abs_dev


class DarkProc(object):
    """dark data accumulation and processing
    """
    def __init__(self, **kwa):

        self.nrecs  = kwa.get('nrecs',1000)
        self.nrecs1 = kwa.get('nrecs1',100)
        self.plotim = kwa.get('plotim', 1)
        self.savebw = kwa.get('savebw', 0xffff)
        self.fraclm = kwa.get('fraclm', 0.1)
        self.int_lo = kwa.get('int_lo', 1)       # lowest  intensity accepted for dark evaluation
        self.int_hi = kwa.get('int_hi', 16000)   # highest intensity accepted for dark evaluation
        self.intnlo = kwa.get('intnlo', 6.0)     # intensity ditribution number-of-sigmas low
        self.intnhi = kwa.get('intnhi', 6.0)     # intensity ditribution number-of-sigmas high
        self.rms_lo = kwa.get('rms_lo', 0.001)   # rms ditribution low
        self.rms_hi = kwa.get('rms_hi', 16000)   # rms ditribution high
        self.rmsnlo = kwa.get('rmsnlo', 6.0)     # rms ditribution number-of-sigmas low
        self.rmsnhi = kwa.get('rmsnhi', 6.0)     # rms ditribution number-of-sigmas high

        self.status = 0 # 0/1/2 stage
        self.kwa    = kwa
        self.block  = None
        self.irec   = -1


    def accumulate_block(self, raw):
        self.block[self.irec,:] = raw # & M14 - already done


    def proc_block(self):
        t0_sec = time()
        self.gate_lo, self.gate_hi, self.arr_med, self.abs_dev = proc_block(self.block, **self.kwa)
        logger.info('data block processing total time %.3f sec' % (time()-t0_sec)\
              +info_ndarr(self.arr_med, '\n  arr_med[100:105]', first=100, last=105)\
              +info_ndarr(self.abs_dev, '\n  abs_dev[100:105]', first=100, last=105)\
              +info_ndarr(self.gate_lo, '\n  gate_lo[100:105]', first=100, last=105)\
              +info_ndarr(self.gate_hi, '\n  gate_hi[100:105]', first=100, last=105))


    def init_proc(self):

        shape_raw = self.arr_med.shape
        dtype_raw = self.gate_lo.dtype

        logger.info('Stage 2 initialization for raw shape %s and dtype %s' % (str(shape_raw), str(dtype_raw)))

        self.arr_sum0   = np.zeros(shape_raw, dtype=np.uint64)
        self.arr_sum1   = np.zeros(shape_raw, dtype=np.float64)
        self.arr_sum2   = np.zeros(shape_raw, dtype=np.float64)

        self.arr0       = np.zeros(shape_raw, dtype=dtype_raw)
        self.arr1       = np.ones (shape_raw, dtype=dtype_raw)
        self.arr1u64    = np.ones (shape_raw, dtype=np.uint64)

        self.sta_int_lo = np.zeros(shape_raw, dtype=np.uint64)
        self.sta_int_hi = np.zeros(shape_raw, dtype=np.uint64)

        self.arr_sum0   = np.zeros(shape_raw, dtype=np.uint64)
        self.arr_sum1   = np.zeros(shape_raw, dtype=np.float64)
        self.arr_sum2   = np.zeros(shape_raw, dtype=np.float64)

        self.gate_hi    = np.minimum(self.arr1 * self.int_hi, self.gate_hi)
        self.gate_lo    = np.maximum(self.arr1 * self.int_lo, self.gate_lo)

        self.arr_max    = np.zeros(shape_raw, dtype=dtype_raw)
        self.arr_min    = np.ones (shape_raw, dtype=dtype_raw) * 0xffff


    def summary(self):
        t0_sec = time()

        logger.info('summary')
        logger.info('%s\nraw data found/selected in %d events' % (80*'_', self.irec+1))

        if self.irec>0:
            logger.info('begin data summary stage')
        else:
            logger.info('irec=%d there are no arrays to save...' % self.irec)
            return 

        savebw  = self.savebw
        int_hi  = self.int_hi
        int_lo  = self.int_lo
        intnhi  = self.intnhi
        intnlo  = self.intnlo
        rms_hi  = self.rms_hi
        rms_lo  = self.rms_lo
        rmsnhi  = self.rmsnhi
        rmsnlo  = self.rmsnlo

        fraclm  = self.fraclm
        counter = self.irec

        arr_av1 = divide_protected(self.arr_sum1, self.arr_sum0)
        arr_av2 = divide_protected(self.arr_sum2, self.arr_sum0)

        frac_int_lo = np.array(self.sta_int_lo/counter, dtype=np.float32)
        frac_int_hi = np.array(self.sta_int_hi/counter, dtype=np.float32)

        arr_rms = np.sqrt(arr_av2 - np.square(arr_av1))
        
        logger.debug(info_ndarr(arr_rms, 'arr_rms'))
        logger.debug(info_ndarr(arr_av1, 'arr_av1'))

        rms_min, rms_max = evaluate_limits(arr_rms, rmsnlo, rmsnhi, rms_lo, rms_hi, cmt='RMS')
        ave_min, ave_max = evaluate_limits(arr_av1, intnlo, intnhi, int_lo, int_hi, cmt='AVE')

        arr_sta_rms_hi = np.select((arr_rms>rms_max,),    (self.arr1,), 0)
        arr_sta_rms_lo = np.select((arr_rms<rms_min,),    (self.arr1,), 0)
        arr_sta_int_hi = np.select((frac_int_hi>fraclm,), (self.arr1,), 0)
        arr_sta_int_lo = np.select((frac_int_lo>fraclm,), (self.arr1,), 0)
        arr_sta_ave_hi = np.select((arr_av1>ave_max,),    (self.arr1,), 0)
        arr_sta_ave_lo = np.select((arr_av1<ave_min,),    (self.arr1,), 0)

        logger.info('bad pixel status:'\
               +'\n  status  1: %8d pixel rms       > %.3f' % (arr_sta_rms_hi.sum(), rms_max)\
               +'\n  status  2: %8d pixel rms       < %.3f' % (arr_sta_rms_lo.sum(), rms_min)\
               +'\n  status  4: %8d pixel intensity > %g in more than %g fraction of events' % (arr_sta_int_hi.sum(), int_hi, fraclm)\
               +'\n  status  8: %8d pixel intensity < %g in more than %g fraction of events' % (arr_sta_int_lo.sum(), int_lo, fraclm)\
               +'\n  status 16: %8d pixel average   > %g'   % (arr_sta_ave_hi.sum(), ave_max)\
               +'\n  status 32: %8d pixel average   < %g'   % (arr_sta_ave_lo.sum(), ave_min)\
               )

        #0/1/2/4/8/16/32 for good/hot-rms/cold-rms/saturated/cold/average above limit/average below limit, 
        arr_sta = np.zeros(arr_av1.shape, dtype=np.uint64)
        arr_sta += arr_sta_rms_hi    # hot rms
        arr_sta += arr_sta_rms_lo*2  # cold rms
        arr_sta += arr_sta_int_hi*4  # satturated
        arr_sta += arr_sta_int_lo*8  # cold
        arr_sta += arr_sta_ave_hi*16 # too large average
        arr_sta += arr_sta_ave_lo*32 # too small average

        arr_msk  = np.select((arr_sta>0,), (self.arr0,), 1)

        self.arr_av1 = arr_av1        
        self.arr_rms = arr_rms
        self.arr_sta = arr_sta
        self.arr_msk = np.select((arr_sta>0,), (self.arr0,), 1)

        logger.debug(self.info_results())
        self.plot_images(titpref='')

        self.block = None
        self.irec = -1
        logger.info('summary consumes %.3f sec' % (time()-t0_sec))


    def add_event(self, raw, irec):
        logger.debug(info_ndarr(raw, 'add_event %3d raw' % irec))
        #raw = raw & M14

        cond_lo = raw<self.gate_lo
        cond_hi = raw>self.gate_hi
        condlist = (np.logical_not(np.logical_or(cond_lo, cond_hi)),)
        
        raw_f64 = raw.astype(np.float64)

        self.arr_sum0   += np.select(condlist, (self.arr1u64,), 0)
        self.arr_sum1   += np.select(condlist, (raw_f64,), 0)
        self.arr_sum2   += np.select(condlist, (np.square(raw_f64),), 0)

        self.sta_int_lo += np.select((cond_lo,), (self.arr1u64,), 0)
        self.sta_int_hi += np.select((cond_hi,), (self.arr1u64,), 0)

        np.maximum(self.arr_max, raw, out=self.arr_max)
        np.minimum(self.arr_min, raw, out=self.arr_min)


    def add_block(self):
        logger.info(info_ndarr(self.block, 'add to gated average statistics the block of initial data'))
        for i,raw in enumerate(self.block): self.add_event(raw,i)


    def event(self, raw, evnum):
        """Switch between gain mode processing objects using igm index of the gain mode (0,1,2).
           - evnum (int) - event number
           - igm (int) - index of the gain mode in DIC_GAIN_MODE
        """
        logger.debug('event %d' % evnum)
            
        if raw is None: return self.status

        if self.block is None :
           self.block=np.zeros((self.nrecs1,)+tuple(raw.shape), dtype=raw.dtype)
           logger.info(info_ndarr(self.block,'created empty data block'))

        self.irec +=1
        if self.irec < self.nrecs1:
            self.accumulate_block(raw)

        elif self.irec > self.nrecs1:
            self.add_event(raw, self.irec)

        else:
            self.proc_block()
            self.init_proc()
            self.add_block()
            print('1st stage event block processing is completed')
            self.add_event(raw, self.irec)

        if self.irec > self.nrecs-2:
            logger.info('record %d event loop is terminated, --nrecs=%d' % (self.irec, self.nrecs))
            self.status = 2

        return self.status


    def info_results(self, cmt='DarkProc results'):
        return cmt\
         +info_ndarr(self.arr_med, '\n  arr_med')\
         +info_ndarr(self.arr_av1, '\n  arr_av1')\
         +info_ndarr(self.abs_dev, '\n  abs_dev')\
         +info_ndarr(self.arr_rms, '\n  arr_rms')\
         +info_ndarr(self.arr_sta, '\n  arr_sta')\
         +info_ndarr(self.arr_msk, '\n  arr_msk')\
         +info_ndarr(self.arr_max, '\n  arr_max')\
         +info_ndarr(self.arr_min, '\n  arr_min')\
         +info_ndarr(self.gate_lo, '\n  gate_lo')\
         +info_ndarr(self.gate_hi, '\n  gate_hi')


    def plot_images(self, titpref=''):
        plotim = self.plotim
        if plotim &   1: plot_image(self.arr_av1,    tit=titpref + 'average')
        if plotim &   2: plot_image(self.arr_rms,    tit=titpref + 'RMS')
        if plotim &   4: plot_image(self.arr_sta,    tit=titpref + 'status')
        if plotim &   8: plot_image(self.arr_msk,    tit=titpref + 'mask')
        if plotim &  16: plot_image(self.arr_max,    tit=titpref + 'maximum')
        if plotim &  32: plot_image(self.arr_min,    tit=titpref + 'minimum')
        if plotim &  64: plot_image(self.sta_int_lo, tit=titpref + 'statistics below threshold')
        if plotim & 128: plot_image(self.sta_int_hi, tit=titpref + 'statistics above threshold')
        if plotim & 256: plot_image(self.arr_med,    tit=titpref + 'median after 1st stage processing of %d frames' % self.nrecs1)
        if plotim & 512: plot_image(self.abs_dev,    tit=titpref + 'abs_dev after 1st stage processing of %d frames' % self.nrecs1)
        if plotim &1024: plot_image(self.arr_av1 - self.arr_med, tit=titpref + 'ave - dev')


def plot_image(nda, tit=''):
    """Plots averaged image
    """
    import pyimgalgos.GlobalGraphics as gg
    from pyimgalgos.GlobalUtils import reshape_to_2d

    #img = det.image(evt, nda)
    img = reshape_to_2d(nda)
    if img is None:
        print('plot_image - image "%s" is not available.'%tit)
        return

    logger.info(info_ndarr(img, 'plot_image of %s' % tit))

    amin = np.quantile(img, 0.01, interpolation='lower')
    amax = np.quantile(img, 0.99, interpolation='higher')
    gg.plotImageLarge(img, amp_range=(amin, amax), title=tit)
    gg.show()


def common_mode_pars(src, arr_ave, arr_rms, arr_msk):
        """Returns detector-dependent common mode parameters as np.array for a few detectors and None for others.
        """
        import PSCalib.GlobalUtils as gu 
        import math

        dettype = gu.det_type_from_source(src)

        ave = arr_ave[arr_msk>0].mean()
        rms = arr_rms[arr_msk>0].mean()

        print('Evaluate common mode for source: %s det: %s, estimated intensity ave: %.3f  rms: %.3f' %\
                  (src, gu.dic_det_type_to_name[dettype], ave, rms))

        if dettype == gu.PNCCD:
            return np.array((3, math.ceil(4*rms), math.ceil(4*rms), 128))

        #elif dettype == gu.EPIX100A:
        #    return np.array((4, 6, math.ceil(2*rms), math.ceil(2*rms)))

        #elif dettype == gu.CSPAD:
        #    return np.array((1, math.ceil(3*rms), math.ceil(2*rms), 100))

        #elif dettype == gu.CSPAD2X2:
        #    return np.array((1, math.ceil(3*rms), math.ceil(2*rms), 100))

        else:
            return None


def find_file_for_timestamp(dirname, pattern, tstamp):
    # list of file names in directory, dirname, containing pattern
    fnames = [name for name in os.listdir(dirname) if os.path.splitext(name)[-1]=='.dat' and pattern in name]

    # list of int tstamps 
    # !!! here we assume specific name structure generated by file_name_prefix
    itstamps = [int(name.split('_',3)[2]) for name in fnames]

    # reverse-sort int timestamps in the list
    itstamps.sort(key=int,reverse=True)

    # find the nearest to requested timestamp
    for its in itstamps:
        if its <= int(tstamp):
            # find and return the full file name for selected timestamp
            ts = str(its)

            for name in fnames:
                if ts in name: 
                     fname = '%s/%s' % (dirname, name)
                     logger.debug('  selected %s for %s and %s' % (os.path.basename(fname),pattern,tstamp))
                     return fname

    logger.debug('directory %s\n          DOES NOT CONTAIN file for pattern %s and timestamp <= %s'%\
                   (dirname,pattern,tstamp))
    return None


def merge_panels(lst):
    """ stack of 16 (or 4 or 1) arrays from list shaped as (7, 1, 352, 384) to (7, 16, 352, 384)
    """
    npanels = len(lst)   # 16 or 4 or 1
    shape = lst[0].shape # (7, 1, 352, 384)
    ngmods = shape[0]    # 7

    logger.debug('In merge_panels: number of panels %d number of gain modes %d' % (npanels,ngmods))

    # make list for merging of (352,384) blocks in right order
    mrg_lst = []
    for igm in range(ngmods):
        nda1gm = np.stack([lst[ind][igm,0,:] for ind in range(npanels)])
        mrg_lst.append(nda1gm)
    return np.stack(mrg_lst)


def calib_group(dettype):
    """Returns subdirecrory name under calib/<subdir>/... for 
       dettype, which is one of EPIX10KA2M, EPIX10KAQUAD, EPIX10KA, 
       i.g. 'Epix10ka::CalibV1'
    """
    return dic_det_type_to_calib_group.get(dettype, None)

# EOF
