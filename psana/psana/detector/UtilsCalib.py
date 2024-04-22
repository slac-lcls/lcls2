
"""
:py:class:`UtilsCalib` dark processing algorithms for generic area detector
===============================================================================

Usage::

    from psana.detector.UtilsCalib import *
    #OR
    import psana.detector.UtilsCalib as uac

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2022-01-18 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys
import numpy as np

import psana.detector.utils_psana as up
from psana import DataSource
from psana.detector.Utils import str_tstamp, time, get_login, info_dict  # info_command_line
import psana.pscalib.calib.CalibConstants as cc
from psana.detector.NDArrUtils import info_ndarr, divide_protected, reshape_to_2d, save_ndarray_in_textfile
from psana.detector.RepoManager import RepoManager, init_repoman_and_logger

SCRNAME = sys.argv[0].rsplit('/')[-1]


def selected_record(i, events):
    return i<5\
       or (i<50 and not i%10)\
       or (i<200 and not i%20)\
       or not i%100\
       or i>events-5


def info_pixel_status(status, bits=(1<<64)-1):
    arr1 = np.ones_like(status, dtype=np.int32)
    statist_bits = np.select((status & bits,), (arr1,), 0)
    statist_tot = np.select((status>0,), (arr1,), 0)
    return 'number of pixels containing bits %4d(dec) %4s(oct): %d of total bad %d of total %d'%\
            (bits, oct(bits), statist_bits.sum(), statist_tot.sum(), status.size)


def evaluate_limits(arr, nneg=5, npos=5, lim_lo=1, lim_hi=16000, cmt=''):
    """Evaluates low and high limit of the array, which are used to find bad pixels.
    """
    ave, std = (arr.mean(), arr.std()) if (nneg>0 or npos>0) else (None,None)
    lo = ave-nneg*std if nneg>0 else lim_lo
    hi = ave+npos*std if npos>0 else lim_hi
    lo, hi = max(lo, lim_lo), min(hi, lim_hi)

    logger.debug('  %s: %s ave, std = %.3f, %.3f  low, high limits = %.3f, %.3f'%\
                 (sys._getframe().f_code.co_name, cmt, ave, std, lo, hi))

    return lo, hi


def tstamps_run_and_now(trun_sec): # unix epoch time, e.g. 1607569818.532117 sec
    """Returns (str) tstamp_run, tstamp_now#, e.g. (str) 20201209191018, 20201217140026
    """
    trun_sec = int(trun_sec)
    ts_run = str_tstamp(fmt='%Y%m%d%H%M%S', time_sec=trun_sec)
    ts_now = str_tstamp(fmt='%Y%m%d%H%M%S', time_sec=None)
    return ts_run, ts_now


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
    fraclo     = kwa.get('fraclo', 0.05)    # fraction of statistics below low gate limit
    frachi     = kwa.get('frachi', 0.95)    # fraction of statistics below high gate limit
    frac05     = 0.5

    logger.debug('in proc_dark_block for exp=%s det=%s, block.shape=%s' % (exp, detname, str(block.shape)))
    logger.info(info_ndarr(block, 'begin processing of the data block', first=100, last=105))

    t0_sec = time()
    nrecs1= block.shape[0]
    shape = block.shape[1:] #(ny, nx)

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

    return gate_lo, gate_hi, arr_med, arr_abs_dev


class DarkProc():
    """dark data accumulation and processing"""
    def __init__(self, **kwa):

        self.nrecs  = kwa.get('nrecs',1000)
        self.nrecs1 = kwa.get('nrecs1',100)
        self.plotim = kwa.get('plotim', 0o1)
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
        self.datbits= kwa.get('datbits', 0xffff) # data bits 0xffff - 16-bit mask for detector without gain bit/s

        self.status = 0 # 0/1/2 stage
        self.kwa    = kwa
        self.block  = None
        self.irec   = -1


    def accumulate_block(self, raw):
        self.block[self.irec,:] = raw


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
        self.arr_min    = np.ones (shape_raw, dtype=dtype_raw) * self.datbits


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
        plotim  = self.plotim

        fraclm  = self.fraclm
        counter = self.irec
        nevlm = int(fraclm * counter)

        arr_av1 = divide_protected(self.arr_sum1, self.arr_sum0)
        arr_av2 = divide_protected(self.arr_sum2, self.arr_sum0)

        arr_rms = np.sqrt(arr_av2 - np.square(arr_av1))

        logger.debug(info_ndarr(arr_rms, 'arr_rms'))
        logger.debug(info_ndarr(arr_av1, 'arr_av1'))

        rms_min, rms_max = evaluate_limits(arr_rms, rmsnlo, rmsnhi, rms_lo, rms_hi, cmt='RMS')
        ave_min, ave_max = evaluate_limits(arr_av1, intnlo, intnhi, int_lo, int_hi, cmt='AVE')

        arr_sta_rms_hi = np.select((arr_rms>rms_max,),       (self.arr1,), 0)
        arr_sta_rms_lo = np.select((arr_rms<rms_min,),       (self.arr1,), 0)
        arr_sta_int_hi = np.select((self.sta_int_hi>nevlm,), (self.arr1,), 0)
        arr_sta_int_lo = np.select((self.sta_int_lo>nevlm,), (self.arr1,), 0)
        arr_sta_ave_hi = np.select((arr_av1>ave_max,),       (self.arr1,), 0)
        arr_sta_ave_lo = np.select((arr_av1<ave_min,),       (self.arr1,), 0)

        logger.info('bad pixel status:'\
               +'\n  status  1: %8d pixel rms       > %.3f' % (arr_sta_rms_hi.sum(), rms_max)\
               +'\n  status  2: %8d pixel rms       < %.3f' % (arr_sta_rms_lo.sum(), rms_min)\
               +'\n  status  4: %8d pixel intensity > %g in more than %g fraction (%d/%d) of non-empty events'%\
                     (arr_sta_int_hi.sum(), int_hi, fraclm, nevlm, counter)\
               +'\n  status  8: %8d pixel intensity < %g in more than %g fraction (%d/%d) of non-empty events'%\
                     (arr_sta_int_lo.sum(), int_lo, fraclm, nevlm, counter)\
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
        if plotim: self.plot_images(titpref='')

        self.block = None
        self.irec = -1
        logger.info('summary consumes %.3f sec' % (time()-t0_sec))


    def add_event(self, raw, irec):
        logger.debug(info_ndarr(raw, 'add_event %3d raw' % irec))
        _raw = raw & self.datbits # use data bits only 16-bit default (should be 14 for jungfrau and epix10ka)
        _raw_f64 = _raw.astype(np.float64)

        cond_lo = _raw<self.gate_lo
        cond_hi = _raw>self.gate_hi
        condlist = (np.logical_not(np.logical_or(cond_lo, cond_hi)),)

        self.arr_sum0   += np.select(condlist, (self.arr1u64,), 0)
        self.arr_sum1   += np.select(condlist, (_raw_f64,), 0)
        self.arr_sum2   += np.select(condlist, (np.square(_raw_f64),), 0)

        self.sta_int_lo += np.select((_raw<self.int_lo,), (self.arr1u64,), 0)
        self.sta_int_hi += np.select((_raw>self.int_hi,), (self.arr1u64,), 0)

        np.maximum(self.arr_max, _raw, out=self.arr_max)
        np.minimum(self.arr_min, _raw, out=self.arr_min)


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
            sys.stdout.write('1st stage event block processing is completed\n')
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


    def constants_av1_rms_sta(self):
        return self.arr_av1, self.arr_rms, self.arr_sta

    def constants_max_min(self):
        return self.arr_max, self.arr_min


def plot_image(nda, tit=''):
    """Plots averaged image"""
    from psana.detector.UtilsGraphics import gr

    #img = det.image(evt, nda)
    img = reshape_to_2d(nda)
    if img is None:
        sys.stdout.write('plot_image - image "%s" is not available.\n'%tit)
        return

    logger.info(info_ndarr(img, 'plot_image of %s' % tit))

    amin = np.quantile(img, 0.01, interpolation='lower')
    amax = np.quantile(img, 0.99, interpolation='higher')
    gr.plotImageLarge(img, amp_range=(amin, amax), title=tit)
    gr.show()


def add_metadata_kwargs(orun, odet, **kwa):

    trun_sec = up.seconds(orun.timestamp) # 1607569818.532117 sec

    # check opt "-t" if constants need to be deployed with diffiernt time stamp or run number
    tstamp = kwa.get('tstamp', None)
    use_external_run = tstamp is not None and tstamp<10000
    use_external_ts  = tstamp is not None and tstamp>9999
    tvalid_sec = time_sec_from_stamp(fmt=cc.TSFORMAT_SHORT, time_stamp=str(tstamp))\
                  if use_external_ts else trun_sec
    ivalid_run = tstamp if use_external_run else orun.runnum\
                  if not use_external_ts else 0

    kwa['exp']        = orun.expt
    kwa['experiment'] = orun.expt
    kwa['detector']   = odet.raw._uniqueid
    kwa['longname']   = odet.raw._uniqueid
    kwa['uniqueid']   = odet.raw._uniqueid
    kwa['detname']    = odet.raw._det_name
    kwa['dettype']    = odet.raw._dettype
    kwa['time_sec']   = tvalid_sec
    kwa['time_stamp'] = str_tstamp(fmt=cc.TSFORMAT, time_sec=int(tvalid_sec))
    kwa['tsshort']    = str_tstamp(fmt=cc.TSFORMAT_SHORT, time_sec=int(tvalid_sec))
    kwa['tstamp_orig']= str_tstamp(fmt=cc.TSFORMAT, time_sec=int(trun_sec))
    kwa['run']        = ivalid_run
    kwa['run_end']    = kwa.get('run_end', 'end')
    kwa['run_orig']   = orun.runnum
    kwa['version']    = kwa.get('version', 'N/A')
    kwa['comment']    = kwa.get('comment', 'no comment')
    kwa['extpars']    = {'content':'extended parameters dict->json->str',}
    return kwa


def fname_prefix(detname, tstamp, exp, runnum, dirname):
    return '%s/%s-%s-%s-r%04d' % (dirname, detname, tstamp, exp, runnum)


def deploy_constants(dic_consts, **kwa):

    from psana.pscalib.calib.MDBUtils import data_from_file
    from psana.pscalib.calib.MDBWebUtils import add_data_and_two_docs

    CTYPE_DTYPE = cc.dic_calib_name_to_dtype # {'pedestals': np.float32,...}
    repoman  = kwa.get('repoman', None)
    expname  = kwa.get('exp', None)
    detname  = kwa.get('det', None)
    dettype  = kwa.get('dettype', None)
    deploy   = kwa.get('deploy', False)
    dirrepo  = kwa.get('dirrepo', './work')
    dirmode  = kwa.get('dirmode',  0o2775)
    filemode = kwa.get('filemode', 0o664)
    group    = kwa.get('group', 'ps-users')
    tstamp   = kwa.get('tstamp', '2010-01-01T00:00:00')
    tsshort  = kwa.get('tsshort', '20100101000000')
    runnum   = kwa.get('run_orig',None)
    uniqueid = kwa.get('uniqueid', 'not-def-id')

    fmt_peds   = kwa.get('fmt_peds', '%.3f')
    fmt_rms    = kwa.get('fmt_rms',  '%.3f')
    fmt_status = kwa.get('fmt_status', '%4i')
    fmt_max    = kwa.get('fmt_max', '%i')
    fmt_min    = kwa.get('fmt_min', '%i')

    CTYPE_FMT = {'pedestals'   : fmt_peds,
                 'pixel_rms'   : fmt_rms,
                 'pixel_status': fmt_status,
#                 'pixel_max'   : fmt_max,
#                 'pixel_min'   : fmt_min,
                 'status_extra': fmt_status}

    if repoman is None:
       repoman = RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, group=group, dettype=dettype)
    #dircons = repoman.makedir_constants(dname='constants')
    #fprefix = fname_prefix(detname, tsshort, expname, runnum, dircons)

    panelid = uniqueid.split('_',1)[-1]
    logger.info('use panelid: %s' % panelid)

    for ctype, nda in dic_consts.items():

        dir_ct = repoman.makedir_ctype(panelid, ctype)
        fprefix = fname_prefix(detname, tsshort, expname, runnum, dir_ct)

        fname = '%s-%s.data' % (fprefix, ctype)
        fmt = CTYPE_FMT.get(ctype,'%.5f')
        save_ndarray_in_textfile(nda, fname, filemode, fmt)
        #save_2darray_in_textfile(nda, fname, filemode, fmt)

        logger.info('preserve constants in repository: %s' % fname)

        dtype = 'ndarray'
        kwa['iofname'] = fname
        kwa['ctype'] = ctype
        kwa['dtype'] = dtype
        kwa['extpars'] = {'content':'extended parameters dict->json->str',}
        _ = kwa.pop('exp',None) # remove parameters from kwargs - they passed as positional arguments
        _ = kwa.pop('det',None)

        logger.info('DEPLOY metadata: %s' % info_dict(kwa, fmt='%s: %s', sep='  ')) #fmt='%12s: %s'

        data = data_from_file(fname, ctype, dtype, True)
        logger.info(info_ndarr(data, 'constants loaded from file', last=10))

        if deploy:
            detname = kwa['longname']
            resp = add_data_and_two_docs(data, expname, detname, **kwa) # url=cc.URL_KRB, krbheaders=cc.KRBHEADERS
            if resp:
                #id_data_exp, id_data_det, id_doc_exp, id_doc_det = resp
                logger.debug('deployment id_data_exp:%s id_data_det:%s id_doc_exp:%s id_doc_det:%s' % resp)
            else:
                logger.info('constants are not deployed')
                exit()
        else:
            logger.warning('TO DEPLOY CONSTANTS IN DB ADD OPTION -D')



def pedestals_calibration(parser):

  args = parser.parse_args()
  kwa = vars(args)

  repoman = init_repoman_and_logger(parser=parser, **kwa)

  str_dskwargs = kwa.get('dskwargs', None)
  detname = kwa.get('det', None)
  nrecs   = kwa.get('nrecs', 100)
  stepnum = kwa.get('stepnum', None)
  stepmax = kwa.get('stepmax', 1)
  evskip  = kwa.get('evskip', 0)
  events  = kwa.get('events', 1000)

  dskwargs = up.datasource_kwargs_from_string(str_dskwargs)
  logger.info('DataSource kwargs: %s' % str(dskwargs))
  ds = DataSource(**dskwargs)

  t0_sec = time()
  tdt = t0_sec
  dpo = None
  nevtot = 0
  nevsel = 0
  nsteptot = 0
  break_loop = False
  dettype = None

  expname = dskwargs.get('exp', None)
  runnum  = dskwargs.get('run', None)

  for irun,orun in enumerate(ds.runs()):

    if expname is None: expname = orun.expt
    if runnum is None: runnum = orun.runnum

    nevrun = 0
    logger.info('\n==== %02d run: %d exp: %s' % (irun, runnum, expname))
    logger.info(up.info_run(orun, cmt='run info:\n    ', sep='\n    ', verb=3))

    odet = orun.Detector(detname)
    if dettype is None:
        dettype = odet.raw._dettype
        repoman.set_dettype(dettype)

    logger.info('created %s detector object' % detname)
    logger.info(up.info_detector(odet, cmt='  detector info:\n      ', sep='\n      '))

    runtstamp = orun.timestamp    # 4193682596073796843 relative to 1990-01-01
    trun_sec = up.seconds(runtstamp) # 1607569818.532117 sec
    ts_run, ts_now = tstamps_run_and_now(int(trun_sec))

    for istep,step in enumerate(orun.steps()):
      nsteptot += 1
      logger.info('Step %1d' % istep)
      ss = ''

      if istep>=stepmax:
          logger.info('==== Step:%02d loop is terminated --stepmax=%d' % (istep, stepmax))
          break_loop = True
          break

      elif stepnum is not None:
          if istep < stepnum:
              logger.info('==== Step:%02d is skipped --stepnum=%d' % (istep, stepnum))
              continue
          elif istep > stepnum:
              logger.info('==== Step:%02d loop is terminated --stepnum=%d' % (istep, stepnum))
              break_loop = True
              break

      if dpo is None:
         dpo = DarkProc(**kwa)
         dpo.runnum = orun.runnum
         dpo.exp = expname
         dpo.ts_run, dpo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)

      for ievt,evt in enumerate(step.events()):
        #print('Event %04d' % ievt, end='\r')
        sys.stdout.write('Event %04d\r' % ievt)
        nevrun += 1
        nevtot += 1

        if ievt < evskip:
            logger.debug('==== Ev:%04d is skipped --evskip=%d' % (ievt,evskip))
            continue
        elif evskip>0 and (ievt == evskip):
            s = 'Events < --evskip=%d are skipped' % evskip
            logger.info(s)

        if ievt > events-1:
            logger.info(ss)
            logger.info('\n==== Ev:%04d event loop is terminated --events=%d' % (ievt,events))
            break_loop = True
            break

        raw = odet.raw.raw(evt)

        if raw is None:
            logger.info('==== Ev:%04d raw is None' % (ievt))
            continue

        nevsel += 1

        tsec = time()
        dt   = tsec - tdt
        tdt  = tsec
        if selected_record(ievt+1, events):
            ss = 'run[%d] %d  step %d  events total/run/step/selected: %4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                 (irun, orun.runnum, istep, nevtot, nevrun, ievt+1, nevsel, time()-t0_sec, dt)
            logger.info(ss)

        status = dpo.event(raw,ievt)
        if status == 2:
            logger.info('requested statistics --nrecs=%d is collected - terminate loops' % nrecs)
            break_loop = True
            break
        # End of event-loop

      if ievt < events: logger.info('==== Ev:%04d end of events in run %d step %d'%\
                                     (ievt, orun.runnum, istep))
      if True:
          dpo.summary()
          #ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min') # 'status_extra'
          ctypes = ('pedestals', 'pixel_rms', 'pixel_status') # 'status_extra'
          arr_av1, arr_rms, arr_sta = dpo.constants_av1_rms_sta()
          arr_max, arr_min = dpo.constants_max_min()
          #consts = (arr_av1, arr_rms, arr_sta, arr_max, arr_min)
          consts = (arr_av1, arr_rms, arr_sta)

          logger.info('evaluated constants: \n  %s\n  %s\n  %s\n  %s\n  %s' % (
                      info_ndarr(arr_av1, 'arr_av1', first=0, last=5),\
                      info_ndarr(arr_rms, 'arr_rms', first=0, last=5),\
                      info_ndarr(arr_sta, 'arr_sta', first=0, last=5),\
                      info_ndarr(arr_max, 'arr_max', first=0, last=5),\
                      info_ndarr(arr_min, 'arr_min', first=0, last=5)))
          dic_consts = dict(zip(ctypes, consts))
          kwa_depl = add_metadata_kwargs(orun, odet, **kwa)
          kwa_depl['repoman'] = repoman
          deploy_constants(dic_consts, **kwa_depl)
          del(dpo)
          dpo=None

      if break_loop:
        logger.info('terminate_steps')
        break # break step loop

    if break_loop:
      logger.info('terminate_runs')
      break # break run loop

  logger.debug('run/step/event loop is completed')
  repoman.logfile_save()


if __name__ == "__main__":

    sys.stdout.write(80*'_', '\n')
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.INFO)

    kwa = {\
        'fname'   : None,\
        'exp'     : 'tmoc00118',\
        'runs'    : '123',\
        'det'     : 'tmoopal',\
        'nrecs1'  : 100,\
        'nrecs'   : 200,\
        'plotim'  : 0o17777,\
    }

    pedestals_calibration(**kwa)

    sys.exit('End of %s' % sys.argv[0])

# EOF
