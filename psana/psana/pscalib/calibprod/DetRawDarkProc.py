#!/usr/bin/env python


"""
   TODO:
   1. #ecm = EventCodeManager(evcode, verbos) - selection of events for event code is commented
   2. cmod = None - file is not saved
   3. deploy calib files in db
"""

#----------
import logging
logger = logging.getLogger(__name__)

import os
import sys
import psana
import numpy as np
from time import time, localtime, strftime

#from Detector.EventCodeManager import EventCodeManager
from psana.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr, divide_protected
from psana.pyalgos.generic.Utils import do_print
from psana import DataSource
from psana.pscalib.calib.NDArrIO import save_txt
from psana.pscalib.calib.CalibConstants import TSFORMAT

#----------

def plot_det_image(det, evt, nda, tit=''):
    """Plots averaged image
    """
    import psana.pyalgos.generic.Graphics as gr
    from psana.pyalgos.generic.NDArrUtils import reshape_to_2d
    #img = det.image(evt, nda)
    img = np.array(nda)
    if img is None: sys.exit('Image is not available. PROCESSING IS TERMINATED')
    img = reshape_to_2d(img)
    
    ave, rms = nda.mean(), nda.std()
    gr.plotImageLarge(img, amp_range=(ave-1*rms, ave+3*rms), title=tit)
    gr.show()

#----------

def evaluate_limits(arr, nneg=5, npos=5, lim_lo=1, lim_hi=1000, cmt=''):
    """Evaluates low and high limit of the array, which are used to find bad pixels.
    """
    ave, std = (arr.mean(), arr.std()) if (nneg>0 or npos>0) else (None,None)
    lo = ave-nneg*std if nneg>0 else lim_lo
    hi = ave+npos*std if npos>0 else lim_hi
    lo, hi = max(lo, lim_lo), min(hi, lim_hi)

    logger.debug('  %s: %s ave, std = %.3f, %.3f  low, high limits = %.3f, %.3f'%\
              (sys._getframe().f_code.co_name, cmt, ave, std, lo, hi))

    return lo, hi

#----------

def str_tstamp(fmt=TSFORMAT, time_sec=None): #fmt='%Y-%m-%dT%H:%M:%S
    return strftime(fmt, localtime(time_sec))

#----------

def replace(in_tmp, pattern, subst):
    """If pattern in the in_tmp replace it with subst.
       Returns str object in_tmp with replaced patterns. 
    """
    fields = in_tmp.split(pattern, 1) 
    if len(fields) > 1:
        return '%s%s%s' % (fields[0], subst, fields[1])
    else:
        return in_tmp

#----------

def fname_template(orun, detname, ofname, nevts, tsec=None, tnsec=None):
    """Replaces parts of the file name specified as
       #src, #exp, #run, #evts, #type, #date, #time, #fid, #sec, #nsec
       with actual values
    """
    template = replace(ofname,   '#src', detname)
    template = replace(template, '#exp', orun.expt)
    template = replace(template, '#run', 'r%04d'%orun.runnum)
    template = replace(template, '#type', '%s')

    t_sec  = tsec  if tsec  is not None else int(orun.timestamp>>32 & 0xFFFFFFFF)
    t_nsec = tnsec if tnsec is not None else int(orun.timestamp & 0xFFFFFFFF)

    template = replace(template, '#date', str_tstamp('%Y-%m-%d', t_sec))
    template = replace(template, '#time', str_tstamp('%H%M%S', t_sec))
    template = replace(template, '#sec',  '%d' % t_sec)
    template = replace(template, '#nsec', '%09d' % t_nsec)
    template = replace(template, '#evts', 'e%06d' % nevts)
    if not '%s' in template: template += '-%s'
    return template

#----------

class DetRawDarkProc:

    def __init__(self, orun, detname, args):

        self.args    = args
        self.orun    = orun
        self.detname = detname

        self.ofname = args.ofname
        self.events = args.events
        self.evskip = args.evskip

        self.int_lo = args.intlow if args.intlow is not None else 1
        self.int_hi = args.inthig if args.inthig is not None else 16000
        self.rms_lo = args.rmslow # if None - will be evaluated later
        self.rms_hi = args.rmshig # if None - will be evaluated later
        self.fraclm = args.fraclm

        self.plotim = args.plotim
        self.savebw = args.savebw

        self.intnlo = args.intnlo
        self.intnhi = args.intnhi
        self.rmsnlo = args.rmsnlo
        self.rmsnhi = args.rmsnhi

        self.evstg1 = max(self.events*0.05, 5)
        self.nsigma = args.nsigma
        self.addcdb = args.addcdb

        self.loglev = args.loglev
        self.verbos = o0377 if self.loglev=='DEBUG' else 0 # deprecated
        
        self.det     = None
        self.shape   = None
        self.counter = 0
        self.stage   = 0

        self.print_attrs()
        logger.info('Begin processing')


    def print_attrs(self):
        msg = '\n%s\nAttributes of the %s object' % (60*'_', self.__class__.__name__)\
            + '\ndetector name                   : %s'   % self.detname\
            + '\noutput file name template       : %s'   % self.ofname\
            + '\nnumber of events to collect     : %d'   % self.events\
            + '\nnumber of events to skip        : %d'   % self.evskip\
            + '\nnumber of events for stage 1    : %d'   % self.evstg1\
            + '\nnumber of sigma intensity range : %.3f' % self.nsigma\
            + '\ncontrol bit-word to plot images : %d'   % self.plotim\
            + '\ncontrol bit-word to save arrays : %d'   % self.savebw\
            + '\nINTENSITY low limit (ADU)       : %d'   % self.int_lo\
            + '\nINTENSITY high limit (ADU)      : %d'   % self.int_hi\
            + '\nRMS low limit (ADU)             : %d'   % self.rms_lo\
            + '\nRMS high limit (ADU)            : %d'   % self.rms_hi\
            + '\nallowed low/high fraction limit : %.3f' % self.fraclm\
            + '\n\nParameters for auto-evaluation of limits:'\
            + '\nnumber of sigma from mean for INTENSITY low limit  : %.3f' % self.intnlo\
            + '\nnumber of sigma from mean for INTENSITY high limit : %.3f' % self.intnhi\
            + '\nnumber of sigma from mean for RMS low limit        : %.3f' % self.rmsnlo\
            + '\nnumber of sigma from mean for RMS high limit       : %.3f' % self.rmsnhi\
            + '\n' + 60*'_'
        logger.info(msg)


    def _common_mode_pars(self, arr_ave, arr_rms, arr_msk):
        """Returns detector-dependent common mode parameters as np.array for a few detectors and None for others.
        """
        import PSCalib.GlobalUtils as gu 
        import math

        dettype = gu.det_type_from_source(self.detname)

        ave = arr_ave[arr_msk>0].mean()
        #rms = arr_ave[arr_msk>0].std()
        rms = arr_rms[arr_msk>0].mean()

        logger.info('Evaluate common mode for source: %s det: %s, estimated intensity ave: %.3f  rms: %.3f' %\
                  (self.detname, gu.dic_det_type_to_name[dettype], ave, rms))

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


    def _init_stage1(self, evt):

        if self.shape is not None: return True

        if self.det is None: self.det = self.orun.Detector(self.detname)

        if self.det is None:
            raise ValueError('Can not create the Detector object for name %s' % self.detname)

        ndaraw = self.det.raw(evt)

        if ndaraw is None: return False
        if ndaraw.size == 0: return False
        if ndaraw.shape[0] == 0: return False
        if not ndaraw.any(): return False

        self.ev1_sec = evt._seconds
        self.ev1_nsec = evt._nanoseconds
        logger.info('The 1st non-empty event time %d sec %d nsec' % (self.ev1_sec, self.ev1_nsec))
        #print('  YYYY evt.timestamp     :', evt.timestamp)
        #print('  YYYY evt.timestamp sec :', int(evt.timestamp>>32 & 0xFFFFFFFF))
        #print('  YYYY evt.timestamp nsec:', int(evt.timestamp     & 0xFFFFFFFF))

        self.shape=ndaraw.shape

        logger.info('Begin stage 1 for %s, raw.shape = %s, intensity limits low: %.1f  high: %.1f'%\
                    (self.detname, str(self.shape), self.int_lo, self.int_hi))
    
        self.arr0       = np.zeros(self.shape, dtype=np.int64)
        self.arr1       = np.ones (self.shape, dtype=np.int64)

        self.sta_int_lo = np.zeros(self.shape, dtype=np.int64)
        self.sta_int_hi = np.zeros(self.shape, dtype=np.int64)

        self.arr_sum0   = np.zeros(self.shape, dtype=np.int64)
        self.arr_sum1   = np.zeros(self.shape, dtype=np.double)
        self.arr_sum2   = np.zeros(self.shape, dtype=np.double)

        self.gate_lo    = self.arr1 * self.int_lo
        self.gate_hi    = self.arr1 * self.int_hi

        if self.savebw & 16: self.arr_max = np.zeros(self.shape, dtype=ndaraw.dtype)
        if self.savebw & 32: self.arr_min = np.ones (self.shape, dtype=ndaraw.dtype) * 0xffff

        self.stage = 1
        
        return True


    def event(self, evt):

        if not self._init_stage1(evt): return

        ndaraw = self.det.raw(evt)
        if ndaraw is None: return
        ndadbl = np.array(ndaraw, dtype=np.double)

        self._init_stage2(ndaraw)
        self._proc_event(ndadbl, ndaraw)


    def _proc_event(self, ndadbl, ndaraw):

        self.counter += 1

        cond_lo = ndaraw<self.gate_lo
        cond_hi = ndaraw>self.gate_hi
        condlist = (np.logical_not(np.logical_or(cond_lo, cond_hi)),)

        #logger.info('XXX:')
        #print_ndarr(self.arr1, 'arr1')
        #print_ndarr(self.arr_sum0, 'arr_sum0')
        #print_ndarr(self.arr_sum1, 'arr_sum1')
        #print_ndarr(self.arr_sum2, 'arr_sum2')

        self.arr_sum0   += np.select(condlist, (self.arr1,), 0)
        self.arr_sum1   += np.select(condlist, (ndadbl,), 0)
        self.arr_sum2   += np.select(condlist, (np.square(ndadbl),), 0)

        self.sta_int_lo += np.select((ndaraw<self.int_lo,), (self.arr1,), 0)
        self.sta_int_hi += np.select((ndaraw>self.int_hi,), (self.arr1,), 0)

        if self.savebw & 16: self.arr_max = np.maximum(self.arr_max, ndaraw)
        if self.savebw & 32: self.arr_min = np.minimum(self.arr_min, ndaraw)


    def _init_stage2(self, ndaraw):

        if self.stage & 2: return
        if self.counter < self.evstg1: return
 
        t0_sec = time()

        arr_av1 = divide_protected(self.arr_sum1, self.arr_sum0)
        arr_av2 = divide_protected(self.arr_sum2, self.arr_sum0)

        arr_rms = np.sqrt(arr_av2 - np.square(arr_av1))
        rms_ave = arr_rms.mean()

        gate_half = self.nsigma*rms_ave

        logger.info('%s\nBegin stage 2 for %s after %d events' % (80*'_', self.detname, self.counter))
        logger.info('  mean rms=%.3f x %.1f = intensity gate= +/- %.3f around pixel average intensity' %\
                  (rms_ave, self.nsigma, gate_half))

        #print_ndarr(arr_av1, 'arr_av1')

        self.gate_hi = np.minimum(arr_av1 + gate_half, self.arr1*self.int_hi)
        self.gate_lo = np.maximum(arr_av1 - gate_half, self.arr1*self.int_lo)

        self.gate_hi = np.array(self.gate_hi, dtype=ndaraw.dtype)
        self.gate_lo = np.array(self.gate_lo, dtype=ndaraw.dtype)

        self.arr_sum0 = np.zeros(self.shape, dtype=np.int64)
        self.arr_sum1 = np.zeros(self.shape, dtype=np.double)
        self.arr_sum2 = np.zeros(self.shape, dtype=np.double)

        self.stage = 2

        logger.info('Stage 2 initialization for %s consumes dt=%7.3f sec' % (self.detname, time()-t0_sec))


    def summary(self, evt):

        logger.info('%s\nRaw data for %s found/selected in %d events' % (80*'_', self.detname, self.counter))

        if self.counter:
            logger.info(', begin data summary stage')
        else:
            logger.info(', STOP processing, there are no arrays to save...')
            return 

        t0_sec = time()

        # make shorter references
        det     = self.det
        ofname  = self.ofname
        plotim  = self.plotim
        savebw  = self.savebw
        int_hi  = self.int_hi
        int_lo  = self.int_lo
        fraclm  = self.fraclm
        counter = self.counter

        arr_av1 = divide_protected(self.arr_sum1, self.arr_sum0)
        arr_av2 = divide_protected(self.arr_sum2, self.arr_sum0)

        frac_int_lo = np.array(self.sta_int_lo/counter, dtype=np.float32)
        frac_int_hi = np.array(self.sta_int_hi/counter, dtype=np.float32)

        arr_rms = np.sqrt(arr_av2 - np.square(arr_av1))
        
        rms_min, rms_max = evaluate_limits(arr_rms, self.rmsnlo, self.rmsnhi, self.rms_lo, self.rms_hi, cmt='RMS')
        ave_min, ave_max = evaluate_limits(arr_av1, self.intnlo, self.intnhi, self.int_lo, self.int_hi, cmt='AVE')

        arr_sta_rms_hi = np.select((arr_rms>rms_max,),    (self.arr1,), 0)
        arr_sta_rms_lo = np.select((arr_rms<rms_min,),    (self.arr1,), 0)
        arr_sta_int_hi = np.select((frac_int_hi>fraclm,), (self.arr1,), 0)
        arr_sta_int_lo = np.select((frac_int_lo>fraclm,), (self.arr1,), 0)
        arr_sta_ave_hi = np.select((arr_av1>ave_max,),    (self.arr1,), 0)
        arr_sta_ave_lo = np.select((arr_av1<ave_min,),    (self.arr1,), 0)

        logger.info('  Bad pixel status:') 
        logger.info('  status  1: %8d pixel rms       > %.3f' % (arr_sta_rms_hi.sum(), rms_max))
        logger.info('  status  8: %8d pixel rms       < %.3f' % (arr_sta_rms_lo.sum(), rms_min))
        logger.info('  status  2: %8d pixel intensity > %g in more than %g fraction of events' % (arr_sta_int_hi.sum(), int_hi, fraclm))
        logger.info('  status  4: %8d pixel intensity < %g in more than %g fraction of events' % (arr_sta_int_lo.sum(), int_lo, fraclm))
        logger.info('  status 16: %8d pixel average   > %g'   % (arr_sta_ave_hi.sum(), ave_max))
        logger.info('  status 32: %8d pixel average   < %g'   % (arr_sta_ave_lo.sum(), ave_min))

        #0/1/2/4/8/16/32 for good/hot-rms/saturated/cold/cold-rms/average above limit/average below limit, 
        arr_sta = np.zeros(self.shape, dtype=np.int64)
        arr_sta += arr_sta_rms_hi    # hot rms
        arr_sta += arr_sta_rms_lo*8  # cold rms
        arr_sta += arr_sta_int_hi*2  # satturated
        arr_sta += arr_sta_int_lo*4  # cold
        arr_sta += arr_sta_ave_hi*16 # too large average
        arr_sta += arr_sta_ave_lo*32 # too small average
        
        arr_msk  = np.select((arr_sta>0,), (self.arr0,), 1)

        cmod = None # self._common_mode_pars(arr_av1, arr_rms, arr_msk)

        if plotim &  1: plot_det_image(det, evt, arr_av1,         tit='average')
        if plotim &  2: plot_det_image(det, evt, arr_rms,         tit='RMS')
        if plotim &  4: plot_det_image(det, evt, arr_sta,         tit='status')
        if plotim &  8: plot_det_image(det, evt, arr_msk,         tit='mask')
        if plotim & 16: plot_det_image(det, evt, self.arr_max,    tit='maximum')
        if plotim & 32: plot_det_image(det, evt, self.arr_min,    tit='minimum')
        if plotim & 64: plot_det_image(det, evt, self.sta_int_lo, tit='statistics below threshold')
        if plotim &128: plot_det_image(det, evt, self.sta_int_hi, tit='statistics above threshold')
        
        cmts = ['DATASET  %s' % self.detname, 'STATISTICS  %d' % counter]
        
        # Save n-d array in text file % 
        template = fname_template(self.orun, self.detname, ofname, counter, self.ev1_sec, self.ev1_nsec)
        addmetad=True

        verbos = self.verbos

        if savebw &  1: save_txt(template % 'ave', arr_av1,      cmts + ['ARR_TYPE  average'], '%8.2f', verbos, addmetad)
        if savebw &  2: save_txt(template % 'rms', arr_rms,      cmts + ['ARR_TYPE  RMS'],     '%8.2f', verbos, addmetad)
        if savebw &  4: save_txt(template % 'sta', arr_sta,      cmts + ['ARR_TYPE  status'],  '%d',    verbos, addmetad)
        if savebw &  8: save_txt(template % 'msk', arr_msk,      cmts + ['ARR_TYPE  mask'],    '%1d',   verbos, addmetad)
        if savebw & 16: save_txt(template % 'max', self.arr_max, cmts + ['ARR_TYPE  max'],     '%d',    verbos, addmetad)
        if savebw & 32: save_txt(template % 'min', self.arr_min, cmts + ['ARR_TYPE  min'],     '%d',    verbos, addmetad)
        if savebw & 64 and cmod is not None:
            np.savetxt(template % 'cmo', cmod, fmt='%d', delimiter=' ', newline=' ')
            save_txt(template % 'cmm', cmod, cmts + ['ARR_TYPE  common_mode'],'%d', verbos, False)

        args    = vars(self.args)
        version = args.get('version', str_tstamp('V%Y%m%dT%H%M%S', t0_sec))
        orun = self.orun
        loglev = args.get('loglev', 'INFO')

        # args.expnam # args.runnum, orun.timestamp
        kwa = {'detname'   : self.detname,
               'experiment': orun.expt,
               'runnum'    : orun.runnum,
               'timestamp' : str_tstamp(time_sec=self.ev1_sec),
               'version'   : version,
               'comment'   : str(cmts),
               'loglev'    : loglev,
              }

        logger.debug('kwa:', kwa)
        logger.debug('args:', self.args)

        if self.addcdb:
            from psana.pscalib.calib.MDBCommands import add_constants
        
            if savebw &  1: add_constants(template % 'ave', 'pedestals',    **kwa)
            if savebw &  2: add_constants(template % 'rms', 'pixel_rms',    **kwa) 
            if savebw &  4: add_constants(template % 'sta', 'pixel_status', **kwa) 
            if savebw &  8: add_constants(template % 'msk', 'pixel_mask',   **kwa)  
            if savebw & 16: add_constants(template % 'max', 'pixel_max',    **kwa)   
            if savebw & 32: add_constants(template % 'min', 'pixel_min',    **kwa)    

        logger.info('Data summary for %s is completed, dt=%7.3f sec' % (self.detname, time()-t0_sec))

#----------

def detectors_dark_proc(parser):

    args = parser.parse_args()
    #expnam, runnum, dnames, ifname, ofname, events, evskip, intlow, inthig, rmslow, rmshig, fraclm, nsigma,\
    #                    plotim, savebw, intnlo, intnhi, rmsnlo, rmsnhi, evcode

    EVSKIP = args.evskip
    EVENTS = args.events + EVSKIP

    logger.info('Raw data processing of exp: %s run: %d for detector(s): %s' % (args.expnam, args.runnum, args.dnames))
    logger.info('input file: %s' % (args.ifname))

    ds = DataSource(files=args.ifname)
    run = next(ds.runs())
    logger.info('\t RunInfo expt: %s runnum: %d\n' % (run.expt, run.runnum))

    print('dir(run):',dir(run))
    print(':', run.timestamp)
    print(':', run._get_runinfo())
    print(':', run.xtcinfo)
    print(':', run.id)
    #sys.exit('TEST EXIT')
    ###==================

    lst_dpo = [DetRawDarkProc(run, dname, args) for dname in args.dnames.split(',')]

    #ecm = EventCodeManager(evcode, verbos)
           
    t0_sec = time()
    tdt = t0_sec

    for i,evt in enumerate(run.events()):
        if i<EVSKIP: continue
        if not i<EVENTS: break
        #if not ecm.select(evt): continue 

        for dpo in lst_dpo: dpo.event(evt)
        if do_print(i):
            tsec = time()
            dt   = tsec - tdt
            tdt  = tsec
            logger.info('  Event: %4d,  time=%7.3f sec,  dt=%5.3f sec' % (i, time()-t0_sec, dt))

    for dpo in lst_dpo: dpo.summary(evt)

    logger.info('%s\n Processed %d events, consumed time = %f sec.' % (80*'_', i, time()-t0_sec))

#----------

if __name__ == "__main__":
    sys.exit('run it by command det_raw_dark_proc')

#----------
