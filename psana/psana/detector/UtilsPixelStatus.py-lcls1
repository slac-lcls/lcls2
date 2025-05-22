
""" UtilsPixelStatus.py - utilities for command det_pixel_status

    import Detector.UtilsPixelStatus as us

    us.save_constants_in_repository(arr, **kwargs)
    - saves user's array of constants in reppository defined by **kwargs
"""

from time import time
import os
import sys
import numpy as np

import Detector.UtilsLogging as ul
logger = ul.logging.getLogger(__name__)  # where __name__ = 'Detector.UtilsPixelStatus'
from psana import DataSource, Detector
from Detector.GlobalUtils import info_ndarr  # print_ndarr, divide_protected
import PSCalib.GlobalUtils as gu
import Detector.UtilsCalib as uc
import Detector.RepoManager as rm
from pyimgalgos.HBins import HBins

def metadata(ds, orun, det):
    """ returns (dict) metadata evaluated from input objects"""
    import Detector.UtilsDeployConstants as udc
    env = ds.env()
    dettype = det.pyda.dettype
    ts_run, ts_now = uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)
    runnum = orun.run()
    return {
      'dettype': dettype,
      'typename': gu.dic_det_type_to_name.get(dettype, None).lower(),
      'detid': udc.id_det(det, env),
      'detname': det.name,
      'ts_run': ts_run,
      'ts_now': ts_now,
      'expname': env.experiment(),
      'calibdir': env.calibDir(),
      'runnum': runnum,
      'pedestals': det.pedestals(runnum),
      'rms': det.rms(runnum),
      }

def fplane(x, y, p):
    return x*p[0] + y*p[1] + p[2]

def fit_to_plane(xs, ys, zs):
    """xs, ys, zs - np.arrays (OF THE SAME SIZE) of (x,y) points and value z.
       Input coordinates xs, ys are not necessarily on grid.
    """
    assert isinstance(xs, np.ndarray)
    assert isinstance(ys, np.ndarray)
    assert isinstance(zs, np.ndarray)
    A = np.matrix([[x, y, 1] for x,y in zip(xs.ravel(), ys.ravel())])
    b = np.matrix([z for z in zs.ravel()]).T
    pars = (A.T * A).I * A.T * b
    resids = b - A * pars
    pars = [p[0] for p in pars.tolist()]  # np.matrix -> list
    resids = np.array([r[0] for r in resids.tolist()])
    return pars, resids

def residuals_to_plane(a):
    """uses a.shape to make a grid, fit array to plane on grid, find and return residuals."""
    assert isinstance(a, np.ndarray)
    sh = a.shape
    cs, rs = np.meshgrid(range(sh[1]), range(sh[0]))
    logger.debug('in residuals_to_plane input to fit_to_plane:\n  %s\n  %s\n  %s'%\
                 (info_ndarr(a,  '   arr '),
                  info_ndarr(cs, '   cols'),
                  info_ndarr(rs, '   rows')))
    _, res = fit_to_plane(cs, rs, a)
    res.shape = sh
    return res

def find_outliers(arr, title='', vmin=None, vmax=None, fmt='%.3f'):
    assert isinstance(arr, np.ndarray)
    size = arr.size
    arr0 = np.zeros_like(arr, dtype=bool)
    arr1 = np.ones_like(arr, dtype=np.uint64)
    bad_lo = arr0 if vmin is None else arr <= vmin
    bad_hi = arr0 if vmax is None else arr >= vmax
    arr1_lo = np.select((bad_lo,), (arr1,), 0)
    arr1_hi = np.select((bad_hi,), (arr1,), 0)
    sum_lo = arr1_lo.sum()
    sum_hi = arr1_hi.sum()
    s_lo = '%8d / %d (%6.3f%%) pixels %s <= %s'%\
            (sum_lo, size, 100*sum_lo/size, title, 'unlimited' if vmin is None else fmt % vmin)
    s_hi = '%8d / %d (%6.3f%%) pixels %s >= %s'%\
            (sum_hi, size, 100*sum_hi/size, title, 'unlimited' if vmax is None else fmt % vmax)
    return bad_lo, bad_hi, arr1_lo, arr1_hi, s_lo, s_hi

def evaluate_pixel_status(arr, title='', vmin=None, vmax=None, snrmax=8):
    """vmin/vmax - absolutly allowed min/max of the value"""
    assert isinstance(arr, np.ndarray)
    bad_lo, bad_hi, arr1_lo, arr1_hi, s_lo, s_hi = find_outliers(arr, title=title, vmin=vmin, vmax=vmax, fmt='%.0f')

    arr_sel = arr[np.logical_not(np.logical_or(bad_lo, bad_hi))]
    med = np.median(arr_sel)
    spr = np.median(np.absolute(arr_sel-med))  # axis=None, out=None, overwrite_input=False, keepdims=False
    if spr == 0:
       spr = np.std(arr_sel)
       logger.warning('MEDIAN OF SPREAD FOR INT VALUES IS 0 replaced with STD = % .3f' % spr)

    _vmin = med - snrmax*spr if vmin is None else max(med - snrmax*spr, vmin)
    _vmax = med + snrmax*spr if vmax is None else min(med + snrmax*spr, vmax)

    s_sel = '%s selected %d of %d pixels in' % (title, arr_sel.size, arr.size)\
          + ' range (%s, %s)' % (str(vmin), str(vmax))\
          + ' med: %.3f spr: %.3f' % (med, spr)

    s_range = u're-defined range for med \u00B1 %.1f*spr: (%.3f, %.3f)' % (snrmax, _vmin, _vmax)
    _, _, _arr1_lo, _arr1_hi, _s_lo, _s_hi = find_outliers(arr, title=title, vmin=_vmin, vmax=_vmax)

    gap = 13*' '
    logger.info('%s\n    %s\n         absolute limits:\n  %s\n  %s\n         evaluated limits:\n%s%s\n%s%s\n  %s\n  %s' %\
        (20*'=', info_ndarr(arr, title, last=0), s_lo, s_hi, gap, s_sel, gap, s_range, _s_lo, _s_hi))

    return _arr1_lo, _arr1_hi, _s_lo, _s_hi

def set_pixel_status_bits(status, arr, title='', vmin=None, vmax=None,\
                          snrmax=8, bit_lo=1<<0, bit_hi=1<<1):
    assert isinstance(arr, np.ndarray)
    arr1_lo, arr1_hi, s_lo, s_hi = evaluate_pixel_status(arr, title=title,\
                                        vmin=vmin, vmax=vmax, snrmax=snrmax)
    status += arr1_lo * bit_lo
    status += arr1_hi * bit_hi
    s_lo = '%20s: %s' % (oct(bit_lo), s_lo)
    s_hi = '%20s: %s' % (oct(bit_hi), s_hi)
    return s_lo, s_hi

def edges1d(size, step):
    """returns (list) [0, step, 2*step, ..., (size-step)]"""
    assert isinstance(size, int)
    assert isinstance(step, int)
    assert size>step
    es = list(range(0, size, step))
    if es[-1] > (size-step): es[-1] = size-step
    return es

def corners2d(shf, shw):
    """returns list of 2-d corner indices [(r0,c0), (r0,c1), ...]
       parameters: 2-d shape of the frame and 2-d shape of the window
    """
    assert len(shf) == 2
    assert len(shw) == 2
    rows = edges1d(shf[0], shw[0])
    cols = edges1d(shf[1], shw[1])
    cs, rs = np.meshgrid(cols, rows)
    return list(zip(rs.ravel(), cs.ravel()))

def selected_number(n):
    return n<5\
       or (n<50 and not n%10)\
       or (n<500 and not n%100)\
       or (not n%1000)

def bifurgaus(x, *p):
    """Bifurcated Gaussian - assumes ydata = f(xdata, *params) + eps."""
    a0, x0, s1, s2 = p
    sig = np.select((x<x0,), (s1,), default=s2)
    return a0 * np.exp(-0.5*((x-x0)/sig)**2)

def fit_bifurgaus(x, y, p0=None):
    from scipy.optimize import curve_fit
    return curve_fit(bifurgaus, x, y, p0=p0)

class DataProc:
    """data accumulation and processing.
    dpo = DataProc(...)
    dpo.event(raw, evnum)
    arr_status = dpo.summary()
    save_constants(arr_status, args, metad)
    """
    def __init__(self, args, metad, **kwa):
        self.args = args
        self.kwa = kwa
        self.aslice   = args.aslice  # None or eval('np.s_[%s]' % args.slice)
        self.segind   = args.segind
        self.nrecs    = args.nrecs
        self.snrmax   = args.snrmax
        self.gainbits = args.gainbits
        self.databits = args.databits
        self.irec     = -1
        self.state    = 0
        self.shwind   = eval('(%s)' % args.shwind)
        self.block    = None
        self.nda_max  = None
        self.features = eval('(%s)' % args.features)
        self.metad    = metad
        self.fname    = fname_part(args, metad)
        logger.info('file name for data block: %s' % self.fname)
        self._do_max = 11 in self.features
        self._do_block = not self._do_max
        self._dir_work = None

    def init_block(self, raw, do_load=False):
        self.shape_fr = tuple(raw.shape)
        self.shape_bl = (self.nrecs,) + self.shape_fr
        if do_load: self.load_block_file()
        if self.block is not None: return
        self.block = np.zeros(self.shape_bl, dtype=raw.dtype)
        logger.info(info_ndarr(self.block, 'created empty data block'))
        self.state = 0
        self.irec = -1
        self.add_record(raw)

    def add_record(self, raw):
        """check if block has space for the next record, increment irec and add record,
           othervise set self.state = 1 - block is full
        """
        if self.irec < self.nrecs-1:
            self.irec +=1
            self.block[self.irec,:] = raw
        else:
            self.state = 1 # block is full
            logger.info('Ev total: %d accumulated requested number of records --nrecs: %d - break' % (self._evnum, self.irec))

    def feature_01(self):
        logger.info("""Feature 1: mean intensity of frames in good range""")
        block_data = self.block & self.databits
        intensity_mean = np.sum(block_data, axis=(-2,-1)) / self.ssize
        logger.info(info_ndarr(intensity_mean, '\n  per-record intensity MEAN IN FRAME:', last=20))

        intensity_med = np.median(block_data, axis=(-2,-1))
        logger.info(info_ndarr(intensity_med, '\n  per-record intensity MEDIAN IN FRAME:', last=20))
        arr1_lo, arr1_hi, s_lo, s_hi = evaluate_pixel_status(intensity_med, title='Feat.1: intensity_med',\
                                             vmin=0, vmax=self.databits, snrmax=self.snrmax)
        arr0 = np.zeros_like(arr1_lo, dtype=np.uint64)  # dtype=bool
        arr1_good_frames = np.select((arr1_lo>0, arr1_hi>0), (arr0, arr0), 1)
        logger.info('Total number of good events: %d' % arr1_good_frames.sum())
        return arr1_good_frames

    def feature_02(self):
        logger.info("""Feature 2: dark mean in good range""")
        block_good = self.block[self.bool_good_frames,:] & self.databits
        logger.info(info_ndarr(block_good, 'block of good records:', last=5))
        return np.mean(block_good, axis=0, dtype=np.float)

    def feature_03(self):
        logger.info("""Feature 3: dark RMS in good range""")
        block_good = self.block[self.bool_good_frames,:] & self.databits
        logger.info(info_ndarr(block_good, 'block of good records:', last=5))
        return np.std(block_good, axis=0, dtype=np.float)

    def feature_04(self):
        logger.info("""Feature 4: TBD""")
        return None

    def feature_05(self):
        logger.info("""Feature 5: TBD""")
        return None

    def residuals_frame_f06(self, frame):
        assert isinstance(frame, np.ndarray)
        assert frame.ndim==2
        corners = corners2d(frame.shape, self.shwind)
        logger.debug('Feature 6: in frame %s window %s corners:\n %s\nNumber of corners: %d'%\
                     (str(frame.shape), str(self.shwind), str(corners), len(corners)))

        residuals = np.zeros_like(frame, dtype=np.float)
        wr, wc = self.shwind
        for r,c in corners: # evaluate residuals to the plane fit in frame windows
            sl = np.s_[r:r+wr, c:c+wc]
            arrw = frame[sl]
            res = residuals_to_plane(arrw)
            res_med = np.median(res)
            res_spr = np.median(np.absolute(res - res_med))
            residuals[sl] = res
            logger.debug('  corner r: %d c:%d %s\n   residuals med: %.3f spr: %.3f'%\
                         (r, c, info_ndarr(res, 'residuals'), res_med, res_spr))
        return residuals

    def feature_06(self):
        logger.info("""Feature 6: light average SNR of pixels over time""")

        ngframes = self.inds_good_frames.size
        shape_res = (ngframes,) + self.shape_fr
        block_res = np.zeros(shape_res, dtype=np.float)

        for i, igood in enumerate(self.inds_good_frames):
            frame = self.block[igood,:] & self.databits
            logger.debug(info_ndarr(frame, '%04d frame data' % igood))
            block_res[i,:] = res = self.residuals_frame_f06(frame) #  self.block[0,:])

            res_med = np.median(res)
            res_spr = np.median(np.absolute(res - res_med))

            s = 'frame: %04d res_med: %.3f res_spr: %.3f frame residuals' % (igood, res_med, res_spr)
            logger.info(info_ndarr(res, s, last=3))

        return block_res

    def feature_11(self):
        logger.info("""Feature 11: light intensity max-peds in good range""")
        snrmax = self.snrmax

        peds = self.metad['pedestals']
        logger.info(info_ndarr(peds, 'pedestals:', last=5))
        self.max_peds = max_peds = self.nda_max - peds
        med = np.median(max_peds)
        d = max_peds - med
        logger.info(info_ndarr(d, 'max-peds-med:', last=5))
        absspr = np.median(np.absolute(d))
        pos = med
        logger.info('\n  median(max-peds): %.3f\n  absolute spread: %.3f' % (med, absspr))

        diff_pos = np.median(d[d>=0])
        diff_neg = np.median(d[d<0])

        self.max_lim_hi = pos + snrmax*diff_pos
        self.max_lim_lo = pos + snrmax*diff_neg

        logger.info('\n  limits for median(max-peds) + snrmax * diff_pos/neg = %.3f + %.3f * %.3f/%.3f\n  lim_lo: %.3f\n  lim_hi: %.3f' %\
                    (pos, snrmax, diff_pos, diff_neg, self.max_lim_lo, self.max_lim_hi))

        self.save_max_files()
        return max_peds

    def summary(self):
        logger.info(info_ndarr(self.block, '\nSummary:'))
        #logger.info('%s\nraw data found/selected in %d events' % (80*'_', self.irec+1))
        _nrecs = self.irec + 1
        if self._do_block:
            self.block = self.block[:_nrecs,:]
            logger.info(info_ndarr(self.block, 'block of all records:', last=5))
            bsh = self.block.shape
            self.ssize = bsh[-1]*bsh[-2]

        if 1 in self.features:
            arr1_good_frames = self.feature_01()
            logger.info(info_ndarr(arr1_good_frames, 'arr1_good_frames:'))
            self.bool_good_frames = arr1_good_frames>0
            self.inds_good_frames = np.where(self.bool_good_frames)[0] # array if indices for good frames
            logger.info('%s\n%s' % (info_ndarr(self.inds_good_frames, 'inds_good_frames:', last=0),\
                    str(self.inds_good_frames)))
        else:
            logger.info('Feature 1 for mean intensity of frames is not requested. All frames are used for further processing.')
            self.inds_good_frames = np.arange(_nrecs, dtype=np.uint)
            self.bool_good_frames = np.ones(_nrecs, dtype=bool)

        arr_sta = np.zeros(self.shape_fr, dtype=np.uint64)
        f = '\n  %s\n  %s'
        ss = '\n\nSummary of the bad pixel status evaluation for SNR=%.2f, %s array' % (self.snrmax, self.args.ctype)

        if 2 in self.features:
            arr_mean = self.feature_02()
            logger.info(info_ndarr(arr_mean, 'median over good records:', last=5))
            ss += f % set_pixel_status_bits(arr_sta, arr_mean, title='Feat.2 mean', vmin=0, vmax=self.databits,
                                            snrmax=self.snrmax, bit_lo=1<<0, bit_hi=1<<1)

        if 3 in self.features:
            arr_std = self.feature_03()
            logger.info(info_ndarr(arr_std, 'std over good records:', last=5))
            ss += f % set_pixel_status_bits(arr_sta, arr_std, title='Feat.3 std', vmin=0, vmax=self.databits,
                                            snrmax=self.snrmax, bit_lo=1<<2, bit_hi=1<<3)

        if 4 in self.features:
            arr = self.feature_04()

        if 5 in self.features:
            arr = self.feature_05()

        if 6 in self.features:
            block_res = self.feature_06()

            res_med = np.median(block_res, axis=0)
            res_spr = np.median(np.absolute(block_res - res_med), axis=0)

            logger.info(info_ndarr(block_res, 'block of residuals:', last=20))
            logger.info(info_ndarr(res_med, 'median over frames per-pixel residuals:', last=20))
            logger.info(info_ndarr(res_spr, 'median over frames per-pixel spread of res:', last=20))

            ss += f % set_pixel_status_bits(arr_sta, res_med, title='Feat.6 res_med', vmin=-self.databits, vmax=self.databits,
                                            snrmax=self.snrmax, bit_lo=1<<4, bit_hi=1<<5)
            ss += f % set_pixel_status_bits(arr_sta, res_spr, title='Feat.6 res_spr', vmin=-self.databits, vmax=self.databits,\
                                            snrmax=self.snrmax, bit_lo=1<<6, bit_hi=1<<7)
        if 11 in self.features:
            max_peds = self.feature_11()
            ss += f % set_pixel_status_bits(arr_sta, max_peds, title='Feat.11 max-peds', vmin=self.max_lim_lo, vmax=self.max_lim_hi,\
                                            snrmax=self.snrmax, bit_lo=1<<8, bit_hi=1<<9)

        arr1 = np.ones(self.shape_fr, dtype=np.uint64)
        stus_bad_total = np.select((arr_sta>0,), (arr1,), 0)
        num_bad_pixels = stus_bad_total.sum()

        size = arr_sta.size
        ss += '\n    Any bad status bit: %8d / %d (%6.3f%%) pixels' % (num_bad_pixels, size, 100*num_bad_pixels/size)
        logger.info(ss)

        return arr_sta

    def event(self, raw, evnum):
        self._evnum = evnum # use it for messages only
        logger.debug(info_ndarr(raw, 'event %d raw:' % evnum))
        if raw is None: return self.state

        ndim = raw.ndim
        assert ndim > 1, 'raw.ndim: %d' % ndim
        assert ndim < 4, 'raw.ndim: %d' % ndim
        #if ndim >3: raw = gu.reshape_to_3d(raw)

        _raw = raw if ndim==2 else\
               raw[self.segind,:]

        if self.aslice is not None:
            _raw = _raw[self.aslice]

        if self._do_block:
            if self.block is None: self.init_block(_raw, do_load=False)
            else: self.add_record(_raw)

        #elif self._do_max:
        else:
            _raw = _raw & self.databits
            if self.nda_max is None: self.init_nda_max(_raw)
            else: np.maximum(_raw, self.nda_max, out=self.nda_max)

        return self.state

    def dir_work(self):
        if self._dir_work is None:
            panel_ids  = self.metad['detid'].split('_')
            segind     = self.args.segind
            panel_id   = panel_ids[segind]
            self._dir_work = self.args.repoman.makedir_ctype(panel_id, 'work')
        return self._dir_work

    def fname_arr_max(self):
        return '%s/arr-%s-max.npy' % (self.dir_work(), self.fname)

    def fname_arr_max_peds(self):
        return '%s/arr-%s-max-peds.npy' % (self.dir_work(), self.fname)

    def fname_arr_blk(self):
        return '%s/arr-%s-blk.npy' % (self.dir_work(), self.fname)

    def save_max_files(self):
        fname = self.fname_arr_max()
        if os.path.exists(fname):
            logger.info('exists: %s' % fname)
        else:
            np.save(fname, self.nda_max)
            logger.info('saved: %s' % fname)
        fname = self.fname_arr_max_peds()
        if os.path.exists(fname):
            logger.info('exists: %s' % fname)
        else:
            np.save(fname, self.max_peds)
            logger.info('saved: %s' % fname)

    def shaped_as_frame(self, shfr, msg=''):
        cond = shfr == self.shape_fr
        if not cond:
            logger.warning('%s array from file has shape: %s INCONSISTENT with current raw[slice] shape: %s - IGNORE FILE'%\
                           (msg, str(shfr), str(self.shape_fr)))
        return cond

    def init_nda_max(self, raw):
        fname = self.fname_arr_max()
        self.shape_fr = tuple(raw.shape)
        if os.path.exists(fname):
            logger.info('load max aray from file: %s' % fname)
            _nda_max = np.load(fname)
            if not self.shaped_as_frame(tuple(_nda_max.shape), msg='nda_max'):
                return
            self.nda_max = _nda_max
            self.state = 1
        else:
            self.nda_max = np.array(raw)

    def load_block_file(self):
        if 6 in self.features: return
        fname = self.fname_arr_blk()
        if fname is None: return
        if not os.path.exists(fname): return
        _block = np.load(fname)
        if not self.shaped_as_frame(tuple(_block.shape)[-2:], msg='block'):
            return
        self.block = _block
        logger.warning('DATA BLOCK IS LOADED FROM FILE: %s' % fname)
        logger.info(info_ndarr(self.block, 'loaded data block'))
        self.irec = self.block.shape[0] - 1
        #self.state = 1

    def save_data_file(self):
        fname = self.fname_arr_blk()
        s = ''
        if fname is None:
           s = 'file name for data block is not defined\n    DO NOT SAVE'
        elif os.path.exists(fname):
           s = 'EXISTING FILE %s\n    DO NOT SAVE' % fname
        elif self.block is None:
           s = 'data block is None\n    DO NOT SAVE'
        else:
           np.save(fname, self.block)
           s = info_ndarr(self.block, 'data block') + '\n    saved in %s' % fname
        logger.info(s)

def fname_part(args, metad):
    detname = args.detname.replace(':','-').replace('.','-')
    feats = args.features.strip(',').replace(',','-')
    part = '%s-r%04d-%s-seg%s-evs%06d-feat-%s' % (metad['expname'], metad['runnum'],\
             detname, args.segind, args.events, feats)
    if not '11' in feats:
        part += '-recs%04d' % args.nrecs
        if args.slice is not None:
            #sslice = str(args.slice).replace(':','-').replace(',','_').strip('-')
            part += '-slice'   # % sslice
    return part

def event_loop(parser):

    args = parser.parse_args() # namespace # kwargs = vars(args) # dict
    defs = parser.parse_args([])
    logger.debug('Arguments: %s\n' % str(args))

    dskwargs = args.dskwargs  # dsname 'exp=xpplw3319:run=293'
    detname  = args.detname   # source
    events   = args.events
    evskip   = args.evskip
    steps    = args.steps
    stskip   = args.stskip
    evcode   = args.evcode
    segind   = args.segind
    logmode  = args.logmode
    if not ',' in args.features: args.features += ','  # to evaluate it as a tuple
    features= eval('(%s)' % args.features)
    aslice  = args.aslice = None if args.slice is None else\
                            eval('np.s_[%s]' % args.slice)
    args.ctype  = args.ctype if args.ctype != defs.ctype else\
                  'status_light' if  6 in features else\
                  'status_max'   if 11 in features else\
                  'status_dark'

    repoman = args.repoman = rm.init_repoman_and_logger(args, parser)

    t0_sec = time()

    ds  = DataSource(dskwargs)
    det = Detector(detname)
    #cd  = Detector('ControlData')

    ecm = False
    if evcode is not None:
        from Detector.EventCodeManager import EventCodeManager
        ecm = EventCodeManager(evcode, verbos=0)
        logger.info('requested event-code list %s' % str(ecm.event_code_list()))

    nrun_tot  = 0
    nstep_tot = 0
    nevt_tot  = 0
    metad = None
    kwa_dpo = {}
    dpo = None

    for orun in ds.runs():
      nrun_tot += 1
      if metad is None:
         metad = metadata(ds, orun, det)
         logger.info('metadata from DataSource and Detector:\n' + ' '.join(['\n%s: %s'%(k.ljust(10), info_ndarr(v)\
                     if isinstance(v, np.ndarray) else str(v)) for k, v in metad.items()]))
         if args.logmode == 'DEBUG':
            peds = metad['pedestals']
            rms  = metad['rms']
            fname_peds = os.path.join(self.dir_work(), 'cc_peds.npy')
            fname_rms = os.path.join(self.dir_work(), 'cc_rms.npy')
            np.save(fname_peds, peds)
            np.save(fname_rms, rms)
            logger.info('saved files for debugging:\n  %s\n  %s' % (fname_peds, fname_rms))

         dpo = DataProc(args, metad, **kwa_dpo)

      logger.info('==== run %s' % str(orun.run()))

      break_runs = False

      for nstep_run, step in enumerate(orun.steps()):
        nstep_tot += 1
        logger.info('  == step %02d ==' % nstep_tot)

        if steps is not None and nstep_tot >= steps:
            logger.info('nstep_tot:%d >= number of steps:%d - break' % (nstep_tot, steps))
            break

        elif stskip is not None and nstep_tot < stskip:
            logger.info('nstep:%d < number of steps to skip:%d - continue' % (nstep_tot, stskip))
            continue

        break_steps = False

        for i, evt in enumerate(step.events()):
            nevt = i + 1
            nevt_tot += 1
            raw = det.raw(evt)

            if raw is None: #skip empty frames
                logger.info('Ev total:%05d in step:%05d rec:%05d raw=None' % (nevt_tot, nevt, dpo.irec))
                continue

            if evskip is not None and nevt < evskip:
                logger.info('Ev total:%05d in step:%05d < --evskip:%d - continue' % (nevt_tot, nevt, evskip))
                continue

            if selected_number(nevt):
                logger.info(info_ndarr(raw, 'Ev total:%05d in step:%05d rec:%05d raw' % (nevt_tot, nevt, dpo.irec)))

            if ecm and not ecm.select(evt):
                logger.debug('==== Ev total:%05d in step:%d is skipped due to event code selection - continue' % (nevt_tot, nevt))
                continue

            resp = dpo.event(raw, nevt_tot)

            if resp == 1:
                break_steps = True
                break_runs = True
                logger.info('BREAK EVENTS')
                break

            if events is not None and nevt_tot >= events:
                logger.info('BREAK EVENTS nevt_tot:%d >= --events:%d' % (nevt_tot, events))
                break_steps = True
                break_runs = True
                break

        if break_steps:
            logger.info('BREAK STEPS')
            break

      if break_runs:
          logger.info('BREAK RUNS')
          break

    arr_status = dpo.summary()
    save_constants(arr_status, args, metad)

    if False: dpo.save_data_file()

    logger.info('Consumed time %.3f sec' % (time()-t0_sec))
    repoman.logfile_save()


def save_constants(arr_status, args, dmd):
    fmt        = '%d'
    runnum     = dmd['runnum']
    itype      = dmd['dettype']
    typename   = dmd['typename']
    expname    = dmd['expname']
    ts_run     = dmd['ts_run']
    panel_ids  = dmd['detid'].split('_')
    detname    = dmd['detname']
    ctype      = args.ctype # 'status_data'
    dirrepo    = args.dirrepo
    filemode   = args.filemode
    group      = args.group
    segind     = args.segind
    gmode      = args.gmode
    panel_id   = panel_ids[segind]
    repoman    = args.repoman
    features   = eval('(%s)' % args.features)  # list of int
    #detname    = args.detname
    #dirdettype = repoman.makedir_dettype(dettype=typename)
    fname_aliases = repoman.fname_aliases(dettype=typename) # , fname='.aliases_%s.txt' % typename)
    dirpanel   = repoman.dir_panel(panel_id)
    fname_prefix, panel_alias = uc.file_name_prefix(typename, panel_id, ts_run, expname, runnum, fname_aliases, **{'detname':str(detname)})
    logger.debug('panel index: %02d alias: %s dir: %s\n  fname_prefix: %s' % (segind, panel_alias, dirpanel, fname_prefix))
    dirname = repoman.makedir_ctype(panel_id, ctype)
    #_ctype = ctype+'_f11' if 11 in features else ctype
    fname = '%s/%s_%s' % (dirname, fname_prefix, ctype)
    if gmode is not None: fname += '_%s' % gmode
    if args.slice is not None: fname += '_slice'
    fname += '.dat'
    uc.save_2darray_in_textfile(arr_status, fname, filemode, fmt, umask=0o0, group=group)

class Empty:
    """reserved for name sapace"""
    pass

def save_constants_in_repository(arr, **kwa):
    """User's interface method to save any constants in repository.

    PARAMETERS of **kwa
    -------------------

    dskwargs (str) - parameter for DataSource, e.g. 'exp=xpplw3319:run=293'
    detname (str) - detector name
    ctype (str) - calibration constants type, ex. pedestals, gain, offset, status_user, etc.
    dirrepo (str) - root level of repository
    segind (int) - segment index for multipanel detectors
    gmodes (tuple) - gain mode names, ex. for epix10ka ('FH', 'FM', 'FL', 'AHL-H','AML-H', 'AHL-L','AML-L')
    slice (str) - numpy style of array slice for debugging
    """
    logger.debug('kwargs: %s' % str(kwa))
    from Detector.dir_root import DIR_REPO_STATUS, DIR_LOG_AT_START # os, DIR_ROOT
    #import Detector.UtilsEpix10ka as ue

    args = Empty()
    args.dskwargs = kwa.get('dskwargs', None) # 'exp=xpplw3319:run=293'
    args.detname  = kwa.get('detname', None) # 'XppGon.0:Epix100a.3' or its alias 'epix_alc3'
    args.ctype    = kwa.get('ctype', 'status_user')
    args.dirrepo  = kwa.get('dirrepo', DIR_REPO_STATUS)
    args.filemode = kwa.get('filemode', 0o664)
    args.dirmode  = kwa.get('filemode', 0o2775)
    args.group    = kwa.get('group', 'ps-users')
    args.segind   = kwa.get('segind', 0)
    args.gmodes   = kwa.get('gmodes', None)
    args.slice    = kwa.get('slice', None)
    args.repoman = rm.RepoManager(dirrepo=args.dirrepo, dir_log_at_start=DIR_LOG_AT_START,\
                                  dirmode=args.dirmode, filemode=args.filemode, group=args.group)
    args.gmode    = None

    ndim = arr.ndim
    shape = arr.shape

    assert isinstance(arr, np.ndarray)
    assert ndim >= 2
    assert ndim <= 4
    assert args.dskwargs is not None
    assert args.detname is not None

    ds  = DataSource(args.dskwargs)
    det = Detector(args.detname)
    args.detname = det.name
    orun = next(ds.runs())
    metad = metadata(ds, orun, det)
    logger.info(' '.join(['\n%s: %s'%(k.ljust(10), str(v)) for k, v in metad.items()]))
    panel_ids = metad['detid'].split('_')
    nsegs = len(panel_ids)

    if ndim == 2:
        save_constants(arr, args, metad)

    elif ndim == 3:
        assert arr.shape[0] == nsegs, 'number of segments in the array shape %s should be the same as in metadata %s' % (str(shape), nsegs)
        for i in range(nsegs):
            args.segind = i
            save_constants(arr[i,:], args, metad)

    elif ndim == 4:
        assert isinstance(args.gmodes, tuple),\
            'gain mode names should be defined in tuple, gmodes=%s' % str(args.gmodes)
        assert arr.shape[0] == len(args.gmodes),\
            'number of gain modes in the array shape %s should be the same as in tuple gmodes %s' % (str(shape), str(args.gmodes))
        assert arr.shape[1] == nsegs,\
            'number of segments in the array shape %s should be the same as in metadata %s' % (str(shape), nsegs)
        for n, gm in enumerate(args.gmodes):
          for i in range(nsegs):
            args.segind = i
            args.gmode = gm
            save_constants(arr[n,i,:], args, metad)


det_pixel_status = event_loop

# EOF
