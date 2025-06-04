"""
:py:class:`UtilsPixelStatus` psana2 methods to evaluate pixel status in dark and light data.
============================================================================================

Usage::

    import psana.detector.UtilsPixelStatus as ups

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@date 2025-05-20
@author Mikhail Dubrovin
"""

import os
from psana.detector.UtilsEventLoop import *
import numpy as np
#import psana.detector.UtilsCalib as uc
from psana.detector.NDArrUtils import info_ndarr

def tmp_filename(fname=None, suffix='_EventLoopStatus.npy'):
   """returns file name in
      /lscratch/<username>/tmp/fname   if fname is not None or
      /lscratch/<username>/tmp/<random-str>suffix
   """
   import tempfile
   tmp_file = tempfile.NamedTemporaryFile(mode='r+b',suffix=suffix)
   return tmp_file.name if fname is None else\
          os.path.join(os.path.dirname(tmp_file.name), fname)


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

    logger.info('XXX vmin %.3f: vmax: %.3f' % (vmin, vmax))

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






def feature_01(block, databits=0x3FFF, snrmax=8):
    
    logger.info("""Feature 1: mean intensity of frames in good range""")
    #block = block & databits
    #block = np.bitwise_and(block, databits)
    intensity_mean = np.sum(block, axis=(-2,-1)) / block.size
    logger.info(info_ndarr(intensity_mean, '\n  per-record intensity MEAN IN FRAME:', last=20))

    intensity_med = np.median(block, axis=(-2,-1))
    logger.info(info_ndarr(intensity_med, '\n  per-record intensity MEDIAN IN FRAME:', last=20))
    arr1_lo, arr1_hi, s_lo, s_hi = evaluate_pixel_status(intensity_med, title='Feat.1: intensity_med',\
                                         vmin=0, vmax=databits, snrmax=snrmax)
    arr0 = np.zeros_like(arr1_lo, dtype=np.uint64)  # dtype=bool
    arr1_good_frames = np.select((arr1_lo>0, arr1_hi>0), (arr0, arr0), 1)
    logger.info('Total number of good events: %d' % arr1_good_frames.sum())

    return arr1_good_frames




class EventLoopStatus(EventLoop):
    msgels='EventLoopStatus'

    def __init__(self, parser):
        EventLoop.__init__(self, parser)

    def init_event_loop(self):
        message(msg=self.msgels, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print('init_event_loop - dskwargs: %s detname: %s' % (str(self.dskwargs), self.detname))
        kwa = self.kwa
        #nrecs   = kwa.get('nrecs', 10)
        #self.nrecs  = kwa.get('nrecs',1000)
        #kwa['init_event_loop'] = 'OK'
        self.dbl = None
        self.status = None
        self.dic_consts_tot = {} # {<gain_mode>:{<ctype>:nda3d_shape:(4, 192, 384)}}
        self.kwa_depl = {}

    def begin_run(self):
        #message(msg=self.msgels, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print('=== begin_run expname: %s runnum: %s' % (self.expname, str(self.runnum)))

    def end_run(self):
        #message(msg=self.msgels, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print('=== end_run expname: %s runnum: %s' % (self.expname, str(self.runnum)))

    def fname_data_block(self, ext='.npz'):
        fname = '%s-data_block-%s-r%04d-%s' % (self.msgels, self.expname, self.runnum, self.detname)
        fname += '-seg%s' % ('ALL' if self.segind is None else ('%03d' % self.segind))
        fname += '-step%02d' % self.istep
        fname += '-nrecs%04d' % self.nrecs
        return tmp_filename(fname+ext)

    def begin_step(self):
        #message(msg=self.msgels, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print('begin_step istep/nevtot: %d/%s' % (self.istep, str(self.metadic)))

        dbl = self.dbl
        if dbl is None:
            odet = self.odet
            kwa = self.kwa
            kwa['rms_hi'] = odet.raw._data_bit_mask - 10
            kwa['int_hi'] = odet.raw._data_bit_mask - 10
            kwa.setdefault('nrecs',10)
            kwa.setdefault('datbits', 0xffff) # data bits 0xffff - 16-bit mask for detector without gain bit/s

            dbl = self.dbl = uc.DataBlock(**kwa)
            dbl.runnum = self.runnum
            dbl.exp = self.expname
            dbl.ts_run, dbl.ts_now = self.ts_run, self.ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FOR

            self.fname_block = self.fname_data_block()
            self.exists_fdb = os.path.exists(self.fname_block)
            print('XXX tmp file: %s   %s' % (self.fname_block, {True:'EXISTS', False:'DOES NOT EXIST'}[self.exists_fdb]))

    def end_step(self):
        #message(msg=self.msgels, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        print(self.dbl.info_data_block(cmt='XXX berofe saving data_block'))
        dbl = self.dbl
        dbl.save(self.fname_block)
        del(dbl)
        dbl=None
        print('==== End of step %1d ====\n' % self.istep)

    def proc_event(self, msgmaxnum=5):
        #print('proc_event ievt/nevtot: %d/%d' % (self.ievt, self.nevtot))
        raw = self.odet.raw.raw(self.evt)
        if self.segind is not None: raw = raw[self.segind,:]
        if self.aslice is not None: raw = raw[self.aslice]
        is_full = self.dbl.event(raw, self.ievt)
        self.status = 2 if is_full else 1

    def summary(self):
        message(msg=self.msgels, metname=sys._getframe().f_code.co_name, logmethod=logger.info)
        gainmodes = [k for k in self.dic_consts_tot.keys()]
        logger.info('constants'\
                   +'\n  created  for gain modes: %s' % str(gainmodes)\
                   +'\n  expected for gain modes: %s' % str(self.odet.raw._gain_modes))

        ctypes = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min')
        gmodes = self.odet.raw._gain_modes #  or gainmodes
        kwa_depl = self.kwa_depl
        kwa_depl['shape_as_daq'] = None if self.odet.raw is None else self.odet.raw._shape_as_daq()
        kwa_depl['exp']          = self.expname
        kwa_depl['det']          = self.detname
        kwa_depl['run_orig']     = self.runnum

        #deploy_constants(ctypes, gmodes, **kwa_depl)



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


def test_features(fname='fname.npz'):
    dbl = uc.DataBlock()
    dbl.load(fname=fname)
    print(dbl.info_data_block(cmt=''))
    good_frames = feature_01(dbl.block, databits=0x3FFF, snrmax=8)
    print('XXX good_frames', good_frames)


if __name__ == "__main__":

  def USAGE():
    import inspect
    return '\n  %s <TNAME>\n' % sys.argv[0].split('/')[-1]\
          +'\n  test dataset: datinfo -k exp=mfxdaq23,run=7,dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc -d jungfrau'
    #+ '\n'.join([s for s in inspect.getsource(selector).split('\n') if "TNAME in" in s])
    #   datinfo -k exp=mfx101332224,run=204 -d jungfrau

  def argument_parser():
    from argparse import ArgumentParser
    d_tname    = '0'
    d_dirrepo  = './work1' # DIR_REPO_JUNGFRAU
    #d_dskwargs = 'exp=mfxdaq23,run=7' # dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc
    d_dskwargs = 'exp=mfx101332224,run=204' # dir=/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc
    d_detname  = 'jungfrau'
    d_loglevel = 'INFO' # 'DEBUG'
    d_subtest  = None
    d_segind   = 3 # None
    d_nrecs    = 100
    d_evskip   = 0       # number of events to skip in the beginning of each step
    d_events   = 1000000 # last event number in the step to process

    h_dirrepo  = 'non-default repository of calibration results, default = %s' % d_dirrepo
    h_tname    = 'test name, usually numeric number, default = %s' % d_tname
    h_dskwargs = '(str) dataset kwargs for DataSource(**kwargs), default = %s' % d_dskwargs
    h_detname  = 'detector name, default = %s' % d_detname
    h_subtest  = '(str) subtest name, default = %s' % d_subtest
    h_loglevel = 'logging level, one of %s, default = %s' % (', '.join(tuple(logging._nameToLevel.keys())), d_loglevel)
    h_segind   = 'segment index in det.raw.raw array to process, default = %s' % str(d_segind)
    h_nrecs    = 'number of records to collect in data block, default = %d' % d_nrecs
    h_evskip    = 'number of events to skip in the beginning of each step, default = %s' % str(d_evskip)
    h_events  = 'number of events to process from the beginning of each step, default = %s' % str(d_events)

    parser = ArgumentParser(description='%s is a bunch of tests for annual issues' % SCRNAME, usage=USAGE())
    #parser.add_argument('tname',            default=d_tname,   type=str, help=h_tname)
    parser.add_argument('-o', '--dirrepo',  default=d_dirrepo,  type=str, help=h_dirrepo)
    parser.add_argument('-k', '--dskwargs', default=d_dskwargs, type=str, help=h_dskwargs)
    parser.add_argument('-d', '--detname',  default=d_detname,  type=str, help=h_detname)
    parser.add_argument('-L', '--loglevel', default=d_loglevel, type=str, help=h_loglevel)
    parser.add_argument('-s', '--subtest',  default=d_subtest,  type=str, help=h_subtest)
    parser.add_argument('-I', '--segind',   default=d_segind,   type=int, help=h_segind)
    parser.add_argument('-n', '--nrecs',    default=d_nrecs,    type=int, help=h_nrecs)
    parser.add_argument('--evskip',         default=d_evskip,   type=int, help=h_evskip)
    parser.add_argument('--events',         default=d_events,   type=int, help=h_events)
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    #tname = args.tname
    print(80*'_')

#    kwa = vars(args)
    STRLOGLEV = args.loglevel
    INTLOGLEV = logging._nameToLevel[STRLOGLEV]
#    print('XXX tname:', tname)

#    print('XXX tmp_filename:', tmp_filename(fname='data-block.npy', suffix='EventLoopStatus.npy'))
#    sys.exit('TEST EXIT')fname

#    fname = '/lscratch/dubrovin/tmp/EventLoopStatus-data_block-mfxdaq23-r0007-jungfrau-seg003-step00-nrecs0100.npz'
    fname = '/lscratch/dubrovin/tmp/EventLoopStatus-data_block-mfx101332224-r0204-jungfrau-seg003-step00-nrecs0100.npz'
    
    if os.path.exists(fname):
        logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=INTLOGLEV)
        test_features(fname=fname)

    else:
        #evl = EventLoop(parser)
        evl = EventLoopStatus(parser)
        evl.event_loop()

    sys.exit('End of %s' % SCRNAME)

# EOF
