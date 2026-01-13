
"""
:py:class:`UtilsJungfrauCalib`
==============================

Usage::
    from psana.detector.UtilsJungfrauCalib import *

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2021-04-05 by Mikhail Dubrovin
2025-02-20 - adopted to lcls2
"""

import logging
logger = logging.getLogger(__name__)

from time import time
import os
import sys
import psana
import numpy as np
from time import time #, localtime, strftime
import json

import psana.detector.dir_root as dr
import psana.detector.UtilsCalib as uc
from psana.detector.RepoManager import init_repoman_and_logger, fname_prefix, fname_prefix_merge
import psana.detector.utils_psana as ups # seconds, data_source_kwargs#
from psana.detector.NDArrUtils import info_ndarr, save_ndarray_in_textfile # import divide_protected
import psana.detector.Utils as uts # info_dict
import psana.pscalib.calib.CalibConstants as cc
from psana.detector.UtilsCalibRepo import save_constants_in_repository
from psana.pscalib.calib.MDBWebUtils import add_data_and_two_docs, add_data_and_doc_to_detdb_extended

SCRNAME = os.path.basename(sys.argv[0])
MAX_DETNAME_SIZE = 20
NUMBER_OF_GAIN_MODES = 3

CTYPES_DARK = ('pedestals', 'pixel_rms', 'pixel_max', 'pixel_min', 'pixel_status')
CTYPES_DEPL = CTYPES_DARK + ('pixel_gain', 'pixel_offset', 'status_extra')

dic_calib_char_to_name = cc.dic_calib_char_to_name # {'p':'pedestals', 'r':'pixel_rms', 's':'pixel_status',...}
# "p"-pedestals, "r"-rms, "s"-status, "g" or "c" - gain or charge-injection gain,

#DIC_GAIN_MODE = {'DYNAMIC':         0,
#                 'FORCE_SWITCH_G1': 1,
#                 'FORCE_SWITCH_G2': 2}

DIC_GAIN_MODE = {'g0': 0,
                 'g1': 1,
                 'g2': 2}

DIC_IND_TO_GAIN_MODE = {v:k for k,v in DIC_GAIN_MODE.items()} # or uts.inverse_dict(DIC_GAIN_MODE)

M14 = 0x3fff # 16383, 14-bit mask
#FNAME_PANEL_ID_ALIASES = '%s/.aliases_jungfrau.txt' % dr.DIR_REPO_JUNGFRAU


#print('\n  DIR_ROOT', dr.DIR_ROOT,\
#      '\n  DIR_REPO', dr.DIR_REPO)


dic_ctype_fmt = uc.dic_ctype_fmt

class DarkProcJungfrau(uc.DarkProc):
    """dark data accumulation and processing for Jungfrau.
       Extends DarkProc to account for bad gain mode switch state in self.bad_switch array and pixel_status.
    """
    def __init__(self, **kwa):
        kwa.setdefault('datbits', M14)
        uc.DarkProc.__init__(self, **kwa)
        self.modes = ['g0-00', 'g1-01', 'g2-11', 'BAD-10']


    def init_proc(self):
        uc.DarkProc.init_proc(self)
        shape_raw = self.arr_med.shape
        self.bad_switch = np.zeros(shape_raw, dtype=np.uint8)


    def add_event(self, raw, irec):
        uc.DarkProc.add_event(self, raw, irec)
        self.add_statistics_bad_gain_switch(raw, irec)


    def add_statistics_bad_gain_switch(self, raw, irec, evgap=10):
        igm    = self.gmindex
        gmname = self.gmname

        t0_sec = time()
        gbits = raw>>14 # 00/01/11/01 - gain bits for mode 0,1,2,bad
        fg0, fg1, fg2, fgx = gbits==0, gbits==1, gbits==3, gbits==2

        if irec%evgap==0:
           dt_sec = time()-t0_sec
           sums = [fg0.sum(), fg1.sum(), fg2.sum(), fgx.sum()]
           logger.debug('Rec: %4d found pixels %s gain definition time: %.6f sec igm=%d:%s'%\
                    (irec,' '.join(['%s:%d' % (self.modes[i], sums[i]) for i in range(4)]), dt_sec, igm, gmname))

        np.logical_or(self.bad_switch, fgx, self.bad_switch)


    def summary(self):
        t0_sec = time()
        uc.DarkProc.summary(self)
        logger.info('\n  status 64: %8d pixel with bad gain mode switch' % self.bad_switch.sum())
        self.arr_sta += self.bad_switch*64 # add bad gain mode switch to pixel_status
        self.arr_msk = np.select((self.arr_sta>0,), (self.arr0,), 1) #re-evaluate mask
        self.block = None
        self.irec = -1
        logger.info('summary consumes %.3f sec' % (time()-t0_sec))


    def info_results(self, cmt='DarkProc results'):
        return uc.DarkProc.info_results(self, cmt)\
         +info_ndarr(self.bad_switch, '\n  badswch')\


    def plot_images(self, titpref=''):
        uc.DarkProc.plot_images(self, titpref)
        plotim = self.plotim
        if plotim &2048: plot_image(self.bad_switch, tit=titpref + 'bad gain mode switch')


def info_gain_modes(gm):
    s = 'gm.names:'
    for name in gm.names.items(): s += '\n    %s' % str(name)
    s += '\ngm.values:'
    for value in gm.values.items(): s += '\n    %s' % str(value)
    return s


def selected_record(i, events):
    return i<5\
       or (i<50 and not i%10)\
       or (i<200 and not i%20)\
       or not i%100\
       or i>events-5


def print_uniqueid(uniqueid, segind):
    s = 'panel_ids:'
    for i,pname in enumerate(uniqueid.split('_')):
        s += '\n  %02d panel id %s' % (i, pname)
        if i == segind: s += '  <-- selected for processing'
    logger.info(s)


def get_jungfrau_gain_mode_object(odet):
    """Returns gain mode object, usage: gmo=..., gmo.name, gmo.names.items(), gm.values.items(), etc.
    """
    dcfg = odet.raw._config_object()


def open_DataSource(**kwargs):
    #ds = psana.DataSource(exp='mfxdaq23', run=7, dir='/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc')
    dskwargs = ups.data_source_kwargs(**kwargs)
    logger.info('DataSource dskwargs: %s' % (dskwargs))
    try: ds = psana.DataSource(**dskwargs)
    except Exception as err:
        logger.error('DataSource(**dskwargs) does not work for **dskwargs: %s\n    %s' % (dskwargs, err))
        sys.exit('EXIT - requested DataSource does not exist or is not accessible.')

    logger.debug('ds.runnum_list = %s' % str(ds.runnum_list))
    logger.debug('ds.detectors = %s' % str(ds.detectors))
    xtc_files = getattr(ds, 'xtc_files', None)
    logger.info('ds.xtc_files:\n  %s' % ('None' if xtc_files is None else '\n  '.join(ds.xtc_files)))
    return ds, dskwargs


def jungfrau_dark_proc(parser):
    """jungfrau dark data processing for single (of 3) gain mode.
    """
    t0_sec = time()
    tdt = t0_sec

    args = parser.parse_args() # namespae of parameters
    kwargs = vars(args) # dict of parameters

    repoman = init_repoman_and_logger(parser=parser, **kwargs)
    kwargs['repoman'] = repoman

    detname = args.detname
    evskip  = args.evskip
    events  = args.events
    stepnum = args.stepnum
    stepmax = args.stepmax
    evcode  = args.evcode
    segind  = args.segind
    igmode  = args.igmode
    dirrepo = args.dirrepo

    dirmode  = kwargs.get('dirmode',  0o2775)
    filemode = kwargs.get('filemode', 0o664)
    group    = kwargs.get('group', 'ps-users')

    #ecm = False
    #if evcode is not None:
    #    from Detnameector.EventCodeManager import EventCodeManager
    #    ecm = EventCodeManager(evcode, verbos=0)
    #    logger.info('requested event-code list %s' % str(ecm.event_code_list()))

    s = 'DIC_GAIN_MODE {<name> : <number>}'
    for k,v in DIC_GAIN_MODE.items(): s += '\n%16s: %d' % (k,v)
    logger.info(s)

    ds, dskwargs = open_DataSource(**kwargs)

    dpo = None
    igm0 = None
    nevtot = 0
    nevsel = 0
    nsteptot = 0
    ss = ''
    uniqueid = None
    dettype = None
    step_docstring = None
    terminate_runs = False

    for irun, orun in enumerate(ds.runs()):
        logger.info('\n%s Run %d %s' % (20*'=', orun.runnum, 20*'='))

        trun_sec = ups.seconds(orun.timestamp) # 1607569818.532117 sec
        ts_run, ts_now = ups.tstamps_run_and_now(trun_sec) #, fmt=uc.TSTAMP_FORMAT)

        odet = orun.Detector(detname, **kwargs)
        if dettype is None:
            dettype = odet.raw._dettype
            repoman.set_dettype(dettype)
            uniqueid = odet.raw._uniqueid
            logger.info('det.raw._uniqueid.split: %s' % str('\n'.join(uniqueid.split('_'))))

        logger.info('created %s detector object of type %s' % (detname, dettype))
        logger.info(ups.info_detector(odet, cmt='detector info:\n      ', sep='\n      '))

        try:
          step_docstring = orun.Detector('step_docstring')
        except:
          step_docstring = None

        terminate_steps = False
        nevrun = 0
        nnones = 0
        for istep, step in enumerate(orun.steps()):
            nsteptot += 1

            metadic = json.loads(step_docstring(step)) if step_docstring is not None else {}
            logger.info((100*'=') + '\n step_docstring ' +
                  'is None' if step_docstring is None else\
                  str(metadic))

            if stepmax is not None and nsteptot>stepmax:
                logger.info('==== Step:%02d loop is terminated, --stepmax=%d' % (nsteptot, stepmax))
                terminate_runs = True
                break

            if stepnum is not None:
                # process calibcycle stepnum ONLY if stepnum is specified
                if istep < stepnum:
                    logger.info('Skip step %d < --stepnum = %d' % (istep, stepnum))
                    continue
                elif istep > stepnum:
                    logger.info('Break further processing due to step %d > --stepnum = %d' % (istep, stepnum))
                    terminate_runs = True
                    break

            ############################### TBD

            igm = igmode if igmode is not None else\
                  metadic['gainMode'] if step_docstring is not None\
                  else istep
            #gmo = get_jungfrau_gain_mode_object(odet)
            #igm = DIC_GAIN_MODE[gmo.name]
            #igm = igmode if igmode is not None else istep
            gmname = DIC_IND_TO_GAIN_MODE.get(igm, None)
            kwargs['gainmode'] = gmname
            logger.info('gain mode: %s igm: %d' % (gmname, igm))
            ###############################

            if dpo is None:
               kwargs['dettype'] = dettype
               dpo = DarkProcJungfrau(**kwargs)
               dpo.runnum = orun.runnum
               dpo.exp = orun.expt
               dpo.ts_run, dpo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)
               dpo.detid = uniqueid
               dpo.gmindex = igm
               dpo.gmname = gmname

               igm0 = igm

            if igm != igm0:
               logger.warning('event for wrong gain mode index %d, expected %d' % (igm, igm0))

            if istep==0:
                logger.info('TBD gain mode info from jungfrau configuration') #\n%s' % info_gain_modes(gmo))

            logger.info('%s\n== begin step %d gain mode "%s" index %d' % (120*'-',istep, gmname, igm))

            for ievt, evt in enumerate(step.events()):

                nevrun += 1
                nevtot += 1

                if ievt<evskip:
                    s = 'skip event %d < --evskip=%d' % (ievt, evskip)
                    #print(s, end='\r')
                    if (selected_record(ievt+1, events) and ievt<evskip-1)\
                    or ievt==evskip-1: logger.info(s)
                    continue

                #if ievt>=evstep:
                #    print()
                #    logger.info('break at ievt %d == --evstep=%d' % (ievt, evstep))
                #    break

                if nevtot>=events:
                    print()
                    logger.info('break at nevtot %d == --events=%d' % (nevtot, events))
                    terminate_steps = True
                    terminate_runs = True
                    break

                #if ecm:
                #  if not ecm.select(evt):
                #    print('    skip event %d due to --evcode=%s selected %d ' % (ievt, evcode, nevsel), end='\r')
                #    continue

                raw = odet.raw.raw(evt)
                if raw is None:
                    logger.debug('det.raw.raw(evt) is None in event %d' % ievt)
                    nnones =+ 1
                    continue

                raw = (raw if segind is None else raw[segind,:]) # NO & M14 here

                nevsel += 1

                tsec = time()
                dt   = tsec - tdt
                tdt  = tsec
                if selected_record(ievt+1, events):
                    #print()
                    ss = 'run[%d] %d  step %d  events total/run/step/selected/none: %4d/%4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                         (irun, orun.runnum, istep, nevtot, nevrun, ievt+1, nevsel, nnones, time()-t0_sec, dt)
                    #if ecm:
                    #   print()
                    #   ss += ' event codes: %s' % str(ecm.event_codes(evt))
                    logger.info(ss)
                #else: print(ss, end='\r')

                if dpo is not None:
                    #print(info_ndarr(raw,'XXX raw'))
                    status = dpo.event(raw,ievt)
                    if status == 2:
                        logger.info('requested statistics --nrecs=%d is collected - terminate loops' % args.nrecs)
                        #if ecm:
                        #    terminate_runs = True
                        #    terminate_steps = True
                        break # evt loop

                # End of event-loop

            print()
            ss = 'run[%d] %d  end of step %d  events total/run/step/selected: %4d/%4d/%4d/%4d'%\
                 (irun, orun.runnum, istep, nevtot, nevrun, ievt+1, nevsel)
            logger.info(ss)


            #if ecm:
            #    logger.info('continue to accumulate statistics, due to --evcode=%s' % evcode)
            #else:
            #    logger.info('reset statistics for next step')

            save_results(dpo, orun, odet, **kwargs)
            dpo=None

            if terminate_steps:
                logger.info('terminate_steps')
                break

            # End of step-loop

        logger.info(ss)
        logger.info('run %d, number of steps processed %d' % (orun.runnum, istep+1))

        #if is_single_run:
        #    logger.info('terminated due to is_single_run:%s' % is_single_run)
        #    break

        if dpo is not None:
            save_results(dpo, orun, odet, **kwargs)
            dpo=None

        if terminate_runs:
            logger.info('terminate_runs')
            break

        # End of run-loop

    logger.info('number of runs processed %d' % (irun+1))
    logger.info('%s\ntotal consumed time = %.3f sec.' % (40*'_', time()-t0_sec))
    repoman.logfile_save()


def save_results(dpo, orun, odet, **kwa):
    logger.info('begin save_results')
    t0_sec = time()
    if dpo is None: return
    dpo.summary()
    dpo.show_plot_results()

    ctypes = CTYPES_DARK # ('pedestals', 'pixel_rms', 'pixel_max', 'pixel_min', 'pixel_status')
    arr_av1, arr_rms, arr_sta = dpo.constants_av1_rms_sta()
    arr_max, arr_min = dpo.constants_max_min()
    consts = arr_av1, arr_rms, arr_max, arr_min, arr_sta
    logger.info('evaluated constants: \n  %s\n  %s\n  %s\n  %s\n  %s' % (
                info_ndarr(arr_av1, 'arr_av1', first=0, last=5),\
                info_ndarr(arr_rms, 'arr_rms', first=0, last=5),\
                info_ndarr(arr_max, 'arr_max', first=0, last=5),\
                info_ndarr(arr_min, 'arr_min', first=0, last=5),\
                info_ndarr(arr_sta, 'arr_sta', first=0, last=5)))
    dic_consts = dict(zip(ctypes, consts))

    kwa.setdefault('max_detname_size', MAX_DETNAME_SIZE)
    kwa_depl = uc.add_metadata_kwargs(orun, odet, **kwa)
    save_constants_in_repository(dic_consts, **kwa_depl)
    del(dpo)
    logger.info('save_results time %.3f sec' % (time()-t0_sec))


def fname_merged_gmodes(dir_ctype, fnprefix, ctype):
    """ <dirname>/jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-pixel_status.txt """
    return '%s/%s-%s.txt' % (dir_ctype, fnprefix, ctype)


def find_file_for_timestamp(dirname, pattern, tstamp, fnext='.data'):
    """fname ejungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-pedestals-Normal.data
    """
    # list of file names in directory, dirname, containing pattern
    logger.debug('\n  dirname: %s\n  pattern: %s\n, tstamp: %s' % (dirname, pattern, tstamp))
    fnames = [name for name in os.listdir(dirname) if os.path.splitext(name)[-1]==fnext and pattern in name]\
             if os.path.exists(dirname) else []

    # list of int tstamps
    # !!! here we assume specific name structure generated by file_name_prefix: jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-
    itstamps = [int(name.split('-',3)[2]) for name in fnames]

    # reverse-sort int timestamps in the list
    itstamps.sort(key=int, reverse=True)
    logger.debug('found tstamps: %s' % str(itstamps))

    # find the nearest to requested timestamp
    for its in itstamps:
        if its <= int(tstamp):
            # find and return the full file name for selected timestamp
            ts = str(its)
            for name in fnames:
                if ts in name:
                     fname = '%s/%s' % (dirname, name)
                     logger.info('  selected %s for %s and %s' % (os.path.basename(fname),pattern,tstamp))
                     return fname

    logger.debug('directory %s\n         DOES NOT CONTAIN file for pattern %s and timestamp <= %s'%\
                 (dirname,pattern,tstamp))
    return None


def merge_jf_panel_gain_ranges(dir_ctype, panel_id, ctype, tstamp, shape, ofname, fmt='%.3f', fac_mode=0o664, errskip=True, group='ps-users'):
    """ work/jungfrauemu/00b1ed0000-0000000000-0000000000-0000000000-0000000000-0000000000-0000000000/pedestals/
        jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-pedestals-Normal.data
        ofname = jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-pedestals.txt
    """
    logger.debug('In merge_panel_gain_ranges for\n  dir_ctype: %s\n  id: %s\n  ctype=%s tstamp=%s shape=%s'%\
                 (dir_ctype, panel_id, ctype, str(tstamp), str(shape)))

    logger.info('merge gain ranges in: %s tstamp: %s shape: %s'%\
                 (dir_ctype, str(tstamp), str(shape)))

    # define default constants to substitute missing
    nda_def = np.ones(shape, dtype=np.float32) if ctype in ('gain', 'rms', 'dark_max') else\
              np.zeros(shape, dtype=np.float32) # 'pedestals', 'status', 'dark_min'
    if ctype == 'dark_max': nda_def *= M14

    lstnda = []
    for gmind, gmname in DIC_IND_TO_GAIN_MODE.items():
        pattern = '%s-%s' % (ctype, gmname)
        fname = find_file_for_timestamp(dir_ctype, pattern, tstamp)
        nda = None
        if fname is not None:
            nda = np.loadtxt(fname, dtype=np.float32)
        check_exists(fname, errskip, 'panel constants "%s" for gm:%d and tstamp %s NOT FOUND %s' % (ctype, gmind, str(tstamp), fname))
        logger.debug('merge gm:%d %s' % (gmind, fname))
        lstnda.append(nda if nda is not None else nda_def)

    nda = np.stack(tuple(lstnda))
    logger.debug('merge_panel_gain_ranges - merged with shape %s' % str(nda.shape))

    #logger.info(info_ndarr(nda, 'nda before reshaping %s'%ctype))
    nda.shape = (3, 1,) + shape # (3, 1, 512, 1024)
    logger.debug(info_ndarr(nda, 'merged %s'%ctype))
    uc.save_ndarray_in_textfile(nda, ofname, fac_mode, fmt, umask=0o0, group=group)

    nda.shape = (3, 1,) + shape # (3, 1, 512, 1024) # because save_ndarray_in_textfile changes shape

    return nda


def check_exists(path, errskip, msg):
    if path is None or (not os.path.exists(path)):
        if errskip: logger.warning(msg)
        else:
            msg += '\n  to fix this issue please process this or previous dark run using command jungfrau_dark_proc'\
                   '\n  or add the command line parameter -E or --errskip to skip missing file errors, use default,'\
                   'and force to deploy constants.'
            logger.error(msg)
            sys.exit(1)


def jungfrau_deploy_constants(parser):

    import psana.detector.UtilsJungfrau as uj

    args = parser.parse_args() # namespae of parameters
    kwargs = vars(args) # dict of parameters

    errskip   = kwargs.get('errskip', True)
    fac_mode  = kwargs.get('fac_mode', 0o664)
    group     = kwargs.get('group', 'ps-users')
    shape_seg = kwargs.get('shape_seg', (512,1024)) # pixel_gain:  shape:(3, 1, 512, 1024)  size:1572864  dtype:float32
    nsegstot  = kwargs.get('nsegstot', None)
    deploy    = kwargs.get('deploy', False)
    detname   = kwargs.get('detname', None)
    ctdepl    = kwargs.get('ctdepl', None) # 'prs'
    dbsuffix  = kwargs.get('dbsuffix', '')
    max_detname_size = kwargs.setdefault('max_detname_size', MAX_DETNAME_SIZE)

    DIC_CTYPE_FMT = dic_ctype_fmt(**kwargs)

    repoman = init_repoman_and_logger(parser=parser, **kwargs)
    #logger.info(uts.info_command_line_parameters(parser))
    kwargs['repoman'] = repoman

    logger.debug('open_DataSource for kwargs:\n%s' % uts.info_dict(kwargs, fmt='  %12s: %s', sep='\n'))
    ds, dskwargs = open_DataSource(**kwargs)

    orun = next(ds.runs())

    logger.info('\n%s Run %d %s' % (20*'=', orun.runnum, 20*'='))

    trun_sec = ups.seconds(orun.timestamp) # 1607569818.532117 sec
    ts_run, ts_now = ups.tstamps_run_and_now(trun_sec) #, fmt=uc.TSTAMP_FORMAT)

    odet = orun.Detector(detname, **kwargs)

    dettype = odet.raw._dettype
    repoman.set_dettype(dettype)

    uniqueid = odet.raw._uniqueid
    seginds = odet.raw._sorted_segment_inds
    segids = uniqueid.split('_')[1:]
    logger.debug('det.raw._uniqueid.split:\n%s' % ('\n'.join(uniqueid.split('_')))\
                +'det.raw._sorted_segment_inds: %s' % str(seginds))

    longname = uniqueid
    shortname = uc.detector_name_short(longname, maxsize=max_detname_size)
    logger.debug('detector names:\n  long name: %s\n  short name: %s' % (longname, shortname))

    ctypes = CTYPES_DEPL
    ctypes = [dic_calib_char_to_name[c] for c in ctdepl]

    logger.debug('ctdepl: %s ctypes: %s' % (ctdepl, str(ctypes)))

    #for ctype, fmt in DIC_CTYPE_FMT.items():
    for ctype in ctypes:
        fmt = DIC_CTYPE_FMT[ctype]
        #if ctype == 'status_extra':
        #    logger.warning('FOR NOW SKIP ctype: status_extra')
        #    continue
        octype = ctype
        logger.info('\n%s merge constants for calib type %s %s' % (70*'_', ctype, 70*'_'))

        dic_cons = {}

        for i,(segind, segid) in enumerate(zip(seginds, segids)):
            #logger.info('%s next segment\n  segment constants in repo for raw ind:%02d segment ind:%02d id: %s'%\
            #            (20*'-', i, segind, segid))

            dirpanel = repoman.dir_panel(segid)
            #logger.info('%s\nmerge gain range constants for panel %02d dir: %s' % (110*'_', segind, dirpanel))
            check_exists(dirpanel, errskip, 'panel directory does not exist %s' % dirpanel)

            dir_ctype = repoman.dir_ctype(segid, ctype) # <repo>/jungfrauemu/00b1ed0000-0000000000-...-0000000000/pedestals/
            if not os.path.exists(dir_ctype):
                dir_ctype = repoman.makedir_ctype(segid, ctype)

            fnprefix = fname_prefix(shortname, segind, ts_run, orun.expt, orun.runnum, dirname=None) # <dirname>/jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007
            #logger.debug('prefix: %s' % fnprefix)

            fname = fname_merged_gmodes(dir_ctype, fnprefix, ctype) # jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-pedestals.txt # -Normal
            logger.debug('fname_merged_gmodes: %s' % fname)
            nda = merge_jf_panel_gain_ranges(dir_ctype, segid, ctype, ts_run, shape_seg, fname,\
                                             fmt=fmt, fac_mode=fac_mode, errskip=errskip, group='ps-users')

            logger.info('-- save array of panel constants "%s" merged for 3 gain ranges shaped as %s in\n%s%s'\
                        % (ctype, str(nda.shape), 4*' ', fname))

#            if octype in dic_consts: dic_consts[octype].append(nda) # append for panel per ctype
#            else:                    dic_consts[octype] = [nda,]

            dic_cons[segind] = nda

        logger.debug('\n%s\nmerge all panel constants for ctype %s and deploy them' % (80*'_', ctype))

        #nda_def = np.ones(shape, dtype=np.float32) if ctype in ('gain', 'rms', 'dark_max') else\
        #      np.zeros(shape, dtype=np.float32) # 'pedestals', 'status', 'dark_min'
        vtype = cc.dic_calib_name_to_dtype[ctype]
        nda_def = np.zeros((3,1,)+shape_seg, dtype=vtype) # np.float32) # (3,1,512,1024)

        indmax = max(list(dic_cons.keys()))
        nsegs = uj.jungfrau_segments_tot(indmax) if nsegstot is None else nsegstot
        #nsegs = indmax+1 if nsegstot is None else nsegstot   # 1,2,8, or 32

        lst_cons = [(dic_cons[i] if i in dic_cons.keys() else nda_def) for i in range(nsegs)]

        dmerge = repoman.makedir_merge()
        fmerge_prefix = fname_prefix_merge(dmerge, shortname, ts_run, orun.expt, orun.runnum)
        logger.debug('fmerge_prefix: %s' % fmerge_prefix)
        nda = uc.merge_panels(lst_cons)
        fmerge = '%s-%s.txt' % (fmerge_prefix, ctype)
        logger.info(info_ndarr(nda, '%s\n    merged detector constants of %s' % (10*'-', ctype))\
                    + '\n    save in %s\n' % fmerge)
        save_ndarray_in_textfile(nda, fmerge, fac_mode, fmt, umask=0o0, group=group)

        if True:
          kwa_depl = uc.add_metadata_kwargs(orun, odet, **kwargs)
          kwa_depl['repoman'] = repoman
          kwa_depl['shape_as_daq'] = odet.raw._shape_as_daq()
          kwa_depl['run_orig'] = orun.runnum
          kwa_depl['extpars'] = {'content':'extended parameters dict->json->str',}
          kwa_depl['iofname'] = fmerge
          kwa_depl['ctype'] = ctype
          kwa_depl['dtype'] = 'ndarray'
          kwa_depl['extpars'] = {'content':'extended parameters dict->json->str',}
          kwa_depl['shortname'] = shortname
          kwa_depl['dbsuffix'] = dbsuffix
          kwa_depl.pop('exp',None) # remove parameters from kwargs - they passed as positional arguments
          kwa_depl.pop('repoman',None) # remove repoman parameters from kwargs

          d = ups.dict_filter(kwa_depl, list_keys=('dskwargs', 'dirrepo','dettype', 'tsshort', \
                'run', 'run_orig', 'run_beg', 'run_end',\
                'longname', 'shortname', 'segment_ids', 'segment_inds', 'shape_as_daq', 'nsegstot', 'version'))
          logger.info('DEPLOY partial metadata: %s' % uts.info_dict(d, fmt='%12s: %s', sep='\n  '))

        if deploy:
          expname = orun.expt  #'test' # FOR TEST ONLY > cdb_test

          if dbsuffix:
              resp = add_data_and_doc_to_detdb_extended(nda, expname, longname, **kwa_depl)
          else:
              # url=cc.URL_KRB, krbheaders=cc.KRBHEADERS
              resp = add_data_and_two_docs(nda, expname, longname, **kwa_depl)
          if resp:
              #id_data_exp, id_data_det, id_doc_exp, id_doc_det = resp
              if dbsuffix:
                  fmt = (None, resp[0], None, resp[1])
                  logger.debug('deployment id_data_exp:%s id_data_det:%s id_doc_exp:%s id_doc_det:%s' % fmt)
              else:
                  logger.debug('deployment id_data_exp:%s id_data_det:%s id_doc_exp:%s id_doc_det:%s' % resp)
          else:
              logger.error('constants are not deployed')
              exit()
        else:
          logger.warning('TO DEPLOY CONSTANTS IN DB ADD OPTION -D')

    repoman.logfile_save()

# EOF
