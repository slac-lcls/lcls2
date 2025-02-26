
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

import psana.detector.dir_root as dr
import psana.detector.UtilsCalib as uc
from psana.detector.RepoManager import init_repoman_and_logger
import psana.detector.utils_psana as ups # seconds, data_source_kwargs#
from psana.detector.NDArrUtils import info_ndarr, save_2darray_in_textfile, save_ndarray_in_textfile # import divide_protected
import psana.detector.Utils as uts # info_dict
import psana.pscalib.calib.CalibConstants as cc
from psana.pscalib.calib.MDBWebUtils import add_data_and_two_docs

SCRNAME = os.path.basename(sys.argv[0])

NUMBER_OF_GAIN_MODES = 3

CTYPES = ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min', 'status_extra')

dic_calib_char_to_name = cc.dic_calib_char_to_name # {'p':'pedestals', 'r':'pixel_rms', 's':'pixel_status',...}
# "p"-pedestals, "r"-rms, "s"-status, "g" or "c" - gain or charge-injection gain,

DIC_GAIN_MODE = {'FixedGain1':  1,
                 'FixedGain2':  2,
                 'ForcedGain1': 1,
                 'ForcedGain2': 2,
                 'HighGain0':   0,
                 'Normal':      0}

DIC_IND_TO_GAIN_MODE = {v:k for k,v in DIC_GAIN_MODE.items()} # or uts.inverse_dict(DIC_GAIN_MODE)

M14 = 0x3fff # 16383, 14-bit mask
FNAME_PANEL_ID_ALIASES = '%s/.aliases_jungfrau.txt' % dr.DIR_REPO_JUNGFRAU

#DIR_REPO = os.path.join(dr.DIR_ROOT, 'detector/gains/jungfrau')  # for jungfrau_gain_constants
#CALIB_REPO_JUNGFRAU = os.path.join(dr.DIR_ROOT, 'detector/gains/jungfrau/panels')  # for jungfrau_dark_proc, jungfrau_deploy_constants
#JUNGFRAU_REPO_SUBDIRS = ('pedestals', 'rms', 'status', 'dark_min', 'dark_max', 'plots')
# jungfrau repository naming scheme:
# /reg/g/psdm/detector/gains/jungfrau/panels/logs/2021/2021-04-21T101714_log_jungfrau_dark_proc_dubrovin.txt
# /reg/g/psdm/detector/gains/jungfrau/panels/logs/2021_log_jungfrau_dark_proc.txt
# /reg/g/psdm/detector/gains/jungfrau/panels/190408-181206-50c246df50010d/pedestals/jungfrau_0001_20201201085333_cxilu9218_r0238_pedestals_gm0-Normal.dat


print('\n  DIR_ROOT', dr.DIR_ROOT,\
      '\n  DIR_REPO', dr.DIR_REPO)


def dic_ctype_fmt(**kwargs):
    return {'pedestals'   : kwargs.get('fmt_peds', '%.3f'),
            'pixel_rms'   : kwargs.get('fmt_rms',  '%.3f'),
            'pixel_status': kwargs.get('fmt_status', '%4i'),
            'pixel_max'   : kwargs.get('fmt_max', '%i'),
            'pixel_min'   : kwargs.get('fmt_min', '%i'),
            'status_extra': kwargs.get('fmt_status', '%4i')}


class DarkProcJungfrau(uc.DarkProc):

    """dark data accumulation and processing for Jungfrau.
       Extends DarkProc to account for bad gain mode switch state in self.bad_switch array and pixel_status.
    """
    def __init__(self, **kwa):
        kwa.setdefault('datbits', M14)
        uc.DarkProc.__init__(self, **kwa)
        self.modes = ['Normal-00', 'Med-01', 'Low-11', 'UNKNOWN-10']


    def init_proc(self):
        uc.DarkProc.init_proc(self)
        shape_raw = self.arr_med.shape
        self.bad_switch = np.zeros(shape_raw, dtype=np.uint8)


    def add_event(self, raw, irec):
        uc.DarkProc.add_event(self, raw, irec)
        self.add_statistics_bad_gain_switch(raw, irec)


    def add_statistics_bad_gain_switch(self, raw, irec, evgap=10):
        #if irec%evgap: return #parsify events

        igm    = self.gmindex
        gmname = self.gmname

        t0_sec = time()
        gbits = raw>>14 # 00/01/11 - gain bits for mode 0,1,2
        fg0, fg1, fg2 = gbits==0, gbits==1, gbits==3
        bad = (np.logical_not(fg0),\
               np.logical_not(fg1),\
               np.logical_not(fg2))[igm]

        if irec%evgap==0:
           dt_sec = time()-t0_sec
           fgx = gbits==2
           sums = [fg0.sum(), fg1.sum(), fg2.sum(), fgx.sum()]
           logger.debug('Rec: %4d found pixels %s gain definition time: %.6f sec igm=%d:%s'%\
                    (irec,' '.join(['%s:%d' % (self.modes[i], sums[i]) for i in range(4)]), dt_sec, igm, gmname))

        np.logical_or(self.bad_switch, bad, self.bad_switch)


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
        #if plotim &   1: plot_image(self.arr_av1, tit=titpref + 'average')
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
    #print('odet.raw._config_object():', dir(dcfg))
    #co = get_jungfrau_config_object(env, _psana.Source(src))
    #return co.gainMode()


def open_DataSource(**kwargs):
    #ds = psana.DataSource(exp='mfxdaq23', run=7, dir='/sdf/data/lcls/drpsrcf/ffb/MFX/mfxdaq23/xtc')
    dskwargs = ups.data_source_kwargs(**kwargs)
    logger.info('dskwargs: %s' % (dskwargs))

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

    #(popts, pargs) = parser.parse_args()
    args = parser.parse_args() # namespae of parameters
    kwargs = vars(args) # dict of parameters

    repoman = init_repoman_and_logger(parser=parser, **kwargs)
    #logger.info(uts.info_command_line_parameters(parser))
    kwargs['repoman'] = repoman

    detname = args.det
    evskip = args.evskip
    #evstep = args.evstep
    events = args.events
    #source = args.source
    #dsname = args.dsname
    stepnum= args.stepnum
    stepmax= args.stepmax
    evcode = args.evcode
    segind = args.segind
    igmode = args.igmode
    dirrepo = args.dirrepo
    #expname = args.expname

    dirmode  = kwargs.get('dirmode',  0o2775)
    filemode = kwargs.get('filemode', 0o664)
    group    = kwargs.get('group', 'ps-users')

    ecm = False
    if evcode is not None:
        from Detector.EventCodeManager import EventCodeManager
        ecm = EventCodeManager(evcode, verbos=0)
        logger.info('requested event-code list %s' % str(ecm.event_code_list()))

    s = 'DIC_GAIN_MODE {<name> : <number>}'
    for k,v in DIC_GAIN_MODE.items(): s += '\n%16s: %d' % (k,v)
    logger.info(s)

    #_name = sys._getframe().f_code.co_name
    #uc.save_log_record_at_start(dirrepo, _name, dirmode, filemode)


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
    #is_single_run = uc.is_single_run_dataset(dsname)
    #logger.info('dsname: %s  detname: %s  is_single_run: %s' % (dsname, det.name, is_single_run))
    #dic_consts_tot = {} # {<gain_mode>:{<ctype>:nda3d_shape=(4, 192, 384)}}
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
            print('det.raw._uniqueid.split:', ('\n'.join(uniqueid.split('_'))))

        logger.info('created %s detector object of type %s' % (detname, dettype))
        logger.info(ups.info_detector(odet, cmt='detector info:\n      ', sep='\n      '))

        try:
          step_docstring = orun.Detector('step_docstring')
        except:
          step_docstring = None


        terminate_steps = False
        nevrun = 0
        for istep, step in enumerate(orun.steps()):
            nsteptot += 1

            print('step_docstring ' +
                  'is None' if step_docstring is None else\
                  str(json.loads(step_docstring(step))))

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


            ###############################
            #gmo = get_jungfrau_gain_mode_object(odet)
            #igm = DIC_GAIN_MODE[gmo.name]
            igm = igmode if igmode is not None else 0
            gmname = DIC_IND_TO_GAIN_MODE.get(igm, None)
            kwargs['gainmode'] = gmname
            logger.info('TBD gain mode: %s igm: %d' % (gmname, igm))
            ###############################

            if dpo is None:
               kwargs['dettype'] = dettype
               dpo = DarkProcJungfrau(**kwargs)
               dpo.runnum = orun.runnum
               dpo.exp = orun.expt
               #dpo.calibdir = env.calibDir().replace('//','/')
               dpo.ts_run, dpo.ts_now = ts_run, ts_now #uc.tstamps_run_and_now(env, fmt=uc.TSTAMP_FORMAT)
               dpo.detid = uniqueid
               dpo.gmindex = igm
               dpo.gmname = gmname

               igm0 = igm

            if igm != igm0:
               logger.warning('event for wrong gain mode index %d, expected %d' % (igm, igm0))

            if istep==0:
                logger.info('TBD gain mode info from jungfrau configuration') #\n%s' % info_gain_modes(gmo))

            logger.info('\n== begin step %d gain mode "%s" index %d' % (istep, gmname, igm))

            for ievt, evt in enumerate(step.events()):

                nevrun += 1
                nevtot += 1

                #print('xxx event %d'%ievt)#, end='\r')

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

                if ecm:
                  if not ecm.select(evt):
                    print('    skip event %d due to --evcode=%s selected %d ' % (ievt, evcode, nevsel), end='\r')
                    continue
                  #else: print()

                raw = odet.raw.raw(evt)
                if raw is None:
                    logger.info('det.raw.raw(evt) is None in event %d' % ievt)
                    continue

                raw = (raw if segind is None else raw[segind,:]) # NO & M14 herte

                nevsel += 1

                tsec = time()
                dt   = tsec - tdt
                tdt  = tsec
                if selected_record(ievt+1, events):
                    #print()
                    ss = 'run[%d] %d  step %d  events total/run/step/selected: %4d/%4d/%4d/%4d  time=%7.3f sec dt=%5.3f sec'%\
                         (irun, orun.runnum, istep, nevtot, nevrun, ievt+1, nevsel, time()-t0_sec, dt)
                    if ecm:
                       print()
                       ss += ' event codes: %s' % str(ecm.event_codes(evt))
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


            if ecm:
                logger.info('continue to accumulate statistics, due to --evcode=%s' % evcode)
            else:
                logger.info('reset statistics for next step')

            if dpo is not None:
                    dpo.summary()
                    dpo.show_plot_results()
                    kwa_save = uc.add_metadata_kwargs(orun, odet, **kwargs)
                    save_results(dpo, **kwa_save)
                    del(dpo)
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
            dpo.summary()
            dpo.show_plot_results()
            kwa_save = uc.add_metadata_kwargs(orun, odet, **kwargs)
            save_results(dpo, **kwa_save)
            del(dpo)
            dpo=None

        if terminate_runs:
            logger.info('terminate_runs')
            break

        # End of run-loop

    logger.info('number of runs processed %d' % (irun+1))
    logger.info('%s\ntotal consumed time = %.3f sec.' % (40*'_', time()-t0_sec))


def save_results(dpo, **kwa):
    logger.info('TBD save_results')
    #dpo.summary()
    ctypes = CTYPES # ('pedestals', 'pixel_rms', 'pixel_status', 'pixel_max', 'pixel_min') # 'status_extra'
    arr_av1, arr_rms, arr_sta = dpo.constants_av1_rms_sta()
    arr_max, arr_min = dpo.constants_max_min()
    consts = arr_av1, arr_rms, arr_sta, arr_max, arr_min
    logger.info('evaluated constants: \n  %s\n  %s\n  %s\n  %s\n  %s' % (
                info_ndarr(arr_av1, 'arr_av1', first=0, last=5),\
                info_ndarr(arr_rms, 'arr_rms', first=0, last=5),\
                info_ndarr(arr_sta, 'arr_sta', first=0, last=5),\
                info_ndarr(arr_max, 'arr_max', first=0, last=5),\
                info_ndarr(arr_min, 'arr_min', first=0, last=5)))
    dic_consts = dict(zip(ctypes, consts))

    kwa_depl = kwa
    longname = kwa_depl['longname'] # odet.raw._uniqueid
    shortname = uc.detector_name_short(longname)
    print('detector long  name: %s' % longname)
    print('detector short name: %s' % shortname)
    kwa_depl['shortname'] = shortname
    logger.info('kwa_depl:\n%s' % uts.info_dict(kwa_depl, fmt='  %12s: %s', sep='\n'))

    save_constants_in_repository(dic_consts, **kwa_depl)
    #dic_consts_tot[gainmode] = dic_consts
    del(dpo)
    dpo=None


def save_constants_in_repository(dic_consts, **kwa):

    logger.debug('save_constants_in_repository kwa:', kwa)

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
    runnum   = kwa.get('run_orig', None)
    uniqueid = kwa.get('uniqueid', 'not-def-id')
    shortname= kwa.get('shortname', 'not-def-shortname')
    segids   = kwa.get('segment_ids', [])
    seginds  = kwa.get('segment_inds', [])
    gainmode = kwa.get('gainmode', None)

    DIC_CTYPE_FMT = dic_ctype_fmt(**kwa)

    if repoman is None:
       repoman = RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, group=group, dettype=dettype)

    #segids = self._uniqueid.split('_')[1]
    #seginds = self._sorted_segment_inds # _segment_numbers
    for i,(segind,segid) in enumerate(zip(seginds, segids)):
      logger.info('%s next segment\n   save segment constants for gain mode:%s in repo for raw ind:%02d segment ind:%02d id: %s'%\
                  (20*'-', gainmode, i, segind, segid))

      for ctype, nda in dic_consts.items():

        dir_ct = repoman.makedir_ctype(segid, ctype)
        fprefix = fname_prefix(shortname, segind, tsshort, expname, runnum, dir_ct)

        fname = calib_file_name(fprefix, ctype, gainmode)
        fmt = DIC_CTYPE_FMT.get(ctype,'%.5f')
        arr2d = nda[i,:]
        print(info_ndarr(arr2d, '   %s' % ctype))  # shape:(4, 192, 384)

        #save_ndarray_in_textfile(nda, fname, filemode, fmt)
        save_2darray_in_textfile(arr2d, fname, filemode, fmt)

        logger.info('saved: %s' % fname)


def fname_prefix(detname, ind, tstamp, exp, runnum, dirname=None):
    """ <dirname>/jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007     -pixel_status-Normal.data """
    fnpref = '%s-s%02d-%s-%s-r%04d' % (detname, ind, tstamp, exp, runnum)
    return fnpref if dirname is None else '%s/%s' % (dirname, fnpref)

def fname_merged_gmodes(dir_ctype, fnprefix, ctype):
    """ <dirname>/jungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-pixel_status.txt """
    return '%s/%s-%s.txt' % (dir_ctype, fnprefix, ctype)

def calib_file_name(fprefix, ctype, gainmode, fmt='%s-%s-%s.data'):
    return fmt % (fprefix, ctype, gainmode)


def fname_prefix_merge(dmerge, detname, tstamp, exp, irun):
    return '%s/%s-%s-%s-r%04d' % (dmerge, detname, tstamp, exp, irun)


#def file_name_prefix(dirrepo, dettype, panel_id, tstamp, exp, irun):
#    """ epix10ka """
#    assert dettype is not None
#    panel_alias = alias_for_id(panel_id, fname=os.path.join(dirrepo, dettype, FNAME_PANEL_ID_ALIASES), exp=exp, run=irun)
#    return '%s_%s_%s_%s_r%04d' % (dettype, panel_alias, tstamp, exp, irun), panel_alias





#def fname_panel_id_aliases(dirrepo):
#    fname = FNAME_PANEL_ID_ALIASES if dirrepo != 'work' else\
#            os.path.join(dirrepo, os.path.basename(FNAME_PANEL_ID_ALIASES))
#    logger.info('file name for aliases: %s' % fname)
#    return fname

#def info_object_dir(o, sep=',\n  '):
#    return 'dir(%s):\n  %s' % (str(o), sep.join([v for v in dir(o) if v[:1]!='_']))


#def jungfrau_config_info(dsname, detname, idx=0):
#    from psana import DataSource, Detector
#    ds = DataSource(dsname)
#    det = Detector(detname)
#    env = ds.env()
#    co = jungfrau_config_object(env, det.source)
#
#    #print(info_object_dir(env))
#    #print(info_object_dir(co))
#    logger.debug('jungfrau config. object: %s' % str(co))
#
#    cpdic = {}
#    cpdic['expname'] = env.experiment()
#    cpdic['calibdir'] = env.calibDir().replace('//','/')
#    cpdic['strsrc'] = det.pyda.str_src
#    cpdic['shape'] = shape_from_config_jungfrau(co)
#    cpdic['panel_ids'] = jungfrau_uniqueid(ds, detname).split('_')
#    cpdic['dettype'] = det.dettype
#    #cpdic['gain_mode'] = find_gain_mode(det, data=None) #data=raw: distinguish 5-modes w/o data
#    for orun in ds.runs():
#      cpdic['runnum'] = orun.runnum
#      #for step in orun.steps():
#      #for nevt,evt in enumerate(ds.events()):
#      for nevt,evt in enumerate(orun.events()):
#        raw = det.raw(evt)
#        if raw is not None:
#            tstamp, tstamp_now = uc.tstamps_run_and_now(env)
#            cpdic['tstamp'] = tstamp
#            del ds
#            del det
#            break
#      break
#    logger.info('configuration info for %s %s segment=%d:\n%s' % (dsname, detname, idx, str(cpdic)))
#    return cpdic
#


def find_file_for_timestamp(dirname, pattern, tstamp, fnext='.data'):
    """fname ejungfrauemu_000001-s00-20250203095124-mfxdaq23-r0007-pedestals-Normal.data
    """
    # list of file names in directory, dirname, containing pattern
    logger.debug('\n  dirname: %s\n  pattern: %s\n, tstamp: %s' % (dirname, pattern, tstamp))
    fnames = [name for name in os.listdir(dirname) if os.path.splitext(name)[-1]==fnext and pattern in name]

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
                     logger.debug('  selected %s for %s and %s' % (os.path.basename(fname),pattern,tstamp))
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
    args = parser.parse_args() # namespae of parameters
    kwargs = vars(args) # dict of parameters

    errskip   = kwargs.get('errskip', True)
    fac_mode  = kwargs.get('fac_mode', 0o664)
    group     = kwargs.get('group', 'ps-users')
    shape_seg = kwargs.get('shape_seg', (512,1024)) # pixel_gain:  shape:(3, 1, 512, 1024)  size:1572864  dtype:float32
    nsegstot  = kwargs.get('nsegstot', 32)
    deploy    = kwargs.get('deploy', False)
    detname   = kwargs.get('det', None)
    ctdepl    = kwargs.get('ctdepl', None) # 'prs'

    DIC_CTYPE_FMT = dic_ctype_fmt(**kwargs)

    repoman = init_repoman_and_logger(parser=parser, **kwargs)
    #logger.info(uts.info_command_line_parameters(parser))
    kwargs['repoman'] = repoman

    logger.info('open_DataSource for kwargs:\n%s' % uts.info_dict(kwargs, fmt='  %12s: %s', sep='\n'))
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
    print('det.raw._uniqueid.split:\n%s' % ('\n'.join(uniqueid.split('_'))))
    print('det.raw._sorted_segment_inds: %s' % str(seginds))

    #logger.info('created %s detector object of type %s' % (detname, dettype))
    #logger.info(ups.info_detector(odet, cmt='detector info:\n      ', sep='\n      '))

    longname = uniqueid
    shortname = uc.detector_name_short(longname)
    print('detector long name: %s' % longname)
    print('detector short name: %s' % shortname)

    ctypes = CTYPES
    ctypes = [dic_calib_char_to_name[c] for c in ctdepl]

    print('ctdepl:', ctdepl)
    print('ctypes:', ctypes)
    #shape=(3, <num-segments>, 352, 384)

#   dict_consts for constants octype: 'pedestals','status', 'rms',  etc. {ctype:<list-of-per-panel-constants-merged-for-3-gains>}
#    dic_consts = {}
    #for octype, (ctype, fmt) in mparsDIC_CTYPE_FMT.items():

    for ctype, fmt in DIC_CTYPE_FMT.items():
        if ctype == 'status_extra':
            logger.warning('FOR NOW SKIP ctype: status_extra')
            continue
        octype = ctype
        logger.info('\n%s merge constants for calib type %s %s' % (50*'_', ctype, 50*'_'))

        dic_cons = {}

        for i,(segind, segid) in enumerate(zip(seginds, segids)):
            #logger.info('%s next segment\n  segment constants in repo for raw ind:%02d segment ind:%02d id: %s'%\
            #            (20*'-', i, segind, segid))

            dirpanel = repoman.dir_panel(segid)
            #logger.info('%s\nmerge gain range constants for panel %02d dir: %s' % (110*'_', segind, dirpanel))
            check_exists(dirpanel, errskip, 'panel directory does not exist %s' % dirpanel)

            dir_ctype = repoman.dir_type(segid, ctype) # <repo>/jungfrauemu/00b1ed0000-0000000000-...-0000000000/pedestals/

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

        nda_def = np.zeros((3,1,)+shape_seg, dtype=np.float32) # (3,1,512,1024)
        lst_cons = [(dic_cons[i] if i in dic_cons.keys() else nda_def) for i in range(nsegstot)]

        dmerge = repoman.makedir_merge()
        fmerge_prefix = fname_prefix_merge(dmerge, shortname, ts_run, orun.expt, orun.runnum)
        logger.debug('fmerge_prefix: %s' % fmerge_prefix)
        nda = uc.merge_panels(lst_cons)
        fmerge = '%s-%s.txt' % (fmerge_prefix, ctype)
        logger.info(info_ndarr(nda, '%s\n    merged for detector constants of %s' % (10*'-', ctype))\
                    + '\n    save in %s\n' % fmerge)
        save_ndarray_in_textfile(nda, fmerge, fac_mode, fmt, umask=0o0, group=group)

        if deploy:
          #kwa_depl = dict(kwargs)
          kwa_depl = uc.add_metadata_kwargs(orun, odet, **kwargs)
          #kwa_depl['gainmode'] = gainmode
          kwa_depl['repoman'] = repoman
          kwa_depl['detname'] = shortname
          kwa_depl['shortname'] = shortname
          kwa_depl['shape_as_daq'] = odet.raw._shape_as_daq()
          kwa_depl['run_orig'] = orun.runnum
          kwa_depl['extpars'] = {'content':'extended parameters dict->json->str',}
          kwa_depl['iofname'] = fmerge
          kwa_depl['ctype'] = ctype
          kwa_depl['dtype'] = 'ndarray'
          kwa_depl['extpars'] = {'content':'extended parameters dict->json->str',}

          _ = kwa_depl.pop('exp',None) # remove parameters from kwargs - they passed as positional arguments
          _ = kwa_depl.pop('det',None)

          #print('XXXXX kwa_depl', kwa_depl)
          #logger.info('kwa_depl:\n%s' % uts.info_dict(kwa_depl, fmt='  %12s: %s', sep='\n'))
          #sys.exit('TEST EXIT')

          logger.info('DEPLOY metadata: %s' % uts.info_dict(kwa_depl, fmt='%12s: %s', sep='\n  ')) #fmt='%12s: %s'

          resp = add_data_and_two_docs(nda, orun.expt, longname, **kwa_depl) # url=cc.URL_KRB, krbheaders=cc.KRBHEADERS
          if resp:
              #id_data_exp, id_data_det, id_doc_exp, id_doc_det = resp
              logger.debug('deployment id_data_exp:%s id_data_det:%s id_doc_exp:%s id_doc_det:%s' % resp)
          else:
              logger.info('constants are not deployed')
              exit()
        else:
          logger.warning('TO DEPLOY CONSTANTS IN DB ADD OPTION -D')



        
#        if dircalib is not None: calibdir = dircalib
#        #ctypedir = .../calib/Epix10ka::CalibV1/MfxEndstation.0:Epix10ka.0/'
#        calibgrp = uc.calib_group(dettype) # 'Epix10ka::CalibV1'
#        ctypedir = '%s/%s/%s' % (calibdir, calibgrp, strsrc)
#
#        if deploy:
#            ofname   = '%s.data'%runrange if runrange is not None else '%d-end.data'%irun
#            lfname   = None
#            verbos   = True
#            logger.info('deploy file %s/%s/%s' % (ctypedir, octype, ofname))
#            gu.deploy_file(fmerge, ctypedir, octype, ofname, lfname, verbos=(logmode=='DEBUG'))
#        else:
#            logger.warning('Add option -D to deploy files under directory %s' % ctypedir)





        
        
def deploy_constants_epix(ctypes, gainmodes, **kwa):

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
    shortname= kwa.get('shortname', 'not-def-shortname')
    shape_as_daq = kwa.get('shape_as_daq', (4, 192, 384))

    DIC_CTYPE_FMT = {'pedestals'   : kwargs.get('fmt_peds', '%.3f'),
                     'pixel_rms'   : kwargs.get('fmt_rms',  '%.3f'),
                     'pixel_status': kwargs.get('fmt_status', '%4i'),
                     'pixel_max'   : kwargs.get('fmt_max', '%i'),
                     'pixel_min'   : kwargs.get('fmt_min', '%i'),
                     'status_extra': kwargs.get('fmt_status', '%4i')}


    logger.info('kwa_depl:\n%s' % uts.info_dict(kwa_depl, fmt='  %12s: %s', sep='\n'))


    if repoman is None:
       repoman = init_repoman_and_logger(parser=parser, **kwargs)
       kwargs['repoman'] = repoman

#    if repoman is None:
#       repoman = RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, group=group, dettype=dettype)
    #dircons = repoman.makedir_constants(dname='constants')
    #fprefix = fname_prefix(detname, tsshort, expname, runnum, dircons)

    #segid = uniqueid.split('_',1)[-1]
    segid = uniqueid.split('_')[1]
    logger.info('\n\n\ndeploy_constants'\
               +'\n  segment id: %s' % segid\
               +'\n  gainmodes : %s' % str(gainmodes))

    #for ctype, nda in dic_consts.items():
    for ctype in ctypes:

      dir_ct = repoman.makedir_ctype(segid, ctype)
      #fprefix = fname_prefix(detname, tsshort, expname, runnum, dir_ct)
      #fprefix = fname_prefix(dettype, tsshort, expname, runnum, dir_ct)
      fprefix = fname_prefix(shortname, tsshort, expname, runnum, dir_ct)
      logger.info('=========================== combine calib array for %s' % ctype
                 +'\n  shortname:%s\n  tsshort:%s\n  expname:%s\n  runnum:%d\n  dir_ct:%s\n  fprefix:%s\n\n'%\
                  (shortname, tsshort, orun.expt, orun.runnum, dir_ct, fprefix))
      dic_nda = {}

      for gm in gainmodes:

        fname = calib_file_name(fprefix, ctype, gm)
        fmt = DIC_CTYPE_FMT.get(ctype,'%.5f')
        #save_ndarray_in_textfile(nda, fname, filemode, fmt)
        #save_2darray_in_textfile(nda, fname, filemode, fmt)

        logger.info('extract constants from repo file: %s' % fname)

        dtype = 'ndarray'
        kwa['iofname'] = fname
        kwa['ctype'] = ctype
        kwa['dtype'] = dtype
        kwa['extpars'] = {'content':'extended parameters dict->json->str',}
        _ = kwa.pop('exp',None) # remove parameters from kwargs - they passed as positional arguments
        _ = kwa.pop('det',None)

        try:
          data = data_from_file(fname, ctype, dtype, True)
          logger.info(info_ndarr(data, 'constants loaded from file', last=10))
        except AssertionError as err:
          logger.warning(err)
          data = np.zeros(shape_as_daq, np.uint16)
          logger.info(info_ndarr(data, 'substitute array with', last=10))
        dic_nda[gm] = data

      nda = np.stack([dic_nda[gm] for gm in gainmodes])
      fname = calib_file_name(fprefix, ctype, 'comb')
      fmt = DIC_CTYPE_FMT.get(ctype,'%.5f')
      save_ndarray_in_textfile(nda, fname, filemode, fmt)
      logger.info(info_ndarr(nda, 'constants combined for %s' % ctype))  # shape:(3, 4, 192, 384)
      logger.info('saved in file: %s' % fname)

      if deploy:
            logger.info('DEPLOY metadata: %s' % info_dict(kwa, fmt='%s: %s', sep='  ')) #fmt='%12s: %s'

            detname = kwa['longname']
            resp = add_data_and_two_docs(nda, expname, detname, **kwa) # url=cc.URL_KRB, krbheaders=cc.KRBHEADERS
            if resp:
                #id_data_exp, id_data_det, id_doc_exp, id_doc_det = resp
                logger.debug('deployment id_data_exp:%s id_data_det:%s id_doc_exp:%s id_doc_det:%s' % resp)
            else:
                logger.info('constants are not deployed')
                exit()
      else:
            logger.warning('TO DEPLOY CONSTANTS IN DB ADD OPTION -D')



































    
#def jungfrau_deploy_constants(pargs, popts):
#    """jungfrau deploy constants
#    """
#    t0_sec = time()
#
#    #(popts, pargs) = parser.parse_args()
#    kwa = vars(args) # dict of options
#
#    #logger.info(uts.info_command_line_parameters(parser))
#    #logger.info(info_kwargs(**kwa))
#
#    exp        = kwa.get('exp', None)
#    detname    = kwa.get('det', None)
#    run        = kwa.get('run', None)
#    runrange   = kwa.get('runrange', None) # '0-end'
#    tstamp     = kwa.get('tstamp', None)
#    dsnamex    = kwa.get('dsnamex', None)
#    dirrepo    = kwa.get('dirrepo', CALIB_REPO_JUNGFRAU)
#    dircalib   = kwa.get('dircalib', None)
#    deploy     = kwa.get('deploy', False)
#    errskip    = kwa.get('errskip', False)
#    logmode    = kwa.get('logmode', 'DEBUG')
#    dirmode    = kwa.get('dirmode',  0o2775)
#    filemode   = kwa.get('filemode', 0o664)
#    group      = kwa.get('group', 'ps-users')
#    gain0      = kwa.get('gain0', 41.5)    # ADU/keV ? /reg/g/psdm/detector/gains/jungfrau/MDEF/g0_gain.npy
#    gain1      = kwa.get('gain1', -1.39)   # ADU/keV ? /reg/g/psdm/detector/gains/jungfrau/MDEF/g1_gain.npy
#    gain2      = kwa.get('gain2', -0.11)   # ADU/keV ? /reg/g/psdm/detector/gains/jungfrau/MDEF/g2_gain.npy
#    offset0    = kwa.get('offset0', 0.01)  # ADU
#    offset1    = kwa.get('offset1', 300.0) # ADU
#    offset2    = kwa.get('offset2', 50.0)  # ADU
#    proc       = kwa.get('proc', None)
#    paninds    = kwa.get('paninds', None)
#    panel_type = kwa.get('panel_type', 'jungfrau')
#    fmt_peds   = kwa.get('fmt_peds',   '%.3f')
#    fmt_rms    = kwa.get('fmt_rms',    '%.3f')
#    fmt_status = kwa.get('fmt_status', '%4i')
#    fmt_minmax = kwa.get('fmt_status', '%6i')
#    fmt_gain   = kwa.get('fmt_gain',   '%.6f')
#    fmt_offset = kwa.get('fmt_offset', '%.6f')
#
#    fname_aliases = fname_panel_id_aliases(dirrepo)
#
#    panel_inds = None if paninds is None else [int(i) for i in paninds.split(',')] # conv str '0,1,2,3' to list [0,1,2,3]
#    dsname = uc.str_dsname(exp, run, dsnamex)
#    _name = sys._getframe().f_code.co_name
#
#    logger.info('In %s\n      dataset: %s\n      detector: %s' % (_name, dsname, detname))
#
#    #uc.save_log_record_at_start(dirrepo, _name, dirmode, filemode)
#
#    cpdic = jungfrau_config_info(dsname, detname)
#    tstamp_run  = cpdic.get('tstamp',    None)
#    expname     = cpdic.get('expname',   None)
#    shape       = cpdic.get('shape',     None)
#    #calibdir    = cpdic.get('calibdir',  None)
#    strsrc      = cpdic.get('strsrc',    None)
#    panel_ids   = cpdic.get('panel_ids', None)
#    dettype     = cpdic.get('dettype',   None)
#    irun        = cpdic.get('runnum',    None)
#
#    shape_panel = shape[-2:]
#    logger.info('shape of the detector: %s panel: %s' % (str(shape), str(shape_panel)))
#
#    tstamp = tstamp_run if tstamp is None else\
#             tstamp if int(tstamp)>9999 else\
#             uc.tstamp_for_dataset('exp=%s:run=%d'%(exp,tstamp))
#
#    logger.debug('search for calibration files with tstamp <= %s' % tstamp)
#
#    #repoman = uc.RepoManager(dirrepo=dirrepo, dirmode=dirmode, filemode=filemode, dir_log_at_start=DIR_LOG_AT_START, group=group)
#    repoman = kwa.get('repoman', None)
#
#    mpars = {\
#      'pedestals':    ('pedestals', fmt_peds),\
#      'pixel_rms':    ('rms',       fmt_rms),\
#      'pixel_status': ('status',    fmt_status),\
#      'dark_min':     ('dark_min',  fmt_minmax),\
#      'dark_max':     ('dark_max',  fmt_minmax),\
#    }
#
#    # dict_consts for constants octype: 'pedestals','status', 'rms',  etc. {ctype:<list-of-per-panel-constants-merged-for-3-gains>}
#    dic_consts = {}
#    for ind, panel_id in enumerate(panel_ids):
#
#        if panel_inds is not None and not (ind in panel_inds):
#            logger.info('skip panel %d due to -I or --paninds=%s' % (ind, panel_inds)),
#            continue # skip non-selected panels
#
#        dirpanel = repoman.dir_panel(panel_id)
#        logger.info('%s\nmerge gain range constants for panel %02d dir: %s' % (110*'_', ind, dirpanel))
#
#        check_exists(dirpanel, errskip, 'panel directory does not exist %s' % dirpanel)
#
#        fname_prefix, panel_alias = uc.file_name_prefix(panel_type, panel_id, tstamp, exp, irun, fname_aliases)
#        logger.debug('fname_prefix: %s' % fname_prefix)
#
#        for octype, (ctype, fmt) in mpars.items():
#            dir_ctype = repoman.dir_type(panel_id, ctype)
#            #logger.info('  dir_ctype: %s' % dir_ctype)
#            fname = '%s/%s_%s.txt' % (dir_ctype, fname_prefix, ctype)
#            nda = merge_jf_panel_gain_ranges(dir_ctype, panel_id, ctype, tstamp, shape_panel, fname, fmt, filemode, errskip=errskip, group=group)
#            logger.info('-- save array of panel constants "%s" merged for 3 gain ranges shaped as %s in file\n%s%s\n'\
#                        % (ctype, str(nda.shape), 21*' ', fname))
#
#            if octype in dic_consts: dic_consts[octype].append(nda) # append for panel per ctype
#            else:                    dic_consts[octype] = [nda,]
#
#    logger.info('\n%s\nmerge all panel constants and deploy them' % (80*'_'))
#
#    dmerge = repoman.makedir_merge()
#    fmerge_prefix = fname_prefix_merge(dmerge, detname, tstamp, exp, irun)
#
#    logger.info('fmerge_prefix: %s' % fmerge_prefix)
#
#    for octype,lst in dic_consts.items():
#        lst_nda = uc.merge_panels(lst)
#        logger.info(info_ndarr(lst_nda, 'merged constants for %s' % octype))
#        fmerge = '%s-%s.txt' % (fmerge_prefix, octype)
#        fmt = mpars[octype][1]
#        uc.save_ndarray_in_textfile(lst_nda, fmerge, filemode, fmt, umask=0o0, group=group)
#
#        if dircalib is not None: calibdir = dircalib
#        #ctypedir = .../calib/Epix10ka::CalibV1/MfxEndstation.0:Epix10ka.0/'
#        calibgrp = uc.calib_group(dettype) # 'Epix10ka::CalibV1'
#        ctypedir = '%s/%s/%s' % (calibdir, calibgrp, strsrc)
#
#        if deploy:
#            ofname   = '%s.data'%runrange if runrange is not None else '%d-end.data'%irun
#            lfname   = None
#            verbos   = True
#            logger.info('deploy file %s/%s/%s' % (ctypedir, octype, ofname))
#            gu.deploy_file(fmerge, ctypedir, octype, ofname, lfname, verbos=(logmode=='DEBUG'))
#        else:
#            logger.warning('Add option -D to deploy files under directory %s' % ctypedir)

# EOF
