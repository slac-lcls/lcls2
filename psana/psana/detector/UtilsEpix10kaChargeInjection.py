#!/usr/bin/env python
"""
   import psana.detector.UtilsEpix10kaChargeInjection as ueci
"""
import os
import sys
from time import time
import psana.pyalgos.generic.Graphics as gr
from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL  # , init_stream_handler
import psana.detector.UtilsEpix10kaCalib as uec
from psana.detector.utils_psana import seconds, str_tstamp  # info_run, info_detector
from psana.detector.NDArrUtils import info_ndarr, divide_protected, save_2darray_in_textfile, save_ndarray_in_textfile
#from psana.detector.Utils import info_ndarr
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

ASAT = 16000 # 16384 or 1<<14 (15-th bit starting from 1)

create_directory = uec.create_directory

np, ue, find_gain_mode, GAIN_MODES, GAIN_MODES_IN, selected_record =\
 uec.np, uec.ue, uec.ue.find_gain_mode, uec.ue.GAIN_MODES, uec.ue.GAIN_MODES_IN, uec.selected_record


def get_panel_id(panel_ids, idx=0):
    panel_id = panel_ids[idx] if panel_ids is not None and idx is not None else None
    assert panel_id is not None, 'get_panel_id: panel_id is None for idx=%s' % str(idx)
    return panel_id


def file_name_npz(dir_work, fname_prefix, nspace):
    return '%s/%s_sp%02d_df.npz' % (dir_work, fname_prefix, nspace)


def print_statistics(nevt, nrec):
    logger.debug('statistics nevt:%d nrec:%d lost frames:%d' % (nevt, nrec, nevt-nrec))


class Storage:
    def __init__(self):
        pass

STORE = Storage() # singleton


def list_of_cc_collected():
  if not hasattr(STORE, 'cc_collected'):
    STORE.cc_collected = []
  return STORE.cc_collected


def pixel_row_col_from_str(pixrc):
    """converts str like '11,15' to the list of int (11,15)"""
    if pixrc is None: return None, None
    try:
        pixrow, pixcol = [int(v) for v in pixrc.split(',')]
        logger.info('use pixel row:%d col:%d for graphics' % (pixrow, pixcol))
        return pixrow, pixcol
    except:
        logger.error('vaiable pixrc="%s" can not be converted to pixel row,col' % str(pixrc))
        sys.exit()


def injection_row_col(nstep, space=5):
    """Charge injection point  vs step number."""
    irow = space - (nstep // space) + 1
    irow = irow % space
    icol = space - (nstep % space) - 1
    return irow, icol


def selected_pixel(pixrow, pixcol, jr, jc, nr, nc, nspace):
    """if pixel with panel indexes is in current block, returns tuple of its panel and block indexes,
       None oterwice.
    """
    blkrows = range(jr,nr,nspace)
    blkcols = range(jc,nc,nspace)
    if pixrow not in blkrows\
    or pixcol not in blkcols:
        return None
    ibr = blkrows.index(pixrow)
    ibc = blkcols.index(pixcol)
    msg = 'pixel panel r:%d c:%d    block r:%d c:%d' % (pixrow,pixcol,ibr,ibc)
    logger.info(msg)
    return pixrow, pixcol, ibr, ibc # tuple of panel and block indexes


def plot_fit_results(ifig, fitres, fnameout, filemode, gm, titles):
        fig = gr.plt.figure(ifig, facecolor='w', figsize=(11,8.5), dpi=72.27); gr.plt.clf()
        gr.plt.suptitle(gm)
        for i in range(4):
            gr.plt.subplot(2,2,i+1)
            test=fitres[:,:,i//2,i%2]; testm=np.median(test); tests=3*np.std(test)
            gr.plt.imshow(test, interpolation='nearest', cmap='Spectral', vmin=testm-tests, vmax=testm+tests)
            gr.plt.colorbar()
            gr.plt.title(gm+': '+titles[i])
        gr.plt.pause(0.1)
        fexists = os.path.exists(fnameout)
        fig.savefig(fnameout)
        logger.info('saved: %s' % fnameout)
        if not fexists: os.chmod(fnameout, filemode)


def saw_edges(trace, evnums, gainbitw, gap=50, do_debug=True):
    """
        2021-06-11 privious version neds at least two saw-tooth full cycles to find edgese...
        Returns list of triplet indexes [(ibegin, iswitch, iend), ...]
        in the arrays trace and evnums for found full periods of the charge injection pulser.
    """
    traceB14 = trace & gainbitw # trace & ue.B14,   np.bitwise_and(trace, B14)
    indsB14 = np.flatnonzero(traceB14) #shape:(604,) size:604 dtype:int64 [155 156 157 158 159...]
    evnumsB14 = evnums[indsB14]
    ixoff = np.where(np.diff(evnumsB14)>gap)[0]+1

    if do_debug:
        logger.debug(info_ndarr(trace, 'trace', last=10))
        logger.debug(info_ndarr(traceB14, 'trace & B14', last=10))
        logger.debug(info_ndarr(indsB14, 'indsB14'))
        logger.debug(info_ndarr(evnumsB14, 'evnumsB14'))
        logger.debug(info_ndarr(ixoff, 'ixoff', last=15))

    if len(ixoff)<1: return []

    grinds = np.split(indsB14, ixoff)
    edges_sw = [(g[0],g[-1]) for g in grinds]  #[(678, 991), (1702, 2015), (2725, 3039), (3751, 4063)]
    #print('XXX edges_sw:', str(edges_sw))

    edges = [] if len(edges_sw)<2 else\
            [((g0[1]+1,) + g1) for g0,g1 in zip(edges_sw[:-1],edges_sw[1:])]

    #print('XXX saw_edges:', str(edges))
    #np.save('trace.npy', trace)
    #np.save('evnums.npy', evnums)
    #sys.exit('TEST EXIT')

    return edges


def plot_fit_figaxis():
    if not hasattr(STORE, 'plot_fit_figax'):
        fig=gr.plt.figure(100,facecolor='w')
        ax=fig.add_subplot(111)
        STORE.plot_fit_figax = fig, ax
    return STORE.plot_fit_figax


def plot_fit(x, y, pf0, pf1, fname):
    print('plot_fit %s' % fname)
    fig, ax = plot_fit_figaxis()

    #fig.clf()
    ax.cla()
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 16384)
    ax.set_xticks(np.arange(0, 1100, 200))
    ax.set_yticks(np.arange(0, 16385, 2048))

    ax.plot(x, y, 'ko', markersize=1)
    ax.plot(x, np.polyval(pf0, x), 'b-', linewidth=1)
    ax.plot(x, np.polyval(pf1, x), 'r-', linewidth=1)

    ax.set_title(fname.rstrip('.png').rsplit('/',1)[-1], fontsize=6)#, color=color, fontsize=fstit, **kwargs)
    fig.canvas.manager.set_window_title(fname)
    gr.move_fig(fig, 10, 10)
    #plt.plot()
    fig.canvas.draw()
    gr.plt.pause(3)

    gr.plt.savefig(fname)
    logger.info('saved: %s' % fname)

    #plt.ioff()
    gr.plt.show()
    #plt.ion()


def plot_avsi_figaxis():
    if not hasattr(STORE, 'plot_avsi_figax'):
        fig = gr.plt.figure(101, facecolor='w')
        fig.clf()
        ax = fig.add_subplot(111)
        STORE.plot_avsi_figax = fig, ax
    return STORE.plot_avsi_figax


def plot_avsi(x, y, fname, gainbitw, databitw):

    fig, ax = plot_avsi_figaxis()
    gbit = np.bitwise_and(y, gainbitw) /8
    _y = y & databitw
    ax.cla()
    if _y.max()>2048: ax.set_ylim(0, 16384)
    ax.set_yticks(np.arange(0, 16385, 2048))
    line0,=ax.plot(x, _y, 'b-', linewidth=1)
    line1,=ax.plot(x, gbit, 'r-', linewidth=1)
    ax.set_title(fname.rstrip('.png').rsplit('/',1)[-1], fontsize=6)#, color=color, fontsize=fstit, **kwargs)
    fig.canvas.manager.set_window_title(fname)

    gr.move_fig(fig, 650, 200)
    #gr.plt.plot()
    fig.canvas.draw()
    gr.plt.pause(3)

    fig.savefig(fname)
    logger.info('saved: %s' % fname)
    #gr.plt.ioff()
    gr.plt.show()
    #gr.plt.ion()


def plot_data_block(block, evnums, prefix, gainbitw, databitw, selpix=None):
    ts = str_tstamp(fmt='%Y%m%dT%H%M%S', time_sec=time())
    mf,mr,mc=block.shape
    print('block shape:', mf, mr, mc)
    trace=block[:, 0, 0]
    x = np.arange(mf)
    print(info_ndarr(x, 'x'))
    print(info_ndarr(trace, 'trace'))

    for iy in range(mr):
        for ix in range(mc):
            selected = (iy*mc+ix)%256==255 if selpix is None\
                       else (iy==selpix[2] and ix==selpix[3])

            if selected:  #display a subset of plots

                trace=block[:,iy,ix]
                logger.info('==== saw_edge for %s-proc-ibr%02d-ibc%02d:' % (prefix, iy, ix))
                logger.info(' saw_edges: %s' % str(saw_edges(trace, evnums, gainbitw, gap=50, do_debug=True)))

                fname = '%s-dat-ibr%02d-ibc%02d.png' % (prefix, iy, ix) if selpix is None else\
                        '%s-dat-r%03d-c%03d-ibr%02d-ibc%02d.png' % (prefix, selpix[0], selpix[1], iy, ix)
                plot_avsi(evnums, trace, fname, gainbitw, databitw)


def fit(block, evnum, gainbitw, databitw, display=True, prefix='fig-fit', ixoff=10, nperiods=False, savechi2=False, selpix=None, npmin=5):

    mf, mr, mc=block.shape
    fits = np.zeros((mr, mc, 2, 2))
    chi2 = np.zeros((mr, mc, 2))
    nsp = np.zeros((mr, mc), dtype=np.int16)
    msg = ' fit '

    logger.info('fit selpix:' + str(selpix)) #selpix=(20, 97, 2, 13)
    logger.debug(info_ndarr(evnum, 'in fit evnum:'))
    logger.debug(info_ndarr(block, 'in fit block:'))
    #ts = str_tstamp(fmt='%Y%m%dT%H%M%S', time_sec=time())

    if display:
        plot_data_block(block, evnum, prefix, gainbitw, databitw, selpix)

    for iy in range(mr):
        for ix in range(mc):
            selected = (iy*mc+ix)%256==255 if selpix is None\
                       else (iy==selpix[2] and ix==selpix[3])

            trace=block[:, iy, ix]

            edges = saw_edges(trace, evnum, gainbitw, do_debug=(logger.level==logging.DEBUG))
            if len(edges)==0:
                 logger.warning('pulser saw edges are not found, skip processing for ix%02d-iy%02d:' % (ix, iy))
                 continue

            ixb, ixs, ixe = edges[0]
            nsp[iy,ix]=ixs
            tracem = trace & databitw

            x0 =  evnum[ixb:ixs-ixoff]-evnum[ixb]
            y0 = tracem[ixb:ixs-ixoff]
            # 2021-067-11 protection against overflow
            nonsaturated = np.where(y0<ASAT)[0] # [0] because where returns tuple of arrays - for dims?
            if nonsaturated.size != y0.size:
                x0 = x0[nonsaturated]
                y0 = y0[nonsaturated]

            x1 =  evnum[ixs+ixoff:ixe]-evnum[ixb]
            y1 = tracem[ixs+ixoff:ixe]

            if nperiods:
               for ixb,ixs,ixe in edges[1:]:
                 x0 = np.hstack((x0,  evnum[ixb:ixs-ixoff]-evnum[ixb]))
                 y0 = np.hstack((y0, tracem[ixb:ixs-ixoff]))
                 x1 = np.hstack((x1,  evnum[ixs+ixoff:ixe]-evnum[ixb]))
                 y1 = np.hstack((y1, tracem[ixs+ixoff:ixe]))

            if x0.size<npmin:
                 logger.warning(info_ndarr(x0, '\n    too short array x0', last=10))
                 continue
            if x1.size<npmin:
                 logger.warning(info_ndarr(x1, '\n    too short array x1', last=10))
                 continue

            pf0 = np.polyfit(x0, y0, 1, full=savechi2)
            pf1 = np.polyfit(x1, y1, 1, full=savechi2)

            if savechi2:
                pf0, res0, _, _, _ = pf0
                pf1, res1, _, _, _ = pf1

                chisq0 = res0 / (x0.size - 3)
                chisq1 = res1 / (x1.size - 3)
                chi2[iy,ix,:] = (chisq0, chisq1)

            fits[iy,ix,:] = (pf0, pf1)

            if selected: # for selected ix, iy
                s = '==== ibr%02d-ibc%02d:' % (iy, ix)
                if selpix is not None: s+=' === selected pixel panel r:%03d c:%03d' % (selpix[0], selpix[1])
                for ixb, ixs, ixe in edges:
                    s += '\n  saw edges begin: %4d switch: %4d end: %4d period: %4d' % (ixb, ixs, ixe, ixe-ixb+1)
                    s += info_ndarr(tracem, '\n    tracem', last=10)
                    s += info_ndarr(x0,     '\n    x0',  last=10)
                    s += info_ndarr(y0,     '\n    y0',  last=10)
                    s += info_ndarr(x1,     '\n    x1',  last=10)
                    s += info_ndarr(y1,     '\n    y1',  last=10)
                    s += info_ndarr(pf0,    '\n    pf0', last=10)
                    s += info_ndarr(pf1,    '\n    pf1', last=10)

                if savechi2:
                    s += '\n    chi2/ndof H/M %.3f' % chisq0
                    s += '\n    chi2/ndof L   %.3f' % chisq1

                logger.debug(s)

                msg+='.'
                if display:
                    fname = '%s-fit-ibr%02d-ibc%02d.png' % (prefix, iy, ix) if selpix is None else\
                            '%s-fit-r%03d-c%03d-ibr%02d-ibc%02d.png' % (prefix, selpix[0], selpix[1], iy, ix)

                    x = np.hstack((x0, x1))
                    y = np.hstack((y0, y1))
                    logger.debug(info_ndarr(x, '\n    x')\
                               + info_ndarr(y, '\n    y'))

                    #gr.plt.ioff() # hold control on plt.show()
                    plot_fit(x, y, pf0, pf1, fname)

    return fits, nsp, msg, chi2


def charge_injection(**kwa):

    str_dskwargs = kwa.get('dskwargs', None)
    detname    = kwa.get('det', None)
    idx        = kwa.get('idx', 0)
    nbs        = kwa.get('nbs', 4600)
    nspace     = kwa.get('nspace', 5)
    dirrepo    = kwa.get('dirrepo', uec.DIR_REPO_EPIX10KA)
    display    = kwa.get('display', True)
    fmt_offset = kwa.get('fmt_offset', '%.6f')
    fmt_peds   = kwa.get('fmt_peds',   '%.3f')
    fmt_rms    = kwa.get('fmt_rms',    '%.3f')
    fmt_status = kwa.get('fmt_status', '%4i')
    fmt_gain   = kwa.get('fmt_gain',   '%.6f')
    fmt_chi2   = kwa.get('fmt_chi2',   '%.3f')
    savechi2   = kwa.get('savechi2', False)
    dopeds     = kwa.get('dopeds', False)
    dooffs     = kwa.get('dooffs', True)
    dirmode    = kwa.get('dirmode', 0o2775)
    filemode   = kwa.get('filemode', 0o664)
    group      = kwa.get('group', 'ps-users')
    ixoff      = kwa.get('ixoff', 10)
    nperiods   = kwa.get('nperiods', True)
    ccnum      = kwa.get('ccnum', None)
    ccmax      = kwa.get('ccmax', 103)
    skipncc    = kwa.get('skipncc', 0)
    logmode    = kwa.get('logmode', 'DEBUG')
    errskip    = kwa.get('errskip', False)
    pixrc      = kwa.get('pixrc', None) # ex.: '23,123'
    nbs_half   = int(nbs/2)
    irun       = None
    exp        = None
    nstep_peds = 0
    step_docstring = None

    logger.setLevel(DICT_NAME_TO_LEVEL[logmode])
    uec.save_log_record_at_start(dirrepo, SCRNAME, dirmode, filemode, logmode, group=group)
    logger.info('\n  SCRNAME : %s\n  dskwargs: %s\n  detector: %s' % (SCRNAME, str_dskwargs, detname))

    dskwargs = uec.datasource_kwargs_from_string(str_dskwargs)
    try: ds = uec.DataSource(**dskwargs)
    except Exception as err:
        logger.error('DataSource(**dskwargs) does not work for **dskwargs: %s\n    %s' % (dskwargs, err))
        sys.exit('EXIT - requested DataSource does not exist or is not accessible.')

    logger.debug('ds.runnum_list = %s' % str(ds.runnum_list))
    logger.debug('ds.detectors = %s' % str(ds.detectors))
    xtc_files = getattr(ds, 'xtc_files', None)
    logger.info('ds.xtc_files:\n  %s' % ('None' if xtc_files is None else '\n  '.join(xtc_files)))

    cpdic = uec.get_config_info_for_dataset_detname(**kwa)
    logger.info('config_info:%s' % uec.info_dict(cpdic))  # fmt=fmt, sep=sep+sepnext)

    tstamp    = cpdic.get('tstamp', None)
    panel_ids = cpdic.get('panel_ids', None)
    exp       = cpdic.get('expname', None)
    shape     = cpdic.get('shape', None)
    irun      = cpdic.get('runnum', None)
    dettype   = cpdic.get('dettype', None)
    dsname    = dskwargs
    nr,nc     = shape

    gainbitw = ue.gain_bitword(dettype)  # 0o100000
    databitw = ue.data_bitword(dettype)  # 0o077777
    logger.info('gainbitw %s databitw %s' % (oct(gainbitw), oct(databitw)))
    assert gainbitw is not None, 'gainbitw has to be defined for dettype %s' % str(dettype)
    assert databitw is not None, 'databitw has to be defined for dettype %s' % str(dettype)

    ASAT = databitw - 100

    if display:
        fig2, axim2, axcb2 = gr.fig_img_cbar_axes()
        gr.move_fig(fig2, 500, 10)
        gr.plt.ion() # do not hold control on plt.show()

    selpix = None
    pixrow, pixcol = pixel_row_col_from_str(pixrc)  # converts str like '11,15' to the list of int (11,15)

    panel_id = get_panel_id(panel_ids, idx)
    logger.info('panel_id: %s' % panel_id)

    dir_panel, dir_offset, dir_peds, dir_plots, dir_work, dir_gain, dir_rms, dir_status = uec.dir_names(dirrepo, panel_id)
    fname_prefix, panel_alias = uec.file_name_prefix(dirrepo, dettype, panel_id, tstamp, exp, irun)
    prefix_offset, prefix_peds, prefix_plots, prefix_gain, prefix_rms, prefix_status =\
        uec.path_prefixes(fname_prefix, dir_offset, dir_peds, dir_plots, dir_gain, dir_rms, dir_status)
    fname_work = file_name_npz(dir_work, fname_prefix, nspace)

    create_directory(dir_panel,  mode=dirmode, group=group)
    create_directory(dir_offset, mode=dirmode, group=group)
    create_directory(dir_peds,   mode=dirmode, group=group)
    create_directory(dir_plots,  mode=dirmode, group=group)
    create_directory(dir_work,   mode=dirmode, group=group)
    create_directory(dir_gain,   mode=dirmode, group=group)
    create_directory(dir_rms,    mode=dirmode, group=group)
    create_directory(dir_status, mode=dirmode, group=group)

    s = '\n    '.join([d for d in (dir_panel, dir_offset, dir_peds, dir_plots, dir_work, dir_gain, dir_rms, dir_status)])
    logger.info('created or existing directories:\n    %s' % s)

    #sys.exit('TEST EXIT')

    chi2_ml = np.zeros((nr,nc,2))
    chi2_hl = np.zeros((nr,nc,2))
    nsp_ml  = np.zeros((nr,nc),dtype=np.int16)
    nsp_hl  = np.zeros((nr,nc),dtype=np.int16)

    #try:
    if os.path.exists(fname_work):

        logger.info('file %s\n  exists, begin to load charge injection results from file' % fname_work)

        npz = np.load(fname_work)
        logger.info('Charge-injection data loaded from file:'\
                    '\n  %s\nSKIP CALIBRATION CYCLES' % fname_work)

        darks   = npz['darks']
        fits_ml = npz['fits_ml']
        fits_hl = npz['fits_hl']

    #except IOError:
    else:

        logger.info('DOES NOT EXIST charge-injection data file:'\
                    '\n  %s\nBEGIN CALIBRATION CYCLES' % fname_work)

        darks   = np.zeros((7,nr,nc))
        fits_ml = np.zeros((nr,nc,2,2))
        fits_hl = np.zeros((nr,nc,2,2))

        nstep_tot = -1
        for orun in ds.runs():
          print('==== run:', orun.runnum)

          det = orun.Detector(detname)

          #cdet = orun.Detector('ControlData') # in lcls
          try: step_docstring = orun.Detector('step_docstring')
          except Exception as err:
            logger.error('run.Detector("step_docstring") does not work:\n    %s' % err)
            sys.exit('Exit processing due to missing info about dark data step.')

          for nstep_run, step in enumerate(orun.steps()):
            nstep_tot += 1
            docstr = step_docstring(step)
            logger.info('=============== step %02d ===============\n  step_docstring: %s' % (nstep_tot, docstr))

            metadic = uec.json.loads(docstr)
            nstep = uec.step_counter(metadic, nstep_tot, nstep_run, stype='chargeinj')
            if nstep is None: continue

            if nstep_tot<skipncc:
                logger.info('skip %d consecutive steps' % skipncc)
                continue

            elif nstep_tot>=ccmax:
                logger.info('total number of steps %d exceeds ccmax %d' % (nstep_tot, ccmax))
                break

            elif ccnum is not None:
                # process step ccnum ONLY if ccnum is specified
                if nstep < ccnum:
                    logger.info('step number %d is below selected ccnum %d - continue' % (nstep, ccnum))
                    continue
                elif nstep > ccnum:
                    logger.info('step number %d exceeds selected ccnum %d - break' % (nstep, ccnum))
                    break

            mode = find_gain_mode(det.raw, evt=None).upper()
            logger.info('gain mode %s' % str(mode))

            if mode in GAIN_MODES_IN:
                logger.info('========== step %d: known gain mode %s' %(nstep, mode))
            else:
                logger.info('========== step %d: UNKNOWN gain mode %s - SKIP STEP' %(nstep, mode))
                continue

#            if mode in GAIN_MODES_IN and nstep < len(GAIN_MODES_IN):
#                mode_in_meta = GAIN_MODES_IN[nstep]
#                logger.info('========== step %d: dark run processing for gain mode in configuration %s and metadata %s'\
#                            %(nstep, mode, mode_in_meta))
#                if mode != mode_in_meta:
#                  logger.warning('INCONSISTENT GAIN MODES IN CONFIGURATION AND METADATA')
#                  if not errskip: sys.exit()
#                  logger.warning('FLAG ERRSKIP IS %s - keep processing next steps' % errskip)
#                  continue

            figprefix = '%s-%s-seg%02d-cc%03d-%s'%\
                        (prefix_plots, detname.replace(':','-').replace('.','-'), idx, nstep, mode)

            nrec,nevt = -1,0
            #First nstep_peds (5?) steps correspond to darks:
            if dopeds and nstep<nstep_peds:
                msg = 'DARK step %d ' % nstep
                logger.warning('skip %s' % msg)

#                block=np.zeros((nbs,nr,nc),dtype=np.int16)

#                for nevt,evt in enumerate(step.events()):
#                    raw = det.raw.raw(evt)
#                    if raw is None: #skip empty frames
#                        logger.warning('Ev:%04d rec:%04d panel:%02d raw=None, t(sec)=%.6f' % (nevt, nrec, idx, evt._seconfs))
#                        msg += 'none'
#                        continue
#                    if nrec>nbs-2:
#                        break
#                    else:
#                        nrec += 1
#                        if raw.ndim > 2: raw=raw[idx,:]
#                        if selected_record(nevt):
#                           logger.info(info_ndarr(raw & databitw, 'Ev:%04d rec:%04d panel:%02d raw & databitw' % (nevt, nrec, idx)))
#                        if display and nevt<3:
#                            imsh, cbar = gr.imshow_cbar(fig2, axim2, axcb2, raw, amin=None, amax=None, extent=None,\
#                                                     interpolation='nearest', aspect='auto', origin='upper',\
#                                                     orientation='vertical', cmap='inferno')
#                            fig2.canvas.manager.set_window_title('Run:%d step:%d mode:%s panel:%02d' % (orun.runnum, nstep, mode, idx))
#                            fname = '%s-ev%02d-img-dark' % (figprefix, nevt)
#                            axim2.set_title(fname.rsplit('/',1)[-1], fontsize=6)
#                            fig2.savefig(fname+'.png')
#                            logger.info('saved: %s' % fname+'.png')

#                        block[nrec]=raw & databitw
#                        if nrec%200==0: msg += '.%s' % find_gain_mode(det, raw)

#                print_statistics(nevt, nrec)

#                darks[nstep,:,:], nda_rms, nda_status = proc_dark_block(block[:nrec,:,:], **kwa)
#                logger.debug(msg)

#                fname = '%s_rms_%s.dat' % (prefix_rms, GAIN_MODES[nstep])
#                save_2darray_in_textfile(nda_rms, fname, filemode, fmt_rms, umask=0o0, group=group)

#                fname = '%s_status_%s.dat' % (prefix_status, GAIN_MODES[nstep])
#                save_2darray_in_textfile(nda_status, fname, filemode, fmt_status, umask=0o0, group=group)

            ####################
            elif not dooffs:
                #logger.debug(info_ndarr(darks, 'darks'))
                if nstep>nstep_peds-1:
                    logger.info('nstep %d > %d (nstep_peds-1) - break' % (nstep, nstep_peds-1))
                    break
                logger.info('dooffs is %s - continue' % str(dooffs))
                continue
            ####################

            #Next nspace**2 steps correspond to pulsing in Auto Medium-to-Low
            elif nstep>nstep_peds-1 and nstep<nstep_peds+nspace**2:
                msg = ' AML %2d/%2d '%(nstep-nstep_peds+1, nspace**2)

                istep = nstep - nstep_peds
                #jr = istep // nspace
                #jc = istep % nspace

                jr, jc = injection_row_col(istep, nspace)

                if pixrc is not None:
                    selpix = selected_pixel(pixrow, pixcol, jr, jc, nr, nc, nspace)
                    if selpix is None:
                        logger.info(msg + ' skip, due to pixrc=%s'%pixrc)
                        continue
                    else:
                        logger.info(msg + ' process selected pixel:%s' % str(selpix))

                fid_old = None
                block = np.zeros((nbs, nr, nc), dtype=np.int16)
                evnum = np.zeros((nbs,), dtype=np.int16)
                for nevt, evt in enumerate(step.events()):   #read all frames
                    raw = det.raw.raw(evt)
                    if raw is None:
                        logger.warning('Ev:%04d rec:%04d panel:%02d AML raw=None' % (nevt, nrec, idx))
                        msg += 'none'
                        continue
                    if nrec>nbs-2:
                        break
                    else:
                        #---- 2021-06-10: check fiducial for consecutive events
                        tstamp = evt.timestamp    # 4193682596073796843 relative to 1990-01-01
                        fid = seconds(tstamp) # evt.get(EventId).fiducials()
                        if fid_old is not None:
                            dfid = fid-fid_old
                            if not (dfid < 0.009):  # dfid:0.008389     != 3:
                                logger.warning('TIME SYSTEM FAULT dfid!=3: Ev:%04d rec:%04d panel:%02d AML raw=None fiducials:%.6f dfid:%.6f'%\
                                            (nevt, nrec, idx, fid, dfid))
                                if nrec < nbs_half:
                                   logger.info('reset statistics in block and keep accumulation')
                                   nrec = -1
                                else:
                                   logger.info('terminate event loop and process block data')
                                   break
                        fid_old = fid
                        #print('nevt, nrec, fid: %04d %04d %d ' % (nevt, nrec, evt.get(EventId).fiducials()))
                        #----

                        nrec += 1
                        if raw.ndim > 2: raw=raw[idx,:]
                        if selected_record(nevt):
                           logger.info(info_ndarr(raw, 'Ev:%04d rec:%04d panel:%02d AML raw' % (nevt, nrec, idx)))
                        block[nrec] = raw
                        evnum[nrec] = nevt
                        if nevt%200==0: msg += '.'

                if display:
                    #imsh, cbar = gr.imshow_cbar(fig2, axim2, axcb2, block[nrec][:100,:100], amin=None, amax=None, extent=None,\
                    imsh, cbar = gr.imshow_cbar(fig2, axim2, axcb2, block[nrec], amin=None, amax=None, extent=None,\
                                             interpolation='nearest', aspect='auto', origin='upper',\
                                             orientation='vertical', cmap='inferno')
                    fig2.canvas.manager.set_window_title('Run:%d step:%d events:%d' % (orun.runnum, nstep, evnum[nrec])) #, **kwargs)
                    fname = '%s-img-charge' % figprefix
                    axim2.set_title(fname.rsplit('/',1)[-1], fontsize=6)
                    fig2.savefig(fname + '.png')
                    logger.info('saved: %s' % fname+'.png')

                print_statistics(nevt, nrec)

                block = block[:nrec, jr:nr:nspace, jc:nc:nspace]        # select only pulsed pixels
                evnum = evnum[:nrec]                                    # list of non-empty events
                fits0, nsp0, msgf, chi2 = fit(block, evnum, gainbitw, databitw, display, figprefix, ixoff, nperiods, savechi2, selpix) # fit offset, gain
                fits_ml[jr:nr:nspace, jc:nc:nspace] = fits0             # collect results
                nsp_ml[jr:nr:nspace, jc:nc:nspace] = nsp0               # collect switching points
                if savechi2: chi2_ml[jr:nr:nspace, jc:nc:nspace] = chi2 # collect chi2/dof
                s = '\n  block fit results AML'\
                  + info_ndarr(fits0[:,:,0,0],'\n  M gain',   last=5)\
                  + info_ndarr(fits0[:,:,1,0],'\n  L gain',   last=5)\
                  + info_ndarr(fits0[:,:,0,1],'\n  M offset', last=5)\
                  + info_ndarr(fits0[:,:,1,1],'\n  L offset', last=5)

                logger.info(msg + msgf + s)

            #Next nspace**2 Steps correspond to pulsing in Auto High-to-Low
            elif nstep>nstep_peds-1+nspace**2 and nstep<nstep_peds+2*nspace**2:
                msg = ' AHL %2d/%2d '%(nstep-nstep_peds-nspace**2+1, nspace**2)

                istep = nstep-nstep_peds-nspace**2
                #jr = istep // nspace
                #jc = istep % nspace
                jr, jc = injection_row_col(istep, nspace)

                if pixrc is not None:
                    selpix = selected_pixel(pixrow, pixcol, jr, jc, nr, nc, nspace)
                    if selpix is None:
                        logger.info(msg + ' skip, due to pixrc=%s'%pixrc)
                        continue

                fid_old = None
                block = np.zeros((nbs, nr, nc), dtype=np.int16)
                evnum = np.zeros((nbs,), dtype=np.int16)
                for nevt, evt in enumerate(step.events()):   #read all frames
                    raw = det.raw.raw(evt)
                    if raw is None:
                        logger.warning('Ev:%04d rec:%04d panel:%02d AHL raw=None' % (nevt, nrec, idx))
                        msg += 'None'
                        continue
                    if nrec>nbs-2:
                        break
                    else:
                        #---- 2021-06-10: check fiducial for consecutive events
                        tstamp = evt.timestamp    # 4193682596073796843 relative to 1990-01-01
                        fid = seconds(tstamp) # evt.get(EventId).fiducials()
                        if fid_old is not None:
                            dfid = fid-fid_old
                            if not (dfid < 0.009):  # dfid:0.008389
                                logger.warning('TIME SYSTEM FAULT dfid!=3: Ev:%04d rec:%04d panel:%02d AML raw=None fiducials:%.6f dfid:%.6f'%\
                                            (nevt, nrec, idx, fid, dfid))
                                if nrec < nbs_half:
                                   logger.info('reset statistics in block and keep accumulation')
                                   nrec = -1
                                else:
                                   logger.info('terminate event loop and process block data')
                                   break
                        fid_old = fid
                        #print('nevt, nrec, fid: %04d %04d %d ' % (nevt, nrec, evt.get(EventId).fiducials()))
                        #----

                        nrec += 1
                        if raw.ndim > 2: raw = raw[idx,:]
                        if selected_record(nevt):
                           logger.info(info_ndarr(raw, 'Ev:%04d rec:%04d panel:%02d AHL raw' % (nevt, nrec, idx)))
                        block[nrec] = raw
                        evnum[nrec] = nevt
                        if nevt%200 == 0: msg += '.'

                if display:
                    #imsh, cbar = gr.imshow_cbar(fig2, axim2, axcb2, block[nrec][:100,:100], amin=None, amax=None, extent=None,\
                    imsh, cbar = gr.imshow_cbar(fig2, axim2, axcb2, block[nrec], amin=None, amax=None, extent=None,\
                                             interpolation='nearest', aspect='auto', origin='upper',\
                                             orientation='vertical', cmap='inferno')
                    fig2.canvas.manager.set_window_title('Run:%d step:%d events:%d' % (orun.runnum, nstep, evnum[nrec])) #, **kwargs)
                    fname = '%s-img-charge' % figprefix
                    axim2.set_title(fname.rsplit('/',1)[-1], fontsize=6)
                    fig2.savefig(fname+'.png')
                    logger.info('saved: %s' % fname+'.png')

                print_statistics(nevt, nrec)

                block = block[:nrec, jr:nr:nspace, jc:nc:nspace]        # select only pulsed pixels
                evnum = evnum[:nrec]                                    # list of non-empty events
                fits0, nsp0, msgf, chi2=fit(block, evnum, gainbitw, databitw, display, figprefix, ixoff, nperiods, savechi2, selpix) # fit offset, gain
                fits_hl[jr:nr:nspace, jc:nc:nspace] = fits0             # collect results
                nsp_hl[jr:nr:nspace, jc:nc:nspace] = nsp0
                if savechi2: chi2_hl[jr:nr:nspace, jc:nc:nspace] = chi2 # collect chi2/dof
                s = '\n  block fit results AHL'\
                  + info_ndarr(fits0[:,:,0,0],'\n  H gain',   last=5)\
                  + info_ndarr(fits0[:,:,1,0],'\n  L gain',   last=5)\
                  + info_ndarr(fits0[:,:,0,1],'\n  H offset', last=5)\
                  + info_ndarr(fits0[:,:,1,1],'\n  L offset', last=5)
                logger.info(msg + msgf + s)

            elif nstep>=nstep_peds+2*nspace**2:
                break

            list_of_cc_collected().append(nstep)

        logger.debug(info_ndarr(fits_ml, '  fits_ml', last=10)) # shape:(352, 384, 2, 2)
        logger.debug(info_ndarr(fits_hl, '  fits_hl', last=10)) # shape:(352, 384, 2, 2)
        logger.debug(info_ndarr(darks,   '  darks',   last=10)) # shape:(352, 384, 7)

        #darks[6,:,:]=darks[4,:,:]-fits_ml[:,:,1,1] # 2020-06-19 M.D. - commented out, it is done later
        #darks[5,:,:]=darks[3,:,:]-fits_hl[:,:,1,1] # 2020-06-19 M.D. - commented out, it is done later

        #Save diagnostics data, can be commented out:
        #save fitting results
        fexists = os.path.exists(fname_work)
        np.savez_compressed(fname_work, darks=darks, fits_hl=fits_hl, fits_ml=fits_ml, nsp_hl=nsp_hl, nsp_ml=nsp_ml)
        if not fexists: os.chmod(fname_work, filemode)
        logger.info('Saved:  %s' % fname_work)

    #Save gains:
    gain_ml_m = fits_ml[:,:,0,0]
    gain_ml_l = fits_ml[:,:,1,0]
    gain_hl_h = fits_hl[:,:,0,0]
    gain_hl_l = fits_hl[:,:,1,0]
    fname_gain_AML_M = '%s_gainci_AML-M.dat' % prefix_gain
    fname_gain_AML_L = '%s_gainci_AML-L.dat' % prefix_gain
    fname_gain_AHL_H = '%s_gainci_AHL-H.dat' % prefix_gain
    fname_gain_AHL_L = '%s_gainci_AHL-L.dat' % prefix_gain
    save_2darray_in_textfile(gain_ml_m, fname_gain_AML_M, filemode, fmt_gain, umask=0o0, group=group)
    save_2darray_in_textfile(gain_ml_l, fname_gain_AML_L, filemode, fmt_gain, umask=0o0, group=group)
    save_2darray_in_textfile(gain_hl_h, fname_gain_AHL_H, filemode, fmt_gain, umask=0o0, group=group)
    save_2darray_in_textfile(gain_hl_l, fname_gain_AHL_L, filemode, fmt_gain, umask=0o0, group=group)

    #Save gain ratios:
    #fname_gain_RHL = '%s_gainci_RHoL.dat' % prefix_gain
    #fname_gain_RML = '%s_gainci_RMoL.dat' % prefix_gain
    #save_2darray_in_textfile(divide_protected(gain_hl_h, gain_hl_l), fname_gain_RHL, filemode, fmt_gain, umask=0o0, group=group)
    #save_2darray_in_textfile(divide_protected(gain_ml_m, gain_ml_l), fname_gain_RML, filemode, fmt_gain, umask=0o0, group=group)

    if savechi2:
        #Save chi2s:
        chi2_ml_m = chi2_ml[:,:,0]
        chi2_ml_l = chi2_ml[:,:,1]
        chi2_hl_h = chi2_hl[:,:,0]
        chi2_hl_l = chi2_hl[:,:,1]
        fname_chi2_AML_M = '%s_chi2ci_AML-M.dat' % prefix_gain
        fname_chi2_AML_L = '%s_chi2ci_AML-L.dat' % prefix_gain
        fname_chi2_AHL_H = '%s_chi2ci_AHL-H.dat' % prefix_gain
        fname_chi2_AHL_L = '%s_chi2ci_AHL-L.dat' % prefix_gain
        save_2darray_in_textfile(chi2_ml_m, fname_chi2_AML_M, filemode, fmt_chi2, umask=0o0, group=group)
        save_2darray_in_textfile(chi2_ml_l, fname_chi2_AML_L, filemode, fmt_chi2, umask=0o0, group=group)
        save_2darray_in_textfile(chi2_hl_h, fname_chi2_AHL_H, filemode, fmt_chi2, umask=0o0, group=group)
        save_2darray_in_textfile(chi2_hl_l, fname_chi2_AHL_L, filemode, fmt_chi2, umask=0o0, group=group)

    #Save offsets:
    offset_ml_m = fits_ml[:,:,0,1]
    offset_ml_l = fits_ml[:,:,1,1]
    offset_hl_h = fits_hl[:,:,0,1]
    offset_hl_l = fits_hl[:,:,1,1]
    fname_offset_AML_M = '%s_offset_AML-M.dat' % prefix_offset
    fname_offset_AML_L = '%s_offset_AML-L.dat' % prefix_offset
    fname_offset_AHL_H = '%s_offset_AHL-H.dat' % prefix_offset
    fname_offset_AHL_L = '%s_offset_AHL-L.dat' % prefix_offset
    save_2darray_in_textfile(offset_ml_m, fname_offset_AML_M, filemode, fmt_offset, umask=0o0, group=group)
    save_2darray_in_textfile(offset_ml_l, fname_offset_AML_L, filemode, fmt_offset, umask=0o0, group=group)
    save_2darray_in_textfile(offset_hl_h, fname_offset_AHL_H, filemode, fmt_offset, umask=0o0, group=group)
    save_2darray_in_textfile(offset_hl_l, fname_offset_AHL_L, filemode, fmt_offset, umask=0o0, group=group)

    #Save offsets:
    offset_ahl = offset_hl_h - offset_hl_l # 2020-06-19 M.D. - difference at 0 is taken as offset for peds
    offset_aml = offset_ml_m - offset_ml_l # 2020-06-19 M.D. - difference at 0 is taken as offset for peds
    fname_offset_AHL = '%s_offset_AHL.dat' % prefix_offset
    fname_offset_AML = '%s_offset_AML.dat' % prefix_offset
    save_2darray_in_textfile(offset_ahl, fname_offset_AHL, filemode, fmt_offset, umask=0o0, group=group)
    save_2darray_in_textfile(offset_aml, fname_offset_AML, filemode, fmt_offset, umask=0o0, group=group)

    #Save darks accounting offset whenever appropriate:
    for i in range(5):  #looping through darks measured in Jack's order
        fname = '%s_pedestals_%s.dat' % (prefix_peds, GAIN_MODES[i])
        save_2darray_in_textfile(darks[i,:,:], fname, filemode, fmt_peds, umask=0o0, group=group)

        if i==3:    # evaluate AHL_L from AHL_H
            ped_hl_h = darks[i,:,:]
            #ped_hl_l = ped_hl_h - offset_ahl # V0
            #ped_hl_l = ped_hl_h - offset_ahl + (offset_hl_h - ped_hl_h) * divide_protected(gain_hl_l, gain_hl_h) #V2
            ped_hl_l = offset_hl_l - (offset_hl_h - ped_hl_h) * divide_protected(gain_hl_l, gain_hl_h) #V3 Gabriel's
            fname = '%s_pedestals_AHL-L.dat' % prefix_peds
            save_2darray_in_textfile(ped_hl_l, fname, filemode, fmt_peds, umask=0o0, group=group)

        elif i==4:  # evaluate AML_L from AML_M
            ped_ml_m = darks[i,:,:]
            #ped_ml_l = ped_ml_m - offset_aml # V0
            #ped_ml_l = ped_ml_m - offset_aml + (offset_ml_m - ped_ml_m) * divide_protected(gain_ml_l, gain_ml_m) #V2
            ped_ml_l = offset_ml_l - (offset_ml_m - ped_ml_m) * divide_protected(gain_ml_l, gain_ml_m) #V3 Gabriel's
            fname = '%s_pedestals_AML-L.dat' % prefix_peds
            save_2darray_in_textfile(ped_ml_l, fname, filemode, fmt_peds, umask=0o0, group=group)

    if display:
        gr.plt.close("all")
        fnameout='%s_plot_AML.png' % prefix_plots
        gm='AML'; titles=['M Gain','M Pedestal', 'L Gain', 'M-L Offset']
        plot_fit_results(0, fits_ml, fnameout, filemode, gm, titles)

        fnameout='%s_plot_AHL.png' % prefix_plots
        gm='AHL'; titles=['H Gain','H Pedestal', 'L Gain', 'H-L Offset']
        plot_fit_results(1, fits_hl, fnameout, filemode, gm, titles)

        gr.plt.pause(5)


# EOF
