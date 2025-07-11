#!/usr/bin/env python
""" import psana.detector.UtilsEpix10kaChargeInjection as ueci
"""
import os
import sys
from time import time, sleep
import psana2.pyalgos.generic.Graphics as gr
from psana2.detector.UtilsLogging import logging  # DICT_NAME_TO_LEVEL, init_stream_handler
import psana2.detector.UtilsEpix10kaCalib as uec
from psana2.detector.utils_psana import seconds, str_tstamp  # info_run, info_detector
from psana2.detector.NDArrUtils import info_ndarr, divide_protected, save_2darray_in_textfile, save_ndarray_in_textfile
#from psana2.detector.Utils import info_ndarr
from psana2.detector.RepoManager import init_repoman_and_logger
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

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
    """Charge injection point vs step number."""
    irow = space - (nstep // space) + 1
    irow = irow % space
    icol = space - (nstep % space) - 1
    return irow, icol


def selected_pixel(pixrc, jr, jc, nr, nc, nspace):
    """if pixel with panel indexes is in current block, returns tuple of its panel and block indexes,
       None oterwice.
    """
    pixrow, pixcol = pixel_row_col_from_str(pixrc)  # '11,15' > (11,15)

    blkrows = range(jr,nr,nspace)
    blkcols = range(jc,nc,nspace)
    if pixrow not in blkrows\
    or pixcol not in blkcols:
        return None
    ibr = blkrows.index(pixrow)
    ibc = blkcols.index(pixcol)
    logger.info('pixel on panel r:%d c:%d shape:(%d, %d)    in block r:%d c:%d' % (pixrow, pixcol, nr, nc, ibr, ibc))
    return pixrow, pixcol, ibr, ibc # tuple of panel and block indexes


def plot_array(arr, title='', vmin=None, vmax=None, prefix='', filemode=0o664):
    flimg = ug.fleximagespec(arr, arr=None, amin=vmin, amax=vmax)
    flimg.fig.canvas.manager.set_window_title(title)
    ug.gr.show(mode='DO NOT HOLD')

    fname='%s_plot_%s.png' % (prefix, title)
    fexists = os.path.exists(fname)
    flimg.fig.savefig(fname)
    logger.info('saved: %s' % fname)
    if not fexists: os.chmod(fname, filemode)


def plot_fit_results(ifig, fitres, fnameout, filemode, gm, titles, rcslices):
    fig = gr.plt.figure(ifig, facecolor='w', figsize=(11,8.5), dpi=72.27); gr.plt.clf()
    gr.plt.suptitle(gm)
    for i in range(4):
        gr.plt.subplot(2,2,i+1)
        test=fitres[rcslices[0],rcslices[1],i//2,i%2]; testm=np.median(test); tests=3*np.std(test)
        gr.plt.imshow(test, interpolation='nearest', cmap='Spectral', vmin=testm-tests, vmax=testm+tests)
        gr.plt.colorbar()
        gr.plt.title(gm+': '+titles[i])
    gr.plt.pause(0.1)
    fexists = os.path.exists(fnameout)
    fig.savefig(fnameout)
    logger.info('saved: %s' % fnameout)
    if not fexists: os.chmod(fnameout, filemode)


def saw_edges(trace, evnums, gainbitw, gap=10, do_debug=True):
    """ 2021-06-11 privious version needs at least two saw-tooth full cycles to find edges...
        Returns list of triplet indexes [(ibegin, iswitch, iend), ...]
        in the arrays trace and evnums for found full periods of the charge injection pulser.
    """
    trace_gbit = trace & gainbitw # trace & ue.B14,   np.bitwise_and(trace, B14)
    inds_gbit = np.flatnonzero(trace_gbit) #shape:(604,) size:604 dtype:int64 [155 156 157 158 159...]
    evnums_gbit = evnums[inds_gbit]
    noff = np.where(np.diff(evnums_gbit)>gap)[0]+1

    if do_debug:
        logger.debug(info_ndarr(trace, 'trace', last=10))
        logger.debug(info_ndarr(trace_gbit, 'trace & gainbit', last=10))
        logger.debug(info_ndarr(inds_gbit, 'inds_gbit'))
        logger.debug(info_ndarr(evnums_gbit, 'evnums_gbit'))
        logger.debug(info_ndarr(noff, 'noff', last=15))

    if len(noff)<1: return []

    grinds = np.split(inds_gbit, noff)
    edges_sw = [(g[0], g[-1]) for g in grinds]  #[(678, 991), (1702, 2015), (2725, 3039), (3751, 4063)]
    #print('XXX edges_sw:', str(edges_sw))

    edges = [] if len(edges_sw)<2 else\
            [((g0[1]+1,) + g1) for g0,g1 in zip(edges_sw[:-1], edges_sw[1:])]

    #print('XXX saw_edges:', str(edges))
    #np.save('trace.npy', trace)
    #np.save('evnums.npy', evnums)
    #sys.exit('TEST EXIT')

    return edges


def plot_fit_figaxis():
    if not hasattr(STORE, 'plot_fit_figax'):
        fig = gr.plt.figure(100,facecolor='w')
        ax  = fig.add_subplot(3, 1, (2, 3))
        axr = fig.add_subplot(311)
        STORE.plot_fit_figax = fig, ax, axr
    return STORE.plot_fit_figax


def plot_fit(x, y, pf0, pf1, fname, databitw):
    print('plot_fit %s' % fname)
    fig, ax, axr = plot_fit_figaxis()

    xmin, xmax, xtick = 0, 1100, 200
    ymin, ymax = -500, 500  # -int(databitw/64), int(databitw/64)
    #fig.clf()
    ax.cla()
    ax.set_ylim(0, databitw)
    ax.set_xticks(np.arange(xmin, xmax, xtick))
    ax.set_yticks(np.arange(0, databitw, 5000))
    ax.plot(x, y, 'ko', markersize=1)
    ax.plot(x, np.polyval(pf0, x), 'b-', linewidth=1)
    ax.plot(x, np.polyval(pf1, x), 'r-', linewidth=1)
    ax.set_ylabel('data and fit')

    axr.cla()
    axr.set_ylim(ymin, ymax)
    axr.set_xticks(np.arange(xmin, xmax, xtick))
    axr.set_yticks(np.arange(ymin, ymax, int(ymax/2)))
    axr.plot((x[0],x[-1]), (0,0), 'k-', linewidth=1)
    axr.plot(x, y - np.polyval(pf0, x), 'b-', linewidth=1)
    axr.plot(x, y - np.polyval(pf1, x), 'r-', linewidth=1)
    axr.set_ylabel('residuals')

    axr.set_title(fname.rstrip('.png').rsplit('/',1)[-1], fontsize=6)#, color=color, fontsize=fstit, **kwargs)
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


def plot_avsi(x, y, fname, gainbitw, databitw, tsec_show=10):

    fig, ax = plot_avsi_figaxis()
    gbit = np.bitwise_and(y, gainbitw) /8
    _y = y & databitw
    ax.cla()
    ax.set_ylim(-1000, gainbitw)
    #if _y.max()>2048: ax.set_ylim(0, 16384)
    ax.set_yticks(np.arange(0, gainbitw, 5000)) #  2048
    line0,=ax.plot(x, _y, 'b-', linewidth=1)
    line1,=ax.plot(x, gbit, 'r-', linewidth=1)
    ax.set_title(fname.rstrip('.png').rsplit('/',1)[-1], fontsize=6)#, color=color, fontsize=fstit, **kwargs)
    fig.canvas.manager.set_window_title(fname)

    gr.move_fig(fig, 650, 200)
    #gr.plt.plot()
    fig.canvas.draw()
    gr.plt.pause(tsec_show)

    fig.savefig(fname)
    logger.info('saved: %s' % fname)
    #gr.plt.ioff()
    gr.plt.show()
    #gr.plt.ion()

    #sys.exit('TEST EXIT')


def selected_to_show(ir, ic, selpix=None, irb_def=2, icb_def=21):
    """ returns True for selected pixel to plot trace and fit, othervice False"""
    #return (ir*mc+ic)%256==255 if selpix is None\
    #return (ir<rows_max and ic<cols_max) if selpix is None\
    return (ir==irb_def and ic==icb_def) if selpix is None\
           else (ir==selpix[2] and ic==selpix[3])


def plot_data_block(block, evnums, prefix, gainbitw, databitw, selpix=None, tsec_show=10):
    ts = str_tstamp(fmt='%Y%m%dT%H%M%S', time_sec=time())
    mf, mr, mc=block.shape
    print('block shape:', mf, mr, mc)
    trace=block[:, 0, 0]
    logger.info(info_ndarr(trace, 'trace'))

    for ir in range(mr):
        for ic in range(mc):
            if selected_to_show(ir, ic, selpix):  # display a subset of plots for traces

                trace=block[:,ir,ic]
                logger.info('==== saw_edge for %s-proc-ibr%02d-ibc%02d:' % (prefix, ir, ic))
                logger.info(' saw_edges: %s' % str(saw_edges(trace, evnums, gainbitw, gap=10, do_debug=True)))

                fname = '%s-dat-ibr%02d-ibc%02d.png' % (prefix, ir, ic) if selpix is None else\
                        '%s-dat-r%03d-c%03d-ibr%02d-ibc%02d.png' % (prefix, selpix[0], selpix[1], ir, ic)
                plot_avsi(evnums, trace, fname, gainbitw, databitw, tsec_show)


def fit(block, evnum, gainbitw, databitw, display=True, prefix='fig-fit', npoff=10,\
        nperiods=False, savechi2=False, selpix=None, npmin=5, tsec_show=5):

    mf, mr, mc = block.shape
    fits = np.zeros((mr, mc, 2, 2))
    chi2 = np.zeros((mr, mc, 2))
    neg = np.zeros((mr, mc), dtype=np.int16)  # number of found edge groups / saw periods
    msg = ' fit '

    logger.info('block.shape: %s fit selpix: %s' % (str(block.shape), str(selpix)))  # selpix=(20, 97, 2, 13)
    logger.debug(info_ndarr(evnum, 'in fit evnum:'))
    logger.debug(info_ndarr(block, 'in fit block:'))
    #ts = str_tstamp(fmt='%Y%m%dT%H%M%S', time_sec=time())

    if display:
        plot_data_block(block, evnum, prefix, gainbitw, databitw, selpix, tsec_show)

    #sys.exit('TEST EXIT')
    msgmax = 10  # max number of repeatyed messages
    msgnum = 0  # counter of repeatyed messages

    for ir in range(mr):
        for ic in range(mc):
            trace=block[:, ir, ic]

            edges = saw_edges(trace, evnum, gainbitw, gap=10, do_debug=(logger.level==logging.DEBUG))
            if len(edges)==0:
                msgnum += 1
                if msgnum > msgmax:
                    pass
                elif msgnum == msgmax:
                    logger.warning('pulser saw edges are not found, skip repeating messages')
                else:
                    logger.warning('pulser saw edges are not found, skip processing for ic%02d-ir%02d:' % (ic, ir))
                continue

            ibeg, iswt, iend = edges[0]
            #nsp[ir,ic] = iswt
            neg[ir,ic] = len(edges)
            tracem = trace & databitw

            x0 =  evnum[ibeg:iswt-npoff] - evnum[ibeg]
            y0 = tracem[ibeg:iswt-npoff]
            # 2021-067-11 protection against overflow
            nonsaturated = np.where(y0 < databitw)[0] # [0] because where returns tuple of arrays - for dims?
            if nonsaturated.size != y0.size:
                x0 = x0[nonsaturated]
                y0 = y0[nonsaturated]

            x1 =  evnum[iswt+npoff:iend] - evnum[ibeg]
            y1 = tracem[iswt+npoff:iend]

            if nperiods:
               for ibeg, iswt, iend in edges[1:]:
                 x0 = np.hstack((x0,  evnum[ibeg:iswt-npoff] - evnum[ibeg]))
                 y0 = np.hstack((y0, tracem[ibeg:iswt-npoff]))
                 x1 = np.hstack((x1,  evnum[iswt+npoff:iend] - evnum[ibeg]))
                 y1 = np.hstack((y1, tracem[iswt+npoff:iend]))

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
                chi2[ir,ic,:] = (chisq0, chisq1)

            fits[ir,ic,:] = (pf0, pf1)

            if selected_to_show(ir, ic, selpix):  # display a subset of plots
                s = '==== ibr%02d-ibc%02d:' % (ir, ic)
                if selpix is not None: s+=' === selected pixel panel r:%03d c:%03d' % (selpix[0], selpix[1])
                for ibeg, iswt, iend in edges:
                    s += '\n  saw edges begin: %4d switch: %4d end: %4d period: %4d' % (ibeg, iswt, iend, iend-ibeg+1)
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
                    fname = '%s-fit-ibr%02d-ibc%02d.png' % (prefix, ir, ic) if selpix is None else\
                            '%s-fit-r%03d-c%03d-ibr%02d-ibc%02d.png' % (prefix, selpix[0], selpix[1], ir, ic)

                    x = np.hstack((x0, x1))
                    y = np.hstack((y0, y1))
                    logger.debug(info_ndarr(x, '\n    x')\
                               + info_ndarr(y, '\n    y'))

                    #gr.plt.ioff() # hold control on plt.show()
                    plot_fit(x, y, pf0, pf1, fname, databitw)

    return fits, neg, msg, chi2


def wait_and_exit(tsec=5):
    sleep(tsec)
    sys.exit('EXIT after %d sec timout' % tsec)


def event_loop_and_fit(det, timing, orun, step, istep, nstep,\
                       idx, nspace, pixrc, nbs, nr, nc, dfid_med, msg, fig2, axim2, axcb2,\
                       figprefix, gainbitw, databitw, display, npoff, nperiods, savechi2,\
                       npmin, tsec_show):

    nbs_half = int(nbs/2)
    dfid_spr = int(dfid_med/10)

    jr, jc = injection_row_col(istep, nspace)
    logger.info('==== charge injection point minimal indices jr:%d jc:%d separation space:%d' % (jr, jc, nspace))

    selpix = None
    if pixrc is not None:
        selpix = selected_pixel(pixrc, jr, jc, nr, nc, nspace)
        if selpix is None:
            logger.info(msg + ' skip, due to pixrc=%s'%pixrc)
            return None  # continue steps
        else:
            logger.info(msg + ' process selected pixel:%s' % str(selpix))

    fid_old = None
    dfid = 0
    #tsec_old = 0
    #pulse_old = 0
    block = np.zeros((nbs, nr, nc), dtype=np.int16)
    evnum = np.zeros((nbs,), dtype=np.int16)

    nrec = -1
    nevt = 0

    for nevt, evt in enumerate(step.events()):   #read all frames
        raw = det.raw.raw(evt)
        if raw is None:
            logger.warning('Ev:%04d rec:%04d panel:%02d AML raw=None' % (nevt, nrec, idx))
            msg += 'None'
            continue
        if nrec>nbs-2:
            break
        else:
            #---- 2021-06-10: check fiducial for consecutive events
            #tstamp = evt.timestamp  # 4193682596073796843 relative to 1990-01-01
            #pulse = timing.raw.pulseId(evt)
            #print('dir(timing)', dir(timing))
            #print('dir(timing.raw)', dir(timing.raw))
            #print('timing.raw.pulseId(evt)', pulse)
            #print('dir(evt)', dir(evt))
            #print('evt.datetime()', evt.datetime())
            #tsec = seconds(evt.timestamp)
            #print('ev %03d diff(seconds(evt.timestamp)): %.6f' % (nevt, tsec-tsec_old))
            #tsec_old = tsec
            #pulse_old = pulse
            #if nevt>100: sys.exit('TEST EXIT')

            fid = timing.raw.pulseId(evt) # evt.get(EventId).fiducials() at lcls1
            if fid_old is not None:
                dfid = fid-fid_old
                logger.debug('EVENT FIDUCIAL CHECK |dfid-%d|<%d: Ev:%04d rec:%04d panel:%02d AML raw=None dfid:%d'%\
                              (dfid_med, dfid_spr, nevt, nrec, idx, dfid))
                if not (abs(dfid-dfid_med)<dfid_spr):  # dfid_med=7761
                    logger.warning('FAILED EVENT FIDUCIAL CHECK dfid<%d: Ev:%04d rec:%04d panel:%02d AML raw=None fiducials:%d dfid:%d'%\
                                (dfid_med, nevt, nrec, idx, fid, dfid))
                    if nrec < nbs_half:
                       logger.info('reset statistics in block and keep accumulation')
                       nrec = -1
                    else:
                       logger.info('terminate event loop and process block data')
                       break
            fid_old = fid

            #----

            nrec += 1
            if raw.ndim > 2: raw=raw[idx,:]
            if selected_record(nevt):
               logger.info(info_ndarr(raw, 'Ev:%04d dfid:%04d rec:%04d panel:%02d AML raw' % (nevt, dfid, nrec, idx)))
            block[nrec] = raw
            evnum[nrec] = nevt
            if nevt%200==0: msg += '.'

    if display:
        #imsh, cbar = gr.imshow_cbar(fig2, axim2, axcb2, block[nrec][:144,:192], amin=None, amax=None, extent=None,\
        imsh, cbar = gr.imshow_cbar(fig2, axim2, axcb2, block[nrec], amin=None, amax=None, extent=None,\
                                 interpolation='nearest', aspect='auto', origin='upper',\
                                 orientation='vertical', cmap='inferno')
        fig2.canvas.manager.set_window_title('Run:%d step:%d events:%d' % (orun.runnum, nstep, evnum[nrec])) #, **kwargs)
        fname = '%s-img-charge' % figprefix
        axim2.set_title(fname.rsplit('/',1)[-1], fontsize=6)
        fig2.savefig(fname + '.png')
        logger.info('saved: %s' % fname+'.png')

    print_statistics(nevt, nrec)

    block = block[:nrec, jr:nr:nspace, jc:nc:nspace]        # only pulsed pixels
    evnum = evnum[:nrec]                                    # list of non-empty events
    fits0, neg0, msgf, chi2 = fit(block, evnum, gainbitw, databitw, display, figprefix, npoff,\
                                  nperiods, savechi2, selpix, npmin, tsec_show) # fit offset, gain

    return fits0, neg0, msgf, chi2, jr, jc


def charge_injection(parser):

    args = parser.parse_args()
    kwa = vars(args)
    repoman = init_repoman_and_logger(parser=parser, **kwa)

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
    fmt_status = kwa.get('fmt_status', '%d')
    fmt_gain   = kwa.get('fmt_gain',   '%.6f')
    fmt_chi2   = kwa.get('fmt_chi2',   '%.3f')
    savechi2   = kwa.get('savechi2', False)
    dopeds     = kwa.get('dopeds', False)
    dirmode    = kwa.get('dirmode', 0o2775)
    filemode   = kwa.get('filemode', 0o664)
    group      = kwa.get('group', 'ps-users')
    npoff      = kwa.get('npoff', 10)
    nperiods   = kwa.get('nperiods', True)
    ccnum      = kwa.get('ccnum', None)
    ccmax      = kwa.get('ccmax', 50)
    ccskip     = kwa.get('ccskip', 0)
    logmode    = kwa.get('logmode', 'DEBUG')
    errskip    = kwa.get('errskip', False)
    pixrc      = kwa.get('pixrc', None) # ex.: '23,123'
    nsigm      = kwa.get('nsigm', 8)
    sslice     = kwa.get('slice', '0:,0:')
    irun       = None
    exp        = None
    npmin      = 5
    nstep_peds = 0
    tsec_show  = 2
    tsec_show_end = 60
    step_docstring = None
    dfid_med = 7761 # THIS VALUE DEPENDS ON EVENT RATE -> SHOULD BE AUTOMATED

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

    repoman.set_dettype(dettype)

    gainbitw = ue.gain_bitword(dettype)  # 0o100000
    databitw = ue.data_bitword(dettype)  # 0o077777
    logger.info('gainbitw %s databitw %s' % (oct(gainbitw), oct(databitw)))
    assert gainbitw is not None, 'gainbitw has to be defined for dettype %s' % str(dettype)
    assert databitw is not None, 'databitw has to be defined for dettype %s' % str(dettype)

    if display:
        fig2, axim2, axcb2 = gr.fig_img_cbar_axes()
        gr.move_fig(fig2, 500, 10)
        gr.plt.ion() # do not hold control on plt.show()
    else:
        fig2, axim2, axcb2 = None, None, None

    panel_id = get_panel_id(panel_ids, idx)
    logger.info('panel_id: %s' % panel_id)

    dir_panel, dir_offset, dir_peds, dir_plots, dir_work, dir_gain, dir_rms, dir_status = uec.dir_names(repoman, panel_id)
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

    darks   = np.zeros((7,nr,nc))

    chi2_ml = np.zeros((nr, nc, 2))
    chi2_hl = np.zeros((nr, nc, 2))
    neg_ml  = np.zeros((nr, nc), dtype=np.int16)
    neg_hl  = np.zeros((nr, nc), dtype=np.int16)

    if os.path.exists(fname_work):

        logger.info('file %s\n  exists, begin to load charge injection results from file' % fname_work)

        npz = np.load(fname_work)
        logger.info('Charge-injection data loaded from file:'\
                    '\n  %s\nSKIP CALIBRATION CYCLES' % fname_work)

        #darks   = npz['darks']
        fits_ml = npz['fits_ml']
        fits_hl = npz['fits_hl']
        neg_ml  = npz['neg_ml']
        neg_hl  = npz['neg_hl']
        chi2_ml = npz['chi2_ml']
        chi2_hl = npz['chi2_hl']

    else:

        logger.info('DOES NOT EXIST charge-injection data file:'\
                    '\n  %s\nBEGIN CALIBRATION CYCLES' % fname_work)

        fits_ml = np.zeros((nr,nc,2,2))
        fits_hl = np.zeros((nr,nc,2,2))

        nstep_tot = -1
        for orun in ds.runs():
          print('==== run:', orun.runnum)

          det = orun.Detector(detname)
          timing = orun.Detector('timing')
          timing.raw._add_fields()

          #if dettype is None:
          #   dettype = odet.raw._dettype
          #   repoman.set_dettype(dettype)

          #Cdet = orun.Detector('ControlData') # in lcls
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

            if nstep_tot<ccskip:
                logger.info('skip %d consecutive steps' % ccskip)
                continue

            elif nstep_tot>=ccmax:
                logger.info('total number of steps %d exceeds ccmax %d' % (nstep_tot, ccmax))
                wait_and_exit(tsec_show)
                #break

            elif ccnum is not None:
                # process step ccnum ONLY if ccnum is specified
                if nstep < ccnum:
                    logger.info('step number %d < selected ccnum %d - continue' % (nstep, ccnum))
                    continue
                elif nstep > ccnum:
                    logger.info('step number %d > selected ccnum %d - break' % (nstep, ccnum))
                    wait_and_exit(tsec_show)
                    #break

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

            #First nstep_peds (5?) steps correspond to darks:
            if dopeds and nstep<nstep_peds:
                msg = 'DARK step %d ' % nstep
                logger.warning('skip %s' % msg)

            # Next nspace**2 steps correspond to pulsing in AML - Auto Medium-to-Low
            elif nstep>nstep_peds-1 and nstep<nstep_peds+nspace**2:
                msg = ' AML %2d/%2d '%(nstep-nstep_peds+1, nspace**2)
                istep = nstep - nstep_peds

                resp = event_loop_and_fit(det, timing, orun, step, istep, nstep,\
                    idx, nspace, pixrc, nbs, nr, nc, dfid_med, msg, fig2, axim2, axcb2,\
                    figprefix, gainbitw, databitw, display, npoff, nperiods, savechi2,\
                    npmin, tsec_show)

                if resp is None: continue  # steps
                fits0, neg0, msgf, chi2, jr, jc = resp

                fits_ml[jr:nr:nspace, jc:nc:nspace] = fits0             # collect results
                neg_ml[jr:nr:nspace, jc:nc:nspace] = neg0               # collect switching points
                if savechi2: chi2_ml[jr:nr:nspace, jc:nc:nspace] = chi2 # collect chi2/dof
                s = '\n  block fit results AML'\
                  + info_ndarr(fits0[:,:,0,0],'\n  M gain',   last=5)\
                  + info_ndarr(fits0[:,:,1,0],'\n  L gain',   last=5)\
                  + info_ndarr(fits0[:,:,0,1],'\n  M offset', last=5)\
                  + info_ndarr(fits0[:,:,1,1],'\n  L offset', last=5)

                logger.info(msg + str(msgf) + s)

            # Next nspace**2 Steps correspond to pulsing in AHL - Auto High-to-Low
            elif nstep>nstep_peds-1+nspace**2 and nstep<nstep_peds+2*nspace**2:
                msg = ' AHL %2d/%2d '%(nstep-nstep_peds-nspace**2+1, nspace**2)
                istep = nstep-nstep_peds-nspace**2

                resp = event_loop_and_fit(det, timing, orun, step, istep, nstep,\
                    idx, nspace, pixrc, nbs, nr, nc, dfid_med, msg, fig2, axim2, axcb2,\
                    figprefix, gainbitw, databitw, display, npoff, nperiods, savechi2,\
                    npmin, tsec_show)

                if resp is None: continue  # steps
                fits0, neg0, msgf, chi2, jr, jc = resp

                fits_hl[jr:nr:nspace, jc:nc:nspace] = fits0             # collect results
                neg_hl[jr:nr:nspace, jc:nc:nspace] = neg0               # collect switching points
                if savechi2: chi2_hl[jr:nr:nspace, jc:nc:nspace] = chi2 # collect chi2/dof
                s = '\n  block fit results AHL'\
                  + info_ndarr(fits0[:,:,0,0],'\n  H gain',   last=5)\
                  + info_ndarr(fits0[:,:,1,0],'\n  L gain',   last=5)\
                  + info_ndarr(fits0[:,:,0,1],'\n  H offset', last=5)\
                  + info_ndarr(fits0[:,:,1,1],'\n  L offset', last=5)
                logger.info(msg + str(msgf) + s)

            elif nstep>=nstep_peds+2*nspace**2:
                break

            list_of_cc_collected().append(nstep)

        logger.debug(info_ndarr(fits_ml, '  fits_ml', last=10)) # shape:(352, 384, 2, 2)
        logger.debug(info_ndarr(fits_hl, '  fits_hl', last=10)) # shape:(352, 384, 2, 2)
        #logger.debug(info_ndarr(darks,   '  darks',   last=10)) # shape:(352, 384, 7)

        #darks[6,:,:]=darks[4,:,:]-fits_ml[:,:,1,1] # 2020-06-19 M.D. - commented out, it is done later
        #darks[5,:,:]=darks[3,:,:]-fits_hl[:,:,1,1] # 2020-06-19 M.D. - commented out, it is done later

        #Save diagnostics data, can be commented out:
        #save fitting results
        fexists = os.path.exists(fname_work)
        #np.savez_compressed(fname_work, darks=darks, fits_hl=fits_hl, fits_ml=fits_ml, neg_hl=neg_hl, neg_ml=neg_ml)
        np.savez_compressed(fname_work, fits_hl=fits_hl, fits_ml=fits_ml, neg_hl=neg_hl, neg_ml=neg_ml,\
                            chi2_hl=chi2_hl, chi2_ml=chi2_ml)
        if not fexists: os.chmod(fname_work, filemode)
        logger.info('Saved:  %s' % fname_work)


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

    if False: # 2023-02-13 for now do not save any evaluated offset values
        #Save offsets:
        offset_ahl = offset_hl_h - offset_hl_l # 2020-06-19 M.D. - difference at 0 is taken as offset for peds
        offset_aml = offset_ml_m - offset_ml_l # 2020-06-19 M.D. - difference at 0 is taken as offset for peds
        fname_offset_AHL = '%s_offset_AHL.dat' % prefix_offset
        fname_offset_AML = '%s_offset_AML.dat' % prefix_offset
        save_2darray_in_textfile(offset_ahl, fname_offset_AHL, filemode, fmt_offset, umask=0o0, group=group)
        save_2darray_in_textfile(offset_aml, fname_offset_AML, filemode, fmt_offset, umask=0o0, group=group)


    #Find and load darks:
    #dir_peds = prefix_peds.rsplit('/',1)[0]
    logger.info('== Find and load dark arrays with earlier timestamp in the directory:\n  %s' %  dir_peds)
      # prefix_peds = <path>/<panelid>/pedestals/epixhr2x2_0001_19900121012401_ascdaq18_r0171
    #pattern = 'pedestals_AHL-H'
    #for i in range(5):
    #    pattern = 'pedestals_%s' % GAIN_MODES[i]
    #    msg = '  for pattern: %s tstamp: %s' % (pattern, tstamp)
    #    fname_dark = uec.find_file_for_timestamp(dir_peds, pattern, tstamp)
    #    logger.debug('%s\n    found: %s' % (msg, fname_dark))

    ped_fh   = uec.load_panel_constants(dir_peds, 'pedestals_FH', tstamp)
    ped_fm   = uec.load_panel_constants(dir_peds, 'pedestals_FM', tstamp)
    ped_fl   = uec.load_panel_constants(dir_peds, 'pedestals_FL', tstamp)
    ped_hl_h = uec.load_panel_constants(dir_peds, 'pedestals_AHL-H', tstamp)
    ped_ml_m = uec.load_panel_constants(dir_peds, 'pedestals_AML-M', tstamp)

    darks[0,:,:] = ped_fh
    darks[1,:,:] = ped_fm
    darks[2,:,:] = ped_fl
    darks[3,:,:] = ped_hl_h
    darks[4,:,:] = ped_ml_m

    #ped_hl_h = dark #[3,:,:]
    #darks = np.stack([ped_fh, ped_fm, ped_fl, ped_hl_h, ped_ml_m])

    logger.info(info_ndarr(ped_fh, '  ped_fh', last=10))
    logger.info(info_ndarr(darks, '  darks', last=10))

    #for i in range(5): sys.exit('TEST EXIT')

    #Save darks accounting offset whenever appropriate:
    for i in range(5):  #looping through darks measured in Jack's order
        fname = '%s_pedestals_%s.dat' % (prefix_peds, GAIN_MODES[i])
        #save_2darray_in_textfile(darks[i,:,:], fname, filemode, fmt_peds, umask=0o0, group=group)

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
import psana2.detector.UtilsGraphics as ug
        global ug
        rcslices = [eval('np.s_[%s]' % s) for s in sslice.split(',')]  # sslice="0:,0:" > list of slices for rows and cols
        gr.plt.close("all")
        fnameout='%s_plot_AML.png' % prefix_plots
        gm='AML'; titles=['M Gain','M Pedestal', 'L Gain', 'M-L Offset']
        plot_fit_results(0, fits_ml, fnameout, filemode, gm, titles, rcslices)

        fnameout='%s_plot_AHL.png' % prefix_plots
        gm='AHL'; titles=['H Gain','H Pedestal', 'L Gain', 'H-L Offset']
        plot_fit_results(1, fits_hl, fnameout, filemode, gm, titles, rcslices)

        #gr.plt.pause(5)


    if True: # evaluate pixel status

        kwa = {'myslice'   : eval('np.s_[%s]' % sslice),
               'databitw'  : databitw,
               'nsigm'     : nsigm,
               'prefix'    : prefix_plots if display else '',
               'neg_min'   : 1,
               'neg_max'   : 3,
               'offset_min': 0,
               'offset_max': databitw,
               'gain_min'  : 0.0,
               'gain_max'  : databitw/10,
               'chi2_min'  : 0.0,
               'chi2_max'  : None
              }

        status = ci_pixel_status(
          offset_ml_m,
          offset_ml_l,
          offset_hl_h,
          offset_hl_l,
          gain_ml_m,
          gain_ml_l,
          gain_hl_h,
          gain_hl_l,
          chi2_ml_m,
          chi2_ml_l,
          chi2_hl_h,
          chi2_hl_l,
          neg_ml,
          neg_hl,
          **kwa
        )
        #  myslice = np.s_[0:int(nr/2), 0:int(nc/2)], ASIC0 for 1-st epixhr debugging OR regular np.s_[0:, 0:]

        fname = '%s_status_ci.dat' % prefix_status
        save_2darray_in_textfile(status, fname, filemode, fmt_status, umask=0o0, group=group)

    if display:
        print('TO EXIT - close all graphical windows or wait for %d sec' % tsec_show_end)
        gr.plt.pause(tsec_show_end)
        #gr.show()  # mode='DO NOT HOLD'

    repoman.logfile_save()


def ci_pixel_status(
          offset_ml_m,
          offset_ml_l,
          offset_hl_h,
          offset_hl_l,
          gain_ml_m,
          gain_ml_l,
          gain_hl_h,
          gain_hl_l,
          chi2_ml_m,
          chi2_ml_l,
          chi2_hl_h,
          chi2_hl_l,
          neg_ml,
          neg_hl,
          **kwargs
        ):

    prefix     = kwargs.get('prefix', '')
    myslice    = kwargs.get('myslice', np.s_[0:, 0:])
    databitw   = kwargs.get('databitw', (1<<16)-1)
    neg_min    = kwargs.get('neg_min', 1)
    neg_max    = kwargs.get('neg_max', 3)
    offset_min = kwargs.get('offset_min', 0)
    offset_max = kwargs.get('offset_max', databitw)
    gain_min   = kwargs.get('gain_min', 0.0)
    gain_max   = kwargs.get('gain_max', databitw/10)
    chi2_min   = kwargs.get('chi2_min', 0.0)
    chi2_max   = kwargs.get('chi2_max', None)
    nsigm      = kwargs.get('nsigm', 8)

    logger.info('in ci_pixel_status for slice: %s' % str(myslice))
    nvals = 4
    logger.info(info_ndarr(neg_ml,      '  neg_ml', last=nvals))
    logger.info(info_ndarr(neg_hl,      '  neg_hl', last=nvals))
    logger.info(info_ndarr(offset_ml_m, '  offset_ml_m', last=nvals))
    logger.info(info_ndarr(gain_ml_m,   '  gain_ml_m', last=nvals))
    logger.info(info_ndarr(chi2_ml_m,   '  chi2_ml_m', last=nvals))

    shape = offset_ml_m.shape

    arr1 = np.ones(shape, dtype=np.uint64)

    stus_neg_ml_lo = np.select((neg_ml < neg_min,), (arr1,), 0)[myslice]
    stus_neg_hl_lo = np.select((neg_hl < neg_min,), (arr1,), 0)[myslice]
    stus_neg_ml_hi = np.select((neg_ml > neg_max,), (arr1,), 0)[myslice]
    stus_neg_hl_hi = np.select((neg_hl > neg_max,), (arr1,), 0)[myslice]

    sum_neg_ml_lo = stus_neg_ml_lo.sum()
    sum_neg_hl_lo = stus_neg_hl_lo.sum()
    sum_neg_ml_hi = stus_neg_ml_hi.sum()
    sum_neg_hl_hi = stus_neg_hl_hi.sum()

    arr_sta = np.zeros(shape, dtype=np.uint64)[myslice]
    arr_sta += stus_neg_ml_lo*(1<<16)  #  AML number of pulser periods < %d
    arr_sta += stus_neg_hl_lo*(1<<17)  #  AHL number of pulser periods < %d
    arr_sta += stus_neg_ml_hi*(1<<18)
    arr_sta += stus_neg_ml_hi*(1<<19)

    slsize = arr_sta.size # stus_neg_ml_lo.size

    f = '\n  %s\n  %s'
    s = info_ndarr(arr_sta, '\nSummary of the bad pixel status evaluation, pixel_status array', last=0)\
      + '\n  %20s: %8d / %d (%6.3f%%) pixels AML number of CI pulser periods < %d' % (oct(1<<16), sum_neg_ml_lo, slsize, 100*sum_neg_ml_lo/slsize, neg_min)\
      + '\n  %20s: %8d / %d (%6.3f%%) pixels AML number of CI pulser periods > %d' % (oct(1<<17), sum_neg_ml_hi, slsize, 100*sum_neg_ml_hi/slsize, neg_max)\
      + '\n  %20s: %8d / %d (%6.3f%%) pixels AHL number of CI pulser periods < %d' % (oct(1<<18), sum_neg_hl_lo, slsize, 100*sum_neg_hl_lo/slsize, neg_min)\
      + '\n  %20s: %8d / %d (%6.3f%%) pixels AHL number of CI pulser periods > %d' % (oct(1<<19), sum_neg_hl_hi, slsize, 100*sum_neg_hl_hi/slsize, neg_max)

    s += f % set_status_bits(arr_sta, offset_ml_m[myslice], title='offset_ml_m', vmin=offset_min, vmax=offset_max, nsigm=nsigm, bit_lo=1<<20, bit_hi=1<<21, prefix=prefix)
    s += f % set_status_bits(arr_sta, offset_ml_l[myslice], title='offset_ml_l', vmin=offset_min, vmax=offset_max, nsigm=nsigm, bit_lo=1<<22, bit_hi=1<<23, prefix=prefix)
    s += f % set_status_bits(arr_sta, offset_hl_h[myslice], title='offset_hl_h', vmin=offset_min, vmax=offset_max, nsigm=nsigm, bit_lo=1<<24, bit_hi=1<<25, prefix=prefix)
    s += f % set_status_bits(arr_sta, offset_hl_l[myslice], title='offset_hl_l', vmin=offset_min, vmax=offset_max, nsigm=nsigm, bit_lo=1<<26, bit_hi=1<<27, prefix=prefix)

    s += f % set_status_bits(arr_sta, gain_ml_m[myslice], title='gain_ml_m', vmin=gain_min, vmax=gain_max, nsigm=nsigm, bit_lo=1<<28, bit_hi=1<<29, prefix=prefix)
    s += f % set_status_bits(arr_sta, gain_ml_l[myslice], title='gain_ml_l', vmin=gain_min, vmax=gain_max, nsigm=nsigm, bit_lo=1<<30, bit_hi=1<<31, prefix=prefix)
    s += f % set_status_bits(arr_sta, gain_hl_h[myslice], title='gain_hl_h', vmin=gain_min, vmax=gain_max, nsigm=nsigm, bit_lo=1<<32, bit_hi=1<<33, prefix=prefix)
    s += f % set_status_bits(arr_sta, gain_hl_l[myslice], title='gain_hl_l', vmin=gain_min, vmax=gain_max, nsigm=nsigm, bit_lo=1<<34, bit_hi=1<<35, prefix=prefix)

    s += f % set_status_bits(arr_sta, chi2_ml_m[myslice], title='chi2_ml_m', vmin=chi2_min, vmax=chi2_max, nsigm=nsigm, bit_lo=1<<38, bit_hi=1<<39, prefix=prefix)
    s += f % set_status_bits(arr_sta, chi2_ml_l[myslice], title='chi2_ml_l', vmin=chi2_min, vmax=chi2_max, nsigm=nsigm, bit_lo=1<<40, bit_hi=1<<41, prefix=prefix)
    s += f % set_status_bits(arr_sta, chi2_hl_h[myslice], title='chi2_hl_h', vmin=chi2_min, vmax=chi2_max, nsigm=nsigm, bit_lo=1<<42, bit_hi=1<<43, prefix=prefix)
    s += f % set_status_bits(arr_sta, chi2_hl_l[myslice], title='chi2_hl_l', vmin=chi2_min, vmax=chi2_max, nsigm=nsigm, bit_lo=1<<44, bit_hi=1<<45, prefix=prefix)

    stus_bad_total = np.select((arr_sta>0,), (arr1[myslice],), 0)
    num_bad_pixels = stus_bad_total.sum()

    s += '\n    Any bad status bit: %8d / %d (%6.3f%%) pixels' % (num_bad_pixels, slsize, 100*num_bad_pixels/slsize)

    logger.info(s)

    return arr_sta


def find_outliers(arr, title='', vmin=None, vmax=None):
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
            (sum_lo, size, 100*sum_lo/size, title, 'unlimited' if vmin is None else '%.3f' % vmin)
    s_hi = '%8d / %d (%6.3f%%) pixels %s >= %s'%\
            (sum_hi, size, 100*sum_hi/size, title, 'unlimited' if vmax is None else '%.3f' % vmax)
    return bad_lo, bad_hi, arr1_lo, arr1_hi, s_lo, s_hi


def evaluate_pixel_status(arr, title='', vmin=None, vmax=None, nsigm=8, prefix=''):
    """vmin/vmax - absolutly allowed min/max of the value
    """
    bad_lo, bad_hi, arr1_lo, arr1_hi, s_lo, s_hi = find_outliers(arr, title=title, vmin=vmin, vmax=vmax)

    arr_sel = arr[np.logical_not(np.logical_or(bad_lo, bad_hi))]
    med = np.median(arr_sel)
    spr = np.median(np.absolute(arr_sel-med))  # axis=None, out=None, overwrite_input=False, keepdims=False

    _vmin = med - nsigm*spr if vmin is None else max(med - nsigm*spr, vmin)
    _vmax = med + nsigm*spr if vmax is None else min(med + nsigm*spr, vmax)

    if prefix: plot_array(arr, title=title, vmin=_vmin, vmax=_vmax, prefix=prefix)

    s_sel = '%s selected %d of %d pixels in' % (title, arr_sel.size, arr.size)\
          + ' range (%s, %s)' % (str(vmin), str(vmax))\
          + ' med: %.3f spr: %.3f' % (med, spr)

    s_range = u're-defined range for med \u00B1 %.1f*spr: (%.3f, %.3f)' % (nsigm, _vmin, _vmax)
    _, _, _arr1_lo, _arr1_hi, _s_lo, _s_hi = find_outliers(arr, title=title, vmin=_vmin, vmax=_vmax)

    gap = 13*' '
    logger.info('%s\n    %s\n  %s\n  %s\n%s%s\n%s%s\n  %s\n  %s' %\
        (20*'=', info_ndarr(arr, title, last=0), s_lo, s_hi, gap, s_sel, gap, s_range, _s_lo, _s_hi))

    return _arr1_lo, _arr1_hi, _s_lo, _s_hi


def set_status_bits(status, arr, title='', vmin=None, vmax=None, nsigm=8, bit_lo=1<<0, bit_hi=1<<1, prefix=''):
    arr1_lo, arr1_hi, s_lo, s_hi = evaluate_pixel_status(arr, title=title, vmin=vmin, vmax=vmax, nsigm=nsigm, prefix=prefix)
    status += arr1_lo * bit_lo
    status += arr1_hi * bit_hi
    s_lo = '%20s: %s' % (oct(bit_lo), s_lo)
    s_hi = '%20s: %s' % (oct(bit_hi), s_hi)
    return s_lo, s_hi

# EOF
