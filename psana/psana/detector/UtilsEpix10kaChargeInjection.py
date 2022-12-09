#!/usr/bin/env python

import sys
import psana.pyalgos.generic.Graphics as gr
from psana.detector.UtilsLogging import logging, DICT_NAME_TO_LEVEL  # , init_stream_handler
import psana.detector.UtilsEpix10kaCalib as uec
logger = logging.getLogger(__name__)

SCRNAME = sys.argv[0].rsplit('/')[-1]

create_directory = uec.create_directory


def get_panel_id(panel_ids, idx=0):
    panel_id = panel_ids[idx] if panel_ids is not None and idx is not None else None
    assert panel_id is not None, 'get_panel_id: panel_id is None for idx=%s' % str(idx)
    return panel_id


def dir_names(dirrepo, panel_id):
    """Defines structure of subdirectories in calibration repository.
    """
    dir_panel  = '%s/%s' % (dirrepo, panel_id)
    dir_offset = '%s/offset'    % dir_panel
    dir_peds   = '%s/pedestals' % dir_panel
    dir_plots  = '%s/plots'     % dir_panel
    dir_work   = '%s/work'      % dir_panel
    dir_gain   = '%s/gain'      % dir_panel
    dir_rms    = '%s/rms'       % dir_panel
    dir_status = '%s/status'    % dir_panel
    return dir_panel, dir_offset, dir_peds, dir_plots, dir_work, dir_gain, dir_rms, dir_status


def file_name_prefix(panel_id, tstamp, exp, irun):
    logger.info('FNAME_PANEL_ID_ALIASES % s' % uec.FNAME_PANEL_ID_ALIASES)
    panel_alias = uec.alias_for_id(panel_id, fname=uec.FNAME_PANEL_ID_ALIASES)
    return 'epix10ka_%s_%s_%s_r%04d' % (panel_alias, tstamp, exp, irun), panel_alias


def path_prefixes(fname_prefix, dir_offset, dir_peds, dir_plots, dir_gain, dir_rms, dir_status):
    prefix_offset= '%s/%s' % (dir_offset, fname_prefix)
    prefix_peds  = '%s/%s' % (dir_peds,   fname_prefix)
    prefix_plots = '%s/%s' % (dir_plots,  fname_prefix)
    prefix_gain  = '%s/%s' % (dir_gain,   fname_prefix)
    prefix_rms   = '%s/%s' % (dir_rms,    fname_prefix)
    prefix_status= '%s/%s' % (dir_status, fname_prefix)
    return prefix_offset, prefix_peds, prefix_plots, prefix_gain, prefix_rms, prefix_status


def file_name_npz(dir_work, fname_prefix, nspace):
    return '%s/%s_sp%02d_df.npz' % (dir_work, fname_prefix, nspace)



#def dir_merge(dirrepo):
#    return '%s/merge_tmp' % dirrepo


#def fname_prefix_merge(dmerge, detname, tstamp, exp, irun):
#    return '%s/%s-%s-%s-r%04d' % (dmerge, detname, tstamp, exp, irun)










def charge_injection(**kwa):

#    exp        = kwa.get('exp', None)
#    run        = kwa.get('run', None)
#    dsnamex    = kwa.get('dsnamex', None)

    str_dskwargs = kwa.get('dskwargs', None)
    detname    = kwa.get('det', None)
    idx        = kwa.get('idx', 0)
    nbs        = kwa.get('nbs', 4600)
    nspace     = kwa.get('nspace', 7)
    dirrepo    = kwa.get('dirrepo', uec.DIR_REPO_EPIX10KA)
    display    = kwa.get('display', True)
    fmt_offset = kwa.get('fmt_offset', '%.6f')
    fmt_peds   = kwa.get('fmt_peds',   '%.3f')
    fmt_rms    = kwa.get('fmt_rms',    '%.3f')
    fmt_status = kwa.get('fmt_status', '%4i')
    fmt_gain   = kwa.get('fmt_gain',   '%.6f')
    fmt_chi2   = kwa.get('fmt_chi2',   '%.3f')
    savechi2   = kwa.get('savechi2', False)
    dopeds     = kwa.get('dopeds', True)
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

    logger.setLevel(DICT_NAME_TO_LEVEL[logmode])

    uec.save_log_record_at_start(dirrepo, SCRNAME, dirmode, filemode, logmode, group=group)

    logger.info('\n  SCRNAME : %s\n  dskwargs: %s\n  detector: %s' % (SCRNAME, str_dskwargs, detname))

    #dskwargs = data_source_kwargs(**kwa)
    dskwargs = uec.datasource_kwargs_from_string(str_dskwargs)
    try: ds = uec.DataSource(**dskwargs)
    except Exception as err:
        logger.error('DataSource(**dskwargs) does not work for **dskwargs: %s\n    %s' % (dskwargs, err))
        sys.exit('EXIT - requested DataSource does not exist or is not accessible.')

    logger.debug('ds.runnum_list = %s' % str(ds.runnum_list))
    logger.debug('ds.detectors = %s' % str(ds.detectors))
    xtc_files = getattr(ds, 'xtc_files', None)
    logger.info('ds.xtc_files:\n  %s' % ('None' if xtc_files is None else '\n  '.join(xtc_files)))

    #irun = int(run.split(',',1)[0].split('-',1)[0]) # int first run number from str of run(s)
    #dsname = str_dsname(exp, run, dsnamex)
    #_name = sys._getframe().f_code.co_name
    #cpdic = get_config_info_for_dataset_detname(dsname, detname, idx)

    cpdic = uec.get_config_info_for_dataset_detname(**kwa)

    logger.info('config_info:%s' % uec.info_dict(cpdic))  # fmt=fmt, sep=sep+sepnext)

    tstamp    = cpdic.get('tstamp', None)
    panel_ids = cpdic.get('panel_ids', None)
    #expnum    = cpdic.get('expnum', None)
    shape     = cpdic.get('shape', None)
    irun      = cpdic.get('runnum', None)
    dsname    = dskwargs
    ny,nx     = shape

    if display:
        fig2, axim2, axcb2 = gr.fig_img_cbar_axes()
        gr.move_fig(fig2, 500, 10)
        gr.plt.ion() # do not hold control on plt.show()

    selpix = None
    pixrow, pixcol = None, None
    if pixrc is not None:
      try:
        pixrow, pixcol = [int(v) for v in pixrc.split(',')]
        logger.info('use pixel row:%d col:%d for graphics' % (pixrow, pixcol))
      except:
        logger.error('vaiable pixrc="%s" can not be converted to pixel row,col' % str(pixrc))
        sys.exit()


    panel_id = get_panel_id(panel_ids, idx)

    dir_panel, dir_offset, dir_peds, dir_plots, dir_work, dir_gain, dir_rms, dir_status = dir_names(dirrepo, panel_id)
    fname_prefix, panel_alias = file_name_prefix(panel_id, tstamp, exp, irun)
    prefix_offset, prefix_peds, prefix_plots, prefix_gain, prefix_rms, prefix_status =\
            path_prefixes(fname_prefix, dir_offset, dir_peds, dir_plots, dir_gain, dir_rms, dir_status)
    fname_work = file_name_npz(dir_work, fname_prefix, nspace)

    create_directory(dir_panel,  mode=dirmode, group=group)
    create_directory(dir_offset, mode=dirmode, group=group)
    create_directory(dir_peds,   mode=dirmode, group=group)
    create_directory(dir_plots,  mode=dirmode, group=group)
    create_directory(dir_work,   mode=dirmode, group=group)
    create_directory(dir_gain,   mode=dirmode, group=group)
    create_directory(dir_rms,    mode=dirmode, group=group)
    create_directory(dir_status, mode=dirmode, group=group)







    sys.exit('TEST EXIT')









    chi2_ml=np.zeros((ny,nx,2))
    chi2_hl=np.zeros((ny,nx,2))
    nsp_ml=np.zeros((ny,nx),dtype=np.int16)
    nsp_hl=np.zeros((ny,nx),dtype=np.int16)

    try:
        npz=np.load(fname_work)
        logger.info('Charge-injection data loaded from file: %s' % fname_work)
        logger.info('SKIP CALIBRATION CYCLES')

        darks=npz['darks']
        fits_ml=npz['fits_ml']
        fits_hl=npz['fits_hl']

    except IOError:
        logger.info('Unavailable charge-injection data file: %s' % fname_work)
        logger.info('BEGIN CALIBRATION CYCLES')

        darks=np.zeros((7,ny,nx))
        fits_ml=np.zeros((ny,nx,2,2))
        fits_hl=np.zeros((ny,nx,2,2))

        ds = DataSource(dsname)
        det = Detector(detname)
        cd = Detector('ControlData')

        nstep_tot = -1
        for orun in ds.runs():
          print('==== run:', orun.run())

          for nstep_run, step in enumerate(orun.steps()):
            nstep_tot += 1
            logger.info('=============== calibcycle %02d ===============' % nstep_tot)

            nstep = step_counter(cd, det, nstep_tot, nstep_run, nspace)
            if nstep is None: continue

            if nstep_tot<skipncc:
                logger.info('skip %d consecutive calib-cycles' % skipncc)
                continue

            elif nstep_tot>=ccmax:
                logger.info('total number of calib-cycles %d exceeds ccmax %d' % (nstep_tot, ccmax))
                break

            elif ccnum is not None:
                # process calibcycle ccnum ONLY if ccnum is specified
                if   nstep < ccnum: continue
                elif nstep > ccnum: break

            mode = find_gain_mode(det, data=None).upper()

            if mode in GAIN_MODES_IN and nstep < len(GAIN_MODES_IN):
                mode_in_meta = GAIN_MODES_IN[nstep]
                logger.info('========== calibcycle %d: dark run processing for gain mode in configuration %s and metadata %s'\
                            %(nstep, mode, mode_in_meta))
                if mode != mode_in_meta:
                  logger.warning('INCONSISTENT GAIN MODES IN CONFIGURATION AND METADATA')
                  if not errskip: sys.exit()
                  logger.warning('FLAG ERRSKIP IS %s - keep processing next calib-cycle' % errskip)
                  continue

            figprefix = '%s-%s-seg%02d-cc%03d-%s'%\
                        (prefix_plots, detname.replace(':','-').replace('.','-'), idx, nstep, mode)

            nrec,nevt = -1,0
            #First 5 Calib Cycles correspond to darks:
            if dopeds and nstep<5:
                msg = 'DARK Calib Cycle %d ' % nstep
                block=np.zeros((nbs,ny,nx),dtype=np.int16)

                for nevt,evt in enumerate(step.events()):
                    raw = det.raw(evt)
                    do_print = selected_record(nevt)
                    if raw is None: #skip empty frames
                        logger.warning('Ev:%04d rec:%04d panel:%02d raw=None' % (nevt,nrec,idx))
                        msg += 'none'
                        continue
                    if nrec>nbs-2:
                        break
                    else:
                        nrec += 1
                        if raw.ndim > 2: raw=raw[idx,:]
                        if do_print: logger.info(info_ndarr(raw & M14, 'Ev:%04d rec:%04d panel:%02d raw & M14' % (nevt,nrec,idx)))
                        if display and nevt<3:
                            imsh, cbar = imshow_cbar(fig2, axim2, axcb2, raw, amin=None, amax=None, extent=None,\
                                                     interpolation='nearest', aspect='auto', origin='upper',\
                                                     orientation='vertical', cmap='inferno')
                            fig2.canvas.set_window_title('Run:%d calib-cycle:%d mode:%s panel:%02d' % (orun.run(), nstep, mode, idx))
                            fname = '%s-ev%02d-img-dark' % (figprefix, nevt)
                            axim2.set_title(fname.rsplit('/',1)[-1], fontsize=6)
                            fig2.savefig(fname+'.png')
                            logger.info('saved: %s' % fname+'.png')

                        block[nrec]=raw & M14
                        if nrec%200==0: msg += '.%s' % find_gain_mode(det, raw)

                print_statistics(nevt, nrec)

                darks[nstep,:,:], nda_rms, nda_status = proc_dark_block(block[:nrec,:,:], **kwa)
                logger.debug(msg)

                fname = '%s_rms_%s.dat' % (prefix_rms, GAIN_MODES[nstep])
                save_2darray_in_textfile(nda_rms, fname, filemode, fmt_rms, umask=0o0, group=group)

                fname = '%s_status_%s.dat' % (prefix_status, GAIN_MODES[nstep])
                save_2darray_in_textfile(nda_status, fname, filemode, fmt_status, umask=0o0, group=group)

            ####################
            elif not dooffs:
                logger.debug(info_ndarr(darks, 'darks'))
                if nstep>4: break
                continue
            ####################

            #Next nspace**2 Calib Cycles correspond to pulsing in Auto Medium-to-Low
            elif nstep>4 and nstep<5+nspace**2:
                msg = ' AML %2d/%2d '%(nstep-5+1,nspace**2)

                istep=nstep-5
                jy=istep//nspace
                jx=istep%nspace

                if pixrc is not None:
                    selpix = selected_pixel(pixrow, pixcol, jy, jx, ny, nx, nspace)
                    if selpix is None:
                        logger.info(msg + ' skip, due to pixrc=%s'%pixrc)
                        continue
                    else:
                        logger.info(msg + ' process selected pixel:%s' % str(selpix))

                fid_old=None
                block=np.zeros((nbs,ny,nx),dtype=np.int16)
                evnum=np.zeros((nbs,),dtype=np.int16)
                for nevt,evt in enumerate(step.events()):   #read all frames
                    raw = det.raw(evt)
                    do_print = selected_record(nevt)
                    if raw is None:
                        logger.warning('Ev:%04d rec:%04d panel:%02d AML raw=None' % (nevt,nrec,idx))
                        msg += 'none'
                        continue
                    if nrec>nbs-2:
                        break
                    else:
                        #---- 2021-06-10: check fiducial for consecutive events
                        fid = evt.get(EventId).fiducials()
                        if fid_old is not None:
                            dfid = fid-fid_old
                            if dfid != 3:
                                logger.warning('TIME SYSTEM FAULT dfid!=3: Ev:%04d rec:%04d panel:%02d AML raw=None fiducials:%7d dfid:%d'%\
                                            (nevt,nrec,idx,fid,dfid))
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
                        if do_print: logger.info(info_ndarr(raw, 'Ev:%04d rec:%04d panel:%02d AML raw' % (nevt,nrec,idx)))
                        block[nrec]=raw
                        evnum[nrec]=nevt
                        if nevt%200==0: msg+='.'

                if display:
                    #imsh, cbar = imshow_cbar(fig2, axim2, axcb2, block[nrec][:100,:100], amin=None, amax=None, extent=None,\
                    imsh, cbar = imshow_cbar(fig2, axim2, axcb2, block[nrec], amin=None, amax=None, extent=None,\
                                             interpolation='nearest', aspect='auto', origin='upper',\
                                             orientation='vertical', cmap='inferno')
                    fig2.canvas.set_window_title('Run:%d calib-cycle:%d events:%d' % (orun.run(), nstep, evnum[nrec])) #, **kwargs)
                    fname = '%s-img-charge' % figprefix
                    axim2.set_title(fname.rsplit('/',1)[-1], fontsize=6)
                    fig2.savefig(fname+'.png')
                    logger.info('saved: %s' % fname+'.png')

                print_statistics(nevt, nrec)

                block=block[:nrec,jy:ny:nspace,jx:nx:nspace]         # select only pulsed pixels
                evnum=evnum[:nrec]                                   # list of non-empty events
                fits0,nsp0,msgf,chi2=fit(block,evnum,display,figprefix,ixoff,nperiods,savechi2,selpix) # fit offset, gain
                fits_ml[jy:ny:nspace,jx:nx:nspace]=fits0             # collect results
                nsp_ml[jy:ny:nspace,jx:nx:nspace]=nsp0               # collect switching points
                if savechi2: chi2_ml[jy:ny:nspace,jx:nx:nspace]=chi2 # collect chi2/dof
                s = '\n  block fit results AML'\
                  + info_ndarr(fits0[:,:,0,0],'\n  M gain',   last=5)\
                  + info_ndarr(fits0[:,:,1,0],'\n  L gain',   last=5)\
                  + info_ndarr(fits0[:,:,0,1],'\n  M offset', last=5)\
                  + info_ndarr(fits0[:,:,1,1],'\n  L offset', last=5)

                logger.info(msg + msgf + s)

            #Next nspace**2 Calib Cycles correspond to pulsing in Auto High-to-Low
            elif nstep>4+nspace**2 and nstep<5+2*nspace**2:
                msg = ' AHL %2d/%2d '%(nstep-5-nspace**2+1,nspace**2)

                istep=nstep-5-nspace**2
                jy=istep//nspace
                jx=istep%nspace

                if pixrc is not None:
                    selpix = selected_pixel(pixrow, pixcol, jy, jx, ny, nx, nspace)
                    if selpix is None:
                        logger.info(msg + ' skip, due to pixrc=%s'%pixrc)
                        continue

                fid_old=None
                block=np.zeros((nbs,ny,nx),dtype=np.int16)
                evnum=np.zeros((nbs,),dtype=np.int16)
                for nevt,evt in enumerate(step.events()):   #read all frames
                    raw = det.raw(evt)
                    do_print = selected_record(nevt)
                    if raw is None:
                        logger.warning('Ev:%04d rec:%04d panel:%02d AHL raw=None' % (nevt,nrec,idx))
                        msg+='None'
                        continue
                    if nrec>nbs-2:
                        break
                    else:
                        #---- 2021-06-10: check fiducial for consecutive events
                        fid = evt.get(EventId).fiducials()
                        if fid_old is not None:
                            dfid = fid-fid_old
                            if dfid != 3:
                                logger.warning('TIME SYSTEM FAULT dfid!=3: Ev:%04d rec:%04d panel:%02d AML raw=None fiducials:%7d dfid:%d'%\
                                            (nevt,nrec,idx,fid,dfid))
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
                        if do_print: logger.info(info_ndarr(raw, 'Ev:%04d rec:%04d panel:%02d AHL raw' % (nevt,nrec,idx)))
                        block[nrec]=raw
                        evnum[nrec]=nevt
                        if nevt%200==0: msg+='.'

                if display:
                    #imsh, cbar = imshow_cbar(fig2, axim2, axcb2, block[nrec][:100,:100], amin=None, amax=None, extent=None,\
                    imsh, cbar = imshow_cbar(fig2, axim2, axcb2, block[nrec], amin=None, amax=None, extent=None,\
                                             interpolation='nearest', aspect='auto', origin='upper',\
                                             orientation='vertical', cmap='inferno')
                    fig2.canvas.set_window_title('Run:%d calib-cycle:%d events:%d' % (orun.run(), nstep, evnum[nrec])) #, **kwargs)
                    fname = '%s-img-charge' % figprefix
                    axim2.set_title(fname.rsplit('/',1)[-1], fontsize=6)
                    fig2.savefig(fname+'.png')
                    logger.info('saved: %s' % fname+'.png')

                print_statistics(nevt, nrec)

                block=block[:nrec,jy:ny:nspace,jx:nx:nspace]       # select only pulsed pixels
                evnum=evnum[:nrec]                                 # list of non-empty events
                fits0,nsp0,msgf,chi2=fit(block,evnum,display,figprefix,ixoff,nperiods,savechi2,selpix) # fit offset, gain
                fits_hl[jy:ny:nspace,jx:nx:nspace]=fits0           # collect results
                nsp_hl[jy:ny:nspace,jx:nx:nspace]=nsp0
                if savechi2: chi2_hl[jy:ny:nspace,jx:nx:nspace]=chi2 # collect chi2/dof
                s = '\n  block fit results AHL'\
                  + info_ndarr(fits0[:,:,0,0],'\n  H gain',   last=5)\
                  + info_ndarr(fits0[:,:,1,0],'\n  L gain',   last=5)\
                  + info_ndarr(fits0[:,:,0,1],'\n  H offset', last=5)\
                  + info_ndarr(fits0[:,:,1,1],'\n  L offset', last=5)
                logger.info(msg + msgf + s)

            elif nstep>=5+2*nspace**2:
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
        plt.close("all")
        fnameout='%s_plot_AML.png' % prefix_plots
        gm='AML'; titles=['M Gain','M Pedestal', 'L Gain', 'M-L Offset']
        plot_fit_results(0, fits_ml, fnameout, filemode, gm, titles)

        fnameout='%s_plot_AHL.png' % prefix_plots
        gm='AHL'; titles=['H Gain','H Pedestal', 'L Gain', 'H-L Offset']
        plot_fit_results(1, fits_hl, fnameout, filemode, gm, titles)

        plt.pause(5)


# EOF
