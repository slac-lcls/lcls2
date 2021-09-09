
"""Class :py:class:`DMQWControl` is a Data Manager QWidget with control fields
==============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/DMQWControl.py

    from psana.graphqt.DMQWControl import DMQWControl
    w = DMQWControl()

Created on 2021-08-10 by Mikhail Dubrovin
"""

from psana.graphqt.CMWControlBase import * # CMWControlBase, QApplication, ..., icon, style, cp, logging, os, sys

logger = logging.getLogger(__name__)

COMMAND_SET_ENV_LCLS1 = '. /cds/sw/ds/ana/conda1/manage/bin/psconda.sh; echo "PATH: $PATH"; echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"; '
ENV1 = {} #'PATH':'/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin'}


def detname_contains_pattern(detname, pattern):
    r = pattern in detname
    if not r: logger.warning('DETECTOR NAME "%s" DOES NOT CONTAIN "%s"' % (detname, pattern))
    return r

def is_epix10ka(detname): return detname_contains_pattern(detname, 'Epix10')

def is_jungfrau(detname): return detname_contains_pattern(detname, 'Jungfrau')

def is_area_detector(detname): 
    logger.warning('TBD: is_area_detector for DETECTOR NAME "%s"' % detname)
    return True


class DMQWControl(CMWControlBase):
    """QWidget for Data Manager control fields"""

    det_list0 = ['select','run mast be','selected first']
    cmd_list = ['detnames', 'datinfo']
    cmd_list_lcls1 = [\
     'detnames',
     'datinfo',
     'event_keys',
     'DetInfo',
     'det_calib_ave_and_max',
     'det_ndarr_raw_proc',
     'epix10ka_id',
     'epix10ka_offset_calibration',
     'epix10ka_pedestals_calibration',
     'epix10ka_deploy_constants',
     'epix10ka_test_calibcycles',
     'jungfrau_id',
     'jungfrau_dark_proc',
     'jungfrau_deploy_constants',
    ]
    cmd_list_lcls2 = [\
     'detnames',
     'datinfo',
     'det_dark_proc',
     'epix10ka_pedestals_calibration',
     'epix10ka_deploy_constants',
    ]
    lclsv_list = ['LCLS','LCLS2']
    instr_lcls1 = ['AMO','CXI','XPP','SXR','MEC','DET','MFX','MOB','USR','MON','DIA']
    instr_lcls2 = ['TST','TMO','RIX','UED']
    instr_exp_is_changed = pyqtSignal('QString', 'QString')

    def __init__(self, **kwa):

        #parent = kwa.get('parent', None)

        CMWControlBase.__init__(self, **kwa)
        cp.dmqwcontrol = self
        self.osp = None
        self.dt_msec = 1000

        expname = kwa.get('expname', expname_def())

        self.lab_exp = QLabel('exp:')
        self.lab_cmd = QLabel('cmd:')
        self.lab_det = QLabel('det:')
        self.but_exp = QPushButton(expname)
        self.but_start = QPushButton('start')
        self.but_stop = QPushButton('stop')
        self.cmb_det = QComboBox()
        #self.cmb_det.addItems(self.det_list)
        self.set_cmb_det()
        self.cmb_lclsv = QComboBox()
        self.cmb_lclsv.addItems(self.lclsv_list)
        self.cmb_cmd = QComboBox()
        self.set_lclsv_for_experiment(expname)

        self.box2 = QHBoxLayout()
        self.box2.addStretch(1)
        self.box2.addWidget(self.lab_det)
        self.box2.addWidget(self.cmb_det)
        self.box = QHBoxLayout()
        self.box.addWidget(self.lab_exp)
        self.box.addWidget(self.but_exp)
        #self.box.addStretch(1)
        self.box.addWidget(self.cmb_lclsv)
        self.box.addWidget(self.lab_cmd)
        self.box.addWidget(self.cmb_cmd)
        self.box.addWidget(self.but_start)
        self.box.addWidget(self.but_stop)
        self.box.addWidget(self.but_save)
        self.box.addWidget(self.but_view)
        self.box.addWidget(self.but_tabs)
        self.boxv = QVBoxLayout()
        self.boxv.addLayout(self.box)
        self.boxv.addLayout(self.box2)
        self.setLayout(self.boxv)
 
        self.but_exp.clicked.connect(self.on_but_exp)
        self.but_start.clicked.connect(self.on_but_start)
        self.but_stop.clicked.connect(self.on_but_stop)
        self.cmb_cmd.currentIndexChanged[int].connect(self.on_cmb_cmd)
        self.cmb_det.currentIndexChanged[int].connect(self.on_cmb_det)
        #self.cmb_lclsv.currentIndexChanged[int].connect(self.on_cmb_lclsv)

        self.set_tool_tips()
        self.set_style()

#        if cp.dmqwmain is not None:
#            global full_path_for_item
#            from psana.graphqt.FSTree import full_path_for_item
#            cp.dmqwmain.wlist.connect_item_selected_to(self.on_item_selected)
#            cp.dmqwmain.wlist.clicked[QModelIndex].connect(self.on_click)


    def set_cmb_cmd(self):
        self.cmb_cmd.clear()
        self.cmb_cmd.addItems(self.cmd_list)
        self.cmb_cmd.setCurrentIndex(self.cmd_list.index('detnames'))


    def is_lcls1(self, expname=None):
        """use instrument part of the passed experiment name or field content if expname=None
        """
        exp = expname if expname is not None else self.but_exp.text()
        resp = expname[:3].upper() in self.instr_lcls1
        logger.debug('instrument/experiment: %s belonds to LCLS%s' % (exp, '' if resp else '2'))
        return resp


    def is_lcls2(self, expname, runnum):
        """checks db file name extension for *.xtc2
        """
        if cp.dmqwmain is None: return None
        return cp.dmqwmain.is_lcls2(expname, runnum)


    def set_lclsv(self, is_lcls1):
        lclsv = 'LCLS' if is_lcls1 else 'LCLS2'
        self.cmd_list = self.cmd_list_lcls1 if is_lcls1 else self.cmd_list_lcls2
        self.cmb_lclsv.setCurrentIndex(self.lclsv_list.index(lclsv))
        self.cmb_lclsv.setToolTip('%s data type is assumed from\ninstrument name' % lclsv)
        self.set_cmb_cmd()


    def set_lclsv_for_instrument(self, instr):
        self.set_lclsv(self.is_lcls1(instr))
        #self.set_lclsv(instr.upper() in self.instr_lcls1)


    def set_lclsv_for_experiment(self, expname):
        self.set_lclsv_for_instrument(expname[:3])


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        self.but_exp.setToolTip('Select experiment')
        self.but_start.setToolTip('Start command execution in subprocess')
        self.but_stop.setToolTip('Stop subprocess')
        self.cmb_cmd.setToolTip('Select command to execute for selected run(s)')
        self.cmb_det.setToolTip('Select detector name for command')


    def set_style(self):
        CMWControlBase.set_style(self)
        self.lab_exp.setStyleSheet(style.styleLabel)
        self.lab_cmd.setStyleSheet(style.styleLabel)
        self.lab_det.setStyleSheet(style.styleLabel)
        self.lab_exp.setFixedWidth(28)
        self.lab_cmd.setFixedWidth(28)
        self.lab_det.setFixedWidth(28)
        self.but_exp.setFixedWidth(80)
        self.cmb_lclsv.setFixedWidth(65)
        self.cmb_cmd.setFixedWidth(120)
        self.cmb_det.setFixedWidth(200)
        self.but_start.setFixedWidth(35)
        self.but_stop.setFixedWidth(35)
        #self.but_buts.setStyleSheet(style.styleButton)
        #self.but_tabs.setVisible(True)
        self.layout().setContentsMargins(5,0,5,0)
        self.cmb_lclsv.setEnabled(False)
        self.wfnm.setVisible(False)
        self.wfnm.setEnabled(False)
        self.but_save.setVisible(False)
        self.but_view.setVisible(False)
        self.but_save.setEnabled(False)
        self.but_view.setEnabled(False)


    def current_detnames(self):
        count = self.cmb_det.count()
        lst = [self.cmb_det.itemText(i) for i in range(count)]
        logger.debug('count %d current_detnames %s' % (count, str(lst)))
        return lst


    def on_cmb_det(self, ind):
        logger.debug('on_cmb_det selected index %d: %s' % (ind, self.current_detnames()[ind] if ind>0 else 'None'))


    def on_cmb_cmd(self, ind):
        logger.debug('on_cmb_cmd selected index %d: %s' % (ind, self.cmd_list[ind]))


    def on_cmb_lclsv(self, ind):
        txt = self.lclsv_list[ind]
        logger.debug('on_cmb_lclsv selected index %d: %s' % (ind, txt))
        self.set_lclsv(txt=='LCLS')


    def on_but_stop(self):
        logger.debug('on_but_stop')
        self.force_stop = True


    def on_but_start(self, **kwa):
        events   = kwa.get('events', 1000)
        evskip   = kwa.get('evskip', 0)
        calibdir = kwa.get('calibdir', None)
        loglevel = kwa.get('loglevel', 'INFO')

        command = self.cmb_cmd.currentText()
        expname, runnum = cp.exp_name.value(), cp.last_selected_run
        logger.info('on_but_start command:%s exp:%s run:%s' % (command, expname, str(runnum)))
        #if cp.last_selected_run is None:
        #    logger.info('RUN IS NOT SELECTED - click/select run first')
        #    return

        self.force_stop = False
        if cp.dmqwmain is None: return

        dsname = cp.dmqwmain.fname_info(expname, runnum)
        cp.dmqwmain.append_info(dsname, cp.dmqwmain.fname_info(expname, runnum))
        #cp.dmqwmain.dump_info_exp_run(expname, runnum)

        selruns = [int(i.accessibleText()) for i in cp.dmqwmain.wlist.selected_items()]
        s = 'Exp: %s %d selected runs:\n  %s\nLast selected run: %s'%\
               (expname, len(selruns), str(selruns), str(cp.last_selected_run))
        cp.dmqwmain.append_info(s)

        if runnum is None:
            logger.warning('RUN IS NOT SELECTED - command terminated')
            return

        is_lcls1 = self.is_lcls1(expname)
        is_lcls2 = not is_lcls1
        #is_lcls2 = self.is_lcls2(expname, runnum)
        s = 'dataset exp=%s,run=%d is_lcls2:%s' % (expname, runnum, is_lcls2)
        cp.dmqwmain.append_info(s)

        detname = self.cmb_det.currentText()

        if is_lcls2:
          cmd = ''
          if command == 'detnames':
            cmd = 'detnames exp=%s,run=%d -r' % (expname, runnum)

          elif command == 'datinfo':
            cmd = 'datinfo -e %s -r %d -t D' % (expname, runnum)
            if detname != 'select': cmd += ' -d %s' % detname

          elif command == 'det_dark_proc':
            cmd = 'det_dark_proc -e %s -r %d -d %s' % (expname, runnum, detname)\
                + '   !!! other options: -o ./work -L DEBUG'

          elif command == 'epix10ka_pedestals_calibration':
            cmd = 'epix10ka_pedestals_calibration -e %s -r %d -d %s' % (expname, runnum, detname)\
                + '   !!! other options: -o ./work -L DEBUG'

          elif command == 'epix10ka_deploy_constants':
            cmd = 'epix10ka_deploy_constants -e %s -r %d -d %s' % (expname, runnum, detname)\
                + '   !!! other options: -o ./work -D -c ./calib'

          else:
            cp.dmqwmain.append_info('LCLS2 COMMAND "%s" IS NOT IMPLEMENTED...' % command)
            return

          cmd = qwu.edit_and_confirm_or_cancel_dialog_box(parent=self, text=cmd, title='Edit and confirm or cancel command')
          if cmd is None:
            logger.info('command is cancelled')
            return

          self.subprocess_command(cmd, shell=True)

        else: # LCLS1
          cmd_pref = COMMAND_SET_ENV_LCLS1
          detname = detname.replace('-','.').replace('|',':')
          if command == 'detnames':
            cmd = 'detnames exp=%s:run=%d' % (expname, runnum)

          elif command == 'event_keys':
            cmd = 'event_keys -d exp=%s:run=%d -m3' % (expname, runnum)

          elif command == 'DetInfo':
            cmd = 'event_keys -d exp=%s:run=%d -pDetInfo -n10' % (expname, runnum)

          elif command == 'datinfo':
            if detname == self.det_list0[0]:
                logger.warning('PLEASE SELECT THE DETECTOR NAME')
                return
            cmd = 'datinfo -e %s -r %d -d %s' % (expname, runnum, detname)

          elif command == 'det_calib_ave_and_max':
            if not is_area_detector(detname): return
            #det_calib_ave_and_max <dataset-name> <detector-name> <number-of-events> <number-events-to-skip> <calib-dir> <log-level-str>
            cmd = 'det_calib_ave_and_max exp=%s:run=%d %s %d %d %s %s'%\
                   (expname, runnum, detname, events, evskip, str(calibdir), loglevel)

          elif command == 'det_ndarr_raw_proc':
            if not is_area_detector(detname): return
            #det_ndarr_raw_proc -d <dataset> [-s <source>] [-f <file-name-template>] [-n <events-collect>] [-m <events-skip>] 
            cmd = 'det_ndarr_raw_proc -d exp=%s:run=%d -s %s -n %d -m %d -f nda-#exp-#run-#src-#type.txt'%\
                   (expname, runnum, detname, events, evskip)

          elif command == 'det_calib_ave_and_max':
            if not is_area_detector(detname): return
            #det_calib_ave_and_max <dataset-name> <detector-name> <number-of-events> <number-events-to-skip> <calib-dir> <log-level-str>
            cmd = 'det_calib_ave_and_max exp=%s:run=%d %s %d %d %s %s'%\
                   (expname, runnum, detname, events, evskip, str(calibdir), loglevel)

          elif command == 'epix10ka_id':
            if not is_epix10ka(detname): return
            cmd = 'epix10ka_id exp=%s:run=%d %s' % (expname, runnum, detname)

          elif command == 'jungfrau_id':
            if not is_jungfrau(detname): return
            cmd = 'jungfrau_id exp=%s:run=%d %s' % (expname, runnum, detname)

          elif command == 'epix10ka_offset_calibration':
            cmd = 'epix10ka_offset_calibration -e %s -r %d -d %s' % (expname, runnum, detname)\
                + '    !!! other options: -i0 -o ./work -p -P -O -s 60'

          elif command == 'epix10ka_pedestals_calibration':
            cmd = 'epix10ka_pedestals_calibration -e %s -r %d -d %s' % (expname, runnum, detname)\
                + '    !!! other options: -c1 -i15 -o ./work'

          elif command == 'epix10ka_deploy_constants':
            cmd = 'epix10ka_deploy_constants -e %s -r %d -d %s -D' % (expname, runnum, detname)\
                + '    !!! other options: -t 396 -o ./work -c ./calib -L DEBUG --proc=g --low=0.25 --medium=1 --high=1'

          elif command == 'epix10ka_test_calibcycles':
            cmd = 'epix10ka_test_calibcycles exp=%s:run=%d -e -p -s --detname %s -f ofname.txt' % (expname, runnum, detname)

          elif command == 'jungfrau_dark_proc':
            cmd = 'jungfrau_dark_proc -d exp=%s:run=%d:smd -s %s' % (expname, runnum, detname)\
                + '    !!! other options: -I 1 --evcode 162'

          elif command == 'jungfrau_deploy_constants':
            cmd = 'jungfrau_deploy_constants -e %s -r %d -d %s -D' % (expname, runnum, detname)\
                + '   !!! other options: -c ./calib'

          else:
            cp.dmqwmain.append_info('LCLS1 COMMAND "%s" IS NOT IMPLEMENTED...' % command)
            return

          cmd = qwu.edit_and_confirm_or_cancel_dialog_box(parent=self, text=cmd, title='Edit and confirm or cancel command')
          if cmd is None:
            logger.info('command is cancelled')
            return

          cmd_seq = ['/bin/bash', '-l', '-c', cmd_pref + cmd]
          self.subprocess_command(cmd_seq, shell=False, env=ENV1, executable='/bin/bash')


    def subprocess_command(self, cmd, **kwa):
        import psana.graphqt.UtilsSubproc as usp
        cp.dmqwmain.append_info(cmd)
        env   = kwa.get('env', None)
        shell = kwa.get('shell', False)
        executable = kwa.get('executable', '/bin/bash')
        self.osp = usp.SubProcess()
        self.osp(cmd, stdout=usp.subprocess.PIPE, stderr=usp.subprocess.STDOUT, env=env, shell=shell, executable=executable)
        logger.info('\n== creates subprocess for command: %s' % cmd)

        QTimer().singleShot(self.dt_msec, self.on_timeout)


    def on_timeout(self):
        if self.force_stop:
           if self.osp: self.osp.kill()
           return
        s = self.osp.stdout_incriment().rstrip('\n')
        if cp.dmqwmain is None: print(s)
        else: cp.dmqwmain.append_info(s)
        if self.osp.is_compleated():
           cp.dmqwmain.append_info('subprocess is completed')
           return
        QTimer().singleShot(self.dt_msec, self.on_timeout)


    def on_but_exp(self):
        from psana.graphqt.PSPopupSelectExp import select_instrument_experiment
        from psana.graphqt.CMConfigParameters import dir_calib #cp, 

        dir_instr = cp.instr_dir.value()
        instr_name, exp_name = select_instrument_experiment(None, dir_instr, show_frame=True) # parent=self.but_exp
        logger.debug('selected experiment: %s' % exp_name)
        if instr_name and exp_name and exp_name!=cp.exp_name.value():
            self.but_exp.setText(exp_name)
            cp.instr_name.setValue(instr_name)
            cp.exp_name.setValue(exp_name)

            if cp.dmqwmain is not None:
               cp.dmqwmain.wlist.fill_list_model(experiment=exp_name)
               cp.dmqwmain.winfo.winfo.clear()

            self.set_lclsv_for_instrument(instr_name)
            self.set_cmb_det()


    def on_but_view(self):
        """Re-implementation of the CMWControlBase.on_but_view"""
        logger.info('on_but_view - set file name and switch to IV')
        fname = cp.last_selected_fname.value()
        if fname is None:
            logger.warning('run is not selected to view anything...')
            return
        if cp.cmwmaintabs is not None:
            cp.cmwmaintabs.view_data(fname=fname)
        else:
            logger.info('tab bar object is not defined - do not switch to IV')


    def on_but_save(self):
        """Re-implementation of the CMWControlBase.on_but_save"""
        logger.warning('on_but_save - show the list of selected files')
        if cp.dmqwmain is None: return
        wtree = cp.dmqwmain.wlist

        names_selected = [full_path_for_item(i) for i in wtree.selected_items()]
        if len(names_selected)==0:
            logger.warning('nothing is selected - click on desired dataset(s) it the tree.')
            return

        logger.info('selected file names:\n  %s' % '\n  '.join(names_selected))

#        self.instr_exp_is_changed.emit(instr_name, exp_name)

#    def connect_instr_exp_is_changed(self, recip):
#        self.instr_exp_is_changed.connect(recip)

#    def disconnect_instr_exp_is_changed(self, recip):
#        self.instr_exp_is_changed.disconnect(recip)

#    def on_instr_exp_is_changed(self, instr, exp):
#        logger.debug('selected instrument: %s experiment: %s' % (instr, exp))


    def view_hide_tabs(self):
        CMWControlBase.view_hide_tabs(self)
        # set file manager tabs in/visible too
        wtabs = cp.fmwtabs
        if wtabs is None: return
        is_visible = wtabs.tab_bar_is_visible()
        wtabs.set_tabs_visible(not is_visible)


    def save_item_path(self, index):
        i = cp.dmqwmain.wlist.model.itemFromIndex(index)
        if i is None:
           logger.info('on_item_selected: item is None')
           cp.last_selected_fname.setValue(None)
           cp.last_selected_data = None
           return
        fname = full_path_for_item(i)
        if os.path.exists(fname) and (not os.path.isdir(fname)):
            logger.info('save_item_path save last_selected_fname: %s path: %s' % (str(i.text()), fname))
            cp.last_selected_fname.setValue(fname)
            cp.last_selected_data = None


    def on_item_selected(self, selected, deselected):
        self.save_item_path(selected)


    def detnames(self, expname, runnum):
        import psana.graphqt.UtilsWebServ as uws
        lst = uws.detnames(expname, runnum)
        s = 'detnames:\n  %s' % '\n  '.join(lst)
        cp.dmqwmain.append_info(s)
        return lst


    def set_cmb_det(self, detnames=None):
        lst = ['select',] + detnames if isinstance(detnames, list) else self.det_list0
        lst = [v[:-2] if (len(v)>2 and v[-2:]=='_0') else v for v in lst]

        detname_old = self.cmb_det.currentText()
        i = lst.index(detname_old) if detname_old in lst else 0
        self.cmb_det.clear()
        self.cmb_det.addItems(lst)
        self.cmb_det.setCurrentIndex(i)


    def on_selected_exp_run(self, expname, runnum): # called from DMQWMain <- DMQWList
        logger.debug('on_selected_exp_run: %s %d'%(expname, runnum))
        lst = self.detnames(expname, runnum)
        if not lst: return
        self.set_cmb_det(detnames=lst)


    def on_click(self, index):
        self.save_item_path(index)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        CMWControlBase.closeEvent(self, e)
        cp.dmqwcontrol = None


if __name__ == "__main__":
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = DMQWControl()
    w.setGeometry(100, 50, 500, 70)
    w.setWindowTitle('FM Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
