
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

class DMQWControl(CMWControlBase):
    """QWidget for Data Manager control fields"""

    instr_exp_is_changed = pyqtSignal('QString', 'QString')

    def __init__(self, **kwa):

        #parent = kwa.get('parent', None)

        CMWControlBase.__init__(self, **kwa)
        cp.dmqwcontrol = self
        self.osp = None
        self.dt_msec = 1000

        expname = kwa.get('expname', expname_def())

        self.lab_exp = QLabel('Exp:')
        self.but_exp = QPushButton(expname)
        self.but_cmd = QPushButton('Command')
        self.but_stop = QPushButton('Stop')

        self.box = QHBoxLayout()
        self.box.addWidget(self.lab_exp)
        self.box.addWidget(self.but_exp)
        self.box.addStretch(1)
        self.box.addWidget(self.but_cmd)
        self.box.addWidget(self.but_stop)
        self.box.addWidget(self.but_save)
        self.box.addWidget(self.but_view)
        self.box.addWidget(self.but_tabs)
        self.setLayout(self.box)
 
        self.but_exp.clicked.connect(self.on_but_exp)
        self.but_cmd.clicked.connect(self.on_but_cmd)
        self.but_stop.clicked.connect(self.on_but_stop)

        self.set_tool_tips()
        self.set_style()

#        if cp.dmqwmain is not None:
#            global full_path_for_item
#            from psana.graphqt.FSTree import full_path_for_item
#            cp.dmqwmain.wlist.connect_item_selected_to(self.on_item_selected)
#            cp.dmqwmain.wlist.clicked[QModelIndex].connect(self.on_click)


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        self.but_exp.setToolTip('Select experiment')
        self.but_cmd.setToolTip('Infromation about run from DB')


    def set_style(self):
        CMWControlBase.set_style(self)
        self.lab_exp.setStyleSheet(style.styleLabel)
        self.lab_exp.setFixedWidth(25)
        self.but_exp.setFixedWidth(80)
        self.but_stop.setFixedWidth(45)
        #self.but_buts.setStyleSheet(style.styleButton)
        #self.but_tabs.setVisible(True)
        self.layout().setContentsMargins(5,0,5,0)

        self.wfnm.setVisible(False)
        self.wfnm.setEnabled(False)
        self.but_save.setVisible(False)
        self.but_view.setVisible(False)
        self.but_save.setEnabled(False)
        self.but_view.setEnabled(False)


    def on_but_stop(self):
        logger.debug('on_but_stop')
        self.force_stop = True


    def on_but_cmd(self):
        expname, runnum = cp.exp_name.value(), cp.last_selected_run
        logger.info('TBD on_but_cmd exp:%s run:%s' % (expname, str(runnum)))
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

        is_lcls2 = cp.dmqwmain.is_lcls2(expname, runnum)
        s = 'dataset exp=%s,run=%d is_lcls2:%s' % (expname, runnum, is_lcls2)
        cp.dmqwmain.append_info(s)

        if is_lcls2:
          cmd = 'detnames exp=%s,run=%d -r' % (expname, runnum)
          #cmd = 'datinfo -e %s -r %d' % (expname, runnum)
          self.subprocess_command(cmd)


    def subprocess_command(self, cmd):
        import psana.graphqt.UtilsSubproc as usp
        cp.dmqwmain.append_info(cmd)

        self.osp = usp.SubProcess()
        self.osp(cmd, stdout=usp.subprocess.PIPE, env=None, shell=False)
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
    w.setGeometry(100, 50, 500, 40)
    w.setWindowTitle('FM Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
