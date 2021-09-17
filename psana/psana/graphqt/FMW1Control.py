
"""Class :py:class:`FMW1Control` is a File Manager QWidget with control fields
==============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/FMW1Control.py

    from psana.graphqt.FMW1Control import FMW1Control
    w = FMW1Control()

Created on 2021-07-27 by Mikhail Dubrovin
"""

from psana.graphqt.CMWControlBase import * # CMWControlBase, QApplication, ..., icon, style, cp, logging, os, sys
from psana.graphqt.PSPopupSelectExp import select_instrument_experiment

logger = logging.getLogger(__name__)


def instr_exp_cvers_detname_ctype_runrange_from_calibpath(path):
    """split name like /cds/data/psdm/XCS/xcsx47519/calib/Epix100a::CalibV1/XcsEndstation.0:Epix100a.1/pedestals/23-end.data"""
    flds = path.split('/')
    if len(flds)<5 or flds[-5]!='calib':
        logger.debug('non-expected calib file path: %s' % path)
        return None
    return flds[-7], flds[-6], flds[-4], flds[-3], flds[-2], flds[-1][:-5]


class FMW1Control(CMWControlBase):
    """QWidget for File Manager control fields"""

    instr_exp_is_changed = pyqtSignal('QString', 'QString')

    cmd_list = ['calibfile', 'dcs', 'cdb']
    cmd_tips = 'Choose command to deploy constants\nfrom selected file(s)'\
      '\n    calibfile - in calib directory for LCLS'\
      '\n    dcs - in hdf5 file LCLS'\
      '\n    cdb - in constants DB LCLS2'

    def __init__(self, **kwa):

        CMWControlBase.__init__(self, **kwa)
        cp.fmw1control = self
        self.osp = None
        self.dt_msec = 1000

        #expname = kwa.get('expname', cp.exp_name.value())
        expname = kwa.get('expname', expname_def())

        self.wfnm.setVisible(False)
        self.wfnm.setEnabled(False)

        self.lab_exp     = QLabel('Exp:')
        self.lab_expto   = QLabel('Deploy:')
        self.but_exp     = QPushButton(expname)
        self.but_exp_col = QPushButton('Collapse')
        self.but_expto   = QPushButton('Select')
        self.but_deploy  = QPushButton('Deploy')
        self.but_stop    = QPushButton('Stop')
        self.cmb_cmd     = QComboBox()
        #self.cmb_cmd.clear()
        self.cmb_cmd.addItems(self.cmd_list)

        self.box = QHBoxLayout() #QGridLayout()
        self.box.addWidget(self.lab_exp)
        self.box.addWidget(self.but_exp)
        self.box.addWidget(self.but_exp_col)
        self.box.addStretch(1)
        self.box.addWidget(self.lab_expto)
        self.box.addWidget(self.cmb_cmd)
        self.box.addWidget(self.but_expto)
        self.box.addWidget(self.but_deploy)
        self.box.addWidget(self.but_stop)
        self.box.addStretch(1)
        self.box.addWidget(self.but_save)
        self.box.addWidget(self.but_view)
        self.box.addWidget(self.but_tabs)
        self.setLayout(self.box)

        self.but_exp.clicked.connect(self.on_but_exp)
        self.but_expto.clicked.connect(self.on_but_expto)
        self.but_exp_col.clicked.connect(self.on_but_exp_col)
        self.but_deploy.clicked.connect(self.on_but_deploy)
        self.but_stop.clicked.connect(self.on_but_stop)
        self.cmb_cmd.currentIndexChanged[int].connect(self.on_cmb_cmd)

#        self.but_save.clicked.connect(self.on_but_save)
#        self.connect_instr_exp_is_changed(self.on_instr_exp_is_changed)

        self.set_tool_tips()
        self.set_style()

        if cp.fmw1main is not None:
            global full_path_for_item
            from psana.graphqt.FSTree import full_path_for_item
            cp.fmw1main.wfstree.connect_item_selected_to(self.on_item_selected)
            cp.fmw1main.wfstree.clicked[QModelIndex].connect(self.on_click)


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        self.but_exp.setToolTip('Select experiment')
        self.but_expto.setToolTip('Select destination experiment')
        self.but_save.setToolTip('To save array in file\nclick on *.data file in the tree\nthen click on this Save button')
        self.but_exp_col.setToolTip('Collapse/expand directory/file tree')
        self.but_deploy.setToolTip('Execute deploy command for selected parameters')
        self.but_stop.setToolTip('Stop deployment subprocess\nif something going wrong')
        self.cmb_cmd.setToolTip(self.cmd_tips)


    def set_style(self):
        CMWControlBase.set_style(self)
        self.lab_exp.setStyleSheet(style.styleLabel)
        self.lab_exp.setFixedWidth(25)
        self.lab_expto.setStyleSheet(style.styleLabel)
        self.lab_expto.setFixedWidth(50)
        self.but_exp_col.setIcon(icon.icon_folder_open)
        self.but_exp.setFixedWidth(80)
        self.but_expto.setFixedWidth(80)
        self.but_exp_col.setFixedWidth(80)
        self.cmb_cmd.setFixedWidth(80)
        self.but_deploy.setFixedWidth(60)
        self.but_stop.setFixedWidth(40)
         #self.but_buts.setStyleSheet(style.styleButton)
        #self.but_tabs.setVisible(True)
        self.layout().setContentsMargins(5,0,5,0)


    def on_but_expto(self):
        dir_instr = cp.instr_dir.value()
        instr_name, exp_name = select_instrument_experiment(None, dir_instr, show_frame=True) # parent=self.but_exp
        logger.debug('selected experiment: %s' % exp_name)
        if exp_name==str(self.but_expto.text()): return
        self.but_expto.setText(exp_name)


    def on_cmb_cmd(self, ind):
        logger.debug('on_cmb_cmd selected index %d: %s' % (ind, self.cmd_list[ind]))


    def on_but_deploy(self):
        logger.debug('TBD on_but_deploy')

        if cp.fmw1main is None: return
        wtree = cp.fmw1main.wfstree

        fnames_selected = [full_path_for_item(i) for i in wtree.selected_items()]
        if len(fnames_selected)==0:
            logger.warning('nothing is selected - click on desired dataset(s) it the tree.')
            return

        logger.info('selected file names:\n  %s' % '\n  '.join(fnames_selected))

        expto = self.but_expto.text()
        command = self.cmb_cmd.currentText()
        logger.info('command "%s" deploys constants for exp: %s' % (command, expto))

        lclsv = 'LCLS2' if command[:3] == 'cdb' else 'LCLS'

        cmd = ''

        for fname in fnames_selected:
            r = instr_exp_cvers_detname_ctype_runrange_from_calibpath(fname)
            if r is None: continue
            instrpath, exppath, cvers, detname, ctype, runrange = r
            # XcsEndstation.0:Epix100a.1 -> epix100a_test
            detname_lcls2 = detname.split(':')[-1].split('.')[0].lower() + '_test'

            if command == 'calibfile':
                #calibfile deploy -f myfile.txt -e xpptut15 -r 54-59 -t pedestals -s XppGon.0:Cspad.0
                cmd += 'calibfile deploy -e %s -s %s -t %s -r 0-end -f %s -c ./calib;' % (expto, detname, ctype, fname)

            elif command == 'dcs':
                #dcs print      -e mfxn8316 -r 11 -d Epix100a
                #dcs add        -e mfxn8316 -r 11 -d Epix100a -t geometry     -f geo.txt -m "my geo" -c ./calib
                cmd += 'dcs add -e %s -d %s -t %s -r 0 -f %s    -T 1500000000 -c ./calib;' % (expto, detname, ctype, fname)

            elif command == 'cdb':
                #cdb add -e amox27716 -d ele_opal -c pop_rbfs -r 50 -f /reg/g/.../calib-amox27716-r50-opal-pop-rbfs-xiangli.pkl -i pkl
               cmd += 'cdb add -e %s -d %s -c %s -r 0 -f %s -i txt -l DEBUG;' % (expto, detname_lcls2, ctype, fname)

            else:
                logger.warning('Not recognized command "%s"' % command)
                return

        cmd = qwu.edit_and_confirm_or_cancel_dialog_box(parent=self, text=cmd, title='Edit and confirm or cancel command')
        if cmd is None:
          logger.info('command is cancelled')
          return

        if lclsv=='LCLS':
          cmd_seq = ['/bin/bash', '-l', '-c', COMMAND_SET_ENV_LCLS1 + cmd]
          self.subprocess_command(cmd_seq, shell=False, env=ENV1, executable='/bin/bash')

        elif lclsv=='LCLS2':
          self.subprocess_command(cmd, shell=True)

        else:
            logger.warning('Not recognized LCLS version "%s"' % lclsv)
            return


    def subprocess_command(self, cmd, **kwa):
        logger.warning('TBD subprocess_command\n%s' % cmd)
        cp.fmw1main.append_info(cmd)
        self.force_stop = False
        #return

        import psana.graphqt.UtilsSubproc as usp
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
        s = self.osp.stdout_incriment().rstrip('\n')
        if cp.fmw1main is None: print(s)
        else: cp.fmw1main.append_info(s)
        if self.osp.is_compleated():
           cp.fmw1main.append_info('subprocess is completed')
           return
        if self.force_stop:
           msg = 'forced stop, process may not be completed'
           cp.fmw1main.append_info(msg)
           logger.debug(msg)
           return
        QTimer().singleShot(self.dt_msec, self.on_timeout)


    def on_but_stop(self):
        logger.debug('on_but_stop')
        self.force_stop = True


    def on_but_exp(self):

        dir_instr = cp.instr_dir.value()
        instr_name, exp_name = select_instrument_experiment(None, dir_instr, show_frame=True) # parent=self.but_exp
        logger.debug('selected experiment: %s' % exp_name)
        if instr_name and exp_name and exp_name!=cp.exp_name.value():
            self.but_exp.setText(exp_name)
            cp.instr_name.setValue(instr_name)
            cp.exp_name.setValue(exp_name)

            if cp.fmw1main is not None:
               cp.fmw1main.wfstree.update_tree_model(dir_calib(exp_name))


    def on_but_exp_col(self):
        logger.debug('on_but_exp_col')
        if cp.fmw1main is None: return
        wtree = cp.fmw1main.wfstree
        but = self.but_exp_col
        if but.text() == 'Expand':
            wtree.process_expand()
            self.but_exp_col.setIcon(icon.icon_folder_closed)
            but.setText('Collapse')
        else:
            wtree.process_collapse()
            self.but_exp_col.setIcon(icon.icon_folder_open)
            but.setText('Expand')


    def on_but_view(self):
        """Re-implementation of the CMWControlBase.on_but_view"""
        logger.info('on_but_view - set file name and switch to IV')
        fname = cp.last_selected_fname.value()
        if fname is None:
            logger.warning('data file is not selected, do not switch to IV')
            return
        if cp.cmwmaintabs is not None:
            cp.cmwmaintabs.view_data(fname=fname)
        else:
            logger.info('tab bar object is not defined - do not switch to IV')


    def on_but_save(self):
        """Re-implementation of the CMWControlBase.on_but_save"""
        logger.warning('on_but_save - show the list of selected files')
        if cp.fmw1main is None: return
        wtree = cp.fmw1main.wfstree

        fnames_selected = [full_path_for_item(i) for i in wtree.selected_items()]
        if len(fnames_selected)==0:
            logger.warning('nothing is selected - click on desired dataset(s) it the tree.')
            return

        logger.info('selected file names:\n  %s' % '\n  '.join(fnames_selected))

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
        i = cp.fmw1main.wfstree.model.itemFromIndex(index)
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
        cp.fmw1control = None


if __name__ == "__main__":
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = FMW1Control()
    w.setGeometry(100, 50, 600, 40)
    w.setWindowTitle('FM Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
