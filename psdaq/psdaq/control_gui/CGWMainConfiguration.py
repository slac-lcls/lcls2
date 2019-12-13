#--------------------
"""
:py:class:`CGWMainConfiguration` - widget for configuration
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainConfiguration import CGWMainConfiguration

    # Methods - see test

See:
    - :py:class:`CGWMainConfiguration`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QHBoxLayout, QVBoxLayout #, QCheckBox, QComboBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QCursor

from psdaq.control_gui.CGConfigParameters import cp
from psdaq.control_gui.CGWConfigEditor import CGWConfigEditor
from psdaq.control_gui.QWPopupSelectItem import popup_select_item_from_list
from psdaq.control_gui.CGConfigDBUtils import get_configdb
from psdaq.control_gui.CGJsonUtils import str_json

from psdaq.control_gui.QWUtils import confirm_or_cancel_dialog_box
from psdaq.control_gui.CGDaqControl import daq_control, DaqControlEmulator

from psdaq.control_gui.QWDialog import QWDialog
from psdaq.control_gui.CGWConfigSelect import CGWConfigSelect

#--------------------
char_expand  = u' \u25BC' # down-head triangle

class CGWMainConfiguration(QGroupBox) :
    """
    """
    list_of_aliases = ['NOBEAM', 'BEAM']

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Configuration', parent)

        cp.cgwmainconfiguration = self

        self.lab_type = QLabel('Type')
        self.but_type = QPushButton('Select %s' % char_expand)
        self.but_edit = QPushButton('Edit')

        self.hbox1 = QHBoxLayout() 
        self.hbox1.addWidget(self.lab_type)
        self.hbox1.addWidget(self.but_type) 
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.but_edit, 0, Qt.AlignCenter)

        self.vbox = QVBoxLayout() 
        self.vbox.addLayout(self.hbox1)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_edit.clicked.connect(self.on_but_edit)
        self.but_type.clicked.connect(self.on_but_type)
        #self.box_seq.currentIndexChanged[int].connect(self.on_box_seq)
        #self.cbx_seq.stateChanged[int].connect(self.on_cbx_seq)

        self.device_edit = None
        self.cfgtype_edit = None
        self.w_edit = None
        self.type_old = None
        self.set_config_type('init')

#--------------------

    def set_tool_tips(self) :
        #self.setToolTip('Configuration') 
        self.but_edit.setToolTip('Edit configuration dictionary.')
        self.but_type.setToolTip('Select configuration type.') 

#--------------------

    def set_buts_enabled(self) :
        is_selected_type = self.but_type.text()[:6] != 'Select'
        #is_selected_det  = self.but_dev .text()[:6] != 'Select'
        #self.but_dev .setEnabled(is_selected_type)
        #self.but_edit.setEnabled(is_selected_type and is_selected_det)
        self.but_type.setEnabled(True)
        self.but_edit.setEnabled(True)

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)
        self.but_edit.setFixedWidth(60)
        self.set_buts_enabled()

        #self.layout().setContentsMargins(0,0,0,0)
        #self.setMinimumWidth(350)
        #self.setWindowTitle('File name selection widget')
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
    def inst_configdb(self, msg=''):
        uris = getattr(cp.cgwmain, 'uris', 'mcbrowne:psana@psdb-dev:9306')
        inst = getattr(cp,         'inst', 'TST')
        logger.debug('%sconnect to configdb(uri_suffix=%s, inst=%s)' % (msg, uris, inst))
        return inst, get_configdb(uri_suffix=uris, inst=inst)

#--------------------

    def save_dictj_in_db(self, dictj, msg='') :
        logger.debug('%ssave_dictj_in_db' % msg)
        cfgtype, devname = self.cfgtype_and_device()
        inst, confdb = self.inst_configdb('CGWConfigEditor.on_but_apply: ')

        logger.debug('cfgtype:%s devname:%s' % (cfgtype, devname))
        logger.debug('inst:%s confdb:%s' % (inst, confdb))
        logger.debug('dictj:%s' % str(dictj))

        resp = confirm_or_cancel_dialog_box(parent=None,
                                            text='Save changes in configuration DB',\
                                            title='Confirm or cancel')
        if resp :
            try:
                new_key = confdb.modify_device(cfgtype, dictj, hutch=inst)
                logger.debug('save_dictj_in_db new_key: %d' % new_key)
            except ValueError as err:
                logger.error('ValueError: %s' % err)
            except Exception as err:            
                logger.error('Exception: %s' % err)

        else :
            logger.warning('Saving of configuration in DB is cancelled...')

#--------------------
 
    def on_but_type(self):
        #logger.debug('on_but_type')
        inst, confdb = self.inst_configdb('on_but_type: ')
        list_of_aliases = confdb.get_aliases(hutch=inst) # ['NOBEAM', 'BEAM']
        if not list_of_aliases :
            list_of_aliases = self.list_of_aliases # ['NOBEAM', 'BEAM']
            logger.warning('List of configdb-s IS EMPTY... Use default: %s' % str(list_of_aliases))

        selected = popup_select_item_from_list(self.but_type, list_of_aliases, dx=-46, dy=-33) #, use_cursor_pos=True)

        msg = 'selected %s of the list %s' % (selected, str(list_of_aliases))
        logger.debug(msg)

        if selected in (None, self.type_old) : return

        rv = daq_control().setConfig(selected)
        if rv is None: 
            self.set_config_type(selected)
        else :
            logger.error('setConfig("%s"): %s' % (selected,rv))
            self.set_config_type('error')

#--------------------
 
    def set_config_type(self, config_type):

        cfgtype = config_type if config_type in ('error','init') else cp.s_cfgtype

        if cfgtype == self.type_old : return

        if not (cfgtype in self.list_of_aliases) : return

        self.set_but_type_text(cfgtype)
        self.type_old = cfgtype
        self.set_buts_enabled()

#--------------------
 
    def set_but_type_text(self, txt='Select'): self.but_type.setText('%s %s' % (txt, char_expand))
    #def set_but_dev_text (self, txt='Select'): self.but_dev .setText('%s %s' % (txt, char_expand))

    def but_type_text(self): return str(self.but_type.text()).split(' ')[0] # 'NOBEAM' or 'BEAM'
    #def but_dev_text (self): return str(self.but_dev .text()).split(' ')[0] # 'testdev0'

#--------------------

    def cfgtype_and_device(self):
        return self.cfgtype_edit, self.device_edit #self.but_dev_text()

#--------------------
 
    """
    def on_but_dev(self):
        #logger.debug('on_but_dev')
        inst, confdb = self.inst_configdb('on_but_dev: ')
        cfgtype = str(self.but_type.text()).split(' ')[0] # 'NOBEAM' or 'BEAM'
        list_of_device_names = confdb.get_devices(cfgtype, hutch=inst)

        if not list_of_device_names :
            logger.warning('list_of_device_names IS EMPTY... Check configuration DB')
            return

        selected = popup_select_item_from_list(self.but_dev, list_of_device_names, dx=-46, dy=-33)
        self.set_but_dev_text(selected)
        msg = 'selected %s of the list %s' % (selected, str(list_of_device_names))
        logger.debug(msg)

        self.set_buts_enabled()
    """
#--------------------
 
#    def on_box_seq(self, ind):
#        selected = str(self.box_seq.currentText())
#        msg = 'selected ind:%d %s' % (ind,selected)
#        logger.debug(msg)

#--------------------
 
#    def on_cbx_seq(self, ind):
#        #if self.cbx.hasFocus() :
#        cbx = self.cbx_seq
#        tit = cbx.text()
#        #self.cbx_runc.setStyleSheet(style.styleGreenish if cbx.isChecked() else style.styleYellowBkg)
#        msg = 'Check box "%s" is set to %s' % (tit, cbx.isChecked())
#        logger.info(msg)

#--------------------
 
    def select_config_type_and_dev(self):

        wd = CGWConfigSelect(parent=self, type_def=self.but_type_text())
        w = QWDialog(None, wd, is_frameless=False)
        w.but_apply.setText('Edit')
        w.but_apply.setEnabled(False)
        #w.setWindowTitle('Select to edit')
        #w.move(QCursor.pos() + QPoint(-20, -20))
        w.move(self.mapToGlobal(self.but_edit.pos()) + QPoint(0, 0))

        resp=w.exec_()
        logger.debug('resp=%s' % resp)

        if resp == QWDialog.Rejected : return None

        cfgtype = wd.but_type_text()
        dev     = wd.but_dev_text()
        del w
        del wd

        return cfgtype, dev

#--------------------
 
    def on_but_edit(self):
        #logger.debug('on_but_edit')
        if self.w_edit is None :
            logger.debug("TBD Open configuration editor window")
            resp = self.select_config_type_and_dev()
            if resp is None : return
            cfgtype, dev = resp
            self.device_edit = dev
            self.cfgtype_edit = cfgtype

            inst, confdb = self.inst_configdb('on_but_edit: ')

            try :
                self.config = confdb.get_configuration(cfgtype, dev, hutch=inst)
            except ValueError as err:
                logger.error('ValueError: %s' % err)
                return

            msg = 'get_configuration(%s, %s, %s):\n' % (cfgtype, dev, inst)\
                + '%s\n    type(config): %s'%(str_json(self.config), type(self.config))
            logger.debug(msg)

            self.w_edit = CGWConfigEditor(dictj=self.config)
            self.w_edit.move(self.mapToGlobal(QPoint(self.width()+10,0)))
            self.w_edit.show()

        else :
            logger.debug("Close configuration editor window")
            self.w_edit.close()
            self.w_edit = None

#--------------------
 
#    def on_but_scan(self):
#        logger.debug('on_but_scan')

#--------------------

    def closeEvent(self, e):
        logger.debug('CGWMainConfiguration.closeEvent')
        if self.w_edit is not None :
           self.w_edit.close()
        QGroupBox.closeEvent(self, e)
        cp.cgwmainconfiguration = None

#--------------------
 
if __name__ == "__main__" :

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    daq_control.set_daq_control(DaqControlEmulator())

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CGWMainConfiguration(parent=None)
    w.show()
    app.exec_()

#--------------------
