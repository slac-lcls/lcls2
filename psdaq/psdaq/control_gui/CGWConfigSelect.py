#--------------------
"""
:py:class:`CGWConfigSelect` - widget to select configuration file from db
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWConfigSelect import CGWConfigSelect

    # Methods - see test

See:
    - :py:class:`CGWConfigSelect`
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

from psdaq.control_gui.CGWConfigEditor import CGWConfigEditor
from psdaq.control_gui.QWPopupSelectItem import popup_select_item_from_list
from psdaq.control_gui.CGConfigDBUtils import get_configdb, URI_CONFIGDB, ROOT_CONFIGDB
from psdaq.control_gui.CGDaqControl import daq_control_get_instrument
from psdaq.control_gui.CGJsonUtils import str_json
from psdaq.control_gui.CGConfigParameters import cp

#from psdaq.control_gui.QWUtils import confirm_or_cancel_dialog_box

#--------------------
char_expand  = u' \u25BC' # down-head triangle

class CGWConfigSelect(QGroupBox):
    """
    """
    def __init__(self, parent=None, type_def='Select', dev_def='Select'):

        QGroupBox.__init__(self, 'Edit configuration', parent)

        self.but_apply = None

        self.lab_type = QLabel('Type')
        self.lab_dev  = QLabel('Detector')

        self.but_type = QPushButton()
        self.but_dev  = QPushButton() # ('Select %s' % char_expand)

        self.set_but_type_text(type_def)
        self.set_but_dev_text(dev_def)

        self.hbox1 = QHBoxLayout() 
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.lab_type)
        self.hbox1.addWidget(self.but_type) 
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.lab_dev)
        self.hbox1.addWidget(self.but_dev)

        self.vbox = QVBoxLayout() 
        self.vbox.addLayout(self.hbox1)
        #self.vbox.addWidget(self.but_edit, 0, Qt.AlignCenter)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_type.clicked.connect(self.on_but_type)
        self.but_dev .clicked.connect(self.on_but_dev)

        self.type_old = None

#--------------------

    def set_tool_tips(self):
        #self.setToolTip('Configuration') 
        #self.but_edit.setToolTip('Edit configuration dictionary.')
        self.but_type.setToolTip('Select configuration type.') 
        self.but_dev .setToolTip('Select device for configuration.') 

#--------------------

    def set_buts_enabled(self):
        is_selected_type = self.but_type.text()[:6] != 'Select'
        is_selected_det  = self.but_dev .text()[:6] != 'Select'
        self.but_dev.setEnabled(is_selected_type)
        #self.but_edit.setEnabled(is_selected_type and is_selected_det)
        if self.but_apply is not None:
            is_enabled = is_selected_type and is_selected_det
            self.but_apply.setEnabled(is_enabled)
            #self.but_apply.setFlat(not is_enabled)

#--------------------

    def set_style(self):
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)
        self.set_buts_enabled()
        self.setWindowTitle('Select config type & device')

        #self.layout().setContentsMargins(0,0,0,0)
        #self.but_edit.setFixedWidth(60)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
 
#--------------------
 
    def inst_configdb(self, msg=''):
        uris = getattr(cp.cgwmain, 'uris', URI_CONFIGDB)
        inst = getattr(cp, 'instr', None)
        user = getattr(cp.cgwmain, 'user', None)
        pwd  = getattr(cp.cgwmain, 'password', None)
        if inst is None: inst = daq_control_get_instrument()
        logger.debug('%sconnect to configdb(uris=%s, inst=%s, user=%s, password=...)' % (msg, uris, inst, user))
        return inst, get_configdb(uri=uris, hutch=inst, create=False, root=ROOT_CONFIGDB, user=user, password=pwd)

#--------------------
 
    def on_but_type(self):
        #logger.debug('on_but_type')
        inst, confdb = self.inst_configdb('on_but_type: ')
        list_of_aliases = confdb.get_aliases(hutch=inst) # ['NOBEAM', 'BEAM', ??? 'PROD']

        if not list_of_aliases:
            list_of_aliases = ['NOBEAM', 'BEAM']
            logger.warning('List of configdb-s IS EMPTY... Use default: %s' % str(list_of_aliases))

        selected = popup_select_item_from_list(self.but_type, list_of_aliases, dx=-46, dy=-20)
        self.set_but_type_text(selected)
        msg = 'selected %s of the list %s' % (selected, str(list_of_aliases))
        logger.debug(msg)

        if selected != self.type_old:
            self.set_but_dev_text()
            self.type_old = selected

            # save selected configuration type in control json
            #rv = daq_control().setConfig(selected)
            #if rv is not None: logger.error('setState("%s"): %s' % (selected,rv))

        self.set_buts_enabled()

#--------------------
 
    def set_config_type(self, config_type):
        if config_type == self.type_old: return

        self.set_but_type_text(config_type)
        self.set_but_dev_text()
        self.type_old = config_type

        self.set_buts_enabled()

#--------------------
 
    def set_but_type_text(self, txt='Select'): self.but_type.setText('%s %s' % (txt, char_expand))
    def set_but_dev_text (self, txt='Select'): self.but_dev .setText('%s %s' % (txt, char_expand))

    def but_type_text(self): return str(self.but_type.text()).split(' ')[0] # 'NOBEAM' or 'BEAM'
    def but_dev_text (self): return str(self.but_dev .text()).split(' ')[0] # 'testdev0'

#--------------------

    def cfgtype_and_device(self):
        return self.but_type_text(), self.but_dev_text()

#--------------------
 
    def on_but_dev(self):
        #logger.debug('on_but_dev')
        inst, confdb = self.inst_configdb('on_but_dev: ')
        cfgtype = str(self.but_type.text()).split(' ')[0] # 'NOBEAM' or 'BEAM'
        list_of_device_names = confdb.get_devices(cfgtype, hutch=inst)

        if not list_of_device_names:
            logger.warning('list_of_device_names IS EMPTY... Check configuration DB')
            return

        selected = popup_select_item_from_list(self.but_dev, list_of_device_names, dx=-190, dy=-20)
        self.set_but_dev_text(selected)
        msg = 'selected %s of the list %s' % (selected, str(list_of_device_names))
        logger.debug(msg)

        self.set_buts_enabled()

#--------------------

    def closeEvent(self, e):
        print('CGWConfigSelect.closeEvent')
        QGroupBox.closeEvent(self, e)

#--------------------
 
if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CGWConfigSelect(parent=None)
    w.show()
    app.exec_()

#--------------------
