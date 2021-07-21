
"""Class :py:class:`CMWConfigPars` is a QWidget for configuration parameters
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWConfigPars.py

    # Import
    from psana.graphqt.CMConfigParameters import

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`CMWConfig`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-05 by Mikhail Dubrovin
"""
import os

import logging
logger = logging.getLogger(__name__)
from psana.graphqt.QWDirNameV2 import QWDirNameV2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QComboBox, QLineEdit, QPushButton
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.Styles import style


class CMWConfigPars(QWidget):
    """QWidget for managements of configuration parameters"""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._name = 'CMWConfigPars'

        self.log_level_names = list(logging._levelToName.values())

        self.lab_host = QLabel('Host:')
        self.lab_port = QLabel('Port:')
        self.lab_level= QLabel('Log level:')
        self.lab_exp  = QLabel('Experiment:')
        self.fld_log_file = QWDirNameV2(self, label='Log dir:', path=cp.log_prefix.value(), split=False)
        self.fld_dir_ins = QWDirNameV2(self, label='Instrument dir:', path=cp.instr_dir.value(), split=False)
        self.cmb_host = QComboBox(self)        
        self.cmb_host.addItems(cp.list_of_hosts)
        hostname = cp.cdb_host.value()
        idx = cp.list_of_hosts.index(hostname) if hostname in cp.list_of_hosts else 1
        self.cmb_host.setCurrentIndex(idx)

        self.cmb_port = QComboBox(self)        
        self.cmb_port.addItems(cp.list_of_str_ports)
        self.cmb_port.setCurrentIndex(cp.list_of_str_ports.index(str(cp.cdb_port.value())))

        self.cmb_level = QComboBox(self)        
        self.cmb_level.addItems(self.log_level_names)
        self.cmb_level.setCurrentIndex(self.log_level_names.index(cp.log_level.value()))

        self.but_exp = QPushButton('Select')

        self.grid = QGridLayout()
        self.grid.addWidget(self.lab_host, 0, 0)
        self.grid.addWidget(self.cmb_host, 0, 1, 1, 1)

        self.grid.addWidget(self.lab_port, 1, 0)
        self.grid.addWidget(self.cmb_port, 1, 1, 1, 1)

        self.grid.addWidget(self.lab_level, 2, 0)
        self.grid.addWidget(self.cmb_level, 2, 1, 1, 1)

        self.grid.addWidget(self.fld_log_file, 3, 0, 1, 2)
        self.grid.addWidget(self.fld_dir_ins,  4, 0, 1, 2)

        self.grid.addWidget(self.lab_exp, 5, 0)
        self.grid.addWidget(self.but_exp, 5, 1)

        self.setLayout(self.grid)
        
        self.cmb_host.currentIndexChanged[int].connect(self.on_cmb_host_changed)
        self.cmb_port.currentIndexChanged[int].connect(self.on_cmb_port_changed)
        self.cmb_level.currentIndexChanged[int].connect(self.on_cmb_level_changed)
        self.fld_log_file.connect_path_is_changed_to_recipient(self.on_fld)
        self.fld_dir_ins.connect_path_is_changed_to_recipient(self.on_fld)
        self.but_exp.clicked.connect(self.on_but_exp)
        self.set_tool_tips()
        self.set_style()

    def set_tool_tips(self):
        self.cmb_host.setToolTip('Select DB host')
        self.cmb_port.setToolTip('Select DB port')

    def set_style(self):
        self.         setStyleSheet(style.styleBkgd)
        self.lab_exp.setStyleSheet(style.styleLabel)
        self.lab_host.setStyleSheet(style.styleLabel)
        self.lab_port.setStyleSheet(style.styleLabel)
        self.lab_level.setStyleSheet(style.styleLabel)
        self.fld_log_file.lab.setStyleSheet(style.styleLabel)
        self.fld_dir_ins.lab.setStyleSheet(style.styleLabel)
        self.setMaximumSize(400,600)
        #self.lab_dir_ins.setStyleSheet(style.styleLabel)

    #def resizeEvent(self, e):
    #    logger.debug('resizeEvent size: %s' % str(e.size())) 


    #def moveEvent(self, e):
        #logger.debug('moveEvent pos: %s' % str(e.pos())) 
        #cp.posGUIMain = (self.pos().x(),self.pos().y())

    def closeEvent(self, event):
        logger.debug('closeEvent')
        #try   : del cp.guiworkresdirs # CMWConfigPars
        #except: pass # silently ignore


#    def on_cbx(self, par, cbx):
#        #if cbx.hasFocus():
#        par.setValue(cbx.isChecked())
#        msg = 'check box %s is set to: %s' % (cbx.text(), str(par.value()))
#        logger.info(msg)


#    def on_cbx_host_changed(self, i):
#        logger.debug('XXX: %s' % str(type(i))
#        self.on_cbx(cp.cdb_host, self.cbx_host)


#    def on_cbx_port_changed(self, i):
#        self.on_cbx(cp.cdb_port, self.cbx_port)

    def on_cmb_host_changed(self):
        selected = self.cmb_host.currentText()
        cp.cdb_host.setValue(selected) 
        logger.info('Set DB host: %s' % selected)

    def on_cmb_port_changed(self):
        selected = self.cmb_port.currentText()
        cp.cdb_port.setValue(int(selected)) 
        logger.info('Set DB port: %s' % selected)

    def on_cmb_level_changed(self):
        selected = self.cmb_level.currentText()
        cp.log_level.setValue(selected) 
        logger.info('Set logger level %s' % selected)

#    def on_edi(self, par, but):
#        #logger.debug('on_edi')
#        par.setValue(str(but.displayText()))
#        logger.info('Set field: %s' % str(par.value()))

#    def on_edi_log_file(self):
#        ##logger.debug('on_edi_log_file')
#        self.on_edi(cp.log_prefix, self.edi_log_file)
#        #cp.log_prefix.setValue(str(self.edi_log_file.displayText()))
#        #logger.info('Set logger file name: ' + str(cp.log_prefix.value()))

    def on_fld(self,s):
        logger.debug('on_fld selected:%s'%s)
        if self.fld_log_file.hasFocus(): cp.log_prefix.setValue(s)
        elif self.fld_dir_ins.hasFocus(): cp.instr_dir.setValue(s)


    def on_but_exp(self):
        from psana.graphqt.PSPopupSelectExp import select_instrument_experiment
        dir_instr = cp.instr_dir.value()
        exp_name = select_instrument_experiment(self.but_exp, dir_instr)
        logger.debug('selected experiment: %s' % exp_name)
        if exp_name: self.but_exp.setText(exp_name)


if __name__ == "__main__":
    import sys
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    app = QApplication(sys.argv)
    w = CMWConfigPars()
    w.setGeometry(200, 400, 500, 200)
    w.setWindowTitle('Config Parameters')
    w.show()
    app.exec_()
    del w
    del app

# EOF
