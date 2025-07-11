
"""Class :py:class:`CMWConfigPars` is a QWidget for configuration parameters
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWConfigPars.py

    # Import
    from psana2.graphqt.CMConfigParameters import

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
from psana2.graphqt.QWDirNameV2 import QWDirNameV2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QComboBox, QLineEdit, QPushButton
from psana2.graphqt.CMConfigParameters import cp
from psana2.graphqt.Styles import style


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
        self.fld_log_file = QWDirNameV2(self, label='Log dir:', path=cp.log_prefix.value(), hide_path=False)
        self.fld_dir_ins = QWDirNameV2(self, label='Instrument dir:', path=cp.instr_dir.value(), hide_path=False)
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

        self.but_exp = QPushButton(cp.exp_name.value())

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
        self.fld_log_file.connect_path_is_changed(self.on_fld)
        self.fld_dir_ins.connect_path_is_changed(self.on_fld)
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


    def closeEvent(self, event):
        logger.debug('closeEvent')


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


    def on_fld(self,s):
        logger.debug('on_fld selected:%s'%s)
        if self.fld_log_file.hasFocus(): cp.log_prefix.setValue(s)
        elif self.fld_dir_ins.hasFocus(): cp.instr_dir.setValue(s)


    def on_but_exp(self):
        from psana2.graphqt.PSPopupSelectExp import select_instrument_experiment
        dir_instr = cp.instr_dir.value()
        instr_name, exp_name = select_instrument_experiment(self.but_exp, dir_instr)
        logger.debug('selected experiment: %s' % exp_name)
        if exp_name:
            self.but_exp.setText(exp_name)
            cp.exp_name.setValue(exp_name)
        if instr_name: cp.instr_name.setValue(instr_name)


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
