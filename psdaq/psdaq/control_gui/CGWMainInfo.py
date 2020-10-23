#----
"""
:py:class:`CGWMainInfo` - widget for configuration
===================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainInfo import CGWMainInfo

    # Methods - see test

See:
    - :py:class:`CGWMainInfo`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-10-23 by Mikhail Dubrovin
"""
#----

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QLabel, QLineEdit, QGridLayout
# QPushButton, QHBoxLayout, QVBoxLayout #, QCheckBox, QComboBox
from psdaq.control_gui.CGConfigParameters import cp
from psdaq.control_gui.Styles import style
#from psdaq.control_gui.CGDaqControl import daq_control, DaqControlEmulator, daq_control_get_instrument
#from PyQt5.QtCore import Qt, QPoint
#from PyQt5.QtGui import QCursor

#----

class CGWMainInfo(QGroupBox):
    """
    """
    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Info', parent)
        cp.cgwmaininfo = self

        self.lab_exp = QLabel('exp:')
        self.lab_run = QLabel('run:')
        self.lab_evt = QLabel('event:')
        self.lab_evd = QLabel('ev. dropped:')
        self.edi_exp = QLineEdit('N/A')
        self.edi_run = QLineEdit('N/A')
        self.edi_evt = QLineEdit('N/A')
        self.edi_evd = QLineEdit('N/A')

        self.grid = QGridLayout()
        self.grid.addWidget(self.lab_exp,      0, 0, 1, 1)
        self.grid.addWidget(self.edi_exp,      0, 1, 1, 1)
        self.grid.addWidget(self.lab_run,      0, 2, 1, 1)
        self.grid.addWidget(self.edi_run,      0, 3, 1, 1)
        self.grid.addWidget(self.lab_evt,      1, 0, 1, 1)
        self.grid.addWidget(self.edi_evt,      1, 1, 1, 1)
        self.grid.addWidget(self.lab_evd,      1, 2, 1, 1)
        self.grid.addWidget(self.edi_evd,      1, 3, 1, 1)
        self.setLayout(self.grid)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        self.setToolTip('Information') 


#    def set_buts_enabled(self):
#        self.but_type.setEnabled(cp.s_state in ('reset','unallocated','allocated','connected'))
#        self.but_edit.setEnabled(True)


    def set_style(self):
        self.setStyleSheet(style.qgrbox_title)
        self.layout().setContentsMargins(2,2,2,2)
        #self.layout().setContentsMargins(5,5,5,5)
        #self.layout().setContentsMargins(0,0,0,0)
        #self.but_edit.setFixedWidth(60)
        #self.setMinimumWidth(350)
        #self.setWindowTitle('File name selection widget')
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumHeight(800)
 

    def closeEvent(self, e):
        logger.debug('CGWMainInfo.closeEvent')
        QGroupBox.closeEvent(self, e)
        cp.cgwmaininfo = None

#----
 
if __name__ == "__main__":

    logging.basicConfig(format='[%(levelname).1s] %(asctime)s %(name)s %(lineno)d: %(message)s', level=logging.DEBUG)

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CGWMainInfo(parent=None)
    w.show()
    logger.info('show window')
    app.exec_()

#----
