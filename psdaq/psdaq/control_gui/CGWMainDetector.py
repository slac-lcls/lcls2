#--------------------
"""
:py:class:`CGWMainDetector` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainDetector import CGWMainDetector

    # Methods - see test

See:
    - :py:class:`CGWMainDetector`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-28 by Mikhail Dubrovin
"""
#--------------------

from time import time
import psdaq.control_gui.Utils as gu

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout # , QWidget,  QLabel, QLineEdit, QFileDialog
from PyQt5.QtCore import QTimer # pyqtSignal, Qt, QRectF, QPointF


from psdaq.control_gui.CGDaqControl import daq_control # , DaqControl #, worker_get_state
#from psdaq.control_gui.DoWorkInThread import DoWorkInThread
#from psdaq.control_gui.CGParameters import cp

#--------------------

class CGWMainDetector(QGroupBox) :
    """
    """
    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Detector', parent)

        self.lab_state = QLabel('Control state')
        self.but_state = QPushButton('Ready')

        self.vbox = QVBoxLayout() 
        self.vbox.addWidget(self.lab_state)
        self.vbox.addWidget(self.but_state)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_state.clicked.connect(self.on_but_state)

        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timeout)
        self.timer.start(1000)

        self.state = 'undefined'
        self.ts = 'N/A'

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Detector GUI')
        self.but_state.setToolTip('Click on button.') 

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)
        self.but_state.setStyleSheet(style.styleButtonGood)

        #self.layout().setContentsMargins(0,0,0,0)

        #self.setWindowTitle('File name selection widget')
        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
    def on_but_state(self):
        logger.debug('on_but_state')
        self.check_state()

#--------------------
 
    def on_timeout(self) :
        #logger.debug('CGWMainDetector Timeout %.3f sec' % time())
        self.ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        self.lab_state.setText('Control state on %s' % self.ts)
        self.check_state()
        self.timer.start(1000)

#--------------------

    def check_state(self) :
        #logger.debug('CGWMainDetector.check_state -> daq_control().getState()')
        state = daq_control().getState()
        if state == self.state : return
        self.state = state
        #logger.debug('daq_control().getState() response %s' % state)
        self.but_state.setText(state.upper() + ' since %s' % self.ts)

#--------------------

    def closeEvent(self, e) :
        logger.debug('closeEvent', __name__)
        self.timer.stop()
        self.timer.timeout.disconnect(self.on_timeout)

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainDetector(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
