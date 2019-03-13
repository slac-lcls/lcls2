#--------------------
"""
:py:class:`CGWMainControl` - widget for configuration
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainControl import CGWMainControl

    # Methods - see test

See:
    - :py:class:`CGWMainControl`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

#from time import time

import psdaq.control_gui.Utils as gu

from PyQt5.QtWidgets import QGroupBox, QLabel, QCheckBox, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout, QComboBox
     #QGridLayout, QLineEdit, QFileDialog, QWidget
from PyQt5.QtCore import Qt, QTimer # pyqtSignal, QRectF, QPointF

from psdaq.control_gui.Styles import style

from psdaq.control_gui.CGDaqControl import daq_control, DaqControl #, worker_set_state
#from psdaq.control_gui.DoWorkInThread import DoWorkInThread
#from psdaq.control_gui.CGParameters import cp

#--------------------

class CGWMainControl(QGroupBox) :
    """
    """
    def __init__(self, parent=None, parent_ctrl=None):

        QGroupBox.__init__(self, 'Control', parent)

        self.parent_ctrl = parent_ctrl

        self.lab_state = QLabel('Target State')
        self.lab_trans = QLabel('Last Transition')
        #self.box_type = QComboBox(self)
        #self.box_type.addItems(self.LIST_OF_CONFIG_OPTIONS)
        #self.box_type.setCurrentIndex(1)

        self.cbx_runc       = QCheckBox('Record Run')
        self.box_state      = QComboBox()
        self.but_transition = QPushButton('Unknown')

        self.states = ['Select',] + [s.upper() for s in DaqControl.states]
        self.box_state.addItems(self.states)

        #self.edi = QLineEdit(path)
        #self.edi.setReadOnly(True) 

        self.hbox1 = QHBoxLayout() 
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.cbx_runc)
        self.hbox1.addStretch(1)

        self.hbox2 = QHBoxLayout() 
        self.hbox2.addWidget(self.lab_state)
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.lab_trans)

        self.hbox3 = QHBoxLayout() 
        self.hbox3.addWidget(self.box_state, 0, Qt.AlignCenter)
        self.hbox3.addStretch(1)
        self.hbox3.addWidget(self.but_transition, 0, Qt.AlignCenter)
        #self.hbox3.addStretch(1)

        self.vbox = QVBoxLayout() 
        #self.vbox.addWidget(self.cbx_runc, 0, Qt.AlignCenter)
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addLayout(self.hbox3)

        #self.grid = QGridLayout()
        #self.grid.addWidget(self.lab_state,       0, 0, 1, 1)
        #self.grid.addWidget(self.but_type,        0, 2, 1, 1)
        #self.grid.addWidget(self.box_state,       1, 1, 1, 1)
        #self.grid.addWidget(self.but_transition,  2, 1, 1, 1)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.box_state.currentIndexChanged[int].connect(self.on_box_state)
        self.but_transition.clicked.connect(self.on_but_transition)
        #self.box_type.currentIndexChanged[int].connect(self.on_box_type)
        self.cbx_runc.stateChanged[int].connect(self.on_cbx_runc)

        #self.timer = QTimer()
        #self.timer.timeout.connect(self.on_timeout)
        #self.timer.start(1000)

        self.transition = 'undefined'
        self.ts = 'N/A'
        self.check_transition()

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Configuration') 
        self.cbx_runc.setToolTip('Use checkbox to on/off recording.')
        self.box_state.setToolTip('Select desirable state.')
        self.but_transition.setToolTip('Info about last transition.')

#--------------------

    def set_style(self) :

        self.setStyleSheet(style.qgrbox_title)
        #self.box_state.setFixedWidth(60)
        #self.but_transition.setFixedWidth(60)
        #self.but_transition.setStyleSheet(style.styleButtonGood)

        self.cbx_runc.setFixedSize(100,40)
        #self.cbx_runc.setStyleSheet(style.styleYellowBkg)
        self.cbx_runc.setStyleSheet(style.style_cbx_off)

        #self.setWindowTitle('File name selection widget')
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 
        #self.layout().setContentsMargins(0,0,0,0)
        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
    def on_box_state(self, ind) :
        if not ind : return
        state = self.states[ind]
        logger.info('CGWMainDetector.on_box_state -> daq_control().setState %s' % state)
        #self.parent_ctrl.kick_zmq()
        daq_control().setState(state.lower())
        logger.debug('command daq_control().setState is committed...')

#--------------------
 
    def on_but_transition(self) :
        #logger.debug('on_but_transition') # NO ACTION')
        self.check_transition()

#--------------------
 
    def on_cbx_runc(self, ind) :
        #if self.cbx.hasFocus() :
        cbx = self.cbx_runc
        tit = cbx.text()
        self.cbx_runc.setStyleSheet(style.styleGreenish if cbx.isChecked() else style.styleYellowBkg)
        msg = 'Check box "%s" is set to %s' % (tit, cbx.isChecked())
        logger.info(msg)

#--------------------
 
#    def on_timeout(self) :
#        #logger.debug('CGWMainDetector Timeout %.3f sec' % time())
#        self.ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
#        #self.lab_state.setText('Control state on %s' % self.ts)
#        self.check_transition()
#        self.timer.start(1000)

#--------------------

    def check_transition(self) :
        """Uses getStatus() to get last transition and set the info button status.
        """
        logger.debug('CGWMainDetector.check_transition')
        #t0_sec = time() # takes 0.001s
        transition, state = daq_control().getStatus() # submits request to check transition and state
        logger.debug('CGWMainDetector.check_transition transition:%s state:%s' % (str(transition), str(state)))
        self.but_transition.setText(transition.upper()) # + ' since %s' % self.ts)
        #state = daq_control().getState()
        #self.but_state.setText(state.upper() + ' since %s' % self.ts)

#--------------------

    def set_transition(self, s) :
        #ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        #self.but_transition.setText('%s since %s' % (s.upper(), ts))
        self.but_transition.setText(s.upper())

#--------------------
#--------------------
#--------------------
#--------------------
 
if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainControl(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
