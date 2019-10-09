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

from PyQt5.QtWidgets import QGroupBox, QLabel, QCheckBox, QPushButton, QComboBox,\
                            QGridLayout, QHBoxLayout, QVBoxLayout, QSizePolicy
 
from PyQt5.QtCore import Qt, QTimer, QSize # pyqtSignal, QRectF, QPointF

from psdaq.control_gui.QWIcons import icon
from psdaq.control_gui.Styles import style
from psdaq.control_gui.CGDaqControl import daq_control, DaqControl

#--------------------

class CGWMainControl(QGroupBox) :
    """
    """
    status_record = ['Begin', 'End']

    def __init__(self, parent=None, parent_ctrl=None):

        QGroupBox.__init__(self, 'Control', parent)

        self.parent_ctrl = parent_ctrl

        self.lab_state = QLabel('Target State')
        self.lab_trans = QLabel('Last Transition')
        self.lab_ctrls = QLabel('Control State')
        #self.box_type = QComboBox(self)
        #self.box_type.addItems(self.LIST_OF_CONFIG_OPTIONS)
        #self.box_type.setCurrentIndex(1)

        icon.set_icons()

        self.but_record = QPushButton(icon.icon_record_sym, '') # icon.icon_record
        self.but_record.setAccessibleName(self.status_record[0])
        self.lab_record = QLabel('Recording')

        self.box_state      = QComboBox()
        self.but_transition = QPushButton('Unknown')
        self.but_ctrls      = QPushButton('Ready')

        self.states = ['Select',] + [s.upper() for s in DaqControl.states]
        self.box_state.addItems(self.states)

        if False :
            self.hbox1 = QHBoxLayout() 
            self.hbox1.addStretch(1)
            self.hbox1.addWidget(self.lab_record)
            self.hbox1.addWidget(self.but_record)
            self.hbox1.addStretch(1)
        
            self.hbox2 = QHBoxLayout() 
            self.hbox2.addWidget(self.lab_state)
            self.hbox2.addStretch(1)
            self.hbox2.addWidget(self.lab_trans)
        
            self.hbox3 = QHBoxLayout() 
            self.hbox3.addWidget(self.box_state, 0, Qt.AlignCenter)
            self.hbox3.addStretch(1)
            self.hbox3.addWidget(self.but_transition, 0, Qt.AlignCenter)
        
            self.vbox = QVBoxLayout() 
            self.vbox.addLayout(self.hbox1)
            self.vbox.addLayout(self.hbox2)
            self.vbox.addLayout(self.hbox3)
        
            self.setLayout(self.vbox)

        else :
            self.grid = QGridLayout()
            self.grid.addWidget(self.lab_record,      0, 0, 1, 1)
            self.grid.addWidget(self.but_record,      0, 4, 1, 1)
            self.grid.addWidget(self.lab_state,       1, 0, 1, 1)
            self.grid.addWidget(self.lab_trans,       1, 9, 1, 1)
            self.grid.addWidget(self.box_state,       2, 0, 1, 1)
            self.grid.addWidget(self.but_transition,  2, 9, 1, 1)
            self.grid.addWidget(self.lab_ctrls,       3, 0, 1, 1)
            self.grid.addWidget(self.but_ctrls,       4, 0, 1,10)
            self.setLayout(self.grid)

        self.set_tool_tips()
        self.set_style()

        self.box_state.currentIndexChanged[int].connect(self.on_box_state)
        self.but_transition.clicked.connect(self.on_but_transition)
        self.but_record.clicked.connect(self.on_but_record)
        self.but_ctrls.clicked.connect(self.on_but_ctrls)

        #self.timer = QTimer()
        #self.timer.timeout.connect(self.on_timeout)
        #self.timer.start(1000)

        self.state = 'undefined'
        self.transition = 'undefined'
        self.ts = 'N/A'
        self.check_state()
        self.check_transition()

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Configuration') 
        s = '%s recording' % self.but_record.accessibleName()
        self.but_record.setToolTip(s)
        self.lab_record.setText(s+':')
        self.box_state.setToolTip('Select desirable state.')
        self.but_transition.setToolTip('Last transition info.')
        self.but_ctrls.setToolTip('State info.') 

#--------------------

    def set_style(self) :
        self.setStyleSheet(style.qgrbox_title)
        self.lab_record.setFixedWidth(100)
        self.but_record.setFixedSize(50, 50)
        self.but_record.setIconSize(QSize(48, 48))

        self.but_ctrls.setStyleSheet(style.styleButtonGood)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.layout().setContentsMargins(4,4,4,4)
        self.setMinimumSize(270,140)

#--------------------

    def sizeHint(self):
        return QSize(270,160)
 
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

    def on_but_ctrls(self):
        logger.debug('on_but_ctrls')
        self.check_state()

#--------------------
 
    def on_cbx_runc(self, ind) :
        #if self.cbx.hasFocus() :
        cbx = self.cbx_runc
        tit = cbx.text()
        self.cbx_runc.setStyleSheet(style.styleGreenish if cbx.isChecked() else style.styleYellowBkg)
        msg = 'Check box "%s" is set to %s' % (tit, cbx.isChecked())
        logger.info(msg)

#--------------------

    def on_but_record(self) :
        txt = self.but_record.accessibleName()
        logger.debug('TBD - on_but_record %s' % txt)
        ind = self.status_record.index(txt)
        ico = icon.icon_record_sym if ind==1 else\
              icon.icon_record
        self.but_record.setIcon(ico)
        self.but_record.setAccessibleName(self.status_record[0 if ind==1 else 1])

        s = '%s recording' % self.but_record.accessibleName()
        self.but_record.setToolTip(s)
        self.lab_record.setText(s+':')

#--------------------

    def check_state(self) :
        #logger.debug('CGWMainDetector.check_state -> daq_control().getState()')
        state = daq_control().getState()
        if state is None : return
        if state == self.state : return
        self.set_but_ctrls(state)

#--------------------

    def set_but_ctrls(self, s) :
        self.ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        self.state = s 
        #self.but_state.setText('%s since %s' % (s.upper(), self.ts))
        self.but_ctrls.setText(s.upper())
        self.parent_ctrl.wpart.set_buts_enable(s.upper()) # enable/disable button plat in other widget

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
        transition, state, config_alias, recording = daq_control().getStatus() # submits request to check transition, state, config_alias, and recording
        logger.debug('CGWMainDetector.check_transition transition:%s state:%s config_alias:%s recording:%s' % (str(transition), str(state), str(config_alias), str(recording)))
        self.but_transition.setText(transition.upper()) # + ' since %s' % self.ts)
        #state = daq_control().getState()
        #self.but_state.setText(state.upper() + ' since %s' % self.ts)

#--------------------

    def set_transition(self, s) :
        #ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        #self.but_transition.setText('%s since %s' % (s.upper(), ts))
        self.but_transition.setText(s.upper())

#--------------------

    if __name__ == "__main__" :
 
      def resizeEvent(self, e):
        print('CGWMainControl.resizeEvent: %s' % str(self.size()))

#--------------------
#--------------------
#--------------------
#--------------------
 
if __name__ == "__main__" :

    from psdaq.control_gui.CGDaqControl import daq_control, DaqControlEmulator, Emulator
    daq_control.set_daq_control(DaqControlEmulator())

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainControl(None, parent_ctrl=Emulator())
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
