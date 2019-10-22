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
from psdaq.control_gui.CGDaqControl import daq_control_set_state, daq_control_get_state,\
                                           daq_control_set_record, daq_control_get_status, DaqControl
from psdaq.control_gui.CGConfigParameters import cp

#--------------------

class CGWMainControl(QGroupBox) :
    """
    """
    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Control', parent)

        cp.cgwmaincontrol = self

        self.lab_state = QLabel('Target State')
        self.lab_trans = QLabel('Last Transition')
        self.lab_ctrls = QLabel('Control State')
        #self.box_type = QComboBox(self)
        #self.box_type.addItems(self.LIST_OF_CONFIG_OPTIONS)
        #self.box_type.setCurrentIndex(1)

        icon.set_icons()

        self.but_record = QPushButton(icon.icon_record_sym, '') # icon.icon_record
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
        self.set_but_ctrls()

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Configuration') 
        self.but_record.setToolTip('sets flag for recording')
        self.lab_record.setText('Recording:')
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
        logger.info('-> daq_control_set_state(%s)' % state)
        if not daq_control_set_state(state.lower()):
            logger.warning('on_box_state: STATE %s IS NOT SET' % state)

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
        logger.debug('on_but_record')

        s = daq_control_get_status()
        if s is None :
            logger.warning('on_but_record: STATUS IS NOT AVAILABLE')
            return
        transition, state, cfgtype, recording = s

        if not daq_control_set_record(not recording) :
            logger.warning('on_but_record: RECORDING FLAG IS NOT SET')

#--------------------

#    def set_but_record(self, recording=None) :
#        """ Callback from CGWMain.process_zmq_message is used to change button status
#        """
#        logger.debug('DEPRICATED set_but_record: %s' % recording)

#--------------------

    def check_state(self) :
        #logger.debug('check_state -> daq_control_get_state()')
        s = daq_control_get_state()
        if s is None :
            logger.warning('check_state: STATE IS NOT AVAILABLE')
            return
        if s == self.state : return
        self.set_but_ctrls()

#--------------------

    def set_but_enabled(self, but, is_enabled=True) :
        but.setEnabled(is_enabled)
        but.setFlat(not is_enabled)
        #but.setVisible(is_enabled)

    def set_but_record_enabled(self, is_enabled=True) :
        self.set_but_enabled(self.but_record, is_enabled)

#--------------------

    def set_but_ctrls(self, s_status=None) :

        logger.debug('in set_but_ctrls received status %s' % str(s_status))

        s = daq_control_get_status() if s_status is None else s_status
        if s is None :
            logger.warning('set_but_ctrls: STATUS IS NOT AVAILABLE')
            return

        transition, state, cfgtype, recording = s

        #state_zmq = str(s_state).lower() if s_state is not None else None
        #if (s_state is not None) and state_zmq != state :
        #    logger.debug('set_but_ctrls ZMQ msg state:%s inconsistent with current:%s'%\
        #                 (state_zmq,state))

        self.but_record.setIcon(icon.icon_record if recording else icon.icon_record_sym)
        self.set_but_record_enabled(state in ('reset','unallocated','allocated','connected','configured'))

        self.ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        self.state = state 
        #self.but_state.setText('%s since %s' % (s.upper(), self.ts))
        self.but_ctrls.setText(state.upper())

        wpart = cp.cgwmainpartition
        if wpart is not None : wpart.set_buts_enable(state.upper()) # enable/disable button plat in other widget

        self.set_transition(transition)

#--------------------
 
#    def on_timeout(self) :
#        #logger.debug('CGWMainDetector Timeout %.3f sec' % time())
#        self.ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
#        #self.lab_state.setText('Control state on %s' % self.ts)
#        self.check_transition()
#        self.timer.start(1000)

#--------------------

    def check_transition(self) :
        """Uses daq_control_get_status() to get last transition and set the info button status.
        """
        logger.debug('check_transition')
        #t0_sec = time() # takes 0.001s
        s = daq_control_get_status()
        if s is None :
            logger.warning('check_transition: STATUS IS NOT AVAILABLE')
            return

        transition, state, cfgtype, recording = s

        logger.debug('check_transition transition:%s state:%s cfgtype:%s recording:%s'%\
                     (str(transition), str(state), str(cfgtype), str(recording)))
        self.but_transition.setText(transition.upper()) # + ' since %s' % self.ts)
        #state = daq_control_get_state()
        #self.but_state.setText(state.upper() + ' since %s' % self.ts)

#--------------------

    def set_transition(self, s) :
        #ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        #self.but_transition.setText('%s since %s' % (s.upper(), ts))
        self.but_transition.setText(s.upper())

#--------------------

    def closeEvent(self, e) :
        #logger.debug('closeEvent')
        QGroupBox.closeEvent(self, e)
        cp.cgwmaincontrol = None

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
    w = CGWMainControl(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
