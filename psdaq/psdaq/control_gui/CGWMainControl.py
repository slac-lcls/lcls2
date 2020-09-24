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
from psdaq.control_gui.CGDaqControl import daq_control_set_state, daq_control_set_record, DaqControl
                                           #daq_control_get_state, daq_control_get_status
from psdaq.control_gui.CGConfigParameters import cp
from psdaq.control_gui.QWProgressBar import QWProgressBar

#--------------------

class CGWMainControl(QGroupBox):
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

        self.but_record = QPushButton(icon.icon_record_start, '') # icon.icon_record_stop
        self.lab_record = QLabel('Recording')

        self.box_state      = QComboBox()
        self.but_transition = QPushButton('Unknown')
        self.but_ctrls      = QPushButton('Ready')
        self.bar_progress   = QWProgressBar() # label='', vmin=0, vmax=100

        self.states = ['Select',] + [s.upper() for s in DaqControl.states]
        self.box_state.addItems(self.states)

        self.state_is_after_reset = False

        if False:
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
            self.hbox3.addWidget(self.bar_progress, 0, Qt.AlignCenter)
            self.hbox3.addStretch(1)
            self.hbox3.addWidget(self.but_transition, 0, Qt.AlignCenter)
        
            self.vbox = QVBoxLayout() 
            self.vbox.addLayout(self.hbox1)
            self.vbox.addLayout(self.hbox2)
            self.vbox.addLayout(self.hbox3)
        
            self.setLayout(self.vbox)

        else:
            self.grid = QGridLayout()
            self.grid.addWidget(self.lab_record,      0, 0, 1, 1)
            self.grid.addWidget(self.but_record,      0, 4, 1, 1)
            self.grid.addWidget(self.lab_state,       1, 0, 1, 1)
            self.grid.addWidget(self.lab_trans,       1, 9, 1, 1)
            self.grid.addWidget(self.box_state,       2, 0, 1, 1)
            self.grid.addWidget(self.bar_progress,    2, 4, 1, 3)
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
        self.set_buts_enabled()

#--------------------

    def set_tool_tips(self):
        self.setToolTip('Configuration') 
        self.but_record.setToolTip('sets flag for recording')
        self.lab_record.setText('Recording:')
        self.box_state.setToolTip('Select desirable state.')
        self.but_transition.setToolTip('Last transition info.')
        self.but_ctrls.setToolTip('State info.') 

#--------------------

    def set_style(self):
        self.setStyleSheet(style.qgrbox_title)
        self.lab_record.setFixedWidth(100)
        self.but_record.setFixedSize(50, 50)
        self.but_record.setIconSize(QSize(48, 48))
        self.bar_progress.setFixedWidth(100)
        self.bar_progress.setVisible(False)
        self.but_ctrls.setStyleSheet(style.styleButtonGood)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.layout().setContentsMargins(4,4,4,4)
        self.setMinimumSize(270,140)

#--------------------

    def update_progress_bar(self, value=0.3, is_visible=False, trans_name=''):
        self.bar_progress.setVisible(is_visible)
        self.bar_progress.set_value(value)
        self.bar_progress.set_label(trans_name)

#--------------------

    def sizeHint(self):
        return QSize(270,160)
 
#--------------------
 
    def on_box_state(self, ind):
        if not ind: return
        state = self.states[ind]

        if self.state_is_after_reset: cp.cgwmain.wlogr.add_separator_err()

        logger.info('-> daq_control_set_state(%s)' % state)
        if not daq_control_set_state(state.lower()):
            logger.warning('on_box_state: STATE %s IS NOT SET' % state)
        self.state_is_after_reset = (state=='RESET')

#--------------------
 
    def on_but_transition(self):
        #logger.debug('on_but_transition') # NO ACTION')
        self.check_transition()

#--------------------

    def on_but_ctrls(self):
        logger.debug('on_but_ctrls')
        self.check_state()

#--------------------
 
    def on_cbx_runc(self, ind):
        #if self.cbx.hasFocus():
        cbx = self.cbx_runc
        tit = cbx.text()
        self.cbx_runc.setStyleSheet(style.styleGreenish if cbx.isChecked() else style.styleYellowBkg)
        msg = 'Check box "%s" is set to %s' % (tit, cbx.isChecked())
        logger.info(msg)

#--------------------

    def on_but_record(self):
        logger.debug('on_but_record')

        if not daq_control_set_record(not cp.s_recording):
            logger.warning('on_but_record: RECORDING FLAG IS NOT SET')

#--------------------

#    def set_but_record(self, recording=None):
#        """ Callback from CGWMain.process_zmq_message is used to change button status
#        """
#        logger.debug('DEPRICATED set_but_record: %s' % recording)

#--------------------

    def check_state(self):
        #logger.debug('check_state -> daq_control_get_state()')
        s = cp.s_state # daq_control_get_state()
        if s is None:
            logger.warning('check_state: STATE IS NOT AVAILABLE')
            return
        if s == self.state: return
        self.set_buts_enabled()

#--------------------

    def set_but_enabled(self, but, is_enabled=True):
        but.setEnabled(is_enabled)
        but.setFlat(not is_enabled)
        #but.setVisible(is_enabled)

    def set_but_record_enabled(self, is_enabled=True):
        self.set_but_enabled(self.but_record, is_enabled)

#--------------------

    def set_buts_enabled(self):

        status = transition, state, cfgtype, recording =\
             (cp.s_transition, cp.s_state, cp.s_cfgtype, cp.s_recording)

        logger.debug('in set_buts_enabled current status %s' % str(status))

        self.but_record.setIcon(icon.icon_record_stop if recording else icon.icon_record_start)
        self.set_but_record_enabled(state in ('reset','unallocated','allocated','connected','configured'))
        self.box_state.setEnabled(state in ('allocated','connected','configured','started','paused','running'))

        self.ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        self.state = state 
        #self.but_state.setText('%s since %s' % (s.upper(), self.ts))
        self.but_ctrls.setText(state.upper() if state is not None else 'None')

        self.set_transition(transition)

#--------------------
 
#    def on_timeout(self):
#        #logger.debug('CGWMainDetector Timeout %.3f sec' % time())
#        self.ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
#        #self.lab_state.setText('Control state on %s' % self.ts)
#        self.check_transition()
#        self.timer.start(1000)

#--------------------

    def check_transition(self):
        """Uses cp.cached parameters to get last transition and set the info button status.
        """
        logger.debug('check_transition transition:%s state:%s cfgtype:%s recording:%s'%\
                     (cp.s_transition, cp.s_state, cp.s_cfgtype, cp.s_recording))
        self.but_transition.setText(cp.s_transition.upper() if cp.s_transition is not None else 'None')

#--------------------

    def set_transition(self, s):
        #ts = gu.str_tstamp(fmt='%H:%M:%S', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
        #self.but_transition.setText('%s since %s' % (s.upper(), ts))
        self.but_transition.setText(s.upper() if s is not None else None)

#--------------------

    def closeEvent(self, e):
        #logger.debug('closeEvent')
        QGroupBox.closeEvent(self, e)
        cp.cgwmaincontrol = None

#--------------------

    if __name__ == "__main__":
 
      def resizeEvent(self, e):
        print('CGWMainControl.resizeEvent: %s' % str(self.size()))

#--------------------
#--------------------
#--------------------
#--------------------
 
if __name__ == "__main__":

    from psdaq.control_gui.CGDaqControl import daq_control, DaqControlEmulator, Emulator
    daq_control.set_daq_control(DaqControlEmulator())

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    cp.test_cpinit()
    w = CGWMainControl(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
