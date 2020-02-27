"""
Class :py:class:`CGWMainTabUser` is a QWidget for interactive image
=======================================================================

Usage ::

    import sys
    from PyQt5.QtWidgets import QApplication
    from psdaq.control_gui.CGWMainTabUser import CGWMainTabUser
    app = QApplication(sys.argv)
    w = CGWMainTabUser(None, app)
    w.show()
    app.exec_()

See:
    - :class:`CGWMainTabUser`
    - :class:`CGWMainPartition`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

Created on 2019-05-07 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

#------------------------------

from time import time

from PyQt5.QtWidgets import QGroupBox, QPushButton, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit, QSizePolicy
from PyQt5.QtCore import Qt, QSize

from psdaq.control_gui.CGWMainPartition import CGWMainPartition
from psdaq.control_gui.QWIcons import icon
from psdaq.control_gui.Styles import style

from psdaq.control_gui.CGDaqControl import daq_control_set_state, daq_control_set_record
                                           #, daq_control_get_status, daq_control_get_state
from psdaq.control_gui.CGConfigParameters import cp

#------------------------------

class CGWMainTabUser(QGroupBox) :

    _name = 'CGWMainTabUser'

    s_running, s_paused = 'running', 'paused'
    s_play_start, s_play_pause, s_play_wait, s_play_stop = 'Start', 'Pause', 'Wait', 'Stop'

    #status_record = ['Begin', 'End', 'Wait']

    def __init__(self, **kwargs) :

        parent      = kwargs.get('parent', None)

        #QWidget.__init__(self, parent=None)
        QGroupBox.__init__(self, 'Control', parent)

        cp.cgwmaintabuser = self

        logger.debug('In %s' % self._name)
        icon.set_icons()
        self.hbox = self.hbox_buttons()
        self.setLayout(self.hbox)

        self.set_style()
        self.set_tool_tips()
        self.set_but_ctrls()

#------------------------------

    def set_but_ctrls(self) :
        """interface method sets button states,
           is called from CGWMain on zmq poll and at initialization of the object.
        """
        transition, state, cfgtype, recording =\
           cp.s_transition, cp.s_state, cp.s_cfgtype, cp.s_recording

        logger.debug('set_but_ctrls transition:%s state:%s config:%s recording:%s'%\
                      (transition, state, cfgtype, recording))

        #s = daq_control_get_status() if s_status is None else s_status
        #if s is None :
        #    logger.warning('set_but_ctrls: STATUS IS NOT AVAILABLE')
        #    return

        #state_zmq = str(s_state).lower() if s_state is not None else None
        #if (s_state is not None) and state_zmq != state :
        #    logger.debug('set_but_ctrls ZMQ msg state:%s inconsistent with current:%s'%\
        #                 (state_zmq,state))

        if state == self.s_running :
            self.but_play.setIcon(icon.icon_playback_pause_sym)
            #self.but_play.setAccessibleName(self.s_play_pause)
        else :
            self.but_play.setIcon(icon.icon_playback_start_sym)
            #self.but_play.setAccessibleName(self.s_play_start)

        self.but_play.setIconSize(QSize(48, 48))
        self.set_tool_tips()

        self.but_record.setIcon(icon.icon_record_stop if recording else icon.icon_record_start)

        self.set_but_play_enabled(True)   # unlock play button
        self.set_but_record_enabled(state in ('reset','unallocated','allocated','connected','configured'))
        #self.set_but_stop_enabled(state in ('starting', 'paused', 'running'))
        self.but_stop.setVisible(state in ('starting', 'paused', 'running'))

#------------------------------

#    def set_transition(self, s_transition) :
#        """interface method called from CGWMain on zmq poll
#        """
#        logger.debug('in set_transition received state: %s' % s_transition)

#------------------------------

    def set_tool_tips(self) :
        self.but_play  .setToolTip('Start/Pause running')
        self.but_record.setToolTip('ON/OFF recording')
        self.but_stop  .setToolTip('Stop running')

#--------------------

    def sizeHint(self):
        return QSize(110, 70)

#--------------------

    def set_style(self) :

        self.setStyleSheet(style.qgrbox_title)
        #self.setCheckable(True)
        self.layout().setContentsMargins(0,0,0,0)
        self.layout().setSpacing(0)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.setMinimumSize(145, 60)
        #self.setMinimumSize(90, 60)

        self.but_play  .setFixedSize(50, 50)
        self.but_record.setFixedSize(50, 50)
        self.but_stop  .setFixedSize(50, 50)

        self.but_play  .setIconSize(QSize(48, 48))
        self.but_record.setIconSize(QSize(48, 48))
        self.but_stop  .setIconSize(QSize(64, 64))


    def closeEvent(self, e) :
        logger.debug('%s.closeEvent' % self._name)

        try :
            pass
            #self.wpart.close()
            #self.wctrl.close()
        except Exception as ex:
            print('Exception: %s' % ex)

 
    #def resizeEvent(self, e):
        #logger.debug('resizeEvent', self._name) 

    #def moveEvent(self, e) :
        #logger.debug('moveEvent', self._name) 


    def hbox_buttons(self) :
        hbox = QHBoxLayout()

        #state = daq_control_get_state()
        #logger.debug('current state is "%s"' % state)

        #is_running = state==self.s_running

        #self.but_play   = QPushButton(icon.icon_playback_pause_sym if is_running else\
        #                              icon.icon_playback_start_sym, '')
        #self.but_record = QPushButton(icon.icon_record_start, '')         # icon.icon_record_stop
        #self.but_stop   = QPushButton(icon.icon_playback_stop_sym, '')

        #self.but_play  .setAccessibleName(self.s_play_pause if is_running else\
        #                                  self.s_play_start)
        #self.but_record.setAccessibleName(self.status_record[0])
        #self.but_stop  .setAccessibleName(self.s_play_stop)

        self.but_play   = QPushButton(icon.icon_playback_pause_sym, '')
        self.but_record = QPushButton(icon.icon_record_start, '')
        self.but_stop   = QPushButton(icon.icon_playback_stop_sym, '')

        self.but_play  .setAccessibleName('play')
        self.but_record.setAccessibleName('record')
        self.but_stop  .setAccessibleName('stop')

        self.but_play  .clicked.connect(self.on_but_play)
        self.but_record.clicked.connect(self.on_but_record)
        self.but_stop  .clicked.connect(self.on_but_stop)

        hbox.addSpacing(24) 
        hbox.addWidget(self.but_play,   alignment=Qt.AlignLeft)
        hbox.addWidget(self.but_stop,   alignment=Qt.AlignCenter)
        hbox.addWidget(self.but_record, alignment=Qt.AlignRight)
        hbox.addSpacing(24) 
        #hbox.addStretch(1) 

        #self.but_play.setStyleSheet('QPushButton{border: 0px solid;}')
        #self.but_record.setStyleSheet("background-image: url('image.jpg'); border: none;")
        return hbox


    def on_but_play(self) :
        #txt = self.but_play.accessibleName()
        logger.debug('on_but_play')

        # submit command for "running" or "pause"
        transition, state, cfgtype, recording =\
           cp.s_transition, cp.s_state, cp.s_cfgtype, cp.s_recording

        cmd = self.s_paused if state==self.s_running else self.s_running
        if not daq_control_set_state(cmd):
            logger.warning('on_but_play: STATE %s IS NOT SET' % cmd)
            return

        # set WAIT temporary icon
        ico = icon.icon_wait
        self.but_play.setIconSize(QSize(32, 32) if ico == icon.icon_wait else QSize(48, 48))
        #self.but_play.setAccessibleName(self.s_play_wait)
        self.but_play.setIcon(ico)
        self.set_tool_tips()
        self.set_but_play_enabled(False) # lock button untill RUNNING status is received


    def set_but_enabled(self, but, is_enabled=True) :
        but.setEnabled(is_enabled)
        but.setFlat(not is_enabled)
        #but.setVisible(is_enabled)


    def set_but_play_enabled(self, is_enabled=True) :
        self.set_but_enabled(self.but_play, is_enabled)


    def set_but_stop_enabled(self, is_enabled=True) :
        self.set_but_enabled(self.but_stop, is_enabled)


    def set_but_record_enabled(self, is_enabled=True) :
        self.set_but_enabled(self.but_record, is_enabled)


    def on_but_record(self) :
        logger.debug('on_but_record')

        #cp.s_transition, cp.s_state, cp.s_cfgtype, cp.s_recording

        if not daq_control_set_record(not cp.s_recording) :
            logger.warning('on_but_record: RECORDING FLAG IS NOT SET')


#    def set_but_record(self, recording=False) :
#        """ Callback from CGWMain.process_zmq_message is used to change button status
#        """
#        logger.debug('DEPRICATED set_but_record to: %s' % recording)

    def update_progress_bar(self, value=0, is_visible=False) :
        """ is used in CGWMainControl ONLY
        """
        pass 
        #self.bar_progress.setVisible(is_visible)
        #self.bar_progress.set_value(value)

    def on_but_stop(self) :
        #txt = self.but_stop.accessibleName()
        logger.debug('on_but_stop')
        daq_control_set_state('configured')
        #self.set_but_stop_enabled(False) # set depending on state

#--------------------

    def closeEvent(self, e) :
        #logger.debug('closeEvent')
        QGroupBox.closeEvent(self, e)
        cp.cgwmaintabuser = None

#------------------------------
#------------------------------
#------------------------------
#------------------------------

    if __name__ == "__main__" :

      def hbox_test(self) :
        list_of_icons = [\
          icon.icon_eject\
        , icon.icon_eject_sym\
        , icon.icon_playback_pause\
        , icon.icon_playback_pause_sym\
        , icon.icon_playback_start\
        , icon.icon_playback_start_rtl\
        , icon.icon_playback_start_sym_rtl\
        , icon.icon_playback_start_sym\
        , icon.icon_playback_stop\
        , icon.icon_playback_stop_sym\
        , icon.icon_record_stop\
        , icon.icon_record_start\
        , icon.icon_seek_backward\
        , icon.icon_seek_backward_rtl\
        , icon.icon_seek_backward_sym_rtl\
        , icon.icon_seek_backward_sym\
        , icon.icon_seek_forward\
        , icon.icon_seek_forward_rtl\
        , icon.icon_seek_forward_sym_rtl\
        , icon.icon_seek_forward_sym\
        , icon.icon_skip_backward\
        , icon.icon_skip_backward_rtl\
        , icon.icon_skip_backward_sym_rtl\
        , icon.icon_skip_backward_sym\
        , icon.icon_skip_forward\
        , icon.icon_skip_forward_rtl\
        , icon.icon_skip_forward_sym_rtl\
        , icon.icon_skip_forward_sym\
        , icon.icon_view_subtitles_sym\
        ]
        hbox = QHBoxLayout() 
        for ico in list_of_icons :
            but = QPushButton(ico, '')
            but.setFixedWidth(25)
            hbox.addWidget(but)
        return hbox

#------------------------------

if __name__ == "__main__" :

    from psdaq.control_gui.CGDaqControl import daq_control, DaqControlEmulator, Emulator
    daq_control.set_daq_control(DaqControlEmulator())

    import sys
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    kwargs = {'parent':None}
    w = CGWMainTabUser(**kwargs)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
