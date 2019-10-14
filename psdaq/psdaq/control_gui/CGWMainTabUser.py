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
#from psdaq.control_gui.CGWMainControl   import CGWMainControl
from psdaq.control_gui.QWIcons import icon
from psdaq.control_gui.Styles import style

from psdaq.control_gui.CGDaqControl import daq_control

#------------------------------

class CGWMainTabUser(QGroupBox) :

    _name = 'CGWMainTabUser'

    s_running, s_paused = 'running', 'paused'
    s_play_start, s_play_pause, s_play_wait, s_play_stop = 'Start', 'Pause', 'Wait', 'Stop'

    status_record = ['Begin', 'End', 'Wait']

    def __init__(self, **kwargs) :

        parent      = kwargs.get('parent', None)
        parent_ctrl = kwargs.get('parent_ctrl', None)

        #QWidget.__init__(self, parent=None)
        QGroupBox.__init__(self, 'Control', parent)

        logger.debug('In %s' % self._name)

        icon.set_icons()

        #self.hbox = self.hbox_test()
        self.hbox = self.hbox_buttons()

        parent_ctrl.wctrl = self

        self.setLayout(self.hbox)

        self.set_style()
        self.set_tool_tips()

#------------------------------

    def set_but_ctrls(self, s_state) :
        """interface method called from CGWMain on zmq poll
        """
        logger.debug('In %s.set_but_ctrls received state: %s' % (self._name, s_state))
        state =  s_state.lower()

        if state == self.s_running :
            self.but_play.setIcon(icon.icon_playback_pause_sym)
            self.but_play.setAccessibleName(self.s_play_pause)

        else : # elif state == self.s_paused :
            self.but_play.setIcon(icon.icon_playback_start_sym)
            self.but_play.setAccessibleName(self.s_play_start)

        self.but_play.setIconSize(QSize(48, 48))
        self.set_tool_tips()
        self.set_but_play_enabled(True) # unlock play button

        self.set_but_stop_enabled(state in ('starting', 'paused', 'running'))

        #daq_ctrl = daq_control()
        #if daq_ctrl is not None :
        #    transition, state, cfgtype, recording = daq_control.getStatus()
        #    logger.debug('CGWMainTabUser.set_but_ctrls transition:%s state:%s config_alias:%s recording:%s'%\
        #                 (str(transition), str(state), str(config_alias), str(recording)))
        #else :
        #    logger.warning('CGWMainTabUser.set_but_ctrls daq_control is None')

#------------------------------

    def set_transition(self, s_transition) :
        """interface method called from CGWMain on zmq poll
        """
        logger.debug('In %s.set_transition received state: %s' % (self._name, s_transition))

#------------------------------

    def set_tool_tips(self) :
        self.but_play  .setToolTip('%s running'  % self.but_play.accessibleName())
        self.but_record.setToolTip('%s recording' % self.but_record.accessibleName())
        self.but_stop  .setToolTip('%s running and request state "configured"' % self.but_stop.accessibleName())

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

        daq_ctrl = daq_control()
        state = daq_ctrl.getState() if daq_ctrl is not None else 'unknown'
        logger.debug('current state is "%s"' % state)
        is_running = state==self.s_running

        self.but_play   = QPushButton(icon.icon_playback_pause_sym if is_running else\
                                      icon.icon_playback_start_sym, '')
        self.but_record = QPushButton(icon.icon_record_sym, '')         # icon.icon_record
        self.but_stop   = QPushButton(icon.icon_playback_stop_sym, '')

        self.but_play  .setAccessibleName(self.s_play_pause if is_running else\
                                          self.s_play_start)
        self.but_record.setAccessibleName(self.status_record[0])
        self.but_stop  .setAccessibleName(self.s_play_stop)

        self.but_play  .clicked.connect(self.on_but_play)
        self.but_record.clicked.connect(self.on_but_record)
        self.but_stop  .clicked.connect(self.on_but_stop)

        hbox.addWidget(self.but_play)
        hbox.addWidget(self.but_stop)
        hbox.addWidget(self.but_record)

        #self.but_play.setStyleSheet('QPushButton{border: 0px solid;}')
        #self.but_record.setStyleSheet("background-image: url('image.jpg'); border: none;")
        return hbox


    def on_but_play(self) :
        txt = self.but_play.accessibleName()
        logger.debug('on_but_play %s' % txt)
        #ind = self.status_play.index(txt)
        #ico = icon.icon_playback_start_sym if txt==self.status_play_start else icon.icon_wait
              #icon.icon_playback_pause_sym

        ico = icon.icon_wait
        self.but_play.setIconSize(QSize(32, 32) if ico == icon.icon_wait else QSize(48, 48))
        self.but_play.setAccessibleName(self.s_play_wait)
        #self.but_play.setAccessibleName(self.status_play[0 if ind==1 else 1])

        self.but_play.setIcon(ico)
        self.set_tool_tips()

        daq_ctrl = daq_control()
        state = daq_ctrl.getState() if daq_ctrl is not None else 'unknown'
        logger.debug('current state is %s' % state)

        cmd = self.s_paused if txt==self.s_play_pause else self.s_running

        daq_ctrl = daq_control()
        if daq_ctrl is not None :
            daq_ctrl.setState(cmd)
            logger.debug('daq_control.setState("%s")' % cmd)
        else :
            logger.warning('daq_control() is None')

        self.set_but_play_enabled(False) # lock button untill RUNNING status is received


    def set_but_enabled(self, but, is_enable=True) :
        but.setEnabled(is_enable)
        but.setFlat(not is_enable)


    def set_but_play_enabled(self, is_running=True) :
        self.set_but_enabled(self.but_play, is_running)


    def set_but_stop_enabled(self, is_enable=True) :
        self.set_but_enabled(self.but_stop, is_enable)


    def on_but_record(self) :
        txt = self.but_record.accessibleName()
        logger.debug('on_but_record %s' % txt)
        ind = self.status_record.index(txt) # 0/1/2 = Begin/End/Wait

        daq_ctrl = daq_control()
        if daq_ctrl is not None :
            daq_ctrl.setRecord(ind==0) # switches button record state
            logger.debug('on_but_record daq_control.setRecord("%s")' % (ind==0))
        else :
            logger.warning('on_but_record daq_control() is None')


    def set_but_record(self, recording=False) :
        """ Callback from CGWMain.process_zmq_message is used to change button status
        """
        txt = self.but_record.accessibleName()
        logger.debug('CGWMainTabUser.set_but_record status: %s request: %s' % (txt,recording))
        ind = self.status_record.index(txt) # 0/1/2 = Begin/End/Wait

        if ind==1 and recording       : return # recording state has not changed
        if ind==0 and (not recording) : return # recording state has not changed

        ico = icon.icon_record_sym if recording else\
              icon.icon_record
        self.but_record.setIcon(ico)
        self.but_record.setAccessibleName(self.status_record[0 if recording else 1])
        self.set_tool_tips()


    def on_but_stop(self) :
        txt = self.but_stop.accessibleName()
        logger.debug('on_but_stop %s' % txt)
        cmd = 'configured'
        daq_ctrl = daq_control()
        if daq_ctrl is not None :
            daq_ctrl.setState(cmd)
            logger.debug('on_but_stop daq_control.setState("%s")' % cmd)
        else :
            logger.warning('on_but_stop daq_control() is None')
        #self.set_but_stop_enabled(False) # set depending on state

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
        , icon.icon_record\
        , icon.icon_record_sym\
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

    kwargs = {'parent':None, 'parent_ctrl':Emulator()}
    w = CGWMainTabUser(**kwargs)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
