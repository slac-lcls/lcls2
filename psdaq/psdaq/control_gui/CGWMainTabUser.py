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
from psdaq.control_gui.CGWMainControl   import CGWMainControl
from psdaq.control_gui.QWIcons import icon
from psdaq.control_gui.Styles import style

from psdaq.control_gui.CGDaqControl import daq_control

#------------------------------

class CGWMainTabUser(QGroupBox) :

    _name = 'CGWMainTabUser'

    s_enabled, s_paused = 'enabled', 'paused'
    s_play_start, s_play_pause, s_play_wait = 'Start', 'Pause', 'Wait'

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

        if state == self.s_enabled :
            self.but_play.setIcon(icon.icon_playback_pause_sym)
            self.but_play.setAccessibleName(self.s_play_pause)

        else : # elif state == self.s_paused :
            self.but_play.setIcon(icon.icon_playback_start_sym)
            self.but_play.setAccessibleName(self.s_play_start)

        self.but_play.setIconSize(QSize(48, 48))
        self.set_tool_tips()
        self.set_but_play_enabled(True) # unlock play button

#------------------------------

    def set_transition(self, s_transition) :
        """interface method called from CGWMain on zmq poll
        """
        logger.debug('In %s.set_transition received state: %s' % (self._name, s_transition))

#------------------------------

    def set_tool_tips(self) :
        self.but_play  .setToolTip('%s triggers'  % self.but_play  .accessibleName())
        self.but_record.setToolTip('%s recording' % self.but_record.accessibleName())

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
        self.setMinimumSize(90, 60)
        #self.setFixedHeight(70)

        self.but_play  .setFixedSize(50, 50)
        self.but_record.setFixedSize(50, 50)

        self.but_play  .setIconSize(QSize(48, 48))
        self.but_record.setIconSize(QSize(48, 48))


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
        is_enabled = state==self.s_enabled

        self.but_play   = QPushButton(icon.icon_playback_pause_sym if is_enabled else\
                                      icon.icon_playback_start_sym, '')
        self.but_record = QPushButton(icon.icon_record_sym, '')         # icon.icon_record

        self.but_play  .setAccessibleName(self.s_play_pause if is_enabled else\
                                          self.s_play_start)
        self.but_record.setAccessibleName(self.status_record[0])

        self.but_play  .clicked.connect(self.on_but_play)
        self.but_record.clicked.connect(self.on_but_record)

        hbox.addWidget(self.but_play)
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

        cmd = self.s_paused if txt==self.s_play_pause else self.s_enabled

        daq_ctrl = daq_control()
        if daq_ctrl is not None :
            daq_ctrl.setState(cmd)
            logger.debug('daq_control.setState("%s")' % cmd)
        else :
            logger.warning('daq_control() is None')

        self.set_but_play_enabled(False) # lock button untill RUNNING status is received


    def set_but_play_enabled(self, is_enabled=True) :
        self.but_play.setEnabled(is_enabled)
        self.but_play.setFlat(not is_enabled)


    def on_but_record(self) :
        txt = self.but_record.accessibleName()
        logger.debug('TBD - on_but_record %s' % txt)
        ind = self.status_record.index(txt)
        ico = icon.icon_record_sym if ind==1 else\
              icon.icon_record
        self.but_record.setIcon(ico)
        self.but_record.setAccessibleName(self.status_record[0 if ind==1 else 1])
        self.set_tool_tips()

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
