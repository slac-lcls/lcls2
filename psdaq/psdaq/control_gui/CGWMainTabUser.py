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

import json
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

    status_cmd = ['running', 'paused']
    status_play = ['Start', 'Stop']
    status_record = ['Begin', 'End']

    def __init__(self, **kwargs) :

        parent      = kwargs.get('parent', None)
        parent_ctrl = kwargs.get('parent_ctrl', None)

        #QWidget.__init__(self, parent=None)
        QGroupBox.__init__(self, 'Control', parent)

        logger.debug('In %s' % self._name)

        icon.set_icons()

        #self.hbox = self.hbox_test()
        self.hbox = self.hbox_buttons()

        #parent_ctrl.wpart = None
        #parent_ctrl.wcoll = None
        #parent_ctrl.wctrl = None

        #self.hbox.addWidget(self.but_record)
        #self.hbox.addWidget(self.but_pause)
        self.setLayout(self.hbox)

        self.set_style()
        self.set_tool_tips()

        #print(dir(self))
        print(self.frameGeometry())
        print(self.frameSize())


#------------------------------

    def set_tool_tips(self) :
        self.but_play  .setToolTip('%s running'   % self.but_play  .accessibleName())
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

        self.but_play  .setFixedSize(40, 40)
        self.but_record.setFixedSize(40, 40)

        #self.setGeometry(self.main_win_pos_x .value(),\
        #                 self.main_win_pos_y .value(),\
        #                 self.main_win_width .value(),\
        #                 self.main_win_height.value())
        #w_height = self.main_win_height.value()


    def closeEvent(self, e) :
        print('%s.closeEvent' % self._name)

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
        self.but_play   = QPushButton(icon.icon_playback_start_sym, '') # icon.icon_playback_pause_sym
        self.but_record = QPushButton(icon.icon_record_sym, '')         # icon.icon_record

        state = daq_control().getState()
        logger.debug('current state is "%s"' % state)

        self.but_play  .setAccessibleName('Stop' if state=='running' else 'Start')
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
        ind = self.status_play.index(txt)
        ico = icon.icon_playback_start_sym if ind==1 else\
              icon.icon_playback_pause_sym
        self.but_play.setAccessibleName(self.status_play[0 if ind==1 else 1])
        self.but_play.setIcon(ico)
        self.set_tool_tips()

        state = daq_control().getState()
        logger.debug('current state is %s' % state)

        #status_cmd = ['running', 'paused']
        #status_play = ['Start', 'Stop']

        cmd = 'paused' if txt=='Stop' else 'running'
        daq_control().setState(cmd)
        logger.debug('daq_control.setState("%s")' % cmd)


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

    import sys
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    kwargs = {'parent':None, 'parent_ctrl':None}
    w = CGWMainTabUser(**kwargs)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
