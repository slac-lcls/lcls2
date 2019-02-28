#--------------------
"""
:py:class:`CGWMainPartition` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainPartition import CGWMainPartition

    # Methods - see test

See:
    - :py:class:`CGWMainPartition`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QPushButton, QHBoxLayout # , QWidget,  QLabel, QLineEdit, QFileDialog
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

from psdaq.control_gui.CGWPartitionSelection import CGWPartitionSelection
from psdaq.control_gui.QWDialog import QDialog, QWDialog
from psdaq.control_gui.CGDaqControl import daq_control, DaqControl #, worker_set_state

#--------------------

#class CGWMainPartition(QWidget) :
class CGWMainPartition(QGroupBox) :
    """
    """
    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Partition', parent)

        self.but_plat    = QPushButton('Raw call')
        self.but_select  = QPushButton('Select')
        self.but_display = QPushButton('Display')

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.but_plat)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_select)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_display)
        self.setLayout(self.hbox)

        self.set_tool_tips()
        self.set_style()

        self.but_plat.clicked.connect(self.on_but_plat)
        self.but_select.clicked.connect(self.on_but_select)
        self.but_display.clicked.connect(self.on_but_display)

        self.w_select = None
        self.w_display = None

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Partition GUI')
        self.but_select.setToolTip('Click on button.') 
        self.but_plat.setToolTip('Submits "plat" command.')

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)

        #self.setWindowTitle('File name selection widget')
        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 
        #self.layout().setContentsMargins(0,0,0,0)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
    def on_but_select(self):
        logger.debug('on_but_select')

        #w_select = QLineEdit('Test window')
        w_select = CGWPartitionSelection()
        w_dialog = QWDialog(self.but_select, w_select)
        w_dialog.setWindowTitle('Partition Selection')
        #w_dialog.setGeometry(20, 40, 500, 200)

        #w.show()
        resp=w_dialog.exec_()
        logger.debug('resp=%s' % resp)
        logger.debug('QtWidgets.QDialog.Rejected: %d' % QDialog.Rejected)
        logger.debug('QtWidgets.QDialog.Accepted: %d' % QDialog.Accepted)

        del w_dialog
        del w_select

#--------------------
 
    def on_but_display(self):
        logger.debug('on_but_display')
        if self.w_display is None :
            self.w_display = CGWPartitionSelection(parent=None, parent_ctrl=self)
            self.w_display.show()
        else :
            self.w_display.close()
            self.w_display = None

#--------------------
 
    def on_but_plat(self) :
        """Equivalent to CLI: daqstate -p6 --transition plat
           https://github.com/slac-lcls/lcls2/blob/collection_front/psdaq/psdaq/control/daqstate.py
        """
        logger.debug('on_but_plat - command to set transition "plat"')
        rv = daq_control().setTransition('plat')
        if rv is not None : logger.error('Error: %s' % rv)

#--------------------

    def set_but_plat(self, s) :
        logger.debug('set_but_plat for state %s' % s)
        self.but_plat.setEnabled(s.upper() in ('RESET', 'UNALLOCATED'))

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainPartition(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
