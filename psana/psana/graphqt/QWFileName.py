#------------------------------
"""
:py:class:`QWFileName` - widget to enter file name
============================================================================================

Usage::

    # Import
    from psana.graphqt.QWFileName import QWFileName

    # Methods - see test

See:
    - :py:class:`QWFileName`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-12-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""
#------------------------------

import os
import sys

import logging
logger = logging.getLogger(__name__)

#from psana.graphqt.Frame import Frame
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QHBoxLayout, QFileDialog
from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

#------------------------------

class QWFileName(QWidget) :
    """Widget for file name input
    """
    path_is_changed = pyqtSignal('QString')

    def __init__(self, parent=None, butname='Browse', label='File:',\
                 path='/reg/neh/home/dubrovin/LCLS/rel-expmon/log.txt',\
                 mode='r',\
                 fltr='*.txt *.data *.png *.gif *.jpg *.jpeg\n *',\
                 show_frame=False) :

        QWidget.__init__(self, parent)
        #Frame.__init__(self, parent, mlw=1, vis=show_frame)

        self.mode = mode
        self.path = path
        self.fltr = fltr
        self.show_frame = show_frame

        self.lab = QLabel(label)
        self.but = QPushButton(butname)
        self.edi = QLineEdit(path)
        self.edi.setReadOnly(True) 

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.lab)
        self.hbox.addWidget(self.edi)
        self.hbox.addWidget(self.but)
        self.hbox.addStretch(1)
        self.setLayout(self.hbox)

        self.set_tool_tips()
        self.set_style()

        self.but.clicked.connect(self.on_but)

#------------------------------

    def path(self):
        return self.path

#------------------------------

    def set_tool_tips(self) :
        self.but.setToolTip('Select input file.')
        self.edi.setToolTip('Path to the file (read-only).\nClick on button to change it.') 

#------------------------------

    def set_style(self) :
        self.setWindowTitle('File name selection widget')
        self.setMinimumWidth(300)
        self.edi.setMinimumWidth(210)
        self.setFixedHeight(50 if self.show_frame else 34)
        if not self.show_frame : self.setContentsMargins(-9,-9,-9,-9)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#------------------------------
 
    def on_but(self):
        logger.debug('on_but')
        resp = QFileDialog.getSaveFileName(self, 'Output file', self.path, filter=self.fltr) \
               if self.mode == 'w' else \
               QFileDialog.getOpenFileName(self, 'Input file', self.path, filter=self.fltr)

        logger.debug('response: %s len=%d' % (resp, len(resp)))

        self.path, filter = resp

        dname, fname = os.path.split(self.path)

        if self.mode == 'r' and not os.path.lexists(self.path) :
            logger.debug('pass does not exist: %s' % self.path)
            return
            #raise IOError('File %s is not available' % self.path)

        elif dname == '' or fname == '' :
            logger.debug('Input directiry name or file name is empty... use default values')
            return

        else :
            logger.debug('Selected file: %s' % self.path)
            self.edi.setText(self.path)
            self.path_is_changed.emit(self.path)

#------------------------------

    def connect_path_is_changed_to_recipient(self, recip) :
        self.path_is_changed['QString'].connect(recip)

#------------------------------
 
    def test_signal_reception(self, s) :
        logger.debug('test_signal_reception: %s' % s)

#------------------------------

if __name__ == "__main__" :

    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = QWFileName(None, butname='Select', label='Path:',\
                   path='/reg/neh/home/dubrovin/LCLS/rel-expmon/log.txt', show_frame=True)
    w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#------------------------------
