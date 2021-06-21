
"""
:py:class:`QWFileNameV2` - widget to enter file name
=====================================================

Usage::

    # Import
    from psana.graphqt.QWFileNameV2 import QWFileNameV2

    # Methods - see test

See:
    - :py:class:`QWFileNameV2`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-12-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""

import os
import sys

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QHBoxLayout, QFileDialog
from PyQt5.QtCore import pyqtSignal

class QWFileNameV2(QWidget):
    """Widget for file name input
    """
    path_is_changed = pyqtSignal('QString')

    def __init__(self, parent=None, label='File:',\
                 path='/cds/group/psdm/detector/data2_test/npy/Select',\
                 mode='r',\
                 fltr='*.txt *.data *.png *.gif *.jpg *.jpeg\n *',\
                 show_frame=False,\
                 but_style_on_start = 'background-color: rgb(100, 255, 100); color: rgb(0, 0, 0);',\
                 but_style_selected = ''):

        QWidget.__init__(self, parent)

        self.mode = mode
        self.path = path
        self.fltr = fltr
        self.show_frame = show_frame
        self.but_style_on_start = but_style_on_start
        self.but_style_selected = but_style_selected

        self.lab = QLabel(label)
        self.but = QPushButton(path.rsplit('/',1)[-1])
        #self.edi = QLineEdit(path)
        #self.edi.setReadOnly(True) 

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.lab)
        #self.hbox.addWidget(self.edi)
        self.hbox.addWidget(self.but)
        self.hbox.addStretch(1)
        self.setLayout(self.hbox)

        self.set_tool_tips()
        self.set_style()

        self.but.clicked.connect(self.on_but)


    def path(self):
        return self.path


    def set_tool_tips(self):
        self.but.setToolTip('Click and select input file.')
        #self.edi.setToolTip('Path to the file (read-only).\nClick on button to change it.') 


    def set_style(self):
        self.setWindowTitle('File name selection widget')
        self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        self.but.setMinimumWidth(250)
        self.setFixedHeight(34)
        self.layout().setContentsMargins(5,0,5,0)
        self.but.setStyleSheet(self.but_style_on_start)


    def on_but(self):
        logger.debug('on_but')

        path_old = self.path

        resp = QFileDialog.getSaveFileName(self, 'Output file', self.path, filter=self.fltr) \
               if self.mode == 'w' else \
               QFileDialog.getOpenFileName(self, 'Input file', self.path, filter=self.fltr)

        logger.debug('response: %s len=%d' % (resp, len(resp)))

        self.path, filter = resp

        dname, fname = os.path.split(self.path)

        if self.mode == 'r' and not os.path.lexists(self.path):
            logger.debug('pass does not exist: %s' % self.path)
            return
            #raise IOError('file %s is not available' % self.path)

        elif dname == '' or fname == '':
            logger.debug('input directiry name "%s" or file name "%s" is empty... use default values'%(dname, fname))
            return

        elif self.path == path_old:
            logger.debug('path has not been changed: %s' % str(self.path))
            return

        else:
            logger.debug('selected file: %s' % self.path)
            #self.edi.setText(self.path)
            self.but.setText(self.path.rsplit('/',1)[-1])
            self.path_is_changed.emit(self.path)
            self.but.setStyleSheet(self.but_style_selected)


    def connect_path_is_changed_to_recipient(self, recip):
        self.path_is_changed['QString'].connect(recip)

 
    def test_signal_reception(self, s):
        logger.debug('test_signal_reception: %s' % s)


if __name__ == "__main__":

    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = QWFileNameV2(None, label='Path:', path='/cds/group/psdm/detector/data2_test/npy/Select', show_frame=True)
    w.setGeometry(100, 50, 350, 80)
    w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

# EOF
