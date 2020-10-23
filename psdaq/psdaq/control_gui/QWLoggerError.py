#------------------------------
"""
:py:class:`QWLoggerError` - logger ERROR window
=======================================================

Usage::
    # Test: python lcls2/psdaq/psdaq/control_gui/QWLoggerError.py

    # Import
    from psdaq.control_gui.QWLoggerError import QWLoggerError

    # Methods - see test

See:
    - :py:class:`QWLoggerError`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-10-22 by Mikhail Dubrovin
"""
#----

import logging

#from psalg.utils.syslog import SysLog

logger = logging.getLogger() # need in root to intercept messages from all other loggers
##logger = logging.getLogger(__name__)

#from psdaq.control_gui.CGConfigParameters import cp

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QGroupBox, QTextEdit, QHBoxLayout, QSizePolicy
#,QWidget, QLabel, QPushButton, QComboBox, QHBoxLayout, QFileDialog, QSplitter
from PyQt5.QtGui import QTextCursor, QColor
#import psdaq.control_gui.Utils as gu
from psdaq.control_gui.Styles import style

#----

class QWLoggerError(QGroupBox):
    """
    """
    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Error messages', parent)

        #cp.qwloggererror = self

        self.edi_err = QTextEdit()
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.edi_err)
        self.setLayout(self.hbox)
        self.set_style()
        #self.set_tool_tips()

    def set_tool_tips(self):
        self.edi_err.setToolTip('Window for ERROR messages')

    def set_style(self):
        self.edi_err.setReadOnly(True)
        self.edi_err.setStyleSheet(style.styleYellowish) 
        self.edi_err.setMinimumHeight(50)
        self.edi_err.setTextColor(Qt.red)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        self.setStyleSheet(style.qgrbox_title)
        self.layout().setContentsMargins(2,0,2,2)

    def sizeHint(self):
        return QSize(300,300)

 
    def append_qwlogger_err(self, msg='...'):
        self.edi_err.append(msg)
        self.scroll_down()


    def add_separator_err(self, sep='\n\n\n\n\n%s'%(50*'_')):
        self.append_qwlogger_err(msg=sep)


    def scroll_down(self):
        #logger.debug('scroll_down')
        self.edi_err.moveCursor(QTextCursor.End)
        self.edi_err.repaint()
        #self.edi_err.update()


    def setReadOnly(self, state):
        self.edi_err.setReadOnly(state)


    def setStyleSheet(self, s):
        QGroupBox.setStyleSheet(self, s)
         #self.edi_err.setStyleSheet(s)


    def setTextColor(self, c):
        self.edi_err.setTextColor(c)


    def append(self, msg):
        self.edi_err.append(msg)


    #def closeEvent(self, e):
    #    logger.debug('QWLoggerError.closeEvent')
    #    QGroupBox.closeEvent(self, e)
    #    cp.qwloggererror = None


    if __name__ == "__main__" :

      def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  M - add message in error logger window'\
               '\n  S - add separator in error logger window'\
               '\n'


      def keyPressEvent(self, e) :
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_S : 
            self.add_separator_err()

        elif e.key() == Qt.Key_M : 
            self.append_qwlogger_err(msg='new message')

        else :
            logger.info(self.key_usage())

#----

if __name__ == "__main__" :
    import sys

    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = QWLoggerError()
    w.setWindowTitle('QWLoggerError')
    w.setGeometry(200, 400, 600, 300)

    from psdaq.control_gui.QWIcons import icon # should be imported after QApplication
    icon.set_icons()
    w.setWindowIcon(icon.icon_logviewer)

    w.show()
    app.exec_()
    sys.exit(0)

#----
