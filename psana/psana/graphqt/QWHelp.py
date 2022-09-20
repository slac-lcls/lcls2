
"""
:py:class:`QWHelp` - GUI for Logger
========================================

Usage::

    # Import
    from psana.graphqt.QWHelp import QWHelp

    # Methods - see test

See:
    - :py:class:`QWHelp`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Adopted for LCLS2 on 2018-02-16 by Mikhail Dubrovin
"""


from PyQt5 import QtWidgets

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.Frame    import Frame
from psana.graphqt.Styles import style
from psana.graphqt.QWIcons import icon


#class QWHelp(QtWidgets.QWidget):
class QWHelp(Frame):

    def __init__(self, parent=None, msg='No message in QWHelp...'):

        #QtWidgets.QWidget.__init__(self, parent)
        Frame.__init__(self, parent, mlw=1)

        self.setGeometry(100, 100, 730, 200)
        self.setWindowTitle('GUI Help')
        try:
             icon.set_icons()
             self.setWindowIcon(icon.icon_monitor)
        except: pass

        self.box_txt    = QtWidgets.QTextEdit()
        self.tit_status = QtWidgets.QLabel('Status:')
        self.but_close  = QtWidgets.QPushButton('Close')

        self.hboxM = QtWidgets.QHBoxLayout()
        self.hboxM.addWidget( self.box_txt )

        self.hboxB = QtWidgets.QHBoxLayout()
        self.hboxB.addWidget(self.tit_status)
        self.hboxB.addStretch(4)
        self.hboxB.addWidget(self.but_close)

        self.vbox  = QtWidgets.QVBoxLayout()
        self.vbox.addLayout(self.hboxM)
        self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        self.but_close.clicked.connect(self.onClose)

        self.setHelpMessage(msg)

        self.showToolTips()
        self.setStyle()


    def showToolTips(self):
        #self           .setToolTip('This GUI is intended for run control and monitoring.')
        self.but_close .setToolTip('Close this window.')


    def setStyle(self):
        self.setMinimumHeight(300)
        self.           setStyleSheet(style.styleBkgd)
        self.tit_status.setStyleSheet(style.styleTitle)
        self.but_close .setStyleSheet(style.styleButton)
        self.box_txt   .setReadOnly(True)
        self.box_txt   .setStyleSheet(style.styleWhiteFixed)
        self.layout().setContentsMargins(0,0,0,0)


    def setParent(self,parent):
        self.parent = parent


    def closeEvent(self, event):
        logger.debug('closeEvent')
        self.box_txt.close()


    def onClose(self):
        logger.debug('onClose')
        self.close()


    def setHelpMessage(self, msg):
        logger.debug('Set help message')
        self.box_txt.setText(msg)
        self.setStatus(0, 'Status: show help info...')


    def setStatus(self, status_index=0, msg=''):
        list_of_states = ['Good','Warning','Alarm']
        if status_index == 0: self.tit_status.setStyleSheet(style.styleStatusGood)
        if status_index == 1: self.tit_status.setStyleSheet(style.styleStatusWarning)
        if status_index == 2: self.tit_status.setStyleSheet(style.styleStatusAlarm)
        self.tit_status.setText(msg)


if __name__ == "__main__":
    import sys
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QtWidgets.QApplication(sys.argv)
    w = QWHelp()
    w.setHelpMessage('This is a test message to test methods of QWHelp...')
    w.show()
    app.exec_()

# EOF
