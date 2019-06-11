#------------------------------
"""
:py:class:`QWDialog` - Popup GUI
========================================

Usage::

    # Test: python psdaq.control_gui.QWDialog.py

    # Import
    from psdaq.control_gui.QWDialog import QWDialog

    # Methods - see test

See:
    - :py:class:`QWDialog`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Adopted for LCLS2 on 2019-01-29 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox
from PyQt5.QtCore import Qt

#from psdaq.control_gui.Styles import style

#------------------------------

class QWDialog(QDialog) :
    """ use wdialog in popup dialog gui.
    """
    def __init__(self, parent=None, wdialog=None, is_frameless=False):
        QDialog.__init__(self, parent)
 
        self.is_frameless = is_frameless

        #self.setModal(True)
        self.vbox = QVBoxLayout()
        if wdialog is not None :
            self.vbox.addWidget(wdialog)

        self.but_cancel = QPushButton('Cancel') 
        self.but_apply  = QPushButton('Apply') 

        self.but_cancel.clicked.connect(self.onCancel)
        self.but_apply.clicked.connect(self.onApply)

        wdialog.but_apply  = self.but_apply
        wdialog.but_cancel = self.but_cancel

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_cancel)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_apply)
        self.hbox.addStretch(1)

        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.but_cancel.setFocusPolicy(Qt.NoFocus)

        self.set_style()
        self.set_icons()
        self.set_tool_tips()

#-----------------------------  

    def set_tool_tips(self):
        self.but_apply .setToolTip('Apply changes to the list')
        self.but_cancel.setToolTip('Use default list')
        
    def set_style(self):
        self.setStyleSheet("background-color: rgb(250, 250, 250);")
        if self.is_frameless : self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        #if self.is_frameless : self.setWindowFlags(Qt.FramelessWindowHint)

        self.layout().setContentsMargins(4,4,4,4)
        #self.layout().setContentsMargins(0,0,0,0)

        #self.setFixedWidth(200)
        self.setMinimumWidth(200)
        #styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);"\
        #            ":disabled {color:rgb(0, 250, 0); background-color:rgb(0, 0, 250);}"

        styleDefault = ""

        self.setStyleSheet(styleDefault)
        # needs in correct style for Enabled/Disabled button
        #self.but_cancel.setStyleSheet(style.styleButton)
        #self.but_apply .setStyleSheet(style.styleButton)
        self.but_cancel.setStyleSheet(styleDefault)
        self.but_apply .setStyleSheet(styleDefault)


    def set_icons(self):
        from psdaq.control_gui.QWIcons import icon
        icon.set_icons()
        self.but_cancel.setIcon(icon.icon_button_cancel)
        self.but_apply .setIcon(icon.icon_button_ok)

    #def resizeEvent(self, e):
        #logger.debug('resizeEvent') 

    #def moveEvent(self, e):
        #logger.debug('moveEvent')

    #def closeEvent(self, event):
        #logger.debug('closeEvent')
        #try    : self.widg_pars.close()
        #except : pass

    #def event(self, event):
        #logger.debug('Event happens...: %s' % str(event))

    def onCancel(self):
        logger.debug('onCancel')
        self.reject()

    def onApply(self):
        logger.debug('onApply')  
        self.accept()

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication, QLineEdit

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    wd = QLineEdit('Test window')
    w  = QWDialog(None, wd, is_frameless=True)
    #w.setGeometry(20, 40, 500, 200)
    w.setWindowTitle('Win title')
    #w.show()
    resp=w.exec_()
    logger.debug('resp=%s' % resp)
    logger.debug('QtWidgets.QDialog.Rejected: %d' % QDialog.Rejected)
    logger.debug('QtWidgets.QDialog.Accepted: %d' % QDialog.Accepted)

    del w
    del app

#------------------------------
