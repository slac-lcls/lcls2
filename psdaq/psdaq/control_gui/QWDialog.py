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

#------------------------------

class QWDialog(QDialog) :
    """ use wdialog in popup dialog gui.
    """
    def __init__(self, parent=None, wdialog=None):
        QDialog.__init__(self,  parent)
 
        #self.setModal(True)
        self.vbox = QVBoxLayout()

        if wdialog is not None :
            self.vbox.addWidget(wdialog)

        self.but_cancel = QPushButton('&Cancel') 
        self.but_apply  = QPushButton('&Apply') 
        
        self.but_cancel.clicked.connect(self.onCancel)
        self.but_apply.clicked.connect(self.onApply)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.but_cancel)
        self.hbox.addWidget(self.but_apply)
        self.hbox.addStretch(1)

        self.vbox.addLayout(self.hbox)
        self.setLayout(self.vbox)

        self.but_cancel.setFocusPolicy(Qt.NoFocus)

        self.setStyle()
        self.setIcons()
        self.showToolTips()

#-----------------------------  

    def showToolTips(self):
        self.but_apply .setToolTip('Apply changes to the list')
        self.but_cancel.setToolTip('Use default list')
        
    def setStyle(self):
        #self.setFixedWidth(200)
        self.setMinimumWidth(200)
        styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        styleDefault = ""

        self.setStyleSheet(styleDefault)
        self.but_cancel.setStyleSheet(styleGray)
        self.but_apply .setStyleSheet(styleGray)

    def setIcons(self):
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
    w  = QWDialog(None, wd)
    #w.setGeometry(20, 40, 500, 200)
    w.setWindowTitle('Win title')
    #w.show()
    resp = w.exec_()
    logger.debug('resp=%s' % resp)
    logger.debug('QtWidgets.QDialog.Rejected: %d' % QDialog.Rejected)
    logger.debug('QtWidgets.QDialog.Accepted: %d' % QDialog.Accepted)

    del w
    del app

#------------------------------
