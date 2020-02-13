#------------------------------
"""
:py:class:`QWProgressBar` - progress bar widget
===============================================

Usage::

    # Test: python lcls2/psdaq/psdaq/control_gui/QWProgressBar.py

    # Import
    from psdaq.control_gui.QWProgressBar import QWProgressBar

    # Methods - see test

See:
    - :py:class:`QWProgressBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui/>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-02-12 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

import os

from PyQt5.QtGui import QPalette, QColor 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
                            # QSizePolicy, QCheckBox, QFrame
from psdaq.control_gui.Utils import load_textfile, save_textfile
#from psana.pyalgos.generic.Utils import load_textfile, save_textfile

#------------------------------

class QWProgressBar(QWidget) :
    """ ProgressBar with comment.
    """
    def __init__(self, **kwargs) :
        parent      = kwargs.get('parent', None)
        win_title   = kwargs.get('win_title', 'Progress bar')
        vmin        = kwargs.get('vmin', 0)
        vmax        = kwargs.get('vmax', 100)
        value       = kwargs.get('value', 50)
        label       = kwargs.get('label', None)

        QWidget.__init__(self, parent)
        if win_title is not None : self.setWindowTitle(win_title)

        self.timer = None 

        self.hbox = QHBoxLayout()

        if label is not None :
            self.plab = QLabel(label)
            self.hbox.addWidget(self.plab)
            #self.hbox.addStretch(1)

        self.pbar = QProgressBar(self)
        self.hbox.addWidget(self.pbar)

        self.setLayout(self.hbox)

        self.set_range(vmin,vmax)
        self.set_value(value)

        self.set_style()
        #self.set_icons()
        self.set_tool_tips()

#-----------------------------  

    def set_tool_tips(self):
        self.setToolTip('Progress bar')
        self.pbar.setToolTip('Progress bar indicator')

    def set_style(self):
        #self.setFixedWidth(200)
        self.setMinimumWidth(150)

        self.pbar.setTextVisible(True)
        #self.plab.setAlignment(Qt.AlignCenter)
        self.plab.setStyleSheet('text-align: center;')

        styleDefault = ""
        #styleGray = "background-color: rgb(230, 240, 230); color: rgb(0, 0, 0);" # Gray
        #styleGreenish = "background-color: rgb(100, 240, 200); color: rgb(255, 0, 0);" # Greenish
        #styleGreenish = "background-color: rgb(100, 240, 240);" # Greenish

        #self.setWindowFlags(Qt.FramelessWindowHint)
        #self.layout().setContentsMargins(5,5,5,5)
        self.layout().setContentsMargins(0,0,0,0)

        self.setStyleSheet(styleDefault)
        self.pbar.setStyleSheet(styleDefault)
        #self.pbar.setStyleSheet('background-color: magenta;')
        #self.pbar.setStyleSheet('QProgressBar::chunk {background-color: yellow;};')

        p = QPalette()
        p.setColor(QPalette.Highlight, Qt.green)
        p.setColor(QPalette.Base, Qt.white) # QColor(230, 230, 255, 255))
        self.pbar.setPalette(p)


    def set_range(self, vmin, vmax):
        self.pbar.setRange(vmin,vmax)

    def set_value(self, value):
        self.pbar.setValue(value)

    def set_icons(self):
        try :
          from psdaq.control_gui.QWIcons import icon
          #from psana.graphqt.QWIcons import icon
          icon.set_icons()
          #self.but_cancel.setIcon(icon.icon_button_cancel)
          #self.but_apply .setIcon(icon.icon_button_ok)
        except : pass
 
    #def resizeEvent(self, e):
        #logger.debug('resizeEvent') 

    #def moveEvent(self, e):
        #logger.debug('moveEvent') 

    #def event(self, event):
        #logger.debug('Event happens...: %s' % str(event))

    #def closeEvent(self, event):
    #    logger.debug('closeEvent')

    def on_timeout(self, timeout_msec=100):
        """ for test only
        """
        if self.timer is None :
            self.timer = QTimer()
            self.timer.timeout.connect(self.on_timeout)
            self.counter = -1
        self.counter += 1
        value = (self.pbar.maximum() - self.pbar.minimum())*self.counter/100.
        if value > self.pbar.maximum() :
            self.timer.stop()
            self.timer.timeout.disconnect(self.on_timeout)
            
        self.pbar.setValue(value)
        self.timer.start(timeout_msec)

#--------------------
#--------------------
#--------------------
#--------------------

if __name__ == "__main__" :

    from PyQt5.QtCore import QTimer # pyqtSignal, Qt, QRectF, QPointF, QTimer

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    import sys
    from PyQt5.QtCore import QPoint #Qt
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = QWProgressBar(label='procname')
    w.move(QPoint(100,50))

    w.on_timeout()

    w.show()
    app.exec_()
    #t = w.get_content()
    #logger.debug("edited text: %s" % str(t))
    del w
    del app

#------------------------------
