#--------------------
"""
:py:class:`CGWMainRunStatistics` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainRunStatistics import CGWMainRunStatistics

    # Methods - see test

See:
    - :py:class:`CGWMainRunStatistics`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-28 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout # , QWidget,  QLabel, QLineEdit, QFileDialog
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

#--------------------

class CGWMainRunStatistics(QGroupBox) :
    """
    """
    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Run Statistics', parent)

        self.lab_duration = QLabel('Duration')
        self.lab_events   = QLabel('Events')
        self.lab_dameged  = QLabel('Dameged')
        self.lab_size     = QLabel('Size')
        self.but_stats = QPushButton('Damage Stats')

        self.vbox = QVBoxLayout() 
        self.vbox.addWidget(self.lab_duration)
        self.vbox.addWidget(self.lab_events  )
        self.vbox.addWidget(self.lab_dameged )
        self.vbox.addWidget(self.lab_size    )
        self.vbox.addWidget(self.but_stats)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_stats.clicked.connect(self.on_but_stats)

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Detector GUI')
        self.but_stats.setToolTip('Click on button.') 

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)
        self.but_stats.setStyleSheet(style.styleButtonGood)

        #self.layout().setContentsMargins(0,0,0,0)

        #self.setWindowTitle('File name selection widget')
        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
    def on_but_stats(self):
        logger.debug('on_but_stats')

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainRunStatistics(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
