#--------------------
"""
:py:class:`CGWMainStatus` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainStatus import CGWMainStatus

    # Methods - see test

See:
    - :py:class:`CGWMainStatus`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-04-26 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QPushButton # QLabel, QWidget,  QLabel, QLineEdit, QFileDialog
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

from psdaq.control_gui.QWTableOfCheckBoxes import QWTableOfCheckBoxes
from psdaq.control_gui.CGJsonUtils import get_status

#--------------------

class CGWMainStatus(QGroupBox) :
    """
    """
    TABTITLE_H = ['R.G.','drp','meb','teb']

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Status', parent)

        self.wtable = None
        self.update_table()

        #self.lab_duration = QLabel('Duration')
        #self.lab_events   = QLabel('Events')
        #self.lab_dameged  = QLabel('Damaged')
        #self.lab_size     = QLabel('Size')
        self.but_status = QPushButton('Update status')

        self.vbox = QVBoxLayout() 
        #self.vbox.addWidget(self.lab_duration)
        #self.vbox.addWidget(self.lab_events  )
        #self.vbox.addWidget(self.lab_dameged )
        #self.vbox.addWidget(self.lab_size    )
        self.vbox.addWidget(self.wtable)
        self.vbox.addWidget(self.but_status)
        #self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_status.clicked.connect(self.on_but_status)

#--------------------

    def update_table(self) :

        list2d_status = get_status(self.TABTITLE_H)
        logger.debug('list2d processes status:\n%s' % str(list2d_status))

        if self.wtable is None :
            self.wtable = QWTableOfCheckBoxes(parent=None, tableio=list2d_status,\
                                          win_title='Selected processes',\
                                          title_h=self.TABTITLE_H,\
                                          is_visv=False)

        else :
            self.wtable.fill_table_model(parent=None, tableio=list2d_status,\
                                       win_title='Selected processes',\
                                       title_h=self.TABTITLE_H,\
                                       is_visv=False)

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('CGWMainStatus')
        self.but_status.setToolTip('Click on button.') 

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)
        self.but_status.setStyleSheet(style.styleButtonGood)

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
 
    def on_but_status(self):
        logger.debug('on_but_status')
        self.update_table()

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainStatus(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
