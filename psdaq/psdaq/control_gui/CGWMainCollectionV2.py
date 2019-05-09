#--------------------
"""
:py:class:`CGWMainCollection` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainCollection import CGWMainCollection

    # Methods - see test

See:
    - :py:class:`CGWMainCollection`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-04-26 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QPushButton, QGridLayout, QLabel, QSizePolicy 
# QWidget, QLabel, QLineEdit, QFileDialog
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

from psdaq.control_gui.CGWPanelList import CGWPanelList, QSize
from psdaq.control_gui.CGJsonUtils  import get_status

#--------------------

class CGWMainCollection(QGroupBox) :
    """
    """
    TABTITLE_H = ['drp','teb','meb']

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Collection', parent)

        self.grid = None
        self.update_table()

        #self.lab_duration = QLabel('Duration')
        #self.lab_events   = QLabel('Events')
        #self.lab_dameged  = QLabel('Damaged')
        #self.lab_size     = QLabel('Size')
        #self.but_status = QPushButton('Update')

        self.vbox = QVBoxLayout() 
        #self.vbox.addWidget(self.lab_duration)
        #self.vbox.addWidget(self.lab_events  )
        #self.vbox.addWidget(self.lab_dameged )
        #self.vbox.addWidget(self.lab_size    )
        self.vbox.addLayout(self.grid)
        #self.vbox.addWidget(self.but_status)
        #self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        #self.but_status.clicked.connect(self.on_but_status)

#--------------------

    def update_table(self) :
        #list2d = [['drp1','teb1','meb1'],\
        #          ['drp2','teb2','meb2'],\
        #          ['drp3','teb3','meb3']]

        list2d = get_status(self.TABTITLE_H)
        logger.debug('list2d processes status:\n%s' % str(list2d))

        nrows = len(list2d)

        if self.grid is None :
            self.grid = QGridLayout()

            for col,s in enumerate(self.TABTITLE_H) :
                self.grid.addWidget(QLabel(s), 0, col)
                self.grid.setColumnStretch(col, 0)
                self.grid.setColumnMinimumWidth(col, 100)
                lst_col = [list2d[r][col] for r in range(nrows)]
                self.grid.addWidget(CGWPanelList(list_str=lst_col), 1, col)

        else :
            for col,s in enumerate(self.TABTITLE_H) :
                w = self.grid.itemAtPosition(1, col).widget()
                lst_col = [list2d[r][col] for r in range(nrows)]
                w.fill_list_model(list_str=lst_col)

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('CGWMainCollection')
        #self.but_status.setToolTip('Click on button.') 

#--------------------

    def sizeHint(self):
        return QSize(350, 200)

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)
        #self.but_status.setStyleSheet(style.styleButtonGood)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        #self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        #self.grid.setMinimumWidth(300)

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
 
#    def on_but_status(self):
#        logger.debug('on_but_status')
#        self.update_table()

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainCollection(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
