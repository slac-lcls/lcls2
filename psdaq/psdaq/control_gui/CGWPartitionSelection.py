#--------------------
"""
:py:class:`CGWPartitionSelection` - widget for control_gui
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWPartitionSelection import CGWPartitionSelection

    # Methods - see test

See:
    - :py:class:`CGWPartitionSelection`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QGroupBox, QPushButton, QVBoxLayout # , QWidget,  QLabel, QLineEdit, QFileDialog
#from PyQt5.QtCore import pyqtSignal #, Qt, QRectF, QPointF, QTimer

#--------------------

class CGWPartitionSelection(QWidget) :
#class CGWPartitionSelection(QGroupBox) :
    """
    """
    def __init__(self, parent=None, parent_ctrl=None):

        #QGroupBox.__init__(self, 'Partition', parent)
        QWidget.__init__(self, parent)
        self.parent_ctrl = parent_ctrl

        self.grb_read_nodes = QGroupBox('Readout Nodes', parent)
        self.grb_proc_nodes = QGroupBox('Processing Nodes', parent)
        self.grb_bld        = QGroupBox('Beamline Data', parent)
        self.grb_camera_ioc = QGroupBox('Camera IOC', parent)

        self.vbox = QVBoxLayout() 
        self.vbox.addWidget(self.grb_read_nodes)
        self.vbox.addWidget(self.grb_proc_nodes)
        self.vbox.addWidget(self.grb_bld)
        self.vbox.addWidget(self.grb_camera_ioc)
        self.vbox.addStretch(1)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        #self.but_select.clicked.connect(self.on_but_select)
        #self.but_display.clicked.connect(self.on_but_display)

#--------------------

    def set_tool_tips(self) :
        self.setToolTip('Partition Selection GUI')

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.grb_read_nodes.setStyleSheet(style.qgrbox_title)
        self.grb_proc_nodes.setStyleSheet(style.qgrbox_title)
        self.grb_bld       .setStyleSheet(style.qgrbox_title)
        self.grb_camera_ioc.setStyleSheet(style.qgrbox_title)

        self.setWindowTitle('Partition Selection')

        self.setMinimumSize(300,200)

        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 
        #self.layout().setContentsMargins(0,0,0,0)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
#    def on_but_select(self):
#        logger.debug('on_but_select')

#--------------------
 
#    def on_but_display(self):
#        logger.debug('on_but_display')

#--------------------

    def closeEvent(self, e):
        logger.debug('closeEvent')
        self.parent_ctrl.w_display = None
        QWidget.closeEvent(self, e)

#--------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWPartitionSelection(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
