#--------------------
"""
:py:class:`CGWMainControl` - widget for configuration
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainControl import CGWMainControl

    # Methods - see test

See:
    - :py:class:`CGWMainControl`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QLabel, QCheckBox, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout, QComboBox
     #QGridLayout, QLineEdit, QFileDialog, QWidget
from PyQt5.QtCore import Qt # pyqtSignal, QRectF, QPointF, QTimer

from psdaq.control_gui.Styles import style

from psdaq.control_gui.CGDaqControl import daq_control, DaqControl #, worker_set_state
#from psdaq.control_gui.DoWorkInThread import DoWorkInThread
#from psdaq.control_gui.CGParameters import cp

#--------------------

class CGWMainControl(QGroupBox) :
    """
    """

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Control', parent)

        self.lab_state = QLabel('Target State')
        self.lab_trans = QLabel('Last Transition')
        #self.box_type = QComboBox(self)
        #self.box_type.addItems(self.LIST_OF_CONFIG_OPTIONS)
        #self.box_type.setCurrentIndex(1)

        self.cbx_runc       = QCheckBox('Record Run')
        self.box_state      = QComboBox()
        self.but_transition = QPushButton('Enable')

        self.states = ['Select',] + [s.upper() for s in DaqControl.states]
        self.box_state.addItems(self.states)

        #self.edi = QLineEdit(path)
        #self.edi.setReadOnly(True) 

        self.hbox1 = QHBoxLayout() 
        self.hbox1.addWidget(self.lab_state)
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.lab_trans)

        self.hbox2 = QHBoxLayout() 
        self.hbox2.addWidget(self.box_state, 0, Qt.AlignCenter)
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.but_transition, 0, Qt.AlignCenter)
        #self.hbox2.addStretch(1)

        self.vbox = QVBoxLayout() 
        self.vbox.addWidget(self.cbx_runc, 0, Qt.AlignCenter)
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)

        #self.grid = QGridLayout()
        #self.grid.addWidget(self.lab_state,       0, 0, 1, 1)
        #self.grid.addWidget(self.but_type,       0, 2, 1, 1)
        #self.grid.addWidget(self.box_state,       1, 1, 1, 1)
        #self.grid.addWidget(self.but_transition,       2, 1, 1, 1)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.box_state.currentIndexChanged[int].connect(self.on_box_state)
        self.but_transition.clicked.connect(self.on_but_transition)
        #self.box_type.currentIndexChanged[int].connect(self.on_box_type)
        self.cbx_runc.stateChanged[int].connect(self.on_cbx_runc)
 
#--------------------

    def set_tool_tips(self) :
        #self.box_state.setToolTip('Select input file.')
        self.setToolTip('Configuration') 
        #self.box_type.setToolTip('Click and select.') 

#--------------------

    def set_style(self) :
        self.setStyleSheet(style.qgrbox_title)
        #self.box_state.setFixedWidth(60)
        #self.but_transition.setFixedWidth(60)
        #self.but_transition.setStyleSheet(style.styleButtonGood)

        self.cbx_runc.setFixedSize(100,40)
        #self.cbx_runc.setStyleSheet(style.styleYellowBkg)
        self.cbx_runc.setStyleSheet(style.style_cbx_off)

        #self.setMinimumWidth(350)
        #self.setWindowTitle('File name selection widget')
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame : 
        #self.layout().setContentsMargins(0,0,0,0)
        #self.setMinimumSize(725,360)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------
 
#    def on_box_type(self, ind):
#        selected = str(self.box_type.currentText())
#        msg = 'selected ind:%d %s' % (ind,selected)
#        logger.debug(msg)

#--------------------
 
    def on_box_state(self, ind):
        if not ind : return
        state = self.states[ind]
        logger.info('CGWMainDetector.on_box_state -> daq_control().setState %s' % state)
        daq_control().setState(state.lower())
        logger.debug('command daq_control().setState is committed...')

#--------------------
 
    def on_but_transition(self):
        logger.debug('on_but_transition')

#--------------------
 
    def on_cbx_runc(self, ind):
        #if self.cbx.hasFocus() :
        cbx = self.cbx_runc
        tit = cbx.text()
        self.cbx_runc.setStyleSheet(style.styleGreenish if cbx.isChecked() else style.styleYellowBkg)
        msg = 'Check box "%s" is set to %s' % (tit, cbx.isChecked())
        logger.info(msg)

#--------------------
 
if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainControl(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
