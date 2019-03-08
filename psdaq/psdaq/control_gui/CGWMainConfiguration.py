#--------------------
"""
:py:class:`CGWMainConfiguration` - widget for configuration
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWMainConfiguration import CGWMainConfiguration

    # Methods - see test

See:
    - :py:class:`CGWMainConfiguration`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-25 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGroupBox, QLabel, QCheckBox, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout
     #QGridLayout, QLineEdit, QFileDialog, QWidget
from PyQt5.QtCore import Qt, QPoint # pyqtSignal, QRectF, QPointF, QTimer

from psdaq.control_gui.CGWConfigEditor import CGWConfigEditor

#--------------------

class CGWMainConfiguration(QGroupBox) :
    """
    """
    LIST_OF_CONFIG_OPTIONS = ('BEAM', 'NO_BEAM', 'ONE_SHOT', 'CAMERA')
    LIST_OF_SEQUENCES = ('1', '2', '3', '4', '5', '6', '7', '8')
    #path_is_changed = pyqtSignal('QString')

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Configuration', parent)

        self.lab_type = QLabel('Type')
        self.box_type = QComboBox(self)
        self.box_type.addItems(self.LIST_OF_CONFIG_OPTIONS)
        self.box_type.setCurrentIndex(1)

        self.but_edit = QPushButton('Edit')
        self.but_scan = QPushButton('Scan')

        self.cbx_seq = QCheckBox('Sync Sequence')
        self.box_seq = QComboBox(self)
        self.box_seq.addItems(self.LIST_OF_SEQUENCES)
        self.box_seq.setCurrentIndex(0)

        #self.edi = QLineEdit(path)
        #self.edi.setReadOnly(True) 

        self.hbox1 = QHBoxLayout() 
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.lab_type)
        self.hbox1.addWidget(self.box_type) 
        self.hbox1.addStretch(1)

        self.hbox2 = QHBoxLayout() 
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.cbx_seq)
        self.hbox2.addWidget(self.box_seq) 
        self.hbox2.addStretch(1)

        self.vbox = QVBoxLayout() 
        self.vbox.addLayout(self.hbox1)
        self.vbox.addWidget(self.but_edit, 0, Qt.AlignCenter)
        self.vbox.addWidget(self.but_scan, 0, Qt.AlignCenter)
        self.vbox.addLayout(self.hbox2)

        #self.grid = QGridLayout()
        #self.grid.addWidget(self.lab_type,       0, 0, 1, 1)
        #self.grid.addWidget(self.but_type,       0, 2, 1, 1)
        #self.grid.addWidget(self.but_edit,       1, 1, 1, 1)
        #self.grid.addWidget(self.but_scan,       2, 1, 1, 1)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_edit.clicked.connect(self.on_but_edit)
        self.but_scan.clicked.connect(self.on_but_scan)
        self.box_type.currentIndexChanged[int].connect(self.on_box_type)
        self.box_seq.currentIndexChanged[int].connect(self.on_box_seq)
        self.cbx_seq.stateChanged[int].connect(self.on_cbx_seq)

        self.w_edit = None

#--------------------

    def set_tool_tips(self) :
        #self.but_edit.setToolTip('Select input file.')
        self.setToolTip('Configuration') 
        self.box_type.setToolTip('Click and select.') 

#--------------------

    def set_style(self) :
        from psdaq.control_gui.Styles import style
        self.setStyleSheet(style.qgrbox_title)

        self.but_edit.setFixedWidth(60)
        self.but_scan.setFixedWidth(60)

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
 
    def on_box_type(self, ind):
        selected = str(self.box_type.currentText())
        msg = 'selected ind:%d %s' % (ind,selected)
        logger.debug(msg)

#--------------------
 
    def on_box_seq(self, ind):
        selected = str(self.box_seq.currentText())
        msg = 'selected ind:%d %s' % (ind,selected)
        logger.debug(msg)

#--------------------
 
    def on_cbx_seq(self, ind):
        #if self.cbx.hasFocus() :
        cbx = self.cbx_seq
        tit = cbx.text()
        #self.cbx_runc.setStyleSheet(style.styleGreenish if cbx.isChecked() else style.styleYellowBkg)
        msg = 'Check box "%s" is set to %s' % (tit, cbx.isChecked())
        logger.info(msg)

#--------------------
 
    def on_but_edit(self):
        logger.debug('on_but_edit')
        if self.w_edit is None :
            self.w_edit = CGWConfigEditor()
            self.w_edit.move(self.pos() + QPoint(self.width()+30, 0))
            self.w_edit.show()
        else :
           self.w_edit.close()
           self.w_edit = None

#--------------------
 
    def on_but_scan(self):
        logger.debug('on_but_scan')

#--------------------

    def closeEvent(self, e):
        print('CGWMainConfiguration.closeEvent')
        if self.w_edit is not None :
           self.w_edit.close()
        QGroupBox.closeEvent(self, e)

#--------------------
 
if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CGWMainConfiguration(None)
    #w.connect_path_is_changed_to_recipient(w.test_signal_reception)
    w.show()
    app.exec_()

#--------------------
