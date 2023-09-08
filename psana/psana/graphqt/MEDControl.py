
"""Class :py:class:`MEDControl` is a QWidget with control fields for Mask Editor
================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDControl.py

    from psana.graphqt.MEDControl import MEDControl
    w = MEDControl()

Created on 2023-09-07 by Mikhail Dubrovin
"""

#from psana.graphqt.CMWControlBase import * #cp, CMWControlBase, QApplication, os, sys, logging, QRectF, QWFileNameV2
#from psana.graphqt.IVControlSpec import IVControlSpec
#import psana.pyalgos.generic.PSUtils as psu
#from psana.pyalgos.generic.NDArrUtils import reshape_to_2d, info_ndarr, np
#import psana.graphqt.QWUtils as qu
#from psana.detector.dir_root import DIR_DATA_TEST

import os
import sys

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,\
                            QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit
from PyQt5.QtCore import QSize, QRectF, pyqtSignal, QModelIndex, QTimer

from psana.graphqt.MEDUtils import *
from psana.graphqt.Styles import style

import psana.graphqt.GWROIUtils as roiu
from psana.graphqt.QWPopupSelectItem import popup_select_item_from_list
from psana.graphqt.QWPopupFileName import popup_file_name

#from psana.detector.dir_root import DIR_DATA_TEST
#from psana.graphqt.QWIcons import icon

class MEDControl(QWidget):
    """QWidget with control fields for Mask Editor"""

    def __init__(self, **kwa):

        #d = DIR_DATA_TEST + '/misc/'
        #kwa.setdefault('dirs', dirs_to_search())

        QWidget.__init__(self, None)

        self.fname_geo = kwa.get('fname_geo', 'geometry.txt')
        self.wmain = kwa.get('parent', None)
        if self.wmain is not None:
            self.wisp = self.wmain.wisp
            self.wimax = self.wmain.wisp.wimax
            self.wspec = self.wmain.wisp.wspec
            self.wim = self.wmain.wisp.wimax.wim

        self.lab_geo = QLabel('Geometry:')
        self.but_geo = QPushButton('Select')

        self.lab_ins = QLabel('Instr:')
        self.but_ins = QPushButton('Select')

        self.list_of_buts = (
          self.but_geo,
          self.but_ins,
        )

        self.hbox0 = QHBoxLayout()
        self.hbox0.addWidget(self.lab_ins)
        self.hbox0.addWidget(self.but_ins)
        self.hbox0.addStretch()

        self.hbox1 = QHBoxLayout()
        self.hbox1.addWidget(self.lab_geo)
        self.hbox1.addWidget(self.but_geo)
        self.hbox1.addStretch()

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox0)
        self.vbox.addLayout(self.hbox1)
        self.setLayout(self.vbox)

        #for but in self.list_of_buts:
        #    but.clicked.connect(self.on_but_clicked)

        self.but_geo.clicked.connect(self.on_but_geo)
        self.but_ins.clicked.connect(self.on_but_ins)

        self.set_tool_tips()
        self.set_style()

    def set_style(self):
        self.layout().setContentsMargins(5,5,5,0)
        self.lab_ins.setStyleSheet(style.styleLabel)
        self.lab_geo.setStyleSheet(style.styleLabel)
        self.but_ins.setFixedWidth(50)
        #self.set_buttons_visiable()
        for but in self.list_of_buts: but.setStyleSheet(style.styleButton)

    def set_tool_tips(self):
        self.but_ins.setToolTip('Instrument - click and select')
        self.but_geo.setToolTip('Geometry file name')

    def on_but_geo(self):
        logger.debug(sys._getframe().f_code.co_name)
        path = popup_file_name(parent=self, mode='r', path=self.fname_geo, dirs=[], fltr='*.txt *.data\n*')
        logger.debug('Selected: %s' % str(path))
        if path is None: return
        self.but_geo.setText(path)
        self.but_geo.setStyleSheet(style.styleButtonGood)

    def on_but_ins(self):
        logger.debug(sys._getframe().f_code.co_name)
        insts = sorted(os.listdir('/cds/data/ffb/'))
        logger.debug('list of instruments: %s' % str(insts))

        ins = popup_select_item_from_list(self.but_ins, insts, dx=10, dy=-10, do_sorted=False)
        if ins is None: return
        self.but_ins.setText(ins)

    def closeEvent(self, e):
        QWidget.closeEvent(self, e)
        logger.debug('sys._getframe().f_code.co_name')


if __name__ == "__main__":

    #os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = MEDControl()
    w.setGeometry(100, 50, 500, 40)
    w.setWindowTitle('MED Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
