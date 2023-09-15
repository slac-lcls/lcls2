
"""Class :py:class:`MEDControl` is a QWidget with control fields for Mask Editor
================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDControl.py

    from psana.graphqt.MEDControl import MEDControl
    w = MEDControl()

Created on 2023-09-07 by Mikhail Dubrovin
"""

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
import psana.graphqt.MEDUtils as mu

class MEDControl(QWidget):
    """QWidget with control fields for Mask Editor"""

    def __init__(self, **kwa):

        #d = DIR_DATA_TEST + '/misc/'
        #kwa.setdefault('dirs', dirs_to_search())

        QWidget.__init__(self, None)

        self.geo      = kwa.get('geo', None)
        self.geofname = kwa.get('geofname', 'geometry.txt')
        self.ndafname = kwa.get('ndafname', 'ndarray.npy')
        self.wmain    = kwa.get('parent', None)

        if self.wmain is not None:
            self.wisp = self.wmain.wisp
            self.wimax = self.wmain.wisp.wimax
            self.wspec = self.wmain.wisp.wspec
            self.wim = self.wmain.wisp.wimax.wim

        self.lab_geo = QLabel('Geometry:')
        self.but_geo = QPushButton(str(self.geofname))

        self.lab_nda = QLabel('N-d array:')
        self.but_nda = QPushButton(str(self.ndafname))

        self.lab_exp = QLabel('Experiment:')
        self.but_exp = QPushButton('Select')

        self.list_of_buts = (
          self.but_nda,
          self.but_geo,
          self.but_exp,
        )

        self.hbox0 = QHBoxLayout()
        self.hbox0.addWidget(self.lab_nda)
        self.hbox0.addWidget(self.but_nda)
        self.hbox0.addStretch()

        self.hbox1 = QHBoxLayout()
        self.hbox1.addWidget(self.lab_exp)
        self.hbox1.addWidget(self.but_exp)
        self.hbox1.addWidget(self.lab_geo)
        self.hbox1.addWidget(self.but_geo)
        self.hbox1.addStretch()

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox0)
        self.vbox.addLayout(self.hbox1)
        self.setLayout(self.vbox)

        #for but in self.list_of_buts:
        #    but.clicked.connect(self.on_but_clicked)

        self.but_nda.clicked.connect(self.on_but_nda)
        self.but_geo.clicked.connect(self.on_but_geo)
        self.but_exp.clicked.connect(self.on_but_exp)

        self.set_tool_tips()
        self.set_style()

    def set_style(self):
        self.layout().setContentsMargins(5,5,5,5)
        self.lab_exp.setStyleSheet(style.styleLabel)
        self.lab_nda.setStyleSheet(style.styleLabel)
        self.lab_geo.setStyleSheet(style.styleLabel)
        #self.but_exp.setFixedWidth(50)
        #self.set_buttons_visiable()
        for but in self.list_of_buts: but.setStyleSheet(style.styleButton)

    def set_tool_tips(self):
        self.but_exp.setToolTip('Dataset - click and select')
        self.but_geo.setToolTip('Geometry file name')

    def on_but_geo(self):
        logger.debug('on_but_geo')  # sys._getframe().f_code.co_name
        path = popup_file_name(parent=self, mode='r', path=self.geofname, dirs=[], fltr='*.txt *.data\n*')
        logger.debug('Selected: %s' % str(path))
        if path is None: return
        self.but_geo.setText(path)
        self.but_geo.setStyleSheet(style.styleButtonGood)
        self.set_image()

    def on_but_nda(self):
        logger.debug('on_but_nda')  # sys._getframe().f_code.co_name
        path = popup_file_name(parent=self, mode='r', path=self.ndafname, dirs=[], fltr='*.npy *.txt *.data\n*')
        logger.debug('Selected: %s' % str(path))
        if path is None: return
        self.but_nda.setText(path)
        self.but_nda.setStyleSheet(style.styleButtonGood)
        self.set_image()

    def set_image(self):
        img, self.geo = mu.image_from_kwargs(\
               geofname=str(self.but_geo.text()),\
               ndafname=str(self.but_nda.text())
        )
        self.wim.set_pixmap_from_arr(img, set_def=True)

    def on_but_exp(self):
        insts = mu.list_of_instruments()
        logger.debug('list of instruments: %s' % str(insts))
        ins = popup_select_item_from_list(self.but_exp, insts, dx=10, dy=-10, do_sorted=False)
        if ins is None: return
        #self.but_exp.setText(ins)

        exps = mu.list_of_experiments(ins)
        logger.debug('list_of_experiments: %s' % str(exps))
        exp = popup_select_item_from_list(self.but_exp, exps, dx=10, dy=-10, do_sorted=False)
        if exp is None: return
        self.but_exp.setText(exp)

    def closeEvent(self, e):
        QWidget.closeEvent(self, e)


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
