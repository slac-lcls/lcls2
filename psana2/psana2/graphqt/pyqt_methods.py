import logging
logger = logging.getLogger(__name__)

import os
#import psana.pyalgos.generic.Utils as gu
import os
import sys
from psana2.graphqt.CMWControlBase import cp, CMWControlBase
from psana2.graphqt.QWFileName import QWFileName

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRect, QRectF, QSize, QTimer, QEvent, QMargins, QRegExp, QThread, QModelIndex
from PyQt5.QtGui import QPen, QBrush, QColor, QFont, QCursor, QTextCursor, QIcon, QStandardItemModel, QStandardItem, QIntValidator, QRegExpValidator, QImage, QPixmap, QBitmap, QPolygon, QPolygonF, QPainterPath
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QTabBar, QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit, QFileDialog, QSizePolicy, QComboBox, QCheckBox, QListWidget, QButtonGroup, QRadioButton, QCheckBox, QFrame, QListView, QAbstractItemView, QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QSplitter, QGraphicsItem, QGraphicsPolygonItem, QGraphicsPathItem, QGraphicsRectItem, QGraphicsEllipseItem, QHeaderView

logging.basicConfig(format='[%(levelname).1s] %(asctime)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d : %(message)s', level=logging.DEBUG)
logging.basicConfig(format='[%(levelname).1s] %(asctime)s %(name)s %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)


self.setGeometry(50, 50, 500, 600)
self.setGeometry(self.main_win_pos_x .value(),\
                 self.main_win_pos_y .value(),\
                 self.main_win_width .value(),\
                 self.main_win_height.value())
w_height = self.main_win_height.value()

self.setMinimumSize(500, 400)
self.wrig.setMinimumWidth(350)
self.wrig.setMaximumWidth(450)
self.wlog.setMinimumWidth(500)
self.setFixedSize(800,500)
self.setMinimumSize(500,800)

spl_pos = cp.main_vsplitter.value()
self.vspl.setSizes((spl_pos,w_height-spl_pos,))
self.hspl.moveSplitter(w*0.5,0)

self.wrig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

self.butELog.setStyleSheet(style.styleButton)
self.butFile.setStyleSheet(style.styleButton)

self.butELog    .setVisible(False)
self.butFBrowser.setVisible(False)

self.but1.raise_()
self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)

self.but_tabs.clicked.connect(self.on_but_tabs)


from psana2.graphqt.QWIcons import icon
icon.set_icons()
self.but_exp_col.setIcon(icon.icon_folder_open)


from psana2.graphqt.Styles import style
self.setStyleSheet(style.styleBkgd)
self.lab_db_filter.setStyleSheet(style.styleLabel)
self.lab_ctrl.setStyleSheet(style.styleLabel)

self.box = QHBoxLayout()
self.box.addWidget(self.w_fname_geo)
self.box.addStretch(1)
self.box.addWidget(self.w_fname_nda)
self.box.addSpacing(20)
self.box.addStretch(1)
self.box.addWidget(self.but_tabs)
self.setLayout(self.box)

self.box = QGridLayout()
self.box.addWidget(self.w_fname_nda, 0, 0, 1, 9)
self.box.addWidget(self.w_fname_geo, 1, 0, 1, 9)
self.box.addWidget(self.but_tabs,    0, 10)
self.box.addWidget(self.but_reset,   1, 10)
self.setLayout(self.box)

style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
self.setStyleSheet(style)

self.setMinimumSize(725,360)
self.setFixedSize(750,270)
self.setMaximumWidth(800)


self.dtxt0 = QPointF(0, 0)

self.setStyleSheet('background: transparent; background-color: rgb(0,0,0);') #rgb(0,0,0);')QColor(black)
self.but_reset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
