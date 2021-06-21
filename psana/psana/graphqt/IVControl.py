
"""Class :py:class:`IVControl` is a Image Viewer QWidget with control fields
==========================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVControl.py

    from psana.graphqt.IVControl import IVControl
    w = IVControl()

Created on 2021-06-14 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys
from psana.graphqt.CMWControlBase import cp, CMWControlBase
from PyQt5.QtWidgets import QGridLayout, QPushButton# QHBoxLayout #QWidget, QLabel, QComboBox, QPushButton, QLineEdit
from psana.graphqt.QWFileNameV2 import QWFileNameV2

import psana.pyalgos.generic.PSUtils as psu
from psana.pyalgos.generic.NDArrUtils import reshape_to_2d, info_ndarr


def image_from_ndarray(nda):
    img = psu.table_nxn_epix10ka_from_ndarr(nda) if (nda.size % (352*384) == 0) else\
          psu.table_nxm_cspad2x1_from_ndarr(nda) if (nda.size % (185*388) == 0) else\
          reshape_to_2d(nda)
    logger.debug(info_ndarr(img,'img'))
    return img


class IVControl(CMWControlBase):
    """QWidget for Image Viewer control fields"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent',None)
        d = '/reg/g/psdm/detector/alignment/epix10ka2m/calib-xxx-epix10ka2m.1-2021-02-02/'
        #fname_nda = d + 'det-calib-mfxc00118-r242-e5000-max.txt'
        #fname_geo = d + '2021-02-02-epix10ks2m.1-geometry-recentred-for-psana.txt'
        fname_nda = d + 'Select'
        fname_geo = d + 'Select'

        CMWControlBase.__init__(self, **kwargs)

        self.w_fname_nda = QWFileNameV2(None, label='Array:',\
           path=fname_nda, fltr='*.txt *.npy *.data *.dat\n*', show_frame=True)

        self.w_fname_geo = QWFileNameV2(None, label='Geometry:',\
           path=fname_geo, fltr='*.txt *.data\n*', show_frame=True)

        self.but_reset = QPushButton('Reset')


        self.box = QGridLayout()
        self.box.addWidget(self.w_fname_nda, 0, 0, 1, 9)
        self.box.addWidget(self.w_fname_geo, 1, 0, 1, 9)
        self.box.addWidget(self.but_tabs,    0, 10)
        self.box.addWidget(self.but_reset,   1, 10)

        #self.box1.addLayout(self.grid)
        self.setLayout(self.box)
 
        self.but_reset.clicked.connect(self.on_but_reset)
        self.w_fname_nda.connect_path_is_changed_to_recipient(self.on_changed_fname_nda)
        self.w_fname_geo.connect_path_is_changed_to_recipient(self.on_changed_fname_geo)

        self.set_tool_tips()
        self.set_style()
        #self.set_buttons_visiable()

        #from psana.graphqt.UtilsFS import list_of_instruments
        #print('lst:', list_of_instruments(cp.instr_dir.value()))


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        self.but_reset.setToolTip('Reset original image size')


    def set_style(self):
        self.but_tabs.setFixedWidth(50)
        self.but_reset.setFixedWidth(50)


    def on_but_reset(self):
        logger.debug('on_but_reset')
        if cp.wimage is not None:
           cp.wimage.reset_original_size()


    def on_changed_fname_nda(self, fname):
        logger.debug('on_changed_fname_nda: %s' % fname)
        if cp.wimage is not None:
           nda = psu.load_ndarray_from_file(fname)
           logger.debug(info_ndarr(nda,'nda'))
           img = image_from_ndarray(nda)
           cp.wimage.set_pixmap_from_arr(img, set_def=True)


    def on_changed_fname_geo(self, s):
        logger.debug('on_changed_fname_geo: %s' % s)


if __name__ == "__main__":
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = IVControl()
    w.setGeometry(100, 50, 500, 80)
    w.setWindowTitle('IV Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
