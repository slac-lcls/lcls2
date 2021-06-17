
"""Class :py:class:`IVControl` is a Image Viewer QWidget with control fields
==========================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVControl.py

    from psana.graphqt.IVControl import IVControl
    w = IVControl()

Created on 2021-06-14 by Mikhail Dubrovin
"""

import logging
#logger = logging.getLogger(__name__)

import os
import sys
from psana.graphqt.CMWControlBase import cp, CMWControlBase
from PyQt5.QtWidgets import QVBoxLayout #QWidget, QLabel, QComboBox, QPushButton, QLineEdit
from psana.graphqt.QWFileName import QWFileName


class IVControl(CMWControlBase):
    """QWidget for Image Viewer control fields"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent',None)
        d = '/reg/g/psdm/detector/alignment/epix10ka2m/calib-xxx-epix10ka2m.1-2021-02-02/'
        fname_nda = d + 'det-calib-mfxc00118-r242-e5000-max.txt'
        fname_geo = d + '2021-02-02-epix10ks2m.1-geometry-recentred-for-psana.txt'

        CMWControlBase.__init__(self, **kwargs)

        self.w_fname_nda = QWFileName(None, butname='Select', label='N-d array:',\
           path=fname_nda, fltr='*.txt *.npy\n*', show_frame=True)

        self.w_fname_geo = QWFileName(None, butname='Select', label='Geometry:',\
           path=fname_geo, fltr='*.txt *.data\n*', show_frame=True)

        self.box1 = QVBoxLayout() 
        #self.box1.addWidget(self.lab_ctrl)
        #self.box1.addWidget(self.but_exp_col)
        self.box1.addWidget(self.w_fname_geo)
        self.box1.addStretch(1)
        self.box1.addWidget(self.w_fname_nda)
        #self.box1.addSpacing(20)
        self.box1.addStretch(1)
        self.box1.addWidget(self.but_tabs)

        #self.box1.addLayout(self.grid)
        self.setLayout(self.box1)
 
        #self.but_exp_col.clicked.connect(self.on_but_clicked)
        #self.but_exp_col.clicked.connect(self.on_but_exp_col)

        #if cp.h5vmain is not None:
        #    self.w_fname_nda.connect_path_is_changed_to_recipient(cp.h5vmain.wtree.set_file)

        self.set_tool_tips()
        #self.set_style()
        #self.set_buttons_visiable()

        #from psana.graphqt.UtilsFS import list_of_instruments
        #print('lst:', list_of_instruments(cp.instr_dir.value()))


    def set_tool_tips(self):
        self.setToolTip('Control fields/buttons')


    def set_style(self):
        pass
        #from psana.graphqt.Styles import style
        #self.         setStyleSheet(style.styleBkgd)
        #self.lab_db_filter.setStyleSheet(style.styleLabel)
        #self.lab_ctrl.setStyleSheet(style.styleLabel)

        #icon.set_icons()
        #self.but_exp_col.setIcon(icon.icon_folder_open)


if __name__ == "__main__":
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = IVControl()
    #w.setGeometry(200, 400, 500, 200)
    w.setWindowTitle('IV Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
