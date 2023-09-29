
"""Class :py:class:`MEDControlROI` is a QWidget with ROI control fields for Mask Editor
=======================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDControlROI.py

    from psana.graphqt.MEDControlROI import MEDControlROI
    w = MEDControlROI()

Created on 2023-09-07 by Mikhail Dubrovin
"""

import os
import sys

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,\
                            QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit
from PyQt5.QtCore import Qt, QSize, QRectF, pyqtSignal, QModelIndex, QTimer

import numpy as np
from psana.graphqt.MEDUtils import mask_ndarray_from_2d, info_ndarr
from psana.graphqt.Styles import style

import psana.graphqt.GWROIUtils as roiu
from psana.graphqt.QWPopupSelectItem import popup_select_item_from_list
from psana.graphqt.QWPopupFileName import popup_file_name

class MEDControlROI(QWidget):
    """QWidget with ROI control fields for Mask Editor"""

    def __init__(self, **kwa):

        QWidget.__init__(self, None)

        repoman = kwa.get('repoman', None)
        dirrepo = './' if repoman is None else repoman.dirrepo
        self.fname_json = os.path.join(dirrepo, kwa.get('fname_json', './roi_parameters.json'))
        self.fname_mask = os.path.join(dirrepo, kwa.get('fname_mask', './mask.npy'))
        self.wmain      = kwa.get('parent', None)
        if self.wmain:
            self.wisp = self.wmain.wisp
            self.wimax = self.wmain.wisp.wimax
            self.wspec = self.wmain.wisp.wspec
            self.wim = self.wmain.wisp.wimax.wim
            self.wctl = self.wmain.wctl # MEDControl

        self.but_img,\
        self.but_add,\
        self.but_com,\
        self.but_can,\
        self.but_edi,\
        self.but_sel,\
        self.but_del,\
        self.but_inv,\
        self.but_sav,\
        self.but_loa,\
        self.but_mas = self.list_of_buts\
                     = [QPushButton(s) for s in ('Image', 'Add', 'Compl.', 'Cancel', 'Edit',\
                                                 'Select', 'Delete', 'Invert', 'Save', 'Load', 'Mask')]

        self.box = QVBoxLayout()
        for but in self.list_of_buts:
            self.box.addWidget(but)
            but.clicked.connect(self.on_but_clicked)
        self.box.addStretch()
        self.setLayout(self.box)

        self.set_tool_tips()
        self.set_style()

        self.but_img.setDefault(True)
        self.but_img.setStyleSheet(style.styleButtonGood)

    def set_style(self, width=60):
        self.layout().setContentsMargins(5,10,0,2)
        for but in self.list_of_buts: but.setFixedWidth(width)
        #self.set_buttons_visiable()

    def set_tool_tips(self):
        self.but_add.setToolTip('Add ROI to image\nclick and select ROI type from pop-up menu,'\
                                '\nthen click on image as many times as necessary to define ROI shape')
        self.but_com.setToolTip('Completing add ROI for multiclick input\nin case of PIXGROUP and POLYGON')
        self.but_edi.setToolTip('Click to edit ROI\nthen click on desired ROI and use control handles to edit')
        self.but_sel.setToolTip('Set mode Select\nclick on ROIs to select\nthen click Delete')
        self.but_del.setToolTip('Delete selected ROIs')
        self.but_inv.setToolTip('Invert good/bad pixel selection by ROI\nclick on ROI to swap inversion mode')
        self.but_can.setToolTip('Cancel adding current ROI')
        self.but_loa.setToolTip('Load json file with ROIs')
        self.but_sav.setToolTip('Save current set of ROIs in json file')
        self.but_mas.setToolTip('generate mask for current set of ROIs\nand save it in file')
        self.but_img.setToolTip('Set image control mode\nto scale/translate image by click-and-pan/scroll mouse')

    def on_but_clicked(self):
        for but in self.list_of_buts:
            but.setStyleSheet(style.styleButton)
        for but in self.list_of_buts:
            if but.hasFocus(): break
        logger.debug('Click on button "%s"' % but.text())
        if   but == self.but_add: self.on_but_add()
        elif but == self.but_edi: self.set_mode('E')
        elif but == self.but_sel: self.set_mode('S')
        elif but == self.but_inv: self.set_mode('I')
        elif but == self.but_del: self.wim.delete_selected_roi()
        elif but == self.but_can: self.wim.cancel_add_roi()
        elif but == self.but_com: self.wim.finish()
        elif but == self.but_sav: self.on_but_sav()
        elif but == self.but_loa: self.on_but_loa()
        elif but == self.but_mas: self.on_but_mas()
        elif but == self.but_img: self.on_but_img()
        but.setStyleSheet(style.styleButtonGood)

    def on_but_img(self):
        logger.debug(sys._getframe().f_code.co_name)
        self.set_mode('Q')

    def on_but_add(self):
        logger.debug(sys._getframe().f_code.co_name)
        self.set_mode('A')
        roi_names = ('ROI name:',) + tuple(roiu.roi_names)
        roi_name = popup_select_item_from_list(self.but_add, roi_names, dx=10, dy=-10, do_sorted=False)
        if roi_name == 'ROI name:': roi_name = 'NONE' # roiu.
        roi_key = roiu.dict_roi_name_key[roi_name]
        logger.info('selected ROI: %s\n  click on image as many times as it is necessary to deffine %s parameters' % (roi_name, roi_name))
        self.set_roitype(roi_key)

    def on_but_sav(self):
        path = popup_file_name(parent=self, mode='w', path=self.fname_json, dirs=[], fltr='*.json\n*')
        if path: self.wim.save_parameters_in_file(fname=path)

    def on_but_loa(self):
        path = popup_file_name(parent=self, mode='r', path=self.fname_json, dirs=[], fltr='*.json\n*')
        if path: self.wim.load_parameters_from_file(fname=path)

    def on_but_mas(self):
        path = popup_file_name(parent=self, mode='w', path=self.fname_mask, dirs=[], fltr='*.npy\n*')
        if path:
            path_2d = path.replace('.npy', '-2d.npy')
            mask_2d = self.wim.save_mask(fname=path_2d)
            logger.info(info_ndarr(mask_2d, 'mask 2d') + ' saved in file %s' % path_2d)
            geo = self.wctl.geo
            logger.debug('geo: %s' % str(geo))
            logger.debug('fname_geo: %s' % str(self.wctl.but_geo.text()))

            if geo is not None:
                mask_nda = mask_ndarray_from_2d(mask_2d, geo)
                logger.debug('2d mask converted to ndarray of shape %s' % str(mask_nda.shape))
                np.save(path, mask_nda)
                logger.info(info_ndarr(mask_nda, 'mask_nda') + ' saved in file %s' % path)

    def set_mode(self, ckey):
        if ckey in roiu.mode_keys:
            wim = self.wim
            wim.finish()
            i = roiu.mode_keys.index(ckey)
            wim.mode_type = roiu.mode_types[i]
            wim.mode_name = roiu.mode_names[i]
            logger.info('set mode: %s' % wim.mode_name) # wim.mode_type
            sc = '' if wim.mode_type > roiu.VISIBLE else 'HV'
            wim.set_scale_control(scale_ctl=sc)

    def set_roitype(self, ckey):
        if ckey in roiu.roi_keys:
            wim = self.wim
            i = roiu.roi_keys.index(ckey)
            wim.roi_type = roiu.roi_types[i]
            wim.roi_name = roiu.roi_names[i]
            logger.debug('set roi_type: %d roi_name: %s' % (wim.roi_type, wim.roi_name))


if __name__ == "__main__":

    #os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = MEDControlROI()
    #w.setGeometry(100, 50, 500, 40)
    w.setWindowTitle('MED Control ROI')
    w.show()
    app.exec_()
    del w
    del app

# EOF
