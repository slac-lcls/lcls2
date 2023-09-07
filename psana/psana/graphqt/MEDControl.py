
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

#from psana.graphqt.QWFileNameV2 import QWFileNameV2
#from psana.detector.dir_root import DIR_DATA_TEST
#from psana.graphqt.QWIcons import icon
#import psana.graphqt.QWUtils as qwu

class MEDControl(QWidget):
    """QWidget with control fields for Mask Editor"""

    def __init__(self, **kwa):

        #d = DIR_DATA_TEST + '/misc/'
        #fname_geo = d + 'Select'
        #kwa.setdefault('parent', None)
        #kwa.setdefault('path', d + 'Select')
        #kwa.setdefault('label', 'File:')
        #kwa.setdefault('fltr', '*.text *.txt *.npy *.data *.dat\n*')
        #kwa.setdefault('dirs', dirs_to_search())

        QWidget.__init__(self, None)

        self.wmain = kwa.get('parent', None)
        self.wisp = self.wmain.wisp
        self.wimax = self.wmain.wisp.wimax
        self.wspec = self.wmain.wisp.wspec
        self.wim = self.wmain.wisp.wimax.wim

        self.lab_roi = QLabel('ROI:')
        self.but_add = QPushButton('Add')
        self.but_fin = QPushButton('Finish')
        self.but_can = QPushButton('Cancel')
        self.but_edi = QPushButton('Edit')
        self.but_sel = QPushButton('Select')
        self.but_del = QPushButton('Delete')
        self.but_inv = QPushButton('Invert')
        self.but_sav = QPushButton('Save')
        self.but_loa = QPushButton('Load')
        self.but_mas = QPushButton('Mask')
        self.but_imc = QPushButton('Image ctrl')

        self.list_of_buts = (
          self.but_add,
          self.but_fin,
          self.but_can,
          self.but_edi,
          self.but_sel,
          self.but_del,
          self.but_inv,
          self.but_sav,
          self.but_loa,
          self.but_mas,
        )

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.but_imc)
        self.hbox.addWidget(self.lab_roi)
        for but in self.list_of_buts:
            self.hbox.addWidget(but)
            but.clicked.connect(self.on_but_clicked)
        self.hbox.addStretch()
        self.setLayout(self.hbox)

        self.but_imc.clicked.connect(self.on_but_imc)


#        self.wfnm_geo = QWFileNameV2(None, label='Geometry:',\
#                                     path=fname_geo, fltr='*.txt *.data\n*', dirs=dirs_to_search())

        self.set_tool_tips()
        self.set_style()

    def set_style(self):
        self.layout().setContentsMargins(5,2,5,2)
        self.lab_roi.setStyleSheet(style.styleLabel)
        self.but_add.setFixedWidth(50)
        #self.set_buttons_visiable()

    def set_tool_tips(self):
        self.but_add.setToolTip('Set mode adding ROI to image\nselect ROI type and click on image')
        self.but_fin.setToolTip('Finishing add ROI for multiclick input\nin case of PIXGROUP and POLYGON')
        self.but_edi.setToolTip('Turn on edit mode\nthen click on ROI to select and use control handles to edit')
        self.but_sel.setToolTip('Select ROIs then click Delete')
        self.but_del.setToolTip('Delete selected ROIs')
        self.but_inv.setToolTip('Invert good/bad pixel selection by ROI')
        self.but_can.setToolTip('Cancel adding current ROI')
        self.but_loa.setToolTip('Load json file with ROIs')
        self.but_sav.setToolTip('Save current set of ROIs in json file')
        self.but_mas.setToolTip('generate mask for current set of ROIs\nand save it in file')
        self.but_imc.setToolTip('Set image control mode\nto scale/translate image')

    def on_but_clicked(self):

        for but in self.list_of_buts:
            but.setStyleSheet(style.styleButton)

        for but in self.list_of_buts:
            if but.hasFocus(): break
        logger.info('Click on button "%s"' % but.text())
        if   but == self.but_add: self.on_but_add()
        elif but == self.but_edi: self.set_mode('E')
        elif but == self.but_sel: self.set_mode('S')
        elif but == self.but_inv: self.set_mode('I')
        elif but == self.but_del: self.wim.delete_selected_roi()
        elif but == self.but_can: self.wim.cancel_add_roi()
        elif but == self.but_fin: self.wim.finish()
        elif but == self.but_sav: self.on_but_sav()
        elif but == self.but_loa: self.on_but_loa()
        elif but == self.but_mas: self.on_but_mas()
        but.setStyleSheet(style.styleButtonGood)

    def on_but_imc(self):
        logger.info(sys._getframe().f_code.co_name)
        self.set_mode('Q')

    def on_but_add(self):
        logger.info(sys._getframe().f_code.co_name)
        self.set_mode('A')
        roi_name = popup_select_item_from_list(self.but_add, roiu.roi_names, dx=10, dy=-10)
        roi_key = roiu.dict_roi_name_key[roi_name]
        logger.info('Selected ROI: %s key: %s' % (roi_name, roi_key))
        self.set_roitype(roi_key)

    def on_but_sav(self):
        self.wim.save_parameters_in_file()

    def on_but_loa(self):
        self.wim.load_parameters_from_file()

    def on_but_mas(self):
        self.wim.save_mask()

    def set_mode(self, ckey):
        if ckey in roiu.mode_keys:
            wim = self.wim
            wim.finish()
            i = roiu.mode_keys.index(ckey)
            wim.mode_type = roiu.mode_types[i]
            wim.mode_name = roiu.mode_names[i]
            logger.info('set mode_type: %d roi_name: %s' % (wim.mode_type, wim.mode_name))
            sc = '' if wim.mode_type > roiu.VISIBLE else 'HV'
            wim.set_scale_control(scale_ctl=sc)

    def set_roitype(self, ckey):
        if ckey in roiu.roi_keys:
            wim = self.wim
            i = roiu.roi_keys.index(ckey)
            wim.roi_type = roiu.roi_types[i]
            wim.roi_name = roiu.roi_names[i]
            logger.info('set roi_type: %d roi_name: %s' % (wim.roi_type, wim.roi_name))

"""
    def on_spectrum_range_changed(self, d):
        logger.debug('on_spectrum_range_changed: %s' % str(d))
        w = cp.ivimageaxes
        if w is not None:
            wi = w.wimg
            rs=wi.scene().sceneRect() # preserve current scene rect
            mode, nbins, amin, amax, frmin, frmax = self.spectrum_parameters()
            w.wimg.set_pixmap_from_arr(wi.arr, set_def=True, amin=amin, amax=amax, frmin=frmin, frmax=frmax)
            wi.set_rect_scene(rs, set_def=False)


    def on_color_table_changed(self):
        w = cp.ivspectrum
        if w is None:
            logger.debug('on_color_table_changed - do nothing here')
            return
        ctab =  w.wcbar.color_table()
        logger.debug(info_ndarr(ctab, 'on_color_table_changed: new color table'))
        w = cp.ivimageaxes
        if w is not None:
            wi = w.wimg
            wi.set_coltab(coltab=ctab)
            rs=wi.scene().sceneRect() # preserve current scene rect
            wi.set_pixmap_from_arr(wi.arr, set_def=False)
            wi.set_rect_scene(rs, set_def=False)

    def set_signal_fast(self, is_fast=True):
        for w in (cp.ivimageaxes, cp.ivspectrum):
          if w is not None: w.set_signal_fast(is_fast)

    def connect_image_scene_rect_changed(self):
        w = cp.ivimageaxes
        if w is not None: w.connect_image_scene_rect_changed(self.on_image_scene_rect_changed)

    def disconnect_image_scene_rect_changed(self):
        w = cp.ivimageaxes
        if w is not None: w.disconnect_image_scene_rect_changed(self.on_image_scene_rect_changed)

    def on_image_scene_rect_changed(self, r):
        wimg = cp.ivimageaxes.wimg
        a = wimg.array_in_rect(r)
        logger.debug('on_image_scene_rect_changed: %s' % qu.info_rect_xywh(r))
        #logger.debug('on_image_scene_rect_changed: %s\n    %s' % (qu.info_rect_xywh(r), info_ndarr(a, 'selected array in rect', first=0, last=3)))
        self.set_spectrum_from_arr(a, update_hblimits=False)

    def connect_image_pixmap_changed(self):
        w = cp.ivimageaxes
        if w is not None: w.wimg.connect_image_pixmap_changed(self.on_image_pixmap_changed)

    def disconnect_image_pixmap_changed(self):
        w = cp.ivimageaxes
        if w is not None: w.wimg.disconnect_image_pixmap_changed(self.on_image_pixmap_changed)

    def on_image_pixmap_changed(self):
        logger.debug('on_image_pixmap_changed')
        wimg = cp.ivimageaxes.wimg
        arr = wimg.array_in_rect()
        coltab = wimg.coltab
        self.set_spectrum_from_arr(arr, update_hblimits=False)

    def set_spectrum_from_arr(self, arr, edgemode=0, update_hblimits=True): #, nbins=1000, amin=None, amax=None, frmin=0.001, frmax=0.999, edgemode=0):
        if arr is self.arr_his_old: return
        self.arr_his_old = arr
        w = cp.ivspectrum
        if w is not None:
            #r = w.whis.scene().sceneRect()
            mode, nbins, amin, amax, frmin, frmax = self.spectrum_parameters()
            w.whis.set_histogram_from_arr(arr, nbins, amin, amax, frmin, frmax, edgemode, update_hblimits)

    def connect_histogram_scene_rect_changed(self):
        w = cp.ivspectrum
        if w is not None: w.connect_histogram_scene_rect_changed(self.on_histogram_scene_rect_changed)

    def disconnect_histogram_scene_rect_changed(self):
        w = cp.ivspectrum
        if w is not None: w.disconnect_histogram_scene_rect_changed(self.on_histogram_scene_rect_changed)

    def on_histogram_scene_rect_changed(self, r):
        x1,y1,x2,y2 = r.getCoords()
        logger.debug('on_histogram_scene_rect_changed: %s reset image for spectal value in range %.3f:%.3f '%\
                     (qu.info_rect_xywh(r),y1,y2))
        w = cp.ivimageaxes
        if w is not None:
            wi = w.wimg
            rs=wi.scene().sceneRect() # preserve current scene rect
            wi.set_pixmap_from_arr(wi.arr, set_def=False, amin=y1, amax=y2)
            wi.set_rect_scene(rs, set_def=False)

    def on_but_exp(self):
        from psana.graphqt.PSPopupSelectExp import select_instrument_experiment
        dir_instr = cp.instr_dir.value()
        instr_name, exp_name = select_instrument_experiment(self.but_exp, dir_instr, show_frame=True)
        logger.debug('selected experiment: %s' % exp_name)
        if instr_name:
            cp.instr_name.setValue(instr_name)
        if exp_name:
            self.but_exp.setText(exp_name)
            cp.exp_name.setValue(exp_name)
            dirs = dirs_to_search()
            logger.debug('set dirs_to_search: %s' % str(dirs))
            self.wfnm_nda.set_dirs_to_search(dirs)
            self.wfnm_geo.set_dirs_to_search(dirs)

    def on_but_reset(self):
        logger.debug('on_but_reset')
        if cp.ivimageaxes is not None:
           cp.ivimageaxes.reset_original_size()

        if cp.ivspectrum is not None:
           cp.ivspectrum.reset_original_size()

    def spectrum_parameters(self):
        d = self.wctl_spec.spectrum_parameters()
        return\
          d.get('mode', 'fraction'),\
          d.get('nbins', 1000),\
          d.get('amin',  None),\
          d.get('amax',  None),\
          d.get('frmin', 0.001),\
          d.get('frmax', 0.999)

    def on_changed_fname_nda(self, fname):
        logger.debug('on_changed_fname_nda: %s' % fname)
        wia = cp.ivimageaxes
        if wia is not None:
           nda = psu.load_ndarray_from_file(fname)
           cp.last_selected_fname.setValue(fname)

           logger.debug(info_ndarr(nda,'nda'))
           img = image_from_ndarray(nda)
           self.wctl_spec.set_amin_amax_def(nda.min(), nda.max())
           mode, nbins, amin, amax, frmin, frmax = self.spectrum_parameters()
           wia.wimg.set_pixmap_from_arr(img, set_def=True, amin=amin, amax=amax, frmin=frmin, frmax=frmax)

           h,w = img.shape
           aspect = float(w)/h
           if aspect>1.5 or aspect<0.7:
               ww = max(h,w)
               rs = QRectF(-0.5*(ww-min(h,w)), 0, ww, ww)
               wia.wimg.set_rect_scene(rs, set_def=False)

    def on_changed_fname_geo(self, s):
        logger.debug('on_changed_fname_geo: %s' % s)

    def on_buts(self):
        logger.debug('on_buts')
        from psana.graphqt.QWUtils import change_check_box_dict_in_popup_menu
        d = self.buttons_dict()
        logger.debug('initial visible buttons: %s' % str(d))
        resp = change_check_box_dict_in_popup_menu(d, 'Select buttons',\
                msg='Check visible buttons then click Apply or Cancel', parent=self.but_buts)
        #logger.debug('select_visible_buttons resp: %s' % resp)
        if resp == 0:
            logger.debug('Visible buttons selection is cancelled')
            return
        else:
            logger.debug('selected visible buttons: %s' % str(d))
            self.set_buttons_visiable(d)
            self.set_buttons_config_bitword(d)

    def set_buttons_visiable(self, dic_buts=None):
        d = self.buttons_dict() if dic_buts is None else dic_buts
        #logger.debug('dic_buts: %s' % str(d))
        self.wfnm_geo.setVisible(d['Geometry'])
        self.but_reset.setVisible(d['Reset'])
        self.but_tabs.setVisible(d['Tabs'])
        self.but_exp.setVisible(d['Experiment'])
        self.but_save.setVisible(d['Save'])
        if cp.ivimageaxes is not None: cp.ivimageaxes.but_reset.setVisible(d['Reset image'])
        if cp.ivimageaxes is not None: cp.ivimageaxes.set_info_visible(d['Cursor position'])
        if cp.ivspectrum  is not None: cp.ivspectrum.but_reset.setVisible(d['Reset spectrum'])
        self.wctl_spec.setVisible(d['Control spectrum'])

    def buttons_dict(self):
        r = cp.iv_buttons.value()
        return {'Geometry'        : r & 1,\
                'Tabs'            : r & 2,\
                'Reset'           : r & 4,\
                'Reset image'     : r & 8,\
                'Reset spectrum'  : r & 16,\
                'Cursor position' : r & 32,\
                'Control spectrum': r & 64,\
                'Experiment'      : r & 128,\
                'Save'            : r & 256,\
               }

    def set_buttons_config_bitword(self, d):
        w = 0
        if d['Geometry']        : w |= 1
        if d['Tabs']            : w |= 2
        if d['Reset']           : w |= 4
        if d['Reset image']     : w |= 8
        if d['Reset spectrum']  : w |= 16
        if d['Cursor position'] : w |= 32
        if d['Control spectrum']: w |= 64
        if d['Experiment']      : w |= 128
        if d['Save']            : w |= 256
        cp.iv_buttons.setValue(w)

    def closeEvent(self, e):
        QWidget.closeEvent(self, e)
        logger.debug('closeEvent')
"""

if __name__ == "__main__":

    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
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
