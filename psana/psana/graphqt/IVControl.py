
"""Class :py:class:`IVControl` is a Image Viewer QWidget with control fields
============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVControl.py

    from psana.graphqt.IVControl import IVControl
    w = IVControl()

Created on 2021-06-14 by Mikhail Dubrovin
"""

from psana.graphqt.CMWControlBase import * #cp, CMWControlBase, QApplication, os, sys, logging, QRectF, QWFileNameV2
from psana.graphqt.IVControlSpec import IVControlSpec
import psana.pyalgos.generic.PSUtils as psu
from psana.pyalgos.generic.NDArrUtils import reshape_to_2d, info_ndarr, np
import psana.graphqt.QWUtils as qu

logger = logging.getLogger(__name__)


def image_from_ndarray(nda):
    if nda is None:
       logger.warning('nda is None - return None for image')
       return None

    if not isinstance(nda, np.ndarray):
       logger.warning('nda is not np.ndarray, type(nda): %s - return None for image' % type(nda))
       return None

    img = psu.table_nxn_epix10ka_from_ndarr(nda) if (nda.size % (352*384) == 0) else\
          psu.table_nxm_jungfrau_from_ndarr(nda) if (nda.size % (512*1024) == 0) else\
          psu.table_nxm_cspad2x1_from_ndarr(nda) if (nda.size % (185*388) == 0) else\
          reshape_to_2d(nda)
    logger.debug(info_ndarr(img,'img'))
    return img


class IVControl(CMWControlBase):
    """QWidget for Image Viewer control fields"""

    def __init__(self, **kwa):

        d = '/cds/group/psdm/detector/data2_test/misc/'
        fname_geo = d + 'Select'
        kwa.setdefault('parent', None)
        kwa.setdefault('path', d + 'Select')
        kwa.setdefault('label', 'File:')
        kwa.setdefault('fltr', '*.text *.txt *.npy *.data *.dat\n*')
        kwa.setdefault('dirs', dirs_to_search())

        CMWControlBase.__init__(self, **kwa)
        cp.ivcontrol = self
        self.arr_his_old = None
        self.arr_img_old = None

        self.wfnm_nda = self.wfnm # from base class
        self.wfnm_geo = QWFileNameV2(None, label='Geometry:',\
                                     path=fname_geo, fltr='*.txt *.data\n*', dirs=dirs_to_search())

        self.but_view.setVisible(False) # from CMWControlBase

        self.but_exp   = QPushButton(cp.exp_name.value())
        self.but_reset = QPushButton('Reset')
        self.but_buts  = QPushButton('Buts %s' % cp.char_expand)
        self.wctl_spec = IVControlSpec()

        self.box = QGridLayout()
        self.box.addWidget(self.wfnm_nda, 0, 0, 1, 5)
        self.box.addWidget(self.but_save,    0, 6)
        self.box.addWidget(self.but_exp,     0, 7)
        self.box.addWidget(self.but_reset,   0, 8)
        self.box.addWidget(self.but_buts,    0, 9)
        self.box.addWidget(self.but_tabs,    0, 10)
        self.box.addWidget(self.wfnm_geo, 1, 0, 1, 5)
        self.box.addWidget(self.wctl_spec,1, 5, 1, 6)
        self.setLayout(self.box)

        self.but_exp.clicked.connect(self.on_but_exp)
        self.but_reset.clicked.connect(self.on_but_reset)
        self.but_buts.clicked.connect(self.on_buts)
        self.wfnm_nda.connect_path_is_changed(self.on_changed_fname_nda)
        self.wfnm_geo.connect_path_is_changed(self.on_changed_fname_geo)
        self.wctl_spec.connect_signal_spectrum_range_changed(self.on_spectrum_range_changed)

        self.connect_image_scene_rect_changed()
        self.connect_histogram_scene_rect_changed()
        self.connect_image_pixmap_changed()

        w = cp.ivspectrum
        if w is not None: w.wcbar.connect_new_color_table(self.on_color_table_changed)

        self.set_tool_tips()
        self.set_style()
        self.set_buttons_visiable()


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


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        self.but_reset.setToolTip('Reset original image size')
        self.but_buts.setToolTip('Show/hide buttons')
        self.but_exp.setToolTip('Select experiment')


    def set_style(self):
        CMWControlBase.set_style(self)
        self.but_reset.setFixedWidth(50)
        self.but_buts.setFixedWidth(50)
        self.but_exp.setFixedWidth(80)
        self.wfnm_geo.lab.setStyleSheet(style.styleLabel)


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
        logger.debug('closeEvent')
        CMWControlBase.closeEvent(self, e)
        cp.ivcontrol = None


if __name__ == "__main__":

    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = IVControl()
    w.setGeometry(100, 50, 500, 80)
    w.setWindowTitle('IV Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
