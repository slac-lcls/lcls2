
"""Class :py:class:`GGWImageSpec` is a QWidget with image and spectrum
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/examples/ex_GWImageSpec.py

    from psana.graphqt.GWImageSpec import GWImageSpec
    w = GWImageAxesROI()

Created on 2023-09-01 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)

from psana.graphqt.GWImageAxesROI import *
from psana.graphqt.GWSpectrum import GWSpectrum
import psana.graphqt.ColorTable as ct
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QTextEdit, QSplitter, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from psana.pyalgos.generic.NDArrGenerators import test_image

class GWImageSpec(QWidget):
    """QWidget for Mask Editor"""
    #image_scene_rect_changed = pyqtSignal('QRectF')

    def __init__(self, **kwa):

        parent      = kwa.get('parent', None)
        image       = kwa.get('image', test_image(shape=(256,256)))
        ctab        = kwa.get('ctab', ct.color_table_interpolated())
        signal_fast = kwa.get('signal_fast', False)
        #origin      = kwa.get('origin', 'UL')
        #scale_ctl   = kwa.get('scale_ctl', 'HV')

        QWidget.__init__(self, parent)

        self.wimax = GWImageAxesROI(parent=parent, image=image, ctab=ctab, signal_fast=signal_fast)
        self.wspec = GWSpectrum(parent=parent, image=image, ctab=ctab, signal_fast=signal_fast)

        self.hspl = QSplitter(Qt.Horizontal)
        self.hspl.addWidget(self.wimax)
        self.hspl.addWidget(self.wspec)
        #self.hspl.setSizes((600, 300))  #spl_pos = self.vspl.sizes()[0]
        self.vbox = QVBoxLayout()
        #self.vbox.addWidget(self.wctrl)
        self.vbox.addWidget(self.hspl)
        self.setLayout(self.vbox)

#        self.but_reset = QPushButton('Reset')
#        self.edi_info = QTextEdit('Info')

        self.set_tool_tips()
        self.set_style()

        self.connect_image_scene_rect_changed()
        self.connect_histogram_scene_rect_changed()
        self.connect_new_color_table()
        self.connect_image_pixmap_changed()

    def set_tool_tips(self):
        self.wimax.setToolTip('Image')

    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)

    def set_splitter_pos(self, fr=0.8):
        #print('XXX', sys._getframe().f_code.co_name)
        #self.wimax.setMinimumWidth(200)
        #self.wspec.setMinimumWidth(200)
        wid = self.width()
        s = int(fr*wid)
        self.hspl.setSizes((s, wid-s))  #spl_pos = self.vspl.sizes()[0]

    def resizeEvent(self, e):
        QWidget.resizeEvent(self, e)
        self.set_splitter_pos()

    def connect_image_scene_rect_changed(self):
        #self.wimax.connect_image_scene_rect_changed(self.wimax.test_image_scene_rect_changed)
        self.wimax.connect_image_scene_rect_changed(self.on_image_scene_rect_changed)

    def disconnect_image_scene_rect_changed(self):
        self.wimax.disconnect_image_scene_rect_changed(self.on_image_scene_rect_changed)

    def on_image_scene_rect_changed(self, r=None):
        if r is None: r=self.wimax.wim.scene_rect()
        #print(sys._getframe().f_code.co_name + ' %s' % qu.info_rect_xywh(r), end='\r')
        logger.debug(sys._getframe().f_code.co_name + ' %s' % qu.info_rect_xywh(r))
        self.wimax.wim.arr_limits_old = None
        a = self.wimax.wim.array_in_rect(rect=r)
        self.disconnect_histogram_scene_rect_changed()
        self.wspec.set_spectrum_from_arr(a)
        self.connect_histogram_scene_rect_changed()

    def connect_image_pixmap_changed(self):
        self.wimax.wim.connect_image_pixmap_changed(self.on_image_scene_rect_changed)

    def disconnect_image_pixmap_changed(self):
        self.wimax.wim.disconnect_image_pixmap_changed(self.on_image_scene_rect_changed)

    def connect_histogram_scene_rect_changed(self):
        #self.wspec.connect_histogram_scene_rect_changed(self.wspec.test_histogram_scene_rect_changed)
        self.wspec.connect_histogram_scene_rect_changed(self.on_histogram_scene_rect_changed)

    def disconnect_histogram_scene_rect_changed(self):
        #self.wspec.connect_histogram_scene_rect_changed(self.wspec.test_histogram_scene_rect_changed)
        self.wspec.disconnect_histogram_scene_rect_changed(self.on_histogram_scene_rect_changed)

    def on_histogram_scene_rect_changed(self, r):
        #print(sys._getframe().f_code.co_name + ' %s' % qu.info_rect_xywh(r), end='\r')
        logger.debug(sys._getframe().f_code.co_name + ' %s' % qu.info_rect_xywh(r))
        wim = self.wimax.wim
        self.disconnect_image_scene_rect_changed()
        self.disconnect_image_pixmap_changed()
        wim.set_pixmap_from_arr(wim.arr, set_def=True, amin=r.top(), amax=r.bottom(), frmin=0, frmax=1) # , frmin=0.00001, frmax=0.99999)
        self.connect_image_scene_rect_changed()
        self.connect_image_pixmap_changed()

    def connect_new_color_table(self):
        #self.wspec.wcbar.connect_new_color_table(self.wspec.wcbar.test_new_color_table_reception)
        self.wspec.wcbar.connect_new_color_table(self.on_new_color_table)

    def on_new_color_table(self):
        wcb = self.wspec.wcbar
        wim = self.wimax.wim
        logger.debug(sys._getframe().f_code.co_name + ': %s' % str(wcb._ctab[:5]))
        self.disconnect_image_scene_rect_changed()
        wim.set_coltab(coltab=wcb._ctab)
        wim.set_pixmap_from_arr(wim.arr, frmin=0, frmax=1)
        self.connect_image_scene_rect_changed()


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
