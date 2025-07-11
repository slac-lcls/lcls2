
"""Class :py:class:`GWImageAxes` is a QWidget with image and two axes
=====================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/GWImageAxes.py

    from psana2.graphqt.GWImageAxes import GWImageAxes
    w = GWImageAxes()

Created on 2021-06-22 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)

from psana2.graphqt.GWViewImage import *
from psana2.graphqt.GWViewAxis import GWViewAxis
import psana2.graphqt.ColorTable as ct
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from psana2.pyalgos.generic.NDArrGenerators import test_image

class GWImageAxes(QWidget):
    """QWidget for Image Viewer"""
    image_scene_rect_changed = pyqtSignal('QRectF')

    def __init__(self, **kwargs):

        parent      = kwargs.get('parent', None)
        image       = kwargs.get('image', test_image(shape=(256,256)))
        ctab        = kwargs.get('ctab', ct.color_table_interpolated())
        signal_fast = kwargs.get('signal_fast', False)
        #origin      = kwargs.get('origin', 'UL')
        #scale_ctl   = kwargs.get('scale_ctl', 'HV')

        QWidget.__init__(self, parent)

        self.wim = GWViewImage(self, image, coltab=ctab, origin='UL', scale_ctl='HV', signal_fast=signal_fast)

        self.rs_old = None
        r = self.wim.sceneRect()
        rscx = QRectF(r.x(), 0, r.width(), 1)
        rscy = QRectF(0, r.y(), 1, r.height())

        self.wax = GWViewAxis(None, rscx, side='U', origin='UL', scale_ctl=True, wwidth=30, wlength=200, signal_fast=signal_fast)
        self.way = GWViewAxis(None, rscy, side='R', origin='UR', scale_ctl=True, wwidth=60, wlength=200, signal_fast=signal_fast)

        self.but_reset = QPushButton('Reset')
        self.edi_info = QTextEdit('Info')

        self.box = QGridLayout()
        self.box.setSpacing(0)
        self.box.setVerticalSpacing(0)
        self.box.setHorizontalSpacing(0)
        self.box.addWidget(self.edi_info,  0,  0,  1, 11)
        self.box.addWidget(self.way,       1,  0,  9,  1)
        self.box.addWidget(self.wim,       1,  1,  9, 10)
        self.box.addWidget(self.wax,      10,  1,  1, 10)
        self.box.addWidget(self.but_reset,10,  0, alignment=Qt.AlignCenter)
        self.setLayout(self.box)

        self.set_tool_tips()
        self.set_style()

        self.connect_scene_rect_changed()
        self.but_reset.clicked.connect(self.on_but_reset)
        self.set_info_visible(True)

    def set_signal_fast(self, is_fast=False):
        self.wim.signal_fast = is_fast
        self.wax.signal_fast = is_fast
        self.way.signal_fast = is_fast

    def set_info_visible(self, is_visible=True):
        self.edi_info.setVisible(is_visible)
        if self.edi_info.isVisible() and is_visible or\
           (not self.edi_info.isVisible()) and (not is_visible): return
        elif is_visible: self.wim.connect_mouse_move_event(self.on_mouse_move_event)
        else: self.wim.disconnect_mouse_move_event(self.on_mouse_move_event)

    def connect_scene_rect_changed(self):
        self.wim.connect_scene_rect_changed(self.on_wim_scene_rect_changed)
        self.wax.connect_scene_rect_changed(self.on_wax_scene_rect_changed)
        self.way.connect_scene_rect_changed(self.on_way_scene_rect_changed)

    def disconnect_scene_rect_changed(self):
        self.wim.disconnect_scene_rect_changed(self.on_wim_scene_rect_changed)
        self.wax.disconnect_scene_rect_changed(self.on_wax_scene_rect_changed)
        self.way.disconnect_scene_rect_changed(self.on_way_scene_rect_changed)

    def on_but_reset(self):
        self.wim.reset_scene_rect()
        logger.debug('on_but_reset wim.scene_rect: %s' % qu.info_rect_xywh(self.wim.scene_rect()))
        self.wax.reset_scene_rect()
        self.way.reset_scene_rect()

    def on_wim_scene_rect_changed(self, r):
        self.wax.set_axis_limits(r.x(), r.x()+r.width())
        self.way.set_axis_limits(r.y(), r.y()+r.height())
        #self.wax.fit_in_view(QRectF(r.x(), 0, r.width(), 1))
        #self.way.fit_in_view(QRectF(0, r.y(), 1, r.height()))
        self.emit_signal_if_image_scene_rect_changed()

    def on_wax_scene_rect_changed(self, r):
        #logger.debug('on_wax_scene_rect_changed: %s'%str(r))
        rs = self.wim.scene_rect()
        self.wim.fit_in_view(QRectF(r.x(), rs.y(), r.width(), rs.height()))
        self.emit_signal_if_image_scene_rect_changed()

    def on_way_scene_rect_changed(self, r):
        #logger.debug('on_way_scene_rect_changed: %s'%str(r))
        rs = self.wim.scene_rect()  # scene().sceneRect()
        self.wim.fit_in_view(QRectF(rs.x(), r.y(), rs.width(), r.height()))
        self.emit_signal_if_image_scene_rect_changed()

    def emit_signal_if_image_scene_rect_changed(self):
        """Checks if scene rect have changed and submits signal with new rect."""
        rs = self.wim.scene_rect()
        if rs != self.rs_old:
            self.rs_old = rs
            self.image_scene_rect_changed.emit(rs)

    def connect_image_scene_rect_changed(self, recip):
        self.image_scene_rect_changed.connect(recip)

    def disconnect_image_scene_rect_changed(self, recip):
        self.image_scene_rect_changed.disconnect(recip)

    def set_tool_tips(self):
        self.wim.setToolTip('Image\npixel map')
        self.wax.setToolTip('Image columns\nH-scale')
        self.way.setToolTip('Image rows\nV-scale')
        self.edi_info.setToolTip('Information field')

    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.but_reset.setFixedSize(60,30)
        self.edi_info.setMaximumHeight(30)
        self.edi_info.setStyleSheet(self.wim.style_def)

    def set_pixmap_from_arr(self, arr, set_def=True, amin=None, amax=None, frmin=0.001, frmax=0.999):
        """shortcut to image"""
        self.wim.set_pixmap_from_arr(arr, set_def, amin, amax, frmin, frmax)
        r = self.wim.scene_rect()
        self.wax.set_axis_limits(r.x(), r.x()+r.width())   # fit_in_view(QRectF(r.x(), 0, r.width(), 1))
        self.way.set_axis_limits(r.y(), r.y()+r.height())  # fit_in_view(QRectF(0, r.y(), 1, r.height()))
        self.wax.reset_scene_rect_default()
        self.way.reset_scene_rect_default()

    def on_mouse_move_event(self, e):
        """Overrides method from GWView"""
        wimg = self.wim
        p = wimg.mapToScene(e.pos())
        ix, iy, v = wimg.cursor_on_image_pixcoords_and_value(p)
        fv = 0 if v is None else v
        s = 'x:%d y:%d v:%s%s' % (ix, iy, '%.3f'%fv, 25*' ')
        #self.setWindowTitle('GWViewImage %s'%s)
        self.edi_info.setText(s)

    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)
        #cp.gwimageaxes = None

if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
