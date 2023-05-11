
"""Class :py:class:`GWImageAxes` is a QWidget with image and two axes
=====================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/GWImageAxes.py

    from psana.graphqt.GWImageAxes import GWImageAxes
    w = GWImageAxes()

Created on 2021-06-22 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.GWViewImage import GWViewImage
from psana.graphqt.GWViewAxis import GWViewAxis
import psana.graphqt.ColorTable as ct
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from psana.pyalgos.generic.NDArrGenerators import test_image
from psana.graphqt.CMConfigParameters import cp


class GWImageAxes(QWidget):
    """QWidget for Image Viewer"""
    image_scene_rect_changed = pyqtSignal('QRectF')

    def __init__(self, **kwargs):

        parent = kwargs.get('parent', None)
        image = kwargs.get('image', test_image(shape=(16,16)))
        ctab = kwargs.get('ctab', ct.color_table_interpolated())
        signal_fast = kwargs.get('signal_fast', False)

        QWidget.__init__(self, parent)
        cp.ivimageaxes = self

        self.wimg = GWViewImage(self, image, coltab=ctab, origin='UL', scale_ctl='HV', signal_fast=signal_fast)

        self.rs_old = None
        r = self.wimg.sceneRect()
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
        self.box.addWidget(self.wimg,      1,  1,  9, 10)
        self.box.addWidget(self.wax,      10,  1,  1, 10)
        self.box.addWidget(self.but_reset,10,  0, alignment=Qt.AlignCenter)
        self.setLayout(self.box)

        self.set_tool_tips()
        self.set_style()

        self.connect_scene_rect_changed()
        self.but_reset.clicked.connect(self.on_but_reset)
        self.set_info_visible(True)


    def set_signal_fast(self, is_fast=False):
        self.wimg.signal_fast = is_fast
        self.wax.signal_fast = is_fast
        self.way.signal_fast = is_fast


    def set_info_visible(self, is_visible=True):
        self.edi_info.setVisible(is_visible)
        if self.edi_info.isVisible() and is_visible or\
           (not self.edi_info.isVisible()) and (not is_visible): return
        elif is_visible: self.wimg.connect_mouse_move_event(self.on_mouse_move_event)
        else: self.wimg.disconnect_mouse_move_event(self.on_mouse_move_event)


    def connect_scene_rect_changed(self):
        self.wimg.connect_scene_rect_changed(self.on_wimg_scene_rect_changed)
        self.wax.connect_scene_rect_changed(self.on_wax_scene_rect_changed)
        self.way.connect_scene_rect_changed(self.on_way_scene_rect_changed)


    def disconnect_scene_rect_changed(self):
        self.wimg.disconnect_scene_rect_changed(self.on_wimg_scene_rect_changed)
        self.wax.disconnect_scene_rect_changed(self.on_wax_scene_rect_changed)
        self.way.disconnect_scene_rect_changed(self.on_way_scene_rect_changed)


    def on_but_reset(self):
        logger.debug('on_but_reset')
        #self.wimg.reset_original_size()
        #self.wax.reset_original_size()
        #self.way.reset_original_size()
        self.wimg.reset_scene_rect()
        self.wax.reset_scene_rect()
        self.way.reset_scene_rect()


    def on_wimg_scene_rect_changed(self, r):
        #logger.debug('on_wimg_scene_rect_changed: %s'%str(r))
        #self.wax.set_view(rs=QRectF(r.x(), 0, r.width(), 1))
        #self.way.set_view(rs=QRectF(0, r.y(), 1, r.height()))
        self.wax.fit_in_view(QRectF(r.x(), 0, r.width(), 1))
        self.way.fit_in_view(QRectF(0, r.y(), 1, r.height()))
        self.emit_signal_if_image_scene_rect_changed()


    def on_wax_scene_rect_changed(self, r):
        #logger.debug('on_wax_scene_rect_changed: %s'%str(r))
        rs = self.wimg.scene().sceneRect()
        self.wimg.fit_in_view(QRectF(r.x(), rs.y(), r.width(), rs.height()))
        self.emit_signal_if_image_scene_rect_changed()


    def on_way_scene_rect_changed(self, r):
        #logger.debug('on_way_scene_rect_changed: %s'%str(r))
        rs = self.wimg.scene().sceneRect()
        self.wimg.fit_in_view(QRectF(rs.x(), r.y(), rs.width(), r.height()))
        self.emit_signal_if_image_scene_rect_changed()


    def emit_signal_if_image_scene_rect_changed(self):
        """Checks if scene rect have changed and submits signal with new rect.
        """
        rs = self.wimg.scene().sceneRect()
        if rs != self.rs_old:
            self.rs_old = rs
            self.image_scene_rect_changed.emit(rs)


    def connect_image_scene_rect_changed(self, recip):
        self.image_scene_rect_changed.connect(recip)


    def disconnect_image_scene_rect_changed(self, recip):
        self.image_scene_rect_changed.disconnect(recip)


    def set_tool_tips(self):
        self.wimg.setToolTip('Image\npixel map')
        self.wax.setToolTip('Image columns\nH-scale')
        self.way.setToolTip('Image rows\nV-scale')
        self.edi_info.setToolTip('Information field')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.but_reset.setFixedSize(60,30)
        self.edi_info.setMaximumHeight(30)


    def set_pixmap_from_arr(self, arr, set_def=True, amin=None, amax=None, frmin=0.001, frmax=0.999):
        """shortcut to image"""
        self.wimg.set_pixmap_from_arr(arr, set_def, amin, amax, frmin, frmax)


    def reset_original_size(self):
        """shortcut to image"""
        self.wimg.reset_original_size()


    def on_mouse_move_event(self, e):
        """Overrides method from GWView"""
        wimg = self.wimg
        p = wimg.mapToScene(e.pos())
        ix, iy, v = wimg.cursor_on_image_pixcoords_and_value(p)
        fv = 0 if v is None else v
        s = 'x:%d y:%d v:%s%s' % (ix, iy, '%.3f'%fv, 25*' ')
        #self.setWindowTitle('GWViewImage %s'%s)
        self.edi_info.setText(s)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)
        cp.ivimageaxes = None


if __name__ == "__main__":
    import os
    import sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='[%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = GWImageAxes()
    w.setGeometry(100, 50, 800, 800)
    w.setWindowTitle('Image with two axes')
    w.show()
    app.exec_()
    del w
    del app

# EOF
