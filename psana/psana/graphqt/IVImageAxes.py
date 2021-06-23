
"""Class :py:class:`IVImageAxes` is a QWidget with image and two axes
=====================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVImageAxes.py

    from psana.graphqt.IVImageAxes import IVImageAxes
    w = IVImageAxes()

Created on 2021-06-22 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.FWViewImage import FWViewImage
from psana.graphqt.FWViewAxis import FWViewAxis
import psana.graphqt.ColorTable as ct
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton#, QSizePolicy
from PyQt5.QtCore import Qt, QRectF #, QPointF, QPoint, QRectF

def test_image():
  import psana.pyalgos.generic.NDArrGenerators as ag
  return ag.random_standard((8,12), mu=0, sigma=10)


class IVImageAxes(QWidget):
    """QWidget for Image Viewer"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent', None)
        image = kwargs.get('image', test_image())

        QWidget.__init__(self, parent)

        ctab = ct.color_table_interpolated()

        self.wimg = FWViewImage(self, image, coltab=ctab, origin='UL', scale_ctl='HV')

        r = self.wimg.sceneRect()
        rscx = QRectF(r.x(), 0, r.width(), 1)
        rscy = QRectF(0, r.y(), 1, r.height())

        self.waxx = FWViewAxis(None, rscx, side='U', origin='UL', scale_ctl=True, wwidth=30, wlength=200)
        self.waxy = FWViewAxis(None, rscy, side='R', origin='UR', scale_ctl=True, wwidth=60, wlength=200)

        self.but_reset = QPushButton('Reset')

        self.box = QGridLayout()
        self.box.setSpacing(0)
        self.box.setVerticalSpacing(0)
        self.box.setHorizontalSpacing(0)
        self.box.addWidget(self.waxy,      0,  0, 20,  1)
        self.box.addWidget(self.wimg,      0,  1, 20, 20)
        self.box.addWidget(self.waxx,      20, 1,  1, 20)
        self.box.addWidget(self.but_reset, 20, 0, alignment=Qt.AlignCenter)
        self.setLayout(self.box)
 
        self.set_tool_tips()
        self.set_style()

        self.connect_scene_rect_changed()
        self.but_reset.clicked.connect(self.on_but_reset)


    def connect_scene_rect_changed(self):
        self.wimg.connect_scene_rect_changed_to(self.on_wimg_scene_rect_changed)
        self.waxx.connect_scene_rect_changed_to(self.on_waxx_scene_rect_changed)
        self.waxy.connect_scene_rect_changed_to(self.on_waxy_scene_rect_changed)


    def disconnect_scene_rect_changed(self):
        self.wimg.disconnect_scene_rect_changed_from(self.on_wimg_scene_rect_changed)
        self.waxx.disconnect_scene_rect_changed_from(self.on_waxx_scene_rect_changed)
        self.waxy.disconnect_scene_rect_changed_from(self.on_waxy_scene_rect_changed)


    def on_but_reset(self):
        logger.debug('on_but_reset')
        if self.wimg is not None:
           self.wimg.reset_original_size()


    def on_wimg_scene_rect_changed(self, r):
        #logger.debug('on_wimg_scene_rect_changed: %s'%str(r))
        self.waxx.set_view(rs=QRectF(r.x(), 0, r.width(), 1))
        self.waxy.set_view(rs=QRectF(0, r.y(), 1, r.height()))


    def on_waxx_scene_rect_changed(self, r):
        #logger.debug('on_waxx_scene_rect_changed: %s'%str(r))
        rs = self.wimg.scene().sceneRect()
        self.wimg.set_view(rs=QRectF(r.x(), rs.y(), r.width(), rs.height()))


    def on_waxy_scene_rect_changed(self, r):
        #logger.debug('on_waxy_scene_rect_changed: %s'%str(r))
        rs = self.wimg.scene().sceneRect()
        self.wimg.set_view(rs=QRectF(rs.x(), r.y(), rs.width(), r.height()))


    def set_tool_tips(self):
        self.wimg.setToolTip('Image')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.but_reset.setFixedSize(60,30)


    def set_pixmap_from_arr(self, arr, set_def=True):
        """shortcat to image"""
        self.wimg.set_pixmap_from_arr(arr, set_def)


    def reset_original_size(self):
        """shortcat to image"""
        self.wimg.reset_original_size()


if __name__ == "__main__":
    import os
    import sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d : %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = IVImageAxes()
    w.setGeometry(100, 50, 800, 800)
    w.setWindowTitle('Image with two axes')
    w.show()
    app.exec_()
    del w
    del app

# EOF
