
"""Class :py:class:`IVSpectrum` is a QWidget with histogram, two axes, and color bar
====================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVSpectrum.py

    from psana.graphqt.IVSpectrum import IVSpectrum
    w = IVSpectrum()

Created on 2021-06-22 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)

from psana.graphqt.FWViewHist import FWViewHist
from psana.graphqt.FWViewAxis import FWViewAxis
from psana.graphqt.FWViewColorBar import FWViewColorBar
import psana.graphqt.ColorTable as ct
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, QRectF

def test_image():
  import psana.pyalgos.generic.NDArrGenerators as ag
  return ag.random_standard((8,12), mu=0, sigma=10)


class IVSpectrum(QWidget):
    """QWidget for Image Viewer"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent', None)
        image = kwargs.get('image', test_image())

        QWidget.__init__(self, parent)

        ctab = ct.color_table_interpolated()

        rs=QRectF(0, 0, 100, 1000)
        self.whis = FWViewHist(self, rs, origin='DR', scale_ctl='V', fgcolor='yellow', bgcolor='dark', orient='V')
        self.wcbar = FWViewColorBar(self, coltab=ctab, orient='V')

        r = self.whis.sceneRect()
        rscx = QRectF(r.x(), 0, r.width(), 1)
        rscy = QRectF(0, r.y(), 1, r.height())

        self.wax = FWViewAxis(None, rscx, side='U', origin='UR', scale_ctl=True, wwidth=30, wlength=200)
        self.way = FWViewAxis(None, rscy, side='L', origin='DL', scale_ctl=True, wwidth=60, wlength=200)

        self.but_reset = QPushButton('Reset')
        self.edi_info = QTextEdit('Info')

        self.box = QGridLayout()
        self.box.setSpacing(0)
        self.box.setVerticalSpacing(0)
        self.box.setHorizontalSpacing(0)
        self.box.addWidget(self.edi_info,   0,  0,  1, 11)
        self.box.addWidget(self.way,        1, 10,  9,  1)
        self.box.addWidget(self.whis,       1,  0,  9, 10)
        self.box.addWidget(self.wax,       10,  0,  1,  9)
        self.box.addWidget(self.wcbar,      1,  9,  9,  1)
        self.box.addWidget(self.but_reset, 10,  9,  1,  2, alignment=Qt.AlignCenter)
        self.setLayout(self.box)
 
        self.set_tool_tips()
        self.set_style()

        self.connect_scene_rect_changed()
        self.but_reset.clicked.connect(self.on_but_reset)


    def connect_scene_rect_changed(self):
        self.whis.connect_scene_rect_changed_to(self.on_whis_scene_rect_changed)
        self.wax.connect_scene_rect_changed_to(self.on_wax_scene_rect_changed)
        self.way.connect_scene_rect_changed_to(self.on_way_scene_rect_changed)


    def disconnect_scene_rect_changed(self):
        self.whis.disconnect_scene_rect_changed_from(self.on_whis_scene_rect_changed)
        self.wax.disconnect_scene_rect_changed_from(self.on_wax_scene_rect_changed)
        self.way.disconnect_scene_rect_changed_from(self.on_way_scene_rect_changed)


    def on_but_reset(self):
        logger.debug('on_but_reset')
        if self.whis is not None:
           self.whis.reset_original_size()


    def on_whis_scene_rect_changed(self, r):
        #logger.debug('on_whis_scene_rect_changed: %s'%str(r))
        self.wax.set_view(rs=QRectF(r.x(), 0, r.width(), 1))
        self.way.set_view(rs=QRectF(0, r.y(), 1, r.height()))
        self.update_info()


    def on_wax_scene_rect_changed(self, r):
        #logger.debug('on_wax_scene_rect_changed: %s'%str(r))
        rs = self.whis.scene().sceneRect()
        self.whis.set_view(rs=QRectF(r.x(), rs.y(), r.width(), rs.height()))


    def on_way_scene_rect_changed(self, r):
        #logger.debug('on_way_scene_rect_changed: %s'%str(r))
        rs = self.whis.scene().sceneRect()
        self.whis.set_view(rs=QRectF(rs.x(), r.y(), rs.width(), r.height()))
        self.update_info()


    def update_info(self):
        r = self.whis.scene().sceneRect()
        self.edi_info.setText('Spectrum min: %d max:  %d' % (r.y(), r.y()+r.height()))


    def set_tool_tips(self):
        self.whis.setToolTip('Spectrum')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        #self.but_reset.setFixedSize(60,30)
        self.wcbar.setFixedWidth(25)
        #self.edi_info.setFixedHeight(100)
        self.edi_info.setMaximumHeight(50)


    def set_pixmap_from_arr(self, arr, set_def=True):
        """shortcat to image"""
        self.whis.set_pixmap_from_arr(arr, set_def)


    def reset_original_size(self):
        """shortcat to image"""
        self.whis.reset_original_size()


if __name__ == "__main__":
    import os
    import sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    from PyQt5.QtWidgets import QApplication

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(name)s : %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = IVSpectrum()
    w.setGeometry(100, 50, 300, 800)
    w.setWindowTitle('Image with two axes')
    w.show()
    app.exec_()
    del w
    del app

# EOF
