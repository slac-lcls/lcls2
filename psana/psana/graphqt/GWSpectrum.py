
"""Class :py:class:`GWSpectrum` is a QWidget with histogram, two axes, and color bar
====================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/GWSpectrum.py

    from psana.graphqt.GWSpectrum import GWSpectrum
    w = GWSpectrum()

Created on 2021-06-22 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)

from psana.graphqt.GWViewHist import * # GWViewHist # , test_histogram
from psana.graphqt.GWViewAxis import GWViewAxis
from psana.graphqt.GWViewColorBar import GWViewColorBar
import psana.graphqt.ColorTable as ct
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QTextEdit
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from psana.pyalgos.generic.NDArrGenerators import np, test_image, random_standard

class GWSpectrum(QWidget):
    """QWidget for Image Viewer"""
    histogram_scene_rect_changed = pyqtSignal('QRectF')

    def __init__(self, **kwargs):

        parent= kwargs.get('parent', None)
        image = kwargs.get('image', test_image())
        ctab  = kwargs.get('ctab', ct.color_table_default())
        nbins = kwargs.get('nbins', 1000)
        amin  = kwargs.get('amin', None)
        amax  = kwargs.get('amax', None)
        frmin = kwargs.get('frmin', 0.00001)
        frmax = kwargs.get('frmax', 0.99999)
        edgemode = kwargs.get('edgemode', 0)
        signal_fast = kwargs.get('signal_fast', True)

        QWidget.__init__(self, parent)

        self.nreset = 0

        self.whi = GWViewHist(parent=self, rscene=None, origin='DR', scale_ctl='V',\
                              fgcolor='yellow', bgcolor='dark', orient='V', signal_fast=signal_fast, hbins=None)
        self.whi.set_histogram_from_arr(image, nbins, amin, amax, frmin, frmax, edgemode, update_hblimits=True)
        self.wcbar = GWViewColorBar(self, coltab=ctab, orient='V')

        self.rsh_old = None
        r = self.whi.scene_rect()
        logger.debug('self.whi.scene_rect: %s' % qu.info_rect_xywh(r))

        rscx = QRectF(r.x(), 0, r.width(), 1)
        rscy = QRectF(0, r.y(), 1, r.height())

        self.wax = GWViewAxis(None, rscx, side='U', origin='UR', scale_ctl=True, wwidth=30, wlength=200, signal_fast=signal_fast) #, label_rot=20)
        self.way = GWViewAxis(None, rscy, side='L', origin='DL', scale_ctl=True, wwidth=60, wlength=200, signal_fast=signal_fast) #, label_rot=-70)

        self.but_reset = QPushButton('Reset')
        self.edi_info = QTextEdit('Info')

        self.box = QGridLayout()
        self.box.setSpacing(0)
        self.box.setVerticalSpacing(0)
        self.box.setHorizontalSpacing(0)
        self.box.addWidget(self.edi_info,   0,  0,  1, 11)
        self.box.addWidget(self.way,        1, 10,  9,  1)
        self.box.addWidget(self.whi,        1,  0,  9,  9)
        self.box.addWidget(self.wax,       10,  0,  1,  9)
        self.box.addWidget(self.wcbar,      1,  9,  9,  1)
        self.box.addWidget(self.but_reset, 10,  9,  1,  2, alignment=Qt.AlignCenter)
        self.setLayout(self.box)

        self.set_tool_tips()
        self.set_style()

        self.connect_scene_rect_changed()
        self.but_reset.clicked.connect(self.on_but_reset)
        self.update_info_panel()

    def set_signal_fast(self, is_fast=True):
        self.whi.signal_fast = is_fast
        self.wax.signal_fast = is_fast
        self.way.signal_fast = is_fast

    def connect_scene_rect_changed(self):
        self.whi.connect_scene_rect_changed(self.on_whi_scene_rect_changed)
        self.wax.connect_scene_rect_changed(self.on_wax_scene_rect_changed)
        self.way.connect_scene_rect_changed(self.on_way_scene_rect_changed)

    def disconnect_scene_rect_changed(self):
        self.whi.disconnect_scene_rect_changed(self.on_whi_scene_rect_changed)
        self.wax.disconnect_scene_rect_changed(self.on_wax_scene_rect_changed)
        self.way.disconnect_scene_rect_changed(self.on_way_scene_rect_changed)

    def on_but_reset(self):
        self.nreset += 1
        self.whi.reset_scene_rect()
        r = self.whi.scene_rect()
        logger.debug('on_but_reset:%04d whi.scene_rect: %s' % (self.nreset, qu.info_rect_xywh(r)))
        self.on_whi_scene_rect_changed(r)
        #self.wax.reset_scene_rect()
        #self.way.reset_scene_rect()

    def on_whi_scene_rect_changed(self, r):
        logger.debug('on_whi_scene_rect_changed: %s' % qu.info_rect_xywh(r))
        self.wax.set_axis_limits(r.x(), r.x()+r.width())   # fit_in_view(QRectF(r.x(), 0, r.width(), 1))
        self.way.set_axis_limits(r.y(), r.y()+r.height())  # fit_in_view(QRectF(0, r.y(), 1, r.height()))
        self.update_info_panel()
        self.emit_signal_if_histogram_scene_rect_changed()

    def on_wax_scene_rect_changed(self, r):
        #logger.debug('on_wax_scene_rect_changed: %s' % qu.info_rect_xywh(r))
        rs = self.whi.scene_rect()
        self.whi.fit_in_view(QRectF(r.x(), rs.y(), r.width(), rs.height()))
        self.emit_signal_if_histogram_scene_rect_changed()

    def on_way_scene_rect_changed(self, r):
        #logger.debug('on_way_scene_rect_changed: %s' % qu.info_rect_xywh(r))
        rs = self.whi.scene_rect()  # scene().sceneRect()
        self.whi.fit_in_view(QRectF(rs.x(), r.y(), rs.width(), r.height()))
        self.emit_signal_if_histogram_scene_rect_changed()
        self.update_info_panel()

    def emit_signal_if_histogram_scene_rect_changed(self):
        """Checks if scene rect have changed and submits signal with new rect."""
        rs = self.whi.scene_rect()
        if rs != self.rsh_old:
            self.rsh_old = rs
            self.histogram_scene_rect_changed.emit(rs)

    def connect_histogram_scene_rect_changed(self, recip):
        self.histogram_scene_rect_changed.connect(recip)

    def disconnect_histogram_scene_rect_changed(self, recip):
        self.histogram_scene_rect_changed.disconnect(recip)

    def test_histogram_scene_rect_changed(self, r):
        print(sys._getframe().f_code.co_name + ' %s' % qu.info_rect_xywh(r), end='\r')

    def update_info_panel(self):
        """update text information in stat box."""
        hb = self.whi.hbins
        r = self.whi.scene_rect()  # scene().sceneRect()
        vmin, vmax = r.y(), r.y()+r.height()
        s = 'spectrum min: %d max: %d' % (vmin, vmax)
        if hb is not None:
          resp = hb.histogram_statistics(vmin, vmax)
          if resp is not None:
            mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w, ibeg, iend = resp
            s += '\nentries: %d  nbins: %d [%d:%d]' % (sum_w, iend-ibeg, ibeg, iend)\
              + u'\nmean: %.3f \u00B1 %.3f\nrms: %.3f \u00B1 %.3f' % (mean, err_mean, rms, err_rms)\
              + u'\n\u03B31 skew: %.3f  \u03B32 kurt: %.3f' % (skew, kurt)
        self.edi_info.setText(s)

    def set_tool_tips(self):
        self.whi.setToolTip('Spectral intensity\ndistribution')
        self.wax.setToolTip('Spectral\nintensity')
        self.way.setToolTip('Spectral\nvalue')
        self.edi_info.setToolTip('Spectral statistics')
        self.but_reset.setToolTip('Reset to default\nspectrum')
        self.wcbar.setToolTip('Color bar for color\nto intensity conversion\nclick on it to select another')

    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.wcbar.setFixedWidth(25)
        self.edi_info.setMaximumHeight(85)
        self.edi_info.setStyleSheet(self.whi.style_def)

    def set_spectrum_from_arr(self, arr, nbins=1000, amin=None, amax=None, frmin=0.00001, frmax=0.99999, edgemode=0, update_hblimits=True):
        """shotcut"""
        #logger.info('set_spectrum_from_arr')
        self.whi.set_histogram_from_arr(arr, nbins, amin, amax, frmin, frmax, edgemode, update_hblimits)
        self.update_info_panel()
        self.on_but_reset()

    def reset_original_size(self):
        """Shortcut to whi.reset_original_size."""
        logger.debug('reset_original_size')
        #self.whi.reset_original_size()
        self.on_but_reset()

    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)

if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
