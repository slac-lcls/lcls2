
"""
Class :py:class:`GWViewHist` is a widget with interactive axes
==============================================================

GWViewHist <- GWViewExt <- GWView <- QGraphicsView <- QWidget

Usage ::
    from psana.graphqt.GWViewExt import *

See:
    - graphqt/examples/ex_GWViewExt.py
    - :class:`GWView`
    - :class:`GWViewExt`
    - :class:`GWViewAxis`
    - :class:`GWViewImage`
    - :class:`GWViewHist`
    - :class:`GWViewColorBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-11-02 by Mikhail Dubrovin
Refactored to GWViewHist on 2022-08-30
"""

from psana.graphqt.GWViewExt import * # GWViewExt, QtGui, QtCore, Qt
from psana.graphqt.FWHist import FWHist, test_histogram
from PyQt5.QtGui import QColor, QFont

from psana.pyalgos.generic.HBins import HBins, np
from psana.pyalgos.generic.NDArrUtils import info_ndarr

logger = logging.getLogger(__name__)


class GWViewHist(GWViewExt):

    def __init__(self, parent=None, rscene=QRectF(0, 0, 10, 10), origin='DL', **kwargs):

        self.bgcolor_def = 'black'

        self.kwargs = kwargs
        #self.wlength   = kwargs.get('wlength', 400)
        #self.wwidth    = kwargs.get('wwidth',   60)
        self.fgcolor   = kwargs.get('fgcolor', 'blue')
        self.bgcolor   = kwargs.get('bgcolor', self.bgcolor_def)
        self.hbins     = kwargs.get('hbins', test_histogram())
        scale_ctl      = kwargs.get('scale_ctl', 'H')
        signal_fast    = kwargs.get('signal_fast', True)
        #self.kwargs    = kwargs

        self.hist = None
        #self.side = 'D'
        self.arr_old = None
        GWViewExt.__init__(self, parent, rscene, origin, scale_ctl=scale_ctl, signal_fast=signal_fast)

        #self.set_style() # called in GWViewExt
        self.update_my_scene(self.hbins)


    def print_attributes(self):
        logger.info(self.info_attributes())


    def info_attributes(self):
        return 'scale_control: %s' % self.str_scale_control()\
            +'\norigin       : %s' % self.origin()


    def set_style(self):
        GWViewExt.set_style(self)
        self.colhi = QColor(self.fgcolor)
        self.penhi = QPen(self.colhi, 1, Qt.SolidLine)


    def update_my_scene(self, hbins=None):
        GWViewExt.update_my_scene(self)
        if hbins is None: return

        if self.hist is not None:
           self.hist.remove()
           del self.hist
        view = self
        if self.bgcolor != self.bgcolor_def:
            s = self.scene()
            r = s.sceneRect()
            s.addRect(r, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(QColor(self.bgcolor)))

        orient = self.kwargs.get('orient', 'H')  # histogram orientation H or V
        zvalue = self.kwargs.get('zvalue', 10)  # z value for visibility

        self.hist = FWHist(view, hbins=hbins, color=self.colhi, brush=QBrush(), orient=orient, zvalue=zvalue)


    def reset_original_image_size(self):
         # def in GWViewExt.py with overloaded update_my_scene()
         self.reset_original_size()

#    def reset_original_image_size(self):
#        self.set_view()
#        self.update_my_scene()
#        self.check_axes_limits_changed()

    def mouseReleaseEvent(self, e):
        logger.debug('GWViewHist.mouseReleaseEvent')
        GWViewExt.update_my_scene(self)
        GWViewExt.mouseReleaseEvent(self, e)


    def closeEvent(self, e):
        self.hist.remove()
        GWViewExt.closeEvent(self, e)
        #logger.debug('GWViewHist.closeEvent')


    def set_histogram_from_arr(self, arr, nbins=1000, amin=None, amax=None, frmin=0.001, frmax=0.999, edgemode=0, update_hblimits=True):
        #if np.array_equal(arr, self.arr_old): return
        if arr is self.arr_old: return
        self.arr_old = arr
        if arr.size<1: return

        aravel = arr.ravel()

        vmin, vmax = self.hbins.limits() if self.hbins is not None else (None, None)

        if self.hbins is None or update_hblimits:
          vmin = amin if amin is not None else\
               aravel.min() if frmin in (0,None) else\
               np.quantile(aravel, frmin, axis=0, interpolation='lower')
          vmax = amax if amax is not None else\
               aravel.max() if frmax in (1,None) else\
               np.quantile(aravel, frmax, axis=0, interpolation='higher')
          if not vmin<vmax: vmax=vmin+1

        hb = HBins((vmin,vmax), nbins=nbins)
        hb.set_bin_data_from_array(aravel, dtype=np.float64, edgemode=edgemode)

        hmin, hmax = 0, hb.bin_data_max()
        #logger.debug('set_histogram_from_arr %s\n    vmin(%.5f%%):%.3f vmax(%.5f%%):%.3f hmin: %.3f hmax: %.3f'%\
        #             (info_ndarr(aravel, 'arr.ravel'), frmin,vmin,frmax,vmax,hmin,hmax))
        hgap = 0.05*(hmax-hmin)

        rs0 = self.scene().sceneRect()
        rsy, rsh = (hb.vmin(), hb.vmax()-hb.vmin()) if update_hblimits else (rs0.y(), rs0.height())
        rs = QRectF(hmin-hgap, rsy, hmax-hmin+2*hgap, rsh)
        self.set_rect_scene(rs, set_def=update_hblimits)

        self.update_my_scene(hbins=hb)
        self.hbins = hb


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
