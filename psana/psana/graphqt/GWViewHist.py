
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
from psana.graphqt.GWHist import GWHist, test_histogram
from PyQt5.QtGui import QColor, QFont

from psana.pyalgos.generic.HBins import HBins, np
from psana.pyalgos.generic.NDArrUtils import info_ndarr

logger = logging.getLogger(__name__)


class GWViewHist(GWViewExt):

    def __init__(self, parent=None, rscene=QRectF(0, 0, 10, 10), origin='DL', **kwargs):

        self.bgcolor_def = 'black'

        self.kwargs = kwargs
        self.fgcolor   = kwargs.get('fgcolor', 'blue')
        self.bgcolor   = kwargs.get('bgcolor', self.bgcolor_def)
        self.hbins     = kwargs.get('hbins', test_histogram())
        self.auto_limits = kwargs.get('auto_limits', '')
        scale_ctl      = kwargs.get('scale_ctl', 'H')
        signal_fast    = kwargs.get('signal_fast', True)
        #self.wlength   = kwargs.get('wlength', 400)
        #self.wwidth    = kwargs.get('wwidth',   60)

        self.hist = None
        self.item_rs_bg = None
        #self.side = 'D'
        self.arr_old = None
        GWViewExt.__init__(self, parent, rscene, origin, scale_ctl=scale_ctl, move_fast=signal_fast, wheel_fast=signal_fast)

        #self.set_style() # called in GWViewExt
        self.update_my_scene(self.hbins)
        #self.rs_old = rscene


    def print_attributes(self):
        logger.info(self.info_attributes())


    def info_attributes(self):
        return 'scale_control: %s' % self.str_scale_control()\
            +'\norigin       : %s' % self.origin()


    def set_style(self):
        GWViewExt.set_style(self)
        self.colhi = QColor(self.fgcolor)
        self.penhi = QPen(self.colhi, 1, Qt.SolidLine)


    def set_auto_limits(self, frac=0.05):
        #qu.info_rect_xywh(r)
        print('\nTBD IN set_auto_limits - auto_limits: %s orientation: %s scale_ctl: %s'%\
               (self.auto_limits, self.hist.orient, self.str_scale_ctl))  # , end='\n')
        r = self.scene_rect()
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        hb = self.hbins
        orient = self.hist.orient  # HV
        amin, amax = hb.limits()
        print('XXXX hb.limits: %.3f %.3f' % (amin, amax))
        arr = hb.bin_data()
        print('XXXX hb.data min/max: %.3f / %.3f' % (arr.min(), arr.max()))
        #lim_lo = arr.min()
        #lim_hi = arr.max()
        lim_lo = np.quantile(arr, frac, axis=0)
        lim_hi = np.quantile(arr, 1-frac, axis=0)
        gap = lim_hi - lim_lo
        lim_hi += gap * 0.5
        lim_lo -= gap * 0.1
        print('XXXX limits low/hight: %.3f / %.3f' % (lim_lo, lim_hi))

        rpars = ((amin, y, amax-amin, h) if self.auto_limits == 'H' else\
                 (x, lim_lo, w, lim_hi-lim_lo) if self.auto_limits == 'V' else\
                 (amin, lim_lo, amax-amin, lim_hi-lim_lo))\
              if orient == 'H' else\
                ((x, amin, w, amax-amin) if self.auto_limits == 'V' else\
                 (lim_lo, y, lim_hi-lim_lo, h) if self.auto_limits == 'H' else\
                 (lim_lo, amin, lim_hi-lim_lo, amax-amin))

        self.set_scene_rect(QRectF(*rpars))


    def update_my_scene(self, hbins=None):
        GWViewExt.update_my_scene(self)
        if hbins is None: return

        if self.hist is not None:
           self.hist.remove()
           del self.hist
        view = self
        if self.bgcolor != self.bgcolor_def:
            s = self.scene()
            if self.item_rs_bg:
               s.removeItem(self.item_rs_bg)
            r = s.sceneRect()
            self.item_rs_bg = s.addRect(r, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(QColor(self.bgcolor)))

        orient = self.kwargs.get('orient', 'H')  # histogram orientation H or V
        zvalue = self.kwargs.get('zvalue', 10)  # z value for visibility

        self.hist = GWHist(view, hbins=hbins, color=self.colhi, brush=QBrush(), orient=orient, zvalue=zvalue)
        self.hbins = hbins

        if self.auto_limits:
            self.set_auto_limits()

        self.reset_scene_rect_default()


    def mouseReleaseEvent(self, e):
        logger.debug('GWViewHist.mouseReleaseEvent')
        GWViewExt.update_my_scene(self)
        GWViewExt.mouseReleaseEvent(self, e)


    def closeEvent(self, e):
        self.hist.remove()
        GWViewExt.closeEvent(self, e)
        #logger.debug('GWViewHist.closeEvent')


    def set_histogram_from_arr(self, arr, nbins=1000, amin=None, amax=None,\
                               frmin=0.00001, frmax=0.99999, edgemode=0, update_hblimits=True):
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

        _nbins = int(vmax) - int(vmin)
        if _nbins < nbins: _nbins = nbins
        #arrsize = arr.size
        #print('XXX arrsize: %d vmin: %.3f vmax: %.3f nbins:%d' % (arrsize, vmin, vmax, _nbins))

        hb = HBins((vmin,vmax), nbins=_nbins)
        hb.set_bin_data_from_array(aravel, dtype=np.float64, edgemode=edgemode)

        hmin, hmax = 0, hb.bin_data_max()
        #logger.debug('set_histogram_from_arr %s\n    vmin(%.5f%%):%.3f vmax(%.5f%%):%.3f hmin: %.3f hmax: %.3f'%\
        #             (info_ndarr(aravel, 'arr.ravel'), frmin,vmin,frmax,vmax,hmin,hmax))
        hgap = 0.05*(hmax-hmin)

        #rs0 = self.scene().sceneRect()
        rs0 = self.scene_rect()
        rsy, rsh = (hb.vmin(), hb.vmax()-hb.vmin()) if update_hblimits else (rs0.y(), rs0.height())
        rs = QRectF(hmin-hgap, rsy, hmax-hmin+2*hgap, rsh)
        self.set_scene_rect(rs)

        self.update_my_scene(hbins=hb)
        #self.hbins = hb


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
