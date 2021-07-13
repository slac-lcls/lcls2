
"""
Class :py:class:`FWViewHist` is a widget with interactive axes
==============================================================

FWViewHist <- FWView <- QGraphicsView <- QWidget

Usage ::

    Create FWViewHist object within pyqt QApplication
    --------------------------------------------------
    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.FWViewHist import FWViewHist

    app = QApplication(sys.argv)
    rscene=QRectF(0, 0, 100, 100)
    w = FWViewHist(None, rscene, origin='UL', fgcolor='red', bgcolor='yellow')
    w = FWViewHist(None, rscene, origin='UL')
    w = FWViewHist(None, rscene, origin='DR', scale_ctl=True, wwidth=50, wlength=200)

    w.show()
    app.exec_()

    Connect/disconnecr recipient to signals
    ---------------------------------------

    Methods
    -------
    w.print_attributes()

    Internal methods
    -----------------
    w.reset_original_image_size()

    Re-defines methods
    ------------------
    w.update_my_scene() # FWView.update_my_scene() + draw hist
    w.set_style()       # sets FWView.set_style() + color, font, pen
    w.closeEvent()      # removes rulers, FWView.closeEvent()

See:
    - :class:`FWView`
    - :class:`FWViewImage`
    - :class:`FWViewHist`
    - :class:`FWViewColorBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-11-02 by Mikhail Dubrovin 
"""

from psana.graphqt.FWView import * # FWView, QtGui, QtCore, Qt
from psana.graphqt.FWHist import FWHist, test_histogram
from PyQt5.QtGui import QColor, QFont

from psana.pyalgos.generic.HBins import HBins, np
from psana.pyalgos.generic.NDArrUtils import info_ndarr

logger = logging.getLogger(__name__)


class FWViewHist(FWView):
    
    def __init__(self, parent=None, rscene=QRectF(0, 0, 10, 10), origin='DL', **kwargs):

        self.bgcolor_def = 'black'

        self.scale_ctl = kwargs.get('scale_ctl', 'H')
        self.orient    = kwargs.get('orient',    'H') # histogram orientation H or V
        self.zvalue    = kwargs.get('zvalue',   10)   # z value for visibility
        self.wlength   = kwargs.get('wlength', 400)
        self.wwidth    = kwargs.get('wwidth',   60)
        self.bgcolor   = kwargs.get('bgcolor', self.bgcolor_def)
        self.fgcolor   = kwargs.get('fgcolor', 'blue')
        self.hbins     = kwargs.get('hbins', test_histogram())
        signal_fast    = kwargs.get('signal_fast', True)
        #self.kwargs    = kwargs

        self.hist = None
        self.side = 'D'
        self.arr_old = None
        #scctl = ('H' if self.side in ('U','D') else 'V') if self.scale_ctl else ''
        #scctl = 'HV'
        FWView.__init__(self, parent, rscene, origin, scale_ctl=self.scale_ctl, signal_fast=signal_fast)

        self._name = self.__class__.__name__
        #self.set_style() # called in FWView
        self.update_my_scene(self.hbins)


    def print_attributes(self):
        print('scale_control: ', self.str_scale_control())
        print('origin       : ', self.origin())


    def set_style(self):
        FWView.set_style(self)
        color = QColor(self.fgcolor)
        self.colhi = QColor(color)
        self.penhi = QPen(color, 1, Qt.SolidLine)


    def update_my_scene(self, hbins=None):
        FWView.update_my_scene(self)
        if hbins is None: return

        if self.hist is not None:
           self.hist.remove()
           del self.hist
        view = self
        if self.bgcolor != self.bgcolor_def:
            s = self.scene()
            r = s.sceneRect()
            s.addRect(r, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(QColor(self.bgcolor)))
        self.hist = FWHist(view, hbins=hbins, color=self.colhi, brush=QBrush(), orient=self.orient, zvalue=self.zvalue)


    def reset_original_image_size(self):
         # def in FWView.py with overloaded update_my_scene()
         self.reset_original_size()

#    def reset_original_image_size(self):
#        self.set_view()
#        self.update_my_scene()
#        self.check_axes_limits_changed()
#    def mouseMoveEvent(self, e):
#        self.update_my_scene()
#        FWView.mouseMoveEvent(self, e)


    def mouseReleaseEvent(self, e):
        logger.debug('mouseReleaseEvent')
        FWView.update_my_scene(self)
        FWView.mouseReleaseEvent(self, e)

 
    def closeEvent(self, e):
        self.hist.remove()
        FWView.closeEvent(self, e)
        #print('FWViewHist.closeEvent')


    def set_histogram_from_arr(self, arr, nbins=1000, amin=None, amax=None, frmin=0.001, frmax=0.999, edgemode=0):
        #if np.array_equal(arr, self.arr_old): return
        if arr is self.arr_old: return
        self.arr_old = arr

        aravel = arr.ravel()
        vmin = amin if amin is not None else\
               aravel.min() if frmin==0 else\
               np.quantile(aravel, frmin, axis=0, interpolation='lower')
        vmax = amax if amax is not None else\
               aravel.max() if frmax==0 else\
               np.quantile(aravel, frmax, axis=0, interpolation='higher')

        hb = HBins((vmin,vmax), nbins=nbins)
        hb.set_bin_data_from_array(aravel, dtype=np.float64, edgemode=edgemode)
        hmin, hmax = 0, hb.bin_data_max()

        logger.debug('set_histogram_from_arr %s\n    vmin(%.5f%%):%.3f vmax(%.5f%%):%.3f hmin: %.3f hmax: %.3f'%\
                     (info_ndarr(aravel, 'arr.ravel'), frmin,vmin,frmax,vmax,hmin,hmax))
        hgap = 0.05*(hmax-hmin)
        rs = QRectF(hmin-hgap, hb.vmin(), hmax-hmin+2*hgap, hb.vmax()-hb.vmin())
        self.set_rect_scene(rs, set_def=True)
        self.update_my_scene(hbins=hb)
        self.hbins = hb


if __name__ == "__main__":

  import sys

  def test_guiview(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w = None

    rs = QRectF(0, 0, 100, 1000)

    if   tname ==  '0': w=FWViewHist(None, rs, origin='UL', scale_ctl='V', fgcolor='white', bgcolor='gray')
    elif tname ==  '1': w=FWViewHist(None, rs, origin='DL', scale_ctl='H', fgcolor='black', bgcolor='yellow')
    elif tname ==  '2': w=FWViewHist(None, rs, origin='DR')
    elif tname ==  '3': w=FWViewHist(None, rs, origin='UR')
    elif tname ==  '4': w=FWViewHist(None, rs, origin='DR', scale_ctl='V', fgcolor='yellow', bgcolor='gray', orient='V')
    elif tname ==  '5': w=FWViewHist(None, rs, origin='DR', scale_ctl='V', fgcolor='white', orient='V')
    else:
        print('test %s is not implemented' % tname)
        return

    w.print_attributes()

    #w.connect_axes_limits_changed_to(w.test_axes_limits_changed_reception)
    #w.disconnect_axes_limits_changed_from(w.test_axes_limits_changed_reception)
    w.show()
    app.exec_()

    del w
    del app


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_guiview(tname)
    sys.exit('End of Test %s' % tname)

# EOF
