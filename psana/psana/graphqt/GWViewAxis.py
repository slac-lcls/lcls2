
"""
Class :py:class:`GWViewAxis` is a widget with interactive axis
==============================================================

GWViewAxis <- GWViewExt <- GWView <- QGraphicsView <- QWidget

Usage ::
    from psana.graphqt.GWViewExt import *

See:
    - graphqt/examples/ex_GWViewExt.py
    - :class:`GWView`
    - :class:`GWViewExt`
    - :class:`GWViewImage`
    - :class:`GWViewAxis`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-12-12 by Mikhail Dubrovin
Adopted as FWViewAxis for LCLS2 on 2018-02-20
Refactored to GWViewAxis on 2022-08-18
"""

from psana.graphqt.GWViewExt import * # FWView, QtGui, QtCore, Qt
from psana.graphqt.FWRuler import FWRuler
from PyQt5.QtGui import QColor, QFont


class GWViewAxis(GWViewExt):

    def __init__(self, parent=None, rscene=QRectF(0, 0, 10, 10), origin='UL', side='U', **kwargs):

        self.bgcolor_def = 'black'
        self.scale_ctl = kwargs.get('scale_ctl', True)
        self.wlength   = kwargs.get('wlength',   400)
        self.wwidth    = kwargs.get('wwidth',    60)
        self.bgcolor   = kwargs.get('bgcolor',   self.bgcolor_def)
        self.fgcolor   = kwargs.get('fgcolor',  'yellow')
        signal_fast    = kwargs.get('signal_fast', True)

        self.side  = side.upper()
        self.ruler = None

        scctl = ('H' if self.side.upper() in ('U','D') else 'V') if self.scale_ctl else ''
        GWViewExt.__init__(self, parent, rscene, origin, scale_ctl=scctl, signal_fast=signal_fast)
        self.update_my_scene()


    def info_attributes(self):
        return 'scale_control: %s' % self.str_scale_control()\
           + '\norigin       : %s' % self.origin()\
           + '\nside         : %s' % self.side


    def set_axis_limits(self, vmin, vmax):
        print('GWViewAxis.set_axis_limits vmin: %.1f vmax: %.1f' % (vmin, vmax))
        r = self.scene_rect()
        if self.side in ('U','D'):
            r.setX(vmin)
            r.setWidth(vmax - vmin)
        else:
            r.setY(vmin)
            r.setHeight(vmax - vmin)
        self.fit_in_view(r)
        self.update_my_scene()


    def set_style(self):
        GWViewExt.set_style(self)

        color = QColor(self.fgcolor)
        self.colax = QColor(color)
        self.fonax = QFont('Courier', 10, QFont.Normal)
        self.penax = QPen(color, 1, Qt.SolidLine)
        #self.penax.setCosmetic(True)

        if self.side in ('U','D'):
            self.setMinimumSize(self.wlength, 2)
            self.setFixedHeight(self.wwidth)
        else:
            self.setMinimumSize(2, self.wlength)
            self.setFixedWidth(self.wwidth)


    def update_ruler(self):
        if self.bgcolor != self.bgcolor_def:
            s = self.scene()
            r = s.sceneRect()
            s.addRect(r, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(QColor(self.bgcolor)))
        if self.ruler is not None: self.ruler.remove()
        view = self
        self.ruler = FWRuler(view, side=self.side, color=self.colax, pen=self.penax, font=self.fonax)


    def update_my_scene(self):
        """Re-implementation of GWViewExt.update_my_scene.
           Auto-called when the scene rect is changed."""
        #logger.debug('GWViewAxis.update_my_scene')
        GWViewExt.update_my_scene(self)  # which is GWViewExt.set_cursor_type_rect(self)
        self.update_ruler()


    def mouseReleaseEvent(self, e):
        self.update_my_scene()
        GWViewExt.mouseReleaseEvent(self, e)


    def closeEvent(self, e):
        self.ruler.remove()
        GWViewExt.closeEvent(self, e)
        logging.debug('GWViewAxis.closeEvent')


#    def reset_original_image_size(self):
#         # def in GWViewExt.py with overloaded update_my_scene()
#         #self.reset_original_size()
#        self.reset_scene_rect()
#        self.update_ruler()
#        self.check_axes_limits_changed()


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
