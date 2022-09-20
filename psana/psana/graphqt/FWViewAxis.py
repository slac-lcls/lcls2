
"""
Class :py:class:`FWViewAxis` is a widget with interactive axes
==============================================================

FWViewAxis <- FWView <- QGraphicsView <- QWidget

Usage ::

    Create FWViewAxis object within pyqt QApplication
    --------------------------------------------------
    import sys
    from PyQt4 import QtGui, QtCore
    from psana.graphqt.FWViewAxis import FWViewAxis

    app = QApplication(sys.argv)
    rscene=QRectF(0, 0, 100, 100)
    w = FWViewAxis(None, rscene, side='D', origin='UL', fgcolor='red', bgcolor='yellow')
    w = FWViewAxis(None, rscene, side='U', origin='UL')
    w = FWViewAxis(None, rscene, side='L', origin='DR', scale_ctl=True, wwidth=50, wlength=200)

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
    w.update_my_scene() # FWView.update_my_scene() + draw rulers
    w.set_style()       # sets FWView.set_style() + color, font, pen
    w.closeEvent()      # removes rulers, FWView.closeEvent()

See:
    - :class:`FWView`
    - :class:`FWViewImage`
    - :class:`FWViewAxis`
    - :class:`FWViewColorBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-12-12 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-20
"""

from psana.graphqt.FWView import * # FWView, QtGui, QtCore, Qt
from psana.graphqt.FWRuler import FWRuler
from PyQt5.QtGui import QColor, QFont


class FWViewAxis(FWView):

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

        scctl = ('H' if self.side in ('U','D') else 'V') if self.scale_ctl else ''
        #scctl = 'HV'
        FWView.__init__(self, parent, rscene, origin, scale_ctl=scctl, signal_fast=signal_fast)

        self._name = self.__class__.__name__
        #self.set_style() # called in FWView
        self.update_my_scene()


    def print_attributes(self):
        print('scale_control: ', self.str_scale_control())
        print('origin       : ', self.origin())
        print('side         : ', self.side)


    def set_style(self):
        FWView.set_style(self)

        color = QColor(self.fgcolor)
        self.colax = QColor(color)
        self.fonax = QFont('Courier', 10, QFont.Normal)
        self.penax = QPen(color, 1, Qt.SolidLine)

        if self.side in ('U','D') :
            self.setMinimumSize(self.wlength, 2)
            self.setFixedHeight(self.wwidth)
        else:
            self.setMinimumSize(2, self.wlength)
            self.setFixedWidth(self.wwidth)


    def update_my_scene(self):
        FWView.update_my_scene(self)
        if self.bgcolor != self.bgcolor_def:
            s = self.scene()
            r = s.sceneRect()
            s.addRect(r, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(QColor(self.bgcolor)))
        if self.ruler is not None: self.ruler.remove()
        view = self
        self.ruler = FWRuler(view, side=self.side, color=self.colax, pen=self.penax, font=self.fonax)


    def reset_original_image_size(self):
         # def in FWView.py with overloaded update_my_scene()
         self.reset_original_size()


#    def reset_original_image_size(self):
#        self.set_view()
#        self.update_my_scene()
#        self.check_axes_limits_changed()


    def mouseReleaseEvent(self, e):
        self.update_my_scene()
        FWView.mouseReleaseEvent(self, e)

    def closeEvent(self, e):
        self.ruler.remove()
        FWView.closeEvent(self, e)
        #print('FWViewAxis.closeEvent')


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
