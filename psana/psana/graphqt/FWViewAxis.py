"""
Class :py:class:`FWViewAxis` is a widget with interactive axes
==============================================================

FWViewAxis <- FWView <- QGraphicsView <- QWidget

Usage ::

    Create FWViewAxis object within pyqt QApplication
    --------------------------------------------------
    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.FWViewAxis import FWViewAxis

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

from FWView import * # FWView, QtGui, QtCore, Qt
from FWRuler import FWRuler
from PyQt5.QtGui import QColor, QFont

#----

class FWViewAxis(FWView):
    
    def __init__(self, parent=None, rscene=QRectF(0, 0, 10, 10), origin='UL', side='U', **kwargs):

        self.scale_ctl = kwargs.get('scale_ctl', True)
        self.wlength   = kwargs.get('wlength',   400)
        self.wwidth    = kwargs.get('wwidth',    60)
        self.bgcolor   = kwargs.get('bgcolor',   None)
        self.fgcolor   = kwargs.get('fgcolor',  'black')

        self.side  = side.upper()
        self.ruler = None

        scctl = ('H' if self.side in ('U','D') else 'V') if self.scale_ctl else ''
        #scctl = 'HV'
        FWView.__init__(self, parent, rscene, origin, scale_ctl=scctl)

        self._name = self.__class__.__name__
        #self.set_style() # called in FWView
        self.update_my_scene()


    def print_attributes(self):
        print('scale_control: ', self.str_scale_control())
        print('origin       : ', self.origin())
        print('side         : ', self.side)


    def set_style(self):
        FWView.set_style(self)

        #style_default = "background-color: rgb(239, 235, 231, 255); color: rgb(0, 0, 0);" # Gray bkgd 
        #bgcolor = self.palette().color(QPalette.Background)
        style_default = '' if self.bgcolor is None else 'background-color: %s' % self.bgcolor
        self.setStyleSheet(style_default)

        #color = Qt.white
        color = QColor(self.fgcolor)
        self.colax = QColor(color)
        self.fonax = QFont('Courier', 12, QFont.Normal)
        self.penax = QPen(color, 1, Qt.SolidLine)

        if self.side in ('U','D') :
            self.setMinimumSize(self.wlength, 2)
            self.setFixedHeight(self.wwidth)
        else:
            self.setMinimumSize(2, self.wlength)
            self.setFixedWidth(self.wwidth)

        #self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)


    def update_my_scene(self):
        FWView.update_my_scene(self)
        #sc = self.scene()
        #rs = sc.sceneRect()
        #ra = rs
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


#    def mouseMoveEvent(self, e):
#        self.update_my_scene()
#        FWView.mouseMoveEvent(self, e)


    def mouseReleaseEvent(self, e):
        self.update_my_scene()
        FWView.mouseReleaseEvent(self, e)

 
    def closeEvent(self, e):
        self.ruler.remove()
        FWView.closeEvent(self, e)
        #print('FWViewAxis.closeEvent')

#----

if __name__ == "__main__":

  import sys

  def test_guiview(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    app = QApplication(sys.argv)
    w = None
    rs=QRectF(0, 0, 1000, 10)
    if   tname ==  '0': w=FWViewAxis(None, rs, side='D', origin='UL', fgcolor='red', bgcolor='yellow')
    elif tname ==  '1': w=FWViewAxis(None, rs, side='U', origin='UL')
    elif tname ==  '2': w=FWViewAxis(None, rs, side='L', origin='UL')
    elif tname ==  '3': w=FWViewAxis(None, rs, side='R', origin='UL')

    elif tname == '10': w=FWViewAxis(None, rs, side='D', origin='DL')
    elif tname == '11': w=FWViewAxis(None, rs, side='U', origin='DL')
    elif tname == '12': w=FWViewAxis(None, rs, side='L', origin='DL')
    elif tname == '13': w=FWViewAxis(None, rs, side='R', origin='DL')

    elif tname == '20': w=FWViewAxis(None, rs, side='D', origin='DR')
    elif tname == '21': w=FWViewAxis(None, rs, side='U', origin='DR')
    elif tname == '22': w=FWViewAxis(None, rs, side='L', origin='DR')
    elif tname == '23': w=FWViewAxis(None, rs, side='R', origin='DR')

    elif tname == '30': w=FWViewAxis(None, rs, side='D', origin='UR')
    elif tname == '31': w=FWViewAxis(None, rs, side='U', origin='UR')
    elif tname == '32': w=FWViewAxis(None, rs, side='L', origin='UR')
    elif tname == '33': w=FWViewAxis(None, rs, side='R', origin='UR')
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

#----

if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_guiview(tname)
    sys.exit('End of Test %s' % tname)

#----
