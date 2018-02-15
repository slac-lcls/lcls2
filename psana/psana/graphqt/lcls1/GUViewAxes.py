#!@PYTHON@
"""
Class :py:class:`GUViewAxes` is a QGraphicsView / QWidget with interactive scalable scene with axes
===================================================================================================

Usage ::

    Create GUViewAxes object within pyqt QApplication
    --------------------------------------------------
    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.GUViewAxes import GUViewAxes

    app = QtGui.QApplication(sys.argv)
    w = GUViewAxes(None, raxes=QtCore.QRectF(0, 0, 100, 100), origin='UL',\
                   scale_ctl='HV', rulers='TR', margl=0.02, margr=0.12, margt=0.06, margb=0.02)
    w.show()
    app.exec_()

    Connect/disconnecr recipient to signals
    ---------------------------------------

    Methods
    -------
    w.set_show_rulers(rulers='TBLR')
    w.reset_original_image_size()

    Internal methods
    -----------------

    Re-defines methods
    ------------------
    w.update_my_scene() # GUView.update_my_scene() + draw rulers
    w.set_style()       # sets GUView.set_style() + color, font, pen
    w.closeEvent()      # removes rulers, GUView.closeEvent()

Created on December 14, 2016 by Mikhail Dubrovin
"""

from graphqt.GUView  import GUView, QtGui, QtCore, Qt
from graphqt.GURuler import GURuler
#from graphqt.GUUtils import print_rect

class GUViewAxes(GUView) :
    
    def __init__(self, parent=None, rectax=QtCore.QRectF(0, 0, 10, 10), origin='UL',\
                 scale_ctl='HV', rulers='TBLR', margl=None, margr=None, margt=None, margb=None) :

        self.set_show_rulers(rulers)

        GUView.__init__(self, parent, rectax, origin, scale_ctl, margl, margr, margt, margb)

        self._name = self.__class__.__name__

        self.rulerl = None
        self.rulerb = None
        self.rulerr = None
        self.rulert = None

        self.update_my_scene()


    def set_show_rulers(self, rulers='TBLR') :
        key = rulers.upper()
        self._do_left   = 'L' in key
        self._do_right  = 'R' in key
        self._do_top    = 'T' in key or 'U' in key 
        self._do_bottom = 'B' in key or 'D' in key


    def set_style(self) :
        GUView.set_style(self)
        self.colax = QtGui.QColor(Qt.white)
        self.fonax = QtGui.QFont('Courier', 12, QtGui.QFont.Normal)
        self.penax = QtGui.QPen(Qt.white, 1, Qt.SolidLine)


    def update_my_scene(self) :
        GUView.update_my_scene(self)

        sc = self.scene()
        rs = sc.sceneRect()
        ra = self.rectax #  self.raxes

        #print_rect(ra, cmt='YYY GUViewAxes axes rect')

        if self._origin_ul :
            if self._do_left   : self.rulerl = GURuler(sc, GURuler.VL, rect=ra, color=self.colax, pen=self.penax, font=self.fonax)
            if self._do_bottom : self.rulerb = GURuler(sc, GURuler.HD, rect=ra, color=self.colax, pen=self.penax, font=self.fonax)
            if self._do_right  : self.rulerr = GURuler(sc, GURuler.VR, rect=ra, color=self.colax, pen=self.penax, font=self.fonax)
            if self._do_top    : self.rulert = GURuler(sc, GURuler.HU, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.04)

        elif self._origin_dl :
            if self._do_left   : self.rulerl = GURuler(sc, GURuler.VL, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.055)
            if self._do_bottom : self.rulerb = GURuler(sc, GURuler.HU, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.09)
            if self._do_right  : self.rulerr = GURuler(sc, GURuler.VR, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.055)
            if self._do_top    : self.rulert = GURuler(sc, GURuler.HD, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.045)

        elif self._origin_dr :
            if self._do_left   : self.rulerl = GURuler(sc, GURuler.VR, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.055, txtoff_hfr=0.055)
            if self._do_bottom : self.rulerb = GURuler(sc, GURuler.HU, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.09,  txtoff_hfr=0.05)
            if self._do_right  : self.rulerr = GURuler(sc, GURuler.VL, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.055, txtoff_hfr=0.05)
            if self._do_top    : self.rulert = GURuler(sc, GURuler.HD, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.05,  txtoff_hfr=0.03)

        elif self._origin_ur :
            #print 'UR'
            if self._do_left   : self.rulerl = GURuler(sc, GURuler.VR, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_hfr=0.07)
            if self._do_bottom : self.rulerb = GURuler(sc, GURuler.HD, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_hfr=0.05)
            if self._do_right  : self.rulerr = GURuler(sc, GURuler.VL, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_hfr=0.06)
            if self._do_top    : self.rulert = GURuler(sc, GURuler.HU, rect=ra, color=self.colax, pen=self.penax, font=self.fonax, txtoff_hfr=0.05, txtoff_vfr=0.04)

#------------------------------

    def reset_original_image_size(self) :
         # def in GUView.py with overloaded update_my_scene()
         self.reset_original_size()

#    def reset_original_image_size(self) :
#        self.set_view()
#        self.update_my_scene()
#        self.check_axes_limits_changed()

#------------------------------
 
    def closeEvent(self, e):
        self.rulerl = None
        self.rulerb = None
        self.rulerr = None
        self.rulert = None
        GUView.closeEvent(self, e)
        #print 'GUViewAxes.closeEvent'

#-----------------------------

def test_guiview(tname) :
    print '%s:' % sys._getframe().f_code.co_name
    app = QtGui.QApplication(sys.argv)
    w = None
    rectax=QtCore.QRectF(0, 0, 100, 100)
    if   tname == '0': w=GUViewAxes(None, rectax, origin='DL', scale_ctl='HV', margl=0.12, margr=0.10, margt=0.06, margb=0.06)
    elif tname == '1': w=GUViewAxes(None, rectax, origin='DL', scale_ctl='')
    elif tname == '2': w=GUViewAxes(None, rectax, origin='DL', scale_ctl='H')
    elif tname == '3': w=GUViewAxes(None, rectax, origin='DL', scale_ctl='V')
    elif tname == '4': w=GUViewAxes(None, rectax, origin='UL', rulers='TRLB', margl=0.02, margr=0.12, margt=0.02, margb=0.06)
    elif tname == '5': w=GUViewAxes(None, rectax, origin='DL', rulers='TRLB', margl=0.02, margr=0.12, margt=0.06, margb=0.02)
    elif tname == '6': w=GUViewAxes(None, rectax, origin='DR', rulers='TRLB', margl=0.12, margr=0.12, margt=0.06, margb=0.06)
    elif tname == '7': w=GUViewAxes(None, rectax, origin='UR', rulers='TRLB', margl=0.12, margr=0.12, margt=0.06, margb=0.06)
    else :
        print 'test %s is not implemented' % tname
        return

    w.connect_axes_limits_changed_to(w.test_axes_limits_changed_reception)
    #w.disconnect_axes_limits_changed_from(w.test_axes_limits_changed_reception)
    w.show()
    app.exec_()

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s' % tname
    test_guiview(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
