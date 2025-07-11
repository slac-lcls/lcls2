#!@PYTHON@
"""
Class :py:class:`GUViewAxesDL` is a QGraphicsView / QWidget with interactive scalable scene with axes
=====================================================================================================

DL stands for Down and Lext axes only.

Usage ::

    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.GUViewAxesDL import GUViewAxesDL

    app = QtGui.QApplication(sys.argv)
    w = GUViewAxesDL(None, raxes=QtCore.QRectF(0, 0, 100, 100), origin_up=False, scale_ctl=3)
    w.show()
    app.exec_()

Created on September 9, 2016 by Mikhail Dubrovin
"""

#import os
from math import floor
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from graphqt.GURuler import GURuler

class GUViewAxesDL(QtGui.QGraphicsView) :
    
    def __init__(self, parent=None, rectax=QtCore.QRectF(0, 0, 10, 10), origin_up=True, scale_ctl=3, rulers='LB') :

        self.raxes = rectax
        self.origin_up = origin_up
        self.set_scale_control(scale_ctl)
        self.set_show_rulers(rulers)

        sc = QtGui.QGraphicsScene() # rectax
        #print 'scene rect=', sc.sceneRect()        
        #print 'rect img=', self.rectax

        QtGui.QGraphicsView.__init__(self, sc, parent)
        
        self.set_style()
        self.set_view() #ml=0.12, mr=0.02, mt=0.02, mb=0.06)

        colfld = Qt.magenta
        colori = Qt.red
        #pen=QtGui.QPen(colfld, 0, Qt.SolidLine)
        self.raxi = self.add_rect_to_scene_v1(self.raxes, pen=QtGui.QPen(Qt.NoPen), brush = QtGui.QBrush(colfld))

        ror=QtCore.QRectF(-2, -2, 4, 4)
        self.rori = self.add_rect_to_scene(ror, pen=QtGui.QPen(colori, 0, Qt.SolidLine), brush = QtGui.QBrush(colori))

        if not self.origin_up :
            t = self.transform()
            t2 = t.scale(1,-1)
            self.setTransform(t2)

        self.rslefv = None
        self.rsbotv = None
        self.rslefi = None
        self.rsboti = None
        self.raxesi = None
        self.rulerl = None
        self.rulerd = None
        self.pos_click = None
        self.scalebw = 3

        self.update_my_scene()


    def set_show_rulers(self, rulers='LRBT') :
        self._do_left   = 'L' in rulers
        self._do_right  = 'R' in rulers
        self._do_top    = 'T' in rulers
        self._do_bottom = 'B' in rulers
        

    def set_scale_control(self, scale_ctl=3) :
        """Sets scale control bit-word
           = 0 - x, y frozen scales
           + 1 - x is interactive
           + 2 - y is interactive
           bit value 0/1 frozen/interactive  
        """
        self._scale_ctl = scale_ctl        


    def scale_control(self) :
        return self._scale_ctl


    def set_style(self) :
        self.setGeometry(20, 20, 600, 600)
        self.setWindowTitle("GUViewAxesDL window")
        #w.setContentsMargins(-9,-9,-9,-9)
        self.setStyleSheet("background-color:black; border: 0px solid green")
        #self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        #self.setAttribute(Qt.WA_TranslucentBackground)
        #self.setInteractive(True)

        self.colax = QtGui.QColor(Qt.white)
        self.fonax = QtGui.QFont('Courier', 12, QtGui.QFont.Normal)

        self.brudf = QtGui.QBrush()
        self.brubx = QtGui.QBrush(Qt.black, Qt.SolidPattern)

        self.pendf = QtGui.QPen()
        self.pendf.setStyle(Qt.NoPen)
        self.penbx = QtGui.QPen(Qt.black, 6, Qt.SolidLine)
        self.penax = QtGui.QPen(Qt.white, 1, Qt.SolidLine)


    def set_view(self, ml=0.12, mr=0.02, mt=0.02, mb=0.06) :
        self.margl = ml
        self.margr = mr
        self.margt = mt
        self.margb = mb

        x, y = self.raxes.x(), self.raxes.y()
        sx = self.raxes.width() /(1. - ml - mr)
        sy = self.raxes.height()/(1. - mt - mb)

        # Set scene rect larger than axes rect
        dy = -sy*(mt if self.origin_up else mb)
        rs = QtCore.QRectF(x-ml*sx, y+dy, sx, sy)
        #print 'scene rect=', rs

        self.scene().setSceneRect(rs)
        self.fitInView(rs, Qt.IgnoreAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio


    def update_my_scene(self) :
        sc = self.scene()
        rs = sc.sceneRect()

        #print 'XXX: Scene rect', rs
        #print 80*'_'
        #for item in sc.items() :
        #    print 'scene item:', item

        # set dark rects

        #if self.raxi   is not None : self.scene().removeItem(self.raxi)
        if self.rslefv is not None : self.scene().removeItem(self.rslefv)
        if self.rsbotv is not None : self.scene().removeItem(self.rsbotv)
        if self.rslefi is not None : self.scene().removeItem(self.rslefi)
        if self.rsboti is not None : self.scene().removeItem(self.rsboti)
        if self.raxesi is not None : self.scene().removeItem(self.raxesi)

        x, y, w, h = rs.x(), rs.y(), rs.width(), rs.height()

        rslef=QtCore.QRectF(x, y, w*self.margl, h)
        self.rslefv = self.add_rect_to_scene_v1(rslef, self.brubx, self.penbx)
        self.rslefi = self.add_rect_to_scene(rslef, self.brudf, self.pendf)
        self.rslefi.setCursorHover(Qt.SizeVerCursor)
        self.rslefi.setCursorGrab (Qt.SplitVCursor)

        rsbot=QtCore.QRectF(x+w*self.margl, y, w-w*self.margl, h*self.margb)

        if self.origin_up :
            rsbot.moveBottomRight(rs.bottomRight())
            self.rsbotv = self.add_rect_to_scene_v1(rsbot, self.brubx, self.penbx)
            self.rsboti = self.add_rect_to_scene(rsbot, self.brudf, self.pendf)

            self.rectax = QtCore.QRectF(rslef.topRight(), rsbot.topRight())
            if self._do_left   : self.rulerl = GURuler(sc, GURuler.VL, rect=self.rectax, color=self.colax, pen=self.penax, font=self.fonax)
            if self._do_bottom : self.rulerd = GURuler(sc, GURuler.HD, rect=self.rectax, color=self.colax, pen=self.penax, font=self.fonax)

        else :
            rsbot.moveTopRight(rs.topRight())
            self.rsbotv = self.add_rect_to_scene_v1(rsbot, self.brubx, self.penbx)
            self.rsboti = self.add_rect_to_scene(rsbot, self.brudf, self.pendf)

            self.rectax = QtCore.QRectF(rslef.bottomRight(), rsbot.bottomRight()).normalized()
            if self._do_left   : self.rulerl = GURuler(sc, GURuler.VL, rect=self.rectax, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.055)
            if self._do_bottom : self.rulerd = GURuler(sc, GURuler.HU, rect=self.rectax, color=self.colax, pen=self.penax, font=self.fonax, txtoff_vfr=0.09)
        
        self.raxesi = self.add_rect_to_scene(self.rectax, self.brudf, self.pendf)
        self.raxesi.setCursorHover(Qt.CrossCursor)
        self.raxesi.setCursorGrab (Qt.SizeAllCursor)
        self.rsboti.setCursorHover(Qt.SizeHorCursor)
        self.rsboti.setCursorGrab (Qt.SplitHCursor)

        self.rslefv.setZValue(1)
        self.rsbotv.setZValue(1)
        self.raxesi.setZValue(20)
        self.rslefi.setZValue(20)
        self.rsboti.setZValue(20)

        #self.updateScene([rsbot, rslef, self.rectax])


    def remove(self) :
        #remove ruler lines
        #self.scene.removeItem(self.path_item)
        #remove labels
        #for item in self.lst_txtitems :
        #    self.scene.removeItem(item)
        #self.textitems=[]
        pass


#    def update(self) :
#        print 'update signal is received'
#        self.update_my_scene()


    def __del__(self) :
        self.remove()


    def mouseReleaseEvent(self, e):
        QtGui.QApplication.restoreOverrideCursor()
        QtGui.QGraphicsView.mouseReleaseEvent(self, e)
        #print 'GUViewAxesDL.mouseReleaseEvent, at point: ', e.pos(), ' diff:', e.pos() - self.pos_click
        #self.pos_click = e.pos()
        self.pos_click = None


#    def mouseDoubleCkickEvent(self, e):
#        QtGui.QGraphicsView.mouseDoubleCkickEvent(self, e)
#        print 'mouseDoubleCkickEvent'


    def mousePressEvent(self, e):
        #print 'GUViewAxesDL.mousePressEvent, at point: ', e.pos() #e.globalX(), e.globalY() 
        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(Qt.SizeAllCursor))# ClosedHandCursor
        QtGui.QGraphicsView.mousePressEvent(self, e)

        self.pos_click = e.pos()
        #self.pos_click_sc = self.mapToScene(self.pos_click)
        self.rs_center = self.scene().sceneRect().center()
        self.invscalex = 1./self.transform().m11()
        self.invscaley = 1./self.transform().m22()

        self._select_further_action(e)

        
    def _select_further_action(self, e):
        if self._scale_ctl != 3 :
            self.scalebw = self._scale_ctl
            return

        pos_on_sc = self.mapToScene(e.pos())
        item = self.scene().itemAt(pos_on_sc)

        if   item == self.rsboti : self.scalebw = 1 # print 'bottom rect' # |= 1
        elif item == self.rslefi : self.scalebw = 2 # print 'left rect' # |= 2
        else                     : self.scalebw = 3
        #elif item == self.raxesi   : self.scalebw = 3 # print 'axes rect'
        #print '_select_further_action scalebw:', self.scalebw


    def display_pixel_pos(self, e):
        p = self.mapToScene(e.pos())
        #print 'mouseMoveEvent, current point: ', e.x(), e.y(), ' on scene: %.1f  %.1f' % (p.x(), p.y()) 
        self.setWindowTitle('GUViewAxesDL: x=%.1f y=%.1f' % (p.x(), p.y()))


    def mouseMoveEvent(self, e):
        QtGui.QGraphicsView.mouseMoveEvent(self, e)
        #print 'GUViewAxesDL.mouseMoveEvent, at point: ', e.pos()
        self.display_pixel_pos(e)

        if self._scale_ctl==0 : return

        if self.pos_click is None : return        

        dp = e.pos() - self.pos_click
        #print 'mouseMoveEvent, at point: ', e.pos(), ' diff:', dp
        
        dx = dp.x()*self.invscalex if self.scalebw & 1 else 0
        dy = dp.y()*self.invscaley if self.scalebw & 2 else 0
        dpsc = QtCore.QPointF(dx, dy)

        sc = self.scene()
        rs = sc.sceneRect()
        rs.moveCenter(self.rs_center - dpsc)
        sc.setSceneRect(rs)

        self.update_my_scene()


    def wheelEvent(self, e) :
        QtGui.QGraphicsView.wheelEvent(self, e)

        if self._scale_ctl==0 : return

        self._select_further_action(e)

        #print 'wheelEvent: ', e.delta()
        f = 1 + 0.4 * (1 if e.delta()>0 else -1)
        #print 'Scale factor =', f

        p = self.mapToScene(e.pos())
        px, py = p.x(), p.y() 
        #print 'wheel x,y = ', e.x(), e.y(), ' on scene x,y =', p.x(), p.y() 
        #rectax = self.rectax

        sc = self.scene()
        rs = sc.sceneRect()
        x,y,w,h = rs.x(), rs.y(), rs.width(), rs.height()
        #print 'Scene x,y,w,h:', x,y,w,h

        # zoom relative to axes center
        #dxc = (f-1)*0.55*w 
        #dyc = (f-1)*0.45*h

        # zoom relative to mouse position
        dxc = (f-1)*(px-x)
        dyc = (f-1)*(py-y) 
        dx, sx = (dxc, f*w) if self.scalebw & 1 else (0, w)
        dy, sy = (dyc, f*h) if self.scalebw & 2 else (0, h)

        rs.setRect(x-dx, y-dy, sx, sy)      
        sc.setSceneRect(rs)

        #self.update_transform()
        #sc.update()
        #rs = self.scene().sceneRect()    
        self.fitInView(rs, Qt.IgnoreAspectRatio)
        self.update_my_scene()

        #self.scalebw = 3


    def enterEvent(self, e) :
    #    print 'enterEvent'
        QtGui.QGraphicsView.enterEvent(self, e)
        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(Qt.CrossCursor))
        

    def leaveEvent(self, e) :
    #    print 'leaveEvent'
        QtGui.QGraphicsView.leaveEvent(self, e)
        #QtGui.QApplication.restoreOverrideCursor()


    def closeEvent(self, e) :
        QtGui.QGraphicsView.closeEvent(self, e)
        #print 'closeEvent'
        

    #def moveEvent(self, e) :
    #    print 'moveEvent'
    #    print 'Geometry rect:', self.geometry()


    def resizeEvent(self, e) :
         QtGui.QGraphicsView.resizeEvent(self, e)
         #print 'resizeEvent'
         #print 'Geometry rect:', self.geometry()
         rs = self.scene().sceneRect()    
         #print 'Rect of the scene =', rs
         self.fitInView(rs, Qt.IgnoreAspectRatio)


    #def paintEvent(e):
    #    pass


    def keyPressEvent(self, e) :
        #print 'keyPressEvent, key=', e.key()         
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_R : 
            print 'Reset original size'
            self.set_view()
            self.update_my_scene()


    def add_rect_to_scene_v1(self, rect, brush=QtGui.QBrush(), pen=QtGui.QPen(Qt.yellow, 4, Qt.DashLine)) :
        pen.setCosmetic(True)
        return self.scene().addRect(rect, pen, brush)


    def add_rect_to_scene(self, rect, brush=QtGui.QBrush(), pen=QtGui.QPen(Qt.yellow, 4, Qt.DashLine)) :
        from graphqt.GUQGraphicsRectItem import GUQGraphicsRectItem
        #pen.setCosmetic(True)
        #return self.scene().addRect(rect, pen, brush)
        pen.setCosmetic(True)
        #item = QtGui.QGraphicsRectItem(rect, parent=None, scene=self.scene())
        item = GUQGraphicsRectItem(rect, parent=None, scene=self.scene())
        item.setPen(pen)
        item.setBrush(brush)
        #self.scene().addRect(item)
        #item.setAcceptsHoverEvents(True)

        return item

#-----------------------------

def test_guiview(tname) :
    print '%s:' % sys._getframe().f_code.co_name
    app = QtGui.QApplication(sys.argv)
    w = None
    if tname == '0': w=GUViewAxesDL(None, rectax=QtCore.QRectF(0, 0, 100, 100), origin_up=False, scale_ctl=3)
    if tname == '1': w=GUViewAxesDL(None, rectax=QtCore.QRectF(0, 0, 100, 100), origin_up=False, scale_ctl=0)
    if tname == '2': w=GUViewAxesDL(None, rectax=QtCore.QRectF(0, 0, 100, 100), origin_up=False, scale_ctl=1)
    if tname == '3': w=GUViewAxesDL(None, rectax=QtCore.QRectF(0, 0, 100, 100), origin_up=False, scale_ctl=2)
    if tname == '4': w=GUViewAxesDL(None, rectax=QtCore.QRectF(0, 0, 100, 100), origin_up=True,  scale_ctl=3)
    if tname == '5': w=GUViewAxesDL(None, rectax=QtCore.QRectF(0, 0, 100, 100), origin_up=False, scale_ctl=3, rulers='L')
    if tname == '6': w=GUViewAxesDL(None, rectax=QtCore.QRectF(0, 0, 100, 100), origin_up=False, scale_ctl=3, rulers='B')
    else : print 'test %s is not implemented' % tname
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

