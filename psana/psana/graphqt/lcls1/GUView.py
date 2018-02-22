#!@PYTHON@
"""
Class :py:class:`GUView` is a QGraphicsView / QWidget with interactive scene having scalable axes box
=====================================================================================================

Usage ::

    Create GUView object within pyqt QApplication
    ---------------------------------------------
    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.GUView import GUView
    app = QtGui.QApplication(sys.argv)
    w = GUView(None, raxes=QtCore.QRectF(0, 0, 100, 100), origin='UL',\
               scale_ctl='HV', margl=0.12, margr=0.10, margt=0.06, margb=0.06)
    w.show()
    app.exec_()

    Connect/disconnecr recipient to signals
    ----------------------------
    w.connect_axes_limits_changed_to(recipient)
    w.disconnect_axes_limits_changed_from(recipient)
    w.test_axes_limits_changed_reception(self, x1, x2, y1, y2)

    w.connect_wheel_is_stopped_to(recipient)
    w.disconnect_wheel_is_stopped_from(recipient)
    w.test_wheel_is_stopped_reception(self)

    w.connect_view_is_closed_to(recipient)
    w.disconnect_view_is_closed_from(recipient)
    w.test_view_is_closed_reception(self)

    Major methors
    -----------------
    w.reset_original_size()
    w.set_rect_axes(rectax, set_def=True)

    Methods
    -------
    rs = w.rect_scene_from_rect_axes(raxes=None)
    w.set_rect_scene(rs) # the same as zoom-in/out, do not change default

    w.set_rect_axes_default(rectax)
    w.set_origin(origin='UL')
    w.set_transform_orientation() # if origin is changed
    w.set_scale_control(scale_ctl='HV') # sets dynamically controlled axes
    sc = w.scale_control()
    w.set_margins(margl=None, margr=None, margt=None, margb=None)

    w.set_limits_vertical(yaxmin, yaxmax)
    w.check_limits()
    w.check_axes_limits_changed() # compare axes box with old and sends signal if changed
    w.update_my_scene()
    w.set_view()
    w.reset_original_size() # calls sequence of methods: set_view, update_my_scene, check_axes_limits_changed
    w.add_rect_to_scene_v1(rect, brush=QtGui.QBrush(), pen=QtGui.QPen(Qt.yellow, 4, Qt.DashLine)) # returns QGraphicsRectItem
    w.add_rect_to_scene(rect, brush=QtGui.QBrush(), pen=QtGui.QPen(Qt.yellow, 4, Qt.DashLine))  # returns GUQGraphicsRectItem
    r = w.rect_axes()
    xmin, xmax, ymin, ymax = w.axes_limits()

    Internal methods
    -----------------
    w._select_further_action(e) # checks cursor position relative to margin box items and sets w.scalebw
    w._continue_wheel_event(t_msec=500)
    w.on_timeout(self)


    Re-defines methods
    ------------------
    __del__
    enterEvent, leaveEvent, closeEvent, resizeEvent, keyPressEvent, 
    mouseReleaseEvent, mousePressEvent, mouseMoveEvent, wheelEvent

Created on September 9, 2016 by Mikhail Dubrovin
"""
#------------------------------

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
#from graphqt.GUUtils import print_rect

import pyimgalgos.NDArrGenerators as ag

#------------------------------

class GUView(QtGui.QGraphicsView) :
    
    def __init__(self, parent=None, rectax=QtCore.QRectF(0, 0, 10, 10), origin='UL', scale_ctl='HV',\
                 margl=None, margr=None, margt=None, margb=None, show_mode=0) :

        # set initial parameters, no graphics yet
        self._name = self.__class__.__name__
        self.set_rect_axes_default(rectax) # sets self.rectax
        self.show_mode = show_mode
        self.set_origin(origin)
        self.set_scale_control(scale_ctl)
        self.set_margins(margl, margr, margt, margb)

        # begin graphics
        sc = QtGui.QGraphicsScene() # rectax
        #print 'scene rect=', sc.sceneRect()        
        #print 'rect img=', self.rectax

        QtGui.QGraphicsView.__init__(self, sc, parent)
        
        self.set_transform_orientation() 
        self.set_view()
        self.set_style()

        self.rslefv = None
        self.rsbotv = None
        self.rsrigv = None
        self.rstopv = None

        self.rslefi = None
        self.rsboti = None
        self.rsrigi = None
        self.rstopi = None

        self.raxesv = None
        self.raxesi = None
        self.pos_click = None
        self.scalebw = 3

        self.raxi = None
        self.rori = None

        self.update_my_scene()

        self.timer = QtCore.QTimer()
        self.connect(self.timer, QtCore.SIGNAL('timeout()'), self.on_timeout)

        #self.connect_wheel_is_stopped_to(self.check_axes_limits_changed)
        #self.disconnect_wheel_is_stopped_from(self.check_axes_limits_changed)
        
        #self.connect_axes_limits_changed_to(self.test_axes_limits_changed_reception)
        #self.disconnect_axes_limits_changed_from(self.test_axes_limits_changed_reception)


    def set_rect_axes_default(self, rectax) :
        #print 'XXX: In GUIView.set_rect_axes_default rectax', rectax

        #self.rectax = rectax # self.rectax is a current window which is set later
        self.raxes = rectax

        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None

        self._x1_old = None
        self._x2_old = None
        self._y1_old = None
        self._y2_old = None


    def set_rect_axes(self, rectax, set_def=True) :
        """ Sets new axes rect.
        """
        if set_def :
            self.set_rect_axes_default(rectax)
            self.reset_original_size()
        else :
            rs = self.rect_scene_from_rect_axes(rectax)
            self.set_rect_scene(rs)


    def set_rect_scene(self, rs) :
        self.scene().setSceneRect(rs)
        self.fitInView(rs, Qt.IgnoreAspectRatio)
        #print 'XXX GUView.set_rect_scene after fitInView'
        self.update_my_scene()


    def reset_original_size(self) :
        """call sequence of methods to reset original size.
        """
        self.set_view()
        self.update_my_scene()
        self.check_axes_limits_changed()


    def set_origin(self, origin='UL') :
        """Defines internal (bool) parameters for origin position. 
           Sensitive to U(T)L characters in (str) origin, standing for upper(top) lower.  
        """
        self._origin = origin
        key = origin.upper()

        self._origin_u = 'U' in key or 'T' in key
        self._origin_d = not self._origin_u

        self._origin_l = 'L' in key 
        self._origin_r = not self._origin_l

        self._origin_ul = self._origin_u and self._origin_l
        self._origin_ur = self._origin_u and self._origin_r
        self._origin_dl = self._origin_d and self._origin_l
        self._origin_dr = self._origin_d and self._origin_r

        #self.set_transform_orientation()


    def set_transform_orientation(self) :
        """Flips signs of scalex, scaley depending on origin.
        """
        if not self._origin_ul :
            t = self.transform()
            sx = 1 if self._origin_l else -1
            sy = 1 if self._origin_u else -1
            t2 = t.scale(sx, sy)
            self.setTransform(t2)


    def set_scale_control(self, scale_ctl='HV') :
        """Sets scale control bit-word
           = 0 - x, y frozen scales
           + 1 - x is interactive
           + 2 - y is interactive
           bit value 0/1 frozen/interactive  
        """
        self._scale_ctl = 0
        if 'H' in scale_ctl : self._scale_ctl += 1
        if 'V' in scale_ctl : self._scale_ctl += 2

 
    def scale_control(self) :
        return self._scale_ctl


    def set_style(self) :
        self.setStyleSheet("background-color:black; border: 0px solid green")
        #w.setContentsMargins(-9,-9,-9,-9)
        #self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        #self.setAttribute(Qt.WA_TranslucentBackground) # Qt.WA_NoSystemBackground
        #self.setInteractive(True)

        self.brudf = QtGui.QBrush()
        self.brubx = QtGui.QBrush(Qt.black, Qt.SolidPattern)

        self.pendf = QtGui.QPen()
        self.pendf.setStyle(Qt.NoPen)
        self.penbx = QtGui.QPen(Qt.black, 6, Qt.SolidLine)
        self.pen_gr= QtGui.QPen(Qt.green, 0, Qt.SolidLine)
        self.pen_bl= QtGui.QPen(Qt.blue,  0, Qt.SolidLine)


    def set_margins(self, margl=None, margr=None, margt=None, margb=None) :
        """Sets margins around axes rect to expend viewed scene"""
        self.margl = margl if margl is not None else 0.12
        self.margr = margr if margr is not None else 0.03
        self.margt = margt if margt is not None else 0.01
        self.margb = margb if margb is not None else 0.06


    def rect_scene_from_rect_axes(self, raxes=None) :
        # Sets scene rect larger than axes rect by margins
        ml, mr, mt, mb = self.margl, self.margr, self.margt, self.margb

        r = self.raxes if raxes is None else raxes
        #print_rect(r, cmt='XXX rect axes')
        
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        sx = w/(1. - ml - mr)
        sy = h/(1. - mt - mb)

        my = mt if self._origin_u else mb
        mx = ml if self._origin_l else mr
        return QtCore.QRectF(x-mx*sx, y-my*sy, sx, sy)


    def set_limits_vertical(self, ymin, ymax) :
        """ymin, ymax - axes coordinates
           self._ymin, self._ymax -scene coordinates
        """
        if None in (ymin, ymax) :
            self._ymin = ymin
            self._ymax = ymax
            return

        if ymax == ymin : ymax = ymin+1
        #print 'XXX:  ymin, ymax', ymin, ymax
        ml, mr, mt, mb = self.margl, self.margr, self.margt, self.margb
        hsc = (ymax-ymin)/(1 - mt - mb)
        self._ymin = ymin-hsc*mb if ymin<0 else -hsc*mb
        self._ymax = ymax+hsc*mt if ymax>0 else  hsc*mt


    def set_view(self) :
        rs = self.rect_scene_from_rect_axes()
        #print_rect(rs, cmt='XXX rect scene')
        self.scene().setSceneRect(rs)
        self.fitInView(rs, Qt.IgnoreAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio
        #print 'XXX GUView.set_view after fitInView'


    def check_axes_limits_changed(self):
        """Checks if axes limits have changed and submits signal with new limits.
        """
        r = self.rectax
        x1, x2 = r.left(), r.right()
        y1, y2 = r.bottom(), r.top()

        if x1 != self._x1_old or x2 != self._x2_old \
        or y1 != self._y1_old or y2 != self._y2_old :
            self._x1_old = x1
            self._x2_old = x2
            self._y1_old = y1
            self._y2_old = y2
            #self.evaluate_hist_statistics()
            self.emit(QtCore.SIGNAL('axes_limits_changed(float,float,float,float)'), x1, x2, y1, y2)


    def check_limits(self) :

        if all(v is None for v in (self._xmin, self._xmax, self._ymin, self._ymax)) : return  
        
        sc = self.scene()
        rs = sc.sceneRect()
        x, y, w, h = rs.x(), rs.y(), rs.width(), rs.height()

        x1 = x   if self._xmin is None else self._xmin
        x2 = x+w if self._xmax is None else self._xmax
        y1 = y   if self._ymin is None else self._ymin
        y2 = y+h if self._ymax is None else self._ymax

        rs = QtCore.QRectF(x1, y1, x2-x1, y2-y1)
        sc.setSceneRect(rs)
        self.fitInView(rs, Qt.IgnoreAspectRatio)
        #print 'XXX GUView.check_limits after fitInView'


    def update_my_scene(self) :
        """ Draws content on scene
        """
        #print 'In %s.update_my_scene' % self._name
        self.check_limits()

        sc = self.scene()
        rs = sc.sceneRect()
        x, y, w, h = rs.x(), rs.y(), rs.width(), rs.height()
        #print_rect(rs, cmt='XXX scene rect')

        x1ax, x2ax = x + w*self.margl, x + w - w*self.margr
        y1ax, y2ax = y + h*self.margb, y + h - h*self.margt

        wax = x2ax - x1ax
        hax = y2ax - y1ax

        # set dark rects
        if self.rslefv is not None : self.scene().removeItem(self.rslefv)
        if self.rsbotv is not None : self.scene().removeItem(self.rsbotv)
        if self.rsrigv is not None : self.scene().removeItem(self.rsrigv)
        if self.rstopv is not None : self.scene().removeItem(self.rstopv)

        # set interactive rects
        if self.rslefi is not None : self.scene().removeItem(self.rslefi)
        if self.rsboti is not None : self.scene().removeItem(self.rsboti)
        if self.rsrigi is not None : self.scene().removeItem(self.rsrigi)
        if self.rstopi is not None : self.scene().removeItem(self.rstopi)

        if self.raxesv is not None : self.scene().removeItem(self.raxesv)
        if self.raxesi is not None : self.scene().removeItem(self.raxesi)

        if self._origin_l :            
            #print 'L'
            self.rslef=QtCore.QRectF(x,    y, w*self.margl, h)
            self.rsrig=QtCore.QRectF(x2ax, y, w*self.margr, h)
        else :
            #print 'R'
            self.rslef=QtCore.QRectF(x,    y, w*self.margr, h)
            self.rsrig=QtCore.QRectF(x + w - w*self.margl, y, w*self.margl, h)

        self.rslefv = self.add_rect_to_scene_v1(self.rslef, self.brubx, self.penbx)
        self.rslefi = self.add_rect_to_scene(self.rslef, self.brudf, self.pendf)
        self.rslefi.setCursorHover(Qt.SizeVerCursor)
        self.rslefi.setCursorGrab (Qt.SplitVCursor)

        self.rsrigv = self.add_rect_to_scene_v1(self.rsrig, self.brubx, self.penbx)
        self.rsrigi = self.add_rect_to_scene(self.rsrig, self.brudf, self.pendf)
        self.rsrigi.setCursorHover(Qt.SizeVerCursor)
        self.rsrigi.setCursorGrab (Qt.SplitVCursor)

        if self._origin_ul :            
            #print 'UL'
            self.rstop  = QtCore.QRectF(x1ax, y + h - h*self.margb, wax, h*self.margb)
            self.rsbot  = QtCore.QRectF(x1ax, y, wax, h*self.margt)
            self.rectax = QtCore.QRectF(x1ax, y + h*self.margt, wax, hax)

        elif self._origin_dl :
            #print 'DL'
            self.rsbot  = QtCore.QRectF(x1ax, y,    wax, h*self.margb) #.normalized()
            self.rstop  = QtCore.QRectF(x1ax, y2ax, wax, h*self.margt) #.normalized()
            self.rectax = QtCore.QRectF(x1ax, y1ax, wax, hax) #.normalized()

        elif self._origin_dr : 
            #print 'DR'
            x1ax = x + w*self.margr
            self.rsbot  = QtCore.QRectF(x1ax, y,    wax, h*self.margb) #.normalized()
            self.rstop  = QtCore.QRectF(x1ax, y2ax, wax, h*self.margt) #.normalized()
            self.rectax = QtCore.QRectF(x1ax, y1ax, wax, hax) #.normalized()

        else : #self._origin_ur : 
            #print 'UR'
            x1ax = x + w*self.margr
            y1ax = y + h - h*self.margb
            self.rsbot  = QtCore.QRectF(x1ax, y,    wax, h*self.margt) #.normalized()
            self.rstop  = QtCore.QRectF(x1ax, y1ax, wax, h*self.margb) #.normalized()
            self.rectax = QtCore.QRectF(x1ax, y + h*self.margt, wax, hax)

        self.rsbotv = self.add_rect_to_scene_v1(self.rsbot, self.brubx, self.penbx)
        self.rsboti = self.add_rect_to_scene   (self.rsbot, self.brudf, self.pendf)

        self.rstopv = self.add_rect_to_scene_v1(self.rstop, self.brubx, self.penbx)
        self.rstopi = self.add_rect_to_scene   (self.rstop, self.brudf, self.pendf)

        self.raxesv = self.add_rect_to_scene_v1(self.rectax, self.brubx, self.penbx)
        self.raxesi = self.add_rect_to_scene   (self.rectax, self.brudf, self.pendf) # self.pen_gr
        #print_rect(self.rectax, 'XXX GUView.update_my_scene axes rect')

        self.raxesi.setCursorHover(Qt.CrossCursor)
        self.raxesi.setCursorGrab (Qt.SizeAllCursor)
        self.raxesi.setZValue(20)
        self.raxesv.setZValue(-1)

        self.rsboti.setCursorHover(Qt.SizeHorCursor)
        self.rsboti.setCursorGrab (Qt.SplitHCursor)
        self.rstopi.setCursorHover(Qt.SizeHorCursor)
        self.rstopi.setCursorGrab (Qt.SplitHCursor)

        self.rslefv.setZValue(1)
        self.rsbotv.setZValue(1)
        self.rsrigv.setZValue(1)
        self.rstopv.setZValue(1)

        self.rslefi.setZValue(20)
        self.rsboti.setZValue(20)
        self.rsrigi.setZValue(20)
        self.rstopi.setZValue(20)

        #self.updateScene([self.rsbot, self.rslef, self.rectax])

        #self.check_axes_limits_changed()

        if self.show_mode & 1 :
            colfld = Qt.magenta
            if self.raxi is not None : self.scene().removeItem(self.raxi)
            self.raxi = self.add_rect_to_scene_v1(self.raxes, pen=QtGui.QPen(Qt.NoPen), brush=QtGui.QBrush(colfld))
            self.raxi.setZValue(0.1)

        if self.show_mode & 2 :
            pc = self.raxes.topLeft()
            ror=QtCore.QRectF(pc.x()-2, pc.y()-2, 4, 4)
            colori = Qt.red
            if self.rori is not None : self.scene().removeItem(self.rori)
            self.rori = self.add_rect_to_scene(ror, pen=QtGui.QPen(colori, 0, Qt.SolidLine), brush=QtGui.QBrush(colori))
            self.rori.setZValue(0.2)


    def add_rect_to_scene_v1(self, rect, brush=QtGui.QBrush(), pen=QtGui.QPen(Qt.yellow, 4, Qt.DashLine)) :
        """Adds rect to scene, returns QGraphicsRectItem"""
        pen.setCosmetic(True)
        return self.scene().addRect(rect, pen, brush)


    def add_rect_to_scene(self, rect, brush=QtGui.QBrush(), pen=QtGui.QPen(Qt.yellow, 4, Qt.DashLine)) :
        """Adds rect to scene, returns GUQGraphicsRectItem - for interactive stuff"""
        from graphqt.GUQGraphicsRectItem import GUQGraphicsRectItem
        pen.setCosmetic(True)
        item = GUQGraphicsRectItem(rect, parent=None, scene=self.scene())
        item.setPen(pen)
        item.setBrush(brush)
        return item


    def display_pixel_pos(self, e):
        p = self.mapToScene(e.pos())
        #print 'mouseMoveEvent, current point: ', e.x(), e.y(), ' on scene: %.1f  %.1f' % (p.x(), p.y()) 
        self.setWindowTitle('GUView: x=%.1f y=%.1f' % (p.x(), p.y()))


    def rect_axes(self) :
        return self.rectax
        #return self.raxes


    def axes_limits(self) :
        r = self.rectax
        x1, x2 = r.left(), r.right()
        y1, y2 = r.bottom(), r.top()
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)

 
#------------------------------

    def move_scene(self, dp) :
        """Move scene/axes rect by QtCore.QPointF dp
        """
        sc = self.scene()
        rs = sc.sceneRect()
        rs.moveCenter(rs.center() + dp)
        sc.setSceneRect(rs)
        self.update_my_scene()
 
#------------------------------

    def connect_axes_limits_changed_to(self, recip) :
        self.connect(self, QtCore.SIGNAL('axes_limits_changed(float,float,float,float)'), recip)

    def disconnect_axes_limits_changed_from(self, recip) :
        self.disconnect(self, QtCore.SIGNAL('axes_limits_changed(float,float,float,float)'), recip)

    def test_axes_limits_changed_reception(self, x1, x2, y1, y2) :
        print 'GUView.test_axes_limits_changed_reception x1: %.2f  x2: %.2f  y1: %.2f  y2: %.2f' % (x1, x2, y1, y2)

#------------------------------

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
        #print 'GUView.mouseReleaseEvent, at point: ', e.pos(), ' diff:', e.pos() - self.pos_click
        #self.pos_click = e.pos()
        self.pos_click = None
        self.check_axes_limits_changed()


#    def mouseDoubleCkickEvent(self, e):
#        QtGui.QGraphicsView.mouseDoubleCkickEvent(self, e)
#        print 'mouseDoubleCkickEvent'


    def mousePressEvent(self, e):
        #print 'GUView.mousePressEvent, at point: ', e.pos() #e.globalX(), e.globalY() 
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
        elif item == self.rstopi : self.scalebw = 1 # print 'left rect' # |= 2
        elif item == self.rslefi : self.scalebw = 2 # print 'left rect' # |= 2
        elif item == self.rsrigi : self.scalebw = 2 # print 'left rect' # |= 2
        else                     : self.scalebw = 3
        #elif item == self.raxesi   : self.scalebw = 3 # print 'axes rect'
        #print '_select_further_action scalebw:', self.scalebw


    def mouseMoveEvent(self, e):
        QtGui.QGraphicsView.mouseMoveEvent(self, e)
        #print 'GUView.mouseMoveEvent, at point: ', e.pos()
        self.display_pixel_pos(e) # re-defined in GUViewImage, GUViewHist, etc.

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

        self.set_rect_scene(rs)
        #sc.setSceneRect(rs)
        #self.fitInView(rs, Qt.IgnoreAspectRatio)
        #self.update_my_scene()

        self._continue_wheel_event()


    def _continue_wheel_event(self, t_msec=500) :
        """Reset time interval for timer in order to catch wheel stop
        """
        self.timer.start(t_msec)
        #print 'update_on_wheel_event'


    def on_timeout(self) :
        """Is activated by timer when wheel is stopped and interval is expired
        """
        #print 'on_timeout'
        self.timer.stop()
        self.check_axes_limits_changed()
        self.emit(QtCore.SIGNAL('wheel_is_stopped()'))

#------------------------------

    def connect_wheel_is_stopped_to(self, recip) :
        self.connect(self, QtCore.SIGNAL('wheel_is_stopped()'), recip)

    def disconnect_wheel_is_stopped_from(self, recip) :
        self.disconnect(self, QtCore.SIGNAL('wheel_is_stopped()'), recip)

    def test_wheel_is_stopped_reception(self) :
        print 'GUView.test_wheel_is_stopped_reception'

#------------------------------

    def enterEvent(self, e) :
    #    print 'enterEvent'
        QtGui.QGraphicsView.enterEvent(self, e)
        #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(Qt.CrossCursor))
        

    def leaveEvent(self, e) :
    #    print 'leaveEvent'
        QtGui.QGraphicsView.leaveEvent(self, e)
        #QtGui.QApplication.restoreOverrideCursor()


    def closeEvent(self, e) :
        #print 'In %s.closeEvent' % (self._name) #, sys._getframe().f_code.co_name)
        QtGui.QGraphicsView.closeEvent(self, e)
        #print 'GUView.closeEvent' #% self._name
        self.emit(QtCore.SIGNAL('view_is_closed()'))


    def connect_view_is_closed_to(self, recip) :
        self.connect(self, QtCore.SIGNAL('view_is_closed()'), recip)


    def disconnect_view_is_closed_from(self, recip) :
        self.disconnect(self, QtCore.SIGNAL('view_is_closed()'), recip)


    def test_view_is_closed_reception(self) :
        print '%s.test_view_is_closed_reception' % self._name


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
        #print 'XXX GUView.resizeEvent after fitInView'


    #def paintEvent(e):
    #    pass
    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset original size'\
               '\n  W - change axes rect, do not change default'\
               '\n  D - change axes rect and its default'\
               '\n'

    def keyPressEvent(self, e) :
        #print 'keyPressEvent, key=', e.key()         
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_R : 
            print '%s: Reset original size' % self._name
            self.reset_original_size()

        elif e.key() == Qt.Key_W : 
            print '%s: set rect of axes, do not change default' % self._name
            v = ag.random_standard((4,), mu=0, sigma=20, dtype=np.int)
            rax = QtCore.QRectF(v[0], v[1], v[2]+100, v[3]+100)
            print 'Set axes rect: %s' % str(rax)
            self.set_rect_axes(rax, set_def=False)

        elif e.key() == Qt.Key_D : 
            print '%s: change default axes rect, set new default' % self._name
            v = ag.random_standard((4,), mu=0, sigma=20, dtype=np.int)
            rax = QtCore.QRectF(v[0], v[1], v[2]+100, v[3]+100)
            print 'Set new default axes rect: %s' % str(rax)
            self.set_rect_axes(rax) # def in GUView

        else :
            print self.key_usage()

#-----------------------------

def test_guiview(tname) :
    print '%s:' % sys._getframe().f_code.co_name
    app = QtGui.QApplication(sys.argv)
    w = None
    rectax=QtCore.QRectF(0, 0, 100, 100)
    if   tname == '0': w=GUView(None, rectax, origin='DL', scale_ctl='HV', margl=0.12, margr=0.10, margt=0.06, margb=0.06, show_mode=3)
    elif tname == '1': w=GUView(None, rectax, origin='DL', scale_ctl='',   show_mode=1)
    elif tname == '2': w=GUView(None, rectax, origin='DL', scale_ctl='H',  show_mode=1)
    elif tname == '3': w=GUView(None, rectax, origin='DL', scale_ctl='V',  show_mode=1)
    elif tname == '4': w=GUView(None, rectax, origin='UL', scale_ctl='HV', show_mode=3)
    elif tname == '5': w=GUView(None, rectax, origin='DL', scale_ctl='HV', show_mode=3)
    elif tname == '6': w=GUView(None, rectax, origin='DR', scale_ctl='HV', show_mode=3)
    elif tname == '7': w=GUView(None, rectax, origin='UR', scale_ctl='HV', show_mode=3)
    else :
        print 'test %s is not implemented' % tname
        return

    w.setGeometry(20, 20, 600, 600)
    w.setWindowTitle("GUView window")
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

