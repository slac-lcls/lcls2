#------------------------------
"""
Class :py:class:`FWView` is a QGraphicsView QWidget with interactive scalable scene
=====================================================================================

FW stands for Full Window - no margins for rullers etc.

Usage ::

    # Test
    #-----
    import sys
    from psana.graphqt.FWView import *
    app = QApplication(sys.argv)
    w = FWView(None, rscene=QRectF(0, 0, 100, 100), origin='UL', scale_ctl='HV')
    w.show()
    app.exec_()

    # Constructor and destructor
    #---------------------------
    from psana.graphqt.FWView import FWView
    w = FWView(parent=None, rscene=QRectF(0, 0, 10, 10), origin='UL', scale_ctl='HV', show_mode=0)
    w.__del__() # on w.close()

    # Main methods
    #-------------
    w.set_rect_scene(rs, set_def=True)
    w.reset_original_size()
    w.connect_cursor_on_scene_pos_to(recip)
    w.connect_scene_rect_changed_to(recip)
    w.connect_mouse_press_event_to(recip) 

    # Methods
    #---------
    w.set_origin(origin='UL')           # sets attributes for origin
    w.set_scale_control(scale_ctl='HV') # sets attribute _scale_ctl = 0/+1/+2
    sctrl = w.scale_control()
    w.set_style()
    w._set_signs_of_transform()
    w.set_background_style(bkgd="background-color:black; border: 0px solid green")
    w._set_scene_item_rect_zvalue(sc_zvalue=20)

    # Re-calls
    #---------    
    w.update_my_scene()
    w.set_cursor_type_on_scene_rect(curs_hover=Qt.CrossCursor, curs_grab=Qt.SizeAllCursor)
    w.set_view(rs=None)

    # Methods for tests
    #------------------
    w.add_rect_to_scene_v1(rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine))
    w.add_rect_to_scene(rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine))
    w.add_test_items_to_scene(show_mode=0)

    # Call-back slots
    #----------------
    w.mouseReleaseEvent(e)
    # w.mouseDoubleCkickEvent(e)
    w.mousePressEvent(e)
    w.mouseMoveEvent(e)
    w.wheelEvent(e)
    w.enterEvent(e)
    w.leaveEvent(e)
    w.closeEvent(e)
    w.resizeEvent(e)
    # w.paintEvent(e)
    w.key_usage(self)
    w.keyPressEvent(e)

    w.connect_mouse_move_event_to(recip)
    w.disconnect_mouse_move_event_from(recip)
    w.test_mouse_move_event_reception(e)     # resieves signal mouse_move_event(QMouseEvent)

    w.on_timeout()
    w.emit_signal_if_scene_rect_changed()        # emits signal scene_rect_changed(QRectF)
    w.connect_scene_rect_changed_to(recip)
    w.disconnect_scene_rect_changed_from(recip)
    w.test_scene_rect_changed_reception(rs)

    w.connect_mouse_press_event_to(recip)
    w.disconnect_mouse_press_event_from(recip)
    w.test_mouse_press_event_reception(e)

    # w.connect_wheel_is_stopped_to(recip)
    # w.disconnect_wheel_is_stopped_from(recip)
    # w.test_wheel_is_stopped_reception()

See:
    - :class:`FWView`
    - :class:`FWViewImage`
    - :class:`FWViewAxis`
    - :class:`FWViewColorBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-01-03 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication
from PyQt5.QtGui import QBrush, QPen, QCursor, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer

#from time import time, sleep       
#t0_sec = time()
#logger.debug('time = %.6f' % (time() - t0_sec))

#------------------------------

class FWView(QGraphicsView) :
    #wheel_is_stopped = pyqtSignal()
    #view_is_closed = pyqtSignal()
    mouse_move_event   = pyqtSignal('QMouseEvent')
    mouse_press_event  = pyqtSignal('QMouseEvent')
    scene_rect_changed = pyqtSignal('QRectF')

    def __init__(self, parent=None, rscene=QRectF(0, 0, 10, 10),\
                 origin='UL', scale_ctl='HV',\
                 show_mode=0) :
        """
        Parameters

        - parent (QWidget) - parent object
        - rscene (QRectF) - default rect on scaene
        - origin (str) - location of origin U(T)=Upper(Top), L=Left, other letters are not used but mean Down(Bottom), Right
        - scale_ctl (str) - scale control at mouse move, scroll, H=Horizontal, V=Vertical
        - show_mode (int) - 0(def) - draw nothing, +1 - field of scene rect, +2 - origin
        """

        sc = QGraphicsScene()
        QGraphicsView.__init__(self, sc, parent)

        self._name = self.__class__.__name__

        self.rs = rscene         # default rect on scene, restored at reset_original_size
        self.rs_old = self.rs    # rs_old - is used to check if scene rect changed
        self.rs_item = None      # item on scene to change cursor style at howering

        self.pos_click = None    # QPoint at mousePress
        self._set_scene_item_rect_zvalue()
        self.set_origin(origin)           # sets self._origin_...
        self.set_scale_control(scale_ctl) # sets self._scale_ctl
        self._set_signs_of_transform()        

        self.set_style()
        self.set_view()
        self.add_test_items_to_scene(show_mode)

        self.twheel_msec = 500 # timeout for wheelEvent
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timeout)


    def __del__(self) :
        pass


    def set_origin(self, origin='UL') :
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


    def set_scale_control(self, scale_ctl='HV') :
        """Sets scale control bit-word
           = 0 - x, y frozen scales
           + 1 - x is interactive
           + 2 - y is interactive
           bit value 0/1 frozen/interactive  
        """
        self.str_scale_ctl = scale_ctl
        self._scale_ctl = 0
        if 'H' in scale_ctl : self._scale_ctl += 1
        if 'V' in scale_ctl : self._scale_ctl += 2
 

    def scale_control(self) :
        return self._scale_ctl


    def str_scale_control(self) :
        return self.str_scale_ctl


    def origin(self) :
        return self._origin


    def set_style(self) :
        self.setGeometry(20, 20, 600, 600)
        self.setWindowTitle("FWView")
        #self.setContentsMargins(0,0,0,0)
        #self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        #self.setAttribute(Qt.WA_TranslucentBackground)
        #self.setInteractive(True)
        #self.setFixedSize(600, 22)

        self.brudf = QBrush()
        self.brubx = QBrush(Qt.black, Qt.SolidPattern)
        self.pendf = QPen()
        self.pendf.setStyle(Qt.NoPen)
        self.penbx = QPen(Qt.black, 6, Qt.SolidLine)

        self.set_background_style()

        #self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)


    def set_background_style(self, bkgd_color = QColor(50,5,50)) :
        """Sets self.bkgd (str) - background style"""
        # it seems like in pyqt5 setStyleSheet is not enough to re-paint bkgd at drug, zoom etc.
        #bkgd_style="background-color:black; border: 0px solid green",
        #self.setStyleSheet(bkgd_style)
        self.setBackgroundBrush(QBrush(bkgd_color, Qt.SolidPattern))


    def _set_signs_of_transform(self) :
        if not self._origin_ul :
            sx = 1 if self._origin_l else -1
            sy = 1 if self._origin_u else -1
            ts = self.transform().scale(sx, sy)
            self.setTransform(ts)


    def _set_scene_item_rect_zvalue(self, sc_zvalue=20) :
        """Sets self.sc_zvalue (float) - scene rect item z value for cursor behavir"""
        self.sc_zvalue = sc_zvalue


    def update_my_scene(self) :
        """Should be re-implemented, if necessary
        """
        self.set_cursor_type_on_scene_rect()


    def set_cursor_type_on_scene_rect(self, curs_hover=Qt.CrossCursor, curs_grab=Qt.SizeAllCursor) :
        """Optional method: sets current scene rect item self.rs_item for cursor type

        Parameters

        - curs_hover (QCursor) - cursor type at mouse hovering on scene rect
        - curs_grab  (QCursor) - cursor type at mouse click on scene rect
        """
        sc = self.scene()
        rs = sc.sceneRect()
        if self.rs_item is not None : sc.removeItem(self.rs_item)
        self.rs_item = self.add_rect_to_scene(rs, self.brudf, self.pendf)
        self.rs_item.setCursorHover(curs_hover)
        self.rs_item.setCursorGrab(curs_grab)
        self.rs_item.setZValue(self.sc_zvalue)


    def set_view(self, rs=None) :
        """Sets rs - rect on scene to view in viewport, by default self.rs is used
        """
        r = self.rs if rs is None else rs
        self.scene().setSceneRect(r)
        self.fitInView(r, Qt.IgnoreAspectRatio) # KeepAspectRatioByExpanding KeepAspectRatio
        self.update_my_scene()


    def set_rect_scene(self, rs, set_def=True) :
        """Sets new rect to view, set_def=True - updates default for reset.
        """
        if set_def : 
            self.rs = rs
        self.set_view(rs)
        self.emit_signal_if_scene_rect_changed() # sends signal


    def reset_original_size(self) :
        """Alias with default parameters"""
        self.set_view()
        self.emit_signal_if_scene_rect_changed() # sends signal


    def mouseReleaseEvent(self, e):
        QApplication.restoreOverrideCursor()
        QGraphicsView.mouseReleaseEvent(self, e)
        #logger.debug('FWView.mouseReleaseEvent, at point: ', e.pos(), ' diff:', e.pos() - self.pos_click)
        #self.pos_click = e.pos()
        self.pos_click = None
        ### self.set_cursor_type_on_scene_rect()

        self.emit_signal_if_scene_rect_changed()
        #self.update_my_scene()


#    def mouseDoubleCkickEvent(self, e):
#        QGraphicsView.mouseDoubleCkickEvent(self, e)
#        logger.debug('mouseDoubleCkickEvent')


    def mousePressEvent(self, e):
        #logger.debug('FWView.mousePressEvent, at point: ', e.pos() #e.globalX(), e.globalY())
        #QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor)) #Qt.SizeAllCursor))# ClosedHandCursor
        #self.emit(QtCore.SIGNAL('mouse_press_event(QMouseEvent)'), e)

        logger.debug('FWView.mousePressEvent but=%d %s scene x=%.1f y=%.1f'%\
                     (e.button(), str(e.pos()), self.x(), self.y())) # self.__class__.__name__

        self.mouse_press_event.emit(e)

        QGraphicsView.mousePressEvent(self, e)
        self.pos_click = e.pos()
        self.rs_center = self.scene().sceneRect().center()
        self.invscalex = 1./self.transform().m11()
        self.invscaley = 1./self.transform().m22()
        #self.pos_click_sc = self.mapToScene(self.pos_click)


    def mouseMoveEvent(self, e):
        QGraphicsView.mouseMoveEvent(self, e)
        #logger.debug('FWView.mouseMoveEvent, at point: %s' % str(e.pos()))
        #self.emit(QtCore.SIGNAL('mouse_move_event(QMouseEvent)'), e)
        self.mouse_move_event.emit(e)

        if self._scale_ctl==0 : return
        if self.pos_click is None : return        

        dp = e.pos() - self.pos_click
        dx = dp.x()*self.invscalex if self._scale_ctl & 1 else 0
        dy = dp.y()*self.invscaley if self._scale_ctl & 2 else 0
        dpsc = QPointF(dx, dy)

        sc = self.scene()
        rs = sc.sceneRect()
        rs.moveCenter(self.rs_center - dpsc)
        sc.setSceneRect(rs)

        #self.update_my_scene()


    def wheelEvent(self, e) :
        QGraphicsView.wheelEvent(self, e)

        if self._scale_ctl==0 : return

        #logger.debug('wheelEvent: ', e.angleDelta())
        f = 1 + 0.4 * (1 if e.angleDelta().y()>0 else -1)
        #logger.debug('Scale factor =', f)

        p = self.mapToScene(e.pos())
        px, py = p.x(), p.y() 
        #logger.debug('wheel x,y = ', e.x(), e.y(), ' on scene x,y =', p.x(), p.y())

        sc = self.scene()
        rs = sc.sceneRect()
        x,y,w,h = rs.x(), rs.y(), rs.width(), rs.height()
        #logger.debug('Scene x,y,w,h:', x,y,w,h)

        # zoom relative to mouse position
        dxc = (f-1)*(px-x)
        dyc = (f-1)*(py-y) 
        dx, sx = (dxc, f*w) if self._scale_ctl & 1 else (0, w)
        dy, sy = (dyc, f*h) if self._scale_ctl & 2 else (0, h)

        rs.setRect(x-dx, y-dy, sx, sy)      

        self.set_view(rs)

        self.timer.start(self.twheel_msec)


    def on_timeout(self) :
        """Is activated by timer when wheel is stopped and interval is expired.
        """
        #logger.debug('on_timeout')
        self.timer.stop()
        self.emit_signal_if_scene_rect_changed()
        #self.emit(QtCore.SIGNAL('wheel_is_stopped()'))

#------------------------------

#    def connect_wheel_is_stopped_to(self, recip) :
#        self.connect(self, QtCore.SIGNAL('wheel_is_stopped()'), recip)

#    def disconnect_wheel_is_stopped_from(self, recip) :
#        self.disconnect(self, QtCore.SIGNAL('wheel_is_stopped()'), recip)

#    def test_wheel_is_stopped_reception(self) :
#        logger.debug('GUView.test_wheel_is_stopped_reception')

#------------------------------

    def emit_signal_if_scene_rect_changed(self):
        """Checks if scene rect have changed and submits signal with new rect.
        """
        rs = self.scene().sceneRect()
        #logger.debug('XXX FWView.emit_signal_if_scene_rect_changed old, new:', self.rs_old, rs)

        #if not equal_rects(rs, self.rs_old) :
        if rs != self.rs_old :
            self.rs_old = rs
            #self.emit(QtCore.SIGNAL('scene_rect_changed(QRectF)'), rs)
            self.scene_rect_changed.emit(rs)

#------------------------------

    def connect_scene_rect_changed_to(self, recip) :
        #self.connect(self, QtCore.SIGNAL('scene_rect_changed(QRectF)'), recip)
        #self.scene_rect_changed('QRectF').connect(recip)
        self.scene_rect_changed.connect(recip)

    def disconnect_scene_rect_changed_from(self, recip) :
        self.scene_rect_changed.disconnect(recip)

    def test_scene_rect_changed_reception(self, rs) :
        from psana.graphqt.QWUtils import print_rect
        #logger.debug('GUView.test_scene_rect_changed_reception:', rs)
        print_rect(rs, cmt='FWView.test_scene_rect_changed_reception')

#------------------------------

    def enterEvent(self, e) :
    #    logger.debug('enterEvent')
        QGraphicsView.enterEvent(self, e)
        #QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
        

    def leaveEvent(self, e) :
    #    logger.debug('leaveEvent')
        QGraphicsView.leaveEvent(self, e)
        #QApplication.restoreOverrideCursor()


    def closeEvent(self, e) :
        #logger.debug('XXX FWView.closeEvent')
        sc = self.scene()
        for item in sc.items() : 
            #logger.debug('XXX removeItem:', item)
            sc.removeItem(item)
        QGraphicsView.closeEvent(self, e)
        #logger.debug('XXX closeEvent is over')


    #def moveEvent(self, e) :
    #    logger.debug('moveEvent')
    #    logger.debug('Geometry rect:', self.geometry())


    def resizeEvent(self, e) :
        QGraphicsView.resizeEvent(self, e)
        rs = self.scene().sceneRect()    
        self.set_view(rs)


    #def paintEvent(self, e):
    #    logger.debug('paintEvent')  
    #    QGraphicsView.paintEvent(self, e)


    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset original size'\
               '\n  W - change scene rect, do not change default'\
               '\n  D - change scene rect and its default'\
               '\n'


    def keyPressEvent(self, e) :
        #logger.debug('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_R : 
            print('Reset original size')
            self.reset_original_size()

        elif e.key() in (Qt.Key_W, Qt.Key_D)  : 
            change_def = e.key()==Qt.Key_D
            print('%s: change scene rect %s' % (self._name, 'set new default' if change_def else ''))
            v = ag.random_standard((4,), mu=0, sigma=20, dtype=np.int)
            rs = QRectF(v[0], v[1], v[2]+100, v[3]+100)
            print('Set scene rect: %s' % str(rs))
            self.set_rect_scene(rs, set_def=change_def)

        else :
            print(self.key_usage())

#-----------------------------

    def add_rect_to_scene_v1(self, rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine)) :
        """Adds rect to scene, returns QGraphicsRectItem"""
        pen.setCosmetic(True)
        return self.scene().addRect(rect, pen, brush)


    def add_rect_to_scene(self, rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine)) :
        """Adds rect to scene, returns QWGraphicsRectItem - for interactive stuff"""

        #logger.debug('XXX:QWGraphicsRectItem TBD')

        from psana.graphqt.QWGraphicsRectItem import QWGraphicsRectItem
        pen.setCosmetic(True)
        item = QWGraphicsRectItem(rect, parent=None, scene=self.scene())
        item.setPen(pen)
        item.setBrush(brush)
        return item
        return None


    def add_test_items_to_scene(self, show_mode=0) :
        colfld = Qt.magenta
        colori = Qt.red
        if show_mode & 1 :
            self.rsi = self.add_rect_to_scene_v1(self.rs, pen=QPen(Qt.NoPen), brush=QBrush(colfld))
        if show_mode & 2 :
            ror=QRectF(-1, -1, 2, 2)
            self.rori = self.add_rect_to_scene_v1(ror, pen=QPen(colori, 0, Qt.SolidLine), brush=QBrush(colori))


#------------------------------

    def connect_mouse_press_event_to(self, recip) :
        self.mouse_press_event.connect(recip)

    def disconnect_mouse_press_event_from(self, recip) :
        self.mouse_press_event.disconnect(recip)

    def test_mouse_press_event_reception(self, e) :
        print('FWViewImage.mouse_press_event, QMouseEvent point: x=%d y=%d' % (e.x(), e.y()))

#-----------------------------

    def test_mouse_move_event_reception(self, e) :
        #logger.debug('mouseMoveEvent, current point: ', e.x(), e.y(), ' on scene: %.1f  %.1f' % (p.x(), p.y()))
        p = self.mapToScene(e.pos())
        self.setWindowTitle('FWView: x=%.1f y=%.1f %s' % (p.x(), p.y(), 25*' '))

    def connect_mouse_move_event_to(self, recip) :
        #self.connect(self, QtCore.SIGNAL('mouse_move_event(QMouseEvent)'), recip)
        self.mouse_move_event['QMouseEvent'].connect(recip)

    def disconnect_mouse_move_event_from(self, recip) :
        #self.disconnect(self, QtCore.SIGNAL('mouse_move_event(QMouseEvent)'), recip)
        self.mouse_move_event['QMouseEvent'].disconnect(recip)

#-----------------------------

    #def drawBackground(self, painter, rect):
        #painter.fillRect(rect, QBrush(QColor(0,0,0), Qt.SolidPattern))

#-----------------------------
#-----------------------------
#-----------------------------
if __name__ == "__main__" :
  def test_fwview(tname) :
    print('%s:' % sys._getframe().f_code.co_name)
    b="background-color:yellow; border: 0px solid green"
    app = QApplication(sys.argv)
    w = None
    if   tname == '0': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='DL', show_mode=3, scale_ctl='HV')
    elif tname == '1': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='UL', show_mode=3, scale_ctl='HV')
    elif tname == '2': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='UR', show_mode=3, scale_ctl='HV')
    elif tname == '3': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='DR', show_mode=3, scale_ctl='HV')
    elif tname == '4': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='DL', show_mode=3, scale_ctl='')
    elif tname == '5': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='DL', show_mode=3, scale_ctl='H')
    elif tname == '6': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='DL', show_mode=3, scale_ctl='V')
    elif tname == '7': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='DL', show_mode=1, scale_ctl='HV')
    elif tname == '8': w=FWView(None, rscene=QRectF(0, 0, 100, 100), origin='DL', show_mode=3, scale_ctl='HV')
    else :
        print('test %s is not implemented' % tname)
        return

    w.connect_mouse_move_event_to(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed_to(w.test_scene_rect_changed_reception)
    w.connect_mouse_press_event_to(w.test_mouse_press_event_reception)

    w.show()

    #w.disconnect_mouse_move_event_from(w.test_mouse_move_event_reception)
    #w.disconnect_scene_rect_changed_from(w.test_scene_rect_changed_reception)
    #w.disconnect_mouse_press_event_from(w.test_mouse_press_event_reception)
    #w.close()

    app.exec_()
    app.quit()

    del w
    del app

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    import psana.pyalgos.generic.NDArrGenerators as ag; global ag

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_fwview(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------

