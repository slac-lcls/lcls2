
"""
Class :py:class:`GWViewExt` is a QGraphicsView QWidget with interactive scalable scene
=====================================================================================

FW stands for Full Window - no margins for rullers etc.

Usage ::

    # Test
    #-----
    import sys
    from .GWViewExt import *
    app = QApplication(sys.argv)
    w = GWViewExt(None, rscene=QRectF(0, 0, 100, 100), origin='UL', scale_ctl='HV')
    w.show()
    app.exec_()

    # Constructor and destructor
    #---------------------------
    from .GWViewExt import GWViewExt
    w = GWViewExt(parent=None, rscene=QRectF(0, 0, 10, 10), origin='UL', scale_ctl='HV', show_mode=0)
    w.__del__() # on w.close()

    # Main methods
    #-------------
    w.set_rect_scene(rs, set_def=True)
    w.reset_original_size()
    w.connect_cursor_on_scene_pos_to(recip)
    w.connect_scene_rect_changed(recip)
    w.connect_mouse_press_event(recip)

    # Methods
    #---------
    w.set_origin(origin='UL')           # sets attributes for origin
    w.set_scale_control(scale_ctl='HV') # sets attribute _scale_ctl = 0/+1/+2
    sctrl = w.scale_control()
    w.set_style()
    w._set_signs_of_transform()
    w.set_background_brush(color=QColor(50,5,50), pattern=Qt.SolidPattern)

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

    w.connect_mouse_move_event(recip)
    w.disconnect_mouse_move_event(recip)
    w.test_mouse_move_event_reception(e)     # resieves signal mouse_move_event(QMouseEvent)

    w.on_timeout()
    w.emit_signal_if_scene_rect_changed()        # emits signal scene_rect_changed(QRectF)
    w.connect_scene_rect_changed(recip)
    w.disconnect_scene_rect_changed(recip)
    w.test_scene_rect_changed_reception(rs)

    w.connect_mouse_press_event(recip)
    w.disconnect_mouse_press_event(recip)
    w.test_mouse_press_event_reception(e)

    # w.connect_wheel_is_stopped(recip)
    # w.disconnect_wheel_is_stopped(recip)
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
Re-factored from FWView on 2022-06-29
"""

from psana.graphqt.GWView import *

logger = logging.getLogger(__name__)

from psana.graphqt.QWGraphicsRectItem import QWGraphicsRectItem
#import psana.graphqt.QWUtils as qu # print_rect


class GWViewExt(GWView):
    mouse_move_event   = pyqtSignal('QMouseEvent')
    mouse_press_event  = pyqtSignal('QMouseEvent')
    scene_rect_changed = pyqtSignal('QRectF')

    def __init__(self, parent=None, rscene=QRectF(0, 0, 10, 10),\
                 origin='UL', scale_ctl='HV',\
                 show_mode=0, signal_fast=True, **kwa):
        """
        Parameters

        - parent (QWidget) - parent object
        - rscene (QRectF) - default rect on scaene
        - origin (str) - location of origin U(T)=Upper(Top), L=Left, other letters are not used but mean Down(Bottom), Right
        - scale_ctl (str) - scale control at mouse move, scroll, H=Horizontal, V=Vertical
        - show_mode (int) - 0(def) - draw nothing, +1 - field of scene rect, +2 - origin
        - signal_fast (bool) - send fast signal if the scene rect is changed
        """

        GWView.__init__(self, parent, rscene, scale_ctl, show_mode, **kwa)

        self._name = self.__class__.__name__
        self.rs = rscene         # default rect on scene, restored at reset_original_size
        self.rs_old = self.rs    # rs_old - is used to check if scene rect changed
        self.rs_item = None      # item on scene to change cursor style at howering
        self.sc_zvalue = kwa.get('sc_zvalue', 20)

        self.set_origin(origin)           # sets self._origin_...

        self.twheel_msec = 500 # timeout for wheelEvent
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timeout)
        self.signal_fast = signal_fast


    def __del__(self):
        pass


    def set_origin(self, origin='UL'):
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

        self._set_signs_of_transform()


    def _set_signs_of_transform(self):
        if not self._origin_ul:
            sx = 1 if self._origin_l else -1
            sy = 1 if self._origin_u else -1
            ts = self.transform().scale(sx, sy)
            self.setTransform(ts)


    def origin(self):
        return self._origin


    def scale_control(self):
        return self._scale_ctl


    def str_scale_control(self):
        return self.str_scale_ctl


    def update_my_scene(self):
        """Should be re-implemented, if necessary
        """
        self.set_cursor_type_on_scene_rect()


    def set_cursor_type_on_scene_rect(self, curs_hover=Qt.CrossCursor, curs_grab=Qt.SizeAllCursor):
        """Optional method: sets current scene rect item self.rs_item for cursor type

        Parameters

        - curs_hover (QCursor) - cursor type at mouse hovering on scene rect
        - curs_grab  (QCursor) - cursor type at mouse click on scene rect
        """
        sc = self.scene()
        rs = sc.sceneRect()
        if self.rs_item is not None: sc.removeItem(self.rs_item)
        self.rs_item = self.add_rect_to_scene(rs, self.brudf, self.pendf)
        self.rs_item.setCursorHover(curs_hover)
        self.rs_item.setCursorGrab(curs_grab)
        self.rs_item.setZValue(self.sc_zvalue)


    def set_view(self, rs=None):
        """Sets rs - rect on scene to view in viewport, by default self.rs is used
        """
        r = self.rs if rs is None else rs
        self.scene().setSceneRect(r)
        self.fitInView(r, Qt.IgnoreAspectRatio) # KeepAspectRatioByExpanding KeepAspectRatio
        self.update_my_scene()


    def set_rect_scene(self, rs, set_def=True):
        """Sets new rect to view, set_def=True - updates default for reset.
        """
        if set_def:
            self.rs = rs
        self.set_view(rs)
        self.emit_signal_if_scene_rect_changed() # sends signal


    def reset_original_size(self):
        """Alias with default parameters"""
        self.set_view()
        self.emit_signal_if_scene_rect_changed() # sends signal


    def mouseReleaseEvent(self, e):
        QApplication.restoreOverrideCursor()
        GWView.mouseReleaseEvent(self, e)
        #logger.debug('GWViewExt.mouseReleaseEvent, at point: '+str(e.pos()))
        self.emit_signal_if_scene_rect_changed()


    def mousePressEvent(self, e):
        logger.debug('GWViewExt.mousePressEvent but=%d %s scene x=%.1f y=%.1f'%\
                     (e.button(), str(e.pos()), self.x(), self.y())) # self.__class__.__name__

        self.mouse_press_event.emit(e)
        GWView.mousePressEvent(self, e)


    def mouseMoveEvent(self, e):
        GWView.mouseMoveEvent(self, e)
        #logger.debug('GWViewExt.mouseMoveEvent, at point: %s' % str(e.pos()))
        self.mouse_move_event.emit(e)
        if self.signal_fast: self.emit_signal_if_scene_rect_changed()


    def wheelEvent(self, e):
        GWView.wheelEvent(self, e)
        if self.signal_fast: self.emit_signal_if_scene_rect_changed()
        else: self.timer.start(self.twheel_msec)


    def on_timeout(self):
        """Is activated by timer when wheel is stopped and interval is expired.
        """
        #logger.debug('on_timeout')
        self.timer.stop()
        self.emit_signal_if_scene_rect_changed()
        #self.emit(QtCore.SIGNAL('wheel_is_stopped()'))


    def emit_signal_if_scene_rect_changed(self):
        """Checks if scene rect have changed and submits signal with new rect.
        """
        rs = self.scene().sceneRect()
        if rs != self.rs_old:
            self.rs_old = rs
            self.scene_rect_changed.emit(rs)


    def connect_scene_rect_changed(self, recip):
        self.scene_rect_changed.connect(recip)


    def disconnect_scene_rect_changed(self, recip):
        self.scene_rect_changed.disconnect(recip)


    def test_scene_rect_changed_reception(self, rs):
        #qu.print_rect(rs, cmt='GWViewExt.test_scene_rect_changed_reception')
        print('GWViewExt.test_scene_rect_changed_reception', rs)


    def enterEvent(self, e):
    #    logger.debug('enterEvent')
        GWView.enterEvent(self, e)
        #QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))


    def leaveEvent(self, e):
    #    logger.debug('leaveEvent')
        GWView.leaveEvent(self, e)
        #QApplication.restoreOverrideCursor()


    def closeEvent(self, e):
        #logger.debug('XXX GWViewExt.closeEvent')
        sc = self.scene()
        for item in sc.items():
            #logger.debug('XXX removeItem:', item)
            sc.removeItem(item)
        GWView.closeEvent(self, e)
        #logger.debug('XXX closeEvent is over')


    def add_rect_to_scene(self, rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine)):
        """Adds rect to scene, returns QWGraphicsRectItem - for interactive stuff"""
        #logger.debug('add_rect_to_scene %s' % qu.info_rect_xywh(rect))
        pen.setCosmetic(True)
        item = QWGraphicsRectItem(rect, parent=None, scene=self.scene())
        item.setPen(pen)
        item.setBrush(brush)
        return item


    def connect_mouse_press_event(self, recip):
        self.mouse_press_event.connect(recip)


    def disconnect_mouse_press_event(self, recip):
        self.mouse_press_event.disconnect(recip)


    def test_mouse_press_event_reception(self, e):
        print('GWViewExtImage.mouse_press_event, QMouseEvent point: x=%d y=%d' % (e.x(), e.y()))


    def connect_mouse_move_event(self, recip):
        self.mouse_move_event['QMouseEvent'].connect(recip)


    def disconnect_mouse_move_event(self, recip):
        self.mouse_move_event['QMouseEvent'].disconnect(recip)


    def test_mouse_move_event_reception(self, e):
        p = self.mapToScene(e.pos())
        self.setWindowTitle('GWViewExt: x=%.1f y=%.1f %s' % (p.x(), p.y(), 25*' '))


    def resizeEvent(self, e):
        GWView.resizeEvent(self, e)
        rs = self.scene().sceneRect()
        self.set_view(rs)


    #def paintEvent(self, e):
    #    logger.debug('paintEvent')
    #    GWView.paintEvent(self, e)


    #def drawBackground(self, painter, rect):
        #painter.fillRect(rect, QBrush(QColor(0,0,0), Qt.SolidPattern))


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF

