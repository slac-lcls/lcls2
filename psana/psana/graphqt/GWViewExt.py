
"""
Class :py:class:`GWViewExt`
===========================

GW - Graphical Widget
GWViewExt - is an extension of the class GWView. Adds to GWView
   - orientation by setting origin e.g.='UL'
   - default scene rect with reset_scene_rect
   - emit signals for scene_rect_changed, mouse_move_event, mouse_press_event
   - auto-changing cursor type for scene rect

Usage ::
    from psana.graphqt.GWViewExt import *

See:
    - graphqt/examples/ex_FWViewExt.py
    - :class:`GWView`
    - :class:`GWViewExt`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created as FWView on 2017-01-03 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
Refactored/split FWView to GWView and GWViewExt on 2022-07-12
"""

from psana.graphqt.GWView import *
from psana.graphqt.QWGraphicsRectItem import QWGraphicsRectItem

logger = logging.getLogger(__name__)


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

        GWView.__init__(self, parent, rscene, origin, scale_ctl, show_mode, **kwa)

        self.rs_def = rscene  # default rect on scene, restored at reset_original_size
        self.rs_old = rscene  # rs_old - is used to check if scene rect changed
        self.rs_item = None   # item on scene to change cursor style at howering
        self.sc_zvalue = kwa.get('sc_zvalue', 20)
        self.signal_fast = signal_fast

        self.set_origin(origin)
        self.update_my_scene()
        self.fit_in_view()  # mode=Qt.IgnoreAspectRatio or Qt.KeepAspectRatio
        self.init_timer()


    def init_timer(self):
        self.twheel_msec = self.kwa.get('twheel_msec', 500)  # timeout for wheelEvent
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timeout)


    def set_style(self):
        logger.debug('GWViewExt.set_style')
        GWView.set_style(self)  # set_background_brush
        self.brudf = QBrush()
        self.brubx = QBrush(Qt.black, Qt.SolidPattern)
        self.pendf = QPen()
        self.pendf.setStyle(Qt.NoPen)
        self.penbx = QPen(Qt.black, 6, Qt.SolidLine)


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
        """returns str parameter of origin, e.g. 'UL'"""
        return self._origin


    def scale_control(self):
        """returns int value of scale control bitword"""
        return self._scale_ctl


    def str_scale_control(self):
        """returns str value of scale control, e.g. 'HV'"""
        return self.str_scale_ctl


    def mousePressEvent(self, e):
        """emits mouse_press_event signal for non-derived methods"""
        GWView.mousePressEvent(self, e)
        self.mouse_press_event.emit(e)


    def mouseReleaseEvent(self, e):
        """emits signal for non-derived methods"""
        QApplication.restoreOverrideCursor()
        GWView.mouseReleaseEvent(self, e)
        self.emit_signal_if_scene_rect_changed()


    def mouseMoveEvent(self, e):
        #logger.debug('GWViewExt.mouseMoveEvent, at point: %s' % str(e.pos()))
        GWView.mouseMoveEvent(self, e)
        self.mouse_move_event.emit(e)
        if self.signal_fast: self.emit_signal_if_scene_rect_changed()


    def wheelEvent(self, e):
        GWView.wheelEvent(self, e)
        if self.signal_fast: self.emit_signal_if_scene_rect_changed()
        else: self.timer.start(self.twheel_msec)


    def resizeEvent(self, e):
        GWView.resizeEvent(self, e)
        r = self.scene_rect()
        self._add_cursor_type_rect_to_scene(r, self.brudf, self.pendf)


    def closeEvent(self, e):
        logger.debug('GWViewExt.closeEvent')
        sc = self.scene()
        for item in sc.items():
            sc.removeItem(item)
        GWView.closeEvent(self, e)


    def on_timeout(self):
        """Is activated by timer when wheel is stopped and interval is expired.
        """
        logger.debug('GWViewExt.on_timeout')
        self.timer.stop()
        self.emit_signal_if_scene_rect_changed()
        #self.emit(QtCore.SIGNAL('wheel_is_stopped()'))


    def emit_signal_if_scene_rect_changed(self):
        """Checks if scene rect have changed and submits signal with new rect.
        """
        r = self.scene_rect()
        if r != self.rs_old:
            self.rs_old = r
            self.scene_rect_changed.emit(r)
            self.update_my_scene()


    def connect_mouse_move_event(self, recip):
        self.mouse_move_event['QMouseEvent'].connect(recip)

    def disconnect_mouse_move_event(self, recip):
        self.mouse_move_event['QMouseEvent'].disconnect(recip)

    def test_mouse_move_event_reception(self, e):
        p = self.mapToScene(e.pos())
        self.setWindowTitle('GWViewExt: x=%.1f y=%.1f %s' % (p.x(), p.y(), 25*' '))


    def connect_scene_rect_changed(self, recip):
        self.scene_rect_changed.connect(recip)

    def disconnect_scene_rect_changed(self, recip):
        self.scene_rect_changed.disconnect(recip)

    def test_scene_rect_changed_reception(self, r):
        #logger.debug('GWViewExt.test_scene_rect_changed_reception rect: %s' % qu.info_rect_xywh(r))
        pass

    def connect_mouse_press_event(self, recip):
        self.mouse_press_event.connect(recip)

    def disconnect_mouse_press_event(self, recip):
        self.mouse_press_event.disconnect(recip)

    def test_mouse_press_event_reception(self, e):
        print('GWViewExt.mouse_press_event, QMouseEvent point: x=%d y=%d' % (e.x(), e.y()))


    def reset_scene_rect(self, rs=None, mode=Qt.IgnoreAspectRatio):
        """NEW: uses/reset self.rs_def to set scene rect and fit_in_view."""
        if rs is not None: self.rs_def = rs
        self.set_scene_rect(self.rs_def)
        self.fit_in_view(self.rs_def, mode)


    def _add_cursor_type_rect_to_scene(self, rect, brush=QBrush(), pen=QPen(Qt.yellow, 4, Qt.DashLine)):
        """Adds rect to scene, returns QWGraphicsRectItem - to control cursor type"""
        #logger.debug('GWViewExt._add_cursor_type_rect_to_scene %s' % qu.info_rect_xywh(rect))
        pen.setCosmetic(True)
        item = QWGraphicsRectItem(rect, parent=None, scene=self.scene())
        item.setPen(pen)
        item.setBrush(brush)
        return item


    def set_cursor_type_rect(self, curs_hover=Qt.CrossCursor, curs_grab=Qt.SizeAllCursor):
        """Optional method: sets current scene rect item self.rs_item for cursor type

        Parameters

        - curs_hover (QCursor) - cursor type at mouse hovering on scene rect
        - curs_grab  (QCursor) - cursor type at mouse click on scene rect
        """
        sc = self.scene()
        rs = sc.sceneRect()
        if self.rs_item is not None: sc.removeItem(self.rs_item)
        self.rs_item = self._add_cursor_type_rect_to_scene(rs, self.brudf, self.pendf)
        self.rs_item.setCursorHover(curs_hover)
        self.rs_item.setCursorGrab(curs_grab)
        self.rs_item.setZValue(self.sc_zvalue)


    def update_my_scene(self):
        #logger.debug('GWViewExt.update_my_scene')
        self.set_cursor_type_rect()


class OtherDeprecated:

    def __del__(self):
        pass

    def reset_original_size(self):
        """Alias with default parameters"""
        self.fit_in_view()
        self.emit_signal_if_scene_rect_changed() # sends signal

    def enterEvent(self, e):
        logger.debug('enterEvent')
        GWView.enterEvent(self, e)
        #QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))

    def leaveEvent(self, e):
        logger.debug('leaveEvent')
        GWView.leaveEvent(self, e)
        #QApplication.restoreOverrideCursor()


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF

