
"""
Class :py:class:`GWViewImageROI` is a GWView for interactive image
====================================================================

GWView <- QGraphicsView <- ... <- QWidget

Usage ::

    # Test
    #-----
    import sys
    from psana.graphqt.GWViewImageROI import *
    import psana.graphqt.ColorTable as ct
    app = QApplication(sys.argv)
    ctab = ct.color_table_monochr256()
    w = GWViewImageROI(None, arr, origin='UL', scale_ctl='HV', coltab=ctab)
    w.show()
    app.exec_()

    # Main methods in addition to GWView
    #------------------------------------
    w.set_pixmap_from_arr(arr, set_def=True)
    w.set_coltab(self, coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20))

    w.connect_mouse_press_event(w.test_mouse_press_event_reception)
    w.connect_mouse_move_event(w.test_mouse_move_event_reception)
    w.connect_scene_rect_changed(w.test_scene_rect_changed_reception)

    # Methods
    #--------
    w.set_style()
    ix, iy, v = w.cursor_on_image_pixcoords_and_value(p)

    # Call-back slots
    #----------------
    w.mousePressEvent(e)
    # w.mouseMoveEvent(e)
    # w.closeEvent(e)
    w.key_usage()
    w.keyPressEvent(e)

    # Overrides method from GWView
    #-----------------------------
    w.test_mouse_move_event_reception(e) # signature differs from GWView

    # Global methods for test
    #------------------------
    img = image_with_random_peaks(shape=(500, 500))

See:
    - :class:`GWView`
    - :class:`GWViewImage`
    - :class:`QWSpectrum`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-09-09 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
"""

from psana.graphqt.GWViewImage import *
from psana.pyalgos.generic.NDArrUtils import np, info_ndarr
from PyQt5.QtGui import  QPen, QPainter, QColor, QBrush, QTransform, QPolygonF
from PyQt5.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QLineF
import psana.graphqt.QWUtils as gu
import psana.graphqt.GWROIUtils as roiu
QPEN_DEF, QBRUSH_DEF, QCOLOR_DEF  = roiu.QPEN_DEF, roiu.QBRUSH_DEF, roiu.QCOLOR_DEF
QCOLOR_SEL = QColor('#ffeeaaee') #  Qt.magenta

class GWViewImageROI(GWViewImage):

    image_pixmap_changed = pyqtSignal()

    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', show_mode=0, signal_fast=True):

        GWViewImage.__init__(self, parent, arr, coltab, origin, scale_ctl, show_mode, signal_fast)
        self._name = 'GWViewImageROI' # self.__class__.__name__
        self.set_roi_and_mode()
        self.list_of_rois = []
        self._iscpos_old = None
        self.left_is_pressed = False
        self.roi_active = None
        self.scpos_first = None
        self.clicknum = 0
        #self.set_style_focus()

    def set_style_focus(self):
        #    QGraphicsView::item:focus {
        self.style = """
            QGraphicsRectItem:focus {
                background: red;
                selection-background-color: green;
                border: 1px solid gray;}
            QGraphicsRectItem:focus {border-color: blue;}"""
        self.setStyleSheet(self.style)

#    def paintEvent(self, e):
#        print('XXX in paintEvent') # , dir(e))
#        GWViewImage.paintEvent(self, e)


    def set_roi_and_mode(self, roi_type=roiu.NONE, mode_type=roiu.NONE):
        self.roi_type = roi_type
        self.roi_name = roiu.dict_mode_type_name[roi_type]
        self.mode_type = mode_type
        self.mode_name = roiu.dict_mode_type_name[mode_type]


    def mousePressEvent(self, e):
        GWViewImage.mousePressEvent(self, e)

        if self.mode_type < roiu.ADD: return
        self.left_is_pressed = e.button() == Qt.LeftButton
        if not self.left_is_pressed:
            logging.warning('USE MOUSE LEFT BUTTON ONLY !!!')
            return

        logger.debug('XXX GWViewImageROI.mousePressEvent but=%d (1/2/4 = L/R/M) screen x=%.1f y=%.1f'%\
                     (e.button(), e.pos().x(), e.pos().y()))

        if self.mode_type & roiu.SELECT:
            self.select_roi(e)

        if self.mode_type & roiu.REMOVE:
            self.remove_roi(e)

        if self.mode_type & roiu.ADD:
            self.clicknum += 1
            scpos = self.mapToScene(e.pos())

            if self.roi_active is None: #  1st click
                self.add_roi(e)
                self.scpos_first = scpos

            else: # other clicks
                if self.roi_type == roiu.POLYGON:
                    d = (scpos - self.scpos_first).manhattanLength()
                    logging.info('POLYGON manhattanLength(last-first): %.1f closeng distance: %.1f'%\
                            (d, self.roi_active.tolerance))
                    if self.clicknum < 2 or d > self.roi_active.tolerance:
                        self.roi_active.click_at_add(scpos)
                    else:
                        self.roi_active.set_poly() # set poly at last non-closing click
                        self.roi_active = None

                elif self.roi_type in (roiu.POLYREG, roiu.ARCH): # 3-click input
                    self.roi_active.set_point(scpos, self.clicknum)
                    if self.clicknum > 2:
                        self.roi_active = None
                        self.clicknum = 0
                else:
                    self.roi_active = None


    def mouseMoveEvent(self, e):
        GWViewImage.mouseMoveEvent(self, e)
        if self.mode_type < roiu.ADD: return
        #if not self.left_is_pressed: return

        scpos = self.mapToScene(e.pos())
        iscpos = QPoint(int(scpos.x()), int(scpos.y()))

        if iscpos == self._iscpos_old: return
        self._iscpos_old = iscpos

        if self.mode_type & roiu.ADD:
          if self.roi_type == roiu.PIXEL:
              if self.left_is_pressed:
                  self.add_roi_pixel(iscpos)
          elif self.roi_active is not None:
              self.roi_active.move_at_add(scpos)

        if self.mode_type & roiu.REMOVE:
          if self.roi_type == roiu.PIXEL and self.left_is_pressed:
             self.remove_roi_pixel(e)


    def mouseReleaseEvent(self, e):
        GWViewImage.mouseReleaseEvent(self, e)
        logger.debug('mouseReleaseEvent but=%d %s' % (e.button(), str(e.pos())))

        self.left_is_pressed = False

        if self.mode_type & roiu.ADD:

            if self.roi_type in (roiu.POLYGON, roiu.POLYREG, roiu.ARCH):
                return

            elif self.roi_type == roiu.PIXEL:
                self.roi_active = None
                self.clicknum = 0

            elif self.clicknum > 1: # number of clicks > 1
                self.roi_active = None
                self.clicknum = 0

            else:
                d = (self.mapToScene(e.pos()) - self.scpos_first).manhattanLength()
                if self.roi_active and d > self.roi_active.tolerance: # count as click-drag-release input
                    self.roi_active = None
                    self.clicknum = 0


    def select_roi(self, e):
        """select ROI on mouthPressEvent"""
        logger.debug('GWViewImageROI.select_roi')
        items = self.scene().items(self.mapToScene(e.pos()))
        #logger.debug('select_roi list of scene items at point: %s' % str(items))
        roisel = [o for o in self.list_of_rois if o.scitem in items]
        logger.debug('select_roi list of ROIs at point: %s' % str(roisel))
        for o in roisel:
             color = QCOLOR_DEF if o.scitem.pen().color() == QCOLOR_SEL else QCOLOR_SEL
             o.scitem.setPen(QPen(color, 1, Qt.SolidLine))


    def remove_roi(self, e):
        """remove ROI on mouthPressEvent"""
        logger.debug('GWViewImageROI.remove_roi')
        if self.roi_type == roiu.PIXEL:
            self.remove_roi_pixel(e)
        else:
            logger.debug('GWViewImageROI.remove_roi for non-PIXEL types')
            items = self.scene().items(self.mapToScene(e.pos()))
            roisel = [o for o in self.list_of_rois if o.scitem in items]
            logger.debug('remove_roi list of ROIs at point: %s' % str(roisel))
            for o in roisel:
                self.scene().removeItem(o.scitem)
                self.list_of_rois.remove(o)


    def delete_selected_roi(self):
        roisel = [o for o in self.list_of_rois if o.scitem.pen().color() == QCOLOR_SEL]
        logger.debug('delete_selected_roi: %s' % str(roisel))
        for o in roisel:
            self.scene().removeItem(o.scitem)
            self.list_of_rois.remove(o)


    def remove_roi_pixel(self, e):
        scpos = self.mapToScene(e.pos())
        iscpos = QPoint(int(scpos.x()), int(scpos.y()))
        self._iscpos_old = iscpos
        for o in self.list_of_rois:
            if iscpos == o.pos:
                self.scene().removeItem(o.scitem)
                self.list_of_rois.remove(o)


    def add_roi(self, e):
        """add ROI on mouthPressEvent"""

        scpos = self.mapToScene(e.pos())
        logger.debug('GWViewImageROI.add_roi scene x=%.1f y=%.1f'%(int(scpos.x()), int(scpos.y())))

        if self.roi_type == roiu.PIXEL:
             self.add_roi_pixel(scpos)
        else:
             self.add_roi_to_scene(scpos)


    def add_roi_pixel(self, scpos):
        """add ROIPixel on mouthPressEvent - special treatment for pixels...On/Off at mouthMoveEvent"""
        iscpos = QPoint(int(scpos.x()), int(scpos.y()))
        self._iscpos_old = iscpos
        for o in self.list_of_rois:
            if iscpos == o.pos:
                return
        self.add_roi_to_scene(iscpos)


    def add_roi_to_scene(self, scpos):
        o = roiu.select_roi(self.roi_type, view=self, pos=scpos)
        if o is None:
            logger.warning('ROI of type %d is undefined' % self.roi_type) # roiu.dict_roi_type_name[self.roi_type])
            return
        o.add_to_scene()
        self.list_of_rois.append(o)
        self.roi_active = o


    def set_pixmap_from_arr(self, arr, set_def=True, amin=None, amax=None, frmin=0.00001, frmax=0.99999):
        """Input array is scailed by color table. If color table is None arr set as is."""

        GWViewImage.set_pixmap_from_arr(self, arr, set_def, amin, amax, frmin, frmax)

        image = self.qimage # QImage
        pixmap = self.qpixmap # QPixmap

        arr = ct.image_to_arrcolors(self.qimage, channels=4)
        print(info_ndarr(arr, 'XXXX image_to_arrcolors:'))

#        arrB = ct.pixmap_channel(self.qpixmap, channel=0)
#        arrG = ct.pixmap_channel(self.qpixmap, channel=1)
#        arrR = ct.pixmap_channel(self.qpixmap, channel=2)
        arrA = ct.pixmap_channel(self.qpixmap, channel=3)

        print(info_ndarr(arrA, 'XXXX alpha channel'))

        #mask = pixmap.createMaskFromColor()
        #arr = pixmap_to_arrcolors(pixmap, channels=4)

if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
