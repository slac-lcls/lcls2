#------------------------------
"""
:py:class:`QWGraphicsRectItem` - derived from QWGraphicsRectItem to intercept events
=========================================================================================

Usage::

    # Import
    from psana.graphqt.QWGraphicsRectItem import WGraphicsRectItem

    # Methods - see test

See:
    - :py:class:`QWGraphicsRectItem`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-06-12 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-16
"""
#------------------------------
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal # , QPoint, QEvent, QMargins
from PyQt5.QtGui import QCursor


#from PyQt4 import QtGui#, QtCore
#from PyQt4.QtGui import QGraphicsRectItem
#from PyQt4.QtCore import Qt

#-----------------------------

class QWGraphicsRectItem(QtWidgets.QGraphicsRectItem) :    
    #                  QRectF, QGraphicsItem, QGraphicsScene


    event_on_rect = pyqtSignal('QString')

    def __init__(self, rect, parent=None, scene=None) :
        #QtWidgets.QGraphicsRectItem.__init__(self, rect, parent, scene)
        QtWidgets.QGraphicsRectItem.__init__(self, rect, parent)
        if scene is not None: scene.addItem(self)

        self.setAcceptHoverEvents(True)
        self.setAcceptTouchEvents(True)
        #self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setCursorHover()
        self.setCursorGrab()


    def setCursorHover(self, cursor=Qt.CrossCursor) :
        #QtWidgets.QGraphicsRectItem.setCursor(self, cursor)
        self.hover_cursor = cursor


    def setCursorGrab(self, cursor=Qt.SizeAllCursor) : # Qt.ClosedHandCursor) :
        self.grub_cursor = cursor


    def hoverEnterEvent(self, e) :
        #print('hoverEnterEvent')
        QtWidgets.QGraphicsRectItem.hoverEnterEvent(self, e)
        QtWidgets.QApplication.setOverrideCursor(QCursor(self.hover_cursor))


    def hoverLeaveEvent(self, e) :
        #print('hoverLeaveEvent')
        QtWidgets.QGraphicsRectItem.hoverLeaveEvent(self, e)
        #QtWidgets.QApplication.setOverrideCursor(QCursor(self.hover_cursor))
        QtWidgets.QApplication.restoreOverrideCursor()
        

    def hoverMoveEvent(self, e) :
        #print('hoverMoveEvent')
        QtWidgets.QGraphicsRectItem.hoverMoveEvent(self, e)


    def mouseMoveEvent(self, e) :
        print('QWGraphicsRectItem: mouseMoveEvent')
        QtWidgets.QGraphicsRectItem.mouseMoveEvent(self, e)


    def mousePressEvent(self, e) :
        #print('mousePressEvent, at point: ', e.pos() #e.globalX(), e.globalY())
        QtWidgets.QGraphicsRectItem.mousePressEvent(self, e)
        QtWidgets.QApplication.setOverrideCursor(QCursor(self.grub_cursor))


    def mouseReleaseEvent(self, e) :
        """ !!! This method does not receive control because module is distracted before...
        """
        #print('mouseReleaseEvent')
        QtWidgets.QGraphicsRectItem.mouseReleaseEvent(self, e)
        #QtWidgets.QApplication.setOverrideCursor(QCursor(self.hover_cursor))
        QtWidgets.QApplication.restoreOverrideCursor()


#    def mouseDoubleClickEvent(self, e) :
#        QtWidgets.QGraphicsRectItem.hoverLeaveEvent(self, e)
#        print('mouseDoubleClickEvent, at point: ', e.pos() #e.globalX(), e.globalY())


#    def wheelEvent(self, e) :
#        QtWidgets.QGraphicsRectItem.wheelEvent(self, e)
#        #print('wheelEvent, at point: ', e.pos() #e.globalX(), e.globalY() )


    def emit_signal(self, msg='click') :
        #self.emit(QtCore.SIGNAL('event_on_rect(QString)'), msg)
        self.event_on_rect.emit(msg)
        #print(msg)

#-----------------------------
