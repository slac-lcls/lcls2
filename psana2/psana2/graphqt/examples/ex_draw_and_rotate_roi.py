#!/usr/bin/env python

"""Bare bones interactive example showing how to transform QGraphics*Item on QGraphicsScene.
   Created on 2023-06-06 by Mikhail Dubrovin
"""

from math import degrees, atan2
import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene,\
     QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsEllipseItem
from PyQt5.QtGui import QColor, QPen, QBrush, QPolygonF, QPixmap
from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF

class TestQGraphicsView(QGraphicsView):

    KEY_USAGE = 'Keys:'\
            '\n  ESC - exit'\
            '\n  P - position'\
            '\n  R - rotation'\
            '\n  S - scale'\
            '\n  or click on mouse left for next rotation angle for ellipse\n'

    def keyPressEvent(self, e):
        item = self.ite2

        if e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_P:
            pos = item.pos()
            print('current position: %s' % str(pos))
            dp = QPointF(-50,-50) if pos.manhattanLength()>50 else QPointF(10,10)
            item.setPos(pos + dp) # increment relative to current position !!! applys setTransformOriginPoint

        elif e.key() == Qt.Key_R:
            angle = item.rotation()+10
            print('set new rotation angle to %.2f' % angle)
            item.setRotation(angle)

        elif e.key() == Qt.Key_S:
            scale = item.scale()
            print('current scale: %.2f' % scale)
            scale = 1 if scale>2 else scale*1.2
            item.setScale(scale)

        print(self.KEY_USAGE)

    def mousePressEvent(self, e):
        item = self.ite2
        QGraphicsView.mousePressEvent(self, e)
        scpoint = self.mapToScene(e.pos())
        dp = scpoint - item.rect().center()
        angle_deg = degrees(atan2(dp.y(), dp.x()))
        print('TestQGraphicsView.mousePressEvent ellips new rotation angle (degree): %.2f' % angle_deg)
        item.setRotation(angle_deg)


app = QApplication(sys.argv)

pen1 = QPen(Qt.red, 1, Qt.DashLine)
pen1.setCosmetic(True)
pen2 = QPen(Qt.black, 1, Qt.SolidLine)
pen2.setCosmetic(True)
pen3 = QPen(Qt.red, 2, Qt.SolidLine)
line = QLineF(QPointF(10,5), QPointF(90,95))
poly = QPolygonF((QPointF(15,60), QPointF(20,65), QPointF(20,85), QPointF(25,95)))
brushy = QBrush(Qt.yellow)
brush0 = QBrush()
r = QRectF(30, 10, 40, 80)
rs = QRectF(0, 0, 100, 100)
#pixmap = QPixmap()

s = QGraphicsScene(rs)
w = TestQGraphicsView(s)
w.fitInView(rs, Qt.KeepAspectRatio)

# Draw line, rects, ellipse, and polygon uing convenience methods
itl0 = s.addLine(line, pen3)
itr0 = s.addRect(rs, pen1, brush0)
itr1 = s.addRect(r, pen2, brushy)
ite1 = s.addEllipse(r, pen2, brushy)
itp1 = s.addPolygon(poly, pen2, brushy)
#itm1 = s.addPixmap(pixmap)

# Draw rotated rect using regular QGraphicsRectItem definition
itr2 = QGraphicsRectItem(r)
itr2.setTransformOriginPoint(itr2.rect().center())
itr2.setPen(pen2)
itr2.setRotation(-60)
s.addItem(itr2)

# Draw rotated ellipse using regular QGraphicsEllipseItem definition
ite2 = QGraphicsEllipseItem(r)
ite2.setPen(pen2)
ite2.setBrush(brush0)
ite2.setTransformOriginPoint(ite2.rect().center())
ite2.setRotation(60)
s.addItem(ite2)

# preserve this QGraphicsEllipseItem to use in TestQGraphicsView widget
w.ite2 = ite2

# Draw rotated polygon using regular QGraphicsPolygonItem definition
itp2 = QGraphicsPolygonItem(poly)
itp2.setPen(pen2)
itp2.setBrush(brush0)
itp2.setTransformOriginPoint(itp2.boundingRect().center())
itp2.setRotation(60)
s.addItem(itp2)

w.setWindowTitle(sys.argv[0].rsplit('/')[-1])
w.setGeometry(100, 20, 600, 600)
print(w.KEY_USAGE)
w.show()

app.exec_()

# EOF
