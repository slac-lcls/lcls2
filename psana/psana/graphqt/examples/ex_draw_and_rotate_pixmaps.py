#!/usr/bin/env python

"""Bare bones interactive example showing how to transform QGraphicsPixmapItem on QGraphicsScene.
   Created on 2023-06-06 by Mikhail Dubrovin
"""

from math import degrees, atan2
import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene,\
     QGraphicsRectItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsEllipseItem, QGraphicsPixmapItem
from PyQt5.QtGui import QColor, QPen, QBrush, QPolygonF, QPixmap, QImage
from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF

import psana.graphqt.ColorTable as ct
import psana.pyalgos.generic.NDArrGenerators as ag

class TestQGraphicsView(QGraphicsView):

    KEY_USAGE = 'Keys:'\
            '\n  ESC - exit'\
            '\n  P - position'\
            '\n  R - rotation'\
            '\n  S - scale'\
            '\n  or click on mouse left for rotation'\
            '\n'

    def keyPressEvent(self, e):

        if e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_P:
            pos = itpm2.pos()
            print('current position: %s' % str(pos))
            dx = -10 if pos.x() > 30 else 10
            dy =  -5 if pos.y() > 50 else 5
            dp = QPointF(-50,-50) if pos.manhattanLength()>50 else QPointF(10,10)
            itpm2.setPos(pos + dp)  # increment relative to current position !!! applys setTransformOriginPoint

        elif e.key() == Qt.Key_R:
            angle = itpm1.rotation()
            print('current rotation angle: %.2f' % angle)
            itpm1.setRotation(angle+10)

        elif e.key() == Qt.Key_S:
            scale = itpm1.scale()
            print('current scale: %.2f' % scale)
            scale = 1 if scale>2 else scale*1.2
            itpm1.setScale(scale)

        print(self.KEY_USAGE)

    def mousePressEvent(self, e):
        item = itpm1
        QGraphicsView.mousePressEvent(self, e)
        scpoint = self.mapToScene(e.pos())
        #dp = scpoint - item.rect().center()
        dp = scpoint - item.boundingRect().center()
        angle_deg = degrees(atan2(dp.y(), dp.x()))
        print('TestQGraphicsView.mousePressEvent ellips new rotation angle (degree): %.2f' % angle_deg)
        item.setRotation(angle_deg)


app = QApplication(sys.argv)

pen1 = QPen(Qt.red, 1, Qt.DashLine)
pen1.setCosmetic(True)
rs = QRectF(0, 0, 500, 500)

shape=(100,200)
arr = ag.random_standard(shape, mu=0, sigma=10)
pixmap = ct.qpixmap_from_arr(arr, ctable=ct.next_color_table(3))  # QPixmap()

s = QGraphicsScene(rs)
w = TestQGraphicsView(s)
w.fitInView(rs, Qt.KeepAspectRatio)

# Draw scene rect
itr0 = s.addRect(rs, pen1, QBrush())

# Draw rotated rect using regular QGraphicsRectItem definition
#itr2 = QGraphicsRectItem(QRectF(30, 10, 40, 80))
#itr2.setTransformOriginPoint(itr2.rect().center())
#itr2.setPen(pen2)
#itr2.setBrush(QBrush(Qt.yellow))
#itr2.setRotation(-60)
#s.addItem(itr2)

itpm1 = QGraphicsPixmapItem(pixmap)
itpm1.setOffset(QPointF(100,50))
itpm1.setTransformOriginPoint(itpm1.boundingRect().center())
itpm1.setRotation(-20)
s.addItem(itpm1)
#itpm1 = s.addPixmap(pixmap)

itpm2 = QGraphicsPixmapItem(pixmap)
itpm2.setOffset(QPointF(200,200))
itpm2.setTransformOriginPoint(itpm2.boundingRect().center())
itpm2.setRotation(30)
s.addItem(itpm2)


w.setWindowTitle(sys.argv[0].rsplit('/')[-1])
w.setGeometry(100, 20, 600, 600)
print(w.KEY_USAGE)
w.show()

app.exec_()

# EOF
