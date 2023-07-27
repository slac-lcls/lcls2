#!/usr/bin/env python

from psana.graphqt.GWRuler import *

if __name__ == "__main__":

    import sys
    from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

    app = QApplication(sys.argv)
    rs = QRectF(0, 0, 100, 100)
    ro = QRectF(-1, -1, 3, 2)
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 600)
    v._origin_u = True
    v._origin_l = True

    print('screenGeometry():', app.desktop().screenGeometry())
    print('scene rect=', s.sceneRect())

    v.fitInView(rs, Qt.KeepAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio

    s.addRect(rs, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.yellow))
    s.addRect(ro, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.red))

    ruler1 = GWRuler(v, 'L', label_rot=30)
    ruler2 = GWRuler(v, 'D')
    ruler3 = GWRuler(v, 'U')
    ruler4 = GWRuler(v, 'R')

    v.setWindowTitle("My window")
    v.setContentsMargins(0,0,0,0)
    v.show()
    app.exec_()

    ruler1.remove()
    ruler2.remove()
    ruler3.remove()
    ruler4.remove()

    del app

# EOF
