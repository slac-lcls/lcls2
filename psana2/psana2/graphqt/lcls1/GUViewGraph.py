#!@PYTHON@
"""
Class :py:class:`GUViewGraph` is a QWidget for interactive image
================================================================

Usage ::
    from graphqt.GUViewGraph import GUViewGraph
    #...
    w = GUViewGraph(None, arr, origin='UL', scale_ctl='HV', coltab=ctab)

Created on 2016-09-09 by Mikhail Dubrovin
"""

#import os
#import math
#import math
from math import floor
from graphqt.GUViewAxes import *
from PyQt4.QtCore import QRectF, QPointF

class GUViewGraph(GUViewAxes) :
    
    def __init__(self, parent=None, rectax=QtCore.QRectF(0, 0, 1, 1), origin='DL', scale_ctl='HV', rulers='TBLR',\
                 margl=None, margr=None, margt=None, margb=None) :

        #xmin, xmax = np.amin(x), np.amax(x) 
        #ymin, ymax = np.amin(y), np.amax(y) 
        #w, h = xmax-xmin, ymax-ymin

        GUViewAxes.__init__(self, parent, rectax, origin, scale_ctl, rulers, margl, margr, margt, margb)

        #self.scene().removeItem(self.raxi)
        #self.scene().removeItem(self.rori)

        self.pen1=QtGui.QPen(Qt.white, 0, Qt.DashLine)
        self.pen2=QtGui.QPen(Qt.black, 0, Qt.DashLine)
        #pen1.setCosmetic(True)
        #pen2.setCosmetic(True)

        ptrn = [10,10]
        self.pen1.setDashPattern(ptrn)
        self.pen2.setDashPattern(ptrn)
        #print 'pen1.dashPattern()', self.pen1.dashPattern()
        self.pen2.setDashOffset(ptrn[0])
        self.cline1i = self.scene().addLine(QtCore.QLineF(), self.pen1)
        self.cline2i = self.scene().addLine(QtCore.QLineF(), self.pen2)
        self.cline1i.setZValue(10)
        self.cline2i.setZValue(10)
        
        self.lst_items = []


    def set_style(self) :
        GUViewAxes.set_style(self)
        self.setWindowTitle("GUViewGraph")
        #w.setContentsMargins(-9,-9,-9,-9)
        #self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        #self.setAttribute(Qt.WA_TranslucentBackground)


#    def display_pixel_pos(self, e):
#        p = self.mapToScene(e.pos())
#        ix, iy = floor(p.x()), floor(p.y())
#        v = None
#        if ix<0\
#        or iy<0\
#        or ix>arr.shape[0]-1\
#        or iy>arr.shape[1]-1 : pass
#        else : v = self.arr[ix,iy]
#        vstr = 'None' if v is None else '%.1f' % v 
#        #print 'mouseMoveEvent, current point: ', e.x(), e.y(), ' on scene: %.1f  %.1f' % (p.x(), p.y()) 
#        self.setWindowTitle('GUViewGraph x=%d y=%d v=%s' % (ix, iy, vstr))
#        #return ix, iy, v


#------------------------------

    def mouseMoveEvent(self, e):
        GUViewAxes.mouseMoveEvent(self, e) # calls display_pixel_pos(e)
        p = self.mapToScene(e.pos())
        x, y = p.x(), p.y()
        if x<self.rectax.left() : return
        y1, y2 = self.rectax.bottom(), self.rectax.top()
        self.cline1i.setLine(x, y1, x, y2)
        self.cline2i.setLine(x, y1, x, y2)


    def enterEvent(self, e) :
    #    print 'enterEvent'
        GUViewAxes.enterEvent(self, e)
        self.cline1i.setPen(self.pen1)
        self.cline2i.setPen(self.pen2)
        

    def leaveEvent(self, e) :
    #    print 'leaveEvent'
        GUViewAxes.leaveEvent(self, e)
        self.cline1i.setPen(QtGui.QPen())
        self.cline2i.setPen(QtGui.QPen())

    def key_usage(self) :
        return 'Keys: TBD ???'\
               '\n  ESC - exit'\
               '\n  R - reset original graphic ????'\
               '\n  N - set new graphic ?????'\
               '\n'


    def keyPressEvent(self, e) :
        #print 'keyPressEvent, key=', e.key()         
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_R : 
            print 'Reset original size'
            self.set_view()
            self.update_my_scene()

        elif e.key() == Qt.Key_N : 
            print 'Set new pixel map'
            s = self.pmi.pixmap().size()
            self.set_pixmap_random((s.width(), s.height()))

        else :
            print self.key_usage()

    def _add_path_to_scene(self, path, pen=QtGui.QPen(Qt.yellow), brush=QtGui.QBrush()) :
        item = self.scene().addPath(path, pen, brush)
        self.lst_items.append(item)
        self.update_my_scene()


    def add_graph(self, x, y, pen=QtGui.QPen(Qt.yellow), brush=QtGui.QBrush()) :
        path = QtGui.QPainterPath(QPointF(x[0],y[0]))
        #polygon = QtGui.QPolygonF([QtCore.QPointF(px,py) for px,py in zip(x, y)])
        #path.addPolygon(polygon)
        #path.moveTo(QPointF(x[0],y[0]))
        for px,py in zip(x, y)[:-1] :
            path.lineTo(QPointF(px,py))
        self._add_path_to_scene(path, pen, brush)


    def add_points(self, x, y, pen=QtGui.QPen(Qt.black), brush=QtGui.QBrush(Qt.cyan), fsize=0.01) :
        rect = self.rectax
        rx, ry = fsize*rect.width(), fsize*rect.height()
        path = QtGui.QPainterPath()
        for px,py in zip(x, y) :
            if None in (px,py) : continue
            path.addEllipse(QPointF(px,py), rx, ry)
        self._add_path_to_scene(path, pen, brush)


    def remove_all_graphs(self) :
        for item in self.lst_items :
            self.scene().removeItem(item)
            self.lst_items.remove(item)
        self.scene().update()

#------------------------------
 
    def closeEvent(self, e):
        #print 'GUViewHist.closeEvent'
        self.lst_items = []
        #self.lst_hbins = []
        GUViewAxes.closeEvent(self, e)
        #print '%s.closeEvent' % self._name

#------------------------------ 

#    def __del__(self) :
#        self.remove_all_graphs()

#-----------------------------

if __name__ == "__main__" :

    import sys
    import numpy as np
    shape = (1000,)
    x = np.array(range(shape[0]))
    #y = np.random.random(shape)
    y1 = np.sin(x)
    y2 = np.random.random(shape)

    rectax=QtCore.QRectF(0, -1.25, 1000, 2.5)    

    app = QtGui.QApplication(sys.argv)
    w = GUViewGraph(None, rectax, origin='DL', scale_ctl='HV', rulers='UDLR',\
                    margl=0.12, margr=0.10, margt=0.06, margb=0.06)
    w.add_graph(x, y1, QtGui.QPen(Qt.blue), brush=QtGui.QBrush())
    w.add_graph(x, y2, QtGui.QPen(Qt.red),  brush=QtGui.QBrush())
    w.show()
    app.exec_()

#-----------------------------
