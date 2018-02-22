#!@PYTHON@
"""
Module :py:class:`GUDragFactory` contains factory methods for draggable objects
===============================================================================

Created on 2016-10-09 by Mikhail Dubrovin
"""
#-----------------------------

#import os
#import math
#import graphqt.GUDragBase as drb
from PyQt4 import QtGui#, QtCore
from PyQt4.QtCore import Qt
from graphqt.GUDragPoint   import GUDragPoint
from graphqt.GUDragRect    import GUDragRect
#from graphqt.GUDragLine    import GUDragLine
#from graphqt.GUDragCirc    import GUDragCirc
#from graphqt.GUDragEllipse import GUDragEllipse
#from graphqt.GUDragPoly    import GUDragPoly
#from graphqt.GUDragWedge   import GUDragWedge

from graphqt.GUDragBase    import DELETE 
#-----------------------------

POINT= 0
LINE = 1
RECT = 2
CIRC = 3
POLY = 4
WEDG = 5

drag_types = ( POINT,   LINE,   RECT,   CIRC,   POLY,   WEDG)
drag_names = ('POINT', 'LINE', 'RECT', 'CIRC', 'POLY', 'WEDG')

dic_drag_type_to_name = dict(zip(drag_types, drag_names))
dic_drag_name_to_type = dict(zip(drag_names, drag_types))

#-----------------------------

#class GUDragFactory(GUDragBase) :
#    
#    def __init__(self) :
#        GUView.__init__(self, parent=None)
#        pass

#-----------------------------

def add_item(type, obj, parent=None, scene=None,\
             brush=QtGui.QBrush(),\
             pen=QtGui.QPen(Qt.blue, 2, Qt.SolidLine)) :

    brush_w=QtGui.QBrush(Qt.white, Qt.SolidPattern)

    if   type == POINT : return GUDragPoint(obj, parent, scene, brush_w, pen, orient='r', rsize=8)
    elif type == RECT  : return GUDragRect (obj, parent, scene, brush, pen)
    elif type == LINE  : return None # GUDragLine(view, points)
    elif type == CIRC  : return None # GUDragCirc(view, points)
    elif type == POLY  : return None # GUDragPoly(view, points)
    elif type == WEDG  : return None # GUDragWedge(view, points)
    else : 
        print 'WARNING: Type %s is unknown' % type
        return None

#-----------------------------
if __name__ == "__main__" :
    print 'Self test is not implemented...'
    print 'use > python GUViewImageWithShapes.py'
#-----------------------------
