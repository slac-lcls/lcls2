"""
Module :py:class:`DragFactory` contains factory methods for draggable objects
===============================================================================

Created on 2016-10-09 by Mikhail Dubrovin
"""
#-----------------------------
import logging
logger = logging.getLogger(__name__)

from psana.graphqt.DragPoint   import * # DragPoint, DragBase, Qt, QPen, QBrush
from psana.graphqt.DragRect    import DragRect
#from psana.graphqt.DragLine    import DragLine
#from psana.graphqt.DragCirc    import DragCirc
#from psana.graphqt.DragEllipse import DragEllipse
#from psana.graphqt.DragPoly    import DragPoly
#from psana.graphqt.DragWedge   import DragWedge

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

def add_item(type, obj, parent=None, scene=None,\
             brush=QBrush(),\
             pen=QPen(Qt.blue, 2, Qt.SolidLine)) :

    brush_w=QBrush(Qt.white, Qt.SolidPattern)

    logger.debug('DragFactory add_item %s' % dic_drag_type_to_name[type])

    if   type == POINT : return DragPoint(obj, parent, scene, brush_w, pen, orient='r', rsize=8)
    elif type == RECT  : return DragRect (obj, parent, scene, brush, pen)
    elif type == LINE  : return None # DragLine(view, points)
    elif type == CIRC  : return None # DragCirc(view, points)
    elif type == POLY  : return None # DragPoly(view, points)
    elif type == WEDG  : return None # DragWedge(view, points)
    else : 
        logger.warning('WARNING: Type %s is unknown' % type)
        return None

#-----------------------------
if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')
#-----------------------------
