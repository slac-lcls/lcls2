"""
Module :py:class:`DragFactory` contains factory methods for draggable objects
===============================================================================

Created on 2016-10-09 by Mikhail Dubrovin
"""
#-----------------------------
import logging
logger = logging.getLogger(__name__)

from psana.graphqt.DragTypes   import * # defined in DragPoint <- DragBase <- DragTypes
from psana.graphqt.DragPoint   import * # DragPoint, DragBase, Qt, QPen, QBrush
from psana.graphqt.DragRect    import DragRect
from psana.graphqt.DragPoly    import DragPoly 
from psana.graphqt.DragEllipse import DragEllipse 
#from psana.graphqt.DragLine    import DragLine
#from psana.graphqt.DragCirc    import DragCirc
#from psana.graphqt.DragEllipse import DragEllipse
#from psana.graphqt.DragWedge   import DragWedge

#-----------------------------

def add_item(type, obj, parent=None, scene=None,\
             brush=QBrush(),\
             pen=QPen(Qt.blue, 2, Qt.SolidLine)) :

    brush_w=QBrush(Qt.white, Qt.SolidPattern)

    logger.debug('DragFactory add_item %s' % dic_drag_type_to_name[type])

    if   type == POINT  : return DragPoint  (obj, parent, scene, brush_w, pen, pshape='r', rsize=7)
    elif type == RECT   : return DragRect   (obj, parent, scene, brush,   pen)
    elif type == LINE   : return DragPoint  (obj, parent, scene, brush_w, pen, pshape='v', rsize=9)
    elif type == CIRC   : return DragPoint  (obj, parent, scene, brush_w, pen, pshape='c', rsize=40)
    elif type == WEDG   : return DragPoint  (obj, parent, scene, brush_w, pen, pshape='w', rsize=8)
    elif type == POLY   : return DragPoly   (obj, parent, scene, brush_w, pen) #, pshape='x', rsize=11)
    elif type == ELLIPSE: return DragEllipse(obj, parent, scene, brush_w, pen) #, pshape='x', rsize=11)
                          #return None # DragWedge(view, points) #return None # DragPoly(view, points)
    else : 
        logger.warning('WARNING: Type %s is unknown' % type)
        return None

#-----------------------------
if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')
#-----------------------------
