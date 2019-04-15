"""
Module :py:class:`DragTypes` defines types of draggable objects
===============================================================

Created on 2019-03-06 by Mikhail Dubrovin
"""
#-----------------------------

UNDEF = 0
POINT = 1
LINE = 2
RECT = 3
CIRC = 4
POLY = 5
WEDG = 6

drag_types = ( UNDEF,   POINT,   LINE,   RECT,   CIRC,   POLY,   WEDG)
drag_names = ('UNDEF', 'POINT', 'LINE', 'RECT', 'CIRC', 'POLY', 'WEDG')

dic_drag_type_to_name = dict(zip(drag_types, drag_names))
dic_drag_name_to_type = dict(zip(drag_names, drag_types))

#-----------------------------
