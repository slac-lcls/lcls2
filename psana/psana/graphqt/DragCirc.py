"""
Class :py:class:`DragCirc` - for draggable circle
===================================================

Created on 2016-10-09 by Mikhail Dubrovin
"""
#import math

#from graphqt.DragBase import DragBase
from psana.graphqt.DragPoint import * # DragPoint, DragBase, Qt, QPen, QBrush

#-----------------------------

class DragCirc(DragBase) :
    
    def __init__(self) :
        DragBase.__init__(self, view, points)
        pass

#-----------------------------

if __name__ == "__main__" :
    print('Self test is not implemented...')
    print('use > python FWViewImageShapes.py')

#-----------------------------
