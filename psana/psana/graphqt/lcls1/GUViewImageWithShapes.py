#!@PYTHON@
"""
Class :py:class:`GUViewImageWithShapes` is a QWidget for interactive image
==========================================================================

Usage ::

    import sys
    import numpy as np
    arr = np.random.random((1000, 1000))
    app = QtGui.QApplication(sys.argv)
    w = GUViewImageWithShapes(None, arr, origin='UL', scale_ctl='HV', rulers='UDLR')
    w.show()
    app.exec_()

Created on 2016-10-10 by Mikhail Dubrovin
"""
#------------------------------

#import os
#import math
#import math
import graphqt.ColorTable as ct
from graphqt.GUViewImage import *
from graphqt.GUDragFactory import * # add_item, POINT, RECT, ..., DELETE from GUDragBase

#------------------------------

class GUViewImageWithShapes(GUViewImage) :
    
    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV', rulers='UDLR',\
                 margl=None, margr=None, margt=None, margb=None) :
        GUViewImage.__init__(self, parent, arr, coltab, origin, scale_ctl, rulers, margl, margr, margt, margb)

        self.add_request=None
        self.lst_drag_items = []


    def setShapesEnabled(self, is_enabled=True):
        for item in self.lst_drag_items : item.setEnabled(is_enabled)


    def selected_item(self):
        for item in self.lst_drag_items :
            if item.isSelected() : return item
        return None


    def item_marked_to_delete(self):
        for item in self.lst_drag_items :
            if item.mode() == DELETE : return item
        return None


    def mousePressEvent(self, e):
        sp = self.mapToScene(e.pos())
        #print '%s.mousePressEvent' % self.__class__.__name__, e.pos(),\
        #      ' scene x=%.1f y=%.1f' % (sp.x(), sp.y()), ' button:', e.button()
        GUViewImage.mousePressEvent(self, e) # to select/deselect items

        if self.add_request is not None :
            self.set_scale_control('')

            print 'process request to add %s' % dic_drag_type_to_name[self.add_request]
            parent = None if self.add_request == POINT else self
            item = add_item(self.add_request, sp, parent, scene=self.scene())
            item.setZValue(100 if self.add_request == POINT else 30)
            item.setSelected(True)
            self.lst_drag_items.append(item)
            return

        item = self.item_marked_to_delete()
        #print 'item_marked_to_delete', item
        if item is not None :
            self.delete_item(item)
            return

        if self.selected_item() is None :
            self.set_scale_control(scale_ctl='HV')
            GUViewImage.mousePressEvent(self, e) # to move pixmap on click
        else :
            self.set_scale_control(scale_ctl='')


#    def mouseMoveEvent(self, e):
#        #print '%s.mouseMoveEvent' % self.__class__.__name__, e.pos()
#        GUViewImage.mouseMoveEvent(self, e)


    def mouseReleaseEvent(self, e):
        GUViewImage.mouseReleaseEvent(self, e)
        #print '%s.mouseReleaseEvent' % self.__class__.__name__, e.pos()

        if self.add_request is not None :
            self.setShapesEnabled()
            self.add_request = None

#------------------------------
 
    def closeEvent(self, e):
        GUViewImage.closeEvent(self, e)
        #print 'GUViewImage.closeEvent' # % self._name

#------------------------------

    def delete_item(self, item) :
        if item is None : return
        
        self.set_scale_control('')
        self.scene().removeItem(item)
        self.lst_drag_items.remove(item)
        self.setShapesEnabled()
#------------------------------

    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  M - add point'\
               '\n  A - add rect'\
               '\n  L - add line TBD'\
               '\n  P - add polyline TBD'\
               '\n  C - add circle TBD'\
               '\n  W - add wedge TBD'\
               '\n  S - switch interactive session between scene and shapes'\
               '\n  D - delete selected item'\
               '\n'

#------------------------------

    def keyPressEvent(self, e) :
        #print 'keyPressEvent, key=', e.key()         
        # POINT,   LINE,   RECT,   CIRC,   POLY,   WEDG
        GUViewImage.keyPressEvent(self, e) # uses Key_R and Key_N

        d = {Qt.Key_M : POINT, Qt.Key_A : RECT, Qt.Key_L : LINE,\
             Qt.Key_C : CIRC,  Qt.Key_P : POLY, Qt.Key_W : WEDG}

        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() in d.keys() :
            type = d[e.key()]
            self.add_request = type # e.g. RECT
            print 'click-drag-release mouse button on image to add %s' % dic_drag_type_to_name[type]
            self.setShapesEnabled(False)

        elif e.key() == Qt.Key_D : 
            print 'delete selected item'
            self.delete_item(self.selected_item())

        elif e.key() == Qt.Key_S : 
            print 'switch interactive session between scene and shapes'
            
            if self.scale_control() :
                self.set_scale_control(scale_ctl='')
                self.setShapesEnabled()
            else :
                self.set_scale_control(scale_ctl='HV')
                self.setShapesEnabled(False)
        else :
            print self.key_usage()

#------------------------------

if __name__ == "__main__" :

    import sys
    import numpy as np; global np
    arr = np.random.random((1000, 1000))
    app = QtGui.QApplication(sys.argv)
    w = GUViewImageWithShapes(None, arr, origin='UL', scale_ctl='HV', rulers='DL',\
                              margl=0.12, margr=0.01, margt=0.01, margb=0.06)
    w.show()
    app.exec_()

#------------------------------
