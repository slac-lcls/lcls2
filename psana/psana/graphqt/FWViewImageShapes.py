"""
Class :py:class:`FWViewImageShapes` is a QWidget for interactive image
==========================================================================

Usage ::

    import sys
    import numpy as np
    arr = np.random.random((1000, 1000))
    app = QApplication(sys.argv)
    w = FWViewImageShapes(None, arr, origin='UL', scale_ctl='HV', rulers='UDLR')
    w.show()
    app.exec_()

Created on 2016-10-10 by Mikhail Dubrovin
"""
#------------------------------

#import os
#import math
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtCore import Qt #, QPointF#, QRect, QRectF

import psana.graphqt.ColorTable as ct
from psana.graphqt.FWViewImage import FWViewImage
from psana.graphqt.DragFactory import add_item, POINT, LINE, RECT, CIRC, POLY, WEDG,\
                                      dic_drag_type_to_name, dic_drag_name_to_type
from psana.graphqt.DragBase import FROZEN, ADD, MOVE, EDIT, DELETE

#------------------------------

class FWViewImageShapes(FWViewImage) :
    
    def __init__(self, parent=None, arr=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 origin='UL', scale_ctl='HV') :

        FWViewImage.__init__(self, parent, arr, coltab, origin, scale_ctl)

        self.add_request=None
        self.lst_drag_items = []
        self.scale_ctl_normal = scale_ctl


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
        scpoint = self.mapToScene(e.pos())
        logger.debug('FWViewImageShapes.mousePressEvent but=%d %s scene x=%.1f y=%.1f'%\
                     (e.button(), str(e.pos()), scpoint.x(), scpoint.y())) # self.__class__.__name__

        FWViewImage.mousePressEvent(self, e) # to select/deselect items

        if self.add_request is not None :
            self.set_scale_control('')

            logger.debug('process request to add %s' % dic_drag_type_to_name[self.add_request])
            parent = None if self.add_request == POINT else self
            #parent = None if self.add_request == RECT else self
            item = add_item(self.add_request, scpoint, parent, scene=self.scene())
            item.setZValue(100 if self.add_request == POINT else 30)
            item.setSelected(True)
            self.lst_drag_items.append(item)
            return

        item = self.item_marked_to_delete()
        #print('item_marked_to_delete', item)
        if item is not None :
            self.delete_item(item)
            return

        if self.selected_item() is None :
            self.set_scale_control(scale_ctl=self.scale_ctl_normal)
            #FWViewImage.mousePressEvent(self, e) # to move pixmap on click
        else :
            self.set_scale_control(scale_ctl='')


#    def mouseMoveEvent(self, e):
#        print('%s.mouseMoveEvent' % self.__class__.__name__, e.pos())
#        FWViewImage.mouseMoveEvent(self, e)


    def mouseReleaseEvent(self, e):
        FWViewImage.mouseReleaseEvent(self, e)
        #logger.debug('%s.mouseReleaseEvent pos: %s' % (self.__class__.__name__, str(e.pos())))

        if self.add_request is not None :
            self.setShapesEnabled()
            self.add_request = None

#------------------------------
 
    def closeEvent(self, e):
        FWViewImage.closeEvent(self, e)
        #print('FWViewImage.closeEvent' # % self._name)

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
        #print('keyPressEvent, key=', e.key())        
        # POINT,   LINE,   RECT,   CIRC,   POLY,   WEDG
        #FWViewImage.keyPressEvent(self, e) # uses Key_R and Key_N

        d = {Qt.Key_M : POINT, Qt.Key_A : RECT, Qt.Key_L : LINE,\
             Qt.Key_C : CIRC,  Qt.Key_P : POLY, Qt.Key_W : WEDG}

        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() in d.keys() :
            type = d[e.key()]
            self.add_request = type # e.g. RECT
            logger.info('click-drag-release mouse button on image to add %s' % dic_drag_type_to_name[type])
            self.setShapesEnabled(False)

        elif e.key() == Qt.Key_D : 
            logger.info('delete selected item')
            self.delete_item(self.selected_item())

        elif e.key() == Qt.Key_S : 
            logger.info('switch interactive session between scene and shapes')
            
            if self.scale_control() :
                self.set_scale_control(scale_ctl='')
                self.setShapesEnabled()
            else :
                self.set_scale_control(scale_ctl='HV')
                self.setShapesEnabled(False)
        else :
            logger.info(self.key_usage())

#------------------------------

if __name__ == "__main__" :

    logging.basicConfig(format='%(asctime)s %(levelname)s L:%(lineno)03d %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    from PyQt5.QtWidgets import QApplication
    import sys
    import numpy as np; global np
    arr = np.random.random((1000, 1000))
    app = QApplication(sys.argv)
    w = FWViewImageShapes(None, arr, origin='UL', scale_ctl='HV')
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
