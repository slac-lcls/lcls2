#------------------------------
"""Class :py:class:`QWTree` is a QTreeView->QWidget for tree model
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWTree.py

    from psana.graphqt.QWTree import QWTree
    w = QWTree()

Created on 2019-11-12 by Mikhail Dubrovin
"""
#------------------------------
import logging
logger = logging.getLogger(__name__)

import sys
import h5py

#from PyQt5.QtWidgets import QTreeView, QVBoxLayout, QAbstractItemView
#from PyQt5.QtGui import QStandardItemModel, QStandardItem
#from PyQt5.QtCore import Qt, QModelIndex

#from psana.graphqt.CMConfigParameters import cp
#from psana.pyalgos.generic.Logger import logger
#from psana.graphqt.QWIcons import icon

from psana.graphqt.QWTree import Qt, QWTree, icon, QStandardItem
#------------------------------

def name_in_path(path,sep='/') :
    return path.rsplit(sep,1)[-1]

#------------------------------

class H5VQWTree(QWTree) :
    """Widget for HDF5 tree
    """
    #TEST_FNAME = '/reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5'
    TEST_FNAME = '/reg/g/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5'


    def __init__(self, **kwargs) :

        parent = kwargs.get('parent',None)
        self.fname = kwargs.get('fname', self.TEST_FNAME)

        QWTree.__init__(self, parent)

        #self._name = self.__class__.__name__

        #icon.set_icons()

        #self.model = QStandardItemModel()
        #self.set_selection_mode()

        #self.fill_tree_model() # in QWTree

        #self.expanded.connect(self.on_item_expanded)
        #self.collapsed.connect(self.on_item_collapsed)
        ##self.model.itemChanged.connect(self.on_item_changed)
        #self.connect_item_selected_to(self.on_item_selected)
        #self.clicked[QModelIndex].connect(self.on_click)
        ##self.doubleClicked[QModelIndex].connect(self.on_double_click)

        self.process_expand()
 

    def fill_tree_model(self, g=None, parent_item=None):

        item = None

        # initialization at 1-st call
        if g is None : # on first call
            self.clear_model()
            print('Open file: %s' % self.fname)
            self.model.setHorizontalHeaderLabels(('.../%s'%name_in_path(self.fname),))
            g=h5py.File(self.fname, 'r')
            self.fill_tree_model(g)
            return # !!!

        if isinstance(g, h5py.File) :
            print('(File)', g.filename, g.name)
            #print(g.__dir__())
            root_item = self.model.invisibleRootItem()
            item = QStandardItem('(file) %s'%g.name)
            #item.setCheckable(True) 
            item.setIcon(icon.icon_folder_open)
            item.setData(g)
            root_item.appendRow(item)
        
        elif isinstance(g,h5py.Group) :
            print('(Group)', g.name)
            item = QStandardItem(name_in_path(g.name))
            #item.setCheckable(True) 
            item.setIcon(icon.icon_folder_closed)
            item.setData(g)
            parent_item.appendRow(item)

        elif isinstance(g,h5py.Dataset) :
            print('(Dataset)', g.name, '    len =', g.shape, g.dtype) #, g.dtype
            item = QStandardItem('%s: %s %s' % (name_in_path(g.name), g.shape, g.dtype))
            item.setIcon(icon.icon_table)
            item.setCheckable(True) 
            item.setData(g)
            parent_item.appendRow(item)

        else :
            print('WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
            sys.exit ( "EXECUTION IS TERMINATED" )
        
        if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
            #if item is None : return
            for k,v in dict(g).items() :
                subg = v
                #print('    k:',k, ' v:',v) #,"   ", subg.name #, val, subg.len(), type(subg),
                self.fill_tree_model(subg, item)
            return

    #--------------------------

    def show_tool_tips(self):
        self.setToolTip('HDF5 tree') 


    def set_style(self):
        QWTree.set_style(self)
        #self.header().hide()
        self.header().show()
        self.setMinimumSize(400, 900)

    #def set_style(self):
    #    #from psana.graphqt.Styles import style
    #    self.setWindowIcon(icon.icon_monitor)
    #    self.setContentsMargins(0,0,0,0)
    #    self.setStyleSheet("QWTree::item:hover{background-color:#00FFAA;}")


    #def resizeEvent(self, e):
        #pass
        #self.frame.setGeometry(self.rect())
        #logger.debug('resizeEvent') 


    #def moveEvent(self, e):
        #logger.debug('moveEvent') 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWTree.closeEvent(self, e)

        #try    : self.gui_win.close()
        #except : pass

        #try    : del self.gui_win
        #except : pass


    #def on_exit(self):
    #    logger.debug('on_exit')
    #    self.close()

#------------------------------
#------------------------------

#    if __name__ == "__main__" :

    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  E - expand'\
               '\n  C - collapse'\
               '\n'

    def keyPressEvent(self, e) :
        #logger.debug('keyPressEvent, key = %s'%e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_E : 
            self.process_expand()

        elif e.key() == Qt.Key_C : 
            self.process_collapse()

        else :
            logger.debug(self.key_usage())

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=fmt, datefmt='%H:%M:%S', level=logging.DEBUG)
    #logger.setPrintBits(0o177777)
    app = QApplication(sys.argv)
    w = H5VQWTree()
    #w.setGeometry(10, 25, 200, 600)
    w.setWindowTitle('H5VQWTree')
    w.move(50,20)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------
