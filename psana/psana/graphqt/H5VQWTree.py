
"""Class :py:class:`QWTree` is a QTreeView->QWidget for tree model
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWTree.py

    from psana.graphqt.QWTree import QWTree
    w = QWTree()

Created on 2019-11-12 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys
import h5py
from psana.graphqt.QWTree import Qt, QWTree, icon, QStandardItem


def name_in_path(path,sep='/'):
    return path.rsplit(sep,1)[-1]


class H5VQWTree(QWTree):
    """Widget for HDF5 tree
    """
    TEST_FNAME = '/cds/group/psdm/detector/calib/jungfrau/jungfrau-171113-154920171025-3d00fb.h5'

    def __init__(self, **kwargs):

        wparent = kwargs.get('parent', None)
        self.fname = kwargs.get('fname', self.TEST_FNAME)

        QWTree.__init__(self, wparent)
        self.h5py = h5py
        self.process_expand()


    def set_file(self, fname):
        self.collapseAll()
        self.fname = fname
        self.fill_tree_model()
        self.expandAll()


    def fill_tree_model(self, g=None, parent_item=None):

        item = None

        # initialization at 1-st call
        if g is None: # on first call
            self.clear_model()
            logger.debug('Open file: %s' % self.fname)
            self.model.setHorizontalHeaderLabels(('%s/'%os.path.dirname(self.fname),))
            g=h5py.File(self.fname, 'r')
            self.fill_tree_model(g)
            return # !!!

        if isinstance(g, h5py.File):
            logger.debug('(File) %s %s' % (g.filename, g.name))
            root_item = self.model.invisibleRootItem()
            item = QStandardItem(os.path.basename(self.fname))
            item.setIcon(icon.icon_folder_open)
            item.setData(g)
            root_item.appendRow(item)

        elif isinstance(g,h5py.Group):
            logger.debug('(Group) %s' % g.name)
            item = QStandardItem(name_in_path(g.name))
            item.setIcon(icon.icon_folder_closed)
            item.setData(g)
            parent_item.appendRow(item)

        elif isinstance(g,h5py.Dataset):
            logger.debug('(Dataset) %s   len=%s   dtype=%s' % (g.name, g.shape, g.dtype))
            item = QStandardItem('%s: %s %s' % (name_in_path(g.name), g.shape, g.dtype))
            item.setIcon(icon.icon_table)
            item.setCheckable(True)
            item.setData(g)
            parent_item.appendRow(item)

        else:
            print('WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
            sys.exit ( "EXECUTION IS TERMINATED" )

        if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
            #if item is None: return
            for k,v in dict(g).items():
                subg = v
                #logger.debug('    k: %s v: %s' % (k, str(v))) #,"   ", subg.name #, val, subg.len(), type(subg),
                self.fill_tree_model(subg, item)
            return


    def show_tool_tips(self):
        self.setToolTip('HDF5 tree')


    def set_style(self):
        QWTree.set_style(self)
        #self.header().hide()
        self.header().show()


    def full_path(self, item, path=''):
        if item is None: return path
        if isinstance(item.data(), h5py.File): return 'fd-%s' % path
        else:
          txt = item.text()
          #name = txt.split(':')[0]
          txt = txt.strip().replace(' ','').replace('(','-').replace(')','-')
          txt = txt.replace(',','-').replace(':','').replace('|','-').replace('/','-')
          txt = txt.replace('--','-').replace('--','-')
          path_ext = '%s-%s' % (txt,path) if path else txt
          return self.full_path(item.parent(), path_ext)


    def on_item_selected(self, selected, deselected):
        itemsel = self.model.itemFromIndex(selected)
        if itemsel is not None:
            parent = itemsel.parent()
            parname = parent.text() if parent is not None else None
            msg = 'selected item: %s row: %d parent: %s dtype: %s' % (itemsel.text(), selected.row(), parname, type(itemsel.data())) 
            logger.debug(msg)
            if isinstance(itemsel.data(), h5py.Dataset):
                msg = 'data.value:\n%s' % str(itemsel.data()[()]) #.value)
                logger.debug(msg)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWTree.closeEvent(self, e)


    if __name__ == "__main__":

      def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  E - expand'\
               '\n  C - collapse'\
               '\n'

      def keyPressEvent(self, e):
        logger.debug('keyPressEvent, key = %s'%e.key())
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_E:
            self.process_expand()

        elif e.key() == Qt.Key_C:
            self.process_collapse()

        else:
            logger.debug(self.key_usage())


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    logging.basicConfig(format=fmt, datefmt='%H:%M:%S', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = H5VQWTree()
    w.setGeometry(10, 25, 400, 800)
    w.setWindowTitle('H5VQWTree')
    w.move(50,20)
    w.show()
    app.exec_()
    del w
    del app

# EOF
