
"""Class :py:class:`FSTree` is a QWTree for filesystem tree presentation
========================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/FSTree.py

    # Import
    from psana.graphqt.FSTree import ...

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`FSTree`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2021-07-26 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import os #import psana.pyalgos.generic.UtilsFS as ufs

#from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.QWTree import *
from psana.pyalgos.generic.UtilsFS import safe_listdir
#from PyQt5.QtCore import pyqtSignal # Qt


def full_path_for_item(item, path=''):
    if item is None: return path # stop recursion
    if item.parent() is None: return '%s%s' % (item.text(),path) # stop recursion
    path_ext = ('/%s' % item.text()) if path=='' else\
               ('/%s%s' % (item.text(),path))
    return full_path_for_item(item.parent(), path_ext)


class FSTree(QWTree):
    """GUI for file system tree
    """
    def __init__(self, **kwa):
        logger.debug('FSTree.__init__')
        self.kwa = kwa
        self.topdir = kwa.get('topdir', None)
        QWTree.__init__(self, kwa.get('parent', None))
        self.set_selection_mode(smode='extended')
        #self.db_and_collection_selected.connect(self.on_db_and_collection_selected)
        self.expandAll()


    def update_tree_model(self, topdir):
        if not os.path.isdir(topdir):
            logger.warning('NON-EXISTENT PATH: %s' % str(topdir))
            return
        self.topdir = topdir
        self.fill_tree_model()


    def fill_tree_model(self):
        logger.debug('FSTree.fill_tree_model for %s' % self.topdir)
        from time import time
        t0_sec = time()

        self.clear_model()

        tdir = self.topdir

        if not os.path.isdir(tdir):
            logger.warning('NON-EXISTENT PATH: %s' % str(tdir))
            return

        if not os.access(tdir, os.R_OK): # os.W_OK | os.R_OK
            logger.warning('NON-ACCESSIBLE PATH FOR READOUT: %s' % str(tdir))
            return

        logger.debug('call safe_listdir("%s")' % str(tdir))
        r = safe_listdir(tdir, timeout_sec=5)
        if r is None:
            logger.warning('safe_listdir is None for: %s' % str(tdir))
            return

        logger.debug('fill_tree_model for top directory:%s' % str(tdir))


        self.fill_tree_model_dir(tdir, pitem=None, **self.kwa)

        logger.info('tree-model filling time %.3f sec' % (time()-t0_sec))

        self.expandAll()


    def add_item(self, pitem, name, iconimg=None, **kwa):
        item = QStandardItem(name)
        if iconimg: item.setIcon(iconimg)
        item.setEditable(kwa.get('iseditable', False))
        item.setCheckable(kwa.get('ischeckable', True))
        item.setSelectable(kwa.get('isselectable', True))
        item.setEnabled(kwa.get('isenabled', True))
        atxt = kwa.get('accessibletext', '')
        if atxt: item.setAccessibleText(atxt)
        adsc = kwa.get('accessibledescription', '')
        if atxt: item.setAccessibleDescription(adsc)
        pitem.appendRow(item)
        return item


    def fill_tree_model_dir(self, dirname, pitem=None, **kwa):

        logger.debug('call safe_listdir("%s")' % str(dirname))
        lst = safe_listdir(dirname)
        if lst is None:
            logger.warning('safe_listdir is None for: %s' % str(dirname))
            return

        names = sorted(lst) # ufs.list_of_files_in_dir(pdir)

        logger.debug('FSTree.fill_tree_model_dir len: %d names: %s' % (len(names), str(names)))

        is_selectable_dir  = kwa.get('is_selectable_dir', True)
        ptrns_selectable   = kwa.get('selectable_ptrns', []) #['.data',])
        ptrns_unselectable = kwa.get('unselectable_ptrns', []) #['HISTORY',])

        #if pattern: names = [name for name in names if patternin name]
        if pitem is None:
            self.item_top_invis = self.model.invisibleRootItem()
            item = self.add_item(self.item_top_invis, dirname,\
                                 iconimg=icon.icon_folder_closed, iseditable=False, ischeckable=False, isselectable=False)
            self.fill_tree_model_dir(dirname, pitem=item, **kwa)
            self.item_top = item
            return

        for name in names:
            path = os.path.join(dirname, name)
            isdir = os.path.isdir(path)
            cond = is_selectable_dir if isdir else\
                    ((len(ptrns_selectable)==0 or any([p in name for p in ptrns_selectable])) and\
                     (len(ptrns_unselectable)==0 or all([not(p in name) for p in ptrns_unselectable])))
            iconimg = icon.icon_folder_closed if isdir else icon.icon_data
            item = self.add_item(pitem, name, iconimg, iseditable=False, ischeckable=False, isselectable=cond)

            if isdir:
                self.fill_tree_model_dir(path, pitem=item, **kwa)


    def on_click(self, index):
        """Override method in QWTree"""
        item = self.model.itemFromIndex(index)
        itemname = item.text()
        parent = item.parent()
        parname = parent.text() if parent is not None else None
        msg = 'clicked item: %s parent: %s' % (itemname, parname) # index.row()
        logger.debug(msg)
        #names_selected = [i.text() for i in self.selected_items()]
        names_selected = [full_path_for_item(i) for i in self.selected_items()]
        logger.debug('on_click: %s' % str(names_selected))


    def on_item_selected(self, selected, deselected):
        QWTree.on_item_selected(self, selected, deselected)
        i = self.model.itemFromIndex(selected)
        if i is None:
           logger.debug('on_item_selected: item is None')
           return
        fname = full_path_for_item(i)
        logger.debug('on_item_selected: %s path: %s' % (str(i.text()), fname))


if __name__ == "__main__":

    #logging.getLogger('urllib3').setLevel(logging.WARNING)

    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)
    logger.info('set logger for module %s' % __name__)

    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = FSTree(parent=None,\
               topdir='/cds/data/psdm/MEC/mecx24215/calib',\
               is_selectable_dir=False,\
               selectable_ptrns=['.data'],\
               unselectable_ptrns=['HISTORY']\
              )
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle(w.__class__.__name__)
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

# EOF
