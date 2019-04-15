#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: H5TreeViewModel.py 13101 2017-01-29 21:22:43Z dubrovin@SLAC.STANFORD.EDU $
#
# Description:
#  Module H5TreeViewModel...
#------------------------------------------------------------------------

"""Makes QtGui.QStandardItemModel for QtGui.QTreeView

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@see project modules
    * :py:class:`CalibManager.H5TreeViewModel.py` - this
    * :py:class:`CalibManager.H5WTree.py`

@author Mikhail S. Dubrovin
"""

#------------------------------

import sys
import h5py
from PyQt4 import QtGui, QtCore
from CalibManager.QIcons import icon

#------------------------------

class H5TreeViewModel(QtGui.QStandardItemModel) :
    """Makes QtGui.QStandardItemModel for QtGui.QTreeView.
    """

    def __init__(self, parent=None, fname='/reg/g/psdm/detector/calib/epix100a/epix100a-test.h5'):

        icon.set_icons()
        self.icon_folder_open   = icon.icon_folder_open
        self.icon_folder_closed = icon.icon_folder_closed
        self.icon_data          = icon.icon_data 

        QtGui.QStandardItemModel.__init__(self, parent)

        self.fname = fname

        self.str_file  = 'File'
        self.str_data  = 'Data'
        self.str_group = 'Group'

        #self._model_example()
        self._model_hdf5_tree()


    def close(self) :
        print "%s.close()" % self.__class__.__name__


    def _model_hdf5_tree(self) :
        """Puts the HDF5 file structure in the model tree"""

        print '%s makes the tree view model for HDF5 file: %s' % (self.__class__.__name__, self.fname)
        f = h5py.File(self.fname, 'r') # open read-only
        self._begin_construct_tree(f)
        f.close()
        print '=== EOF ==='

    #---------------------

    def _begin_construct_tree(self, g):
        """Adds the input file/group/dataset (g) name and begin iterations on its content"""

        #print "Add structure of the",
        #if   isinstance(g,h5py.File):    print "'File'",
        #elif isinstance(g,h5py.Group):   print "'Group' from file",
        #elif isinstance(g,h5py.Dataset): print "'Dataset' from file",
        #print g.file,"\n",g.name
        self.parent_item = self.invisibleRootItem()
        self.parent_item.setAccessibleDescription(self.str_file)
        self.parent_item.setAccessibleText(g.name) # Root item does not show this text...
        #self.parent_item.setIcon(self.icon_folder_open) # Root item does not show icon...

        if isinstance(g, h5py.Dataset):
            #print offset, "(Dateset)   len =", g.shape #, subg.dtype
            item = QtGui.QStandardItem(QtCore.QString(g.key()))
            item.setAccessibleDescription(self.str_data)
            self.parent_item.appendRow(item)            
        else:
            self._add_group_to_tree(g,self.parent_item) # start recursions from here


        #self._set_checkable_items_for_level(self.parent_item, level=3)
 
    #---------------------

    def _add_group_to_tree(self, g, parent_item):
        """Adds content of the file/group/dataset iteratively, starting from the sub-groups of g"""

        d = dict(g)
        list_keys = sorted(d.keys())
        list_vals = d.values()
        #print 'list_keys =', list_keys 

        for key in list_keys:
        #for key,val in dict(g).iteritems():

            #subg = val
            subg = d[key]

            item = QtGui.QStandardItem(QtCore.QString(key))
            #print '    k=', key, #,"   ", subg.name #, val, subg.len(), type(subg), 
            if isinstance(subg, h5py.Dataset):
                #print " (Dateset)   len =", subg.shape #, subg.dtype
                item.setIcon(self.icon_data)
                item.setCheckable(True)
                item.setAccessibleDescription(self.str_data)
                item.setAccessibleText(str(key))
                parent_item.appendRow(item)
                #print 'item row, col:', parent_item.row(), parent_item.column()
                
                
            elif isinstance(subg, h5py.Group):
                #print " (Group)   len =",len(subg) 
                #offset_subg = offset + '    '
                item.setCheckable(True)
                item.setIcon(self.icon_folder_closed)
                item.setAccessibleDescription(self.str_group)
                item.setAccessibleText(str(key))
                #item.setDragEnabled(True)
                #item.setDropEnabled(True)
                #item.setSelectable(False)

                parent_item.appendRow(item)

                self._add_group_to_tree(subg,item)

    #---------------------

    def _set_checkable_items_for_level(self, item, level=3):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""

        if self.distance_from_top(item) > level : item.setCheckable(True)

        if item.hasChildren():
            for row in range(item.rowCount()) :
                item_child = item.child(row,0)
                self._set_checkable_items_for_level(item_child, level)                

    #---------------------

    def distance_from_top(self, item, level=1):
        parent_item = item.parent()
        if parent_item is None : return level
        level += 1
        level = self.distance_from_top(parent_item, level)
        return level

    #---------------------
    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def get_full_name_from_item(self, item): 
        """Returns the full name in the HDF5 tree model for the given item"""
        ind = self.indexFromItem(item)        
        return self.get_full_name_from_index(ind)

    #---------------------
    
    def get_full_name_from_index(self, ind): 
        """Begin recursion from item with given ind(ex) and forms the full name in the self._full_name"""
        item = self.itemFromIndex(ind)
        self._full_name = item.text() 
        self._get_full_name(ind) 
        return str(self._full_name) ### QString object is converted to str

    #---------------------

    def _get_full_name(self, ind): 
        """Recursion from child to parent"""
        ind_par  = self.parent(ind)
        if(ind_par.column() == -1) :
            item = self.itemFromIndex(ind)
            self._full_name = '/' + self._full_name
            #print 'Item full name :' + self._full_name
            return self._full_name
        else:
            item_par = self.itemFromIndex(ind_par)
            self._full_name = item_par.text() + '/' + self._full_name
            self._get_full_name(ind_par)

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def set_all_group_icons(self,icon):
        """Iterates over the list of item in the QTreeModel and set icon for all groups"""
        self.new_icon = icon
        self._iteration_over_items_and_set_icon(self.parent_item)

    #---------------------

    def _iteration_over_items_and_set_icon(self,parent_item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        if parent_item.accessibleDescription() == 'Group' :
            parent_item.setIcon(self.new_icon)
            
        if parent_item.hasChildren():
            for row in range(parent_item.rowCount()) :
                item = parent_item.child(row,0)
                self._iteration_over_items_and_set_icon(item)                

    #---------------------
    #---------------------
    #---------------------

    def check_parents(self, item):
        check_state = item.checkState()
        self._set_parents_check_state(item, check_state)

    def _set_parents_check_state(self, item, check_state):
        parent_item = item.parent()
        if parent_item is None : return # for top-most parent
        parent_item.setCheckState(check_state)
        self._set_parents_check_state(parent_item, check_state)

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def check_children(self, item) :

        check_state = item.checkState()
        self._set_children_check_state(item, check_state)

    def _set_children_check_state(self, item, check_state) :
        if item.hasChildren() :
            #print 'item.row, col', item.row(), item.column()
            for row in range(item.rowCount()) :
                item_child = item.child(row,0)
                item_child.setCheckState(check_state)
                self._set_children_check_state(item_child, check_state)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def reset_checked_items(self):
        """Iterates over the list of item in the QTreeModel and uncheck all checked items"""
        self._iteration_over_items_and_uncheck(self.parent_item)

    #---------------------

    def _iteration_over_items_and_uncheck(self,parent_item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][parent_item.checkState()]
        if state == 'CHECKED' or state == 'TRISTATE' :
            print ' Uncheck item.text():', parent_item.text()
            parent_item.setCheckState(0) # 0 means UNCHECKED
            
        if parent_item.hasChildren():
            for row in range(parent_item.rowCount()) :
                item = parent_item.child(row,0)
                self._iteration_over_items_and_uncheck(item)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def expand_checked_items(self,view):
        """Iterates over the list of item in the QTreeModel and expand all checked items"""
        self.view = view
        self._iteration_over_items_and_expand_checked(self.parent_item)

    #---------------------

    def _iteration_over_items_and_expand_checked(self,parent_item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][parent_item.checkState()]
        if state == 'CHECKED' or state == 'TRISTATE' :
            print ' Expand item.text():', parent_item.text()
            #Now we have to expand all parent groups for this checked item...
            self._expand_parents(parent_item)
            
        if parent_item.hasChildren():
            for row in range(parent_item.rowCount()) :
                item = parent_item.child(row,0)
                self._iteration_over_items_and_expand_checked(item)                

    #---------------------

    def _expand_parents(self,item):
        item_parent = item.parent()
        ind_parent  = self.indexFromItem(item_parent)
        if(ind_parent.column() != -1) :
            if item_parent.accessibleDescription() == 'Group' :
                self.view.expand(ind_parent)
                item_parent.setIcon(self.icon_folder_open)
                self._expand_parents(item_parent)

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def retreve_checked_items(self,list_of_checked_item_names):
        """Use the input list of items and check them in the tree model"""
        self.list_of_checked_item_names = list_of_checked_item_names
        for name in self.list_of_checked_item_names :
            print 'Retreve the CHECKMARK for item', name 
        self._iteration_over_items_and_check_from_list(self.parent_item)    

    #---------------------

    def _iteration_over_items_and_check_from_list(self,parent_item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""

        if parent_item.isCheckable():
            item_name = self.get_full_name_from_item(parent_item)
            if item_name in self.list_of_checked_item_names :
                print ' Check the item:', item_name # parent_item.text()
                parent_item.setCheckState(2) # 2 means CHECKED; 1-TRISTATE
            
        if parent_item.hasChildren():
            for row in range(parent_item.rowCount()) :
                item = parent_item.child(row,0)
                self._iteration_over_items_and_check_from_list(item)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------
    
    def get_list_of_checked_item_names_for_model(self):
        """Get the list of checked item names for the self tree model"""
        list_of_checked_items = self.get_list_of_checked_items()
        return self.get_list_of_checked_item_names(list_of_checked_items)

    #---------------------
    
    def get_list_of_checked_item_names(self,list_of_checked_items):
        """Get the list of checked item names from the input list of items"""
        self.list_of_checked_item_names = []
        print 'The number of CHECKED items in the tree model =', len(self.list_of_checked_items)
        
        for item in list_of_checked_items :
            item_full_name = self.get_full_name_from_item(item)
            self.list_of_checked_item_names.append(item_full_name)
            print 'Checked item :', item_full_name
            
        return self.list_of_checked_item_names   

    #---------------------

    def get_list_of_checked_items(self):
        """Returns the list of checked item names in the QTreeModel"""
        self.list_of_checked_items = []
        #self._iteration_over_tree_model_item_children_v1(self.parent_item)
        #self._iteration_over_tree_model_item_children_v2(self.parent_item)

        self._iteration_over_items_find_checked(self.parent_item)
        return self.list_of_checked_items

    #---------------------

    def _iteration_over_items_find_checked(self,parent_item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][parent_item.checkState()]
        if state == 'CHECKED' :
            #print ' checked item.text():', parent_item.text()
            self.list_of_checked_items.append(parent_item)
            
        if parent_item.hasChildren():
            for row in range(parent_item.rowCount()) :
                item = parent_item.child(row,0)
                self._iteration_over_items_find_checked(item)                

    #---------------------
    #---------------------
    #---------------------
    #---------------------

    def _iteration_over_tree_model_item_children_v1(self,parent_item):
        """Recursive iteration over item children in the frame of the QtGui.QStandardItemModel"""
        parentIndex = self.indexFromItem(parent_item)
        print ' item.text():', parent_item.text(),
        print ' row:',         parentIndex.row(),        
        print ' col:',         parentIndex.column()

        if parent_item.hasChildren():
            Nrows = parent_item.rowCount()
            print ' rowCount:', Nrows

            for row in range(Nrows) :
                item = parent_item.child(row,0)
                self._iteration_over_tree_model_item_children_v1(item)                

    #---------------------

    def _iteration_over_tree_model_item_children_v2(self,parent_item):
        """Recursive iteration over item children in the freame of the QtGui.QStandardItemModel"""
        print ' parent_item.text():', parent_item.text()
        if parent_item.hasChildren():
            list_of_items = parent_item.takeColumn(0) # THIS GUY REMOVES THE COLUMN !!!!!!!!
            parent_item.insertColumn(0, list_of_items) 
            for item in list_of_items : 
                self._iteration_over_tree_model_item_children_v2(item)                

    #---------------------

    def _model_example(self) :
        """Makes the model tree for example"""
        for k in range(0, 6):
            parent_item = self.invisibleRootItem()
            for i in range(0, k):
                item = QtGui.QStandardItem(QtCore.QString("itemA %0 %1").arg(k).arg(i))
                item.setIcon(self.icon_data)
                item.setCheckable(True) 
                parent_item.appendRow(item)
                item = QtGui.QStandardItem(QtCore.QString("itemB %0 %1").arg(k).arg(i))
                item.setIcon(self.icon_folder_closed)
                parent_item.appendRow(item)
                parent_item = item
                print 'append item %s' % (item.text())

#------------------------------

if __name__ == "__main__" :    
    import sys

    app = QtGui.QApplication(sys.argv)
    m = H5TreeViewModel(parent=None, fname='/reg/g/psdm/detector/calib/epix100a/epix100a-test.h5')

    #sys.exit('Module is not supposed to be run as main module')

#------------------------------
