#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#   GUICalibDirTree...
#------------------------------------------------------------------------

#--------------------------------
__version__ = "$Revision$"
#--------------------------------

import os

from PyQt4 import QtGui, QtCore

from ConfigParametersForApp import cp
from Logger                 import logger
from FileNameManager        import fnm

#------------------------------

class GUICalibDirTree(QtGui.QWidget):

    calib_types_cspad = [
        'center'
       ,'center_corr_xxx'
       ,'center_global'
       ,'offset'
       ,'offset_corr'
       ,'marg_gap_shift'
       ,'quad_rotation'
       ,'quad_tilt'
       ,'rotation'
       ,'tilt'
       ,'beam_vector'
       ,'beam_intersect'
       ,'pedestals'
       ,'pixel_status'
       ,'common_mode'
       ,'filter'
       ,'pixel_gain'
        ]

    calib_types_cspad2x2 = [
        'center'
       ,'tilt'     
       ,'pedestals'
       ,'pixel_status'
       ,'common_mode'
       ,'filter'
       ,'pixel_gain'
        ]
    
    calib_dets_cspad = [ 
        'XppGon.0:Cspad.0'
       ,'XcsEndstation.0:Cspad.0'
       ,'CxiDs1.0:Cspad.0'
       ,'CxiDs2.0:Cspad.0'
       ,'CxiDsd.0:Cspad.0'
        ]

    calib_dets_cspad2x2 = [ 
        'XppGon.0:Cspad2x2.0'
       ,'XppGon.0:Cspad2x2.1'
       ,'MecTargetChamber.0:Cspad2x2.1'
       ,'MecTargetChamber.0:Cspad2x2.2'
       ,'MecTargetChamber.0:Cspad2x2.3'
       ,'MecTargetChamber.0:Cspad2x2.4'
       ,'MecTargetChamber.0:Cspad2x2.5'
       ,'CxiSc.0:Cspad2x2.0'
       ,'MecTargetChamber.0:Cspad2x2.1'
        ]

    calib_vers = [
        'CsPad::CalibV1'
       ,'CsPad2x2::CalibV1'
        ]


    def __init__(self, parent=None) :
        #super(GUIQTreeView, self).__init__(parent)
        QtGui.QWidget.__init__(self, parent)

        self.setGeometry(100, 100, 200, 600)
        self.setWindowTitle('Item selection tree')

        cp.setIcons()
 
        self.fill_calib_dir_tree()

        #self.view = QtGui.QListView()
        #self.view = QtGui.QTableView()
        self.view = QtGui.QTreeView()
        self.view.setModel(self.model)
        #self.view.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        #self.view.expandAll()
        self.view.setAnimated(True)
 
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.view)

        if parent is None :
            self.setLayout(vbox)

        self.connect(self.view.selectionModel(), QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.itemSelected)
        #self.view.clicked.connect(self.someMethod1)       # This works
        #self.view.doubleClicked.connect(self.someMethod2) # This works
        self.model.itemChanged.connect(self.itemChanged)
        self.view.expanded.connect(self.itemExpanded)
        self.view.collapsed.connect(self.itemCollapsed)

        self.setStyle()


    def fill_calib_dir_tree(self) :

        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderLabels('x')
        #self.model.setHorizontalHeaderItem(1,QtGui.QStandardItem('Project Title'))
        #self.model.setVerticalHeaderLabels('abc')

        for v in self.calib_vers :
            det, vers = v.split('::',1)
            #print 'det, vers =', det, vers

            parentItem = self.model.invisibleRootItem() 
            itemv = QtGui.QStandardItem(QtCore.QString(v))
            itemv.setIcon(cp.icon_folder_closed)
            #itemv.setCheckable(True) 
            parentItem.appendRow(itemv)
  
            if det == 'CsPad' :
                self.calib_type_list = self.calib_types_cspad
                self.calib_det_list  = self.calib_dets_cspad
            elif det == 'CsPad2x2' :
                self.calib_type_list = self.calib_types_cspad2x2
                self.calib_det_list  = self.calib_dets_cspad2x2
            else :
                print 'UNKNOWN DETECTOR' 

            for d in self.calib_det_list :
                itemd = QtGui.QStandardItem(QtCore.QString(d))
                itemd.setIcon(cp.icon_folder_closed)
                #itemd.setCheckable(True) 
                itemv.appendRow(itemd)
 
                for t in self.calib_type_list :
                    itemt = QtGui.QStandardItem(QtCore.QString(t))
                    itemt.setIcon(cp.icon_folder_closed)
                    itemt.setCheckable(True) 
                    itemd.appendRow(itemt)


    def getFullNameFromItem(self, item): 
        #item = self.model.itemFromIndex(ind)        
        ind   = self.model.indexFromItem(item)        
        return self.getFullNameFromIndex(ind)


    def getFullNameFromIndex(self, ind): 
        item = self.model.itemFromIndex(ind)
        if item is None : return 'None'
        self._full_name = item.text()
        self._getFullName(ind)
        return str(self._full_name)


    def _getFullName(self, ind): 
        ind_par  = self.model.parent(ind)
        if(ind_par.column() == -1) :
            item = self.model.itemFromIndex(ind)
            self.full_name = '/' + self._full_name
            #print 'Item full name :' + self._full_name
            return self._full_name
        else:
            item_par = self.model.itemFromIndex(ind_par)
            self._full_name = item_par.text() + '/' + self._full_name
            self._getFullName(ind_par)


    def getTextListOfChildren(self, index): 
        item = self.model.itemFromIndex(index)
        number_of_children = item.rowCount()
        txt_list_of_children = []
        for row in range(number_of_children) :
            child_item = item.child(row)
            txt_list_of_children.append(str(child_item.text()))
        return txt_list_of_children


    def itemChanged(self, item):
        state = ['UNCHECKED', 'TRISTATE', 'CHECKED'][item.checkState()]
        msg = 'Item with full name %s, is at state %s' % ( self.getFullNameFromItem(item),  state)
        #print msg
        logger.info(msg, __name__)       

        
    def itemExpanded(self, ind): 
        item = self.model.itemFromIndex(ind)
        item.setIcon(cp.icon_folder_open)
        msg = 'Item expanded: %s' % item.text()  
        logger.info(msg, __name__)       


    def itemCollapsed(self, ind):
        item = self.model.itemFromIndex(ind)
        item.setIcon(cp.icon_folder_closed)
        msg = 'Item collapsed: %s' % item.text()  
        logger.info(msg, __name__)       


    def itemSelected(self, selected, deselected):
        selected_txt = self.getFullNameFromIndex(selected)
        msg1 = 'Item selected: %s' % self.getFullNameFromIndex(selected)

        txt_list_of_children = self.getTextListOfChildren(selected)
        
        self.onSelectedItem(selected_txt, txt_list_of_children)
        logger.info(msg1, __name__)       

        #deselected_txt = self.getFullNameFromIndex(deselected)
        #msg2 = 'Item deselected: %s' % self.getFullNameFromIndex(deselected)
        #logger.info(msg2, __name__)       
        #self.onDeselectedItem(deselected_txt)


    def onSelectedItem(self, path_from_calib, list_expected) :
        cp.guitabs.setTabByName('Status')
        dir = os.path.join(fnm.path_to_calib_dir(), path_from_calib)        
        cp.guistatus.statusOfDir(dir, list_expected)


    def setStyle(self):
        #self.setMinimumSize(100,400)
        self.setMinimumWidth(150)
        self.setMaximumWidth(500)
        self.setMinimumHeight(500)
        self.setContentsMargins(QtCore.QMargins(-9,-9,-9,-9))


    #def resizeEvent(self, e):
        #logger.debug('resizeEvent', self.name) 
        #pass


    #def moveEvent(self, e):
        #logger.debug('moveEvent', self.name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #pass
 
#-----------------------------

if __name__ == "__main__" :
    import sys
    app = QtGui.QApplication(sys.argv)
    widget = GUICalibDirTree()
    widget.show()
    app.exec_()

#-----------------------------
