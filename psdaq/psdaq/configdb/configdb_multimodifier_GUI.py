#!/usr/bin/env python

"""configdb_multimodifier_GUI.py: Offers a GUI to perform multiple modifications of a variable  over several detectors in configdb. The GUI uses configdb_multimodifier.py which uses configdb.py"""

__author__      = "Riccardo Melchiorri"

import sys
from PyQt5.QtWidgets import QApplication, QLineEdit, QWidget, QMainWindow, QLabel, QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox, QRadioButton,QListWidget, QListWidgetItem, QAbstractItemView
import os
from psdaq.configdb.configdb import configdb
import configdb_multimodifier

class ConfigdbGUI(QWidget):
    
    def __init__(self):
        super().__init__()
        self.dev_list=[]
        self.list_det=[]

        grid_layout = QGridLayout()
        self.list_det_edit = QListWidget()
        self.list_det_edit.setSelectionMode(QAbstractItemView.MultiSelection)
        grid_layout.addWidget(self.list_det_edit, 1,6,5,3)
        self.list_det_edit.itemSelectionChanged.connect(self.get_list_det)

        self.det_selection=QLineEdit()
        self.det_selection.textChanged.connect(self.get_selection)
        grid_layout.addWidget(self.det_selection,0,6)

        Det_Edit=QPushButton('Reload')
        grid_layout.addWidget(Det_Edit,0,7)
        Det_Edit.clicked.connect(self.load_det)

        clear_list_det=QPushButton('Clear')
        clear_list_det.clicked.connect(self.list_clean)
        grid_layout.addWidget(clear_list_det,0,8)

        URI_Label=QLabel('database: ')
        self.radio_prod = QRadioButton("Prod")
        self.radio_prod.setChecked(True)
        self.radio_dev = QRadioButton("Dev")
        self.radio_dev.setChecked(False)

        grid_layout.addWidget(URI_Label,0,0)
        grid_layout.addWidget(self.radio_prod,0,1)
        grid_layout.addWidget(self.radio_dev,0,2)

        Root_Label=QLabel('Root')
        self.Root_Edit=QLineEdit('configDB')
        grid_layout.addWidget(Root_Label,0,3)
        grid_layout.addWidget(self.Root_Edit,0,4)        

        Inst_Label=QLabel('Instrument')
        self.Inst_Edit=QLineEdit('tmo')
        grid_layout.addWidget(Inst_Label,1,0)
        grid_layout.addWidget(self.Inst_Edit,1,1,1,2)
        
        User_Label=QLabel('User')
        self.User_Edit=QLineEdit('tmoopr')
        grid_layout.addWidget(User_Label,1,3)
        grid_layout.addWidget(self.User_Edit,1,4,1,2)
                
        Key_Label=QLabel('Config Key')
        self.Key_Edit=QLineEdit('user')
        grid_layout.addWidget(Key_Label,3,0)
        grid_layout.addWidget(self.Key_Edit,3,1,1,2)

        Value_Label=QLabel('Config Value')
        self.Value_Edit=QLineEdit('1')
        grid_layout.addWidget(Value_Label,3,3)
        grid_layout.addWidget(self.Value_Edit,3,4,1,2)

        
        self.Modify_Edit=QCheckBox('Modify')
        grid_layout.addWidget(self.Modify_Edit,4,1)

        Send_but=QPushButton('Send')
        grid_layout.addWidget(Send_but,4,3)
        Send_but.clicked.connect(self.run)

        self.load_det()

        self.setLayout(grid_layout)
        self.setGeometry(50, 50, 800,40)
        self.setWindowTitle("ConfigDB modifier")
        self.show()

    def get_selection(self):
        self.list_det_edit.clear()
        for w in self.dev_list:
            if (self.det_selection.text() in w): QListWidgetItem(w, self.list_det_edit)

    def list_clean(self):
        self.list_det_edit.clearSelection()
        self.det_selection.setText("")

    def get_list_det(self):
        self.list_det=[item.text() for item in self.list_det_edit.selectedItems()]

    def collect_data(self):
        data={}
        if self.radio_prod.isChecked():
            data["URI"]    = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws'
        else:
            data["URI"]    = 'https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws'
        data["Root"]   = self.Root_Edit.text()
        data["Inst"]   = self.Inst_Edit.text()
        data["User"]   = self.User_Edit.text()
        data["Key"]    = self.Key_Edit.text()
        data["Value"]  = self.Value_Edit.text()
        data["Modify"] = self.Modify_Edit.isChecked()
        
        return data
       
    def load_det(self):
        self.det_selection.setText("")
        self.list_det_edit.clear()
        data = self.collect_data()

        confdb=configdb(data['URI'], data['Inst'], create=False, root=data['Root'], user=data["User"], password=os.getenv('CONFIGDB_AUTH'))
        self.dev_list=(confdb.get_devices('BEAM', hutch=data["Inst"]))
        
        for w in self.dev_list:
            QListWidgetItem(w, self.list_det_edit)
            
        
    def run(self):
        data = self.collect_data()
        data["Det"]=self.list_det
        
        try:
            configdb_multimodifier.dbmultimodifier(URI_CONFIGDB = data["URI"], ROOT_CONFIGDB = data["Root"], INSTRUMENT = data["Inst"], USER=data["User"], DETECTOR=data["Det"], CONFIG_KEY=data["Key"].split(":"), CONFIG_VALUE=data["Value"], MODIFY=data["Modify"])
        except:
            print("There was an error in the form")    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ConfigdbGUI()
    sys.exit(app.exec_())
