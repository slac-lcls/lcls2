#!/usr/bin/env python

"""configdb_GUI.py: Offers a GUI to perform multiple modifications of a variable over several detectors in configdb. The GUI uses configdb_multimod.py which uses configdb.py"""

__author__      = "Riccardo Melchiorri"

import sys
from PyQt5.QtWidgets import QApplication, QLineEdit, QWidget, QMainWindow, QLabel, QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox, QRadioButton,QListWidget, QListWidgetItem, QAbstractItemView, QTextEdit, QMessageBox, QProgressBar
import os
from psdaq.configdb.configdb import configdb
from psdaq.configdb.configdb_multimod import configdb_multimod
import threading
    
class ConfigdbGUI(QWidget):
    
    def __init__(self):
        super().__init__()
        self.data={}
        self.hutch=''
        self.dev=''
        self.confdb=''
        self.detType={}
        
        # Select Database to use Prod or Dev
        URI_Label=QLabel('database: ')            
        self.radio_prod = QRadioButton("Prod")
        self.radio_prod.setChecked(True)
        self.radio_dev = QRadioButton("Dev")
        self.radio_dev.setChecked(False)
        self.radio_dev.toggled.connect(self.radioURIselected)
        #self.radio_prod.toggled.connect(self.radioURIselected)
        # needs to be triggered only once
        
        # Select which detector to be searched
        self.list_detector_Box = QComboBox()
        self.list_detector_Box.currentTextChanged.connect(self.load_data_second_column)

        #progress bar for data load
        self.pbar = QProgressBar(self)

        # String defining the value to changed
        self.paramValue=QLineEdit()
        #button to run the change, button is disabled when no key is selected
        self.modButton=QPushButton('Modify')
        self.modButton.clicked.connect(self.modify_database)
        self.modButton.setEnabled(False)
        
        #devices and hutches
        self.first_column = QTreeWidget()
        #detectors
        self.second_column = QListWidget()
        #detector's parameters (only for one)
        self.third_column = QTreeWidget()
        #disabled (log print output)
        self.forth_column = QTextEdit()

        self.first_column.itemSelectionChanged.connect(self.first_column_select)       
        self.first_column.setHeaderLabel("Prod") 
        self.third_column.setHeaderLabel("Parameters") 
        self.second_column.setSelectionMode(QAbstractItemView.MultiSelection)
        self.second_column.itemSelectionChanged.connect(self.second_column_select)
        self.third_column.itemSelectionChanged.connect(self.third_column_select)
        
        #to speed up start up, data are loaded on a thread
        thread_first_column_data = threading.Thread(target=self.load_data_first_column)
        self.set_default()
        #self.load_data_first_column()
        thread_first_column_data.start()
        
        layout_columns = QHBoxLayout()
        layout_commands = QHBoxLayout()
        layout = QVBoxLayout()
        
        layout.addLayout(layout_commands)
        layout.addLayout(layout_columns)
        
        self.setLayout(layout)
        
        layout_columns.addWidget(self.first_column)
        layout_columns.addWidget(self.second_column)
        layout_columns.addWidget(self.third_column)
        layout_columns.addWidget(self.forth_column)
        
        #layout_columns.addWidget(self.forth_column)
        
        layout_commands.addWidget(self.radio_dev)
        layout_commands.addWidget(self.radio_prod)
        layout_commands.addWidget(self.list_detector_Box)
        layout_commands.addWidget(self.pbar)
        layout_commands.addWidget(self.paramValue)
        layout_commands.addWidget(self.modButton)
        
        self.setGeometry(50, 50, 800,400)
        self.setWindowTitle("ConfigDB modifier")
        self.show()

    # when Modify is clicked, a popup shows which modifications are going to be performed 
    def modify_database(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgBox.buttonClicked.connect(self.msgbtn)
        msgBox.setText(f"You are going to modify the variable :{self.full_path_param[::-1]} \nwith this value {self.paramValue.text()}\n in these detectors: {self.second_column_selection} \n in {self.data['URI']}");
        msgBox.exec();

    #If from popup OK then continue with modification, Cancel otherwise
    def msgbtn(self,i):
        print(i.text())
        if i.text() == "OK":
            if self.paramValue.text() != '':
                stdout = configdb_multimod(URI_CONFIGDB = self.data["URI"], DEV = self.dev, ROOT_CONFIGDB = self.data["Root"], INSTRUMENT = self.hutch, DETECTOR=self.second_column_selection, CONFIG_KEY=self.full_path_param[::-1] , CONFIG_VALUE=self.paramValue.text(), MODIFY=True)
                self.third_column_select()
                
        else:
            return

    # choosing dev or prod
    def radioURIselected(self):
        if self.radio_dev.isChecked():
            self.data['URI']='https://pswww.slac.stanford.edu/ws-kerb/devconfigdb/ws'
            self.first_column.setHeaderLabel("Dev")
        else:
            self.data['URI']='https://pswww.slac.stanford.edu/ws-kerb/configdb/ws'
            self.first_column.setHeaderLabel("Prod")
        
        self.third_column.clear()
        self.second_column.clear()
        self.first_column.clear()
        self.second_column_selection=[]
        self.load_data_first_column()
    
    #function to create tree in column 1 and 3
    def buildtree(self, value, parent):
        for l in value:
            item = QTreeWidgetItem(parent)
            item.setText(0, l)
            if isinstance(value[l], dict):
                self.buildtree(value[l], item)

    # in case something is selected in the first column
    def first_column_select(self):
        self.second_column.clear()
        self.third_column.clear()
        self.modButton.setEnabled(False)
        
        first_column_items = self.first_column.selectedItems()
        try:
            self.dev   = first_column_items[0].text(0)
            self.hutch = first_column_items[0].parent().text(0)
            self.load_data_second_column()
        except:
            self.dev   = ''
            self.hutch = ''
            return
    
    #load data in second column
    def load_data_second_column(self):
        self.second_column.clear()
        self.forth_column.clear()
        try:
            for det in self.config[self.hutch][self.dev]:
                #print(self.data[f'{self.hutch}:{self.dev}:{det}'])
                if self.detType[f'{self.hutch}:{self.dev}:{det}'] == self.list_detector_Box.currentText() or self.list_detector_Box.currentText() == 'All':
                    QListWidgetItem(det, self.second_column)
        except:
            return
                
# in case something is selected in second column
    def second_column_select(self):
        self.third_column.clear()
        self.forth_column.clear()
        self.modButton.setEnabled(False)
        self.second_column_selection = [item.text() for item in self.second_column.selectedItems()] 
        self.load_data_third_column()
        
    #load data in third column    
    def load_data_third_column(self):
        try:
            self.third_column.setHeaderLabel(self.second_column_selection[0])
            config = self.confdb.get_configuration(self.dev, self.second_column_selection[0], hutch=self.hutch)
            self.buildtree( config, self.third_column)
        except:
            return
    
    #search parents in tree dictionary    
    def find_parents(self, val):
        try:
            self.full_path_param.append(val.parent().text(0))
            self.find_parents(val.parent())
        except:    
            return
        
    #in case something is selected in the third column
    def third_column_select(self):
        self.modButton.setEnabled(True)
        try:
            self.full_path_param=[self.third_column.selectedItems()[0].text(0)]    
            self.find_parents(self.third_column.selectedItems()[0])
            from subprocess import Popen, PIPE

            stdout = configdb_multimod(URI_CONFIGDB = self.data["URI"], DEV = self.dev, ROOT_CONFIGDB = self.data["Root"], INSTRUMENT = self.hutch, DETECTOR = self.second_column_selection, CONFIG_KEY = self.full_path_param[::-1], MODIFY=False)
            self.forth_column.clear()
            for s in stdout:
                self.forth_column.append(s)
        except:
            return
        
    #setting default values
    def set_default(self):
        self.data["URI"]    = 'https://pswww.slac.stanford.edu/ws-kerb/configdb/ws'
        self.data["Root"]   = 'configDB'
        self.data["Inst"]   = 'TMO'
         
    #load data for first column and dropdown
    def load_data_first_column(self):
        self.first_column.clear()
        self.second_column.clear()
        self.third_column.clear()
        self.second_column_select=[]
        
        self.confdb = configdb(self.data['URI'], self.data['Inst'], create=False, root=self.data['Root'])
        self.first_column.clear()
        list_of_hutch_names = self.confdb.get_hutches() # ['TMO', 'CXI', etc.]
        #print(list_of_hutch_names)
        self.list_of_device_configuration_names = self.confdb.get_device_configs() #['test']  
        #print(self.list_of_device_configuration_names)      
        self.detType={}
        config_hutch={}
        
        pbar_steps = range(0, 100, int(100./len(list_of_hutch_names)))
        print(len(list_of_hutch_names))
        i=0
        self.pbar.setValue(i) 
        for myhutch in list_of_hutch_names:
            list_of_alias_names = self.confdb.get_aliases(hutch=myhutch)
            #print(list_of_alias_names)
            config_alias={}

            for myalias in list_of_alias_names:
                list_of_device_names = self.confdb.get_devices(myalias, hutch=myhutch) 
                #print(list_of_device_names)
                config_alias[myalias]=list_of_device_names
                for dev in list_of_device_names:
                    #print(f'{myalias} {dev} {myhutch}')
                    try:
                        config = self.confdb.get_configuration( myalias, dev, hutch=myhutch)
                        self.detType[f'{myhutch}:{myalias}:{dev}'] = config['detType:RO']
                    except:
                        self.detType[f'{myhutch}:{myalias}:{dev}'] = '' 
            i=i+1
            self.pbar.setValue(pbar_steps[i]) 
            config_hutch[myhutch]=config_alias        

        self.list_detector_Box.clear()
        self.list_of_device_configuration_names.sort()
        self.list_of_device_configuration_names.append("All")
        self.list_detector_Box.addItems(self.list_of_device_configuration_names)
        
        self.list_detector_Box.setCurrentIndex(self.list_of_device_configuration_names.index('All'))
        
        self.buildtree(config_hutch, self.first_column)   
        self.config = config_hutch 
        self.pbar.setValue(100) 
        

def main():
    app = QApplication(sys.argv)
    ex = ConfigdbGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

