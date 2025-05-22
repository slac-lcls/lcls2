import sys
import socket
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from .xpm_utils import *

NGroups     = 8

def addGroup(tw, base, group):
    pvbase = base+'%d:'%group
    wlo    = QtWidgets.QVBoxLayout()

    trgbox = QtWidgets.QGroupBox('Triggers')
    trglo  = QtWidgets.QVBoxLayout()
    LblEditEvt   (trglo, pvbase, "L0Select"        )
    LblEditDst   (trglo, pvbase, "DstSelect"       )
    trgbox.setLayout(trglo)
    wlo.addWidget(trgbox)

    w = QtWidgets.QWidget()
    w.setLayout(wlo)
    tw.addTab(w,'Group %d'%group)

class PatternTab(QtWidgets.QTabWidget):
    def __init__(self, pvbase):
        super(PatternTab,self).__init__()

        v20b = (1<<20)-1
        self.addTab(PvTableDisplay(f'{pvbase}PATT:GROUPS', [f'Group{i}' for i in range(8)], (0, v20b, v20b, v20b, 0)),"Stats")

        self.coinc = []
        g = QtWidgets.QGridLayout()
        for i in range(8):
            g.addWidget(QtWidgets.QLabel(f'G{i}'),0,i+1)
            g.addWidget(QtWidgets.QLabel(f'G{i}'),i+1,0)
        for i in range(8):
            for j in range(i,8):
                w = QtWidgets.QLabel('-')
                w.setMinimumWidth(70)
                self.coinc.append(w)
                g.addWidget(w, i+1, j+1)
        box = QtWidgets.QGroupBox("Group Coincidences")
        box.setLayout(g)
        self.addTab(box,"Coincidences")
#        l.addStretch()
        initPvMon(self,f'{pvbase}PATT:COINC',isStruct=True)

    def update(self,err):
        if err is None:
            v = self.pv.__value__
            q = v.value.Coinc
            for i,qv in enumerate(q):
                self.coinc[i].setText(str(qv))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        global ATCAWidget
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self._pvlabels = []

        pvbase = title + ':'
        tw  = QtWidgets.QTabWidget()

        for g in range(8):
            addGroup(tw, pvbase+'PART:', g)

        lol = QtWidgets.QVBoxLayout()
        lol.addWidget(PatternTab(pvbase))
        lol.addWidget(tw)

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(lscroll)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(920,600)

        MainWindow.resize(720,600)
        MainWindow.setWindowTitle('xpmpatt')
        MainWindow.setCentralWidget(self.centralWidget)

def main():
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pv", help="pv to monitor")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    pv = args.pv
    if pv == 'tmo':
        pv = 'DAQ:NEH:XPM:6'
    elif pv == 'rix':
        pv = 'DAQ:NEH:XPM:1'
    
    ui.setupUi(MainWindow,pv)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
