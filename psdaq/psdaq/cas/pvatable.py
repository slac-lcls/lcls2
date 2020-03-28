import sys
import argparse
import logging
import socket
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from p4p.client.thread import Context

try:
    QString = unicode
except NameError:
    # Python 3
    QString = str

try:
    QChar = unichr
except NameError:
    # Python 3
    QChar = chr

class PvBase(object):
    def __init__(self, name, field):
        self.fields = [field]
        initPvMon(self,name,isStruct=True)

    def add(self, field):
        self.fields.append(field)

    def update(self, err):
        if err is None:
            q = self.pv.__value__.todict()
            for f in self.fields:
                f.update(q,err)

class PvManager(object):
    def __init__(self):
        self.names = {}
    
    def add(self,pvname,field):
        if pvname in self.names:
            self.names[pvname].add(field)
        else:
            self.names[pvname] = PvBase(pvname,field)

pvmanager = PvManager()

class PvField(QtWidgets.QLabel):
    def __init__(self, base, field):
        super(QtWidgets.QLabel,self).__init__('-')
        args = field.split('.')[1].split('[')
        self.field = args[0]
        if len(args)>1:
            self.index = int(args[1].split(']')[0])
        else:
            self.index = -1
        pvname = base+':'+field.split('.')[0]
        pvmanager.add(pvname,self)

    def update(self,q,err):
        if err is None:
            if self.index>=0:
                f = q[self.field][self.index]
            else:
                f = q[self.field]
            try:
                s = QString(int(f))
            except:
                v = ''
                for i in range(len(f)):
                    if i==0:
                        v = '{:}'.format(f[i])
                    else:
                        v = v+'\n{:}'.format(f[i])
                s = QString(v)
            self.setText(s)

class PvScalarTable(QtWidgets.QGroupBox):
    def __init__(self, bases, fields, edit=False):
        super(PvScalarTable,self).__init__('PvScalarTable')

        self.bases = bases
 
        print('bases {:}'.format(bases))

        #  Remove the greatest common prefix
        ncommon = len(bases[0])
        for i,base in enumerate(bases[1:]):
            while base[0:ncommon] != bases[0][0:ncommon]:
                ncommon -= 1

        names = []
        for i,base in enumerate(bases):
            names.append(base[ncommon:])

        print('names {:}'.format(names))

        print('fields {:}'.format(fields))

        self.fields = []
        for i,field in enumerate(fields):
            self.fields.append(field.split('.')[1])

        print('self.fields {:}'.format(self.fields))

        self.pvnames = []
        for i,base in enumerate(bases):
            pvnames = []
            for j,field in enumerate(fields):
                pvnames.append(base+':'+field.split('.')[0])
            self.pvnames.append(pvnames)

        print('pvnames {:}'.format(pvnames))

        glo = QtWidgets.QGridLayout()
        for i,ttl in enumerate(names):
            glo.addWidget(QtWidgets.QLabel(ttl),0,i+1)
        for i,ttl in enumerate(self.fields):
            glo.addWidget(QtWidgets.QLabel(ttl),i+1,0)

        self.widgets = []
        for i,field in enumerate(fields):
            for j,base in enumerate(bases):
                glo.addWidget(PvField(base,field),i+1,j+1)

        glo.setRowStretch(len(fields)+1,1)
        glo.setColumnStretch(len(bases),1)
        self.setLayout(glo)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, bases, fields):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        lo = QtWidgets.QVBoxLayout()
        lo.addWidget(PvScalarTable(bases,fields))

        self.centralWidget.setLayout(lo)

        MainWindow.resize(500,550)
        MainWindow.setWindowTitle("pvatable")
        MainWindow.setCentralWidget(self.centralWidget)

def main():

    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("--bases", help="pv bases to monitor", nargs='+', default="DAQ:LAB2:HSD:DEV06_3E:A")
    parser.add_argument("--fields", help="pv fields to monitor", nargs='+', default="MONTIMING.trigcntsum")
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.bases,args.fields)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
