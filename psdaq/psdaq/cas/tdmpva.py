import sys
import argparse
import logging
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from p4p.client.thread import Context
from psdaq.pytdm.pvdef import *

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

class PvScalarBox(QtWidgets.QGroupBox):
    def __init__(self, pvname, title, struct, edit=False):
        super(PvScalarBox,self).__init__(title)
        self.struct = struct
        glo = QtWidgets.QGridLayout()
        self.widgets = []
        for i,ttl in enumerate(struct):
            glo.addWidget(QtWidgets.QLabel(ttl),i,0)
            if edit:
                w = QtWidgets.QLineEdit('-')
            else:
                w = QtWidgets.QLabel('-')
            glo.addWidget(w,i,1)
            self.widgets.append(w)

        if edit:
            b = QtWidgets.QPushButton('Apply')
            glo.addWidget(b,len(struct),1)
            glo.setRowStretch(len(struct)+1,1)
            b.clicked.connect(self.put)
        else:
            glo.setRowStretch(len(struct),1)
        glo.setColumnStretch(2,1)
        self.setLayout(glo)
        initPvMon(self,pvname,isStruct=True)

    def update(self,err):
        if err is None:
            q = self.pv.__value__.todict()
            for i,v in enumerate(q):
                self.widgets[i].setText(QString(q[v]))

    def put(self):
        v = {}
        w = self.widgets
        for i,ttl in enumerate(self.struct):
            val = int(w[i].text()) if self.struct[ttl][0]=='i' else float(w[i].text())
            v[ttl] = val
#        print(v)
        self.pv.put(v)

class PvBuf(PvScalarBox):
    def __init__( self, pvname, title):
        super(PvBuf,self).__init__(pvname, title, monBuf)

class PvArrayTable(QtWidgets.QGroupBox):
    def __init__( self, pvname, title, struct, vertical=False, edit=False):
        super(PvArrayTable,self).__init__(title)
        self.struct = struct
        glo = QtWidgets.QGridLayout()
        self.widgets = []
        for i,ttl in enumerate(struct):
            x = 0 if vertical else i
            y = i if vertical else 0
            glo.addWidget(QtWidgets.QLabel(ttl),x,y)
            wl = []
            for j in range(len(struct[ttl][1])):
                x = j+1 if vertical else i
                y = i   if vertical else j+1
                w = QtWidgets.QLineEdit(str(struct[ttl][1][j])) if edit else QtWidgets.QLabel('')
                glo.addWidget(w,x,y)
                wl.append(w)
            self.widgets.append(wl)

        if edit:
            b = QtWidgets.QPushButton('Apply')
            glo.addWidget(b,x+1,1)
            glo.setRowStretch(x+1,1)
            b.clicked.connect(self.put)
        else:
            glo.setRowStretch(x+1,1)

        glo.setColumnStretch(y+1,1)
        self.setLayout(glo)

        initPvMon(self,pvname,isStruct=True)

    def update(self,err):
        if err is None:
            d = self.pv.__value__.todict()
            if not 'value' in d:
                print(d)
                return
            q = d['value']
            for i,v in enumerate(q):
                for j,w in enumerate(q[v]):
                    self.widgets[i][j].setText(QString(w))

    def put(self):
        v = {}
        w = self.widgets
        for i,ttl in enumerate(self.struct):
            q = []
            for j in range(len(self.struct[ttl][1])):
                val = int(w[i][j].text())
                q.append(val)
            v[ttl] = q
#        print(v)
        q = self.pv.__value__
        q['value'] = v
        self.pv.put(q)

class PLLBox(QtWidgets.QWidget):
    def __init__(self,title):
        super().__init__()
        lo = QtWidgets.QVBoxLayout()
        lo.addWidget(PvArrayTable(title+':QPLLSTATUS','QPLL Status', qpllStatus,
                                  vertical=True))
        lo.addWidget(PvArrayTable(title+':PLLSTATUS','Si5317 Status',si5317Status,
                                     vertical=True))
        lo.addWidget(PvArrayTable(title+':PLLCTRL','Si5317 Ctrl',si5317Ctrl,
                                  vertical=True, edit=True))
        self.setLayout(lo)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        maintab = QtWidgets.QTabWidget()
        maintab.addTab( PvArrayTable(title+':TIMSTATUS','TimingStatus',timingStatus),
                        'TimingStatus' )
        maintab.addTab( PvArrayTable(title+':CLKSTATUS','Clocks',clkStatus,
                                     vertical=True),
                        'Clocks' )
        maintab.addTab( PvArrayTable(title+':LINKSTATUS','LinkStatus',linkStatus,
                                     vertical=True),
                        'LinkStatus' )
        maintab.addTab( PvArrayTable(title+':LINKCTRL','LinkCtrl',linkCtrls,
                                     vertical=True, edit=True),
                        'LinkCtrl' )
        maintab.addTab( PLLBox(title), "PLLs" )
        maintab.addTab( PvArrayTable(title+':SFPSTATUS','SfpStatus',sfpStatus,
                                     vertical=True),
                        'SfpStatus' )

        lo = QtWidgets.QVBoxLayout()
        lo.addWidget(maintab)

        self.centralWidget.setLayout(lo)

        MainWindow.resize(500,550)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

def main():
    global NChannels
    global NLanes
    global Patterns

    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", default="DAQ:LAB2:HSD:DEV06_3E:A")
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.base)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
